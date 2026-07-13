#!/bin/bash
# Batched no-hint student rollout collection; submit one deterministic shard per job.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

fail() { printf 'event=failure reason=%q fallback_reason=%q\n' "$1" "${2:-none}" >&2; exit 2; }
for name in PROJECT_DIR DATA OUTPUT MODEL MODEL_REVISION DATASET_SOURCE DATASET_REVISION \
  DATA_SHA256 POLICY_HASH PROMPT_VERSION SEEDS EXPECTED_COMMIT QUERY_MIN_NEW_TOKENS \
  RESPONSE_MIN_NEW_TOKENS; do
  [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" bootstrap_contract_missing
done
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "bootstrap requires one node and one task" allocation_shape_invalid
[[ "${SLURM_GPUS_ON_NODE:?}" == 1 || "${SLURM_GPUS_ON_NODE}" == gpu:1 ]] || fail "bootstrap requires exactly one GPU" gpu_count_invalid
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ -f "$DATA" ]] || fail "bootstrap data does not exist: $DATA" data_missing
[[ "$(sha256sum "$DATA" | awk '{print $1}')" == "$DATA_SHA256" ]] || fail "bootstrap data hash mismatch" data_hash_mismatch
[[ "$POLICY_HASH" =~ ^[0-9a-f]{64}$ ]] || fail "POLICY_HASH is not a lowercase SHA-256" policy_hash_invalid

module load u22/cuda/12.4
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" logs

OUTPUT_MANIFEST="${OUTPUT%.*}.manifest.json"
if [[ -e "$OUTPUT" ]]; then
  [[ -f "$OUTPUT_MANIFEST" ]] || fail "output exists without its completion manifest" incomplete_output_exists
  uv run --frozen python - "$OUTPUT" "$OUTPUT_MANIFEST" "$DATA_SHA256" <<'PY'
import hashlib, json, sys
output, manifest_path, data_hash = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
if manifest.get("command") != "bootstrap-rollouts" or manifest.get("dataset", {}).get("sha256") != data_hash:
    raise SystemExit("existing bootstrap output manifest identity mismatch")
if not open(output, encoding="utf-8").read().strip():
    raise SystemExit("existing bootstrap output is empty")
print("event=bootstrap_reuse status=validated fallback_reason=none")
PY
  exit 0
fi

read -r -a SEED_ARGS <<< "$SEEDS"
(( ${#SEED_ARGS[@]} > 0 )) || fail "SEEDS parsed to an empty list" seeds_empty
printf 'event=bootstrap_launch rows_file=%q seeds=%q fallback_reason=none\n' "$DATA" "$SEEDS"
nvidia-smi
uv run --frozen python -m text_feedback_dpo.cli bootstrap-rollouts \
  --data "$DATA" --output "$OUTPUT" --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" \
  --attention-implementation sdpa --device cuda:0 --policy-hash "$POLICY_HASH" \
  --prompt-version "$PROMPT_VERSION" --seeds "${SEED_ARGS[@]}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-4}" --response-batch-size "${RESPONSE_BATCH_SIZE:-4}" \
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS:-32}" \
  --response-max-new-tokens "${RESPONSE_MAX_NEW_TOKENS:-256}" \
  --query-min-new-tokens "$QUERY_MIN_NEW_TOKENS" \
  --response-min-new-tokens "$RESPONSE_MIN_NEW_TOKENS" \
  --query-temperature "${QUERY_TEMPERATURE:-0.7}" \
  --response-temperature "${RESPONSE_TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.9}" \
  --top-k 8 --k1 1.2 --b 0.75 --context-budget 4096
printf 'event=bootstrap_complete output=%q fallback_reason=none\n' "$OUTPUT"
