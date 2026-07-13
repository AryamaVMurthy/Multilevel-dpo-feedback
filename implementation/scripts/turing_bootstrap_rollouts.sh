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
  RESPONSE_MIN_NEW_TOKENS QUERY_BATCH_SIZE RESPONSE_BATCH_SIZE QUERY_MAX_NEW_TOKENS \
  RESPONSE_MAX_NEW_TOKENS QUERY_TEMPERATURE RESPONSE_TEMPERATURE TOP_P; do
  [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" bootstrap_contract_missing
done
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "bootstrap requires one node and one task" allocation_shape_invalid
[[ "${SLURM_GPUS_ON_NODE:?}" == 1 || "${SLURM_GPUS_ON_NODE}" == gpu:1 ]] || fail "bootstrap requires exactly one GPU" gpu_count_invalid
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ -f "$DATA" ]] || fail "bootstrap data does not exist: $DATA" data_missing
[[ "$(sha256sum "$DATA" | awk '{print $1}')" == "$DATA_SHA256" ]] || fail "bootstrap data hash mismatch" data_hash_mismatch
[[ "$POLICY_HASH" =~ ^[0-9a-f]{64}$ ]] || fail "POLICY_HASH is not a lowercase SHA-256" policy_hash_invalid
MODEL_ARTIFACT_ARGS=()
MODEL_ARTIFACT_IDENTITY=none
if [[ -d "$MODEL" ]]; then
  [[ -n "${MODEL_ARTIFACT_SHA256:-}" ]] || fail "local MODEL requires MODEL_ARTIFACT_SHA256" model_artifact_hash_missing
  [[ -f "$MODEL/model.safetensors" ]] || fail "local model.safetensors is missing" model_artifact_missing
  [[ "$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')" == "$MODEL_ARTIFACT_SHA256" ]] || fail "local model artifact hash mismatch" model_artifact_hash_mismatch
  MODEL_ARTIFACT_ARGS=(--model-artifact-sha256 "$MODEL_ARTIFACT_SHA256")
  MODEL_ARTIFACT_IDENTITY="$MODEL_ARTIFACT_SHA256"
elif [[ -n "${MODEL_ARTIFACT_SHA256:-}" ]]; then
  fail "MODEL_ARTIFACT_SHA256 was supplied for a non-local MODEL" model_artifact_hash_unexpected
fi

module load u22/cuda/12.4
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" logs

OUTPUT_MANIFEST="${OUTPUT%.*}.manifest.json"
if [[ -e "$OUTPUT" ]]; then
  [[ -f "$OUTPUT_MANIFEST" ]] || fail "output exists without its completion manifest" incomplete_output_exists
  uv run --frozen python - "$OUTPUT" "$OUTPUT_MANIFEST" "$DATA_SHA256" "$MODEL" "$MODEL_REVISION" "$POLICY_HASH" "$MODEL_ARTIFACT_IDENTITY" <<'PY'
import hashlib, json, sys
output, manifest_path, data_hash, model, revision, policy_hash, artifact_hash = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
if manifest.get("command") != "bootstrap-rollouts" or manifest.get("dataset", {}).get("sha256") != data_hash:
    raise SystemExit("existing bootstrap output manifest identity mismatch")
model_manifest = manifest.get("model", {})
expected_artifact = None if artifact_hash == "none" else artifact_hash
if model_manifest != {"identity": model, "revision": revision, "policy_hash": policy_hash, "artifact_sha256": expected_artifact}:
    raise SystemExit("existing bootstrap output model identity mismatch")
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
  "${MODEL_ARTIFACT_ARGS[@]}" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" \
  --attention-implementation sdpa --device cuda:0 --policy-hash "$POLICY_HASH" \
  --prompt-version "$PROMPT_VERSION" --seeds "${SEED_ARGS[@]}" \
  --query-batch-size "$QUERY_BATCH_SIZE" --response-batch-size "$RESPONSE_BATCH_SIZE" \
  --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" \
  --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
  --query-min-new-tokens "$QUERY_MIN_NEW_TOKENS" \
  --response-min-new-tokens "$RESPONSE_MIN_NEW_TOKENS" \
  --query-temperature "$QUERY_TEMPERATURE" \
  --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" \
  --top-k 8 --k1 1.2 --b 0.75 --context-budget 4096
printf 'event=bootstrap_complete output=%q fallback_reason=none\n' "$OUTPUT"
