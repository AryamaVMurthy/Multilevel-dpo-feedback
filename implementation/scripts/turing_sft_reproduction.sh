#!/bin/bash
# Deterministic generation audit of every verified SFT target from one checkpoint.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_sft_reproduction
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" reproduction_contract_missing; }

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "SFT_DATA"
require_env "SFT_DATA_HASH"
require_env "CHECKPOINT"
require_env "CHECKPOINT_MODEL_HASH"
require_env "OUTPUT"
require_env "REPORT"
require_env "BATCH_SIZE"
require_env "QUERY_MIN_NEW_TOKENS"
require_env "QUERY_MAX_NEW_TOKENS"
require_env "RESPONSE_MIN_NEW_TOKENS"
require_env "RESPONSE_MAX_NEW_TOKENS"
require_env "SEED"
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "SFT reproduction requires one node and one task" allocation_shape_invalid
[[ "${SLURM_GPUS_ON_NODE:?}" == 1 || "$SLURM_GPUS_ON_NODE" == gpu:1 ]] || fail "SFT reproduction requires exactly one GPU; got $SLURM_GPUS_ON_NODE" gpu_count_invalid
[[ -f "$SFT_DATA" ]] || fail "SFT data is missing: $SFT_DATA" sft_data_missing
[[ -f "$CHECKPOINT/model.safetensors" ]] || fail "checkpoint model is missing: $CHECKPOINT/model.safetensors" checkpoint_model_missing
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ "$(sha256sum "$SFT_DATA" | awk '{print $1}')" == "$SFT_DATA_HASH" ]] || fail "SFT data hash mismatch" sft_data_hash_mismatch
[[ "$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')" == "$CHECKPOINT_MODEL_HASH" ]] || fail "checkpoint model hash mismatch" checkpoint_model_hash_mismatch
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer" batch_size_invalid
[[ ! -e "$OUTPUT" && ! -e "$REPORT" ]] || fail "output or report already exists; refusing stale reuse" output_exists

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs

log_event reproduction_start checkpoint_model_sha256="$CHECKPOINT_MODEL_HASH" sft_data_sha256="$SFT_DATA_HASH" rows=all fallback_reason=none
nvidia-smi
uv run --frozen python -m text_feedback_dpo.cli evaluate-sft-reproduction \
  --data "$SFT_DATA" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_MODEL_HASH" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --batch-size "$BATCH_SIZE" \
  --query-min-new-tokens "$QUERY_MIN_NEW_TOKENS" \
  --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" \
  --response-min-new-tokens "$RESPONSE_MIN_NEW_TOKENS" \
  --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --attention-implementation sdpa \
  --device cuda:0

[[ -s "$OUTPUT" && -s "$REPORT" ]] || fail "reproduction audit did not create non-empty artifacts" reproduction_artifact_missing
log_event reproduction_complete output="$OUTPUT" report="$REPORT" output_sha256="$(sha256sum "$OUTPUT" | awk '{print $1}')" report_sha256="$(sha256sum "$REPORT" | awk '{print $1}')" fallback_reason=none
