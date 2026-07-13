#!/bin/bash
# Verify complete trajectory-cache reuse without loading student or teacher models.
# Submit with: sbatch -A <account> --export=ALL,... scripts/turing_offline_reuse_check.sh
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_offline_reuse_check
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" "gpu_count_parse_unsupported"; fi; }

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${CACHE:?CACHE must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${SEED:?SEED must be supplied with --export}"
: "${PROTOCOL:?PROTOCOL must be archival or active-search}"
[[ "$PROTOCOL" == "archival" || "$PROTOCOL" == "active-search" ]] || fail "unsupported PROTOCOL=$PROTOCOL" "protocol_validation"
: "${SLURM_NNODES:?SLURM_NNODES is required}"; [[ "$SLURM_NNODES" == "1" ]] || fail "offline reuse requires one node" "multi_node_offline_reuse"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"; [[ "$SLURM_NTASKS" == "1" ]] || fail "offline reuse requires one task" "multi_task_offline_reuse"
[[ "$(allocated_gpu_count)" == "2" ]] || fail "offline reuse allocation must expose exactly two GPUs" "offline_reuse_gpu_count"

cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"

[[ -s "$CACHE" ]] || fail "complete trajectory cache is missing or empty: $CACHE" "cache_missing"
[[ "$CACHE" != "$OUTPUT" ]] || fail "offline output must differ from the cache path" "cache_output_collision"
rm -f "$OUTPUT"
log_event offline_reuse_start protocol="$PROTOCOL" cache="$CACHE" fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli collect \
  --data "$DATA" --output "$OUTPUT" \
  --student-model "$STUDENT_MODEL" \
  --student-revision "$STUDENT_REVISION" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-revision "$TEACHER_REVISION" \
  --dataset-revision "$DATASET_REVISION" --prompt-version "$PROMPT_VERSION" --seed "$SEED" \
  --teacher-quantization 4bit \
  --attention-implementation "${ATTENTION_IMPLEMENTATION:-sdpa}" \
  --student-device cuda:1 --teacher-device cuda:0 \
  --trajectory-cache "$CACHE" --policy-hash "$POLICY_HASH" \
  --max-interventions 4 --student-batch-size 1 --teacher-batch-size 1 --student-thinking-mode "${STUDENT_THINKING_MODE:-direct}" --teacher-thinking

cmp -s "$CACHE" "$OUTPUT" || {
  log_event failure reason="offline trajectory replay differs from the cache" fallback_reason=cache_parity_mismatch >&2
  sha256sum "$CACHE" "$OUTPUT" >&2
  exit 2
}
log_event offline_reuse_complete protocol="$PROTOCOL" cache="$CACHE" output="$OUTPUT" generation=skipped_complete_cache byte_identical=true fallback_reason=none
