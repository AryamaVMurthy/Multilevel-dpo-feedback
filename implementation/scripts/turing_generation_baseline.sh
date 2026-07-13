#!/bin/bash
# Refresh the commit-bound measured SDPA generation baseline and frozen launch decision.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_generation_baseline
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() {
  local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${LAUNCH_MAX_STEPS:?LAUNCH_MAX_STEPS must be supplied with --export}"
: "${LEARNING_RATE:?LEARNING_RATE must be supplied with --export}"
: "${EPOCHS:?EPOCHS must be supplied with --export}"
: "${SAVE_STEPS:?SAVE_STEPS must be supplied with --export}"
: "${EVAL_STEPS:?EVAL_STEPS must be supplied with --export}"
: "${QUERY_MAX_NEW_TOKENS:?QUERY_MAX_NEW_TOKENS must be supplied with --export}"
: "${RESPONSE_MAX_NEW_TOKENS:?RESPONSE_MAX_NEW_TOKENS must be supplied with --export}"
: "${STUDENT_THINKING_MODE:?STUDENT_THINKING_MODE must be supplied with --export}"
: "${SCRATCHPAD_MAX_NEW_TOKENS:?SCRATCHPAD_MAX_NEW_TOKENS must be supplied with --export}"
: "${QUERY_TEMPERATURE:?QUERY_TEMPERATURE must be supplied with --export}"
: "${RESPONSE_TEMPERATURE:?RESPONSE_TEMPERATURE must be supplied with --export}"
: "${TOP_P:?TOP_P must be supplied with --export}"
: "${TOP_K:?TOP_K must be supplied with --export}"
: "${BM25_K1:?BM25_K1 must be supplied with --export}"
: "${BM25_B:?BM25_B must be supplied with --export}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
[[ "$SLURM_NNODES" == 1 ]] || fail "generation baseline requires one node; got $SLURM_NNODES" multi_node_probe_forbidden
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
[[ "$SLURM_NTASKS" == 1 ]] || fail "generation baseline requires one task; got $SLURM_NTASKS" multi_task_probe_forbidden
[[ "$(allocated_gpu_count)" == 1 ]] || fail "generation baseline requires exactly one GPU" probe_gpu_count

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR: $PROJECT_DIR" invalid_project_root
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$OUTPUT_ROOT" logs

PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
[[ -x "$PROBE_RUNNER" ]] || fail "probe runner is missing or not executable: $PROBE_RUNNER" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
BASELINE="$OUTPUT_ROOT/baseline-sdpa.json"
DECISION="$OUTPUT_ROOT/generation-optimization-decision.json"

log_event benchmark_start output="$BASELINE" commit_hash="$(git rev-parse HEAD)" fallback_reason=none
run_probe_runner benchmark --probe-name baseline-sdpa --result "$BASELINE" \
  --commit-hash "$(git rev-parse HEAD)" --config "$CONFIG" --data "$DATA" --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --prompt-sha256 "$PROMPT_HASH" \
  --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
  --sample-size "${PROBE_SAMPLE_SIZE:-16}" --warmup-repeats "${PROBE_WARMUP_REPEATS:-2}" \
  --measured-repeats "${PROBE_MEASURED_REPEATS:-5}" --num-generations 1 --rl-generation-batch-size 1 \
  --max-completion-length "$RESPONSE_MAX_NEW_TOKENS" --training-method sft --attention-implementation sdpa \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-4}"

run_probe_runner freeze-decision --purpose generation --baseline "$BASELINE" --output "$DECISION" \
  --launch-max-steps "$LAUNCH_MAX_STEPS" --launch-learning-rate "$LEARNING_RATE" --launch-epochs "$EPOCHS" \
  --launch-save-steps "$SAVE_STEPS" --launch-eval-steps "$EVAL_STEPS" --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" \
  --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" --student-thinking-mode "$STUDENT_THINKING_MODE" \
  --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-temperature "$QUERY_TEMPERATURE" \
  --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"

DECISION_SHA256="$(sha256sum "$DECISION" | awk '{print $1}')"
log_event benchmark_complete baseline="$BASELINE" decision="$DECISION" decision_sha256="$DECISION_SHA256" fallback_reason=sdpa_baseline_selected
