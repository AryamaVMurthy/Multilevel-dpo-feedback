#!/bin/bash
# One resumable primary DPO round. Submit with: sbatch -A <account> --export=ALL,... scripts/turing_primary_round.sh
#SBATCH -p u22
#SBATCH -n 64
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${EVAL_DATA:?EVAL_DATA must be supplied with --export}"
: "${EVAL_PREFERENCES:?EVAL_PREFERENCES must be supplied with --export}"
: "${ROUND_DIR:?ROUND_DIR must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${SEED:?SEED must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: invalid PROJECT_DIR" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$ROUND_DIR" logs
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
echo "<runtime component=primary_round attention_implementation=\"$ATTENTION_IMPLEMENTATION\" fallback_reason=\"${ATTENTION_FALLBACK_REASON:-none}\"/>"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT

TRAJECTORIES="$ROUND_DIR/trajectories.jsonl"
PREFERENCES="$ROUND_DIR/preferences.jsonl"
DPO_OUT="$ROUND_DIR/dpo"
PREDICTIONS="$ROUND_DIR/validation-predictions.jsonl"
METRICS="$ROUND_DIR/validation-metrics.json"

uv run --frozen python -m text_feedback_dpo.cli collect \
  --data "$DATA" --output "$TRAJECTORIES" \
  --student-model "$STUDENT_MODEL" --student-revision "$STUDENT_REVISION" \
  --teacher-model "$TEACHER_MODEL" --teacher-revision "$TEACHER_REVISION" \
  --dataset-revision "$DATASET_REVISION" --prompt-version "$PROMPT_VERSION" --seed "$SEED" \
  --teacher-quantization 4bit --attention-implementation "$ATTENTION_IMPLEMENTATION" \
  --student-device cuda:1 --teacher-device cuda:0 \
  --trajectory-cache "$ROUND_DIR/trajectory-cache.jsonl" \
  --policy-hash "${POLICY_HASH:?POLICY_HASH must be supplied with --export}" \
  --max-interventions 4 --student-thinking-mode "${STUDENT_THINKING_MODE:-direct}" \
  --answer-max-new-tokens 32 --scratchpad-max-new-tokens 256 --teacher-thinking

uv run --frozen python -m text_feedback_dpo.cli build-preferences \
  --trajectories "$TRAJECTORIES" --output "$PREFERENCES"

TRAIN_ARGS=(
  --config "$CONFIG" --train "$PREFERENCES" --eval "$EVAL_PREFERENCES"
  --output "$DPO_OUT" --deepspeed-config configs/deepspeed_zero3.json
  --save-steps 100 --eval-steps 100
)
if [[ -n "${START_MODEL:-}" ]]; then
  TRAIN_ARGS+=(--model "$START_MODEL")
  if [[ -n "${START_REVISION:-}" ]]; then
    TRAIN_ARGS+=(--model-revision "$START_REVISION")
  fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
torchrun --standalone --nproc_per_node=2 -m text_feedback_dpo.cli train-dpo "${TRAIN_ARGS[@]}"

uv run --frozen python -m text_feedback_dpo.cli generate \
  --data "$EVAL_DATA" --output "$PREDICTIONS" \
  --model "$DPO_OUT/final" \
  --attention-implementation "$ATTENTION_IMPLEMENTATION" --batch-size "$GENERATION_BATCH_SIZE" \
  --student-thinking-mode "${STUDENT_THINKING_MODE:-direct}" --max-new-tokens 32 --temperature 0.0 --top-p 1.0 \
  --policy-hash "${POLICY_HASH}:dpo-final"

uv run --frozen python -m text_feedback_dpo.cli evaluate \
  --data "$EVAL_DATA" --predictions "$PREDICTIONS" --output "$METRICS"

uv run --frozen python -m text_feedback_dpo.cli report \
  --metrics "$METRICS" --output "$ROUND_DIR/report.html" \
  --artifact "$TRAJECTORIES" --artifact "$PREFERENCES" --artifact "$PREDICTIONS" --artifact "$METRICS"
