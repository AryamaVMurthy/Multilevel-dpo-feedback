#!/bin/bash
# Submit with: sbatch -A <account> --export=ALL,PROJECT_DIR=...,DATA=...,OUTPUT=...,STUDENT_MODEL=...,TEACHER_MODEL=... scripts/turing_collect.sh
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${SEED:?SEED must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-32}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
STUDENT_THINKING_MODE="${STUDENT_THINKING_MODE:-direct}"
SCRATCHPAD_MAX_NEW_TOKENS="${SCRATCHPAD_MAX_NEW_TOKENS:-128}"
echo "<runtime component=collection attention_implementation=\"$ATTENTION_IMPLEMENTATION\" fallback_reason=\"${ATTENTION_FALLBACK_REASON:-none}\"/>"
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT

COLLECT_ARGS=(
  --data "$DATA" --output "$OUTPUT"
  --student-model "$STUDENT_MODEL"
  --student-revision "$STUDENT_REVISION"
  --teacher-model "$TEACHER_MODEL"
  --teacher-revision "$TEACHER_REVISION"
  --dataset-revision "$DATASET_REVISION" --prompt-version "$PROMPT_VERSION" --seed "$SEED"
  --teacher-quantization 4bit
  --attention-implementation "$ATTENTION_IMPLEMENTATION"
  --student-device cuda:1 --teacher-device cuda:0
  --trajectory-cache "${TRAJECTORY_CACHE:?TRAJECTORY_CACHE must be supplied with --export}"
  --policy-hash "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
  --max-interventions 4
  --student-batch-size "$STUDENT_BATCH_SIZE" --teacher-batch-size "$TEACHER_BATCH_SIZE"
  --student-thinking-mode "$STUDENT_THINKING_MODE"
  --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --answer-max-new-tokens 32
  --teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS" --teacher-thinking
)
uv run --frozen python -m text_feedback_dpo.cli collect "${COLLECT_ARGS[@]}"
