#!/bin/bash
# Teacher-free validation/test evaluation for one checkpoint.
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:?CHECKPOINT_KIND is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
SPLIT="${SPLIT:?SPLIT is required}"
FREEZE_MANIFEST="${FREEZE_MANIFEST:?FREEZE_MANIFEST is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

if [[ "$CHECKPOINT_KIND" == "base" ]]; then
  if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "ERROR: CHECKPOINT must be unset for CHECKPOINT_KIND=base" >&2
    exit 1
  fi
elif [[ "$CHECKPOINT_KIND" == "adapter" ]]; then
  CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required for CHECKPOINT_KIND=adapter}"
else
  echo "ERROR: CHECKPOINT_KIND must be base or adapter" >&2
  exit 1
fi

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: RUNTIME_ROOT must be node-local /scratch storage: $RUNTIME_ROOT" >&2
  exit 1
fi
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
SHARD_OUTPUT_DIR="$OUTPUT_DIR/shard-$(printf '%04d' "$SHARD_INDEX")"
mkdir -p "$SCRATCH_DIR" "$SHARD_OUTPUT_DIR"
mkdir -p "$RUNTIME_ROOT"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
  echo "ERROR: MODEL_CACHE_DIR is not present on $(hostname): $MODEL_CACHE_DIR" >&2
  exit 1
fi
export HF_HOME="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

GPU_LOG="$SHARD_OUTPUT_DIR/gpu-${SLURM_JOB_ID}_${SHARD_INDEX}.csv"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 > "$GPU_LOG" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
    if ! wait "$GPU_MONITOR_PID"; then
      echo "GPU monitor exited after explicit termination" >&2
    fi
  fi
}
trap cleanup EXIT

EVALUATE_ARGS=(
  --config "$CONFIG"
  --checkpoint-kind "$CHECKPOINT_KIND"
  --data "$DATA_PATH"
  --split "$SPLIT"
  --output-dir "$SHARD_OUTPUT_DIR"
  --freeze-manifest "$FREEZE_MANIFEST"
  --source-commit "$SOURCE_COMMIT"
  --shard-index "$SHARD_INDEX"
  --num-shards "$NUM_SHARDS"
)
if [[ "$CHECKPOINT_KIND" == "adapter" ]]; then
  EVALUATE_ARGS+=(--checkpoint "$CHECKPOINT")
fi
uv run --frozen python -m text_feedback_dpo.cli evaluate-paper "${EVALUATE_ARGS[@]}"
