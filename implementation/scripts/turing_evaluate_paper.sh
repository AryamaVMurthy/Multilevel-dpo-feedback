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
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
SPLIT="${SPLIT:?SPLIT is required}"
FREEZE_MANIFEST="${FREEZE_MANIFEST:?FREEZE_MANIFEST is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR"
export UV_CACHE_DIR="$HOME/tfdpo-runs/uv_cache"
export UV_PROJECT_ENVIRONMENT="$HOME/tfdpo-runs/project_venv"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export UV_LINK_MODE=copy
cd "$PROJECT_DIR"

GPU_LOG="$OUTPUT_DIR/gpu-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "$GPU_LOG" &
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

uv run --frozen python -m text_feedback_dpo.cli evaluate-paper \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --data "$DATA_PATH" \
  --split "$SPLIT" --output-dir "$OUTPUT_DIR" --freeze-manifest "$FREEZE_MANIFEST"
