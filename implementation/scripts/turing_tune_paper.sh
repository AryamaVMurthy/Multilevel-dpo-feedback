#!/bin/bash
# One candidate/stage per Slurm task. The caller supplies a deterministic array.
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
METHOD="${METHOD:?METHOD is required}"
CANDIDATE_ID="${CANDIDATE_ID:?CANDIDATE_ID is required}"
STAGE_ID="${STAGE_ID:?STAGE_ID is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
VALIDATION_PATH="${VALIDATION_PATH:?VALIDATION_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
LEDGER_PATH="${LEDGER_PATH:?LEDGER_PATH is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
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

uv run --frozen python -m text_feedback_dpo.cli tune-paper \
  --config "$CONFIG" --method "$METHOD" --candidate-id "$CANDIDATE_ID" \
  --stage "$STAGE_ID" --data "$DATA_PATH" --validation "$VALIDATION_PATH" \
  --output-dir "$OUTPUT_DIR" --ledger "$LEDGER_PATH"
