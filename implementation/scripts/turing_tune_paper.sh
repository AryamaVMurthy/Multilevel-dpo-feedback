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
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

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
mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR"
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
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

if [[ "$METHOD" == "grpo" ]]; then
  GRPO_ENVIRONMENT="${GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required for GRPO methods}"
  if [[ ! -x "$GRPO_ENVIRONMENT/bin/python" ]]; then
    echo "ERROR: frozen GRPO environment is missing: $GRPO_ENVIRONMENT" >&2
    exit 1
  fi
  export UV_PROJECT_ENVIRONMENT="$GRPO_ENVIRONMENT"
  if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
    echo "ERROR: locked GRPO runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
    exit 1
  fi
  RUNNER=(uv run --project "$PROJECT_DIR/environments/grpo" --frozen --no-sync)
else
  RUNNER=(uv run --frozen --no-sync)
fi

GPU_LOG="$OUTPUT_DIR/gpu-${SLURM_JOB_ID}.csv"
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

"${RUNNER[@]}" python -m text_feedback_dpo.cli tune-paper \
  --config "$CONFIG" --method "$METHOD" --candidate-id "$CANDIDATE_ID" \
  --stage "$STAGE_ID" --data "$DATA_PATH" --validation "$VALIDATION_PATH" \
  --output-dir "$OUTPUT_DIR" --ledger "$LEDGER_PATH" "$@"
