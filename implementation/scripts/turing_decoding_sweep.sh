#!/bin/bash
# One-GPU train-only MATH decoding screening or confirmation stage.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
DATASET_MANIFEST="${DATASET_MANIFEST:?DATASET_MANIFEST is required}"
DATASET_AUDIT="${DATASET_AUDIT:?DATASET_AUDIT is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SWEEP_STAGE="${SWEEP_STAGE:?SWEEP_STAGE is required}"
SWEEP_SEED="${SWEEP_SEED:?SWEEP_SEED is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home runtime or model fallback" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* || "$MODEL_CACHE_DIR" != /scratch/* || "$DATA_PATH" != /scratch/* ]]; then
  echo "ERROR: runtime, model cache, and dataset must use node-local /scratch" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing existing decoding-sweep output: $OUTPUT_DIR" >&2
  exit 1
fi
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export HF_HOME="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

GPU_LOG="$(dirname "$OUTPUT_DIR")/gpu-${SLURM_JOB_ID}.csv"
mkdir -p "$(dirname "$OUTPUT_DIR")"
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

ARGS=(
  --config "$CONFIG"
  --data-path "$DATA_PATH"
  --dataset-manifest "$DATASET_MANIFEST"
  --dataset-audit "$DATASET_AUDIT"
  --model-cache-manifest "$MODEL_CACHE_DIR/tfdpo-model-cache-manifest.json"
  --output-dir "$OUTPUT_DIR"
  --source-commit "$SOURCE_COMMIT"
  --stage "$SWEEP_STAGE"
  --seed "$SWEEP_SEED"
)
if [[ "$SWEEP_STAGE" == "confirmation" ]]; then
  PROMOTION_MANIFEST="${PROMOTION_MANIFEST:?PROMOTION_MANIFEST is required for confirmation}"
  ARGS+=(--promotion-manifest "$PROMOTION_MANIFEST")
elif [[ "$SWEEP_STAGE" != "screening" ]]; then
  echo "ERROR: SWEEP_STAGE must be screening or confirmation" >&2
  exit 1
fi
uv run --frozen --no-sync python scripts/run_local_concise_sweep.py "${ARGS[@]}"
