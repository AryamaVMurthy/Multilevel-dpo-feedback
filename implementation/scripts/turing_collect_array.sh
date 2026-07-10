#!/bin/bash
# Submit with: sbatch --array=0-(N-1)%CONCURRENCY --export=ALL,... this script.
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
SPLIT="${SPLIT:?SPLIT is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}_${SHARD_INDEX}"
mkdir -p "$SCRATCH_DIR" "$OUTPUT_DIR"
export UV_CACHE_DIR="$HOME/tfdpo-runs/uv_cache"
export UV_PROJECT_ENVIRONMENT="$HOME/tfdpo-runs/project_venv"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

GPU_LOG="$OUTPUT_DIR/gpu-${SLURM_JOB_ID}_${SHARD_INDEX}.csv"
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

echo "job_id=${SLURM_JOB_ID} array_index=${SHARD_INDEX} account=${TURING_ACCOUNT} host=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
echo "scratch_before=$(df -h /scratch | tail -1)"
cp "$CONFIG" "$SCRATCH_DIR/config.yaml"
uv run --frozen python -m text_feedback_dpo.cli collect-shard \
  --config "$SCRATCH_DIR/config.yaml" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
echo "scratch_after=$(df -h /scratch | tail -1)"
