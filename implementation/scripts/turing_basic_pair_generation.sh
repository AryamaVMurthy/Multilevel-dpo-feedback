#!/bin/bash
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4

export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable on host $(hostname); refusing to use /home for model cache or Python environment" >&2
  exit 1
fi
echo "scratch_df_before=$(df -h /scratch | tail -1)"
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
echo "scratch_dir=${SCRATCH_DIR}"

export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/project_venv"

mkdir -p logs runs

echo "job_id=${SLURM_JOB_ID}"
echo "turing_account=${TURING_ACCOUNT}"
echo "host=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "config=${CONFIG}"
echo "start_time=$(date --iso-8601=seconds)"
nvidia-smi
export HF_HOME="$SCRATCH_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!

uv run python -m text_feedback_dpo.cli generate-pipeline --config "$CONFIG"

kill "$GPU_MONITOR_PID"
cp "logs/gpu-${SLURM_JOB_ID}.csv" "runs/qwen35-basic-smoke/gpu-${SLURM_JOB_ID}.csv"
echo "end_time=$(date --iso-8601=seconds)"
