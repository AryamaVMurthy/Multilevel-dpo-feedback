#!/bin/bash
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

CONFIG="${CONFIG:?CONFIG is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
HF_CACHE_DIR="${HF_CACHE_DIR:?HF_CACHE_DIR is required}"
PIPELINE_COMMAND="${PIPELINE_COMMAND:?PIPELINE_COMMAND is required}"

module load u22/cuda/12.4

export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable on host $(hostname); refusing to use /home for model cache or Python environment" >&2
  exit 1
fi
echo "scratch_df_before=$(df -h /scratch | tail -1)"
SCRATCH_DIR="/scratch/$USER/text-feedback-dpo/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_CACHE_DIR"
echo "scratch_dir=${SCRATCH_DIR}"
echo "hf_cache_dir=${HF_CACHE_DIR}"

export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/project_venv"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

mkdir -p logs runs
RUN_OUTPUT_DIR="$(uv run --frozen python -c '
from pathlib import Path
import sys
from text_feedback_dpo.config import load_config
print(load_config(Path(sys.argv[1]))["output_dir"])
' "$CONFIG")"

echo "job_id=${SLURM_JOB_ID}"
echo "turing_account=${TURING_ACCOUNT}"
echo "host=$(hostname)"
echo "slurm_job_num_nodes=${SLURM_JOB_NUM_NODES:-unset}"
echo "slurm_nnodes=${SLURM_NNODES:-unset}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "config=${CONFIG}"
echo "output_dir=${RUN_OUTPUT_DIR}"
echo "start_time=$(date --iso-8601=seconds)"

allocation_mismatch() {
  echo "ERROR: allocation_mismatch: $*" >&2
  exit 1
}
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || allocation_mismatch "expected one node, got SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset}"
[[ "${SLURM_NNODES:-}" == "1" ]] || allocation_mismatch "expected one node, got SLURM_NNODES=${SLURM_NNODES:-unset}"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "1" ]]; then
  allocation_mismatch "expected one GPU on the node, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || allocation_mismatch "CUDA_VISIBLE_DEVICES is unset"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#visible_gpus[@]}" == "1" ]] || allocation_mismatch "expected one visible GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "logs/gpu-${SLURM_JOB_ID}.csv" &
GPU_MONITOR_PID=$!
cleanup_gpu_monitor() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup_gpu_monitor EXIT

case "$PIPELINE_COMMAND" in
  native-pipeline)
    uv run --frozen python -m text_feedback_dpo.cli native-pipeline --config "$CONFIG"
    ;;
  *)
    echo "ERROR: unsupported pipeline command: ${PIPELINE_COMMAND}" >&2
    exit 1
    ;;
esac

cleanup_gpu_monitor
trap - EXIT
cp "logs/gpu-${SLURM_JOB_ID}.csv" "${RUN_OUTPUT_DIR}/gpu-${SLURM_JOB_ID}.csv"
echo "end_time=$(date --iso-8601=seconds)"
