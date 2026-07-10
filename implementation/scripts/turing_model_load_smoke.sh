#!/bin/bash
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

MODEL_ID="Qwen/Qwen3.5-2B"
RUN_DIR="runs/model-load-smoke"
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

mkdir -p logs "$RUN_DIR"

echo "job_id=${SLURM_JOB_ID}"
echo "turing_account=${TURING_ACCOUNT}"
echo "host=$(hostname)"
echo "slurm_job_num_nodes=${SLURM_JOB_NUM_NODES:-unset}"
echo "slurm_nnodes=${SLURM_NNODES:-unset}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "model_id=${MODEL_ID}"
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
export HF_HOME="$SCRATCH_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_datasets"

uv run --frozen python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

run_dir = Path("runs/model-load-smoke")
events_path = run_dir / "events.jsonl"
model_id = "Qwen/Qwen3.5-2B"

def event(name, **fields):
    payload = {
        "event_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        **fields,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false; refusing CPU fallback")

event("cuda_verified", gpu=torch.cuda.get_device_name(0), model_id=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
prompt = "Return only the number: 2 + 2 ="
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8)
completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
event("model_generated", model_id=model_id, completion=completion)
PY

echo "end_time=$(date --iso-8601=seconds)"
