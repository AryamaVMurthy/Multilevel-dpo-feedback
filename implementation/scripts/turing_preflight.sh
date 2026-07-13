#!/bin/bash
# Submit with: sbatch -A <account> scripts/turing_preflight.sh
#SBATCH -p u22
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${MODEL_ID:?MODEL_ID must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${MODEL_ROLE:?MODEL_ROLE must be student or teacher}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || { echo "ERROR: PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs

set -x
hostname
nvidia-smi
uv run --frozen python - <<'PY'
import json
import torch
import transformers
import trl
import deepspeed
import bitsandbytes

try:
    import flash_attn
    flash_attention = {"available": True, "version": getattr(flash_attn, "__version__", "unknown")}
except ImportError as exc:
    flash_attention = {"available": False, "fallback_reason": f"flash_attn unavailable: {exc}"}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback")
print(json.dumps({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "deepspeed": deepspeed.__version__,
    "bitsandbytes": bitsandbytes.__version__,
    "flash_attn": flash_attention,
    "attention_implementation": "flash_attention_2" if flash_attention["available"] else "sdpa",
    "cuda": torch.cuda.get_device_name(0),
}, sort_keys=True))
PY

uv run --frozen python -m text_feedback_dpo.cli probe-model \
  --role "$MODEL_ROLE" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --teacher-quantization 4bit --attention-implementation sdpa --output "$OUTPUT"
