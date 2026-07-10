#!/bin/bash
# Build the isolated vLLM GRPO environment on an allocated compute node.
#SBATCH -p u22
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
GRPO_ENVIRONMENT="${GRPO_ENVIRONMENT:?GRPO_ENVIRONMENT is required}"
GRPO_UV_CACHE="${GRPO_UV_CACHE:?GRPO_UV_CACHE is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ "$GRPO_ENVIRONMENT" != /scratch/* || "$GRPO_UV_CACHE" != /scratch/* ]]; then
  echo "ERROR: GRPO environment and cache must use node-local /scratch" >&2
  exit 1
fi
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home storage fallback" >&2
  exit 1
fi
mkdir -p "$GRPO_ENVIRONMENT" "$GRPO_UV_CACHE"
export UV_PROJECT_ENVIRONMENT="$GRPO_ENVIRONMENT"
export UV_CACHE_DIR="$GRPO_UV_CACHE"
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"

echo "account=$TURING_ACCOUNT host=$(hostname) scratch_before=$(df -h /scratch | tail -1)"
uv sync --project "$PROJECT_DIR/environments/grpo" --frozen --no-dev
uv run --project "$PROJECT_DIR/environments/grpo" --frozen --no-sync python -c \
  'import torch, transformers, trl, vllm; print({"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "vllm": vllm.__version__})'
echo "scratch_after=$(df -h /scratch | tail -1)"
