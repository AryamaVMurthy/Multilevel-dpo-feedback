#!/bin/bash
# Install the locked project environment once; model/data caches remain node-local scratch.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
SHARED_UV_CACHE="${SHARED_UV_CACHE:?SHARED_UV_CACHE is required}"
SHARED_PROJECT_ENV="${SHARED_PROJECT_ENV:?SHARED_PROJECT_ENV is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing hidden cache fallback" >&2
  exit 1
fi
if [[ "$SHARED_UV_CACHE" != /scratch/* || "$SHARED_PROJECT_ENV" != /scratch/* ]]; then
  echo "ERROR: project environment and cache must use node-local /scratch" >&2
  exit 1
fi
mkdir -p "$SHARED_UV_CACHE" "$SHARED_PROJECT_ENV"
export UV_CACHE_DIR="$SHARED_UV_CACHE"
export UV_PROJECT_ENVIRONMENT="$SHARED_PROJECT_ENV"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
uv sync --frozen
uv run --frozen python -c 'import sys; print(sys.executable); print(sys.version)' \
  > "$SHARED_PROJECT_ENV/environment_verified.txt"
