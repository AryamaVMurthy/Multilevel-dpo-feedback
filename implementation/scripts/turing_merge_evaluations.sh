#!/bin/bash
# CPU-only strict merge of completed teacher-free evaluation shards.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
SPLIT="${SPLIT:?SPLIT is required}"
SHARD_ROOT="${SHARD_ROOT:?SHARD_ROOT is required}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:?EXPECTED_SHARDS is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:?CHECKPOINT_KIND is required}"
FREEZE_MANIFEST="${FREEZE_MANIFEST:?FREEZE_MANIFEST is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
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
mkdir -p "$RUNTIME_ROOT"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

uv run --frozen python -m text_feedback_dpo.cli merge-evaluations \
  --config "$CONFIG" \
  --data "$DATA_PATH" \
  --split "$SPLIT" \
  --shard-root "$SHARD_ROOT" \
  --expected-shards "$EXPECTED_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-kind "$CHECKPOINT_KIND" \
  --freeze-manifest "$FREEZE_MANIFEST" \
  --source-commit "$SOURCE_COMMIT"
