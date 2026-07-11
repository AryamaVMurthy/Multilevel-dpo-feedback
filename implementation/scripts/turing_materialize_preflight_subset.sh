#!/bin/bash
# CPU-only deterministic baseline preflight subset materialization.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

SOURCE_PATH="${SOURCE_PATH:?SOURCE_PATH is required}"
DATASET_MANIFEST="${DATASET_MANIFEST:?DATASET_MANIFEST is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
SUBSET_COUNT="${SUBSET_COUNT:?SUBSET_COUNT is required}"
SUBSET_SEED="${SUBSET_SEED:?SUBSET_SEED is required}"
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
if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "ERROR: source dataset does not exist: $SOURCE_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATASET_MANIFEST" ]]; then
  echo "ERROR: dataset manifest does not exist: $DATASET_MANIFEST" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"
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

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname)"
uv run --frozen python -m text_feedback_dpo.cli materialize-preflight-subset \
  --source-path "$SOURCE_PATH" \
  --output-path "$OUTPUT_PATH" \
  --count "$SUBSET_COUNT" \
  --seed "$SUBSET_SEED"

OUTPUT_MANIFEST="$(dirname "$OUTPUT_PATH")/manifest.json"
if [[ -e "$OUTPUT_MANIFEST" ]] && ! cmp -s "$DATASET_MANIFEST" "$OUTPUT_MANIFEST"; then
  echo "ERROR: preflight output directory has a conflicting manifest: $OUTPUT_MANIFEST" >&2
  exit 1
fi
if [[ ! -e "$OUTPUT_MANIFEST" ]]; then
  cp "$DATASET_MANIFEST" "$OUTPUT_MANIFEST"
fi
