#!/bin/bash
# CPU-only node-local cache verification and immutable decoding freeze.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
SCREENING_DIR="${SCREENING_DIR:?SCREENING_DIR is required}"
CONFIRMATION_DIR="${CONFIRMATION_DIR:?CONFIRMATION_DIR is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
CACHE_PRESERVE_OUTPUT="${CACHE_PRESERVE_OUTPUT:?CACHE_PRESERVE_OUTPUT is required}"
DECODING_FREEZE_OUTPUT="${DECODING_FREEZE_OUTPUT:?DECODING_FREEZE_OUTPUT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable" >&2
  exit 1
fi
if [[ "$MODEL_CACHE_DIR" != /scratch/* || "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: model cache and runtime must use node-local /scratch" >&2
  exit 1
fi
CACHE_MANIFEST="$MODEL_CACHE_DIR/tfdpo-model-cache-manifest.json"
if [[ ! -f "$CACHE_MANIFEST" ]]; then
  echo "ERROR: node-local model-cache manifest is missing: $CACHE_MANIFEST" >&2
  exit 1
fi
for output in "$CACHE_PRESERVE_OUTPUT" "$DECODING_FREEZE_OUTPUT"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite decoding freeze artifact: $output" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$output")"
done
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

python3 - "$CACHE_MANIFEST" "$CONFIRMATION_DIR/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
confirmation = Path(sys.argv[2])
actual = hashlib.sha256(cache.read_bytes()).hexdigest()
expected = json.loads(confirmation.read_text(encoding="utf-8"))["model_cache_manifest_sha256"]
if actual != expected:
    raise ValueError(f"model-cache selection hash mismatch: {actual} != {expected}")
print(f"selection_model_cache_sha256={actual}")
PY
cp "$CACHE_MANIFEST" "$CACHE_PRESERVE_OUTPUT"
uv run --frozen --no-sync python -m text_feedback_dpo.cli freeze-decoding \
  --config "$CONFIG" \
  --screening-dir "$SCREENING_DIR" \
  --confirmation-dir "$CONFIRMATION_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$DECODING_FREEZE_OUTPUT"
