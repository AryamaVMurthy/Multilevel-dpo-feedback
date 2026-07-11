#!/bin/bash
# CPU-only exact row lookup for reproducible one-shard diagnostics.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
ROW_ID="${ROW_ID:?ROW_ID is required}"
LOOKUP_OUTPUT="${LOOKUP_OUTPUT:?LOOKUP_OUTPUT is required}"
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
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: dataset path does not exist: $DATA_PATH" >&2
  exit 1
fi
if [[ -e "$LOOKUP_OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite dataset-row lookup: $LOOKUP_OUTPUT" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT" "$(dirname "$LOOKUP_OUTPUT")"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) row_id=${ROW_ID}"
uv run --frozen --no-sync python - "$DATA_PATH" "$ROW_ID" "$LOOKUP_OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from text_feedback_dpo.io import read_jsonl, read_jsonl_zst, write_json_atomic


data_path = Path(sys.argv[1])
row_id = sys.argv[2]
output_path = Path(sys.argv[3])
rows = read_jsonl_zst(data_path) if data_path.name.endswith(".zst") else read_jsonl(data_path)
matches = [(index, row) for index, row in enumerate(rows) if row.get("id") == row_id]
if len(matches) != 1:
    raise ValueError(f"dataset row lookup requires exactly one match for {row_id}; found {len(matches)}")
index, row = matches[0]
result = {
    "schema": "paper-dataset-row-lookup-v1",
    "data_path": str(data_path),
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "row_count": len(rows),
    "row_id": row_id,
    "zero_based_index": index,
    "row_hash": row.get("row_hash"),
    "stratum": row.get("stratum"),
}
write_json_atomic(output_path, result)
print(json.dumps(result, sort_keys=True))
PY
