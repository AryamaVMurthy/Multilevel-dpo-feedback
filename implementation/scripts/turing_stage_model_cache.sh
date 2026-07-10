#!/bin/bash
# Download exact paper model revisions to one compute node's local scratch cache.
# Submit with an explicit node constraint, then use the same node for GPU evaluation.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home model-cache fallback" >&2
  exit 1
fi
if [[ "$MODEL_CACHE_DIR" != /scratch/* ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be node-local /scratch storage: $MODEL_CACHE_DIR" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: paper config does not exist: $CONFIG" >&2
  exit 1
fi

mkdir -p "$MODEL_CACHE_DIR"
export UV_CACHE_DIR="$HOME/tfdpo-runs/uv_cache"
export UV_PROJECT_ENVIRONMENT="$HOME/tfdpo-runs/project_venv"
export UV_LINK_MODE=hardlink
export HF_HOME="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR/datasets"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) source_commit=${SOURCE_COMMIT}"
echo "model_cache_dir=${MODEL_CACHE_DIR}"

uv run --frozen python - "$CONFIG" "$MODEL_CACHE_DIR" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

config_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
source_commit = sys.argv[3]
config_bytes = config_path.read_bytes()
config = yaml.safe_load(config_bytes)
models = config.get("models")
if not isinstance(models, dict):
    raise ValueError("paper config is missing models mapping")

requested: list[dict[str, str]] = []
seen: set[tuple[str, str]] = set()
for role in ("student", "teacher", "evaluator"):
    model = models.get(role)
    if not isinstance(model, dict) or not model.get("id") or not model.get("revision"):
        raise ValueError(f"paper config is missing pinned {role} model id or revision")
    identity = (str(model["id"]), str(model["revision"]))
    if identity not in seen:
        seen.add(identity)
        requested.append({"role": role, "id": identity[0], "revision": identity[1]})

snapshots: list[dict[str, str]] = []
for model in requested:
    snapshot = snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        cache_dir=str(cache_dir),
    )
    snapshots.append({**model, "snapshot_path": snapshot})

manifest = {
    "schema": "tfdpo-model-cache-v1",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "source_commit": source_commit,
    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    "models": snapshots,
}
target = cache_dir / "tfdpo-model-cache-manifest.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(manifest, sort_keys=True))
PY
