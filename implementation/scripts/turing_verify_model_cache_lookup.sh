#!/bin/bash
# CPU-only proof that the pinned model IDs resolve from the staged offline cache.
# Submit on the same constrained node that will run subsequent GPU jobs.
#SBATCH -p u22
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
CACHE_LOOKUP_OUTPUT="${CACHE_LOOKUP_OUTPUT:?CACHE_LOOKUP_OUTPUT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing home cache fallback" >&2
  exit 1
fi
if [[ "$MODEL_CACHE_DIR" != /scratch/* ]]; then
  echo "ERROR: MODEL_CACHE_DIR must be node-local /scratch storage: $MODEL_CACHE_DIR" >&2
  exit 1
fi
if [[ "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: RUNTIME_ROOT must be node-local /scratch storage: $RUNTIME_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: paper config does not exist: $CONFIG" >&2
  exit 1
fi
CACHE_MANIFEST="$MODEL_CACHE_DIR/tfdpo-model-cache-manifest.json"
if [[ ! -f "$CACHE_MANIFEST" ]]; then
  echo "ERROR: staged model-cache manifest is missing: $CACHE_MANIFEST" >&2
  exit 1
fi
if [[ -e "$CACHE_LOOKUP_OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite model-cache lookup artifact: $CACHE_LOOKUP_OUTPUT" >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT" "$(dirname "$CACHE_LOOKUP_OUTPUT")"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export UV_LINK_MODE=hardlink
export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) cache_lookup=offline"
uv run --frozen --no-sync python - "$CONFIG" "$CACHE_MANIFEST" "$CACHE_LOOKUP_OUTPUT" <<'PY'
import hashlib
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoTokenizer

from text_feedback_dpo.experiment_config import load_paper_experiment, validate_paper_experiment
from text_feedback_dpo.io import write_json_atomic


config_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
config = load_paper_experiment(config_path)
validate_paper_experiment(config)
config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "tfdpo-model-cache-v1":
    raise ValueError("staged model-cache manifest schema mismatch")
if manifest.get("config_sha256") != config_sha256:
    raise ValueError("staged model-cache config hash does not match the frozen config")

manifest_models = {}
for entry in manifest.get("models", []):
    identity = (entry.get("id"), entry.get("revision"))
    if identity in manifest_models:
        raise ValueError(f"duplicate staged model identity: {identity}")
    snapshot = Path(str(entry.get("snapshot_path", "")))
    if not snapshot.is_dir():
        raise ValueError(f"staged model snapshot is missing: {snapshot}")
    manifest_models[identity] = snapshot

required = []
seen = set()
for role in ("student", "teacher", "evaluator"):
    model = config.models[role]
    identity = (model["id"], model["revision"])
    if identity not in seen:
        seen.add(identity)
        required.append((role, *identity))
if set(manifest_models) != {(model_id, revision) for _, model_id, revision in required}:
    raise ValueError("staged model identities do not exactly match the frozen config")

checks = []
for role, model_id, revision in required:
    started = time.perf_counter()
    resolved_config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )
    if resolved_config.model_type != "qwen3":
        raise ValueError(f"unexpected model_type for {model_id}: {resolved_config.model_type}")
    if not tokenizer.chat_template:
        raise ValueError(f"tokenizer chat template is missing for {model_id}")
    checks.append(
        {
            "role": role,
            "id": model_id,
            "revision": revision,
            "model_type": resolved_config.model_type,
            "tokenizer_class": type(tokenizer).__name__,
            "snapshot_path": str(manifest_models[(model_id, revision)]),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )

artifact = {
    "schema": "paper-model-cache-lookup-v1",
    "status": "passed",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "config_sha256": config_sha256,
    "offline": True,
    "local_files_only": True,
    "transformers_version": transformers.__version__,
    "models": checks,
}
write_json_atomic(output_path, artifact)
print(json.dumps(artifact, sort_keys=True))
PY
