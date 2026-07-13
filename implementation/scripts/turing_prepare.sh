#!/bin/bash
# Prepare aligned SearchQA records; no model fallback is permitted.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_prepare
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
def values(name): return [item for item in os.environ.get(name, "").split("|") if item]
manifest = {"status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"], "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(), "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS")}, "package_versions": values("PACKAGE_VERSIONS"), "gpu_telemetry": os.environ["GPU_TELEMETRY"], "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": values("ARTIFACT_PATHS"), "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"), "max_length": 4096}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${SPLIT:?SPLIT must be train, validation, or test}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${TOKENIZER_MODEL:?TOKENIZER_MODEL must be supplied with --export}"
: "${TOKENIZER_REVISION:?TOKENIZER_REVISION must be supplied with --export}"
: "${SEARCHQA_REVISION:?SEARCHQA_REVISION must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"
COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="${MODEL_HASH:-$(hash_value "$TOKENIZER_MODEL@$TOKENIZER_REVISION")}" DATASET_HASH="${DATASET_HASH:-$(hash_value "kyunghyuncho/search_qa@$SEARCHQA_REVISION/$SPLIT")}" PROMPT_VERSION="${PROMPT_VERSION:-fixed-retrieval-cited-v1}" ATTENTION_FALLBACK_REASON="none" GPU_TELEMETRY="not_allocated" RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT.manifest.json}" ARTIFACT_PATHS="$OUTPUT|$CONFIG"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("datasets", "transformers")))
PY
)"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY MANIFEST_STARTED_AT RUN_MANIFEST ARTIFACT_PATHS ATTENTION_FALLBACK_REASON
log_event prepare_start split="$SPLIT" source=kyunghyuncho/search_qa revision="$SEARCHQA_REVISION" max_length=4096 fallback_reason=none
uv run --frozen python -m text_feedback_dpo.cli prepare-searchqa --source kyunghyuncho/search_qa --split "$SPLIT" --revision "$SEARCHQA_REVISION" --tokenizer-model "$TOKENIZER_MODEL" --tokenizer-revision "$TOKENIZER_REVISION" --output "$OUTPUT" --max-evidence-tokens 3000
log_event prepare_complete artifact="$OUTPUT"
write_manifest complete
