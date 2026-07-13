#!/bin/bash
# Active fixed-retrieval generation on one GPU.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_generate
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
def values(name): return [item for item in os.environ.get(name, "").split("|") if item]
manifest = {
    "status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"],
    "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"],
    "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"],
    "node": socket.gethostname(), "platform": platform.platform(),
    "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE")},
    "package_versions": values("PACKAGE_VERSIONS"), "gpu_telemetry": os.environ.get("GPU_TELEMETRY"),
    "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]},
    "artifact_paths": values("ARTIFACT_PATHS"), "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"),
    "generation": {"query_batch_size": int(os.environ["QUERY_BATCH_SIZE"]), "response_batch_size": int(os.environ["RESPONSE_BATCH_SIZE"]), "query_max_new_tokens": int(os.environ["QUERY_MAX_NEW_TOKENS"]), "response_max_new_tokens": int(os.environ["RESPONSE_MAX_NEW_TOKENS"]), "max_length": 4096},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"

ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"
RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-4}"
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-32}"
RESPONSE_MAX_NEW_TOKENS="${RESPONSE_MAX_NEW_TOKENS:-256}"
SCRATCHPAD_MAX_NEW_TOKENS="${SCRATCHPAD_MAX_NEW_TOKENS:-128}"
QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.0}"
RESPONSE_TEMPERATURE="${RESPONSE_TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-8}"
BM25_K1="${BM25_K1:-1.2}"
BM25_B="${BM25_B:-0.75}"
STUDENT_THINKING_MODE="${STUDENT_THINKING_MODE:-direct}"
export ATTENTION_FALLBACK_REASON QUERY_BATCH_SIZE RESPONSE_BATCH_SIZE QUERY_MAX_NEW_TOKENS RESPONSE_MAX_NEW_TOKENS
COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="${MODEL_HASH:-$(hash_value "$MODEL|${MODEL_REVISION:-none}")}" DATASET_HASH="${DATASET_HASH:-$(hash_path "$DATA")}"
RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT.manifest.json}" GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}" ARTIFACT_PATHS="$OUTPUT|$DATA|$CONFIG"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl")))
PY
)"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT RUN_MANIFEST
log_event runtime attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" query_batch_size="$QUERY_BATCH_SIZE" response_batch_size="$RESPONSE_BATCH_SIZE" max_length=4096
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

GEN_ARGS=(
  --data "$DATA" --output "$OUTPUT" --model "$MODEL" --attention-implementation "$ATTENTION_IMPLEMENTATION"
  --prompt-version "$PROMPT_VERSION" --policy-hash "$POLICY_HASH" --student-thinking-mode "$STUDENT_THINKING_MODE"
  --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-batch-size "$QUERY_BATCH_SIZE"
  --response-batch-size "$RESPONSE_BATCH_SIZE" --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS"
  --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" --query-temperature "$QUERY_TEMPERATURE"
  --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
  --context-budget 4096
)
if [[ -n "${MODEL_REVISION:-}" ]]; then GEN_ARGS+=(--model-revision "$MODEL_REVISION"); fi
if [[ -z "${STUDENT_THINKING_MODE:-}" ]]; then fail "STUDENT_THINKING_MODE must be explicit" "missing_thinking_mode"; fi
log_event generation_launch cli=generate-searchqa protocol=active-search attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON"
uv run --frozen python -m text_feedback_dpo.cli generate-searchqa "${GEN_ARGS[@]}"
log_event generation_complete artifact="$OUTPUT"
write_manifest complete
