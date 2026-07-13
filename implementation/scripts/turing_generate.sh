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
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported; fi; }
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
    "optimization_decision": {"path": os.environ["OPTIMIZATION_DECISION"], "sha256": os.environ["OPTIMIZATION_DECISION_SHA256"]},
    "thinking_mode_selection": {"path": os.environ["THINKING_MODE_SELECTION"], "sha256": os.environ["THINKING_MODE_SELECTION_SHA256"]},
    "dataset": {"source": os.environ["DATASET_SOURCE"], "revision": os.environ["DATASET_REVISION"]},
    "shard": {key: os.environ[key] for key in ("SHARD_INDEX", "SHARD_COUNT", "SHARD_INPUT_SHA256", "MERGE_ID")},
    "generation": {"query_batch_size": int(os.environ["QUERY_BATCH_SIZE"]), "response_batch_size": int(os.environ["RESPONSE_BATCH_SIZE"]), "query_max_new_tokens": int(os.environ["QUERY_MAX_NEW_TOKENS"]), "response_max_new_tokens": int(os.environ["RESPONSE_MAX_NEW_TOKENS"]), "student_thinking_mode": os.environ["STUDENT_THINKING_MODE"], "scratchpad_max_new_tokens": int(os.environ["SCRATCHPAD_MAX_NEW_TOKENS"]), "query_temperature": float(os.environ["QUERY_TEMPERATURE"]), "response_temperature": float(os.environ["RESPONSE_TEMPERATURE"]), "top_p": float(os.environ["TOP_P"]), "top_k": int(os.environ["TOP_K"]), "k1": float(os.environ["BM25_K1"]), "b": float(os.environ["BM25_B"]), "max_length": 4096},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"
: "${SHARD_INDEX:?SHARD_INDEX must be supplied with --export}"
: "${SHARD_COUNT:?SHARD_COUNT must be supplied with --export}"
: "${SHARD_INPUT_SHA256:?SHARD_INPUT_SHA256 must be supplied with --export}"
: "${MERGE_ID:?MERGE_ID must be supplied with --export}"
: "${SLURM_NNODES:?SLURM_NNODES is required}"
[[ "$SLURM_NNODES" == "1" ]] || fail "generation requires exactly one node; got $SLURM_NNODES" multi_node_generation_forbidden
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
[[ "$SLURM_NTASKS" == "1" ]] || fail "generation requires exactly one task; got $SLURM_NTASKS" multi_task_generation_forbidden
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == "1" ]] || fail "generation requires exactly one allocated GPU; got $ALLOCATED_GPU_COUNT" generation_gpu_count
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ && "$SHARD_INDEX" -lt "$SHARD_COUNT" ]] || fail "invalid shard contract index=$SHARD_INDEX count=$SHARD_COUNT" shard_contract_invalid
ACTUAL_SHARD_INPUT_SHA256="$(sha256sum "$DATA" | awk '{print $1}')"
[[ "$ACTUAL_SHARD_INPUT_SHA256" == "$SHARD_INPUT_SHA256" ]] || fail "SHARD_INPUT_SHA256=$SHARD_INPUT_SHA256 differs from actual=$ACTUAL_SHARD_INPUT_SHA256" shard_input_identity_mismatch
: "${STUDENT_THINKING_MODE:?STUDENT_THINKING_MODE must be explicit and match the frozen decision}"
: "${SCRATCHPAD_MAX_NEW_TOKENS:?SCRATCHPAD_MAX_NEW_TOKENS must be explicit and match the frozen decision}"
: "${QUERY_TEMPERATURE:?QUERY_TEMPERATURE must be explicit and match the frozen decision}"
: "${RESPONSE_TEMPERATURE:?RESPONSE_TEMPERATURE must be explicit and match the frozen decision}"
: "${TOP_P:?TOP_P must be explicit and match the frozen decision}"
: "${TOP_K:?TOP_K must be explicit and match the frozen decision}"
: "${BM25_K1:?BM25_K1 must be explicit and match the frozen decision}"
: "${BM25_B:?BM25_B must be explicit and match the frozen decision}"
: "${OPTIMIZATION_DECISION:?OPTIMIZATION_DECISION must be supplied with --export}"
: "${OPTIMIZATION_DECISION_SHA256:?OPTIMIZATION_DECISION_SHA256 must be supplied with --export}"
: "${THINKING_MODE_SELECTION:?THINKING_MODE_SELECTION must be the selected-thinking-mode.manifest.json from prompt preflight}"
: "${THINKING_MODE_SELECTION_SHA256:?THINKING_MODE_SELECTION_SHA256 must hash THINKING_MODE_SELECTION}"

PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
[[ -x "$PROBE_RUNNER" ]] || fail "repository probe runner is not executable: $PROBE_RUNNER" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
CONFIG_IDENTITY="$(hash_path "$CONFIG")"
COMMIT_HASH="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
IFS=$'\t' read -r ATTENTION_IMPLEMENTATION QUERY_BATCH_SIZE RESPONSE_BATCH_SIZE QUERY_MAX_NEW_TOKENS RESPONSE_MAX_NEW_TOKENS FROZEN_THINKING_MODE FROZEN_SCRATCHPAD_MAX_NEW_TOKENS FROZEN_QUERY_TEMPERATURE FROZEN_RESPONSE_TEMPERATURE FROZEN_TOP_P FROZEN_TOP_K FROZEN_BM25_K1 FROZEN_BM25_B ATTENTION_FALLBACK_REASON VALIDATED_DECISION_SHA256 < <(
  run_probe_runner validate-decision --decision "$OPTIMIZATION_DECISION" --expected-sha256 "$OPTIMIZATION_DECISION_SHA256" --purpose generation --output-format generation-tsv \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_IDENTITY" --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-source "$DATASET_SOURCE" \
    --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$ACTUAL_SHARD_INPUT_SHA256" --prompt-sha256 "$PROMPT_HASH" \
    --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" --student-thinking-mode "$STUDENT_THINKING_MODE" \
    --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-temperature "$QUERY_TEMPERATURE" \
    --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
) || fail "frozen optimization decision validation failed" optimization_decision_invalid
[[ "$VALIDATED_DECISION_SHA256" == "$OPTIMIZATION_DECISION_SHA256" ]] || fail "validated decision hash changed" optimization_decision_hash_drift
IFS=$'\t' read -r SELECTED_THINKING_MODE VALIDATED_SELECTION_SHA256 SELECTION_FALLBACK_REASON < <(
  run_probe_runner validate-thinking-selection --manifest "$THINKING_MODE_SELECTION" --expected-sha256 "$THINKING_MODE_SELECTION_SHA256" \
    --expected-mode "$STUDENT_THINKING_MODE" --optimization-decision-sha256 "$OPTIMIZATION_DECISION_SHA256" \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_IDENTITY" --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$ACTUAL_SHARD_INPUT_SHA256" \
    --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
) || fail "selected thinking mode manifest validation failed" thinking_mode_selection_invalid
[[ "$SELECTED_THINKING_MODE" == "$FROZEN_THINKING_MODE" ]] || fail "selected thinking mode does not match optimization decision" thinking_mode_selection_mismatch
STUDENT_THINKING_MODE="$SELECTED_THINKING_MODE"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"

export ATTENTION_FALLBACK_REASON QUERY_BATCH_SIZE RESPONSE_BATCH_SIZE QUERY_MAX_NEW_TOKENS RESPONSE_MAX_NEW_TOKENS STUDENT_THINKING_MODE SCRATCHPAD_MAX_NEW_TOKENS QUERY_TEMPERATURE RESPONSE_TEMPERATURE TOP_P TOP_K BM25_K1 BM25_B THINKING_MODE_SELECTION THINKING_MODE_SELECTION_SHA256
CONFIG_HASH="$CONFIG_IDENTITY" MODEL_HASH="${MODEL_HASH:-$(hash_value "$MODEL|$MODEL_REVISION")}" DATASET_HASH="$ACTUAL_SHARD_INPUT_SHA256"
RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT.manifest.json}" GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}" ARTIFACT_PATHS="$OUTPUT|$DATA|$CONFIG"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl")))
PY
)"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT RUN_MANIFEST OPTIMIZATION_DECISION OPTIMIZATION_DECISION_SHA256 DATASET_SOURCE DATASET_REVISION SHARD_INDEX SHARD_COUNT SHARD_INPUT_SHA256 MERGE_ID
log_event runtime attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" query_batch_size="$QUERY_BATCH_SIZE" response_batch_size="$RESPONSE_BATCH_SIZE" max_length=4096
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

GEN_ARGS=(
  --data "$DATA" --output "$OUTPUT" --model "$MODEL" --model-revision "$MODEL_REVISION"
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --attention-implementation "$ATTENTION_IMPLEMENTATION"
  --prompt-version "$PROMPT_VERSION" --policy-hash "$POLICY_HASH" --student-thinking-mode "$SELECTED_THINKING_MODE"
  --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-batch-size "$QUERY_BATCH_SIZE"
  --response-batch-size "$RESPONSE_BATCH_SIZE" --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS"
  --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" --query-temperature "$QUERY_TEMPERATURE"
  --response-temperature "$RESPONSE_TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --k1 "$BM25_K1" --b "$BM25_B"
  --context-budget 4096
)
log_event generation_launch cli=generate-searchqa protocol=active-search attention_implementation="$ATTENTION_IMPLEMENTATION" student_thinking_mode="$SELECTED_THINKING_MODE" selection_sha256="$VALIDATED_SELECTION_SHA256" fallback_reason="$ATTENTION_FALLBACK_REASON"
uv run --frozen python -m text_feedback_dpo.cli generate-searchqa "${GEN_ARGS[@]}"
log_event generation_complete artifact="$OUTPUT"
write_manifest complete
