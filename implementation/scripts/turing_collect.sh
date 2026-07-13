#!/bin/bash
# One deterministic teacher/student collection shard: exactly two GPUs, one Slurm task.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_collect
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
allocated_gpu_count() {
  local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required inside the allocation}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" "gpu_count_parse_unsupported"; fi
}
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
def values(name): return [item for item in os.environ.get(name, "").split("|") if item]
manifest = {
    "status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"],
    "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(),
    "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE")},
    "package_versions": values("PACKAGE_VERSIONS"), "gpu_telemetry": os.environ.get("GPU_TELEMETRY"),
    "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": values("ARTIFACT_PATHS"),
    "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"), "shard": {k: os.environ[k] for k in ("SHARD_INDEX", "SHARD_COUNT", "SHARD_SEED", "SHARD_INPUT_SHA256", "MERGE_ID")},
    "dataset": {"source": os.environ["DATASET_SOURCE"], "revision": os.environ["DATASET_REVISION"]},
    "collection": {"teacher_device": "cuda:0", "student_device": "cuda:1", "teacher_quantization": "4bit", "max_length": 4096},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied with --export}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied with --export}"
: "${PROMPT_VERSION:?PROMPT_VERSION must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${SEED:?SEED must be supplied with --export}"
: "${SHARD_INDEX:?SHARD_INDEX must be supplied with --export}"
: "${SHARD_COUNT:?SHARD_COUNT must be supplied with --export}"
: "${SHARD_SEED:?SHARD_SEED must be supplied with --export}"
: "${MERGE_ID:?MERGE_ID must be supplied with --export}"
: "${SHARD_INPUT_SHA256:?SHARD_INPUT_SHA256 must be supplied with --export}"
: "${TRAJECTORY_CACHE:?TRAJECTORY_CACHE must be supplied with --export}"
: "${POLICY_HASH:?POLICY_HASH must be supplied with --export}"

: "${SLURM_NNODES:?SLURM_NNODES is required inside the allocation}"
if [[ "$SLURM_NNODES" != "1" ]]; then fail "collection requires one node; got $SLURM_NNODES" "multi_node_collection_forbidden"; fi
: "${SLURM_NTASKS:?SLURM_NTASKS is required inside the allocation}"
if [[ "$SLURM_NTASKS" != "1" ]]; then fail "collection requires one Slurm task; got $SLURM_NTASKS" "multi_task_collection_forbidden"; fi
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
if [[ "$ALLOCATED_GPU_COUNT" != "2" ]]; then fail "collection requires exactly two allocated GPUs; got $ALLOCATED_GPU_COUNT" "collection_gpu_count"; fi
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ && "$SHARD_INDEX" -lt "$SHARD_COUNT" ]] || fail "invalid shard contract index=$SHARD_INDEX count=$SHARD_COUNT" shard_contract_invalid
ACTUAL_SHARD_INPUT_SHA256="$(sha256sum "$DATA" | awk '{print $1}')"
[[ "$ACTUAL_SHARD_INPUT_SHA256" == "$SHARD_INPUT_SHA256" ]] || fail "SHARD_INPUT_SHA256=$SHARD_INPUT_SHA256 differs from actual=$ACTUAL_SHARD_INPUT_SHA256" shard_input_identity_mismatch

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"

ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}" ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-32}" TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}" ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-32}"
SCRATCHPAD_MAX_NEW_TOKENS="${SCRATCHPAD_MAX_NEW_TOKENS:-128}" STUDENT_THINKING_MODE="${STUDENT_THINKING_MODE:-direct}"
RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT.manifest.json}" GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}" ARTIFACT_PATHS="$OUTPUT|$TRAJECTORY_CACHE|$DATA"
COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "${CONFIG:?CONFIG must be supplied with --export}")" MODEL_HASH="${MODEL_HASH:-$(hash_value "$STUDENT_MODEL@$STUDENT_REVISION|teacher=$TEACHER_MODEL@$TEACHER_REVISION")}" DATASET_HASH="$ACTUAL_SHARD_INPUT_SHA256"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl", "deepspeed", "bitsandbytes")))
PY
)"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export ATTENTION_FALLBACK_REASON COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT RUN_MANIFEST SHARD_INDEX SHARD_COUNT SHARD_SEED SHARD_INPUT_SHA256 MERGE_ID DATASET_SOURCE DATASET_REVISION
log_event runtime attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" shard_index="$SHARD_INDEX" shard_count="$SHARD_COUNT" merge_id="$MERGE_ID" allocated_gpus="$ALLOCATED_GPU_COUNT"
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

COLLECT_ARGS=(
  --data "$DATA" --output "$OUTPUT" --student-model "$STUDENT_MODEL" --student-revision "$STUDENT_REVISION"
  --teacher-model "$TEACHER_MODEL" --teacher-revision "$TEACHER_REVISION" --dataset-revision "$DATASET_REVISION"
  --prompt-version "$PROMPT_VERSION" --seed "$SHARD_SEED" --teacher-quantization 4bit
  --attention-implementation "$ATTENTION_IMPLEMENTATION" --student-device cuda:1 --teacher-device cuda:0
  --trajectory-cache "$TRAJECTORY_CACHE" --policy-hash "$POLICY_HASH" --max-interventions 4
  --student-batch-size "$STUDENT_BATCH_SIZE" --teacher-batch-size "$TEACHER_BATCH_SIZE"
  --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS"
  --answer-max-new-tokens "$ANSWER_MAX_NEW_TOKENS" --teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS" --teacher-thinking
)
log_event collection_launch teacher_device=cuda:0 student_device=cuda:1 teacher_batch_size="$TEACHER_BATCH_SIZE" student_batch_size="$STUDENT_BATCH_SIZE" max_length=4096
uv run --frozen python -m text_feedback_dpo.cli collect "${COLLECT_ARGS[@]}"
log_event collection_complete artifact="$OUTPUT" merge_id="$MERGE_ID"
write_manifest complete
