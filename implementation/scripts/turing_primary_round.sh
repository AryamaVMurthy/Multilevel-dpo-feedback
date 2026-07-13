#!/bin/bash
# Resumable round orchestrator. Collection and full training are separate Slurm allocations.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_primary_round
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
manifest = {"status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"], "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(), "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS")}, "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "gpu_telemetry": os.environ["GPU_TELEMETRY"], "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": os.environ["ARTIFACT_PATHS"].split("|"), "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"), "max_length": 4096, "merge_id": os.environ["MERGE_ID"]}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${EVAL_DATA:?EVAL_DATA must be supplied with --export}"
: "${EVAL_PREFERENCES:?EVAL_PREFERENCES must be supplied with --export}"
: "${ROUND_DIR:?ROUND_DIR must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${DATASET_HASH:?DATASET_HASH must be supplied with --export}"
: "${MERGE_ID:?MERGE_ID must be supplied with --export}"
: "${SHARD_INDEX:?SHARD_INDEX must be supplied with --export}"
: "${SHARD_COUNT:?SHARD_COUNT must be supplied with --export}"
: "${COLLECTION_SCRIPT:?COLLECTION_SCRIPT must identify the two-GPU collection script}"
: "${COLLECTION_EXPORT:?COLLECTION_EXPORT must contain the complete collection environment contract}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT must identify the four-GPU training script}"
: "${TRAIN_EXPORT:?TRAIN_EXPORT must contain the complete training environment contract}"
: "${GENERATION_SCRIPT:?GENERATION_SCRIPT must identify the one-GPU generation script}"
: "${GENERATION_EXPORT:?GENERATION_EXPORT must contain the complete generation environment contract}"
GENERATION_COMMAND=generate-searchqa
GENERATION_PROTOCOL=active-search

cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$ROUND_DIR" logs
[[ "${TRAIN_GPUS:-4}" == "4" ]] || fail "primary full training requires TRAIN_GPUS=4" "full_training_gpu_count"
TRAIN_GPUS=4
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}" ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}" RUN_MANIFEST="${RUN_MANIFEST:-$ROUND_DIR/run-manifest.json}" GPU_TELEMETRY="not_allocated" ARTIFACT_PATHS="$ROUND_DIR|$DATA|$CONFIG" MANIFEST_STARTED_AT="$(date -u +%FT%TZ)" COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="${MODEL_HASH:-$(hash_value "${STUDENT_MODEL:-unknown}")}"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl")))
PY
)"
export ATTENTION_FALLBACK_REASON RUN_MANIFEST GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT COMMIT_HASH CONFIG_HASH MODEL_HASH PACKAGE_VERSIONS
log_event round_start merge_id="$MERGE_ID" shard_index="$SHARD_INDEX" shard_count="$SHARD_COUNT" attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" max_length=4096

TRAJECTORIES="$ROUND_DIR/trajectories-${MERGE_ID}.jsonl"
PREFERENCES="$ROUND_DIR/preferences-${MERGE_ID}.jsonl"
DPO_OUT="$ROUND_DIR/dpo"
PREDICTIONS="$ROUND_DIR/validation-predictions.jsonl"
METRICS="$ROUND_DIR/validation-metrics.json"
COLLECTION_ENV="ALL,PROJECT_DIR=$PROJECT_DIR,DATA=$DATA,OUTPUT=$TRAJECTORIES,TRAJECTORY_CACHE=$ROUND_DIR/trajectory-cache-${MERGE_ID}.jsonl,SHARD_INDEX=$SHARD_INDEX,SHARD_COUNT=$SHARD_COUNT,MERGE_ID=$MERGE_ID,SHARD_SEED=${SHARD_SEED:?SHARD_SEED must be supplied with --export},CONFIG=$CONFIG,PROMPT_HASH=$PROMPT_HASH,RETRIEVAL_HASH=$RETRIEVAL_HASH,SOURCE_SCHEMA_HASH=$SOURCE_SCHEMA_HASH,$COLLECTION_EXPORT"
log_event collect_submit nodes=1 gpus=2 tasks=1 merge_id="$MERGE_ID"
sbatch --wait --nodes=1 --ntasks=1 --gres=gpu:2 --export="$COLLECTION_ENV" "$COLLECTION_SCRIPT"
uv run --frozen python -m text_feedback_dpo.cli build-preferences --trajectories "$TRAJECTORIES" --output "$PREFERENCES"

TRAIN_ENV="ALL,PROJECT_DIR=$PROJECT_DIR,METHOD=dpo,TRAIN_GPUS=4,CONFIG=$CONFIG,TRAIN=$PREFERENCES,EVAL=$EVAL_PREFERENCES,OUTPUT=$DPO_OUT,PROMPT_HASH=$PROMPT_HASH,RETRIEVAL_HASH=$RETRIEVAL_HASH,SOURCE_SCHEMA_HASH=$SOURCE_SCHEMA_HASH,$TRAIN_EXPORT"
TRAIN_METHOD=train-dpo
log_event train_submit stage="$TRAIN_METHOD" nodes=1 gpus=4 tasks=1 train_gpus=4 checkpoint_gate=required resume_gate=required
sbatch --wait --nodes=1 --ntasks=1 --gres=gpu:4 --export="$TRAIN_ENV" "$TRAIN_SCRIPT"

GEN_ENV="ALL,PROJECT_DIR=$PROJECT_DIR,DATA=$EVAL_DATA,OUTPUT=$PREDICTIONS,CONFIG=$CONFIG,PROMPT_HASH=$PROMPT_HASH,RETRIEVAL_HASH=$RETRIEVAL_HASH,SOURCE_SCHEMA_HASH=$SOURCE_SCHEMA_HASH,$GENERATION_EXPORT"
log_event generate_submit nodes=1 gpus=1 tasks=1
[[ "$(basename "$GENERATION_SCRIPT")" == "turing_generate.sh" ]] || fail "primary active-search path requires turing_generate.sh; legacy generation is forbidden" "legacy_generation_path"
sbatch --wait --nodes=1 --ntasks=1 --gres=gpu:1 --export="$GEN_ENV" "$GENERATION_SCRIPT"
uv run --frozen python -m text_feedback_dpo.cli evaluate --data "$EVAL_DATA" --predictions "$PREDICTIONS" --output "$METRICS" --protocol active-search
uv run --frozen python -m text_feedback_dpo.cli report --metrics "$METRICS" --output "$ROUND_DIR/report.html" --artifact "$TRAJECTORIES" --artifact "$PREFERENCES" --artifact "$PREDICTIONS" --artifact "$METRICS"
log_event round_complete merge_id="$MERGE_ID" artifact="$ROUND_DIR/report.html"
write_manifest complete
