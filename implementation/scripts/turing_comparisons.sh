#!/bin/bash
# Matched SFT/GRPO/DAPO comparison arms on one four-GPU node.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_comparisons
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" "gpu_count_parse_unsupported"; fi; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
manifest = {"status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"], "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(), "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE")}, "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "gpu_telemetry": os.environ["GPU_TELEMETRY"], "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": os.environ["ARTIFACT_PATHS"].split("|"), "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"), "max_length": 4096}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${BASE_MODEL:?BASE_MODEL must be supplied with --export}"
: "${BASE_REVISION:?BASE_REVISION must be supplied with --export}"
: "${SFT_TRAIN:?SFT_TRAIN must be supplied with --export}"
: "${SFT_EVAL:?SFT_EVAL must be supplied with --export}"
: "${RL_DATA:?RL_DATA must be supplied with --export}"
: "${VAL_DATA:?VAL_DATA must be supplied with --export}"
: "${TEST_DATA:?TEST_DATA must be supplied with --export}"
: "${DPO_METRICS:?DPO_METRICS must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${TRAIN_GPUS:?TRAIN_GPUS must be 4 for comparison training}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${CHECKPOINT_SMOKE_COMMAND:?CHECKPOINT_SMOKE_COMMAND must be supplied; refusing launch without a save gate}"
: "${RESUME_SMOKE_COMMAND:?RESUME_SMOKE_COMMAND must be supplied; refusing launch without a resume gate}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "invalid PROJECT_DIR" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}" HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT" logs
: "${SLURM_NNODES:?SLURM_NNODES is required}"; [[ "$SLURM_NNODES" == "1" ]] || fail "local torchrun requires one node; got $SLURM_NNODES" "multi_node_training_forbidden"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"; [[ "$SLURM_NTASKS" == "1" ]] || fail "local torchrun requires one task; got $SLURM_NTASKS" "multi_task_training_forbidden"
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$TRAIN_GPUS" == "4" ]] || fail "full comparison training requires TRAIN_GPUS=4; got $TRAIN_GPUS" "full_training_gpu_count"
[[ "$ALLOCATED_GPU_COUNT" == "$TRAIN_GPUS" ]] || fail "TRAIN_GPUS=$TRAIN_GPUS differs from allocated GPU count=$ALLOCATED_GPU_COUNT" "gpu_allocation_mismatch"

ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}" ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}" GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}" STUDENT_THINKING_MODE="${STUDENT_THINKING_MODE:-direct}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}" RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-4}" QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-32}" RESPONSE_MAX_NEW_TOKENS="${RESPONSE_MAX_NEW_TOKENS:-256}" SCRATCHPAD_MAX_NEW_TOKENS="${SCRATCHPAD_MAX_NEW_TOKENS:-128}"
COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="${MODEL_HASH:-$(hash_value "$BASE_MODEL@$BASE_REVISION")}" DATASET_HASH="${DATASET_HASH:-$(hash_path "$RL_DATA")}" RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT_ROOT/run-manifest.json}" GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}" ARTIFACT_PATHS="$OUTPUT_ROOT|$CONFIG|$SFT_TRAIN|$RL_DATA|$VAL_DATA|$TEST_DATA" MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl", "deepspeed", "bitsandbytes")))
PY
)"
export ATTENTION_FALLBACK_REASON COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH RUN_MANIFEST GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT PACKAGE_VERSIONS
log_event runtime attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" allocated_gpus="$ALLOCATED_GPU_COUNT" max_length=4096
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT
bash -c "$CHECKPOINT_SMOKE_COMMAND"
log_event checkpoint_smoke_passed
bash -c "$RESUME_SMOKE_COMMAND"
log_event resume_smoke_passed

TRAIN_COMMON=(--deepspeed-config configs/deepspeed_zero3.json --save-steps 100 --eval-steps 100)
if [[ -n "${EXISTING_SFT:-}" ]]; then
  SFT_MODEL="$EXISTING_SFT"; START_MODEL="$EXISTING_SFT"; START_REVISION=""; log_event comparison_sft source=existing_checkpoint model="$SFT_MODEL" fallback_reason=none
else
  uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli train-sft --config "$CONFIG" --train "$SFT_TRAIN" --eval "$SFT_EVAL" --model "$BASE_MODEL" --model-revision "$BASE_REVISION" --output "$OUTPUT_ROOT/sft" "${TRAIN_COMMON[@]}"
  SFT_MODEL="$OUTPUT_ROOT/sft/final"; START_MODEL="$SFT_MODEL"; START_REVISION=""
fi
RL_ARGS=(--config "$CONFIG" --train "$RL_DATA" --output "$OUTPUT_ROOT/grpo" --model "$START_MODEL" "${TRAIN_COMMON[@]}")
if [[ -n "$START_REVISION" ]]; then RL_ARGS+=(--model-revision "$START_REVISION"); fi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli train-grpo "${RL_ARGS[@]}"
RL_ARGS=(--config "$CONFIG" --train "$RL_DATA" --output "$OUTPUT_ROOT/dapo" --model "$START_MODEL" "${TRAIN_COMMON[@]}")
if [[ -n "$START_REVISION" ]]; then RL_ARGS+=(--model-revision "$START_REVISION"); fi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli train-dapo "${RL_ARGS[@]}"

for method in sft grpo dapo; do
  MODEL="$OUTPUT_ROOT/$method/final"; if [[ "$method" == "sft" ]]; then MODEL="$SFT_MODEL"; fi
  for split in validation test; do
    DATA="$VAL_DATA"; if [[ "$split" == "test" ]]; then DATA="$TEST_DATA"; fi
    uv run --frozen python -m text_feedback_dpo.cli generate-searchqa --data "$DATA" --output "$OUTPUT_ROOT/$method-$split-predictions.jsonl" --model "$MODEL" --attention-implementation "$ATTENTION_IMPLEMENTATION" --device cuda:0 --student-thinking-mode "$STUDENT_THINKING_MODE" --scratchpad-max-new-tokens "$SCRATCHPAD_MAX_NEW_TOKENS" --query-batch-size "$QUERY_BATCH_SIZE" --response-batch-size "$RESPONSE_BATCH_SIZE" --query-max-new-tokens "$QUERY_MAX_NEW_TOKENS" --response-max-new-tokens "$RESPONSE_MAX_NEW_TOKENS" --top-p 1.0 --top-k 8 --k1 1.2 --b 0.75 --context-budget 4096 --prompt-version fixed-retrieval-cited-v1 --policy-hash "${POLICY_HASH:?POLICY_HASH must be supplied with --export}:$method:$split"
    uv run --frozen python -m text_feedback_dpo.cli evaluate --data "$DATA" --predictions "$OUTPUT_ROOT/$method-$split-predictions.jsonl" --output "$OUTPUT_ROOT/$method-$split-metrics.json" --protocol active-search
  done
done
uv run --frozen python -m text_feedback_dpo.cli compare --run "dpo=$DPO_METRICS" --run "sft-validation=$OUTPUT_ROOT/sft-validation-metrics.json" --run "grpo-validation=$OUTPUT_ROOT/grpo-validation-metrics.json" --run "dapo-validation=$OUTPUT_ROOT/dapo-validation-metrics.json" --run "sft-test=$OUTPUT_ROOT/sft-test-metrics.json" --run "grpo-test=$OUTPUT_ROOT/grpo-test-metrics.json" --run "dapo-test=$OUTPUT_ROOT/dapo-test-metrics.json" --output "$OUTPUT_ROOT/comparison.json" --html "$OUTPUT_ROOT/comparison.html" --artifact "$DPO_METRICS" --artifact "$OUTPUT_ROOT/sft-validation-metrics.json" --artifact "$OUTPUT_ROOT/grpo-validation-metrics.json" --artifact "$OUTPUT_ROOT/dapo-validation-metrics.json" --artifact "$OUTPUT_ROOT/sft-test-metrics.json" --artifact "$OUTPUT_ROOT/grpo-test-metrics.json" --artifact "$OUTPUT_ROOT/dapo-test-metrics.json"
log_event comparisons_complete artifact="$OUTPUT_ROOT/comparison.json"
write_manifest complete
