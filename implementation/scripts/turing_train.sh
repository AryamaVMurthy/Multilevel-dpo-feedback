#!/bin/bash
# Full Qwen3-4B training. Submit with the frozen four- or eight-GPU Slurm allocation.
# sbatch -A <account> --export=ALL,PROJECT_DIR=...,METHOD=dpo,... scripts/turing_train.sh
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_train
log_event() {
  local event="$1"
  shift
  printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"
}
fail() {
  log_event failure reason="$1" fallback_reason="${2:-none}" >&2
  exit 2
}
hash_value() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}
hash_path() {
  local path="$1"
  if [[ -f "$path" ]]; then
    sha256sum "$path" | awk '{print $1}'
  elif [[ -d "$path" ]]; then
    find "$path" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
  else
    hash_value "$path"
  fi
}
allocated_gpu_count() {
  local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required inside the allocation}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$raw"
  elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    fail "unsupported SLURM_GPUS_ON_NODE format: $raw" "gpu_count_parse_unsupported"
  fi
}
write_manifest() {
  local status="$1"
  local ended_at
  ended_at="$(date -u +%FT%TZ)"
  export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$ended_at"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json
import os
import platform
import socket
import sys

def values(name):
    return [item for item in os.environ.get(name, "").split("|") if item]

manifest = {
    "status": os.environ["MANIFEST_STATUS"],
    "commit_hash": os.environ["COMMIT_HASH"],
    "config_hash": os.environ["CONFIG_HASH"],
    "model_hash": os.environ["MODEL_HASH"],
    "dataset_hash": os.environ["DATASET_HASH"],
    "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"],
    "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"],
    "node": socket.gethostname(),
    "platform": platform.platform(),
    "slurm_allocation": {key: os.environ.get(key) for key in (
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS",
        "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "TRAIN_GPUS",
    )},
    "package_versions": values("PACKAGE_VERSIONS"),
    "gpu_telemetry": os.environ.get("GPU_TELEMETRY"),
    "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]},
    "artifact_paths": values("ARTIFACT_PATHS"),
    "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"),
    "optimization_decision": {"path": os.environ["OPTIMIZATION_DECISION"], "sha256": os.environ["OPTIMIZATION_DECISION_SHA256"]},
    "scale_decision": {"path": os.environ["SCALE_DECISION"], "sha256": os.environ["SCALE_DECISION_SHA256"]},
    "checkpoint_gate": {"path": os.environ["CHECKPOINT_SMOKE_MANIFEST"], "sha256": os.environ["CHECKPOINT_SMOKE_MANIFEST_SHA256"]},
    "dataset": {"source": os.environ["DATASET_SOURCE"], "revision": os.environ["DATASET_REVISION"]},
    "training_contract": {
        "bf16": True,
        "tf32": True,
        "deepspeed": "zero3",
        "optimizer": "fused_adamw",
        "gradient_checkpointing": "non-reentrant",
        "max_length": 4096,
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, sort_keys=True, indent=2)
    handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${METHOD:?METHOD must be sft, dpo, grpo, or dapo}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${TRAIN:?TRAIN must be supplied with --export}"
: "${EVAL:?EVAL must be the explicit evaluation dataset for all methods}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${TRAIN_GPUS:?TRAIN_GPUS must be supplied with --export}"
: "${EFFECTIVE_BATCH_SIZE:?EFFECTIVE_BATCH_SIZE must be supplied with --export}"
: "${LEARNING_RATE:?LEARNING_RATE must be supplied with --export}"
: "${EPOCHS:?EPOCHS must be supplied with --export}"
: "${SAVE_STEPS:?SAVE_STEPS must be supplied with --export}"
: "${EVAL_STEPS:?EVAL_STEPS must be supplied with --export}"
: "${CHECKPOINT_SMOKE_MANIFEST:?CHECKPOINT_SMOKE_MANIFEST must be produced by turing_checkpoint_smoke.sh}"
: "${CHECKPOINT_SMOKE_MANIFEST_SHA256:?CHECKPOINT_SMOKE_MANIFEST_SHA256 must be supplied}"
: "${START_MODEL:?START_MODEL must explicitly pin the full-training start model}"
: "${START_REVISION:?START_REVISION must explicitly pin the full-training start revision}"
: "${OPTIMIZATION_DECISION:?OPTIMIZATION_DECISION must be supplied}"
: "${OPTIMIZATION_DECISION_SHA256:?OPTIMIZATION_DECISION_SHA256 must be supplied}"
: "${SCALE_DECISION:?SCALE_DECISION must be supplied}"
: "${SCALE_DECISION_SHA256:?SCALE_DECISION_SHA256 must be supplied}"
: "${DATASET_SOURCE:?DATASET_SOURCE must be supplied}"
: "${DATASET_REVISION:?DATASET_REVISION must be supplied}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"

if [[ "$TRAIN_GPUS" != "4" && "$TRAIN_GPUS" != "8" ]]; then
  fail "frozen scaling contract supports TRAIN_GPUS=4 or 8; got $TRAIN_GPUS" "full_training_gpu_count"
fi
: "${SLURM_NNODES:?SLURM_NNODES is required inside the allocation}"
if [[ "$SLURM_NNODES" != "1" ]]; then
  fail "local torchrun requires SLURM_NNODES=1; got $SLURM_NNODES" "multi_node_training_forbidden"
fi
: "${SLURM_NTASKS:?SLURM_NTASKS is required inside the allocation}"
if [[ "$SLURM_NTASKS" != "1" ]]; then
  fail "local torchrun requires one Slurm task; got $SLURM_NTASKS" "multi_task_training_forbidden"
fi
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
if [[ "$ALLOCATED_GPU_COUNT" != "$TRAIN_GPUS" ]]; then
  fail "TRAIN_GPUS=$TRAIN_GPUS differs from allocated GPU count=$ALLOCATED_GPU_COUNT" "gpu_allocation_mismatch"
fi
if (( EFFECTIVE_BATCH_SIZE % ALLOCATED_GPU_COUNT != 0 )); then
  fail "EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE must be divisible by allocated GPUs=$ALLOCATED_GPU_COUNT" "batch_divisibility"
fi

case "$METHOD" in
  sft|dpo|grpo|dapo) : ;;
  *) fail "unsupported METHOD=$METHOD" "method_validation" ;;
esac
if [[ "$METHOD" == dpo ]]; then
  : "${REF_LOG_PROBS:?REF_LOG_PROBS must be supplied for DPO}"
  : "${EVAL_REF_LOG_PROBS:?EVAL_REF_LOG_PROBS must be supplied for DPO}"
  : "${REFERENCE_CHECKPOINT_HASH:?REFERENCE_CHECKPOINT_HASH must be supplied for DPO}"
  : "${PROMPT_CONTEXT_SCHEMA:?PROMPT_CONTEXT_SCHEMA must be supplied for DPO}"
fi

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$OUTPUT"

PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
[[ -x "$PROBE_RUNNER" ]] || fail "repository probe runner is not executable: $PROBE_RUNNER" probe_runner_missing
run_probe_runner() { uv run --frozen python "$PROBE_RUNNER" "$@"; }
CONFIG_HASH="$(hash_path "$CONFIG")"
DATASET_HASH="$(hash_path "$TRAIN")"
EVAL_DATASET_HASH="$(hash_path "$EVAL")"
COMMIT_HASH="$(git rev-parse HEAD)"
IFS=$'\t' read -r ATTENTION_IMPLEMENTATION DECISION_MICROBATCH DECISION_GRADIENT_ACCUMULATION_STEPS DECISION_DATALOADER_WORKERS DECISION_EVAL_MICROBATCH DECISION_MAX_STEPS DECISION_MAX_LENGTH DECISION_GRADIENT_CHECKPOINTING DECISION_PACKING DECISION_PADDING_FREE DECISION_USE_LIGER DECISION_LEARNING_RATE DECISION_EPOCHS DECISION_SAVE_STEPS DECISION_EVAL_STEPS DECISION_NUM_GENERATIONS DECISION_GENERATION_BATCH_SIZE DECISION_MAX_COMPLETION_LENGTH ATTENTION_FALLBACK_REASON VALIDATED_DECISION_SHA256 < <(
  run_probe_runner validate-decision --decision "$OPTIMIZATION_DECISION" --expected-sha256 "$OPTIMIZATION_DECISION_SHA256" --purpose training --output-format training-tsv \
    --training-method "$METHOD" \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$START_MODEL" --model-revision "$START_REVISION" --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" \
    --dataset-sha256 "$DATASET_HASH" --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH" \
    --eval-dataset-sha256 "$EVAL_DATASET_HASH"
) || fail "frozen optimization decision validation failed" optimization_decision_invalid
awk -v expected="$LEARNING_RATE" -v frozen="$DECISION_LEARNING_RATE" 'BEGIN { exit !(expected == frozen) }' || fail "LEARNING_RATE differs from frozen decision" training_control_mismatch
awk -v expected="$EPOCHS" -v frozen="$DECISION_EPOCHS" 'BEGIN { exit !(expected == frozen) }' || fail "EPOCHS differs from frozen decision" training_control_mismatch
[[ "$SAVE_STEPS" == "$DECISION_SAVE_STEPS" && "$EVAL_STEPS" == "$DECISION_EVAL_STEPS" ]] || fail "save/eval cadence differs from frozen decision" training_control_mismatch
IFS=$'\t' read -r SELECTED_TRAIN_GPUS VALIDATED_SCALE_DECISION_SHA256 SCALE_FALLBACK_REASON < <(
  run_probe_runner validate-scale-decision --decision "$SCALE_DECISION" --expected-sha256 "$SCALE_DECISION_SHA256" --train-gpus "$TRAIN_GPUS" --training-method "$METHOD" \
    --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$START_MODEL" --model-revision "$START_REVISION" \
    --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_HASH" \
    --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" --source-schema-sha256 "$SOURCE_SCHEMA_HASH"
) || fail "frozen GPU scaling decision validation failed" scale_decision_invalid
export ATTENTION_IMPLEMENTATION ATTENTION_FALLBACK_REASON
MODEL_HASH="$(hash_value "$START_MODEL|$START_REVISION")"
RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT/run-manifest.json}"
PROMPT_HASH="$PROMPT_HASH"
RETRIEVAL_HASH="$RETRIEVAL_HASH"
SOURCE_SCHEMA_HASH="$SOURCE_SCHEMA_HASH"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
names = ("torch", "transformers", "trl", "deepspeed", "bitsandbytes")
print(";".join(f"{name}={importlib.metadata.version(name)}" for name in names))
PY
)"
GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}"
ARTIFACT_PATHS="$OUTPUT|$CONFIG|$TRAIN|${EVAL:-none}|$OPTIMIZATION_DECISION|$SCALE_DECISION|$CHECKPOINT_SMOKE_MANIFEST"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT RUN_MANIFEST OPTIMIZATION_DECISION OPTIMIZATION_DECISION_SHA256 SCALE_DECISION SCALE_DECISION_SHA256 CHECKPOINT_SMOKE_MANIFEST CHECKPOINT_SMOKE_MANIFEST_SHA256 DATASET_SOURCE DATASET_REVISION
log_event runtime attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON" train_gpus="$TRAIN_GPUS" allocated_gpus="$ALLOCATED_GPU_COUNT" max_length=4096

if [[ "${CLEANUP_TRAIN_INPUTS:-false}" == "true" ]]; then
  export TFDPO_CLEANUP_TRAINING_INPUTS=1
  log_event storage cleanup_training_inputs=enabled fallback_reason=derived_inputs_rebuildable_from_raw_searchqa
fi

nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT

log_event checkpoint_gate_started smoke_manifest="$CHECKPOINT_SMOKE_MANIFEST" fallback_reason=none
run_probe_runner validate-checkpoints --smoke-manifest "$CHECKPOINT_SMOKE_MANIFEST" --expected-sha256 "$CHECKPOINT_SMOKE_MANIFEST_SHA256" \
  --commit-hash "$COMMIT_HASH" --config-sha256 "$CONFIG_HASH" --model "$START_MODEL" --model-revision "$START_REVISION" \
  --dataset-source "$DATASET_SOURCE" --dataset-revision "$DATASET_REVISION" --dataset-sha256 "$DATASET_HASH" --prompt-sha256 "$PROMPT_HASH" --retrieval-sha256 "$RETRIEVAL_HASH" \
  --eval-dataset-sha256 "$EVAL_DATASET_HASH" \
  --source-schema-sha256 "$SOURCE_SCHEMA_HASH" --optimization-decision-sha256 "$OPTIMIZATION_DECISION_SHA256" --scale-decision-sha256 "$SCALE_DECISION_SHA256" --method "$METHOD"
log_event checkpoint_resume_gate_passed artifact="$CHECKPOINT_SMOKE_MANIFEST" checkpoint_gate_sha256="$CHECKPOINT_SMOKE_MANIFEST_SHA256" fallback_reason=none

GRADIENT_ACCUMULATION_STEPS="$((EFFECTIVE_BATCH_SIZE / ALLOCATED_GPU_COUNT))"
[[ "$DECISION_GRADIENT_ACCUMULATION_STEPS" == "$GRADIENT_ACCUMULATION_STEPS" ]] || fail "frozen gradient accumulation=$DECISION_GRADIENT_ACCUMULATION_STEPS differs from required=$GRADIENT_ACCUMULATION_STEPS" optimization_decision_batch_mismatch
TRAIN_HELP="$(uv run --frozen python -m text_feedback_dpo.cli "train-$METHOD" --help)" || fail "cannot inspect Task 7 train-$METHOD CLI" task7_train_cli_help_failed
for required_flag in --eval --max-steps --max-length --dataloader-num-workers --per-device-train-batch-size --per-device-eval-batch-size --gradient-accumulation-steps --attention-implementation --gradient-checkpointing --packing --padding-free --use-liger-kernel; do
  [[ "$TRAIN_HELP" == *"$required_flag"* ]] || fail "Task 7 train-$METHOD CLI does not expose $required_flag; cannot launch frozen worker settings" task7_training_worker_cli_missing
done
ARGS=(
  --config "$CONFIG" --train "$TRAIN" --output "$OUTPUT"
  --eval "$EVAL"
  --deepspeed-config configs/deepspeed_zero3.json
  --save-steps "$DECISION_SAVE_STEPS" --eval-steps "$DECISION_EVAL_STEPS"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --per-device-train-batch-size "$DECISION_MICROBATCH"
  --per-device-eval-batch-size "$DECISION_EVAL_MICROBATCH"
  --dataloader-num-workers "$DECISION_DATALOADER_WORKERS"
  --max-steps "$DECISION_MAX_STEPS" --max-length "$DECISION_MAX_LENGTH"
  --attention-implementation "$ATTENTION_IMPLEMENTATION"
  --learning-rate "$DECISION_LEARNING_RATE" --epochs "$DECISION_EPOCHS"
  --num-generations "$DECISION_NUM_GENERATIONS" --generation-batch-size "$DECISION_GENERATION_BATCH_SIZE"
  --max-completion-length "$DECISION_MAX_COMPLETION_LENGTH"
)
if [[ "$DECISION_GRADIENT_CHECKPOINTING" == True ]]; then ARGS+=(--gradient-checkpointing); else ARGS+=(--no-gradient-checkpointing); fi
if [[ "$DECISION_PACKING" == True ]]; then ARGS+=(--packing); else ARGS+=(--no-packing); fi
if [[ "$DECISION_PADDING_FREE" == True ]]; then ARGS+=(--padding-free); else ARGS+=(--no-padding-free); fi
if [[ "$DECISION_USE_LIGER" == True ]]; then ARGS+=(--use-liger-kernel); else ARGS+=(--no-use-liger-kernel); fi
ARGS+=(--model "$START_MODEL" --model-revision "$START_REVISION")
if [[ "$METHOD" == dpo ]]; then
  ARGS+=(--ref-log-probs "$REF_LOG_PROBS" --eval-ref-log-probs "$EVAL_REF_LOG_PROBS" \
    --reference-checkpoint-hash "$REFERENCE_CHECKPOINT_HASH" --prompt-context-schema "$PROMPT_CONTEXT_SCHEMA")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_TF32_CUBLAS_OVERRIDE=1
log_event training_launch bf16=true tf32=true deepspeed=zero3 optimizer=fused_adamw gradient_checkpointing=non-reentrant
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node="$ALLOCATED_GPU_COUNT" -m text_feedback_dpo.cli "train-$METHOD" "${ARGS[@]}"
log_event training_complete artifact="$OUTPUT"
write_manifest complete
