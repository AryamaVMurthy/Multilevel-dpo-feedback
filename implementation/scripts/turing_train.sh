#!/bin/bash
# Full Qwen3-4B training. Submit with one four-GPU Slurm allocation.
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
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${TRAIN_GPUS:?TRAIN_GPUS must be supplied with --export}"
: "${EFFECTIVE_BATCH_SIZE:?EFFECTIVE_BATCH_SIZE must be supplied with --export}"
: "${LEARNING_RATE:?LEARNING_RATE must be supplied with --export}"
: "${EPOCHS:?EPOCHS must be supplied with --export}"
: "${SAVE_STEPS:?SAVE_STEPS must be supplied with --export}"
: "${EVAL_STEPS:?EVAL_STEPS must be supplied with --export}"
: "${CHECKPOINT_SMOKE_COMMAND:?CHECKPOINT_SMOKE_COMMAND must be supplied; refusing to launch without a save gate}"
: "${RESUME_SMOKE_COMMAND:?RESUME_SMOKE_COMMAND must be supplied; refusing to launch without a resume gate}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"

if [[ "$TRAIN_GPUS" != "4" ]]; then
  fail "full Qwen3-4B training requires TRAIN_GPUS=4; got $TRAIN_GPUS" "full_training_gpu_count"
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
  sft|dpo) : "${EVAL:?EVAL must be supplied for $METHOD}" ;;
  grpo|dapo) : ;;
  *) fail "unsupported METHOD=$METHOD" "method_validation" ;;
esac

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

ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}"
export ATTENTION_IMPLEMENTATION ATTENTION_FALLBACK_REASON
COMMIT_HASH="$(git rev-parse HEAD)"
CONFIG_HASH="$(hash_path "$CONFIG")"
MODEL_HASH="${MODEL_HASH:-$(hash_value "${START_MODEL:-config-student}|${START_REVISION:-none}")}"
DATASET_HASH="${DATASET_HASH:-$(hash_path "$TRAIN")}"
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
ARTIFACT_PATHS="$OUTPUT|$CONFIG|$TRAIN|${EVAL:-none}"
MANIFEST_STARTED_AT="$(date -u +%FT%TZ)"
export COMMIT_HASH CONFIG_HASH MODEL_HASH DATASET_HASH PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH PACKAGE_VERSIONS GPU_TELEMETRY ARTIFACT_PATHS MANIFEST_STARTED_AT RUN_MANIFEST
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

log_event checkpoint_smoke_started
bash -c "$CHECKPOINT_SMOKE_COMMAND"
log_event checkpoint_smoke_passed
bash -c "$RESUME_SMOKE_COMMAND"
log_event resume_smoke_passed

GRADIENT_ACCUMULATION_STEPS="$((EFFECTIVE_BATCH_SIZE / ALLOCATED_GPU_COUNT))"
ARGS=(
  --config "$CONFIG" --train "$TRAIN" --output "$OUTPUT"
  --deepspeed-config configs/deepspeed_zero3.json
  --save-steps "$SAVE_STEPS" --eval-steps "$EVAL_STEPS"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS"
)
if [[ "$METHOD" == sft || "$METHOD" == dpo ]]; then
  ARGS+=(--eval "$EVAL")
fi
if [[ -n "${START_MODEL:-}" ]]; then
  ARGS+=(--model "$START_MODEL")
  if [[ -n "${START_REVISION:-}" ]]; then
    ARGS+=(--model-revision "$START_REVISION")
  fi
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
