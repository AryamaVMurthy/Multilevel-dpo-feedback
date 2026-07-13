#!/bin/bash
# First optimizer-stage gate: full-parameter Qwen3-4B SFT overfit with real save/resume.
# This measures the training path before optimization and 4-vs-8 GPU decisions are frozen.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

METHOD=sft
COMPONENT=turing_sft_overfit
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
allocated_gpu_count() {
  local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"
  elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"
  else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported
  fi
}

required=(PROJECT_DIR EXPECTED_COMMIT CONFIG TRAIN TRAIN_SHA256 EVAL EVAL_SHA256 OUTPUT START_MODEL START_REVISION DATASET_SOURCE DATASET_REVISION PROMPT_HASH RETRIEVAL_HASH SOURCE_SCHEMA_HASH INITIAL_MAX_STEPS FINAL_MAX_STEPS LEARNING_RATE EPOCHS SAVE_STEPS EVAL_STEPS GRADIENT_ACCUMULATION_STEPS)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" sft_overfit_contract_missing; done
[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "SFT overfit requires one node and one task" allocation_shape_invalid
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == 4 ]] || fail "SFT overfit requires exactly four GPUs; got $ALLOCATED_GPU_COUNT" gpu_count_invalid
[[ "$INITIAL_MAX_STEPS" =~ ^[1-9][0-9]*$ && "$FINAL_MAX_STEPS" =~ ^[1-9][0-9]*$ && "$FINAL_MAX_STEPS" -gt "$INITIAL_MAX_STEPS" ]] || fail "FINAL_MAX_STEPS must exceed positive INITIAL_MAX_STEPS" step_contract_invalid
[[ "$SAVE_STEPS" =~ ^[1-9][0-9]*$ && "$EVAL_STEPS" =~ ^[1-9][0-9]*$ && "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "save/eval/accumulation controls must be positive integers" training_control_invalid
[[ -f "$CONFIG" && -f "$TRAIN" && -f "$EVAL" ]] || fail "config, train, and eval artifacts must exist" training_input_missing
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ "$(sha256sum "$TRAIN" | awk '{print $1}')" == "$TRAIN_SHA256" ]] || fail "train artifact hash mismatch" train_hash_mismatch
[[ "$(sha256sum "$EVAL" | awk '{print $1}')" == "$EVAL_SHA256" ]] || fail "eval artifact hash mismatch" eval_hash_mismatch
[[ ! -e "$OUTPUT" ]] || fail "OUTPUT already exists; refusing stale checkpoint reuse: $OUTPUT" output_exists

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" PYTORCH_TF32_CUBLAS_OVERRIDE=1
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" logs

uv run --frozen python - "$TRAIN" "$EVAL" <<'PY'
import json, sys
from collections import Counter

for label, path in (("train", sys.argv[1]), ("eval", sys.argv[2])):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"{label} SFT artifact is empty")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise SystemExit(f"{label} SFT IDs must be unique and non-empty")
    counts = Counter(row.get("task") for row in rows)
    if counts.get("query", 0) != counts.get("response", 0) or set(counts) != {"query", "response"}:
        raise SystemExit(f"{label} SFT artifact must be exactly query/response balanced; counts={dict(counts)}")
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("provenance") != "student" or metadata.get("no_hint") is not True:
            raise SystemExit(f"{label} SFT row {row.get('id')} is not verified student no-hint supervision")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or not isinstance(row.get("completion"), str) or not row["completion"].strip():
            raise SystemExit(f"{label} SFT row {row.get('id')} has an empty prompt/completion")
    print(f"event=sft_artifact_validated split={label} rows={len(rows)} query={counts['query']} response={counts['response']} fallback_reason=none")
PY

mkdir -p "$OUTPUT"
GPU_TELEMETRY="$OUTPUT/gpu-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

COMMON_ARGS=(
  --config "$CONFIG" --train "$TRAIN" --eval "$EVAL" --output "$OUTPUT"
  --model "$START_MODEL" --model-revision "$START_REVISION"
  --deepspeed-config configs/deepspeed_zero3.json
  --max-length 4096 --per-device-train-batch-size 1 --per-device-eval-batch-size 1
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --dataloader-num-workers 2
  --attention-implementation sdpa --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS"
  --save-steps "$SAVE_STEPS" --eval-steps "$EVAL_STEPS"
  --gradient-checkpointing --no-packing --no-padding-free --no-use-liger-kernel
  --num-generations 4 --generation-batch-size 4 --max-completion-length 256
)
log_event overfit_initial_start max_steps="$INITIAL_MAX_STEPS" train_sha256="$TRAIN_SHA256" fallback_reason=none
nvidia-smi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=4 -m text_feedback_dpo.cli train-sft "${COMMON_ARGS[@]}" --max-steps "$INITIAL_MAX_STEPS"
INITIAL_CHECKPOINT="$OUTPUT/checkpoint-$INITIAL_MAX_STEPS"
[[ -d "$INITIAL_CHECKPOINT" ]] || fail "initial checkpoint is missing: $INITIAL_CHECKPOINT" checkpoint_save_missing
INITIAL_CHECKPOINT_HASH_FILE="$OUTPUT/initial-checkpoint.sha256"
uv run --frozen python - "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_HASH_FILE" <<'PY'
import hashlib, sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
digest = hashlib.sha256()
files = sorted(item for item in root.rglob("*") if item.is_file())
if not files:
    raise SystemExit(f"checkpoint has no files: {root}")
for item in files:
    digest.update(str(item.relative_to(root)).encode())
    digest.update(hashlib.sha256(item.read_bytes()).digest())
output.write_text(digest.hexdigest() + "\n", encoding="utf-8")
PY

log_event overfit_resume_start max_steps="$FINAL_MAX_STEPS" resume_from="$INITIAL_CHECKPOINT" fallback_reason=none
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=4 -m text_feedback_dpo.cli train-sft "${COMMON_ARGS[@]}" --max-steps "$FINAL_MAX_STEPS" --resume-from-checkpoint "$INITIAL_CHECKPOINT"
FINAL_CHECKPOINT="$OUTPUT/checkpoint-$FINAL_MAX_STEPS"
[[ -d "$FINAL_CHECKPOINT" ]] || fail "resumed checkpoint is missing: $FINAL_CHECKPOINT" checkpoint_resume_missing

cleanup
trap - EXIT
export MANIFEST_PATH="$OUTPUT/run-manifest.json" INITIAL_CHECKPOINT INITIAL_CHECKPOINT_HASH_FILE FINAL_CHECKPOINT GPU_TELEMETRY
uv run --frozen python - <<'PY'
import hashlib, json, os, platform, socket
from pathlib import Path

def tree_hash(path):
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise SystemExit(f"checkpoint has no files: {root}")
    for item in files:
        digest.update(str(item.relative_to(root)).encode())
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()

manifest = {
    "status": "complete",
    "stage": "sft_full_finetune_overfit_save_resume",
    "commit_hash": os.environ["EXPECTED_COMMIT"],
    "config_hash": hashlib.sha256(Path(os.environ["CONFIG"]).read_bytes()).hexdigest(),
    "dataset_hash": os.environ["TRAIN_SHA256"],
    "eval_dataset_hash": os.environ["EVAL_SHA256"],
    "model": {"identity": os.environ["START_MODEL"], "revision": os.environ["START_REVISION"], "full_finetuning": True},
    "dataset": {"source": os.environ["DATASET_SOURCE"], "revision": os.environ["DATASET_REVISION"]},
    "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"],
    "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"],
    "training": {"bf16": True, "tf32": True, "deepspeed": "zero3", "max_length": 4096, "initial_max_steps": int(os.environ["INITIAL_MAX_STEPS"]), "final_max_steps": int(os.environ["FINAL_MAX_STEPS"]), "learning_rate": float(os.environ["LEARNING_RATE"]), "epochs": float(os.environ["EPOCHS"])},
    "checkpoints": {
        "initial": {
            "path": os.environ["INITIAL_CHECKPOINT"],
            "sha256": Path(os.environ["INITIAL_CHECKPOINT_HASH_FILE"]).read_text(encoding="utf-8").strip(),
            "initial_checkpoint_retained": Path(os.environ["INITIAL_CHECKPOINT"]).is_dir(),
        },
        "resumed": {"path": os.environ["FINAL_CHECKPOINT"], "sha256": tree_hash(os.environ["FINAL_CHECKPOINT"])},
    },
    "gpu_telemetry": os.environ["GPU_TELEMETRY"],
    "node": socket.gethostname(),
    "platform": platform.platform(),
    "fallback_reason": "none",
}
Path(os.environ["MANIFEST_PATH"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
log_event overfit_complete manifest="$OUTPUT/run-manifest.json" checkpoint="$FINAL_CHECKPOINT" fallback_reason=none
