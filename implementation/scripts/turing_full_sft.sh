#!/bin/bash
# Full-parameter Qwen3-4B SFT from an exact hash-bound local checkpoint.
# The checkpoint supplies weights only: optimizer and scheduler are intentionally fresh.
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

COMPONENT=turing_full_sft
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
require_env() { local name="$1"; [[ -n "${!name:-}" ]] || fail "required environment variable is missing: $name" full_sft_contract_missing; }
allocated_gpu_count() {
  local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"
  elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"
  else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" gpu_count_parse_unsupported
  fi
}

require_env "PROJECT_DIR"
require_env "EXPECTED_COMMIT"
require_env "CONFIG"
require_env "CONFIG_SHA256"
require_env "TRAIN"
require_env "TRAIN_SHA256"
require_env "EVAL"
require_env "EVAL_SHA256"
require_env "SPLIT_REPORT"
require_env "SPLIT_REPORT_SHA256"
require_env "OUTPUT"
require_env "START_MODEL"
require_env "START_MODEL_SHA256"
require_env "START_REVISION"
require_env "DATASET_SOURCE"
require_env "DATASET_REVISION"
require_env "PROMPT_HASH"
require_env "RETRIEVAL_HASH"
require_env "SOURCE_SCHEMA_HASH"
require_env "LEARNING_RATE"
require_env "EPOCHS"
require_env "SAVE_STEPS"
require_env "EVAL_STEPS"
require_env "GRADIENT_ACCUMULATION_STEPS"
require_env "EXPECTED_TRAIN_PAIRS"
require_env "EXPECTED_EVAL_PAIRS"

[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "full SFT requires one node and one task" allocation_shape_invalid
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"
[[ "$ALLOCATED_GPU_COUNT" == 4 ]] || fail "full SFT requires exactly four GPUs; got $ALLOCATED_GPU_COUNT" gpu_count_invalid
[[ "$SAVE_STEPS" =~ ^[1-9][0-9]*$ && "$EVAL_STEPS" =~ ^[1-9][0-9]*$ && "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "save/eval/accumulation controls must be positive integers" training_control_invalid
[[ "$EXPECTED_TRAIN_PAIRS" =~ ^[1-9][0-9]*$ && "$EXPECTED_EVAL_PAIRS" =~ ^[1-9][0-9]*$ ]] || fail "expected pair counts must be positive integers" pair_count_invalid
[[ "$LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]] || fail "LEARNING_RATE must be a positive decimal" learning_rate_invalid
awk -v value="$LEARNING_RATE" 'BEGIN { exit !(value > 0) }' || fail "LEARNING_RATE must be positive" learning_rate_invalid
[[ "$EPOCHS" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "EPOCHS must be a positive number" epochs_invalid
awk -v value="$EPOCHS" 'BEGIN { exit !(value > 0) }' || fail "EPOCHS must be positive" epochs_invalid
[[ -f "$CONFIG" && -f "$TRAIN" && -f "$EVAL" && -f "$SPLIT_REPORT" ]] || fail "config, train, eval, and split-report artifacts must exist" training_input_missing
[[ -d "$START_MODEL" && -f "$START_MODEL/model.safetensors" ]] || fail "START_MODEL must be a local checkpoint containing model.safetensors" start_model_missing
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "remote commit differs from EXPECTED_COMMIT" commit_mismatch
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || fail "config artifact hash mismatch" config_hash_mismatch
[[ "$(sha256sum "$TRAIN" | awk '{print $1}')" == "$TRAIN_SHA256" ]] || fail "train artifact hash mismatch" train_hash_mismatch
[[ "$(sha256sum "$EVAL" | awk '{print $1}')" == "$EVAL_SHA256" ]] || fail "eval artifact hash mismatch" eval_hash_mismatch
[[ "$(sha256sum "$SPLIT_REPORT" | awk '{print $1}')" == "$SPLIT_REPORT_SHA256" ]] || fail "split-report artifact hash mismatch" split_report_hash_mismatch
[[ "$(sha256sum "$START_MODEL/model.safetensors" | awk '{print $1}')" == "$START_MODEL_SHA256" ]] || fail "initial checkpoint model hash mismatch" start_model_hash_mismatch
[[ ! -e "$OUTPUT" ]] || fail "OUTPUT already exists; refusing stale or partial run reuse: $OUTPUT" output_exists

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR is not the implementation root" project_root_invalid
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" PYTORCH_TF32_CUBLAS_OVERRIDE=1
mkdir -p "$HF_HOME" "$(dirname "$OUTPUT")" logs

uv run --frozen python - "$TRAIN" "$EVAL" "$SPLIT_REPORT" "$EXPECTED_TRAIN_PAIRS" "$EXPECTED_EVAL_PAIRS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

train_path, eval_path, report_path = map(Path, sys.argv[1:4])
expected_train_pairs, expected_eval_pairs = map(int, sys.argv[4:6])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected_report = {
    "train_pairs": expected_train_pairs,
    "train_rows": 2 * expected_train_pairs,
    "eval_pairs": expected_eval_pairs,
    "eval_rows": 2 * expected_eval_pairs,
    "trajectory_overlap": 0,
}
for key, expected in expected_report.items():
    if report.get(key) != expected:
        raise SystemExit(f"split report {key} mismatch: expected {expected}, got {report.get(key)!r}")

trajectory_sets = {}
for label, path, expected_pairs in (
    ("train", train_path, expected_train_pairs),
    ("eval", eval_path, expected_eval_pairs),
):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 2 * expected_pairs:
        raise SystemExit(f"{label} row count mismatch: expected {2 * expected_pairs}, got {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise SystemExit(f"{label} SFT IDs must be unique and non-empty")
    counts = Counter(row.get("task") for row in rows)
    if counts != {"query": expected_pairs, "response": expected_pairs}:
        raise SystemExit(f"{label} SFT task counts are not exactly balanced: {dict(counts)}")
    trajectories = set()
    per_trajectory = {}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise SystemExit(f"{label} SFT row {row.get('id')} is missing metadata")
        if metadata.get("provenance") != "student" or metadata.get("no_hint") is not True or metadata.get("verified_no_hint_success") is not True:
            raise SystemExit(f"{label} SFT row {row.get('id')} is not verified student no-hint supervision")
        trajectory_id = metadata.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise SystemExit(f"{label} SFT row {row.get('id')} has no trajectory_id")
        trajectories.add(trajectory_id)
        tasks = per_trajectory.setdefault(trajectory_id, set())
        tasks.add(row["task"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise SystemExit(f"{label} SFT row {row.get('id')} has an empty prompt")
        if not isinstance(row.get("completion"), str) or not row["completion"].strip():
            raise SystemExit(f"{label} SFT row {row.get('id')} has an empty completion")
    invalid = {key: value for key, value in per_trajectory.items() if value != {"query", "response"}}
    if invalid or len(trajectories) != expected_pairs:
        raise SystemExit(f"{label} SFT trajectories are not exact query/response pairs")
    trajectory_sets[label] = trajectories
    print(f"event=sft_artifact_validated split={label} rows={len(rows)} pairs={expected_pairs} fallback_reason=none")
if trajectory_sets["train"] & trajectory_sets["eval"]:
    raise SystemExit("train/eval trajectory leakage detected")
PY

mkdir -p "$OUTPUT"
GPU_TELEMETRY="$OUTPUT/gpu-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
on_exit() {
  local code=$?
  trap - EXIT
  set +e
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi
  if [[ "$code" -ne 0 ]]; then log_event training_failed exit_code="$code" output="$OUTPUT" fallback_reason=none >&2; fi
  exit "$code"
}
trap on_exit EXIT

log_event training_start model="$START_MODEL" model_sha256="$START_MODEL_SHA256" optimizer_state=fresh max_steps=-1 epochs="$EPOCHS" learning_rate="$LEARNING_RATE" effective_batch_size="$((4 * GRADIENT_ACCUMULATION_STEPS))" fallback_reason=none
nvidia-smi
uv run --frozen python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m text_feedback_dpo.cli train-sft \
  --config "$CONFIG" --train "$TRAIN" --eval "$EVAL" --output "$OUTPUT" \
  --model "$START_MODEL" --model-revision "$START_REVISION" \
  --initial-checkpoint-sha256 "$START_MODEL_SHA256" \
  --deepspeed-config configs/deepspeed_zero3.json \
  --max-steps -1 --max-length 4096 \
  --per-device-train-batch-size 1 --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --dataloader-num-workers 2 \
  --attention-implementation sdpa --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS" \
  --save-steps "$SAVE_STEPS" --eval-steps "$EVAL_STEPS" \
  --gradient-checkpointing --no-packing --no-padding-free --no-use-liger-kernel \
  --num-generations 4 --generation-batch-size 4 --max-completion-length 256

[[ -f "$OUTPUT/final/model.safetensors" ]] || fail "full SFT final model is missing" final_model_missing
FINAL_MODEL_SHA256="$(sha256sum "$OUTPUT/final/model.safetensors" | awk '{print $1}')"
export GPU_TELEMETRY FINAL_MODEL_SHA256
uv run --frozen python - "$OUTPUT" <<'PY'
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

output = Path(sys.argv[1])
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

checkpoints = []
for checkpoint in sorted(output.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])):
    model = checkpoint / "model.safetensors"
    state = checkpoint / "trainer_state.json"
    if not model.is_file() or not state.is_file():
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    state_value = json.loads(state.read_text(encoding="utf-8"))
    checkpoints.append({
        "path": str(checkpoint),
        "step": int(checkpoint.name.split("-")[-1]),
        "model_sha256": file_sha256(model),
        "trainer_state_sha256": file_sha256(state),
        "log_history": state_value.get("log_history", []),
    })
if not checkpoints:
    raise SystemExit("full SFT produced no retained checkpoints")
eval_losses = [
    entry["eval_loss"]
    for checkpoint in checkpoints
    for entry in checkpoint["log_history"]
    if isinstance(entry, dict) and isinstance(entry.get("eval_loss"), (int, float))
]
manifest = {
    "status": "complete",
    "stage": "verified_student_no_hint_full_sft",
    "commit_hash": os.environ["EXPECTED_COMMIT"],
    "config_sha256": os.environ["CONFIG_SHA256"],
    "train_sha256": os.environ["TRAIN_SHA256"],
    "eval_sha256": os.environ["EVAL_SHA256"],
    "split_report_sha256": os.environ["SPLIT_REPORT_SHA256"],
    "model": {
        "initial_path": os.environ["START_MODEL"],
        "initial_model_sha256": os.environ["START_MODEL_SHA256"],
        "revision": os.environ["START_REVISION"],
        "final_model_sha256": os.environ["FINAL_MODEL_SHA256"],
        "full_finetuning": True,
        "optimizer_state": "fresh",
    },
    "dataset": {"source": os.environ["DATASET_SOURCE"], "revision": os.environ["DATASET_REVISION"]},
    "prompt_hash": os.environ["PROMPT_HASH"],
    "retrieval_hash": os.environ["RETRIEVAL_HASH"],
    "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"],
    "training": {
        "bf16": True, "tf32": True, "deepspeed": "zero3", "max_length": 4096,
        "max_steps": -1, "epochs": float(os.environ["EPOCHS"]),
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "save_steps": int(os.environ["SAVE_STEPS"]), "eval_steps": int(os.environ["EVAL_STEPS"]),
        "gradient_accumulation_steps": int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
        "effective_batch_size": 4 * int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
        "train_pairs": int(os.environ["EXPECTED_TRAIN_PAIRS"]),
        "eval_pairs": int(os.environ["EXPECTED_EVAL_PAIRS"]),
    },
    "retained_checkpoints": checkpoints,
    "best_retained_eval_loss": min(eval_losses) if eval_losses else None,
    "gpu_telemetry": os.environ["GPU_TELEMETRY"],
    "node": socket.gethostname(),
    "platform": platform.platform(),
    "fallback_reason": "none",
}
(output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

kill "$GPU_MONITOR_PID"
trap - EXIT
log_event training_complete manifest="$OUTPUT/run-manifest.json" final_model_sha256="$FINAL_MODEL_SHA256" fallback_reason=none
