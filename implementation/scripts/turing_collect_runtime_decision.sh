#!/bin/bash
# Parallel-safe two-GPU collector: freeze the device decision from this allocation,
# then execute the strict collector with that allocation-bound decision.
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

COMPONENT=turing_collect_runtime_decision
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }

: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${STUDENT_MODEL:?STUDENT_MODEL must be supplied with --export}"
: "${STUDENT_REVISION:?STUDENT_REVISION must be supplied with --export}"
: "${TEACHER_MODEL:?TEACHER_MODEL must be supplied with --export}"
: "${TEACHER_REVISION:?TEACHER_REVISION must be supplied with --export}"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -x scripts/turing_probe_runner.py ]] || fail "PROJECT_DIR lacks the pinned probe runner" invalid_project_root
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy

[[ -s "$DATA" ]] || fail "collection input shard is missing or empty: $DATA" collection_input_missing
SHARD_INPUT_SHA256="$(sha256sum "$DATA" | awk '{print $1}')" \
  || fail "could not hash collection input shard: $DATA" collection_input_hash_failed
[[ "$SHARD_INPUT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "collection input shard hash is not a lowercase SHA-256: $SHARD_INPUT_SHA256" collection_input_hash_invalid
log_event runtime_shard_input_identity data="$DATA" shard_input_sha256="$SHARD_INPUT_SHA256" fallback_reason=none

CURRENT_HARDWARE="${CURRENT_HARDWARE:-$OUTPUT.hardware.json}"
COLLECTION_DECISION="${COLLECTION_DECISION:-$OUTPUT.collection-decision.json}"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$CURRENT_HARDWARE")" "$(dirname "$COLLECTION_DECISION")"
[[ ! -e "$COLLECTION_DECISION" ]] || fail "runtime collection decision already exists: $COLLECTION_DECISION" collection_decision_would_overwrite

PROBE_RUNNER="$PROJECT_DIR/scripts/turing_probe_runner.py"
uv run --frozen python "$PROBE_RUNNER" probe-hardware --output "$CURRENT_HARDWARE" \
  || fail "current physical GPU hardware probe failed" collection_current_hardware_probe_failed

DEVICE_SELECTION="$(uv run --frozen python - "$CURRENT_HARDWARE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hardware = payload.get("gpu_hardware") if isinstance(payload, dict) else None
if not isinstance(hardware, dict) or hardware.get("count") != 2 or not isinstance(hardware.get("devices"), list) or len(hardware["devices"]) != 2:
    raise SystemExit("runtime collection requires exactly two measured GPUs")
devices = hardware["devices"]
if any(not isinstance(item, dict) or not isinstance(item.get("free_memory_bytes"), int) for item in devices):
    raise SystemExit("runtime collection hardware lacks integer free-memory evidence")
teacher = max(range(2), key=lambda index: (devices[index]["free_memory_bytes"], -index))
student = 1 - teacher
print(teacher, student)
PY
)" || fail "could not select teacher/student devices from measured hardware" collection_device_selection_failed
read -r TEACHER_DEVICE_INDEX STUDENT_DEVICE_INDEX <<< "$DEVICE_SELECTION" \
  || fail "could not parse measured teacher/student device selection" collection_device_selection_failed
[[ "$TEACHER_DEVICE_INDEX" =~ ^[01]$ && "$STUDENT_DEVICE_INDEX" =~ ^[01]$ && "$TEACHER_DEVICE_INDEX" != "$STUDENT_DEVICE_INDEX" ]] \
  || fail "measured teacher/student device selection is invalid: $DEVICE_SELECTION" collection_device_selection_failed

uv run --frozen python "$PROBE_RUNNER" freeze-collection-decision \
  --hardware-result "$CURRENT_HARDWARE" --output "$COLLECTION_DECISION" \
  --teacher-model "$TEACHER_MODEL" --teacher-revision "$TEACHER_REVISION" \
  --student-model "$STUDENT_MODEL" --student-revision "$STUDENT_REVISION" \
  --teacher-device-index "$TEACHER_DEVICE_INDEX" --student-device-index "$STUDENT_DEVICE_INDEX" \
  || fail "could not freeze runtime collection device decision" collection_decision_freeze_failed

COLLECTION_DECISION_SHA256="$(sha256sum "$COLLECTION_DECISION" | awk '{print $1}')"
log_event runtime_collection_decision_frozen hardware="$CURRENT_HARDWARE" decision="$COLLECTION_DECISION" decision_sha256="$COLLECTION_DECISION_SHA256" teacher_device_index="$TEACHER_DEVICE_INDEX" student_device_index="$STUDENT_DEVICE_INDEX" fallback_reason=none
export CURRENT_HARDWARE COLLECTION_DECISION COLLECTION_DECISION_SHA256 SHARD_INPUT_SHA256
exec bash "$PROJECT_DIR/scripts/turing_collect.sh"
