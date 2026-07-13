#!/bin/bash
# Measured probes; PROBE_RUNNER must write the documented JSON measurement contract.
# Required result keys: status, output_hash, examples_per_second, tokens_per_second,
# peak_gpu_memory_mb, gpu_utilization, fallback_reason.
# TRL config names probed explicitly: use_liger_kernel, precompute_ref_log_probs,
# padding_free, and packing. Liger is never enabled for primary DPO.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_optimization_probe
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be supplied with --export}"
: "${PROBE_RUNNER:?PROBE_RUNNER must be an executable runner}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${MODEL:?MODEL must be supplied with --export}"
: "${DATA:?DATA must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
[[ -x "$PROBE_RUNNER" ]] || fail "PROBE_RUNNER is not executable: $PROBE_RUNNER" "probe_runner_missing"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT" logs
ATTENTION_FALLBACK_REASON="${ATTENTION_FALLBACK_REASON:-none}"
GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-probe-${SLURM_JOB_ID}.csv}"
REPORT="$OUTPUT_ROOT/optimization-probe.jsonl"
BASELINE_RESULT="$OUTPUT_ROOT/baseline.json"
PACKAGE_VERSIONS_FILE="$OUTPUT_ROOT/package-versions.json"
PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
from importlib.metadata import PackageNotFoundError
names = ("torch", "transformers", "trl", "deepspeed", "bitsandbytes", "flash-attn", "liger-kernel")
versions = []
for name in names:
    try:
        versions.append(f"{name}={importlib.metadata.version(name)}")
    except PackageNotFoundError:
        versions.append(f"{name}=missing")
print(";".join(versions))
PY
)"
PACKAGE_VERSIONS="$PACKAGE_VERSIONS" PACKAGE_VERSIONS_FILE="$PACKAGE_VERSIONS_FILE" uv run --frozen python - <<'PY'
import json, os
versions = dict(item.split("=", 1) for item in os.environ["PACKAGE_VERSIONS"].split(";"))
with open(os.environ["PACKAGE_VERSIONS_FILE"], "w", encoding="utf-8") as handle:
    json.dump({"package_versions": versions}, handle, sort_keys=True, indent=2)
    handle.write("\n")
PY
: > "$REPORT"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 1 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT

run_probe() {
  local name="$1" result="$2"; shift 2
  rm -f "$result"
  log_event probe_start probe_name="$name" result="$result"
  if ! env PROBE_NAME="$name" PROBE_RESULT="$result" CONFIG="$CONFIG" MODEL="$MODEL" DATA="$DATA" PROMPT_HASH="$PROMPT_HASH" RETRIEVAL_HASH="$RETRIEVAL_HASH" SOURCE_SCHEMA_HASH="$SOURCE_SCHEMA_HASH" PACKAGE_VERSIONS="$PACKAGE_VERSIONS" PACKAGE_VERSIONS_FILE="$PACKAGE_VERSIONS_FILE" "$@" "$PROBE_RUNNER"; then
    printf '{"probe_name":"%s","status":"rejected","accepted":false,"fallback_reason":"runner_exit_status","package_versions_file":"%s"}\n' "$name" "$PACKAGE_VERSIONS_FILE" >> "$REPORT"
    log_event probe_rejected probe_name="$name" fallback_reason=runner_exit_status
    return 1
  fi
  [[ -s "$result" ]] || { printf '{"probe_name":"%s","status":"rejected","accepted":false,"fallback_reason":"missing_result","package_versions_file":"%s"}\n' "$name" "$PACKAGE_VERSIONS_FILE" >> "$REPORT"; log_event probe_rejected probe_name="$name" fallback_reason=missing_result; return 1; }
  return 0
}

record_decision() {
  local name="$1" result="$2"
  BASELINE_RESULT="$BASELINE_RESULT" CANDIDATE_RESULT="$result" PROBE_NAME="$name" REPORT="$REPORT" PACKAGE_VERSIONS="$PACKAGE_VERSIONS" PACKAGE_VERSIONS_FILE="$PACKAGE_VERSIONS_FILE" uv run --frozen python - <<'PY'
import json, os
from pathlib import Path
name = os.environ["PROBE_NAME"]
try:
    base = json.loads(Path(os.environ["BASELINE_RESULT"]).read_text())
    candidate = json.loads(Path(os.environ["CANDIDATE_RESULT"]).read_text())
    required = ("status", "output_hash", "examples_per_second", "tokens_per_second", "peak_gpu_memory_mb", "gpu_utilization", "fallback_reason")
    missing = [key for key in required if key not in candidate]
    if missing: raise ValueError("missing_measurements:" + ",".join(missing))
    if candidate["status"] != "ok": raise ValueError("runner_status:" + str(candidate["status"]))
    if candidate["output_hash"] != base["output_hash"]: raise ValueError("output_hash_mismatch")
    baseline_rate = float(base["tokens_per_second"]); candidate_rate = float(candidate["tokens_per_second"])
    if candidate_rate <= baseline_rate: raise ValueError(f"throughput_not_improved:{candidate_rate}<={baseline_rate}")
    decision = {"probe_name": name, "status": "accepted", "accepted": True, "fallback_reason": candidate["fallback_reason"], "output_hash": candidate["output_hash"], "baseline_output_hash": base["output_hash"], "examples_per_second": candidate["examples_per_second"], "tokens_per_second": candidate_rate, "baseline_tokens_per_second": baseline_rate, "peak_gpu_memory_mb": candidate["peak_gpu_memory_mb"], "gpu_utilization": candidate["gpu_utilization"], "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "package_versions_file": os.environ["PACKAGE_VERSIONS_FILE"]}
except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    decision = {"probe_name": name, "status": "rejected", "accepted": False, "fallback_reason": str(exc), "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "package_versions_file": os.environ["PACKAGE_VERSIONS_FILE"]}
print(json.dumps(decision, sort_keys=True))
with Path(os.environ["REPORT"]).open("a") as handle: handle.write(json.dumps(decision, sort_keys=True) + "\n")
PY
}

if ! run_probe baseline "$BASELINE_RESULT"; then fail "baseline probe failed" "baseline_probe_failed"; fi
BASELINE_RESULT="$BASELINE_RESULT" uv run --frozen python - <<'PY'
import json, os
data = json.load(open(os.environ["BASELINE_RESULT"]))
required = ("status", "output_hash", "examples_per_second", "tokens_per_second", "peak_gpu_memory_mb", "gpu_utilization", "fallback_reason")
missing = [key for key in required if key not in data]
if missing or data.get("status") != "ok": raise SystemExit(f"invalid baseline measurements: {missing}")
PY

GENERATION_BATCH_SIZES="${GENERATION_BATCH_SIZES:-1 2 4 8 16}"
TRAIN_MICROBATCHES="${TRAIN_MICROBATCHES:-1 2 4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1 2 4 8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0 2 4 8}"
STATIC_CACHE="${STATIC_CACHE:-false true}"
COMPILE="${COMPILE:-false true}"
PACKING="${PACKING:-false true}"
PADDING_FREE="${PADDING_FREE:-false true}"
LIGER="${LIGER:-false true}"
LIGER_PRECOMPUTE_REF_LOG_PROBS="${LIGER_PRECOMPUTE_REF_LOG_PROBS:-true}"
for attention in sdpa flash_attention_2; do name="attention-$attention"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "ATTENTION_IMPLEMENTATION=$attention"; then record_decision "$name" "$result"; fi; done
for value in $GENERATION_BATCH_SIZES; do name="generation-batch-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "GENERATION_BATCH_SIZE=$value"; then record_decision "$name" "$result"; fi; done
for value in $TRAIN_MICROBATCHES; do name="train-microbatch-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "TRAIN_MICROBATCH=$value"; then record_decision "$name" "$result"; fi; done
for value in $GRADIENT_ACCUMULATION_STEPS; do name="grad-accum-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "GRADIENT_ACCUMULATION_STEPS=$value"; then record_decision "$name" "$result"; fi; done
for value in $DATALOADER_WORKERS; do name="workers-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "DATALOADER_WORKERS=$value"; then record_decision "$name" "$result"; fi; done
for value in $STATIC_CACHE; do name="static-cache-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "STATIC_CACHE=$value"; then record_decision "$name" "$result"; fi; done
for value in $COMPILE; do name="compile-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "COMPILE=$value"; then record_decision "$name" "$result"; fi; done
for value in $PACKING; do name="packing-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "PACKING=$value"; then record_decision "$name" "$result"; fi; done
for value in $PADDING_FREE; do name="padding-free-$value"; result="$OUTPUT_ROOT/$name.json"; if run_probe "$name" "$result" "PADDING_FREE=$value" "ATTENTION_IMPLEMENTATION=flash_attention_2"; then record_decision "$name" "$result"; fi; done
for value in $LIGER; do
  name="liger-$value"; result="$OUTPUT_ROOT/$name.json"
  if [[ "$value" == "true" && "$LIGER_PRECOMPUTE_REF_LOG_PROBS" == "true" ]]; then
    printf '{"probe_name":"%s","status":"rejected","accepted":false,"fallback_reason":"use_liger_kernel_incompatible_with_precompute_ref_log_probs","package_versions_file":"%s"}\n' "$name" "$PACKAGE_VERSIONS_FILE" >> "$REPORT"
    log_event probe_rejected probe_name="$name" fallback_reason=use_liger_kernel_incompatible_with_precompute_ref_log_probs
  elif run_probe "$name" "$result" "USE_LIGER_KERNEL=$value" "PRECOMPUTE_REF_LOG_PROBS=$LIGER_PRECOMPUTE_REF_LOG_PROBS"; then
    record_decision "$name" "$result"
  fi
done
log_event probe_complete report="$REPORT" fallback_reason="$ATTENTION_FALLBACK_REASON"
