#!/bin/bash
# Role-specific CUDA/model preflight. Missing optional kernels are reported, never silently enabled.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

COMPONENT=turing_preflight
log_event() { local event="$1"; shift; printf 'event=%s timestamp=%s component=%s %s\n' "$event" "$(date -u +%FT%TZ)" "$COMPONENT" "$*"; }
fail() { log_event failure reason="$1" fallback_reason="${2:-none}" >&2; exit 2; }
hash_value() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }
hash_path() { if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; else hash_value "$1"; fi; }
allocated_gpu_count() { local raw="${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"; if [[ "$raw" =~ ^[0-9]+$ ]]; then printf '%s\n' "$raw"; elif [[ "$raw" =~ ^gpu:([0-9]+)$ ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else fail "unsupported SLURM_GPUS_ON_NODE format: $raw" "gpu_count_parse_unsupported"; fi; }
write_manifest() {
  local status="$1"; export MANIFEST_STATUS="$status" MANIFEST_ENDED_AT="$(date -u +%FT%TZ)"
  uv run --frozen python - "$RUN_MANIFEST" <<'PY'
import json, os, platform, socket, sys
manifest = {"status": os.environ["MANIFEST_STATUS"], "commit_hash": os.environ["COMMIT_HASH"], "config_hash": os.environ["CONFIG_HASH"], "model_hash": os.environ["MODEL_HASH"], "dataset_hash": os.environ["DATASET_HASH"], "prompt_hash": os.environ["PROMPT_HASH"], "retrieval_hash": os.environ["RETRIEVAL_HASH"], "source_schema_hash": os.environ["SOURCE_SCHEMA_HASH"], "node": socket.gethostname(), "platform": platform.platform(), "slurm_allocation": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE")}, "package_versions": os.environ["PACKAGE_VERSIONS"].split(";"), "gpu_telemetry": os.environ["GPU_TELEMETRY"], "timings": {"started_at": os.environ["MANIFEST_STARTED_AT"], "ended_at": os.environ["MANIFEST_ENDED_AT"]}, "artifact_paths": [os.environ["OUTPUT"]], "fallback_reason": os.environ.get("ATTENTION_FALLBACK_REASON", "none"), "max_length": 4096}
with open(sys.argv[1], "w", encoding="utf-8") as handle: json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
PY
}

: "${TURING_ACCOUNT:?TURING_ACCOUNT must be supplied with --export}"
: "${MODEL_ID:?MODEL_ID must be supplied with --export}"
: "${MODEL_REVISION:?MODEL_REVISION must be supplied with --export}"
: "${MODEL_ROLE:?MODEL_ROLE must be student or teacher}"
: "${OUTPUT:?OUTPUT must be supplied with --export}"
: "${PROJECT_DIR:?PROJECT_DIR must be supplied with --export}"
: "${CONFIG:?CONFIG must be supplied with --export}"
: "${PROMPT_HASH:?PROMPT_HASH must be supplied with --export}"
: "${RETRIEVAL_HASH:?RETRIEVAL_HASH must be supplied with --export}"
: "${SOURCE_SCHEMA_HASH:?SOURCE_SCHEMA_HASH must be supplied with --export}"
: "${DATASET_HASH:?DATASET_HASH must be supplied with --export}"

: "${SLURM_NNODES:?SLURM_NNODES is required}"; [[ "$SLURM_NNODES" == "1" ]] || fail "preflight requires one node; got $SLURM_NNODES" "multi_node_preflight"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"; [[ "$SLURM_NTASKS" == "1" ]] || fail "preflight requires one task; got $SLURM_NTASKS" "multi_task_preflight"
ALLOCATED_GPU_COUNT="$(allocated_gpu_count)"; [[ "$ALLOCATED_GPU_COUNT" == "1" ]] || fail "preflight requires exactly one GPU; got $ALLOCATED_GPU_COUNT" "preflight_gpu_count"

module load u22/cuda/12.4
cd "$PROJECT_DIR"
[[ -f pyproject.toml && -d src/text_feedback_dpo ]] || fail "PROJECT_DIR must contain pyproject.toml and src/text_feedback_dpo" "invalid_project_root"
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE_ROOT:-/scratch/$(hostname)/$USER/searchqa-dpo/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs "$(dirname "$OUTPUT")"
ATTENTION_IMPLEMENTATION=sdpa ATTENTION_FALLBACK_REASON=preflight_sdpa_baseline RUN_MANIFEST="${RUN_MANIFEST:-$OUTPUT.manifest.json}" GPU_TELEMETRY="${GPU_TELEMETRY:-logs/gpu-${SLURM_JOB_ID}.csv}" COMMIT_HASH="$(git rev-parse HEAD)" CONFIG_HASH="$(hash_path "$CONFIG")" MODEL_HASH="$(hash_value "$MODEL_ID@$MODEL_REVISION")" MANIFEST_STARTED_AT="$(date -u +%FT%TZ)" PACKAGE_VERSIONS="$(uv run --frozen python - <<'PY'
import importlib.metadata
print(";".join(f"{n}={importlib.metadata.version(n)}" for n in ("torch", "transformers", "trl", "deepspeed", "bitsandbytes")))
PY
)"
export ATTENTION_FALLBACK_REASON RUN_MANIFEST GPU_TELEMETRY COMMIT_HASH CONFIG_HASH MODEL_HASH MANIFEST_STARTED_AT PACKAGE_VERSIONS
export PYTORCH_TF32_CUBLAS_OVERRIDE=1
log_event preflight_start model_role="$MODEL_ROLE" model="$MODEL_ID" node="$(hostname)" attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON"
nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > "$GPU_TELEMETRY" &
GPU_MONITOR_PID=$!
cleanup() { if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi; }
trap cleanup EXIT
uv run --frozen python - <<'PY'
import json
import torch
import transformers
import trl
import deepspeed
import bitsandbytes
try:
    import flash_attn
    flash_attention = {"available": True, "version": getattr(flash_attn, "__version__", "unknown"), "fallback_reason": "none"}
except ImportError as exc:
    flash_attention = {"available": False, "fallback_reason": f"flash_attn unavailable: {exc}"}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback")
print(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "deepspeed": deepspeed.__version__, "bitsandbytes": bitsandbytes.__version__, "flash_attn": flash_attention, "attention_implementation": "flash_attention_2" if flash_attention["available"] else "sdpa", "cuda": torch.cuda.get_device_name(0)}, sort_keys=True))
PY
uv run --frozen python -m text_feedback_dpo.cli probe-model --role "$MODEL_ROLE" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" --teacher-quantization 4bit --teacher-max-new-tokens 1024 --attention-implementation "$ATTENTION_IMPLEMENTATION" --output "$OUTPUT"
log_event preflight_complete artifact="$OUTPUT" attention_implementation="$ATTENTION_IMPLEMENTATION" fallback_reason="$ATTENTION_FALLBACK_REASON"
write_manifest complete
