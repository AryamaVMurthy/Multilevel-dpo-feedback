#!/bin/bash
# One-GPU exact-revision, BF16, non-thinking, and LoRA-inventory preflight.
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?CONFIG is required}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:?PREFLIGHT_OUTPUT is required}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
TURING_ACCOUNT="${TURING_ACCOUNT:?TURING_ACCOUNT is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"

module load u22/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d /scratch || ! -w /scratch ]]; then
  echo "ERROR: /scratch is not writable; refusing CPU, home-cache, or runtime fallback" >&2
  exit 1
fi
if [[ "$MODEL_CACHE_DIR" != /scratch/* || "$RUNTIME_ROOT" != /scratch/* ]]; then
  echo "ERROR: MODEL_CACHE_DIR and RUNTIME_ROOT must use node-local /scratch" >&2
  exit 1
fi
if [[ ! -f "$MODEL_CACHE_DIR/tfdpo-model-cache-manifest.json" ]]; then
  echo "ERROR: staged model cache manifest is missing" >&2
  exit 1
fi
if [[ -e "$PREFLIGHT_OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite model preflight: $PREFLIGHT_OUTPUT" >&2
  exit 1
fi
export UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/project_venv"
if [[ ! -f "$UV_PROJECT_ENVIRONMENT/environment_verified.txt" ]]; then
  echo "ERROR: locked runtime verification is missing: $UV_PROJECT_ENVIRONMENT/environment_verified.txt" >&2
  exit 1
fi
export HF_HOME="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$(dirname "$PREFLIGHT_OUTPUT")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

allocation_mismatch() {
  echo "ERROR: allocation_mismatch: $*" >&2
  exit 1
}
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || allocation_mismatch "expected one node"
[[ "${SLURM_NNODES:-}" == "1" ]] || allocation_mismatch "expected one node"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || allocation_mismatch "CUDA_VISIBLE_DEVICES is unset"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#visible_gpus[@]}" == "1" ]] || allocation_mismatch "expected one visible GPU"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv

echo "job_id=${SLURM_JOB_ID} account=${TURING_ACCOUNT} host=$(hostname) source_commit=${SOURCE_COMMIT}"
uv run --frozen --no-sync python -m text_feedback_dpo.cli validate-paper-config --config "$CONFIG"
uv run --frozen --no-sync python - "$CONFIG" "$MODEL_CACHE_DIR" "$PREFLIGHT_OUTPUT" "$SOURCE_COMMIT" <<'PY'
import gc
import hashlib
import json
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_feedback_dpo.experiment_config import load_paper_experiment, validate_paper_experiment
from text_feedback_dpo.io import write_json_atomic
from text_feedback_dpo.lora_coverage import discover_lora_coverage


config_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
source_commit = sys.argv[4]
if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
    raise ValueError("SOURCE_COMMIT must be an immutable 40-character lowercase Git SHA")
config = load_paper_experiment(config_path)
validate_paper_experiment(config)
cache_manifest_path = cache_dir / "tfdpo-model-cache-manifest.json"
cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
if cache_manifest.get("schema") != "tfdpo-model-cache-v1":
    raise ValueError("staged model cache manifest schema mismatch")
config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
if cache_manifest.get("config_sha256") != config_sha256:
    raise ValueError("staged model cache config hash does not match frozen config")
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false; refusing CPU fallback")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("allocated GPU does not support BF16; refusing dtype fallback")

snapshots = {}
for entry in cache_manifest.get("models", []):
    identity = (entry.get("id"), entry.get("revision"))
    path = Path(str(entry.get("snapshot_path", "")))
    if identity in snapshots:
        raise ValueError(f"duplicate staged model identity: {identity}")
    if not path.is_dir() or cache_dir.resolve() not in path.resolve().parents:
        raise ValueError(f"staged model snapshot is missing or outside cache: {path}")
    snapshots[identity] = path

expected_identities = {
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}
if set(snapshots) != expected_identities:
    raise ValueError(f"staged model identities mismatch: {sorted(snapshots)}")


def load_and_generate(model_id, revision, *, do_sample):
    snapshot = snapshots[(model_id, revision)]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if model.config.model_type != "qwen3":
        raise ValueError(f"unexpected model_type for {model_id}: {model.config.model_type}")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 2 + 2. End with FINAL: \\boxed{answer}."}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    generation = {
        "max_new_tokens": 32,
        "do_sample": do_sample,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation.update(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)
    with torch.inference_mode():
        output = model.generate(**inputs, **generation)
    torch.cuda.synchronize()
    completion = tokenizer.decode(
        output[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    if not completion:
        raise RuntimeError(f"{model_id} generated an empty completion")
    if "<think" in completion.casefold() or "</think" in completion.casefold():
        raise RuntimeError(f"{model_id} emitted thinking markup with enable_thinking=false")
    result = {
        "id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "model_type": model.config.model_type,
        "hidden_layers": int(model.config.num_hidden_layers),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "load_and_generate_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "completion": completion,
    }
    return model, result


student = config.models["student"]
student_model, student_result = load_and_generate(
    student["id"], student["revision"], do_sample=True
)
coverage = discover_lora_coverage(
    student_model,
    rank=config.lora.rank,
    excluded_components=config.lora.excluded_components,
)
if len(coverage.target_modules) != 252:
    raise ValueError(
        f"Qwen3-4B LoRA inventory must contain 252 projections, found {len(coverage.target_modules)}"
    )
if student_result["hidden_layers"] != 36:
    raise ValueError("Qwen3-4B hidden-layer count must be 36")
student_result["lora_coverage"] = {
    "target_count": len(coverage.target_modules),
    "coverage_hash": coverage.coverage_hash,
    "estimated_lora_parameters": coverage.estimated_lora_parameters,
    "targets": list(coverage.target_modules),
}
del student_model
gc.collect()
torch.cuda.empty_cache()

teacher = config.models["teacher"]
teacher_model, teacher_result = load_and_generate(
    teacher["id"], teacher["revision"], do_sample=False
)
del teacher_model
gc.collect()
torch.cuda.empty_cache()

report = {
    "schema": "paper-model-preflight-v1",
    "status": "passed",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "source_commit": source_commit,
    "config_sha256": config_sha256,
    "cache_manifest_sha256": hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest(),
    "cache_source_commit": cache_manifest["source_commit"],
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "bf16_supported": True,
    "models": [student_result, teacher_result],
}
write_json_atomic(output_path, report)
print(json.dumps({
    "schema": report["schema"],
    "status": report["status"],
    "source_commit": source_commit,
    "models": [
        {
            "id": item["id"],
            "revision": item["revision"],
            "peak_gpu_memory_bytes": item["peak_gpu_memory_bytes"],
        }
        for item in report["models"]
    ],
    "lora_target_count": student_result["lora_coverage"]["target_count"],
}, sort_keys=True))
PY
