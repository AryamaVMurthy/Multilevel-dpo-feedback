#!/usr/bin/env python3
"""Measured Turing optimization probes and repository-owned launch gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


DECISION_SCHEMA_VERSION = 1
CORE_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "deepspeed",
    "bitsandbytes",
)
OPTIONAL_PACKAGES = (
    "flash-attn",
    "liger-kernel",
)
REQUIRED_PACKAGES = CORE_PACKAGES + OPTIONAL_PACKAGES
MIN_TELEMETRY_SAMPLES = 3
MIN_TELEMETRY_COVERAGE = 0.75
MAX_TELEMETRY_COVERAGE_DELTA = 0.15


class GateError(RuntimeError):
    def __init__(self, fallback_reason: str, message: str):
        super().__init__(message)
        self.fallback_reason = fallback_reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        reject("checkpoint_directory_missing", f"checkpoint directory does not exist: {path}")
    entries: list[tuple[str, str]] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append((str(child.relative_to(path)), sha256_file(child)))
    return sha256_json(entries)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def required_package_versions() -> dict[str, str]:
    versions = package_versions()
    missing = [name for name in CORE_PACKAGES if versions[name] == "missing"]
    if missing:
        reject("required_package_missing", f"required packages are unavailable: {', '.join(missing)}")
    return versions


def normalize_gpu_name(value: str) -> str:
    return " ".join(value.casefold().split())


def gpu_hardware_identity(torch: Any) -> dict[str, Any]:
    count = torch.cuda.device_count()
    if count <= 0:
        reject("gpu_hardware_identity_missing", "CUDA reports no measured GPU devices")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible and len(visible) != count:
        reject("gpu_hardware_identity_mismatch", f"CUDA_VISIBLE_DEVICES has {len(visible)} entries but torch sees {count}")
    identifiers = visible or [str(index) for index in range(count)]
    devices: list[dict[str, Any]] = []
    for index, identifier in enumerate(identifiers):
        try:
            output = subprocess.run(
                ["nvidia-smi", f"--id={identifier}", "--query-gpu=name,uuid,memory.free", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            reject("gpu_hardware_identity_unavailable", f"cannot query GPU UUID for CUDA device {identifier}: {exc}")
        if len(output) != 1:
            reject("gpu_hardware_identity_unavailable", f"expected one nvidia-smi row for CUDA device {identifier}, got {len(output)}")
        fields = [item.strip() for item in output[0].split(",")]
        if len(fields) != 3 or not fields[1].startswith("GPU-"):
            reject("gpu_hardware_identity_invalid", f"malformed nvidia-smi hardware identity for CUDA device {identifier}")
        properties = torch.cuda.get_device_properties(index)
        name = normalize_gpu_name(str(properties.name))
        if normalize_gpu_name(fields[0]) != name:
            reject("gpu_hardware_identity_mismatch", f"torch GPU name {name!r} differs from nvidia-smi {normalize_gpu_name(fields[0])!r}")
        total_memory = int(properties.total_memory)
        major = int(properties.major)
        minor = int(properties.minor)
        if total_memory <= 0 or major < 1 or minor < 0:
            reject("gpu_hardware_identity_invalid", f"invalid CUDA properties for device {index}")
        devices.append({
            "index": index,
            "name": name,
            "uuid": fields[1],
            "total_memory_bytes": total_memory,
            "free_memory_bytes": int(float(fields[2]) * 1024 * 1024),
            "compute_capability": f"{major}.{minor}",
        })
    return {"count": count, "devices": devices}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reject(reason: str, message: str) -> None:
    raise GateError(reason, message)


def load_decision(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        reject("decision_artifact_missing", f"optimization decision does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        reject("invalid_expected_decision_hash", "expected decision hash must be 64 lowercase hexadecimal characters")
    if actual_sha256 != expected_sha256:
        reject("decision_hash_mismatch", f"optimization decision hash is {actual_sha256}, expected {expected_sha256}")
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reject("decision_json_invalid", f"cannot parse optimization decision: {exc}")
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION or decision.get("status") != "frozen":
        reject("decision_schema_invalid", "optimization decision must be schema_version=1 and status=frozen")
    selected = decision.get("selected")
    if not isinstance(selected, dict):
        reject("decision_schema_invalid", "optimization decision selected config is missing")
    if selected.get("attention_implementation") not in {"sdpa", "flash_attention_2"}:
        reject("decision_attention_invalid", "selected attention implementation is unsupported")
    if decision.get("package_versions") != required_package_versions():
        reject("decision_package_identity_mismatch", "optimization decision package versions differ from the frozen environment")
    if gpu_hardware_error(decision.get("gpu_hardware")):
        reject("decision_gpu_hardware_invalid", "optimization decision lacks valid measured GPU hardware identity")
    if not isinstance(decision.get("selected_result_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", decision["selected_result_sha256"]) is None:
        reject("decision_result_identity_invalid", "optimization decision lacks selected measured result identity")
    return decision, actual_sha256


def verify_identity(decision: dict[str, Any], args: argparse.Namespace) -> None:
    identities = decision.get("identities")
    if not isinstance(identities, dict):
        reject("decision_identity_missing", "optimization decision identities are missing")
    expected = {
        "commit_hash": args.commit_hash,
        "config_sha256": args.config_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_source": args.dataset_source,
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": args.dataset_sha256,
        "prompt_sha256": args.prompt_sha256,
        "retrieval_sha256": args.retrieval_sha256,
        "source_schema_sha256": args.source_schema_sha256,
        "eval_dataset_sha256": args.eval_dataset_sha256,
    }
    for key, value in expected.items():
        if value is not None and identities.get(key) != value:
            reject("decision_identity_mismatch", f"decision {key}={identities.get(key)!r}, expected {value!r}")


def cmd_validate_decision(args: argparse.Namespace) -> None:
    decision, digest = load_decision(args.decision, args.expected_sha256)
    verify_identity(decision, args)
    selected = decision["selected"]
    fallback_reason = decision.get("fallback_reason")
    if not isinstance(fallback_reason, str) or not fallback_reason:
        reject("decision_fallback_reason_missing", "frozen decision must contain an explicit fallback_reason")
    if args.purpose == "generation":
        generation = selected.get("generation")
        if not isinstance(generation, dict):
            reject("decision_generation_missing", "selected generation config is missing")
        required = ("query_batch_size", "response_batch_size", "query_max_new_tokens", "response_max_new_tokens", "scratchpad_max_new_tokens", "top_k")
        if any(not isinstance(generation.get(key), int) or generation[key] <= 0 for key in required):
            reject("decision_generation_invalid", "generation batch and completion-token controls must be positive integers")
        if generation.get("student_thinking_mode") not in {"direct", "two_pass"}:
            reject("decision_generation_invalid", "student_thinking_mode must be direct or two_pass")
        for key in ("query_temperature", "response_temperature", "top_p", "k1", "b"):
            if not isinstance(generation.get(key), (int, float)):
                reject("decision_generation_invalid", f"{key} must be numeric")
        if generation.get("static_cache") or generation.get("compile"):
            reject(
                "active_search_cli_optimization_unsupported",
                "Task 5 generate-searchqa has no committed static-cache/compile flags; freeze a supported decision",
            )
        expected_generation = {
            "student_thinking_mode": args.student_thinking_mode,
            "scratchpad_max_new_tokens": args.scratchpad_max_new_tokens,
            "query_temperature": args.query_temperature,
            "response_temperature": args.response_temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "k1": args.k1,
            "b": args.b,
        }
        if any(value is None for value in expected_generation.values()):
            reject("generation_contract_missing", "all thinking, decoding, and retrieval controls must be explicit")
        for key, value in expected_generation.items():
            if generation.get(key) != value:
                reject("generation_contract_mismatch", f"decision {key}={generation.get(key)!r}, expected {value!r}")
        payload = {key: generation[key] for key in required}
        payload.update({key: generation[key] for key in ("student_thinking_mode", "query_temperature", "response_temperature", "top_p", "k1", "b")})
    else:
        training = selected.get("training")
        if not isinstance(training, dict):
            reject("decision_training_missing", "selected training config is missing")
        required = (
            "microbatch", "gradient_accumulation_steps", "dataloader_workers", "per_device_eval_batch_size",
            "max_steps", "max_length", "save_steps", "eval_steps", "num_generations",
            "generation_batch_size", "max_completion_length",
        )
        if any(not isinstance(training.get(key), int) or training[key] < 0 for key in required):
            reject("decision_training_invalid", "selected training integer controls are invalid")
        if not isinstance(training.get("gradient_checkpointing"), bool):
            reject("decision_training_invalid", "gradient_checkpointing must be frozen explicitly")
        if training.get("use_liger_kernel"):
            reject("use_liger_kernel_incompatible_with_precompute_ref_log_probs", "primary DPO cannot use Liger with precomputed reference log probabilities")
        if args.training_method is None or training.get("method") != args.training_method:
            reject("training_method_mismatch", f"decision training method={training.get('method')!r}, expected {args.training_method!r}")
        payload = {key: training[key] for key in required}
        for key in ("learning_rate", "epochs"):
            value = training.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                reject("decision_training_invalid", f"{key} must be a positive finite number")
            payload[key] = value
        payload.update({key: bool(training.get(key)) for key in ("packing", "padding_free", "use_liger_kernel", "gradient_checkpointing")})
    payload.update(
        {
            "status": "validated",
            "decision_sha256": digest,
            "attention_implementation": selected["attention_implementation"],
            "fallback_reason": fallback_reason,
        }
    )
    if args.output_format == "generation-tsv":
        print("\t".join(str(payload[key]) for key in (
            "attention_implementation", "query_batch_size", "response_batch_size",
            "query_max_new_tokens", "response_max_new_tokens", "student_thinking_mode",
            "scratchpad_max_new_tokens", "query_temperature", "response_temperature", "top_p",
            "top_k", "k1", "b", "fallback_reason", "decision_sha256",
        )))
    elif args.output_format == "training-tsv":
        print("\t".join(str(payload[key]) for key in (
            "attention_implementation", "microbatch", "gradient_accumulation_steps",
            "dataloader_workers", "per_device_eval_batch_size", "max_steps", "max_length",
            "gradient_checkpointing", "packing", "padding_free", "use_liger_kernel",
            "learning_rate", "epochs", "save_steps", "eval_steps", "num_generations",
            "generation_batch_size", "max_completion_length",
            "fallback_reason", "decision_sha256",
        )))
    else:
        print(json.dumps(payload, sort_keys=True))


def selection_identity(args: argparse.Namespace) -> dict[str, str]:
    return {
        "commit_hash": args.commit_hash,
        "config_sha256": args.config_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_source": args.dataset_source,
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": args.dataset_sha256,
        "prompt_sha256": args.prompt_sha256,
        "retrieval_sha256": args.retrieval_sha256,
        "source_schema_sha256": args.source_schema_sha256,
    }


def load_thinking_selection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        reject("thinking_selection_missing", f"thinking-mode selection artifact does not exist: {path}")
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reject("thinking_selection_invalid", f"cannot parse thinking-mode selection artifact: {exc}")
    if not isinstance(selection, dict) or selection.get("selected") not in {"direct", "two_pass"}:
        reject("thinking_selection_invalid", "selection artifact must contain selected=direct or selected=two_pass")
    eligible = selection.get("eligible")
    if not isinstance(eligible, list) or selection["selected"] not in eligible:
        reject("thinking_selection_invalid", "selection artifact eligible modes are missing or inconsistent")
    if not isinstance(selection.get("selection_metric"), list) or not selection["selection_metric"]:
        reject("thinking_selection_invalid", "selection artifact lacks selection_metric")
    return selection


def selection_identity_from_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        reject("thinking_selection_identity_missing", "thinking-mode selection identities are missing")
    for key, value in selection_identity(args).items():
        if identities.get(key) != value:
            reject("thinking_selection_identity_mismatch", f"selection {key}={identities.get(key)!r}, expected {value!r}")


def cmd_freeze_thinking_selection(args: argparse.Namespace) -> None:
    selection = load_thinking_selection(args.selection)
    if selection["selected"] != args.expected_mode:
        reject("thinking_selection_frozen_mode_mismatch", f"selected mode {selection['selected']!r} differs from frozen mode {args.expected_mode!r}")
    if args.row_count != 32:
        reject("thinking_selection_row_count_invalid", f"thinking-mode selection must be based on exactly 32 rows, got {args.row_count}")
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "fallback_reason": "none",
        "selected": selection["selected"],
        "selection_metric": selection["selection_metric"],
        "eligible": selection["eligible"],
        "selection_artifact_sha256": sha256_file(args.selection),
        "row_count": args.row_count,
        "optimization_decision_sha256": args.optimization_decision_sha256,
        "identities": selection_identity(args),
    }
    write_json(args.output, manifest)
    print(f"{selection['selected']}\t{sha256_file(args.output)}\tnone")


def cmd_validate_thinking_selection(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        reject("thinking_selection_missing", f"thinking-mode selection manifest does not exist: {args.manifest}")
    actual_sha256 = sha256_file(args.manifest)
    if actual_sha256 != args.expected_sha256:
        reject("thinking_selection_hash_mismatch", f"selection manifest hash is {actual_sha256}, expected {args.expected_sha256}")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reject("thinking_selection_invalid", f"cannot parse thinking-mode selection manifest: {exc}")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "frozen":
        reject("thinking_selection_schema_invalid", "selection manifest must be schema_version=1 and status=frozen")
    if manifest.get("fallback_reason") != "none":
        reject("thinking_selection_fallback_invalid", "selection manifest must record fallback_reason=none")
    if manifest.get("selected") != args.expected_mode:
        reject("thinking_selection_mode_mismatch", f"selection mode {manifest.get('selected')!r} differs from expected {args.expected_mode!r}")
    if manifest.get("row_count") != 32:
        reject("thinking_selection_row_count_invalid", "selection manifest must record row_count=32")
    if manifest.get("optimization_decision_sha256") != args.optimization_decision_sha256:
        reject("thinking_selection_decision_mismatch", "selection manifest does not match the frozen optimization decision")
    selection_identity_from_manifest(manifest, args)
    print(f"{manifest['selected']}\t{actual_sha256}\tnone")


def checkpoint_step(path: Path) -> tuple[int, Path]:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        reject("trainer_state_missing", f"checkpoint has no trainer_state.json: {path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step = state["global_step"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        reject("trainer_state_invalid", f"cannot read global_step from {state_path}: {exc}")
    if not isinstance(step, int) or step <= 0:
        reject("checkpoint_step_invalid", f"checkpoint global_step must be a positive integer: {step!r}")
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None or int(match.group(1)) != step:
        reject("checkpoint_path_step_mismatch", f"checkpoint directory {path.name!r} does not match global_step={step}")
    return step, state_path


def load_torch_state(path: Path, kind: str) -> dict[str, Any]:
    if path.stat().st_size < 64:
        reject("checkpoint_state_too_small", f"{kind} state is too small to be substantive: {path}")
    try:
        import torch
        if kind == "RNG":
            import numpy as np

            safe_types = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
            with torch.serialization.safe_globals(safe_types):
                value = torch.load(path, map_location="cpu", weights_only=True)
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        reject("checkpoint_state_invalid", f"cannot parse {kind} state {path}: {type(exc).__name__}: {exc}")
    if not isinstance(value, dict) or not value:
        reject("checkpoint_state_invalid", f"{kind} state must be a nonempty mapping: {path}")
    return value


def nested_items(value: object, prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(nested_items(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            items.extend(nested_items(child, f"{prefix}[{index}]"))
    else:
        items.append((prefix, value))
    return items


def load_hf_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        reject("checkpoint_model_config_missing", f"checkpoint has no Hugging Face config.json: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reject("checkpoint_model_config_invalid", f"cannot parse {config_path}: {exc}")
    required = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
    if any(key not in config for key in required):
        reject("checkpoint_model_config_invalid", f"config lacks architecture identity fields: {config_path}")
    if config["model_type"] != "qwen3" or config["architectures"] != ["Qwen3ForCausalLM"]:
        reject("checkpoint_model_config_invalid", f"unsupported checkpoint architecture: {config.get('model_type')!r}")
    if any(not isinstance(config[key], int) or config[key] <= 0 for key in required[2:]):
        reject("checkpoint_model_config_invalid", f"config dimensions must be positive integers: {config_path}")
    return config


QWEN3_KEY_FAMILIES = (
    re.compile(r"^(?:module\.)?model\.embed_tokens\.weight$"),
    re.compile(r"^(?:module\.)?model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$"),
    re.compile(r"^(?:module\.)?model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"),
    re.compile(r"^(?:module\.)?model\.(?:norm|layers\.\d+\.(?:input_layernorm|post_attention_layernorm))\.weight$"),
    re.compile(r"^(?:module\.)?lm_head\.weight$"),
)


def canonical_model_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def validate_model_tensors(tensors: list[tuple[str, Any]], config: dict[str, Any], path: Path) -> None:
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", hidden // heads if heads and hidden % heads == 0 else 0))
    intermediate = int(config.get("intermediate_size", 0))
    vocab = int(config["vocab_size"])
    if not head_dim or not intermediate or heads % kv_heads != 0:
        reject("checkpoint_model_config_invalid", f"Qwen3 config has invalid head/intermediate dimensions: {path}")
    recognized: list[tuple[str, Any]] = []
    layer_indices: set[int] = set()
    for raw_key, tensor in tensors:
        key = canonical_model_key(raw_key)
        layer_match = re.match(r"^model\.layers\.(\d+)\.", key)
        if layer_match:
            layer_index = int(layer_match.group(1))
            if layer_index >= layers:
                reject("checkpoint_model_layer_out_of_range", f"tensor {raw_key} names layer {layer_index} but config has {layers} layers")
            layer_indices.add(layer_index)
        if any(pattern.fullmatch(key) for pattern in QWEN3_KEY_FAMILIES):
            recognized.append((key, tensor))
        elif key.startswith("model.layers."):
            reject("checkpoint_model_key_invalid", f"unrecognized Qwen3 layer tensor key: {raw_key}")
    keys = {key for key, _ in recognized}
    required_exact = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.get("tie_word_embeddings", False):
        required_exact.add("lm_head.weight")
    family_hits = {
        "attention": any(".self_attn." in key for key in keys),
        "mlp": any(".mlp." in key for key in keys),
    }
    minimum_tensors = max(8, min(16, config["num_hidden_layers"] * 4))
    minimum_numel = max(256, config["hidden_size"] * config["vocab_size"] * 2)
    total_numel = sum(int(tensor.numel()) for _, tensor in recognized)
    expected_layer_indices = set(range(layers))
    if layer_indices != expected_layer_indices:
        reject(
            "checkpoint_model_layer_set_invalid",
            f"model state layer indices {sorted(layer_indices)} do not match config range 0..{layers - 1}: {path}",
        )
    if not required_exact.issubset(keys) or not all(family_hits.values()) or len(recognized) < minimum_tensors or total_numel < minimum_numel:
        reject(
            "checkpoint_model_semantics_invalid",
            f"model state is inconsistent with Qwen3 architecture: recognized_tensors={len(recognized)} numel={total_numel} path={path}",
        )
    expected_shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for key, tensor in recognized:
        shape = tuple(int(value) for value in tensor.shape)
        expected: tuple[int, ...] | None = expected_shapes.get(key)
        layer_match = re.match(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$", key)
        if layer_match:
            projection = layer_match.group(2)
            if projection == "q_proj":
                expected = (heads * head_dim, hidden)
            elif projection in {"k_proj", "v_proj"}:
                expected = (kv_heads * head_dim, hidden)
            else:
                expected = (hidden, heads * head_dim)
        if re.fullmatch(r"model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight", key):
            expected = (hidden,)
        mlp_match = re.match(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", key)
        if mlp_match:
            expected = (intermediate, hidden) if mlp_match.group(1) in {"gate_proj", "up_proj"} else (hidden, intermediate)
        if expected is not None and shape != expected:
            reject("checkpoint_model_shape_invalid", f"tensor {key} has shape {shape}, expected {expected} from config")


def validate_model_mapping(value: dict[str, Any], config: dict[str, Any], path: Path) -> None:
    import torch

    tensors = [(key, item) for key, item in nested_items(value) if torch.is_tensor(item)]
    validate_model_tensors(tensors, config, path)


def find_optimizer_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("state"), dict) and isinstance(value.get("param_groups"), list):
            return value
        for child in value.values():
            found = find_optimizer_mapping(child)
            if found is not None:
                return found
    return None


def validate_optimizer_mapping(value: dict[str, Any], path: Path) -> None:
    import torch

    optimizer = find_optimizer_mapping(value)
    if optimizer is None or not optimizer["state"] or not optimizer["param_groups"]:
        reject("checkpoint_optimizer_semantics_invalid", f"optimizer lacks nonempty state/param_groups: {path}")
    if any(not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"] for group in optimizer["param_groups"]):
        reject("checkpoint_optimizer_semantics_invalid", f"optimizer param_groups are malformed: {path}")
    group_params = {parameter for group in optimizer["param_groups"] for parameter in group["params"]}
    if not group_params.issubset(set(optimizer["state"])):
        reject("checkpoint_optimizer_semantics_invalid", f"optimizer param_groups reference missing state: {path}")
    for parameter in group_params:
        state = optimizer["state"][parameter]
        if not isinstance(state, dict) or set(("step", "exp_avg", "exp_avg_sq")) - state.keys():
            reject("checkpoint_optimizer_semantics_invalid", f"AdamW state is missing step/exp_avg/exp_avg_sq: {path}")
        step = state["step"]
        step_value = float(step.item()) if torch.is_tensor(step) and step.numel() == 1 else step
        if isinstance(step_value, bool) or not isinstance(step_value, (int, float)) or not math.isfinite(step_value) or step_value <= 0:
            reject("checkpoint_optimizer_semantics_invalid", f"AdamW step is invalid: {path}")
        moments = (state["exp_avg"], state["exp_avg_sq"])
        if any(not torch.is_tensor(item) or item.numel() < 4 or not torch.isfinite(item).all() for item in moments):
            reject("checkpoint_optimizer_semantics_invalid", f"AdamW moments are not substantive finite tensors: {path}")
        if moments[0].shape != moments[1].shape:
            reject("checkpoint_optimizer_semantics_invalid", f"AdamW moment shapes differ: {path}")


def find_scheduler_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "last_epoch" in value and "_step_count" in value:
            return value
        for child in value.values():
            found = find_scheduler_mapping(child)
            if found is not None:
                return found
    return None


def validate_scheduler_mapping(value: dict[str, Any], step: int, path: Path) -> None:
    scheduler = find_scheduler_mapping(value)
    if scheduler is None:
        reject("checkpoint_scheduler_semantics_invalid", f"scheduler lacks last_epoch/_step_count: {path}")
    progress = (scheduler["last_epoch"], scheduler["_step_count"])
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0 for item in progress):
        reject("checkpoint_scheduler_semantics_invalid", f"scheduler progress is invalid: {path}")
    if int(scheduler["last_epoch"]) not in {step - 1, step} or int(scheduler["_step_count"]) < step:
        reject("checkpoint_scheduler_semantics_invalid", f"scheduler progress is incoherent with trainer step={step}: {path}")


def validate_rng_mapping(value: dict[str, Any], path: Path) -> None:
    import torch

    if set(("python", "numpy", "cpu", "cuda")) - value.keys():
        reject("checkpoint_rng_semantics_invalid", f"Transformers RNG state lacks python/numpy/cpu/cuda fields: {path}")
    if not isinstance(value["python"], tuple) or not isinstance(value["numpy"], tuple):
        reject("checkpoint_rng_semantics_invalid", f"Python or NumPy RNG state has the wrong type: {path}")
    if not torch.is_tensor(value["cpu"]) or value["cpu"].dtype != torch.uint8 or value["cpu"].numel() < 16:
        reject("checkpoint_rng_semantics_invalid", f"CPU RNG state is invalid: {path}")
    cuda_states = value["cuda"] if isinstance(value["cuda"], (list, tuple)) else [value["cuda"]]
    if not cuda_states or any(not torch.is_tensor(item) or item.dtype != torch.uint8 or item.numel() < 16 for item in cuda_states):
        reject("checkpoint_rng_semantics_invalid", f"CUDA RNG state is invalid: {path}")


def validate_trainer_state(state: object, step: int, path: Path) -> str:
    if not isinstance(state, dict):
        reject("trainer_state_invalid", f"trainer state must be a mapping: {path}")
    max_steps = state.get("max_steps")
    history = state.get("log_history")
    if not isinstance(max_steps, int) or max_steps < step or not isinstance(history, list) or not history:
        reject("trainer_state_invalid", f"trainer state lacks coherent max_steps/log_history lineage: {path}")
    history_steps: list[int] = []
    has_metric = False
    for row in history:
        if not isinstance(row, dict) or not isinstance(row.get("step"), int) or row["step"] <= 0 or row["step"] > step:
            reject("trainer_state_invalid", f"trainer log_history has an incoherent step: {path}")
        history_steps.append(row["step"])
        for key, value in row.items():
            if key not in {"step", "epoch"} and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                has_metric = True
    if history_steps != sorted(history_steps) or not has_metric:
        reject("trainer_state_invalid", f"trainer log_history lacks ordered substantive metrics: {path}")
    metrics_identity = {
        "global_step": step,
        "max_steps": max_steps,
        "epoch": state.get("epoch"),
        "log_history": history,
    }
    return sha256_json(metrics_identity)


def inspect_model_state(path: Path, files: list[Path], config: dict[str, Any]) -> list[str]:
    index_paths = [path / name for name in ("pytorch_model.bin.index.json", "model.safetensors.index.json") if (path / name).is_file()]
    if len(index_paths) > 1:
        reject("checkpoint_model_index_invalid", f"multiple incompatible model indexes found: {path}")
    if index_paths:
        index_path = index_paths[0]
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reject("checkpoint_model_index_invalid", f"cannot parse model index {index_path}: {exc}")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
            reject("checkpoint_model_index_invalid", f"model index weight_map is empty or malformed: {index_path}")
        model_files = [path / name for name in sorted(set(weight_map.values()))]
        if any(not item.is_file() for item in model_files):
            reject("checkpoint_model_shard_missing", f"model index references a missing shard: {index_path}")
    else:
        model_files = [item for item in files if item.name in {"model.safetensors", "pytorch_model.bin"} or item.name.endswith("model_states.pt")]
    if not model_files:
        reject("checkpoint_model_state_missing", f"checkpoint has no model state: {path}")
    all_tensors: list[tuple[str, Any]] = []
    for model_file in model_files:
        if model_file.stat().st_size < 64:
            reject("checkpoint_state_too_small", f"model state is too small to be substantive: {model_file}")
        if model_file.suffix in {".bin", ".pt"}:
            import torch

            all_tensors.extend((key, item) for key, item in nested_items(load_torch_state(model_file, "model")) if torch.is_tensor(item))
        elif model_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(model_file, framework="pt", device="cpu") as handle:
                    all_tensors.extend((key, handle.get_tensor(key)) for key in handle.keys())
            except GateError:
                raise
            except Exception as exc:
                reject("checkpoint_state_invalid", f"cannot parse safetensors model {model_file}: {type(exc).__name__}: {exc}")
    validate_model_tensors(all_tensors, config, path)
    return [str(item.relative_to(path)) for item in model_files]


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    step, state_path = checkpoint_step(path)
    files = [item for item in path.rglob("*") if item.is_file()]
    config = load_hf_config(path)
    model_files = inspect_model_state(path, files, config)
    optimizer_files = [item for item in files if item.name == "optimizer.pt" or item.name.endswith("optim_states.pt")]
    if not optimizer_files:
        reject("checkpoint_optimizer_state_missing", f"checkpoint has no optimizer state: {path}")
    scheduler = path / "scheduler.pt"
    if not scheduler.is_file():
        reject("checkpoint_scheduler_state_missing", f"checkpoint has no scheduler.pt: {path}")
    rng_files = [item for item in files if item.name == "rng_state.pth" or re.fullmatch(r"rng_state_\d+\.pth", item.name)]
    if not rng_files:
        reject("checkpoint_rng_state_missing", f"checkpoint has no RNG state: {path}")
    for item in optimizer_files:
        validate_optimizer_mapping(load_torch_state(item, "optimizer"), item)
    validate_scheduler_mapping(load_torch_state(scheduler, "scheduler"), step, scheduler)
    for item in rng_files:
        validate_rng_mapping(load_torch_state(item, "RNG"), item)
    if state_path.stat().st_size < 64:
        reject("trainer_state_invalid", f"trainer state is too small to establish substantive lineage: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    training_metrics_sha256 = validate_trainer_state(state, step, state_path)
    return {
        "path": str(path.resolve()),
        "step": step,
        "tree_sha256": sha256_tree(path),
        "trainer_state_sha256": sha256_file(state_path),
        "model_files": model_files,
        "model_config_sha256": sha256_file(path / "config.json"),
        "training_metrics_sha256": training_metrics_sha256,
    }


def smoke_identities(args: argparse.Namespace) -> dict[str, str]:
    return {
        "commit_hash": args.commit_hash,
        "config_sha256": args.config_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_source": args.dataset_source,
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": args.dataset_sha256,
        "eval_dataset_sha256": args.eval_dataset_sha256,
        "prompt_sha256": args.prompt_sha256,
        "retrieval_sha256": args.retrieval_sha256,
        "source_schema_sha256": args.source_schema_sha256,
        "optimization_decision_sha256": args.optimization_decision_sha256,
        "scale_decision_sha256": args.scale_decision_sha256,
        "method": args.method,
    }


def cmd_create_smoke_manifest(args: argparse.Namespace) -> None:
    initial = inspect_checkpoint(args.initial_checkpoint)
    resumed = inspect_checkpoint(args.resumed_checkpoint)
    initial_step = initial["step"]
    resumed_step = resumed["step"]
    if resumed_step <= initial_step:
        reject("resume_step_not_advanced", f"resumed step {resumed_step} must be greater than saved step {initial_step}")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "fallback_reason": "none",
        "identities": smoke_identities(args),
        "package_versions": required_package_versions(),
        "initial_checkpoint": initial,
        "resumed_checkpoint": resumed,
        "resume_from_checkpoint": initial["path"],
        "step_delta": resumed_step - initial_step,
        "training_metrics_lineage_sha256": sha256_json([initial["training_metrics_sha256"], resumed["training_metrics_sha256"]]),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def cmd_validate_checkpoints(args: argparse.Namespace) -> None:
    if not args.smoke_manifest.is_file():
        reject("smoke_manifest_missing", f"smoke manifest does not exist: {args.smoke_manifest}")
    actual_hash = sha256_file(args.smoke_manifest)
    if actual_hash != args.expected_sha256:
        reject("smoke_manifest_hash_mismatch", f"smoke manifest hash is {actual_hash}, expected {args.expected_sha256}")
    try:
        manifest = json.loads(args.smoke_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reject("smoke_manifest_invalid", f"cannot parse smoke manifest: {exc}")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "passed":
        reject("smoke_manifest_invalid", "smoke manifest must have schema_version=1 and status=passed")
    expected_identities = smoke_identities(args)
    if manifest.get("identities") != expected_identities:
        reject("smoke_identity_mismatch", "smoke manifest identities do not match the full launch")
    if manifest.get("package_versions") != required_package_versions():
        reject("smoke_package_identity_mismatch", "smoke manifest package versions differ from the validation environment")
    initial = inspect_checkpoint(Path(manifest["initial_checkpoint"]["path"]))
    resumed = inspect_checkpoint(Path(manifest["resumed_checkpoint"]["path"]))
    if initial != manifest["initial_checkpoint"] or resumed != manifest["resumed_checkpoint"]:
        reject("smoke_checkpoint_hash_mismatch", "checkpoint contents changed after smoke manifest creation")
    if manifest.get("resume_from_checkpoint") != initial["path"] or resumed["step"] <= initial["step"]:
        reject("smoke_lineage_invalid", "smoke resume lineage or step continuity is invalid")
    expected_metrics_lineage = sha256_json([initial["training_metrics_sha256"], resumed["training_metrics_sha256"]])
    if manifest.get("training_metrics_lineage_sha256") != expected_metrics_lineage:
        reject("smoke_metrics_lineage_mismatch", "smoke training metrics lineage identity is invalid")
    print(json.dumps({"status": "validated", "smoke_manifest_sha256": actual_hash, "fallback_reason": "none"}, sort_keys=True))


class TelemetrySampler:
    def __init__(self, hardware: dict[str, Any], interval_seconds: float) -> None:
        self.hardware = hardware
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.query_errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.started_at = 0.0
        self.ended_at = 0.0

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
                for line in output.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) != 5:
                        continue
                    self.samples.append({
                        "uuid": fields[0], "utilization_percent": float(fields[1]),
                        "memory_used_mb": float(fields[2]), "power_watts": float(fields[3]),
                        "temperature_c": float(fields[4]), "timestamp": time.monotonic(),
                    })
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.query_errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> TelemetrySampler:
        self.started_at = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.ended_at = time.monotonic()

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            reject("gpu_telemetry_unavailable", "nvidia-smi produced no utilization telemetry during the measured probe")
        if self.query_errors:
            reject("gpu_telemetry_query_failed", f"nvidia-smi telemetry query failed: {self.query_errors[0]}")
        duration = self.ended_at - self.started_at
        hardware_uuids = sorted(device["uuid"] for device in self.hardware["devices"])
        monitored_uuids = sorted({str(item["uuid"]) for item in self.samples})
        sample_rounds = len(self.samples) / len(hardware_uuids)
        expected_rounds = max(1.0, duration / self.interval_seconds)
        return {
            "sample_count": int(sample_rounds),
            "monitor_interval_seconds": self.interval_seconds,
            "measured_duration_seconds": duration,
            "coverage_ratio": min(1.0, sample_rounds / expected_rounds),
            "monitored_uuids": monitored_uuids,
            "query_errors": self.query_errors,
            "utilization_mean_percent": sum(item["utilization_percent"] for item in self.samples) / len(self.samples),
            "utilization_peak_percent": max(item["utilization_percent"] for item in self.samples),
            "nvidia_smi_peak_memory_mb": max(item["memory_used_mb"] for item in self.samples),
            "power_peak_watts": max(item["power_watts"] for item in self.samples),
            "temperature_peak_c": max(item["temperature_c"] for item in self.samples),
        }


def read_probe_prompts(path: Path, sample_size: int) -> list[str]:
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                reject("probe_dataset_invalid", f"invalid JSONL at line {line_number}: {exc}")
            prompt = row.get("prompt") or row.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                reject("probe_prompt_missing", f"probe row {line_number} has no nonempty prompt or question")
            prompts.append(prompt)
            if len(prompts) == sample_size:
                break
    if len(prompts) < sample_size:
        reject("probe_sample_too_small", f"probe requires {sample_size} rows, found {len(prompts)}")
    return prompts


def cmd_benchmark(args: argparse.Namespace) -> None:
    started_at = time.time()
    if args.max_steps <= 0:
        reject("max_steps_invalid", f"--max-steps must be positive before a measured probe can launch: {args.max_steps}")
    base_result = {
        "schema_version": 1,
        "probe_name": args.probe_name,
        "status": "rejected",
        "fallback_reason": "probe_not_completed",
        "package_versions": required_package_versions(),
        "config": {
            "probe_kind": args.probe_kind,
            "attention_implementation": args.attention_implementation,
            "generation_batch_size": args.generation_batch_size,
            "static_cache": args.static_cache,
            "compile": args.compile,
            "train_microbatch": args.train_microbatch,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "dataloader_workers": args.dataloader_workers,
            "packing": args.packing,
            "padding_free": args.padding_free,
            "use_liger_kernel": args.use_liger_kernel,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "training_method": args.training_method,
            "num_generations": args.num_generations,
            "rl_generation_batch_size": args.rl_generation_batch_size,
            "max_completion_length": args.max_completion_length,
        },
        "identities": {
            "commit_hash": args.commit_hash,
            "config_sha256": sha256_file(args.config),
            "model": args.model,
            "model_revision": args.model_revision,
            "dataset_source": args.dataset_source,
            "dataset_revision": args.dataset_revision,
            "dataset_sha256": sha256_file(args.data),
            "prompt_sha256": args.prompt_sha256,
            "retrieval_sha256": args.retrieval_sha256,
            "source_schema_sha256": args.source_schema_sha256,
        },
    }
    try:
        if args.use_liger_kernel:
            reject("use_liger_kernel_incompatible_with_precompute_ref_log_probs", "primary DPO requires precompute_ref_log_probs=True")
        if args.packing or args.padding_free:
            if args.attention_implementation != "flash_attention_2":
                reject("flash_attention_2_required", "padding-free and packing probes require FlashAttention2 compatibility first")
        if args.attention_implementation == "flash_attention_2" and base_result["package_versions"]["flash-attn"] == "missing":
            reject("flash_attention_2_import_unavailable", "flash-attn is not installed")

        import torch

        if not torch.cuda.is_available():
            reject("cuda_unavailable", "measured Turing probes refuse CPU fallback")
        base_result["gpu_hardware"] = gpu_hardware_identity(torch)
        if args.probe_kind == "training":
            if args.eval_data is None or args.output_dir is None or args.deepspeed_config is None:
                reject("training_probe_contract_missing", "training probes require --eval-data, --output-dir, and --deepspeed-config")
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={base_result['gpu_hardware']['count']}", "-m", "text_feedback_dpo.cli", f"train-{args.training_method}",
                "--config", str(args.config), "--train", str(args.data), "--eval", str(args.eval_data),
                "--output", str(args.output_dir), "--model", args.model, "--model-revision", args.model_revision,
                "--max-steps", str(args.max_steps), "--max-length", str(args.max_length),
                "--per-device-train-batch-size", str(args.train_microbatch),
                "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
                "--dataloader-num-workers", str(args.dataloader_workers),
                "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                "--attention-implementation", args.attention_implementation,
                "--deepspeed-config", str(args.deepspeed_config), "--save-steps", str(args.max_steps),
                "--eval-steps", str(args.max_steps), "--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing",
                "--packing" if args.packing else "--no-packing",
                "--padding-free" if args.padding_free else "--no-padding-free",
                "--use-liger-kernel" if args.use_liger_kernel else "--no-use-liger-kernel",
                "--num-generations", str(args.num_generations), "--generation-batch-size", str(args.rl_generation_batch_size),
                "--max-completion-length", str(args.max_completion_length),
            ]
            if args.training_method == "dpo":
                if any(value is None for value in (args.ref_log_probs, args.eval_ref_log_probs, args.reference_checkpoint_hash, args.prompt_context_schema)):
                    reject("dpo_training_probe_contract_missing", "DPO probes require ref/eval log probabilities, reference checkpoint hash, and prompt context schema")
                command.extend([
                    "--ref-log-probs", str(args.ref_log_probs), "--eval-ref-log-probs", str(args.eval_ref_log_probs),
                    "--reference-checkpoint-hash", args.reference_checkpoint_hash,
                    "--prompt-context-schema", str(args.prompt_context_schema),
                ])
            torch.cuda.reset_peak_memory_stats()
            with TelemetrySampler(base_result["gpu_hardware"], args.monitor_interval_seconds) as telemetry:
                completed = subprocess.run(command, text=True, capture_output=True, timeout=args.training_timeout_seconds)
            if completed.returncode != 0:
                reject("training_probe_failed", f"bounded trainer exited {completed.returncode}: {completed.stderr[-2000:]}")
            states = sorted(args.output_dir.rglob("trainer_state.json"))
            if not states:
                reject("training_probe_metrics_missing", "bounded trainer emitted no trainer_state.json")
            state = json.loads(states[-1].read_text(encoding="utf-8"))
            history = state.get("log_history")
            if not isinstance(history, list) or not history:
                reject("training_probe_metrics_missing", "bounded trainer emitted no log history")
            numeric: dict[str, float] = {}
            for row in history:
                if isinstance(row, dict):
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                            numeric[key] = float(value)
            train_loss = numeric.get("train_loss", numeric.get("loss"))
            eval_loss = numeric.get("eval_loss")
            runtime = numeric.get("train_runtime")
            input_tokens = numeric.get("num_input_tokens_seen")
            samples_per_second = numeric.get("train_samples_per_second")
            if any(value is None or value <= 0 for value in (runtime, input_tokens, samples_per_second)) or train_loss is None or eval_loss is None:
                reject("training_probe_metrics_missing", "trainer metrics require finite train/eval loss, runtime, samples/s, and num_input_tokens_seen")
            metrics = {"train_loss": train_loss, "eval_loss": eval_loss}
            base_result.update({
                "status": "ok", "fallback_reason": "none", "finite_metrics": metrics,
                "global_examples_per_second": samples_per_second,
                "global_tokens_per_second": input_tokens / runtime,
                "examples_per_second": samples_per_second, "tokens_per_second": input_tokens / runtime,
                "correctness_hash": sha256_json({"eval_dataset_sha256": sha256_file(args.eval_data), "metric_names": sorted(metrics)}),
                "output_hash": sha256_json(state["log_history"]), "decoded_output_hash": sha256_json(metrics),
                "output_token_ids": [[int(state.get("global_step", 0))]], "decoded_outputs": [json.dumps(metrics, sort_keys=True)],
                "peak_gpu_memory_mb": max(torch.cuda.max_memory_allocated() / (1024 * 1024), telemetry.summary()["nvidia_smi_peak_memory_mb"]),
                "gpu_utilization": telemetry.summary(), "trainer_state_sha256": sha256_file(states[-1]),
            })
            base_result["wall_seconds"] = time.time() - started_at
            write_json(args.result, base_result)
            print(json.dumps(base_result, sort_keys=True))
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        prompts = read_probe_prompts(args.data, args.sample_size)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.model_revision,
            attn_implementation=args.attention_implementation,
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        model.eval()
        if args.compile:
            model = torch.compile(model)

        def one_pass() -> tuple[list[list[int]], int]:
            completion_ids: list[list[int]] = []
            generated_tokens = 0
            for start in range(0, len(prompts), args.generation_batch_size):
                batch = prompts[start : start + args.generation_batch_size]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096 - args.max_new_tokens)
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if args.static_cache:
                    generate_kwargs["cache_implementation"] = "static"
                with torch.inference_mode():
                    sequences = model.generate(**encoded, **generate_kwargs)
                prompt_width = encoded["input_ids"].shape[1]
                for sequence in sequences[:, prompt_width:].detach().cpu().tolist():
                    completion_ids.append(sequence)
                    generated_tokens += len(sequence)
            return completion_ids, generated_tokens

        for _ in range(args.warmup_repeats):
            one_pass()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        repeat_seconds: list[float] = []
        repeat_tokens: list[int] = []
        repeat_hashes: list[str] = []
        final_ids: list[list[int]] = []
        with TelemetrySampler(base_result["gpu_hardware"], args.monitor_interval_seconds) as telemetry:
            for _ in range(args.measured_repeats):
                before = time.perf_counter()
                token_ids, generated_tokens = one_pass()
                torch.cuda.synchronize()
                repeat_seconds.append(time.perf_counter() - before)
                repeat_tokens.append(generated_tokens)
                repeat_hashes.append(sha256_json(token_ids))
                final_ids = token_ids
        if len(set(repeat_hashes)) != 1:
            reject("repeated_output_hash_mismatch", "deterministic repeated probe outputs differ")
        total_seconds = sum(repeat_seconds)
        total_examples = args.sample_size * args.measured_repeats
        total_tokens = sum(repeat_tokens)
        decoded = tokenizer.batch_decode(final_ids, skip_special_tokens=False)
        base_result.update(
            {
                "status": "ok",
                "fallback_reason": "none",
                "warmup_repeats": args.warmup_repeats,
                "measured_repeats": args.measured_repeats,
                "repeat_seconds": repeat_seconds,
                "examples_per_second": total_examples / total_seconds,
                "tokens_per_second": total_tokens / total_seconds,
                "generated_tokens": total_tokens,
                "output_hash": repeat_hashes[0],
                "decoded_output_hash": sha256_json(decoded),
                "output_token_ids": final_ids,
                "decoded_outputs": decoded,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "gpu_utilization": telemetry.summary(),
            }
        )
    except GateError as exc:
        base_result.update({"status": "rejected", "fallback_reason": exc.fallback_reason, "error": str(exc)})
    except Exception as exc:  # hardware/model compatibility failures are probe results
        base_result.update(
            {
                "status": "rejected",
                "fallback_reason": "probe_runtime_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    base_result["wall_seconds"] = time.time() - started_at
    write_json(args.result, base_result)
    print(json.dumps(base_result, sort_keys=True))


PROBE_IDENTITY_FIELDS = (
    "commit_hash", "config_sha256", "model", "model_revision", "dataset_source",
    "dataset_revision", "dataset_sha256", "prompt_sha256", "retrieval_sha256",
    "source_schema_sha256",
)
PROBE_CONFIG_FIELDS = (
    "probe_kind", "attention_implementation", "generation_batch_size", "static_cache",
    "compile", "train_microbatch", "gradient_accumulation_steps", "dataloader_workers",
    "packing", "padding_free", "use_liger_kernel", "per_device_eval_batch_size",
    "max_steps", "max_length", "gradient_checkpointing",
    "num_generations", "rl_generation_batch_size", "max_completion_length",
    "training_method",
)
TELEMETRY_FIELDS = (
    "sample_count", "monitor_interval_seconds", "measured_duration_seconds", "coverage_ratio", "monitored_uuids",
    "query_errors",
    "utilization_mean_percent", "utilization_peak_percent",
    "nvidia_smi_peak_memory_mb", "power_peak_watts", "temperature_peak_c",
)


def gpu_hardware_error(value: object) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("count"), int) or value["count"] <= 0:
        return "probe_gpu_hardware_invalid"
    devices = value.get("devices")
    if not isinstance(devices, list) or len(devices) != value["count"]:
        return "probe_gpu_hardware_invalid"
    uuids: set[str] = set()
    for index, device in enumerate(devices):
        if not isinstance(device, dict) or device.get("index") != index:
            return "probe_gpu_hardware_invalid"
        if not isinstance(device.get("name"), str) or not device["name"] or normalize_gpu_name(device["name"]) != device["name"]:
            return "probe_gpu_hardware_invalid"
        uuid = device.get("uuid")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or uuid in uuids:
            return "probe_gpu_hardware_invalid"
        uuids.add(uuid)
        if not isinstance(device.get("total_memory_bytes"), int) or device["total_memory_bytes"] <= 0:
            return "probe_gpu_hardware_invalid"
        if not isinstance(device.get("free_memory_bytes"), int) or not 0 < device["free_memory_bytes"] <= device["total_memory_bytes"]:
            return "probe_gpu_hardware_invalid"
        if not isinstance(device.get("compute_capability"), str) or re.fullmatch(r"[1-9][0-9]*\.[0-9]+", device["compute_capability"]) is None:
            return "probe_gpu_hardware_invalid"
    return None


def probe_artifact_error(result: object) -> str | None:
    if not isinstance(result, dict):
        return "probe_not_object"
    if result.get("schema_version") != DECISION_SCHEMA_VERSION:
        return "probe_schema_invalid"
    if result.get("status") != "ok" or result.get("fallback_reason") != "none":
        return "probe_status_invalid"
    for key in ("output_hash", "decoded_output_hash"):
        if not isinstance(result.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", result[key]) is None:
            return f"probe_{key}_invalid"
    token_ids = result.get("output_token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        not isinstance(row, list) or not row or any(not isinstance(token, int) or token < 0 for token in row)
        for row in token_ids
    ):
        return "probe_output_token_ids_invalid"
    probe_kind = result.get("config", {}).get("probe_kind") if isinstance(result.get("config"), dict) else None
    if probe_kind == "generation" and sha256_json(token_ids) != result["output_hash"]:
        return "probe_output_hash_not_from_token_ids"
    decoded_outputs = result.get("decoded_outputs")
    if not isinstance(decoded_outputs, list) or not decoded_outputs or any(not isinstance(value, str) for value in decoded_outputs):
        return "probe_decoded_outputs_invalid"
    if probe_kind == "generation" and sha256_json(decoded_outputs) != result["decoded_output_hash"]:
        return "probe_decoded_hash_not_from_outputs"
    for key in ("examples_per_second", "tokens_per_second", "peak_gpu_memory_mb"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return f"probe_{key}_invalid"
    telemetry = result.get("gpu_utilization")
    if not isinstance(telemetry, dict) or set(TELEMETRY_FIELDS) - telemetry.keys():
        return "probe_gpu_telemetry_invalid"
    numeric_telemetry_fields = set(TELEMETRY_FIELDS) - {"monitored_uuids", "query_errors"}
    for key in numeric_telemetry_fields:
        value = telemetry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "probe_gpu_telemetry_invalid"
        if key == "sample_count" and (not isinstance(value, int) or value < MIN_TELEMETRY_SAMPLES):
            return "probe_gpu_telemetry_invalid"
        if key != "sample_count" and value <= 0:
            return "probe_gpu_telemetry_invalid"
    hardware_error = gpu_hardware_error(result.get("gpu_hardware"))
    if hardware_error:
        return hardware_error
    hardware_uuids = sorted(device["uuid"] for device in result["gpu_hardware"]["devices"])
    if telemetry["monitored_uuids"] != hardware_uuids:
        return "probe_gpu_telemetry_invalid"
    if not isinstance(telemetry["query_errors"], list) or telemetry["query_errors"]:
        return "probe_gpu_telemetry_query_failed"
    expected_samples = math.floor(telemetry["measured_duration_seconds"] / telemetry["monitor_interval_seconds"])
    if telemetry["sample_count"] < max(MIN_TELEMETRY_SAMPLES, math.floor(expected_samples * MIN_TELEMETRY_COVERAGE)):
        return "probe_gpu_telemetry_invalid"
    if not MIN_TELEMETRY_COVERAGE <= telemetry["coverage_ratio"] <= 1.0:
        return "probe_gpu_telemetry_invalid"
    packages = result.get("package_versions")
    if not isinstance(packages, dict) or set(REQUIRED_PACKAGES) - packages.keys() or any(
        not isinstance(value, str) or not value for value in packages.values()
    ):
        return "probe_package_versions_invalid"
    if any(packages[name] == "missing" for name in CORE_PACKAGES):
        return "probe_package_versions_invalid"
    identities = result.get("identities")
    if not isinstance(identities, dict) or set(PROBE_IDENTITY_FIELDS) - identities.keys() or any(
        not isinstance(identities[key], str) or not identities[key] for key in PROBE_IDENTITY_FIELDS
    ):
        return "probe_identities_invalid"
    config = result.get("config")
    if not isinstance(config, dict) or set(PROBE_CONFIG_FIELDS) - config.keys():
        return "probe_config_invalid"
    if config["probe_kind"] not in {"generation", "training"} or config["attention_implementation"] not in {"sdpa", "flash_attention_2"}:
        return "probe_config_invalid"
    if config["training_method"] not in {"sft", "dpo", "grpo", "dapo"}:
        return "probe_config_invalid"
    for key in ("generation_batch_size", "train_microbatch", "gradient_accumulation_steps", "per_device_eval_batch_size", "max_steps", "max_length", "num_generations", "rl_generation_batch_size", "max_completion_length"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            return "probe_config_invalid"
    if not isinstance(config["dataloader_workers"], int) or isinstance(config["dataloader_workers"], bool) or config["dataloader_workers"] < 0:
        return "probe_config_invalid"
    if any(not isinstance(config[key], bool) for key in ("static_cache", "compile", "packing", "padding_free", "use_liger_kernel", "gradient_checkpointing")):
        return "probe_config_invalid"
    if config["attention_implementation"] == "flash_attention_2" and packages["flash-attn"] == "missing":
        return "probe_package_versions_invalid"
    if config["use_liger_kernel"] and packages["liger-kernel"] == "missing":
        return "probe_package_versions_invalid"
    if config["probe_kind"] == "training":
        metrics = result.get("finite_metrics")
        if not isinstance(metrics, dict) or set(("train_loss", "eval_loss")) - metrics.keys() or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in metrics.values()
        ):
            return "probe_training_metrics_invalid"
        for key in ("global_examples_per_second", "global_tokens_per_second"):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                return "probe_training_metrics_invalid"
        if not isinstance(result.get("correctness_hash"), str) or re.fullmatch(r"[0-9a-f]{64}", result["correctness_hash"]) is None:
            return "probe_training_metrics_invalid"
    return None


def accepted_candidate(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    baseline_error = probe_artifact_error(baseline)
    if baseline_error:
        return False, f"baseline_{baseline_error}"
    candidate_error = probe_artifact_error(candidate)
    if candidate_error:
        return False, f"candidate_{candidate_error}"
    if candidate["identities"] != baseline["identities"]:
        return False, "identity_parity_mismatch"
    if candidate["package_versions"] != baseline["package_versions"]:
        return False, "package_parity_mismatch"
    if candidate["gpu_hardware"] != baseline["gpu_hardware"]:
        return False, "gpu_hardware_parity_mismatch"
    baseline_coverage = baseline["gpu_utilization"]["coverage_ratio"]
    candidate_coverage = candidate["gpu_utilization"]["coverage_ratio"]
    if abs(candidate_coverage - baseline_coverage) > MAX_TELEMETRY_COVERAGE_DELTA:
        return False, "gpu_telemetry_coverage_parity_mismatch"
    if baseline["config"]["probe_kind"] == "training":
        if candidate.get("correctness_hash") != baseline.get("correctness_hash"):
            return False, "training_correctness_mismatch"
        for key in ("train_loss", "eval_loss"):
            baseline_loss = float(baseline["finite_metrics"][key])
            candidate_loss = float(candidate["finite_metrics"][key])
            if abs(candidate_loss - baseline_loss) / max(abs(baseline_loss), 1e-12) > 0.01:
                return False, "training_finite_loss_parity_mismatch"
    else:
        if candidate.get("output_hash") != baseline.get("output_hash"):
            return False, "output_hash_mismatch"
        if candidate.get("decoded_output_hash") != baseline.get("decoded_output_hash"):
            return False, "decoded_output_hash_mismatch"
    if float(candidate.get("tokens_per_second", 0)) <= float(baseline.get("tokens_per_second", 0)):
        return False, "throughput_not_improved"
    return True, "none"


def deployable_candidate(baseline: dict[str, Any], candidate: dict[str, Any], purpose: str) -> tuple[bool, str]:
    accepted, reason = accepted_candidate(baseline, candidate)
    if not accepted:
        return accepted, reason
    config = candidate.get("config", {})
    if config.get("static_cache"):
        return False, "launch_unsupported_static_cache"
    if config.get("compile"):
        return False, "launch_unsupported_compile"
    if config.get("probe_kind") != purpose:
        return False, "probe_kind_mismatch"
    if config.get("use_liger_kernel"):
        return False, "use_liger_kernel_incompatible_with_precompute_ref_log_probs"
    return True, "none"


def cmd_compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    accepted, reason = accepted_candidate(baseline, candidate)
    payload = {"status": "accepted" if accepted else "rejected", "accepted": accepted, "fallback_reason": reason}
    print(json.dumps(payload, sort_keys=True))
    if not accepted:
        raise SystemExit(1)


def cmd_freeze_decision(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_error = probe_artifact_error(baseline)
    if baseline_error or baseline.get("config", {}).get("attention_implementation") != "sdpa" or baseline.get("config", {}).get("probe_kind") != args.purpose:
        reject("baseline_probe_invalid", f"cannot freeze without a complete measured SDPA baseline: {baseline_error or 'attention_not_sdpa'}")
    accepted: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        ok, reason = deployable_candidate(baseline, candidate, args.purpose)
        if ok:
            accepted.append((path, candidate))
        else:
            rejected.append({"path": str(path), "fallback_reason": reason})
    selected_path, selected_result = max(
        accepted,
        key=lambda item: float(item[1]["tokens_per_second"]),
        default=(args.baseline, baseline),
    )
    config = selected_result["config"]
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": "frozen",
        "fallback_reason": "none" if selected_path != args.baseline else "sdpa_baseline_selected",
        "selected": {
            "attention_implementation": config["attention_implementation"],
            "generation": {
                "query_batch_size": config["generation_batch_size"],
                "response_batch_size": config["generation_batch_size"],
                "query_max_new_tokens": args.query_max_new_tokens,
                "response_max_new_tokens": args.response_max_new_tokens,
                "static_cache": config["static_cache"],
                "compile": config["compile"],
                "student_thinking_mode": args.student_thinking_mode,
                "scratchpad_max_new_tokens": args.scratchpad_max_new_tokens,
                "query_temperature": args.query_temperature,
                "response_temperature": args.response_temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "k1": args.k1,
                "b": args.b,
            },
            "training": {
                "microbatch": config["train_microbatch"],
                "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                "dataloader_workers": config["dataloader_workers"],
                "per_device_eval_batch_size": config["per_device_eval_batch_size"],
                "max_steps": args.launch_max_steps,
                "max_length": config["max_length"],
                "gradient_checkpointing": config["gradient_checkpointing"],
                "method": config["training_method"],
                "learning_rate": args.launch_learning_rate,
                "epochs": args.launch_epochs,
                "save_steps": args.launch_save_steps,
                "eval_steps": args.launch_eval_steps,
                "num_generations": config["num_generations"],
                "generation_batch_size": config["rl_generation_batch_size"],
                "max_completion_length": config["max_completion_length"],
                "packing": config["packing"],
                "padding_free": config["padding_free"],
                "use_liger_kernel": config["use_liger_kernel"],
            },
        },
        "identities": baseline["identities"],
        "package_versions": baseline["package_versions"],
        "gpu_hardware": baseline["gpu_hardware"],
        "baseline_result": str(args.baseline.resolve()),
        "baseline_result_sha256": sha256_file(args.baseline),
        "selected_result": str(selected_path.resolve()),
        "selected_result_sha256": sha256_file(selected_path),
        "accepted_candidates": [str(path.resolve()) for path, _ in accepted],
        "rejected_candidates": rejected,
    }
    write_json(args.output, decision)
    print(json.dumps({"status": "frozen", "decision": str(args.output), "decision_sha256": sha256_file(args.output), "fallback_reason": decision["fallback_reason"]}, sort_keys=True))


def scale_hardware_profile(hardware: dict[str, Any]) -> dict[str, Any]:
    error = gpu_hardware_error(hardware)
    if error:
        reject(error, "scale result has invalid GPU hardware identity")
    profiles = {(device["name"], device["total_memory_bytes"], device["compute_capability"]) for device in hardware["devices"]}
    if len(profiles) != 1:
        reject("scale_hardware_not_homogeneous", "each scaling run must use homogeneous GPUs")
    name, memory, capability = next(iter(profiles))
    if "a100" not in name:
        reject("scale_hardware_not_a100", f"scaling contract is restricted to measured A100 hardware, got {name!r}")
    return {"name": name, "total_memory_bytes": memory, "compute_capability": capability}


def cmd_freeze_scale_decision(args: argparse.Namespace) -> None:
    if len(args.result) < 2:
        reject("scale_results_missing", "freeze-scale-decision requires measured 4-GPU and 8-GPU results")
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in args.result:
        result = json.loads(path.read_text(encoding="utf-8"))
        error = probe_artifact_error(result)
        if error or result.get("config", {}).get("probe_kind") != "training":
            reject("scale_probe_invalid", f"invalid measured training result {path}: {error or 'not_training'}")
        records.append((path, result))
    counts = sorted(result["gpu_hardware"]["count"] for _, result in records)
    if counts != [4, 8]:
        reject("scale_gpu_counts_invalid", f"scale decision requires exactly one 4-GPU and one 8-GPU result, got {counts}")
    baseline = records[0][1]
    profile = scale_hardware_profile(baseline["gpu_hardware"])
    parity_config = {key: value for key, value in baseline["config"].items()}
    baseline_eval_loss = float(baseline["finite_metrics"]["eval_loss"])
    for _, result in records[1:]:
        if result["identities"] != baseline["identities"] or result["package_versions"] != baseline["package_versions"] or result["config"] != parity_config:
            reject("scale_identity_parity_mismatch", "scaling runs differ in model/data/config/package/control identity")
        if scale_hardware_profile(result["gpu_hardware"]) != profile:
            reject("scale_hardware_profile_mismatch", "scaling runs differ in GPU model, memory, or compute capability")
        if result["correctness_hash"] != baseline["correctness_hash"]:
            reject("scale_correctness_parity_mismatch", "scaling runs differ in correctness identity")
        loss = float(result["finite_metrics"]["eval_loss"])
        denominator = max(abs(baseline_eval_loss), 1e-12)
        if abs(loss - baseline_eval_loss) / denominator > args.loss_relative_tolerance:
            reject("scale_finite_loss_parity_mismatch", "scaling eval losses exceed the configured relative tolerance")
        coverage_a = baseline["gpu_utilization"]["coverage_ratio"]
        coverage_b = result["gpu_utilization"]["coverage_ratio"]
        if abs(coverage_a - coverage_b) > MAX_TELEMETRY_COVERAGE_DELTA:
            reject("scale_telemetry_coverage_mismatch", "scaling telemetry coverage is not comparable")
    selected_path, selected = max(records, key=lambda item: float(item[1]["global_tokens_per_second"]))
    payload = {
        "schema_version": DECISION_SCHEMA_VERSION, "status": "frozen", "decision_kind": "gpu_scaling",
        "fallback_reason": "none", "compared_gpu_counts": counts,
        "selected_train_gpus": selected["gpu_hardware"]["count"],
        "selected_global_examples_per_second": selected["global_examples_per_second"],
        "selected_global_tokens_per_second": selected["global_tokens_per_second"],
        "loss_relative_tolerance": args.loss_relative_tolerance, "hardware_profile": profile,
        "selected_gpu_hardware": selected["gpu_hardware"], "identities": selected["identities"],
        "package_versions": selected["package_versions"], "training_controls": selected["config"],
        "selected_result": str(selected_path.resolve()), "selected_result_sha256": sha256_file(selected_path),
        "results": [{"path": str(path.resolve()), "sha256": sha256_file(path), "gpu_count": result["gpu_hardware"]["count"],
                     "global_tokens_per_second": result["global_tokens_per_second"], "finite_metrics": result["finite_metrics"]}
                    for path, result in records],
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "frozen", "decision_sha256": sha256_file(args.output), "selected_train_gpus": payload["selected_train_gpus"], "fallback_reason": "none"}, sort_keys=True))


def cmd_validate_scale_decision(args: argparse.Namespace) -> None:
    decision, digest = load_scale_decision(args.decision, args.expected_sha256)
    if decision["selected_train_gpus"] != args.train_gpus:
        reject("scale_allocation_mismatch", f"scale decision selected {decision['selected_train_gpus']} GPUs, launch requested {args.train_gpus}")
    if decision.get("training_controls", {}).get("training_method") != args.training_method:
        reject("scale_training_method_mismatch", "scale decision was measured with a different trainer method")
    expected = {"commit_hash": args.commit_hash, "config_sha256": args.config_sha256, "model": args.model,
                "model_revision": args.model_revision, "dataset_source": args.dataset_source,
                "dataset_revision": args.dataset_revision, "dataset_sha256": args.dataset_sha256,
                "prompt_sha256": args.prompt_sha256, "retrieval_sha256": args.retrieval_sha256,
                "source_schema_sha256": args.source_schema_sha256}
    if any(decision["identities"].get(key) != value for key, value in expected.items()):
        reject("scale_identity_mismatch", "scale decision does not match launch model/data/config identities")
    print(f"{decision['selected_train_gpus']}\t{digest}\t{decision['fallback_reason']}")


def load_scale_decision(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        reject("scale_decision_missing", f"scale decision does not exist: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        reject("scale_decision_hash_mismatch", f"scale decision hash is {digest}, expected {expected_sha256}")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION or decision.get("status") != "frozen" or decision.get("decision_kind") != "gpu_scaling":
        reject("scale_decision_invalid", "scale decision schema/status/kind is invalid")
    if decision.get("selected_train_gpus") not in {4, 8} or decision.get("compared_gpu_counts") != [4, 8]:
        reject("scale_decision_invalid", "scale decision must be measured across 4 and 8 GPUs")
    if decision.get("package_versions") != required_package_versions():
        reject("scale_package_identity_mismatch", "scale decision package versions differ from the frozen environment")
    if not isinstance(decision.get("hardware_profile"), dict) or set(("name", "total_memory_bytes", "compute_capability")) - decision["hardware_profile"].keys():
        reject("scale_decision_invalid", "scale decision hardware profile is missing")
    if "a100" not in normalize_gpu_name(str(decision["hardware_profile"]["name"])):
        reject("scale_decision_invalid", "scale decision hardware profile is not A100")
    if gpu_hardware_error(decision.get("selected_gpu_hardware")):
        reject("scale_decision_invalid", "scale decision selected hardware identity is invalid")
    if decision["selected_gpu_hardware"]["count"] != decision["selected_train_gpus"]:
        reject("scale_decision_invalid", "selected hardware count differs from selected_train_gpus")
    for key in ("selected_global_examples_per_second", "selected_global_tokens_per_second"):
        value = decision.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reject("scale_decision_invalid", f"{key} must be a positive measured value")
    if not isinstance(decision.get("training_controls"), dict) or not isinstance(decision.get("results"), list) or len(decision["results"]) != 2:
        reject("scale_decision_invalid", "scale decision controls/results lineage is missing")
    return decision, digest


def cmd_freeze_collection_decision(args: argparse.Namespace) -> None:
    result = json.loads(args.hardware_result.read_text(encoding="utf-8"))
    error = probe_artifact_error(result)
    if error:
        reject("collection_hardware_probe_invalid", f"collection hardware probe is invalid: {error}")
    hardware = result["gpu_hardware"]
    if hardware["count"] != 2:
        reject("collection_gpu_count", "collection device decisions require exactly two measured GPUs")
    if args.teacher_device_index == args.student_device_index:
        reject("collection_device_collision", "teacher and student must use different devices")
    if any(index < 0 or index >= hardware["count"] for index in (args.teacher_device_index, args.student_device_index)):
        reject("collection_device_out_of_range", "collection device index is outside the measured allocation")
    devices = hardware["devices"]
    maximum_free = max(device["free_memory_bytes"] for device in devices)
    teacher = devices[args.teacher_device_index]
    if teacher["free_memory_bytes"] != maximum_free:
        reject("teacher_not_largest_fit_device", "pinned teacher device is not the largest-free-VRAM measured device")
    payload = {
        "schema_version": 1, "status": "frozen", "decision_kind": "collection_devices", "fallback_reason": "none",
        "teacher": {"model": args.teacher_model, "revision": args.teacher_revision, "device_index": args.teacher_device_index,
                    "uuid": teacher["uuid"], "free_memory_bytes": teacher["free_memory_bytes"], "total_memory_bytes": teacher["total_memory_bytes"]},
        "student": {"model": args.student_model, "revision": args.student_revision, "device_index": args.student_device_index,
                    "uuid": devices[args.student_device_index]["uuid"], "free_memory_bytes": devices[args.student_device_index]["free_memory_bytes"],
                    "total_memory_bytes": devices[args.student_device_index]["total_memory_bytes"]},
        "allocated_gpu_hardware": hardware, "hardware_result": str(args.hardware_result.resolve()),
        "hardware_result_sha256": sha256_file(args.hardware_result),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "frozen", "decision_sha256": sha256_file(args.output), "fallback_reason": "none"}, sort_keys=True))


def cmd_validate_collection_decision(args: argparse.Namespace) -> None:
    if sha256_file(args.decision) != args.expected_sha256:
        reject("collection_decision_hash_mismatch", "collection decision hash differs from expected")
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("status") != "frozen" or decision.get("decision_kind") != "collection_devices":
        reject("collection_decision_invalid", "collection decision schema/status/kind is invalid")
    if decision.get("teacher", {}).get("model") != args.teacher_model or decision["teacher"].get("revision") != args.teacher_revision:
        reject("collection_teacher_identity_mismatch", "teacher model identity differs from frozen collection decision")
    if decision.get("student", {}).get("model") != args.student_model or decision["student"].get("revision") != args.student_revision:
        reject("collection_student_identity_mismatch", "student model identity differs from frozen collection decision")
    frozen_hardware = decision.get("allocated_gpu_hardware")
    if gpu_hardware_error(frozen_hardware) or frozen_hardware["count"] != args.allocated_gpus or args.allocated_gpus != 2:
        reject("collection_hardware_identity_mismatch", "collection requires exactly two GPUs matching frozen hardware evidence")
    try:
        current_payload = json.loads(args.current_hardware.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reject("collection_current_hardware_invalid", f"cannot parse current hardware evidence: {exc}")
    current_hardware = current_payload.get("gpu_hardware") if isinstance(current_payload, dict) else None
    if gpu_hardware_error(current_hardware) or current_hardware["count"] != 2:
        reject("collection_current_hardware_invalid", "current collection hardware evidence is not exactly two GPUs")
    for frozen, current in zip(frozen_hardware["devices"], current_hardware["devices"], strict=True):
        for key in ("name", "uuid", "total_memory_bytes", "compute_capability"):
            if frozen[key] != current[key]:
                reject("collection_gpu_uuid_mismatch", f"current GPU {key} differs from frozen collection evidence")
    frozen_teacher = decision["teacher"]
    frozen_student = decision["student"]
    current_by_uuid = {device["uuid"]: device for device in current_hardware["devices"]}
    if frozen_teacher["uuid"] not in current_by_uuid or frozen_student["uuid"] not in current_by_uuid:
        reject("collection_gpu_uuid_mismatch", "current physical GPU UUIDs do not match frozen teacher/student UUIDs")
    current_teacher = current_by_uuid[frozen_teacher["uuid"]]
    current_student = current_by_uuid[frozen_student["uuid"]]
    if current_teacher["free_memory_bytes"] < frozen_teacher["free_memory_bytes"] or current_student["free_memory_bytes"] < frozen_student["free_memory_bytes"]:
        reject("collection_vram_insufficient", "current free VRAM is below frozen teacher/student selection evidence")
    if current_teacher["free_memory_bytes"] != max(device["free_memory_bytes"] for device in current_hardware["devices"]):
        reject("teacher_not_largest_fit_device", "current teacher UUID is not the largest-free-VRAM device")
    teacher_index = current_teacher["index"]
    student_index = current_student["index"]
    if teacher_index == student_index:
        reject("collection_device_contract_invalid", "frozen teacher and student UUIDs resolve to the same device")
    print(f"cuda:{teacher_index}\tcuda:{student_index}\t{args.expected_sha256}\tnone")


def cmd_probe_hardware(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        reject("cuda_unavailable", "hardware probe refuses CPU fallback")
    payload = {"schema_version": 1, "status": "ok", "fallback_reason": "none", "gpu_hardware": gpu_hardware_identity(torch), "package_versions": required_package_versions()}
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def add_identity_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--commit-hash", required=required)
    parser.add_argument("--config-sha256", required=required)
    parser.add_argument("--model", required=required)
    parser.add_argument("--model-revision", required=required)
    parser.add_argument("--dataset-source", required=required)
    parser.add_argument("--dataset-revision", required=required)
    parser.add_argument("--dataset-sha256", required=required)
    parser.add_argument("--prompt-sha256", required=required)
    parser.add_argument("--retrieval-sha256", required=required)
    parser.add_argument("--source-schema-sha256", required=required)
    parser.add_argument("--eval-dataset-sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-decision")
    validate.add_argument("--decision", required=True, type=Path)
    validate.add_argument("--expected-sha256", required=True)
    validate.add_argument("--purpose", choices=("generation", "training"), required=True)
    validate.add_argument("--output-format", choices=("json", "generation-tsv", "training-tsv"), default="json")
    validate.add_argument("--student-thinking-mode", choices=("direct", "two_pass"))
    validate.add_argument("--scratchpad-max-new-tokens", type=int)
    validate.add_argument("--query-temperature", type=float)
    validate.add_argument("--response-temperature", type=float)
    validate.add_argument("--top-p", type=float)
    validate.add_argument("--top-k", type=int)
    validate.add_argument("--k1", type=float)
    validate.add_argument("--b", type=float)
    validate.add_argument("--training-method", choices=("sft", "dpo", "grpo", "dapo"))
    add_identity_args(validate, required=True)
    validate.set_defaults(func=cmd_validate_decision)

    freeze_thinking = subparsers.add_parser("freeze-thinking-selection")
    freeze_thinking.add_argument("--selection", required=True, type=Path)
    freeze_thinking.add_argument("--output", required=True, type=Path)
    freeze_thinking.add_argument("--expected-mode", choices=("direct", "two_pass"), required=True)
    freeze_thinking.add_argument("--row-count", required=True, type=int)
    freeze_thinking.add_argument("--optimization-decision-sha256", required=True)
    add_identity_args(freeze_thinking, required=True)
    freeze_thinking.set_defaults(func=cmd_freeze_thinking_selection)

    validate_thinking = subparsers.add_parser("validate-thinking-selection")
    validate_thinking.add_argument("--manifest", required=True, type=Path)
    validate_thinking.add_argument("--expected-sha256", required=True)
    validate_thinking.add_argument("--expected-mode", choices=("direct", "two_pass"), required=True)
    validate_thinking.add_argument("--optimization-decision-sha256", required=True)
    add_identity_args(validate_thinking, required=True)
    validate_thinking.set_defaults(func=cmd_validate_thinking_selection)

    create_smoke = subparsers.add_parser("create-smoke-manifest")
    create_smoke.add_argument("--initial-checkpoint", required=True, type=Path)
    create_smoke.add_argument("--resumed-checkpoint", required=True, type=Path)
    create_smoke.add_argument("--output", required=True, type=Path)
    create_smoke.set_defaults(func=cmd_create_smoke_manifest)

    checkpoints = subparsers.add_parser("validate-checkpoints")
    checkpoints.add_argument("--smoke-manifest", required=True, type=Path)
    checkpoints.add_argument("--expected-sha256", required=True)
    checkpoints.set_defaults(func=cmd_validate_checkpoints)
    for checkpoint_parser in (create_smoke, checkpoints):
        checkpoint_parser.add_argument("--commit-hash", required=True)
        checkpoint_parser.add_argument("--config-sha256", required=True)
        checkpoint_parser.add_argument("--model", required=True)
        checkpoint_parser.add_argument("--model-revision", required=True)
        checkpoint_parser.add_argument("--dataset-source", required=True)
        checkpoint_parser.add_argument("--dataset-revision", required=True)
        checkpoint_parser.add_argument("--dataset-sha256", required=True)
        checkpoint_parser.add_argument("--eval-dataset-sha256")
        checkpoint_parser.add_argument("--prompt-sha256", required=True)
        checkpoint_parser.add_argument("--retrieval-sha256", required=True)
        checkpoint_parser.add_argument("--source-schema-sha256", required=True)
        checkpoint_parser.add_argument("--optimization-decision-sha256", required=True)
        checkpoint_parser.add_argument("--scale-decision-sha256", required=True)
        checkpoint_parser.add_argument("--method", choices=("sft", "dpo", "grpo", "dapo"), required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--result", required=True, type=Path)
    benchmark.add_argument("--commit-hash", required=True)
    benchmark.add_argument("--probe-name", required=True)
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--data", required=True, type=Path)
    benchmark.add_argument("--eval-data", type=Path)
    benchmark.add_argument("--output-dir", type=Path)
    benchmark.add_argument("--deepspeed-config", type=Path)
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--model-revision", required=True)
    benchmark.add_argument("--dataset-source", required=True)
    benchmark.add_argument("--dataset-revision", required=True)
    benchmark.add_argument("--prompt-sha256", required=True)
    benchmark.add_argument("--retrieval-sha256", required=True)
    benchmark.add_argument("--source-schema-sha256", required=True)
    benchmark.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    benchmark.add_argument("--probe-kind", choices=("generation", "training"), default="generation")
    benchmark.add_argument("--generation-batch-size", type=int, default=4)
    benchmark.add_argument("--max-new-tokens", type=int, default=32)
    benchmark.add_argument("--sample-size", type=int, default=16)
    benchmark.add_argument("--warmup-repeats", type=int, default=2)
    benchmark.add_argument("--measured-repeats", type=int, default=5)
    benchmark.add_argument("--train-microbatch", type=int, default=1)
    benchmark.add_argument("--gradient-accumulation-steps", type=int, default=32)
    benchmark.add_argument("--dataloader-workers", type=int, default=0)
    benchmark.add_argument("--per-device-eval-batch-size", type=int, default=1)
    benchmark.add_argument("--max-steps", type=int, default=4)
    benchmark.add_argument("--max-length", type=int, default=4096)
    benchmark.add_argument("--training-timeout-seconds", type=int, default=3600)
    benchmark.add_argument("--monitor-interval-seconds", type=float, default=0.2)
    benchmark.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--training-method", choices=("sft", "dpo", "grpo", "dapo"), required=True)
    benchmark.add_argument("--ref-log-probs", type=Path)
    benchmark.add_argument("--eval-ref-log-probs", type=Path)
    benchmark.add_argument("--reference-checkpoint-hash")
    benchmark.add_argument("--prompt-context-schema", type=Path)
    benchmark.add_argument("--num-generations", type=int, required=True)
    benchmark.add_argument("--rl-generation-batch-size", type=int, required=True)
    benchmark.add_argument("--max-completion-length", type=int, required=True)
    benchmark.add_argument("--static-cache", action="store_true")
    benchmark.add_argument("--compile", action="store_true")
    benchmark.add_argument("--packing", action="store_true")
    benchmark.add_argument("--padding-free", action="store_true")
    benchmark.add_argument("--use-liger-kernel", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.set_defaults(func=cmd_compare)

    freeze = subparsers.add_parser("freeze-decision")
    freeze.add_argument("--baseline", required=True, type=Path)
    freeze.add_argument("--candidate", action="append", default=[], type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--purpose", choices=("generation", "training"), default="generation")
    freeze.add_argument("--launch-max-steps", required=True, type=int)
    freeze.add_argument("--launch-learning-rate", required=True, type=float)
    freeze.add_argument("--launch-epochs", required=True, type=float)
    freeze.add_argument("--launch-save-steps", required=True, type=int)
    freeze.add_argument("--launch-eval-steps", required=True, type=int)
    freeze.add_argument("--query-max-new-tokens", required=True, type=int)
    freeze.add_argument("--response-max-new-tokens", required=True, type=int)
    freeze.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), required=True)
    freeze.add_argument("--scratchpad-max-new-tokens", required=True, type=int)
    freeze.add_argument("--query-temperature", required=True, type=float)
    freeze.add_argument("--response-temperature", required=True, type=float)
    freeze.add_argument("--top-p", required=True, type=float)
    freeze.add_argument("--top-k", required=True, type=int)
    freeze.add_argument("--k1", required=True, type=float)
    freeze.add_argument("--b", required=True, type=float)
    freeze.set_defaults(func=cmd_freeze_decision)

    freeze_scale = subparsers.add_parser("freeze-scale-decision")
    freeze_scale.add_argument("--result", action="append", required=True, type=Path)
    freeze_scale.add_argument("--output", required=True, type=Path)
    freeze_scale.add_argument("--loss-relative-tolerance", type=float, default=0.01)
    freeze_scale.set_defaults(func=cmd_freeze_scale_decision)

    validate_scale = subparsers.add_parser("validate-scale-decision")
    validate_scale.add_argument("--decision", required=True, type=Path)
    validate_scale.add_argument("--expected-sha256", required=True)
    validate_scale.add_argument("--train-gpus", required=True, type=int)
    validate_scale.add_argument("--training-method", choices=("sft", "dpo", "grpo", "dapo"), required=True)
    add_identity_args(validate_scale, required=True)
    validate_scale.set_defaults(func=cmd_validate_scale_decision)

    freeze_collection = subparsers.add_parser("freeze-collection-decision")
    freeze_collection.add_argument("--hardware-result", required=True, type=Path)
    freeze_collection.add_argument("--output", required=True, type=Path)
    freeze_collection.add_argument("--teacher-model", required=True)
    freeze_collection.add_argument("--teacher-revision", required=True)
    freeze_collection.add_argument("--student-model", required=True)
    freeze_collection.add_argument("--student-revision", required=True)
    freeze_collection.add_argument("--teacher-device-index", required=True, type=int)
    freeze_collection.add_argument("--student-device-index", required=True, type=int)
    freeze_collection.set_defaults(func=cmd_freeze_collection_decision)

    validate_collection = subparsers.add_parser("validate-collection-decision")
    validate_collection.add_argument("--decision", required=True, type=Path)
    validate_collection.add_argument("--expected-sha256", required=True)
    validate_collection.add_argument("--teacher-model", required=True)
    validate_collection.add_argument("--teacher-revision", required=True)
    validate_collection.add_argument("--student-model", required=True)
    validate_collection.add_argument("--student-revision", required=True)
    validate_collection.add_argument("--allocated-gpus", required=True, type=int)
    validate_collection.add_argument("--current-hardware", required=True, type=Path)
    validate_collection.set_defaults(func=cmd_validate_collection_decision)

    hardware = subparsers.add_parser("probe-hardware")
    hardware.add_argument("--output", required=True, type=Path)
    hardware.set_defaults(func=cmd_probe_hardware)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except GateError as exc:
        print(json.dumps({"status": "rejected", "fallback_reason": exc.fallback_reason, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
