#!/usr/bin/env python3
"""Measured Turing optimization probes and repository-owned launch gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


DECISION_SCHEMA_VERSION = 1
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "deepspeed",
    "bitsandbytes",
    "flash-attn",
    "liger-kernel",
)


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
        required = ("microbatch", "gradient_accumulation_steps", "dataloader_workers")
        if any(not isinstance(training.get(key), int) or training[key] < 0 for key in required):
            reject("decision_training_invalid", "selected training integer controls are invalid")
        if training.get("packing") or training.get("padding_free") or training.get("use_liger_kernel"):
            reject(
                "task7_trainer_optimization_unsupported",
                "Task 7 trainer configuration does not yet expose the selected packing, padding-free, or Liger controls",
            )
        if selected["attention_implementation"] != "sdpa":
            reject(
                "task7_attention_config_integration_missing",
                "train CLI reads attention from Task 7 config and cannot yet consume a frozen non-SDPA decision",
            )
        payload = {key: training[key] for key in required}
        payload.update({key: bool(training.get(key)) for key in ("packing", "padding_free", "use_liger_kernel")})
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
            "dataloader_workers", "fallback_reason", "decision_sha256",
        )))
    else:
        print(json.dumps(payload, sort_keys=True))


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


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    step, state_path = checkpoint_step(path)
    files = [item for item in path.rglob("*") if item.is_file()]
    if not any(item.name in {"model.safetensors", "pytorch_model.bin"} or item.name.endswith("model_states.pt") for item in files):
        reject("checkpoint_model_state_missing", f"checkpoint has no model state: {path}")
    if not any(item.name == "optimizer.pt" or item.name.endswith("optim_states.pt") for item in files):
        reject("checkpoint_optimizer_state_missing", f"checkpoint has no optimizer state: {path}")
    if not (path / "scheduler.pt").is_file():
        reject("checkpoint_scheduler_state_missing", f"checkpoint has no scheduler.pt: {path}")
    if not any(item.name == "rng_state.pth" or re.fullmatch(r"rng_state_\d+\.pth", item.name) for item in files):
        reject("checkpoint_rng_state_missing", f"checkpoint has no RNG state: {path}")
    return {
        "path": str(path.resolve()),
        "step": step,
        "tree_sha256": sha256_tree(path),
        "trainer_state_sha256": sha256_file(state_path),
    }


def smoke_identities(args: argparse.Namespace) -> dict[str, str]:
    return {
        "commit_hash": args.commit_hash,
        "config_sha256": args.config_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_sha256": args.dataset_sha256,
        "prompt_sha256": args.prompt_sha256,
        "retrieval_sha256": args.retrieval_sha256,
        "source_schema_sha256": args.source_schema_sha256,
        "optimization_decision_sha256": args.optimization_decision_sha256,
        "method": args.method,
    }


def cmd_create_smoke_manifest(args: argparse.Namespace) -> None:
    initial_step, _ = checkpoint_step(args.initial_checkpoint)
    resumed_step, _ = checkpoint_step(args.resumed_checkpoint)
    if resumed_step <= initial_step:
        reject("resume_step_not_advanced", f"resumed step {resumed_step} must be greater than saved step {initial_step}")
    initial = inspect_checkpoint(args.initial_checkpoint)
    resumed = inspect_checkpoint(args.resumed_checkpoint)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "fallback_reason": "none",
        "identities": smoke_identities(args),
        "initial_checkpoint": initial,
        "resumed_checkpoint": resumed,
        "resume_from_checkpoint": initial["path"],
        "step_delta": resumed_step - initial_step,
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
    initial = inspect_checkpoint(Path(manifest["initial_checkpoint"]["path"]))
    resumed = inspect_checkpoint(Path(manifest["resumed_checkpoint"]["path"]))
    if initial != manifest["initial_checkpoint"] or resumed != manifest["resumed_checkpoint"]:
        reject("smoke_checkpoint_hash_mismatch", "checkpoint contents changed after smoke manifest creation")
    if manifest.get("resume_from_checkpoint") != initial["path"] or resumed["step"] <= initial["step"]:
        reject("smoke_lineage_invalid", "smoke resume lineage or step continuity is invalid")
    print(json.dumps({"status": "validated", "smoke_manifest_sha256": actual_hash, "fallback_reason": "none"}, sort_keys=True))


class TelemetrySampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
                for line in output.splitlines():
                    values = [float(item.strip()) for item in line.split(",")]
                    self.samples.append(dict(zip(("utilization_percent", "memory_used_mb", "power_watts", "temperature_c"), values, strict=True)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.2)

    def __enter__(self) -> TelemetrySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            reject("gpu_telemetry_unavailable", "nvidia-smi produced no utilization telemetry during the measured probe")
        return {
            "sample_count": len(self.samples),
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
    base_result = {
        "schema_version": 1,
        "probe_name": args.probe_name,
        "status": "rejected",
        "fallback_reason": "probe_not_completed",
        "package_versions": package_versions(),
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
        if args.probe_kind == "training":
            reject("task7_training_probe_support_missing", "Task 7 has not exposed a bounded trainer benchmark entry point")
        if args.use_liger_kernel:
            reject("use_liger_kernel_incompatible_with_precompute_ref_log_probs", "primary DPO requires precompute_ref_log_probs=True")
        if args.train_microbatch != 1 or args.gradient_accumulation_steps != 32 or args.dataloader_workers != 0:
            reject("task7_training_probe_support_missing", "Task 7 has not exposed bounded measured trainer probes")
        if args.packing or args.padding_free:
            if args.attention_implementation != "flash_attention_2":
                reject("flash_attention_2_required", "padding-free and packing probes require FlashAttention2 compatibility first")
            reject("task7_packing_probe_support_missing", "Task 7 has not exposed padding-free or packing trainer controls")
        if args.attention_implementation == "flash_attention_2" and base_result["package_versions"]["flash-attn"] == "missing":
            reject("flash_attention_2_import_unavailable", "flash-attn is not installed")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            reject("cuda_unavailable", "measured Turing probes refuse CPU fallback")
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
        with TelemetrySampler() as telemetry:
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


def accepted_candidate(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    if candidate.get("status") != "ok":
        return False, str(candidate.get("fallback_reason") or "candidate_not_ok")
    if candidate.get("output_hash") != baseline.get("output_hash"):
        return False, "output_hash_mismatch"
    if candidate.get("decoded_output_hash") != baseline.get("decoded_output_hash"):
        return False, "decoded_output_hash_mismatch"
    if float(candidate.get("tokens_per_second", 0)) <= float(baseline.get("tokens_per_second", 0)):
        return False, "throughput_not_improved"
    return True, "none"


def deployable_candidate(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    accepted, reason = accepted_candidate(baseline, candidate)
    if not accepted:
        return accepted, reason
    config = candidate.get("config", {})
    if config.get("static_cache"):
        return False, "launch_unsupported_static_cache"
    if config.get("compile"):
        return False, "launch_unsupported_compile"
    if config.get("attention_implementation") != "sdpa":
        return False, "launch_unsupported_training_attention"
    if config.get("probe_kind") != "generation":
        return False, "launch_unsupported_training_probe"
    if config.get("train_microbatch") != 1 or config.get("dataloader_workers") != 0:
        return False, "launch_unsupported_training_controls"
    if any(config.get(key) for key in ("packing", "padding_free", "use_liger_kernel")):
        return False, "launch_unsupported_task7_controls"
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
    if baseline.get("status") != "ok":
        reject("baseline_probe_invalid", "cannot freeze a decision without a successful SDPA baseline")
    accepted: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        ok, reason = deployable_candidate(baseline, candidate)
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
                "packing": config["packing"],
                "padding_free": config["padding_free"],
                "use_liger_kernel": config["use_liger_kernel"],
            },
        },
        "identities": baseline["identities"],
        "package_versions": baseline["package_versions"],
        "baseline_result": str(args.baseline.resolve()),
        "baseline_result_sha256": sha256_file(args.baseline),
        "selected_result": str(selected_path.resolve()),
        "selected_result_sha256": sha256_file(selected_path),
        "accepted_candidates": [str(path.resolve()) for path, _ in accepted],
        "rejected_candidates": rejected,
    }
    write_json(args.output, decision)
    print(json.dumps({"status": "frozen", "decision": str(args.output), "decision_sha256": sha256_file(args.output), "fallback_reason": decision["fallback_reason"]}, sort_keys=True))


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
    add_identity_args(validate, required=True)
    validate.set_defaults(func=cmd_validate_decision)

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
        checkpoint_parser.add_argument("--dataset-sha256", required=True)
        checkpoint_parser.add_argument("--prompt-sha256", required=True)
        checkpoint_parser.add_argument("--retrieval-sha256", required=True)
        checkpoint_parser.add_argument("--source-schema-sha256", required=True)
        checkpoint_parser.add_argument("--optimization-decision-sha256", required=True)
        checkpoint_parser.add_argument("--method", choices=("sft", "dpo", "grpo", "dapo"), required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--result", required=True, type=Path)
    benchmark.add_argument("--commit-hash", required=True)
    benchmark.add_argument("--probe-name", required=True)
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--data", required=True, type=Path)
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
