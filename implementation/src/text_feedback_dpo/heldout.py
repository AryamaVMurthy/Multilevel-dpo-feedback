from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping

from text_feedback_dpo.evaluation import evaluate_native_examples
from text_feedback_dpo.io import read_jsonl, write_json_atomic, write_jsonl
from text_feedback_dpo.models import ModelGeneration, generate_model_result, normalize_model_generation
from text_feedback_dpo.prompts import build_native_student_prompt


def _require_hex(value: str, *, length: int, field: str) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase {length}-character hexadecimal value")
    return value


def build_baseline_evaluation_freeze(
    *,
    experiment_id: str,
    source_commit: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    student_model: Mapping[str, str],
    evaluator_model: Mapping[str, str],
    prompt_protocol: str,
    student_generation: Mapping[str, Any],
    evaluator_generation: Mapping[str, Any],
    generation_seed: int,
) -> dict[str, Any]:
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    if not prompt_protocol.strip():
        raise ValueError("prompt_protocol must be non-empty")
    if isinstance(generation_seed, bool) or not isinstance(generation_seed, int) or generation_seed <= 0:
        raise ValueError("generation_seed must be a positive integer")
    checkpoint = dict(student_model)
    evaluator = dict(evaluator_model)
    for role, model in (("checkpoint", checkpoint), ("evaluator", evaluator)):
        if not isinstance(model.get("id"), str) or not model["id"].strip():
            raise ValueError(f"{role}.id must be non-empty")
        revision = model.get("revision")
        if not isinstance(revision, str):
            raise ValueError(f"{role}.revision must be a commit SHA")
        _require_hex(revision, length=40, field=f"{role}.revision")
    return {
        "schema": "baseline-evaluation-freeze-v1",
        "experiment_id": experiment_id,
        "source_commit": _require_hex(source_commit, length=40, field="source_commit"),
        "config_sha256": _require_hex(config_sha256, length=64, field="config_sha256"),
        "dataset_manifest_sha256": _require_hex(
            dataset_manifest_sha256,
            length=64,
            field="dataset_manifest_sha256",
        ),
        "checkpoint": checkpoint,
        "evaluator": evaluator,
        "prompt_protocol": prompt_protocol,
        "student_generation": dict(student_generation),
        "evaluator_generation": dict(evaluator_generation),
        "generation_seed": generation_seed,
        "splits": ["validation", "test"],
        "teacher_enabled": False,
    }


def validate_adapter_compatibility(
    adapter_dir: Path,
    *,
    base_model_revision: str,
    lora_coverage_hash: str,
) -> dict[str, Any]:
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"adapter manifest does not exist: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adapter manifest must be a JSON object")
    for field, expected in (
        ("base_model_revision", base_model_revision),
        ("lora_coverage_hash", lora_coverage_hash),
    ):
        if value.get(field) != expected:
            raise ValueError(f"adapter {field} is incompatible with the requested checkpoint")
    return value


def _teacher_free_prompt(example: Mapping[str, Any]) -> str:
    prompt = build_native_student_prompt(
        problem=str(example["problem"]),
        domain=str(example["domain"]),
        evidence=example.get("evidence"),
    )
    lowered = prompt.casefold()
    forbidden = ("gold answer", "teacher guidance", "teacher-only", "privileged training")
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"teacher-free checkpoint prompt contains forbidden context: {found[0]}")
    return prompt


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evaluation seed must be an integer")
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _example_generation_seed(seed: int, example_id: str) -> int:
    payload = f"{seed}:{example_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _attach_generation_metrics(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    exact_prompt_tokens = [row["prompt_tokens"] for row in predictions if row["prompt_tokens"] is not None]
    truncations = [bool(row["truncated"]) for row in predictions if row["truncated"] is not None]
    finish_reasons: dict[str, int] = {}
    for row in predictions:
        reason = str(row["finish_reason"])
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    metrics["common"].update(
        {
            "average_prompt_tokens": (
                sum(exact_prompt_tokens) / len(exact_prompt_tokens) if exact_prompt_tokens else None
            ),
            "truncation_rate": sum(truncations) / len(truncations) if truncations else None,
            "finish_reason_counts": finish_reasons,
            "average_generation_latency_ms": sum(
                float(row["generation_latency_ms"]) for row in predictions
            )
            / len(predictions),
            "average_evaluation_latency_ms": sum(
                float(row["evaluation_latency_ms"]) for row in predictions
            )
            / len(predictions),
        }
    )


def _freeze_sha256(freeze_manifest: Path | None) -> str | None:
    if freeze_manifest is None:
        return None
    if not freeze_manifest.exists():
        raise FileNotFoundError(f"evaluation freeze manifest does not exist: {freeze_manifest}")
    return hashlib.sha256(freeze_manifest.read_bytes()).hexdigest()


def _validate_prediction_record(
    row: Mapping[str, Any],
    *,
    checkpoint_kind: str,
    base_model_revision: str,
) -> None:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("evaluation prediction is missing a non-empty id")
    if row.get("teacher_free") is not True:
        raise ValueError(f"evaluation prediction {row_id} is not teacher-free")
    if row.get("checkpoint_kind") != checkpoint_kind:
        raise ValueError(f"evaluation prediction {row_id} checkpoint kind mismatch")
    if row.get("base_model_revision") != base_model_revision:
        raise ValueError(f"evaluation prediction {row_id} base-model revision mismatch")
    if not isinstance(row.get("response"), str) or not str(row["response"]).strip():
        raise ValueError(f"evaluation prediction {row_id} has no response")
    if not isinstance(row.get("prompt_tokens"), int) or not isinstance(row.get("generated_tokens"), int):
        raise ValueError(f"evaluation prediction {row_id} lacks exact token metadata")
    if not isinstance(row.get("terminated"), bool) or not isinstance(row.get("truncated"), bool):
        raise ValueError(f"evaluation prediction {row_id} lacks termination metadata")
    if row.get("finish_reason") not in {"eos", "final_answer", "length", "other"}:
        raise ValueError(f"evaluation prediction {row_id} has an invalid finish reason")
    if not isinstance(row.get("generation_seed"), int):
        raise ValueError(f"evaluation prediction {row_id} lacks its per-example generation seed")
    if not isinstance(row.get("evaluator_result"), dict):
        raise ValueError(f"evaluation prediction {row_id} has no evaluator result")


def evaluate_checkpoint(
    *,
    examples: list[dict[str, Any]],
    generate: Callable[[str], str | ModelGeneration],
    evaluator: Callable[[dict[str, Any], str], dict[str, Any]],
    output_dir: Path,
    checkpoint_kind: str,
    base_model_revision: str,
    seed: int,
    test: bool,
    freeze_manifest: Path | None = None,
    adapter_manifest: Mapping[str, Any] | None = None,
    require_generation_metadata: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict[str, Any]:
    if checkpoint_kind not in {"base", "adapter"}:
        raise ValueError("checkpoint_kind must be base or adapter")
    if not examples:
        raise ValueError("checkpoint evaluation examples must not be empty")
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be provided together")
    if shard_index is not None and (num_shards is None or num_shards <= 0 or not 0 <= shard_index < num_shards):
        raise ValueError("evaluation shard_index must be in [0, num_shards)")
    if test:
        if freeze_manifest is None or not freeze_manifest.exists():
            raise FileNotFoundError("test evaluation requires an existing freeze manifest")
        freeze_value = json.loads(freeze_manifest.read_text(encoding="utf-8"))
        expected_schema = (
            "baseline-evaluation-freeze-v1" if checkpoint_kind == "base" else "hyperparameter-freeze-v1"
        )
        if not isinstance(freeze_value, dict) or freeze_value.get("schema") != expected_schema:
            raise ValueError(f"freeze manifest must use schema {expected_schema}")
    marker = output_dir / "test_complete.json"
    if test and marker.exists():
        raise FileExistsError(f"refusing to repeat test evaluation; test marker exists: {marker}")
    if adapter_manifest is not None and checkpoint_kind != "adapter":
        raise ValueError("adapter_manifest is only valid for adapter checkpoints")
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_marker = output_dir / "evaluation_complete.json"
    if evaluation_marker.exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation: {evaluation_marker}")
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint predictions: {predictions_path}")
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failures_path = output_dir / "failures.jsonl"

    def persist_failure(example_id: str, stage: str, exc: Exception) -> None:
        failures.append(
            {
                "id": example_id,
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        write_jsonl(failures_path, failures)

    result_by_id: dict[str, dict[str, Any]] = {}
    response_by_id: dict[str, str] = {}
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("checkpoint example is missing id")
        if example_id in response_by_id:
            raise ValueError(f"checkpoint examples contain duplicate id: {example_id}")
        example_seed = _example_generation_seed(seed, example_id)
        _seed_everything(example_seed)
        try:
            prompt = _teacher_free_prompt(example)
        except Exception as exc:
            persist_failure(example_id, "prompt_validation", exc)
            raise
        generation_started = time.monotonic()
        try:
            generation = normalize_model_generation(generate(prompt))
        except Exception as exc:
            persist_failure(example_id, "generation", exc)
            raise
        generation_latency_ms = (time.monotonic() - generation_started) * 1000
        response = generation.text
        try:
            if not response.strip():
                raise ValueError(f"checkpoint generator returned empty response for {example_id}")
            if require_generation_metadata and (
                generation.prompt_tokens is None
                or generation.generated_tokens is None
                or generation.terminated is None
                or generation.truncated is None
                or generation.finish_reason == "unavailable"
            ):
                raise ValueError(f"checkpoint generation metadata is unavailable for {example_id}")
        except Exception as exc:
            persist_failure(example_id, "generation_validation", exc)
            raise
        evaluation_started = time.monotonic()
        try:
            result = evaluator(example, response)
        except Exception as exc:
            persist_failure(example_id, "evaluation", exc)
            raise
        evaluation_latency_ms = (time.monotonic() - evaluation_started) * 1000
        if not isinstance(result, dict):
            exc = ValueError(f"checkpoint evaluator returned non-object result for {example_id}")
            persist_failure(example_id, "evaluation_validation", exc)
            raise exc
        if generation.truncated:
            result = {
                **result,
                "correct": False,
                "deterministic": {**dict(result.get("deterministic", {})), "correct": False},
                "student_truncation_override": True,
            }
        response_by_id[example_id] = response
        result_by_id[example_id] = result
        predictions.append(
            {
                "id": example_id,
                "response": response,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "teacher_free": True,
                "checkpoint_kind": checkpoint_kind,
                "base_model_revision": base_model_revision,
                "generation_seed": example_seed,
                "evaluator_result": result,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "terminated": generation.terminated,
                "truncated": generation.truncated,
                "finish_reason": generation.finish_reason,
                "generation_latency_ms": generation_latency_ms,
                "evaluation_latency_ms": evaluation_latency_ms,
            }
        )
    write_jsonl(failures_path, failures)
    write_jsonl(predictions_path, predictions)

    def replay_evaluator(example: dict[str, Any], response: str) -> dict[str, Any]:
        if response != response_by_id[example["id"]]:
            raise ValueError(f"checkpoint response changed during summary for {example['id']}")
        return result_by_id[example["id"]]

    metrics = evaluate_native_examples(
        examples,
        [
            {
                "id": row["id"],
                "response": row["response"],
                **(
                    {"generated_tokens": row["generated_tokens"]}
                    if row["generated_tokens"] is not None
                    else {}
                ),
            }
            for row in predictions
        ],
        evaluator=replay_evaluator,
    )
    _attach_generation_metrics(metrics, predictions)
    summary = {
        "checkpoint_kind": checkpoint_kind,
        "base_model_revision": base_model_revision,
        "seed": seed,
        "test": test,
        "examples": len(examples),
        "metrics": metrics,
        "adapter_manifest": dict(adapter_manifest) if adapter_manifest is not None else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json_atomic(
        evaluation_marker,
        {
            "schema": "checkpoint-evaluation-shard-v1",
            "checkpoint_kind": checkpoint_kind,
            "base_model_revision": base_model_revision,
            "seed": seed,
            "test": test,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "freeze_sha256": _freeze_sha256(freeze_manifest),
            "examples": len(examples),
            "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            "failures_sha256": hashlib.sha256(failures_path.read_bytes()).hexdigest(),
        },
    )
    if test:
        write_json_atomic(
            marker,
            {
                "schema": "frozen-test-complete-v1",
                "freeze_manifest": str(freeze_manifest),
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "examples": len(examples),
            },
        )
    return metrics


def merge_checkpoint_evaluations(
    *,
    examples: list[dict[str, Any]],
    shard_root: Path,
    expected_shards: int,
    output_dir: Path,
    checkpoint_kind: str,
    base_model_revision: str,
    seed: int,
    test: bool,
    freeze_manifest: Path | None,
) -> dict[str, Any]:
    if expected_shards <= 0:
        raise ValueError("expected_shards must be positive")
    if checkpoint_kind not in {"base", "adapter"}:
        raise ValueError("checkpoint_kind must be base or adapter")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged evaluation: {output_dir}")
    expected_freeze_hash = _freeze_sha256(freeze_manifest)
    predictions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for shard_index in range(expected_shards):
        shard_dir = shard_root / f"shard-{shard_index:04d}"
        marker_path = shard_dir / "evaluation_complete.json"
        if not marker_path.exists():
            raise FileNotFoundError(f"evaluation completion marker does not exist: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_marker = {
            "schema": "checkpoint-evaluation-shard-v1",
            "checkpoint_kind": checkpoint_kind,
            "base_model_revision": base_model_revision,
            "seed": seed,
            "test": test,
            "shard_index": shard_index,
            "num_shards": expected_shards,
            "freeze_sha256": expected_freeze_hash,
        }
        mismatches = [key for key, value in expected_marker.items() if marker.get(key) != value]
        if mismatches:
            raise ValueError(
                f"evaluation shard {shard_index} context mismatch in {', '.join(mismatches)}"
            )
        predictions_path = shard_dir / "predictions.jsonl"
        if hashlib.sha256(predictions_path.read_bytes()).hexdigest() != marker.get("predictions_sha256"):
            raise ValueError(f"evaluation shard {shard_index} prediction hash mismatch")
        shard_predictions = read_jsonl(predictions_path)
        if len(shard_predictions) != marker.get("examples"):
            raise ValueError(f"evaluation shard {shard_index} prediction count mismatch")
        failures_path = shard_dir / "failures.jsonl"
        if not failures_path.exists():
            raise FileNotFoundError(f"evaluation shard {shard_index} failure ledger is missing")
        if hashlib.sha256(failures_path.read_bytes()).hexdigest() != marker.get("failures_sha256"):
            raise ValueError(f"evaluation shard {shard_index} failure-ledger hash mismatch")
        if read_jsonl(failures_path):
            raise ValueError(f"evaluation shard {shard_index} completed with recorded failures")
        for row in shard_predictions:
            _validate_prediction_record(
                row,
                checkpoint_kind=checkpoint_kind,
                base_model_revision=base_model_revision,
            )
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"evaluation shard {shard_index} contains a missing id")
            if row_id in seen_ids:
                raise ValueError(f"merged evaluation contains duplicate id: {row_id}")
            seen_ids.add(row_id)
            predictions.append(row)
    expected_ids = [row.get("id") for row in examples]
    actual_ids = [row["id"] for row in predictions]
    if actual_ids != expected_ids:
        raise ValueError("merged evaluation does not exactly cover the dataset in canonical order")
    result_by_id = {row["id"]: row["evaluator_result"] for row in predictions}

    def replay_evaluator(example: dict[str, Any], _response: str) -> dict[str, Any]:
        return result_by_id[str(example["id"])]

    metrics = evaluate_native_examples(
        examples,
        [
            {"id": row["id"], "response": row["response"], "generated_tokens": row["generated_tokens"]}
            for row in predictions
        ],
        evaluator=replay_evaluator,
    )
    _attach_generation_metrics(metrics, predictions)
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    summary = {
        "checkpoint_kind": checkpoint_kind,
        "base_model_revision": base_model_revision,
        "seed": seed,
        "test": test,
        "examples": len(examples),
        "expected_shards": expected_shards,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json_atomic(
        output_dir / "evaluation_complete.json",
        {
            "schema": "checkpoint-evaluation-merged-v1",
            "checkpoint_kind": checkpoint_kind,
            "base_model_revision": base_model_revision,
            "seed": seed,
            "test": test,
            "freeze_sha256": expected_freeze_hash,
            "examples": len(examples),
            "expected_shards": expected_shards,
            "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
    )
    if test:
        write_json_atomic(
            output_dir / "test_complete.json",
            {
                "schema": "frozen-test-complete-v1",
                "freeze_manifest": str(freeze_manifest),
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "examples": len(examples),
            },
        )
    return metrics


def build_transformers_checkpoint_generator(
    *,
    model_id: str,
    revision: str,
    generation_kwargs: Mapping[str, Any],
    adapter_dir: Path | None = None,
    adapter_base_revision: str | None = None,
    adapter_lora_coverage_hash: str | None = None,
) -> Callable[[str], ModelGeneration]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required for checkpoint evaluation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation; refusing CPU fallback")
    if adapter_dir is not None:
        if adapter_base_revision is None or adapter_lora_coverage_hash is None:
            raise ValueError("adapter loading requires base revision and LoRA coverage hash")
        validate_adapter_compatibility(
            adapter_dir,
            base_model_revision=adapter_base_revision,
            lora_coverage_hash=adapter_lora_coverage_hash,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load an adapter checkpoint") from exc
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

    def generate(prompt: str) -> ModelGeneration:
        return generate_model_result(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            generation_kwargs=dict(generation_kwargs),
        )

    return generate
