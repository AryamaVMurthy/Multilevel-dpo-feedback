from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from text_feedback_dpo.evaluators import make_model_evaluator, make_model_guidance_guard
from text_feedback_dpo.io import append_jsonl_zst, read_jsonl_zst, write_json_atomic
from text_feedback_dpo.methods import build_native_iterative_guidance_pairs
from text_feedback_dpo.models import ModelProvider, TransformersModelProvider
from text_feedback_dpo.observability import JsonlLogger
from text_feedback_dpo.prompts import build_native_student_prompt, build_privileged_guidance_prompt
from text_feedback_dpo.sharding import (
    complete_shard,
    merge_completed_shards,
    next_local_index,
    shard_rows,
    write_progress,
)


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _manifest_hash(dataset_dir: Path) -> str:
    manifest = dataset_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    content_hash = value.get("content_sha256")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError("dataset manifest is missing content_sha256")
    return content_hash


def paper_generation_kwargs(config: Any, *, role: str) -> dict[str, Any]:
    if role not in {"student", "teacher", "evaluator"}:
        raise ValueError(f"unsupported paper generation role: {role}")
    return {
        "max_new_tokens": config.generation.max_completion_tokens,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "presence_penalty": config.generation.presence_penalty,
        # Student thinking remains model-native; structured roles use concise outputs.
        "enable_thinking": role == "student",
    }


def collect_paper_shard(
    *,
    config: Any,
    config_path: Path,
    examples: list[dict[str, Any]],
    dataset_dir: Path,
    output_root: Path,
    split: str,
    shard_index: int,
    num_shards: int,
    model_provider: ModelProvider | None = None,
    evaluator: Any | None = None,
    guidance_guard: Any | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("collection examples must not be empty")
    if split not in {"train", "validation", "test", "hparam_train", "hparam_validation"}:
        raise ValueError(f"unsupported collection split: {split}")
    config_digest = _config_hash(config_path)
    manifest_digest = _manifest_hash(dataset_dir)
    selected = shard_rows(examples, shard_index=shard_index, num_shards=num_shards)
    shard_dir = output_root / f"shard-{shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records_path = shard_dir / "records.jsonl.zst"
    progress_path = shard_dir / "progress.json"
    start_index = next_local_index(
        progress_path,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    existing = read_jsonl_zst(records_path) if records_path.exists() else []
    if len(existing) != start_index:
        raise ValueError(
            f"resume state mismatch for shard {shard_index}: records={len(existing)} next_local_index={start_index}"
        )

    provider = model_provider or TransformersModelProvider(
        model_ids={role: config.models[role]["id"] for role in ("student", "teacher", "evaluator")}
    )
    evaluate = evaluator or make_model_evaluator(
        generate=provider.generate,
        generation_kwargs=paper_generation_kwargs(config, role="evaluator"),
    )
    guard = guidance_guard or make_model_guidance_guard(
        generate=provider.generate,
        generation_kwargs=paper_generation_kwargs(config, role="evaluator"),
    )
    logger = JsonlLogger(shard_dir / "events.jsonl", run_id=f"{config.experiment_id}:{split}:{shard_index}")
    logger.event(
        "shard_start",
        stage="collection",
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
        start_index=start_index,
        expected_records=len(selected),
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
    )

    for local_index in range(start_index, len(selected)):
        example = selected[local_index]
        generation_events: list[dict[str, Any]] = []
        guidance_events: list[dict[str, Any]] = []

        def student_generate(prompt: str) -> str:
            start = time.monotonic_ns()
            result = provider.generate("student", prompt, **paper_generation_kwargs(config, role="student"))
            generation_events.append({
                "role": "student",
                "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                "generated_tokens_estimate": len(result.split()),
            })
            return result

        def teacher_guidance(example_row: dict[str, Any], rollout: str, result: dict[str, Any], attempt: int) -> str:
            prompt = build_privileged_guidance_prompt(
                problem=str(example_row["problem"]),
                gold_answer=str(example_row["gold_answer"]),
                rollout=rollout,
                result=result,
                domain=str(example_row["domain"]),
            )
            start = time.monotonic_ns()
            guidance = provider.generate("teacher", prompt, **paper_generation_kwargs(config, role="teacher"))
            generation_events.append({
                "role": "teacher",
                "attempt": attempt,
                "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                "generated_tokens_estimate": len(guidance.split()),
            })
            guidance_events.append({
                "attempt": attempt,
                "raw_teacher_output": guidance,
                "teacher_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "teacher_model": config.models["teacher"]["id"],
            })
            return guidance

        def base_prompt_builder(example_row: dict[str, Any]) -> str:
            return build_native_student_prompt(
                problem=str(example_row["problem"]),
                domain=str(example_row["domain"]),
                evidence=example_row.get("evidence"),
            )

        def retry_prompt_builder(base_prompt: str, guidance: str) -> str:
            return f"{base_prompt}\n\nTeacher guidance for reconsideration:\n{guidance}\nSolve again."

        result = build_native_iterative_guidance_pairs(
            examples=[example],
            base_prompt_builder=base_prompt_builder,
            retry_prompt_builder=retry_prompt_builder,
            student_generate=student_generate,
            evaluate=evaluate,
            teacher_guidance=teacher_guidance,
            guidance_guard=guard,
            max_guidance_steps=int(config.collection["max_guidance_steps"]),
            max_guidance_regenerations=int(config.collection["max_guidance_regenerations"]),
        )
        record = {
            "id": str(example["id"]),
            "split": split,
            "local_index": local_index,
            "source_key": example.get("source_key"),
            "attempts": [
                {key: value for key, value in attempt.items() if key != "prompt"}
                for attempt in result["attempts"]
            ],
            "pairs": result["pairs"],
            "response_sft": result["response_sft"],
            "failures": result["failures"],
            "guidance": guidance_events,
            "generation_events": generation_events,
            "metrics": result["metrics"],
        }
        append_jsonl_zst(records_path, record)
        write_progress(
            progress_path,
            config_hash=config_digest,
            dataset_manifest_hash=manifest_digest,
            shard_index=shard_index,
            num_shards=num_shards,
            last_completed_local_index=local_index,
            records_written=local_index + 1,
        )
        logger.event(
            "example_complete",
            stage="collection",
            example_id=example["id"],
            local_index=local_index,
            attempts=len(record["attempts"]),
            pairs=len(record["pairs"]),
            unresolved=bool(record["failures"]),
        )

    complete_shard(
        shard_dir,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        shard_index=shard_index,
        num_shards=num_shards,
        expected_records=len(selected),
    )
    logger.event("shard_complete", stage="collection", records=len(selected))
    return {"shard_index": shard_index, "records": len(selected), "output_dir": str(shard_dir)}


def merge_paper_collection(
    *,
    config_path: Path,
    dataset_dir: Path,
    collection_dir: Path,
    expected_shards: int,
    output_path: Path,
) -> dict[str, Any]:
    config_digest = _config_hash(config_path)
    manifest_digest = _manifest_hash(dataset_dir)
    records = merge_completed_shards(
        collection_dir,
        expected_shards=expected_shards,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite merged collection: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        append_jsonl_zst(output_path, record)
    write_json_atomic(
        output_path.with_suffix(".manifest.json"),
        {
            "schema": "paper-collection-v1",
            "config_hash": config_digest,
            "dataset_manifest_hash": manifest_digest,
            "expected_shards": expected_shards,
            "records": len(records),
        },
    )
    return {"records": len(records), "output_path": str(output_path)}
