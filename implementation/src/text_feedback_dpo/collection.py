from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from text_feedback_dpo.evaluators import (
    ModelOutputParseError,
    make_model_evaluator,
    make_model_guidance_critic,
    make_model_guidance_guard,
)
from text_feedback_dpo.experiment_config import load_paper_experiment
from text_feedback_dpo.io import append_jsonl_zst, read_jsonl, read_jsonl_zst, write_json_atomic, write_jsonl
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


def _build_protocol_manifest(
    *,
    config: Any,
    config_hash: str,
    dataset_manifest_hash: str,
    source_commit: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be an immutable 40-character lowercase Git SHA")
    payload = {
        "schema": "paper-collection-protocol-v2",
        "source_commit": source_commit,
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "artifact_schema": config.collection["artifact_schema"],
        "prompt_protocol": config.collection["prompt_protocol"],
        "models": config.models,
        "role_generation": {
            role: asdict(profile) for role, profile in sorted(config.generation.roles.items())
        },
    }
    protocol_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "protocol_hash": protocol_hash}


def _ensure_protocol(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "collection protocol mismatch; use a new output directory for a different source commit or policy"
            )
        return
    write_json_atomic(path, expected)


def paper_generation_kwargs(config: Any, *, role: str) -> dict[str, Any]:
    if role not in config.generation.roles:
        raise ValueError(f"unsupported paper generation role: {role}")
    profile = config.generation.roles[role]
    kwargs: dict[str, Any] = {
        "enable_thinking": profile.enable_thinking,
        "do_sample": profile.do_sample,
        "max_new_tokens": profile.max_new_tokens,
    }
    if profile.do_sample:
        kwargs.update(
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            presence_penalty=profile.presence_penalty,
        )
    if profile.stop_after_final_answer:
        kwargs["stop_after_final_answer"] = True
    return kwargs


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
    source_commit: str,
    model_provider: ModelProvider | None = None,
    evaluator: Any | None = None,
    guidance_guard: Any | None = None,
    guidance_critic: Any | None = None,
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
    protocol = _build_protocol_manifest(
        config=config,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        source_commit=source_commit,
    )
    _ensure_protocol(shard_dir / "protocol.json", protocol)
    protocol_digest = str(protocol["protocol_hash"])
    records_path = shard_dir / "records.jsonl.zst"
    progress_path = shard_dir / "progress.json"
    start_index = next_local_index(
        progress_path,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        protocol_hash=protocol_digest,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    existing = read_jsonl_zst(records_path) if records_path.exists() else []
    if len(existing) != start_index:
        raise ValueError(
            f"resume state mismatch for shard {shard_index}: records={len(existing)} next_local_index={start_index}"
        )

    provider = model_provider or TransformersModelProvider(
        model_ids={
            "student": config.models["student"]["id"],
            "teacher": config.models["teacher"]["id"],
            "evaluator": config.models["evaluator"]["id"],
            "guidance_guard": config.models["evaluator"]["id"],
            "guidance_critic": config.models["evaluator"]["id"],
        },
        model_revisions={
            "student": config.models["student"]["revision"],
            "teacher": config.models["teacher"]["revision"],
            "evaluator": config.models["evaluator"]["revision"],
            "guidance_guard": config.models["evaluator"]["revision"],
            "guidance_critic": config.models["evaluator"]["revision"],
        },
    )
    evaluate = evaluator or make_model_evaluator(
        generate=provider.generate_result,
        generation_kwargs=paper_generation_kwargs(config, role="evaluator"),
        max_regenerations=int(config.collection["max_guidance_regenerations"]),
    )
    guard = guidance_guard or make_model_guidance_guard(
        generate=provider.generate_result,
        generation_kwargs=paper_generation_kwargs(config, role="guidance_guard"),
    )
    critic = guidance_critic or make_model_guidance_critic(
        generate=provider.generate_result,
        generation_kwargs=paper_generation_kwargs(config, role="guidance_critic"),
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
        protocol_hash=protocol_digest,
        source_commit=source_commit,
    )

    for local_index in range(start_index, len(selected)):
        example = selected[local_index]
        generation_events: list[dict[str, Any]] = []
        guidance_events: list[dict[str, Any]] = []

        def student_generate(prompt: str) -> Any:
            start = time.monotonic_ns()
            generation = provider.generate_result(
                "student", prompt, **paper_generation_kwargs(config, role="student")
            )
            generation_events.append(
                {
                    "role": "student",
                    "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                    **asdict(generation),
                }
            )
            return generation

        def teacher_guidance(
            example_row: dict[str, Any],
            rollout: str,
            result: dict[str, Any],
            attempt: int,
            regeneration: int,
            prior_reviews: list[dict[str, Any]],
        ) -> str:
            prompt = build_privileged_guidance_prompt(
                problem=str(example_row["problem"]),
                gold_answer=str(example_row["gold_answer"]),
                rollout=rollout,
                result=result,
                domain=str(example_row["domain"]),
                prior_reviews=prior_reviews,
            )
            start = time.monotonic_ns()
            generation = provider.generate_result(
                "teacher", prompt, **paper_generation_kwargs(config, role="teacher")
            )
            guidance = generation.text
            generation_events.append(
                {
                    "role": "teacher",
                    "attempt": attempt,
                    "regeneration": regeneration,
                    "latency_ms": (time.monotonic_ns() - start) // 1_000_000,
                    **asdict(generation),
                }
            )
            guidance_events.append({
                "attempt": attempt,
                "regeneration": regeneration,
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

        try:
            result = build_native_iterative_guidance_pairs(
                examples=[example],
                base_prompt_builder=base_prompt_builder,
                retry_prompt_builder=retry_prompt_builder,
                student_generate=student_generate,
                evaluate=evaluate,
                teacher_guidance=teacher_guidance,
                guidance_guard=guard,
                guidance_critic=critic,
                max_guidance_steps=int(config.collection["max_guidance_steps"]),
                max_guidance_regenerations=int(config.collection["max_guidance_regenerations"]),
            )
        except ModelOutputParseError as exc:
            failure_path = shard_dir / "model_failures.jsonl"
            failures = read_jsonl(failure_path) if failure_path.exists() else []
            failures.append(
                {
                    "id": str(example["id"]),
                    "local_index": local_index,
                    "role": exc.role,
                    "error_code": "model_output_parse_failed",
                    "message": str(exc),
                    "raw_output": exc.raw,
                    "raw_outputs": exc.raw_outputs,
                    "parse_failures": exc.parse_failures,
                    "source_commit": source_commit,
                    "protocol_hash": protocol_digest,
                }
            )
            write_jsonl(failure_path, failures)
            logger.failure(
                stage=exc.role,
                error_code="model_output_parse_failed",
                message=str(exc),
                example_id=str(example["id"]),
                raw_output_path=str(failure_path),
            )
            raise
        record = {
            "id": str(example["id"]),
            "split": split,
            "local_index": local_index,
            "source_key": example.get("source_key"),
            "source_commit": source_commit,
            "protocol_hash": protocol_digest,
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
            protocol_hash=protocol_digest,
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
        protocol_hash=protocol_digest,
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
    source_commit: str,
) -> dict[str, Any]:
    config_digest = _config_hash(config_path)
    manifest_digest = _manifest_hash(dataset_dir)
    config = load_paper_experiment(config_path)
    protocol = _build_protocol_manifest(
        config=config,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        source_commit=source_commit,
    )
    protocol_digest = str(protocol["protocol_hash"])
    records = merge_completed_shards(
        collection_dir,
        expected_shards=expected_shards,
        config_hash=config_digest,
        dataset_manifest_hash=manifest_digest,
        protocol_hash=protocol_digest,
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite merged collection: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        append_jsonl_zst(output_path, record)
    write_json_atomic(
        output_path.with_suffix(".manifest.json"),
        {
            "schema": "paper-collection-v2",
            "source_commit": source_commit,
            "protocol_hash": protocol_digest,
            "config_hash": config_digest,
            "dataset_manifest_hash": manifest_digest,
            "expected_shards": expected_shards,
            "records": len(records),
        },
    )
    return {"records": len(records), "output_path": str(output_path)}
