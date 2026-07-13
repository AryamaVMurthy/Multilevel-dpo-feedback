from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from text_feedback_dpo.config import load_config
from text_feedback_dpo.dataset import (
    attach_evidence,
    build_sft_rows_from_trajectories,
    load_searchqa_split_with_stats,
    stream_searchqa_split_with_stats,
    stream_stats_report,
    write_jsonl,
)
from text_feedback_dpo.io import iter_jsonl as _iter_jsonl
from text_feedback_dpo.scoring import score_searchqa


def iter_jsonl(path: Path):
    yield from _iter_jsonl(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _require_unique_ids(rows: list[dict], *, label: str) -> list[dict]:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        example_id = row.get("id", row.get("example_id"))
        if not isinstance(example_id, str) or not example_id.strip() or example_id in seen:
            raise ValueError(f"{label} rows require unique non-empty ids; invalid or duplicate id at row {index}: {example_id!r}")
        seen.add(example_id)
    return rows


def read_unique_jsonl(path: Path, *, label: str) -> list[dict]:
    return _require_unique_ids(read_jsonl(path), label=label)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cmd_prepare(args: argparse.Namespace) -> None:
    from text_feedback_dpo.retrieval import canonicalize_materialized_source_records
    from text_feedback_dpo.runtime import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer_model, revision=args.tokenizer_revision)
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    if args.source == "kyunghyuncho/search_qa":
        rows, stream_stats = stream_searchqa_split_with_stats(args.source, args.split, revision=args.revision, limit=args.limit)
        partial_output = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with partial_output.open("w", encoding="utf-8") as handle:
            for row_index, row in enumerate(rows, start=1):
                prepared = attach_evidence([row], max_evidence_tokens=args.max_evidence_tokens, token_count=token_count)[0]
                try:
                    prepared["sources"] = canonicalize_materialized_source_records(prepared.get("sources"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid materialized sources in row {row_index}: {exc}") from exc
                handle.write(json.dumps(prepared, sort_keys=True, ensure_ascii=False) + "\n")
                row_count += 1
        load_stats = stream_stats_report(stream_stats)
        partial_output.replace(args.output)
        manifest = {"source": args.source, "split": args.split, "rows": row_count, "max_length": 4096, "load_stats": load_stats, "required_files": [args.output.name]}
        write_json(args.output.with_suffix(".manifest.json"), manifest)
        write_json(args.output.parent / "manifest.json", manifest)
        return
    rows, load_stats = load_searchqa_split_with_stats(args.source, args.split, revision=args.revision, limit=args.limit)
    rows = attach_evidence(rows, max_evidence_tokens=args.max_evidence_tokens, token_count=token_count)
    for row_index, row in enumerate(rows, start=1):
        try:
            row["sources"] = canonicalize_materialized_source_records(row.get("sources"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid materialized sources in row {row_index}: {exc}") from exc
    write_jsonl(rows, args.output)
    manifest = {"source": args.source, "split": args.split, "rows": len(rows), "max_length": 4096, "load_stats": load_stats, "required_files": [args.output.name]}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.parent / "manifest.json", manifest)


def cmd_shard_jsonl(args: argparse.Namespace) -> None:
    from text_feedback_dpo.sharding import shard_jsonl

    shard_jsonl(args.input, args.output_dir, shard_count=args.shards)


def cmd_merge_predictions(args: argparse.Namespace) -> None:
    from text_feedback_dpo.sharding import merge_prediction_shards

    result = merge_prediction_shards(args.shard_dir, args.output, shard_count=args.shards)
    write_json(args.output.with_suffix(".manifest.json"), result)


def cmd_probe_model(args: argparse.Namespace) -> None:
    from text_feedback_dpo.runtime import (
        extract_qwen_final_content,
        generate_batch,
        generate_student_batch,
        load_student,
        load_teacher,
        load_tokenizer,
        render_teacher_prompts,
    )

    tokenizer = load_tokenizer(args.model, revision=args.model_revision)
    if args.role == "student":
        model = load_student(args.model, revision=args.model_revision, attention_implementation=args.attention_implementation)
        prompt = "Evidence:\nAda Lovelace wrote the first algorithm.\n\nQuestion: Who wrote the first algorithm?\n\nAnswer:"
        result = generate_student_batch(
            model, tokenizer, [prompt], mode="direct", scratchpad_max_new_tokens=256,
            answer_max_new_tokens=32, temperature=0.0, top_p=1.0,
        )[0]
        write_json(args.output, {"role": "student", "model": args.model, "response": result.response, "truncated": result.truncated})
        return
    from text_feedback_dpo.feedback import parse_feedback
    from text_feedback_dpo.prompts import build_teacher_prompt

    model = load_teacher(
        args.model, revision=args.model_revision, quantization=args.teacher_quantization,
        fallback_reason=args.teacher_fallback_reason,
        attention_implementation=args.attention_implementation,
    )
    teacher_source = {
        "source_id": "S001",
        "original_rank": 1,
        "title": "Algorithm history",
        "url": "https://example.test/algorithm-history",
        "snippet": "Ada Lovelace wrote the first algorithm.",
    }
    prompt = build_teacher_prompt(
        {
            "question": "Who wrote the first algorithm?",
            "packed_evidence": "Ada Lovelace wrote the first algorithm.",
            "gold_answer": "Ada Lovelace",
            "sources": [teacher_source],
        },
        "Grace Hopper",
        [],
        raw_query="first algorithm author",
        retrieved_sources=[teacher_source],
        diagnostics={"responsible_region": "answer", "error_code": "answer_mismatch"},
    )
    rendered = render_teacher_prompts(tokenizer, [prompt], enable_thinking=True)
    final = extract_qwen_final_content(generate_batch(model, tokenizer, rendered, max_new_tokens=args.teacher_max_new_tokens, temperature=0.0, top_p=1.0)[0])
    feedback = parse_feedback(final, gold_answer="Ada Lovelace")
    write_json(args.output, {"role": "teacher", "model": args.model, "quantization": args.teacher_quantization, "hint": feedback.hint, "native_thinking": True})


def cmd_build_sft(args: argparse.Namespace) -> None:
    from text_feedback_dpo.runtime import load_tokenizer
    from text_feedback_dpo.training import validate_student_model_selection

    config = load_config(args.config)
    model_id, revision = validate_student_model_selection(config)
    examples = {row["id"]: row for row in read_unique_jsonl(args.data, label="SFT canonical dataset")}
    trajectories = read_unique_jsonl(args.trajectories, label="SFT trajectories")
    tokenizer = load_tokenizer(model_id, revision=revision)
    rows, report = build_sft_rows_from_trajectories(
        trajectories, examples=examples, tokenizer=tokenizer,
        min_coverage=args.min_coverage, min_rows=args.min_rows,
    )
    write_jsonl(rows, args.output)
    write_json(args.report, report)


def cmd_precompute_dpo_refs(args: argparse.Namespace) -> None:
    from text_feedback_dpo.runtime import load_student, load_tokenizer
    from text_feedback_dpo.trainers import precompute_reference_log_probs, validate_prompt_completion_lengths
    from text_feedback_dpo.training import (
        build_reference_manifest,
        dataset_identity_hash,
        validate_student_model_selection,
    )

    config = load_config(args.config)
    model_id, revision = validate_student_model_selection(
        config, requested_model=args.model, requested_revision=args.model_revision,
    )
    rows = read_unique_jsonl(args.data, label="DPO reference precompute")
    tokenizer = load_tokenizer(model_id, revision=revision)
    validate_prompt_completion_lengths(rows, tokenizer, method="dpo")
    schema = json.loads(args.prompt_context_schema.read_text(encoding="utf-8"))
    manifest = build_reference_manifest(
        model=model_id, model_revision=revision,
        reference_checkpoint_hash=args.reference_checkpoint_hash,
        tokenizer=model_id, tokenizer_revision=revision,
        data_hash=dataset_identity_hash(rows), prompt_context_schema=schema, max_length=4096,
    )
    model = load_student(
        model_id, revision=revision,
        attention_implementation=args.attention_implementation, device=args.device,
    )
    precompute_reference_log_probs(
        model=model, tokenizer=tokenizer, rows=rows, output_path=args.output, manifest=manifest,
    )


def cmd_report(args: argparse.Namespace) -> None:
    from text_feedback_dpo.report import write_html_report

    write_html_report(args.output, json.loads(args.metrics.read_text(encoding="utf-8")), args.artifact)


def cmd_validate(args: argparse.Namespace) -> None:
    from text_feedback_dpo.artifacts import validate_artifacts

    write_json(args.output, validate_artifacts(args.directory))


def cmd_compare(args: argparse.Namespace) -> None:
    from text_feedback_dpo.comparison import comparison_metrics
    from text_feedback_dpo.report import write_html_report

    metrics = comparison_metrics(args.run)
    write_json(args.output, metrics)
    write_html_report(args.html, metrics, [args.output.name, *args.artifact])


def cmd_evaluate(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preflight import has_active_search_fields

    example_rows = read_unique_jsonl(args.data, label="example")
    examples = {row["id"]: row for row in example_rows}
    predictions = read_unique_jsonl(args.predictions, label="prediction")
    if set(examples) != {row.get("id", row.get("example_id")) for row in predictions}:
        raise ValueError("prediction/example ID parity mismatch")
    if args.protocol == "active-search":
        _cmd_evaluate_active_search(args, examples, predictions)
        return
    if any(has_active_search_fields(prediction) for prediction in predictions):
        raise ValueError("archival evaluation received active-search fields; choose --protocol active-search explicitly")
    results = []
    for prediction in predictions:
        example_id = prediction.get("id", prediction.get("example_id"))
        if example_id not in examples:
            raise ValueError(f"prediction has unknown example id: {example_id}")
        example = examples[example_id]
        result = score_searchqa(prediction["response"], example["gold_answer"], example["packed_evidence"])
        results.append({"id": example_id, **result})
    exact = sum(row["exact_match"] for row in results) / len(results)
    f1 = sum(row["f1"] for row in results) / len(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, args.output.with_suffix(".jsonl"))
    write_json(args.output, {"examples": len(results), "exact_match": exact, "f1": f1, "correct": sum(row["correct"] for row in results)})


def _cmd_evaluate_active_search(args: argparse.Namespace, examples: dict[str, dict], predictions: list[dict]) -> None:
    from text_feedback_dpo.preflight import evaluate_active_predictions, summarize_active_evaluation

    results = evaluate_active_predictions(list(examples.values()), predictions)
    summary = summarize_active_evaluation(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, args.output.with_suffix(".jsonl"))
    write_json(args.output, summary)


def cmd_preflight_quality(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preflight import (
        assess_preflight,
        evaluate_active_predictions,
        select_preflight_rows,
        summarize_active_evaluation,
        summarize_response_quality,
    )

    if args.split_name != "train-dev":
        raise ValueError("prompt preflight may use only the explicit train-dev split")
    examples = read_jsonl(args.data)
    predictions = read_unique_jsonl(args.predictions, label="prediction")
    canonical_active_rows = None
    if args.protocol == "active-search":
        canonical_active_rows = evaluate_active_predictions(examples, predictions)
        metrics = summarize_active_evaluation(canonical_active_rows)
    else:
        metrics = summarize_response_quality(examples, predictions, protocol=args.protocol)
    metrics["split_name"] = args.split_name
    metrics["gate"] = assess_preflight(metrics)
    selected = select_preflight_rows(examples, sample_size=args.sample_size, seed=args.seed)
    predictions_by_id = {str(row["id"]): row for row in predictions}
    if args.protocol == "active-search":
        canonical_by_id = {row["id"]: row for row in canonical_active_rows}
        samples = [
            {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "raw_query": canonical_by_id[str(example["id"])]["raw_query"],
                "ranked_search_results": canonical_by_id[str(example["id"])]["ranked_search_results"],
                "raw_response": canonical_by_id[str(example["id"])]["raw_response"],
                "parsed_response": canonical_by_id[str(example["id"])]["parsed_response"],
                "rendered_visible_response": canonical_by_id[str(example["id"])]["rendered_visible_response"],
                "cited_score": canonical_by_id[str(example["id"])]["cited_score"],
                "retrieval_metrics": canonical_by_id[str(example["id"])]["retrieval_metrics"],
                "truncation": canonical_by_id[str(example["id"])]["truncation"],
                "timings_ms": canonical_by_id[str(example["id"])]["timings_ms"],
            }
            for example in selected
        ]
    else:
        samples = [
            {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "response": predictions_by_id[str(example["id"])]["response"],
                "truncated": predictions_by_id[str(example["id"])]["truncated"],
            }
            for example in selected
        ]
    write_jsonl(samples, args.samples)
    write_json(args.output, metrics)


def cmd_select_thinking_mode(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preflight import select_thinking_mode

    summaries = {
        "direct": json.loads(args.direct.read_text(encoding="utf-8")),
        "two_pass": json.loads(args.two_pass.read_text(encoding="utf-8")),
    }
    write_json(args.output, select_thinking_mode(summaries))


def cmd_generate(args: argparse.Namespace) -> None:
    from text_feedback_dpo.prompts import build_student_prompt
    from text_feedback_dpo.runtime import generate_student_batch, load_student, load_tokenizer

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = read_jsonl(args.data)
    tokenizer = load_tokenizer(args.model, revision=args.model_revision)
    model = load_student(args.model, revision=args.model_revision, attention_implementation=args.attention_implementation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [row.get("prompt") or build_student_prompt(row, []) for row in batch]
            generations = generate_student_batch(
                model, tokenizer, prompts, mode=args.student_thinking_mode,
                scratchpad_max_new_tokens=args.scratchpad_max_new_tokens,
                answer_max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            )
            for row, generation in zip(batch, generations, strict=True):
                output = {"id": row["id"], "response": generation.response, "truncated": generation.truncated, "thinking_mode": generation.mode, "policy_hash": args.policy_hash}
                if generation.scratchpad is not None:
                    output["private_scratchpad"] = generation.scratchpad
                    output["private_scratchpad_truncated"] = generation.scratchpad_truncated
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()


def cmd_generate_searchqa(args: argparse.Namespace) -> None:
    """Generate structured SearchQA query/search/cited-response trajectories."""
    from text_feedback_dpo.batch_generation import RESPONSE_SCHEMA_VERSION, _validate_rows, run_fixed_retrieval_pipeline
    from text_feedback_dpo.prompts import prompt_builder_identity
    from text_feedback_dpo.runtime import generate_batch_records, generate_student_batch, load_student, load_tokenizer
    from text_feedback_dpo.searchqa import SOURCE_SCHEMA, SOURCE_SCHEMA_VERSION

    if args.query_batch_size <= 0 or args.response_batch_size <= 0:
        raise ValueError("query_batch_size and response_batch_size must be positive")
    if args.context_budget != 4096:
        raise ValueError("active SearchQA generation requires the explicit 4096-token total context budget")
    if args.query_max_new_tokens <= 0 or args.response_max_new_tokens <= 0:
        raise ValueError("query and response max_new_tokens must be positive")
    rows = read_jsonl(args.data)
    _validate_rows(rows)
    tokenizer = load_tokenizer(args.model, revision=args.model_revision)
    model = load_student(args.model, revision=args.model_revision, attention_implementation=args.attention_implementation, device=args.device)

    def generate_stage(prompts: list[str], *, max_new_tokens: int, temperature: float, instruction: str, scratchpad_instruction: str):
        if args.student_thinking_mode == "direct":
            return generate_batch_records(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
                context_budget=args.context_budget,
            )
        return generate_student_batch(
            model,
            tokenizer,
            prompts,
            mode="two_pass",
            scratchpad_max_new_tokens=args.scratchpad_max_new_tokens,
            answer_max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            generation_fn=lambda _model, _tokenizer, batch_prompts, **kwargs: generate_batch_records(
                model,
                tokenizer,
                batch_prompts,
                context_budget=args.context_budget,
                **kwargs,
            ),
            visible_instruction=instruction,
            scratchpad_instruction=scratchpad_instruction,
        )

    results = run_fixed_retrieval_pipeline(
        rows,
        query_generate_batch=lambda prompts: generate_stage(
            prompts,
            max_new_tokens=args.query_max_new_tokens,
            temperature=args.query_temperature,
            instruction="Return exactly one nonempty one-line plain-text search query. Do not use XML, JSON, code fences, or labels.",
            scratchpad_instruction="Privately identify retrieval terms and entities only; do not solve the question or draft the answer.",
        ),
        response_generate_batch=lambda prompts: generate_stage(
            prompts,
            max_new_tokens=args.response_max_new_tokens,
            temperature=args.response_temperature,
            instruction="Return exactly the requested three-line Answer/Reasoning/Sources response in plain text.",
            scratchpad_instruction="Privately reason over the retrieved sources and citation order; do not emit or imitate this scratchpad.",
        ),
        query_batch_size=args.query_batch_size,
        response_batch_size=args.response_batch_size,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        policy_hash=args.policy_hash,
        prompt_version=args.prompt_version,
    )
    write_jsonl(results, args.output)
    prompt_identity = {"identity": args.prompt_version, "builders": prompt_builder_identity()}
    response_identity = {"identity": "cited-response", "schema_version": RESPONSE_SCHEMA_VERSION, "parser": "parse_cited_response", "renderer": "render_cited_response"}
    source_identity = {"identity": SOURCE_SCHEMA, "version": SOURCE_SCHEMA_VERSION}
    pipeline_wall_ms = float(results[0]["timings_ms"]["pipeline_wall_ms"]) if results else 0.0
    manifest = {
        "command": "generate-searchqa",
        "max_length": 4096,
        "rows": len(results),
        "model": {"identity": args.model, "revision": args.model_revision, "policy_hash": args.policy_hash},
        "dataset": {"source": args.dataset_source, "revision": args.dataset_revision, "sha256": _sha256_file(args.data)},
        "source_schema": {**source_identity, "sha256": _identity_hash(source_identity)},
        "retrieval": {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": args.top_k, "k1": args.k1, "b": args.b},
        "prompt": {**prompt_identity, "sha256": _identity_hash(prompt_identity)},
        "response": {**response_identity, "sha256": _identity_hash(response_identity)},
        "generation": {
            "context_budget": args.context_budget,
            "query_batch_size": args.query_batch_size,
            "response_batch_size": args.response_batch_size,
            "query_max_new_tokens": args.query_max_new_tokens,
            "response_max_new_tokens": args.response_max_new_tokens,
            "student_thinking_mode": args.student_thinking_mode,
        },
        "timing": {"pipeline_wall_ms": pipeline_wall_ms},
        "required_files": [args.output.name],
        "artifacts": [{
            "path": args.output.name,
            "format": "jsonl",
            "rows": len(results),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256_file(args.output),
        }],
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.parent / "manifest.json", manifest)


def cmd_build_preferences(args: argparse.Namespace) -> None:
    from text_feedback_dpo.preferences import build_preference_rows
    from text_feedback_dpo.trajectories import revalidate_cached_trajectory

    examples = read_unique_jsonl(args.data, label="preference example")
    trajectories = read_unique_jsonl(args.trajectories, label="preference trajectory")
    example_by_id = {str(example["id"]): example for example in examples}
    trajectory_by_id = {str(trajectory["id"]): trajectory for trajectory in trajectories}
    if set(example_by_id) != set(trajectory_by_id) or len(examples) != len(trajectories):
        raise ValueError("preference trajectory/example ID parity mismatch")
    preference_rows = []
    for example_id in example_by_id:
        trajectory = trajectory_by_id[example_id]
        siblings = trajectory.get("no_hint_siblings")
        if not isinstance(siblings, list):
            raise ValueError(f"preference trajectory {example_id} requires no_hint_siblings")
        sibling_seeds = [sibling.get("seed") if isinstance(sibling, dict) else None for sibling in siblings]
        validated = revalidate_cached_trajectory(
            trajectory, example=example_by_id[example_id], expected_sibling_seeds=sibling_seeds,
        )
        preference_rows.extend(build_preference_rows(validated))
    if not preference_rows:
        args.output.unlink(missing_ok=True)
        raise ValueError("no valid preference rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(preference_rows, args.output)


def cmd_collect(args: argparse.Namespace) -> None:
    from text_feedback_dpo.batch_generation import (
        EVALUATOR_VERSION,
        FIXED_B,
        FIXED_K1,
        FIXED_TOP_K,
        PROMPT_VERSION,
        RESPONSE_SCHEMA_VERSION,
        run_fixed_retrieval_pipeline,
    )
    from text_feedback_dpo.collection import collect_dataset_batchwise
    from text_feedback_dpo.offline import build_cache_manifest, load_or_build_trajectories
    from text_feedback_dpo.prompts import prompt_builder_identity
    from text_feedback_dpo.runtime import (
        extract_qwen_final_content,
        generate_batch,
        generate_batch_records,
        generate_student_batch,
        load_student,
        load_teacher,
        load_tokenizer,
        render_teacher_prompts,
        set_generation_seed,
        validate_teacher_identity,
    )
    from text_feedback_dpo.searchqa import SOURCE_SCHEMA, SOURCE_SCHEMA_VERSION

    examples = read_jsonl(args.data)
    if args.prompt_version != PROMPT_VERSION:
        raise ValueError(f"collect prompt_version must be {PROMPT_VERSION}")
    if args.sibling_count <= 0 or len(args.sibling_seeds) != args.sibling_count:
        raise ValueError("sibling_count must be positive and exactly match sibling_seeds")
    if len(set(args.sibling_seeds)) != len(args.sibling_seeds) or any(seed < 0 for seed in args.sibling_seeds):
        raise ValueError("sibling_seeds must be unique nonnegative integers")
    if args.student_batch_size <= 0 or args.teacher_batch_size <= 0:
        raise ValueError("student_batch_size and teacher_batch_size must be positive")
    teacher_identity = validate_teacher_identity(
        args.teacher_model,
        revision=args.teacher_revision,
        quantization=args.teacher_quantization,
        fallback_reason=args.teacher_fallback_reason,
    )

    def generate_trajectories(pending):
        student_tokenizer = load_tokenizer(args.student_model, revision=args.student_revision)
        student = load_student(args.student_model, revision=args.student_revision, attention_implementation=args.attention_implementation, device=args.student_device)
        teacher_tokenizer = load_tokenizer(args.teacher_model, revision=args.teacher_revision)
        teacher = load_teacher(
            args.teacher_model, revision=args.teacher_revision, quantization=args.teacher_quantization,
            fallback_reason=args.teacher_fallback_reason,
            attention_implementation=args.attention_implementation, device=args.teacher_device,
        )
        def batched_generate(model, tokenizer, prompts, *, batch_size, **kwargs):
            outputs = []
            for start in range(0, len(prompts), batch_size):
                outputs.extend(generate_batch(model, tokenizer, prompts[start : start + batch_size], **kwargs))
            return outputs

        def generate_stage(prompts, *, max_new_tokens, temperature, instruction, scratchpad_instruction):
            if args.student_thinking_mode == "two_pass":
                return generate_student_batch(
                    student, student_tokenizer, prompts, mode="two_pass",
                    scratchpad_max_new_tokens=args.scratchpad_max_new_tokens,
                    answer_max_new_tokens=max_new_tokens, temperature=temperature,
                    top_p=args.student_top_p,
                    generation_fn=lambda _model, _tokenizer, batch_prompts, **kwargs: generate_batch_records(
                        student, student_tokenizer, batch_prompts, context_budget=4096, **kwargs
                    ),
                    visible_instruction=instruction,
                    scratchpad_instruction=scratchpad_instruction,
                )
            outputs = []
            for start in range(0, len(prompts), args.student_batch_size):
                outputs.extend(generate_batch_records(
                    student, student_tokenizer, prompts[start : start + args.student_batch_size],
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    top_p=args.student_top_p, context_budget=4096,
                ))
            return outputs

        def run_student_requests(requests, *, seed):
            if not isinstance(requests, list) or not requests:
                raise ValueError("student active generation requires a nonempty request batch")
            if any(not isinstance(request, dict) for request in requests):
                raise ValueError("student active requests must be mappings")
            expected_ids = [request.get("id") for request in requests]
            rows = [request["example"] for request in requests]
            if [row.get("id") for row in rows] != expected_ids:
                raise ValueError("student active request ID parity mismatch")
            set_generation_seed(seed)
            return run_fixed_retrieval_pipeline(
                rows,
                query_generate_batch=lambda prompts: generate_stage(
                    prompts, max_new_tokens=args.query_max_new_tokens,
                    temperature=args.student_temperature,
                    instruction="Return exactly one nonempty one-line plain-text search query. Do not use XML, JSON, code fences, or labels.",
                    scratchpad_instruction="Privately identify retrieval terms and entities only; do not solve the question or draft the answer.",
                ),
                response_generate_batch=lambda prompts: generate_stage(
                    prompts, max_new_tokens=args.response_max_new_tokens,
                    temperature=args.student_temperature,
                    instruction="Return exactly the requested three-line Answer/Reasoning/Sources response in plain text.",
                    scratchpad_instruction="Privately reason over the retrieved sources and citation order; do not emit or imitate this scratchpad.",
                ),
                query_batch_size=args.student_batch_size,
                response_batch_size=args.student_batch_size,
                top_k=FIXED_TOP_K, k1=FIXED_K1, b=FIXED_B,
                policy_hash=args.policy_hash, prompt_version=args.prompt_version,
                response_schema_version=RESPONSE_SCHEMA_VERSION,
                evaluator_version=EVALUATOR_VERSION,
                hints_by_id={request["id"]: list(request["hints"]) for request in requests},
            )

        def student_batch(requests, **kwargs):
            return run_student_requests(requests, seed=kwargs["seed"])

        def sibling_batch(requests, **_kwargs):
            if not isinstance(requests, list) or not requests:
                raise ValueError("sibling generation requires a nonempty request batch")
            grouped = {}
            for index, request in enumerate(requests):
                if not isinstance(request, dict) or not isinstance(request.get("seed"), int):
                    raise ValueError(f"sibling request {index} requires an integer seed")
                grouped.setdefault(request["seed"], []).append((index, request))
            ordered_outputs = [None] * len(requests)
            for seed, indexed_requests in grouped.items():
                batch = [request for _, request in indexed_requests]
                generated = run_student_requests(batch, seed=seed)
                if len(generated) != len(batch):
                    raise ValueError(
                        f"sibling seed {seed} cardinality mismatch: "
                        f"expected {len(batch)}, got {len(generated)}"
                    )
                for (index, _request), artifact in zip(indexed_requests, generated, strict=True):
                    ordered_outputs[index] = artifact
            if any(output is None for output in ordered_outputs):
                raise ValueError("sibling generation did not populate every requested artifact")
            return ordered_outputs

        def teacher_batch(prompts, **kwargs):
            rendered = render_teacher_prompts(teacher_tokenizer, prompts, enable_thinking=args.teacher_thinking)
            raw = batched_generate(
                teacher, teacher_tokenizer, rendered,
                batch_size=args.teacher_batch_size,
                max_new_tokens=args.teacher_max_new_tokens,
                temperature=kwargs["temperature"], top_p=kwargs["top_p"],
            )
            return [extract_qwen_final_content(text) for text in raw]

        return collect_dataset_batchwise(
            examples=pending,
            student_generate_batch=student_batch,
            teacher_generate_batch=teacher_batch,
            max_interventions=args.max_interventions,
            teacher_max_new_tokens=args.teacher_max_new_tokens,
            sibling_generate_batch=sibling_batch,
            sibling_seeds=args.sibling_seeds,
            student_seed=args.seed,
        )

    source_identity = {"identity": SOURCE_SCHEMA, "version": SOURCE_SCHEMA_VERSION}
    retrieval_config = {
        "identity": "fixed_bm25", "schema_version": 1,
        "requested_top_k": FIXED_TOP_K, "k1": FIXED_K1, "b": FIXED_B,
    }
    prompt_identity = {"identity": PROMPT_VERSION, "builders": prompt_builder_identity()}
    response_identity = {"identity": "cited-response", "schema_version": RESPONSE_SCHEMA_VERSION}
    evaluator_identity = {"identity": EVALUATOR_VERSION}

    cache_manifest = build_cache_manifest(
        student_model=args.student_model,
        student_revision=args.student_revision,
        teacher_model=args.teacher_model,
        teacher_revision=args.teacher_revision,
        teacher_identity=teacher_identity,
        teacher_quantization=args.teacher_quantization,
        teacher_fallback_reason=args.teacher_fallback_reason,
        dataset_revision=args.dataset_revision,
        dataset_hash=_sha256_file(args.data),
        dataset_schema=SOURCE_SCHEMA,
        source_schema_version=SOURCE_SCHEMA_VERSION,
        source_schema_hash=_identity_hash(source_identity),
        retrieval_config=retrieval_config,
        retrieval_hash=_identity_hash(retrieval_config),
        prompt_version=args.prompt_version,
        prompt_hash=_identity_hash(prompt_identity),
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        response_schema_hash=_identity_hash(response_identity),
        evaluator_version=EVALUATOR_VERSION,
        evaluator_hash=_identity_hash(evaluator_identity),
        policy_version=args.policy_version,
        student_thinking_mode=args.student_thinking_mode,
        teacher_thinking=args.teacher_thinking,
        decoding={
            "context_budget": 4096,
            "query_max_new_tokens": args.query_max_new_tokens,
            "response_max_new_tokens": args.response_max_new_tokens,
            "scratchpad_max_new_tokens": args.scratchpad_max_new_tokens,
            "student_temperature": args.student_temperature,
            "student_top_p": args.student_top_p,
            "teacher_max_new_tokens": args.teacher_max_new_tokens,
            "teacher_temperature": 0.0,
            "teacher_top_p": 1.0,
            "student_batch_size": args.student_batch_size,
            "teacher_batch_size": args.teacher_batch_size,
        },
        intervention_policy={"max_interventions": args.max_interventions, "max_hint_words": 24},
        sibling_count=args.sibling_count,
        sibling_seeds=list(args.sibling_seeds),
        seed=args.seed,
        policy_hash=args.policy_hash,
    )
    rows = load_or_build_trajectories(examples=examples, cache_path=args.trajectory_cache, cache_manifest=cache_manifest, generate=generate_trajectories)
    write_jsonl(rows, args.output)
    manifest = {
        "command": "collect", "max_length": 4096, "rows": len(rows),
        "cache_identity": cache_manifest, "teacher_identity": teacher_identity,
        "sibling_count": args.sibling_count, "sibling_seeds": list(args.sibling_seeds),
        "required_files": [args.output.name],
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.parent / "manifest.json", manifest)


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    from text_feedback_dpo.trainers import run_dapo, run_dpo, run_grpo, run_sft
    from text_feedback_dpo.training import validate_student_model_selection

    if args.max_length != 4096:
        raise ValueError("Task 7 combined max_length is fixed at exactly 4096")
    if args.use_liger_kernel:
        raise ValueError("Liger is explicitly disabled for Task 7 training")
    if args.method != "sft" and (args.packing or args.padding_free):
        raise ValueError("packing and padding-free controls are supported only for SFT")
    model_id, revision = validate_student_model_selection(
        config, requested_model=args.model, requested_revision=args.model_revision,
    )
    common = {
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "deepspeed_config": args.deepspeed_config,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_steps": args.save_steps, "eval_steps": args.eval_steps,
        "dapo_enabled": args.method == "dapo",
        "attention_implementation": args.attention_implementation,
        "model_revision": revision,
        "packing": args.packing, "padding_free": args.padding_free,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_completion_length": args.max_completion_length,
        "use_liger_kernel": args.use_liger_kernel,
    }
    if args.method == "dpo":
        schema = json.loads(args.prompt_context_schema.read_text(encoding="utf-8"))
        common.update(
            precomputed_ref_log_probs_path=args.ref_log_probs,
            precomputed_eval_ref_log_probs_path=args.eval_ref_log_probs,
            reference_checkpoint_hash=args.reference_checkpoint_hash,
            prompt_context_schema=schema,
            tokenizer_model=model_id,
            tokenizer_revision=revision,
        )
    kwargs = {
        "model_id": model_id, "train_path": args.train, "eval_path": args.eval,
        "output_dir": args.output, "config": common,
    }
    {"sft": run_sft, "dpo": run_dpo, "grpo": run_grpo, "dapo": run_dapo}[args.method](**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tfdpo")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-searchqa")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--tokenizer-model", required=True)
    prepare.add_argument("--tokenizer-revision", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--max-evidence-tokens", required=True, type=int)
    prepare.add_argument("--limit", type=int)
    prepare.set_defaults(func=cmd_prepare)
    shard = sub.add_parser("shard-jsonl")
    shard.add_argument("--input", required=True, type=Path)
    shard.add_argument("--output-dir", required=True, type=Path)
    shard.add_argument("--shards", required=True, type=int)
    shard.set_defaults(func=cmd_shard_jsonl)
    merge = sub.add_parser("merge-predictions")
    merge.add_argument("--shard-dir", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--shards", required=True, type=int)
    merge.set_defaults(func=cmd_merge_predictions)
    probe = sub.add_parser("probe-model")
    probe.add_argument("--role", choices=("student", "teacher"), required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--model-revision", required=True)
    probe.add_argument("--teacher-quantization", choices=("4bit", "bf16"), required=True)
    probe.add_argument("--teacher-fallback-reason")
    probe.add_argument("--teacher-max-new-tokens", type=int, default=96)
    probe.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    probe.add_argument("--output", required=True, type=Path)
    probe.set_defaults(func=cmd_probe_model)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--protocol", choices=("archival", "active-search"), required=True)
    evaluate.set_defaults(func=cmd_evaluate)
    preflight_quality = sub.add_parser("preflight-quality")
    preflight_quality.add_argument("--data", required=True, type=Path)
    preflight_quality.add_argument("--predictions", required=True, type=Path)
    preflight_quality.add_argument("--output", required=True, type=Path)
    preflight_quality.add_argument("--samples", required=True, type=Path)
    preflight_quality.add_argument("--split-name", required=True)
    preflight_quality.add_argument("--protocol", choices=("archival", "active-search"), required=True)
    preflight_quality.add_argument("--sample-size", type=int, default=32)
    preflight_quality.add_argument("--seed", type=int, default=7)
    preflight_quality.set_defaults(func=cmd_preflight_quality)
    select_thinking = sub.add_parser("select-thinking-mode")
    select_thinking.add_argument("--direct", required=True, type=Path)
    select_thinking.add_argument("--two-pass", required=True, type=Path)
    select_thinking.add_argument("--output", required=True, type=Path)
    select_thinking.set_defaults(func=cmd_select_thinking_mode)
    generate = sub.add_parser("generate")
    generate.add_argument("--data", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--model", required=True)
    generate.add_argument("--model-revision")
    generate.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    generate.add_argument("--batch-size", type=int, default=4)
    generate.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), default="direct")
    generate.add_argument("--scratchpad-max-new-tokens", type=int, default=256)
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--policy-hash", default="unversioned-generate")
    generate.set_defaults(func=cmd_generate)
    active_generate = sub.add_parser("generate-searchqa", help="generate active fixed-search cited-reasoning trajectories")
    active_generate.add_argument("--data", required=True, type=Path)
    active_generate.add_argument("--output", required=True, type=Path)
    active_generate.add_argument("--model", required=True)
    active_generate.add_argument("--model-revision", required=True)
    active_generate.add_argument("--dataset-source", required=True)
    active_generate.add_argument("--dataset-revision", required=True)
    active_generate.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    active_generate.add_argument("--device", default="cuda:0")
    active_generate.add_argument("--policy-hash", required=True)
    active_generate.add_argument("--prompt-version", default="fixed-retrieval-cited-v1")
    active_generate.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), default="direct")
    active_generate.add_argument("--scratchpad-max-new-tokens", type=int, default=256)
    active_generate.add_argument("--query-batch-size", type=int, default=4)
    active_generate.add_argument("--response-batch-size", type=int, default=4)
    active_generate.add_argument("--query-max-new-tokens", type=int, default=32)
    active_generate.add_argument("--response-max-new-tokens", type=int, default=256)
    active_generate.add_argument("--query-temperature", type=float, default=0.0)
    active_generate.add_argument("--response-temperature", type=float, default=0.0)
    active_generate.add_argument("--top-p", type=float, default=1.0)
    active_generate.add_argument("--top-k", type=int, default=8)
    active_generate.add_argument("--k1", type=float, default=1.2)
    active_generate.add_argument("--b", type=float, default=0.75)
    active_generate.add_argument("--context-budget", type=int, default=4096)
    active_generate.set_defaults(func=cmd_generate_searchqa)
    preferences = sub.add_parser("build-preferences")
    preferences.add_argument("--data", required=True, type=Path)
    preferences.add_argument("--trajectories", required=True, type=Path)
    preferences.add_argument("--output", required=True, type=Path)
    preferences.set_defaults(func=cmd_build_preferences)
    sft_data = sub.add_parser("build-sft-data")
    sft_data.add_argument("--config", required=True, type=Path)
    sft_data.add_argument("--data", required=True, type=Path)
    sft_data.add_argument("--trajectories", required=True, type=Path)
    sft_data.add_argument("--output", required=True, type=Path)
    sft_data.add_argument("--report", required=True, type=Path)
    sft_data.add_argument("--min-coverage", required=True, type=float)
    sft_data.add_argument("--min-rows", required=True, type=int)
    sft_data.set_defaults(func=cmd_build_sft)
    refs = sub.add_parser("precompute-dpo-ref-log-probs")
    refs.add_argument("--config", required=True, type=Path)
    refs.add_argument("--data", required=True, type=Path)
    refs.add_argument("--output", required=True, type=Path)
    refs.add_argument("--model")
    refs.add_argument("--model-revision")
    refs.add_argument("--reference-checkpoint-hash", required=True)
    refs.add_argument("--prompt-context-schema", required=True, type=Path)
    refs.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    refs.add_argument("--device", required=True)
    refs.set_defaults(func=cmd_precompute_dpo_refs)
    report = sub.add_parser("report")
    report.add_argument("--metrics", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--artifact", action="append", default=[])
    report.set_defaults(func=cmd_report)
    validate = sub.add_parser("validate-run")
    validate.add_argument("--directory", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.set_defaults(func=cmd_validate)
    compare = sub.add_parser("compare")
    compare.add_argument("--run", action="append", required=True)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--html", required=True, type=Path)
    compare.add_argument("--artifact", action="append", default=[])
    compare.set_defaults(func=cmd_compare)
    collect = sub.add_parser("collect")
    collect.add_argument("--data", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--student-model", required=True)
    collect.add_argument("--teacher-model", required=True)
    collect.add_argument("--student-revision", required=True)
    collect.add_argument("--teacher-revision", required=True)
    collect.add_argument("--dataset-revision", required=True)
    collect.add_argument("--prompt-version", required=True)
    collect.add_argument("--policy-version", required=True)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--teacher-quantization", choices=("4bit", "bf16"), required=True)
    collect.add_argument("--teacher-fallback-reason")
    collect.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
    collect.add_argument("--student-device", required=True)
    collect.add_argument("--teacher-device", required=True)
    collect.add_argument("--max-interventions", type=int, default=4)
    collect.add_argument("--student-batch-size", type=int, default=32)
    collect.add_argument("--teacher-batch-size", type=int, default=8)
    collect.add_argument("--student-thinking-mode", choices=("direct", "two_pass"), default="direct")
    collect.add_argument("--scratchpad-max-new-tokens", type=int, default=256)
    collect.add_argument("--query-max-new-tokens", type=int, default=32)
    collect.add_argument("--response-max-new-tokens", type=int, default=256)
    collect.add_argument("--student-temperature", type=float, default=0.7)
    collect.add_argument("--student-top-p", type=float, default=0.9)
    collect.add_argument("--teacher-max-new-tokens", type=int, default=96)
    collect.add_argument("--teacher-thinking", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--trajectory-cache", required=True, type=Path)
    collect.add_argument("--policy-hash", required=True)
    collect.add_argument("--sibling-count", required=True, type=int)
    collect.add_argument("--sibling-seeds", required=True, nargs="+", type=int)
    collect.set_defaults(func=cmd_collect)
    for method in ("sft", "dpo", "grpo", "dapo"):
        train = sub.add_parser(f"train-{method}")
        train.add_argument("--config", required=True, type=Path)
        train.add_argument("--train", required=True, type=Path)
        train.add_argument("--eval", required=True, type=Path)
        train.add_argument("--output", required=True, type=Path)
        train.add_argument("--model")
        train.add_argument("--model-revision")
        train.add_argument("--learning-rate", type=float, default=1e-6)
        train.add_argument("--epochs", type=float, default=1.0)
        train.add_argument("--max-steps", required=True, type=int)
        train.add_argument("--max-length", required=True, type=int)
        train.add_argument("--per-device-train-batch-size", required=True, type=int)
        train.add_argument("--per-device-eval-batch-size", required=True, type=int)
        train.add_argument("--dataloader-num-workers", required=True, type=int)
        train.add_argument("--gradient-accumulation-steps", required=True, type=int)
        train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
        train.add_argument("--attention-implementation", choices=("sdpa", "flash_attention_2"), required=True)
        train.add_argument("--deepspeed-config", type=Path)
        train.add_argument("--resume-from-checkpoint")
        train.add_argument("--save-steps", type=int, default=100)
        train.add_argument("--eval-steps", type=int, default=100)
        train.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
        train.add_argument("--padding-free", action=argparse.BooleanOptionalAction, default=False)
        train.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=False)
        train.add_argument("--num-generations", type=int, default=4)
        train.add_argument("--generation-batch-size", type=int, default=32)
        train.add_argument("--max-completion-length", type=int, default=256)
        if method == "dpo":
            train.add_argument("--ref-log-probs", required=True, type=Path)
            train.add_argument("--eval-ref-log-probs", required=True, type=Path)
            train.add_argument("--reference-checkpoint-hash", required=True)
            train.add_argument("--prompt-context-schema", required=True, type=Path)
        else:
            train.set_defaults(
                ref_log_probs=None, eval_ref_log_probs=None,
                reference_checkpoint_hash=None, prompt_context_schema=None,
            )
        train.set_defaults(func=cmd_train, method=method)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
