from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_sft_reproduction_report(
    rows: Sequence[Mapping[str, object]],
    generated_by_id: Mapping[str, Any],
) -> tuple[list[dict], dict]:
    """Compare decoded checkpoint continuations with exact verified SFT targets."""
    row_ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("SFT reproduction rows require non-empty string IDs")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("SFT reproduction rows contain a duplicate ID")
    if set(row_ids) != set(generated_by_id):
        missing = sorted(set(row_ids) - set(generated_by_id))
        unexpected = sorted(set(generated_by_id) - set(row_ids))
        raise ValueError(f"SFT reproduction generation ID parity mismatch: missing={missing} unexpected={unexpected}")

    records: list[dict] = []
    for row in rows:
        row_id = str(row["id"])
        task = row.get("task")
        completion = row.get("completion")
        if task not in {"query", "response"}:
            raise ValueError(f"SFT reproduction row {row_id} requires task=query or task=response")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"SFT reproduction row {row_id} requires a non-empty completion")
        generated = generated_by_id[row_id]
        text = getattr(generated, "text", None)
        truncated = getattr(generated, "truncated", None)
        if not isinstance(text, str) or not isinstance(truncated, bool):
            raise TypeError(f"SFT reproduction generation {row_id} must expose text and truncated")
        reference_text = completion.strip()
        generated_text = text.strip()
        records.append({
            "id": row_id,
            "task": task,
            "reference": reference_text,
            "generated": generated_text,
            "exact": generated_text == reference_text,
            "empty": not generated_text,
            "truncated": truncated,
        })

    summary: dict[str, object] = {
        "rows": len(records),
        "exact": sum(record["exact"] for record in records),
        "empty": sum(record["empty"] for record in records),
        "truncated": sum(record["truncated"] for record in records),
        "tasks": {},
        "comparison": "decoded_text_strip_boundary_whitespace_only_no_repair",
    }
    summary["exact_rate"] = summary["exact"] / max(1, len(records))
    for task in ("query", "response"):
        task_records = [record for record in records if record["task"] == task]
        exact = sum(record["exact"] for record in task_records)
        summary["tasks"][task] = {
            "rows": len(task_records),
            "exact": exact,
            "exact_rate": exact / max(1, len(task_records)),
            "empty": sum(record["empty"] for record in task_records),
            "truncated": sum(record["truncated"] for record in task_records),
        }
    return records, summary


def build_sft_capability_report(
    rows: Sequence[Mapping[str, object]],
    reproduction_records: Sequence[Mapping[str, object]],
    *,
    examples_by_id: Mapping[str, Mapping[str, object]],
    bootstrap_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict], dict]:
    """Canonically score generated SFT continuations against their sealed lineage."""
    from text_feedback_dpo.batch_generation import (
        FIXED_B,
        FIXED_K1,
        FIXED_TOP_K,
        SCAFFOLD_PROMPT_VERSION,
        canonical_cited_score,
        parse_search_query,
    )
    from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics
    from text_feedback_dpo.trajectories import _structured_hash, validate_active_artifact

    row_by_id: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(rows):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in row_by_id:
            raise ValueError(f"SFT capability rows require unique non-empty IDs; invalid row {index}")
        row_by_id[row_id] = row
    generated_by_id: dict[str, Mapping[str, object]] = {}
    for index, record in enumerate(reproduction_records):
        row_id = record.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in generated_by_id:
            raise ValueError(f"SFT capability generations require unique non-empty IDs; invalid row {index}")
        generated_by_id[row_id] = record
    if set(row_by_id) != set(generated_by_id):
        missing = sorted(set(row_by_id) - set(generated_by_id))
        unexpected = sorted(set(generated_by_id) - set(row_by_id))
        raise ValueError(f"SFT capability generation ID parity mismatch: missing={missing} unexpected={unexpected}")

    records: list[dict] = []
    for row_id, row in row_by_id.items():
        task = row.get("task")
        metadata = row.get("metadata")
        completion = row.get("completion")
        prompt = row.get("prompt")
        if task not in {"query", "response"}:
            raise ValueError(f"SFT capability row {row_id} requires task=query or task=response")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"SFT capability row {row_id} requires metadata")
        if metadata.get("provenance") != "student" or metadata.get("no_hint") is not True:
            raise ValueError(f"SFT capability row {row_id} is not student-generated no-hint supervision")
        trajectory_id = metadata.get("trajectory_id")
        seed = metadata.get("seed")
        if not isinstance(trajectory_id, str) or not trajectory_id or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"SFT capability row {row_id} has invalid trajectory/seed lineage")
        example = examples_by_id.get(trajectory_id)
        bootstrap = bootstrap_by_id.get(trajectory_id)
        if not isinstance(example, Mapping) or not isinstance(bootstrap, Mapping):
            raise ValueError(f"SFT capability row {row_id} is missing canonical example or bootstrap lineage")
        if bootstrap.get("id") != trajectory_id or bootstrap.get("example_identity") != _structured_hash(example):
            raise ValueError(f"SFT capability row {row_id} bootstrap lineage identity mismatch")
        candidates = bootstrap.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"SFT capability row {row_id} bootstrap lineage has invalid candidates")
        matches = [candidate for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("seed") == seed]
        if len(matches) != 1:
            raise ValueError(f"SFT capability row {row_id} requires exactly one seed-matched bootstrap candidate")
        candidate = matches[0]
        if candidate.get("provenance") != "student" or candidate.get("no_hint") is not True:
            raise ValueError(f"SFT capability row {row_id} bootstrap candidate provenance mismatch")
        artifact = validate_active_artifact(candidate.get("artifact"), example=example, hints=[])
        if candidate.get("artifact_hash") != _structured_hash(artifact):
            raise ValueError(f"SFT capability row {row_id} bootstrap artifact hash mismatch")
        for metadata_field, artifact_field in (
            ("query_prompt_hash", "query_prompt_hash"),
            ("response_prompt_hash", "response_prompt_hash"),
            ("retrieval_context_hash", "retrieval_context_hash"),
        ):
            if metadata.get(metadata_field) != artifact.get(artifact_field):
                raise ValueError(f"SFT capability row {row_id} {metadata_field} lineage mismatch")
        expected_prompt = artifact["query_prompt" if task == "query" else "response_prompt"]
        expected_completion = artifact["raw_query"] if task == "query" else (
            artifact.get("generated_response")
            if artifact.get("prompt_version") == SCAFFOLD_PROMPT_VERSION
            else artifact["raw_response"]
        )
        if prompt != expected_prompt or not isinstance(completion, str) or completion.strip() != str(expected_completion).strip():
            raise ValueError(f"SFT capability row {row_id} prompt/completion lineage mismatch")

        generated_record = generated_by_id[row_id]
        generated = generated_record.get("generated")
        truncated = generated_record.get("truncated")
        if generated_record.get("task") != task or not isinstance(generated, str) or not isinstance(truncated, bool):
            raise ValueError(f"SFT capability generation {row_id} has invalid task/text/truncation fields")
        exact = generated.strip() == completion.strip()
        if generated_record.get("exact") is not exact:
            raise ValueError(f"SFT capability generation {row_id} exact flag mismatch")

        common = {
            "id": row_id,
            "trajectory_id": trajectory_id,
            "task": task,
            "seed": seed,
            "reference": completion.strip(),
            "generated": generated.strip(),
            "exact": exact,
            "empty": not generated.strip(),
            "truncated": truncated,
        }
        if task == "query":
            query_error = None
            ranked: list[dict] = []
            if truncated:
                query_error = "query_truncated"
            else:
                try:
                    parsed_query = parse_search_query(generated)
                except ValueError:
                    query_error = "query_invalid_format"
                else:
                    ranked = FixedBM25Retriever(
                        example["sources"], k1=FIXED_K1, b=FIXED_B,
                    ).search(parsed_query, top_k=FIXED_TOP_K)
            metrics = retrieval_metrics(ranked, str(example["gold_answer"]))
            records.append({
                **common,
                "error_code": query_error,
                "retrieval_metrics": metrics,
                "canonical_ranked_search_results": ranked,
            })
        else:
            prefix = "Answer: " if artifact.get("prompt_version") == SCAFFOLD_PROMPT_VERSION else ""
            raw_response = prefix + generated
            score = canonical_cited_score(
                raw_response,
                str(example["gold_answer"]),
                artifact["canonical_ranked_search_results"],
                truncated=truncated,
            )
            records.append({
                **common,
                "raw_response": raw_response,
                "error_code": "response_truncated" if truncated else score["error_code"],
                "cited_score": score,
            })

    query_records = [record for record in records if record["task"] == "query"]
    response_records = [record for record in records if record["task"] == "response"]
    summary = {
        "rows": len(records),
        "comparison": "canonical_task_revalidation_no_repair",
        "tasks": {
            "query": {
                "rows": len(query_records),
                "exact": sum(record["exact"] for record in query_records),
                "empty": sum(record["empty"] for record in query_records),
                "truncated": sum(record["truncated"] for record in query_records),
                "valid_query": sum(record["error_code"] is None for record in query_records),
                "retrieval_recall@8": sum(
                    record["retrieval_metrics"]["recall@8"] == 1.0 for record in query_records
                ),
            },
            "response": {
                "rows": len(response_records),
                "exact": sum(record["exact"] for record in response_records),
                "empty": sum(record["empty"] for record in response_records),
                "truncated": sum(record["truncated"] for record in response_records),
                "parse_valid": sum(record["cited_score"]["parse_valid"] is True for record in response_records),
                "answer_correct": sum(record["cited_score"]["answer_correct"] is True for record in response_records),
                "correct": sum(record["cited_score"]["correct"] is True for record in response_records),
                "strict_sft_eligible": sum(
                    record["cited_score"]["correct"] is True
                    and record["cited_score"]["lexical_cited_answer_support"] == 1.0
                    and not record["truncated"]
                    for record in response_records
                ),
            },
        },
    }
    return records, summary
