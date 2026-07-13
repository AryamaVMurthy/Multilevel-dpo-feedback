from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import re
import time
from typing import Any

from text_feedback_dpo.prompts import build_cited_response_prompt, build_search_query_prompt
from text_feedback_dpo.responses import parse_cited_response, render_cited_response
from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics, tokenize_query
from text_feedback_dpo.searchqa import SOURCE_SCHEMA_VERSION
from text_feedback_dpo.scoring import score_cited_response


FIXED_TOP_K = 8
FIXED_K1 = 1.2
FIXED_B = 0.75
PROMPT_VERSION = "fixed-retrieval-cited-v1"
RESPONSE_SCHEMA_VERSION = 1
MAX_QUERY_TOKENS = 16
_TAG_PATTERN = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
_RESPONSE_LABEL_PATTERN = re.compile(r"^(?:answer|reasoning|sources|search\s+query)\s*:", re.IGNORECASE)


def generate_batch(provider: Callable[..., list[str]], prompts: list[str], **generation_kwargs: object) -> list[dict]:
    if not prompts:
        return []
    outputs = provider(prompts, **generation_kwargs)
    if not isinstance(outputs, list) or len(outputs) != len(prompts):
        raise ValueError(f"batch generation cardinality mismatch: expected {len(prompts)}, got {len(outputs) if isinstance(outputs, list) else type(outputs).__name__}")
    return [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, outputs, strict=True)]


def _hash(value: Any, *, context: str = "structured input") -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} is not JSON-serializable: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _zero_cited_score(error_code: str, *, truncated: bool) -> dict:
    return {
        "parse_valid": False,
        "error_code": error_code,
        "answer_correct": False,
        "correct": False,
        "malformed_response": True,
        "exact_match": 0.0,
        "f1": 0.0,
        "answer_capability_exact_match": 0.0,
        "answer_capability_f1": 0.0,
        "protocol_exact_match": 0.0,
        "protocol_f1": 0.0,
        "citation_count": 0,
        "valid_citation_rate": 0.0,
        "citation_coverage": 0.0,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "lexical_cited_answer_support": 0.0,
        "unsupported_source_rate": 0.0,
        "answer_words": 0,
        "reasoning_words": 0,
        "answer": "",
        "truncated": truncated,
        "truncation_known": True,
    }


def _record_output(value: Any) -> tuple[str, bool, str | None, bool | None]:
    if isinstance(value, str):
        raise ValueError("active-search batch provider output requires explicit text and truncated metadata")
    if isinstance(value, dict):
        text = value.get("text", value.get("response"))
        truncated = value.get("truncated")
        scratchpad = value.get("scratchpad", value.get("private_scratchpad"))
        scratchpad_truncated = value.get("scratchpad_truncated", value.get("private_scratchpad_truncated"))
    else:
        text = getattr(value, "text", getattr(value, "response", None))
        truncated = getattr(value, "truncated", None)
        scratchpad = getattr(value, "scratchpad", None)
        scratchpad_truncated = getattr(value, "scratchpad_truncated", None)
    if not isinstance(text, str):
        raise ValueError("batch provider output must contain text or response string")
    if not isinstance(truncated, bool):
        raise ValueError("batch provider output requires explicit boolean truncated metadata")
    if scratchpad is not None and not isinstance(scratchpad, str):
        raise ValueError("batch provider scratchpad must be a string or null")
    if scratchpad_truncated is not None and not isinstance(scratchpad_truncated, bool):
        raise ValueError("batch provider scratchpad_truncated must be boolean or null")
    return text, truncated, scratchpad, scratchpad_truncated


def _call_batch(provider: Callable[..., list[Any]], prompts: list[str], *, batch_size: int) -> list[tuple[str, bool, str | None, bool | None]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    records: list[tuple[str, bool, str | None, bool | None]] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        outputs = provider(chunk)
        if not isinstance(outputs, list) or len(outputs) != len(chunk):
            actual = len(outputs) if isinstance(outputs, list) else type(outputs).__name__
            raise ValueError(f"batch generation cardinality mismatch: expected {len(chunk)}, got {actual}")
        records.extend(_record_output(output) for output in outputs)
    return records


def _validate_rows(rows: list[dict]) -> None:
    if not isinstance(rows, list):
        raise TypeError("SearchQA rows must be a list")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"SearchQA row {index} must be a mapping")
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"SearchQA row {index} requires a non-empty string id")
        if example_id in seen:
            raise ValueError(f"SearchQA row ids must be unique: {example_id}")
        seen.add(example_id)
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"SearchQA row {example_id} requires a non-empty question")
        if not isinstance(row.get("gold_answer"), str) or not row["gold_answer"].strip():
            raise ValueError(f"SearchQA row {example_id} requires a non-empty gold_answer")
        if "sources" not in row:
            raise ValueError(f"SearchQA row {example_id} requires structured sources")


def parse_search_query(text: str) -> str:
    """Validate the model's one-line retrieval query without guessing its semantics."""
    if not isinstance(text, str):
        raise ValueError("query_invalid_format: query must be a string")
    stripped = text.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise ValueError("query_invalid_format: query must be exactly one nonempty line")
    query_tokens = tokenize_query(stripped)
    if not query_tokens:
        raise ValueError("query_invalid_format: query must contain searchable tokens")
    if len(query_tokens) > MAX_QUERY_TOKENS:
        raise ValueError(f"query_invalid_format: query exceeds {MAX_QUERY_TOKENS} searchable tokens")
    if "```" in stripped:
        raise ValueError("query_invalid_format: code fences are forbidden")
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("query_invalid_format: JSON is forbidden")
    if _TAG_PATTERN.search(stripped):
        raise ValueError("query_invalid_format: tag-like XML is forbidden")
    if _RESPONSE_LABEL_PATTERN.match(stripped):
        raise ValueError("query_invalid_format: response labels are forbidden")
    return stripped


def _query_error(text: str) -> str | None:
    try:
        parse_search_query(text)
    except ValueError:
        return "query_invalid_format"
    return None


def _add_protocol_metrics(score: dict, *, protocol_valid: bool) -> dict:
    capability_exact = float(score.get("exact_match", 0.0))
    capability_f1 = float(score.get("f1", 0.0))
    return {
        **score,
        "answer_capability_exact_match": capability_exact,
        "answer_capability_f1": capability_f1,
        "protocol_exact_match": capability_exact if protocol_valid else 0.0,
        "protocol_f1": capability_f1 if protocol_valid else 0.0,
        "correct": bool(score.get("correct", False) and protocol_valid),
    }


def run_fixed_retrieval_pipeline(
    rows: list[dict],
    *,
    query_generate_batch: Callable[..., list[Any]],
    response_generate_batch: Callable[..., list[Any]],
    query_batch_size: int = 4,
    response_batch_size: int = 4,
    top_k: int = FIXED_TOP_K,
    k1: float = FIXED_K1,
    b: float = FIXED_B,
    policy_hash: str,
    prompt_version: str = PROMPT_VERSION,
    response_schema_version: int = RESPONSE_SCHEMA_VERSION,
) -> list[dict]:
    """Run structured SearchQA query, fixed retrieval, and cited-response generation."""
    _validate_rows(rows)
    if top_k != FIXED_TOP_K or k1 != FIXED_K1 or b != FIXED_B:
        raise ValueError("active SearchQA retrieval is frozen at top_k=8, k1=1.2, b=0.75")
    if not isinstance(policy_hash, str) or not policy_hash.strip():
        raise ValueError("policy_hash must be a non-empty string")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be a non-empty string")
    if response_schema_version != RESPONSE_SCHEMA_VERSION:
        raise ValueError(f"response_schema_version must be {RESPONSE_SCHEMA_VERSION}")

    pipeline_started = time.perf_counter_ns()
    gold_by_id = {row["id"]: row["gold_answer"] for row in rows}
    query_prompts = [build_search_query_prompt(row, []) for row in rows]
    query_prompt_hashes = [_hash(prompt, context=f"query prompt at index {index}") for index, prompt in enumerate(query_prompts)]
    query_started = time.perf_counter_ns()
    query_records = _call_batch(query_generate_batch, query_prompts, batch_size=query_batch_size)
    query_elapsed_ms = (time.perf_counter_ns() - query_started) / 1_000_000

    results: list[dict] = []
    active: list[dict] = []
    for row, prompt, prompt_hash, (raw_query, query_truncated, scratchpad, scratchpad_truncated) in zip(
        rows, query_prompts, query_prompt_hashes, query_records, strict=True
    ):
        artifact = {
            "id": row["id"],
            "raw_query": raw_query,
            "ranked_search_results": [],
            "raw_response": None,
            "parsed_response": None,
            "rendered_visible_response": None,
            "prompt_version": prompt_version,
            "response_schema_version": response_schema_version,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "schema_version": response_schema_version,
            "policy_hash": policy_hash,
            "input_hash": _hash(
                {"id": row["id"], "question": row["question"], "sources": row["sources"]},
                context=f"input row {row['id']}",
            ),
            "query_prompt_hash": prompt_hash,
            "response_prompt_hash": None,
            "query_truncated": query_truncated,
            "response_truncated": False,
            "truncated": query_truncated,
            "truncation": {"query": query_truncated, "response": False},
            "private_scratchpad": {"query": scratchpad} if scratchpad is not None else None,
            "private_scratchpad_truncated": {"query": scratchpad_truncated} if scratchpad is not None else None,
            "retrieval_metrics": None,
            "cited_score": None,
            "error_code": None,
            "timings_ms": {
                "query_generation_batch_wall_ms": query_elapsed_ms,
                "query_generation_amortized_per_item_ms": query_elapsed_ms / len(rows) if rows else 0.0,
                "retrieval_individual_ms": 0.0,
                "response_generation_batch_wall_ms": 0.0,
                "response_generation_amortized_per_item_ms": 0.0,
                "pipeline_wall_ms": 0.0,
            },
        }
        if query_truncated:
            artifact["error_code"] = "query_truncated"
            artifact["cited_score"] = _zero_cited_score("query_truncated", truncated=True)
            artifact["retrieval_metrics"] = retrieval_metrics([], row["gold_answer"])
            results.append(artifact)
            continue
        query_error = _query_error(raw_query)
        if query_error:
            artifact["error_code"] = query_error
            artifact["cited_score"] = _zero_cited_score(query_error, truncated=False)
            artifact["retrieval_metrics"] = retrieval_metrics([], row["gold_answer"])
            results.append(artifact)
            continue

        normalized_query = parse_search_query(raw_query)
        retrieval_started = time.perf_counter_ns()
        retriever = FixedBM25Retriever(row["sources"], k1=k1, b=b)
        ranked = retriever.search(normalized_query, top_k=top_k)
        artifact["ranked_search_results"] = ranked
        artifact["retrieval_metrics"] = retrieval_metrics(ranked, row["gold_answer"])
        artifact["timings_ms"]["retrieval_individual_ms"] = (time.perf_counter_ns() - retrieval_started) / 1_000_000
        response_prompt = build_cited_response_prompt(row, ranked, [])
        artifact["response_prompt_hash"] = _hash(response_prompt, context=f"response prompt for {row['id']}")
        artifact["response_prompt"] = response_prompt
        active.append(artifact)
        results.append(artifact)

    response_prompts = [artifact["response_prompt"] for artifact in active]
    response_started = time.perf_counter_ns()
    response_records = _call_batch(response_generate_batch, response_prompts, batch_size=response_batch_size) if response_prompts else []
    response_elapsed_ms = (time.perf_counter_ns() - response_started) / 1_000_000
    for artifact, (raw_response, response_truncated, scratchpad, scratchpad_truncated) in zip(active, response_records, strict=True):
        artifact["raw_response"] = raw_response
        artifact["response_truncated"] = response_truncated
        artifact["truncated"] = bool(artifact["query_truncated"] or response_truncated)
        artifact["truncation"]["response"] = response_truncated
        artifact["timings_ms"]["response_generation_batch_wall_ms"] = response_elapsed_ms
        artifact["timings_ms"]["response_generation_amortized_per_item_ms"] = response_elapsed_ms / len(active) if active else 0.0
        if scratchpad is not None:
            if artifact["private_scratchpad"] is None:
                artifact["private_scratchpad"] = {}
            if artifact["private_scratchpad_truncated"] is None:
                artifact["private_scratchpad_truncated"] = {}
            artifact["private_scratchpad"]["response"] = scratchpad
            artifact["private_scratchpad_truncated"]["response"] = scratchpad_truncated
        score = score_cited_response(
            raw_response,
            gold_by_id[artifact["id"]],
            artifact["ranked_search_results"],
            truncated=response_truncated,
        )
        artifact["cited_score"] = _add_protocol_metrics(score, protocol_valid=bool(score["parse_valid"] and not response_truncated))
        if response_truncated:
            artifact["error_code"] = "response_truncated"
            artifact["cited_score"] = {
                **artifact["cited_score"],
                "parse_valid": False,
                "correct": False,
                "malformed_response": True,
                "error_code": "response_truncated",
                "lexical_cited_answer_support": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
            }
            continue
        if not score["parse_valid"]:
            artifact["error_code"] = score["error_code"]
            continue
        parsed = parse_cited_response(raw_response, artifact["ranked_search_results"])
        artifact["parsed_response"] = {"answer": parsed.answer, "reasoning": parsed.reasoning, "source_ids": list(parsed.source_ids)}
        artifact["rendered_visible_response"] = render_cited_response(parsed, artifact["ranked_search_results"])

    pipeline_wall_ms = (time.perf_counter_ns() - pipeline_started) / 1_000_000
    for artifact in results:
        artifact["timings_ms"]["pipeline_wall_ms"] = pipeline_wall_ms
        if artifact["id"] not in gold_by_id:
            raise RuntimeError("internal active-search ID mismatch")
    return results
