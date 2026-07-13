from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence


_CITATION_ID_PATTERN = re.compile(r"(?<!\w)(S\d{3})(?!\w)")
_CITATION_MENTION_PATTERN = re.compile(r"\[(S\d{3})\]")


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_searchqa(response: str, gold_answer: str, packed_evidence: str) -> dict:
    if not isinstance(response, str):
        raise TypeError("SearchQA response must be a string")
    plain_answer = response.strip()
    if not plain_answer:
        return {"correct": False, "exact_match": 0.0, "f1": 0.0, "evidence_support": 0.0, "error_code": "empty_answer", "answer": ""}
    answer = normalize_answer(plain_answer)
    gold = normalize_answer(gold_answer)
    evidence = normalize_answer(packed_evidence)
    exact = float(answer == gold)
    return {
        "correct": bool(exact),
        "exact_match": exact,
        "f1": _f1(plain_answer, gold_answer),
        "evidence_support": float(bool(answer) and answer in evidence),
        "error_code": None,
        "answer": plain_answer,
    }


def _contains_normalized_phrase(source: dict, normalized_gold: str) -> bool:
    target = normalized_gold.split()
    source_tokens = normalize_answer(f"{source['title']} {source['snippet']}").split()
    if not target:
        return False
    return any(source_tokens[index : index + len(target)] == target for index in range(len(source_tokens) - len(target) + 1))


def _line_content(text: object, label: str) -> str:
    if not isinstance(text, str):
        return ""
    expected_line = {"Answer": 0, "Reasoning": 1}.get(label)
    if expected_line is None:
        raise ValueError(f"unsupported response line label: {label}")
    lines = text.splitlines()
    if expected_line >= len(lines):
        return ""
    prefix = f"{label}: "
    line = lines[expected_line]
    return line[len(prefix) :].strip() if line.startswith(prefix) else ""


def _duplicate_rate(values: Sequence[str]) -> float:
    return (len(values) - len(set(values))) / len(values) if values else 0.0


def _raw_citation_metrics(text: object, available_ids: set[str]) -> dict[str, float | int | bool]:
    """Return diagnostics without repairing or accepting a malformed response.

    ``valid_citation_rate`` uses every canonical ID lexically present in the
    Sources line (or reasoning citations when that line is absent) as its
    denominator. ``citation_coverage`` is the fraction of unique reasoning
    citation IDs also listed in Sources; it is zero when reasoning has no IDs.
    Duplicate rate is the maximum repeated-occurrence rate observed in either
    Sources or reasoning citations, so malformed duplicates remain visible.
    """
    if not isinstance(text, str):
        return {
            "citation_count": 0,
            "valid_citation_rate": 0.0,
            "citation_coverage": 0.0,
            "duplicate_citation": False,
            "duplicate_citation_rate": 0.0,
        }
    lines = text.splitlines()
    source_ids: list[str] = []
    if len(lines) >= 3 and lines[2].startswith("Sources: "):
        source_ids = _CITATION_ID_PATTERN.findall(lines[2][len("Sources: ") :])
    reasoning = lines[1][len("Reasoning: ") :] if len(lines) >= 2 and lines[1].startswith("Reasoning: ") else ""
    reasoning_ids = _CITATION_MENTION_PATTERN.findall(reasoning)
    denominator_ids = source_ids if source_ids else reasoning_ids
    valid_count = sum(source_id in available_ids for source_id in denominator_ids)
    unique_reasoning_ids = set(reasoning_ids)
    listed_ids = set(source_ids)
    coverage = (
        sum(source_id in listed_ids for source_id in unique_reasoning_ids) / len(unique_reasoning_ids)
        if unique_reasoning_ids
        else 0.0
    )
    return {
        "citation_count": len(denominator_ids),
        "valid_citation_rate": valid_count / len(denominator_ids) if denominator_ids else 0.0,
        "citation_coverage": coverage,
        "duplicate_citation": bool(_duplicate_rate(source_ids) or _duplicate_rate(reasoning_ids)),
        "duplicate_citation_rate": max(_duplicate_rate(source_ids), _duplicate_rate(reasoning_ids)),
    }


def score_cited_response(
    text: str,
    gold_answer: str,
    retrieved_sources: Sequence[Mapping[str, object]],
    *,
    truncated: bool | None = None,
) -> dict:
    """Score a strict cited response while keeping model-format errors explicit.

    ``correct`` is the primary metric: it requires a valid response grammar
    and answer exact match. ``lexical_cited_answer_support`` is a separate
    title-plus-snippet contiguous-phrase proxy and never changes ``correct``.
    ``truncated`` is returned as supplied; ``None`` means the caller has not
    supplied truncation metadata and is surfaced through ``truncation_known``.
    """
    from text_feedback_dpo.responses import CitedResponseFormatError, parse_cited_response, validate_retrieved_sources

    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ValueError("gold_answer must be a non-empty string")
    normalized_gold = normalize_answer(gold_answer)
    if not normalized_gold:
        raise ValueError("gold_answer must contain normalized words")
    if truncated is not None and not isinstance(truncated, bool):
        raise TypeError("truncated must be a boolean or None")
    source_records = validate_retrieved_sources(retrieved_sources)
    available_ids = {source["source_id"] for source in source_records}
    answer = _line_content(text, "Answer")
    answer_normalized = normalize_answer(answer)
    raw_citation_metrics = _raw_citation_metrics(text, available_ids)
    result = {
        "parse_valid": False,
        "error_code": None,
        "answer_correct": bool(answer_normalized and answer_normalized == normalized_gold),
        "correct": False,
        "malformed_response": True,
        "exact_match": float(answer_normalized == normalized_gold),
        "f1": _f1(answer, gold_answer),
        **raw_citation_metrics,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "lexical_cited_answer_support": 0.0,
        "unsupported_source_rate": 0.0,
        "answer_words": len(answer_normalized.split()),
        "reasoning_words": len(_line_content(text, "Reasoning").split()),
        "answer": answer,
        "truncated": truncated,
        "truncation_known": truncated is not None,
    }
    try:
        parsed = parse_cited_response(text, source_records)
    except CitedResponseFormatError as exc:
        result["error_code"] = exc.error_code
        return result

    cited = [source for source in source_records if source["source_id"] in parsed.source_ids]
    answer_bearing = [source for source in source_records if _contains_normalized_phrase(source, normalized_gold)]
    supported = [source for source in cited if _contains_normalized_phrase(source, normalized_gold)]
    result.update(
        {
            "parse_valid": True,
            "malformed_response": False,
            "citation_count": len(cited),
            "citation_precision": len(supported) / len(cited),
            "citation_recall": len(supported) / len(answer_bearing) if answer_bearing else 0.0,
            "lexical_cited_answer_support": float(bool(supported)),
            "unsupported_source_rate": (len(cited) - len(supported)) / len(cited),
            "answer_words": len(normalize_answer(parsed.answer).split()),
            "reasoning_words": len(parsed.reasoning.split()),
            "answer": parsed.answer,
        }
    )
    result["correct"] = bool(result["parse_valid"] and result["answer_correct"])
    return result
