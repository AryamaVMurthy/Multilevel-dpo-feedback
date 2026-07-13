from __future__ import annotations

import re
from collections import Counter


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


def score_cited_response(text: str, gold_answer: str, retrieved_sources: list[dict]) -> dict:
    """Score a strict cited response while keeping model-format errors explicit."""
    from text_feedback_dpo.responses import CitedResponseFormatError, parse_cited_response, validate_retrieved_sources

    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ValueError("gold_answer must be a non-empty string")
    normalized_gold = normalize_answer(gold_answer)
    if not normalized_gold:
        raise ValueError("gold_answer must contain normalized words")
    source_records = validate_retrieved_sources(retrieved_sources)
    answer = _line_content(text, "Answer")
    answer_normalized = normalize_answer(answer)
    result = {
        "parse_valid": False,
        "error_code": None,
        "answer_correct": bool(answer_normalized and answer_normalized == normalized_gold),
        "correct": False,
        "exact_match": float(answer_normalized == normalized_gold),
        "f1": _f1(answer, gold_answer),
        "citation_count": 0,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "cited_answer_support": 0.0,
        "unsupported_source_rate": 0.0,
        "answer_words": len(answer_normalized.split()),
        "reasoning_words": len(_line_content(text, "Reasoning").split()),
        "answer": answer,
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
            "citation_count": len(cited),
            "citation_precision": len(supported) / len(cited),
            "citation_recall": len(supported) / len(answer_bearing) if answer_bearing else 0.0,
            "cited_answer_support": float(bool(supported)),
            "unsupported_source_rate": (len(cited) - len(supported)) / len(cited),
            "answer_words": len(normalize_answer(parsed.answer).split()),
            "reasoning_words": len(parsed.reasoning.split()),
            "answer": parsed.answer,
        }
    )
    result["correct"] = bool(result["answer_correct"] and result["cited_answer_support"])
    return result
