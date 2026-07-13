from __future__ import annotations

import re
from collections import Counter

from text_feedback_dpo.formatting import XMLFormatError, parse_student_response


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
    try:
        parsed = parse_student_response(response)
    except XMLFormatError as exc:
        return {"correct": False, "exact_match": 0.0, "f1": 0.0, "evidence_support": 0.0, "error_code": "invalid_xml", "error": str(exc)}
    answer = normalize_answer(parsed.answer)
    gold = normalize_answer(gold_answer)
    evidence = normalize_answer(packed_evidence)
    exact = float(answer == gold)
    return {"correct": bool(exact), "exact_match": exact, "f1": _f1(parsed.answer, gold_answer), "evidence_support": float(bool(answer) and answer in evidence), "error_code": None, "answer": parsed.answer, "evidence": parsed.evidence}
