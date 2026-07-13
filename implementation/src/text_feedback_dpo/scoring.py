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
