"""Domain-specific answer checks used alongside model-based judgments.

These checks deliberately evaluate the answer field supplied by the evaluator role, not the
student's full reasoning trace. Cases that cannot be decided safely are returned with an explicit
``requires_model_judgment`` flag so callers can route them to the evaluator rather than guessing.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?%?(?![A-Za-z0-9_])"
)
_ALTERNATIVE_RE = re.compile(r"\b(?:or|and/or)\b", re.IGNORECASE)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _numeric_values(value: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in _NUMBER_RE.finditer(value):
        token = match.group(0).replace(",", "")
        if token.endswith("%"):
            token = token[:-1]
        try:
            values.append(Decimal(token))
        except InvalidOperation as exc:
            raise ValueError(f"could not parse numeric answer token: {match.group(0)!r}") from exc
    return values


def _numeric_result(*, prediction: str, gold_answer: str, values: list[Decimal], gold_values: list[Decimal]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evaluator_source": "deterministic_numeric",
        "extracted_answer": prediction,
        "gold_answer": gold_answer,
        "numeric_values": [str(value) for value in values],
        "numeric_gold_values": [str(value) for value in gold_values],
        "numeric_exact_match": False,
        "correct": False,
        "confidence": 0.0,
        "ambiguous": False,
        "requires_model_judgment": False,
        "error_code": None,
    }
    if not gold_values:
        base.update(
            error_code="invalid_gold_numeric_answer",
            requires_model_judgment=True,
        )
        return base
    if not values:
        base["error_code"] = "missing_numeric_answer"
        return base
    if len(values) != 1:
        base.update(
            ambiguous=True,
            requires_model_judgment=True,
            error_code="ambiguous_numeric_answer",
            confidence=0.5,
        )
        return base
    exact = values[0] == gold_values[-1]
    base.update(
        numeric_exact_match=exact,
        correct=exact,
        confidence=1.0 if exact else 0.0,
    )
    return base


def evaluate_gsm8k_answer(prediction: str, gold_answer: str) -> dict[str, Any]:
    """Evaluate one already-extracted GSM8K answer using exact Decimal equivalence."""

    prediction = _require_text(prediction, "prediction")
    gold_answer = _require_text(gold_answer, "gold_answer")
    return _numeric_result(
        prediction=prediction,
        gold_answer=gold_answer,
        values=_numeric_values(prediction),
        gold_values=_numeric_values(gold_answer),
    )


def _normalize_search_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _normalize_search_text(prediction).split()
    reference_tokens = _normalize_search_text(reference).split()
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _contains_answer(text: str, answer: str) -> bool:
    normalized_text = _normalize_search_text(text)
    normalized_answer = _normalize_search_text(answer)
    if not normalized_answer:
        return False
    return normalized_answer in normalized_text


def _evidence_support(answer: str, evidence: Iterable[str]) -> bool:
    # Evidence support belongs to the submitted answer, not to the gold answer. Otherwise an
    # unrelated prediction would inherit support merely because the reference appears in context.
    return any(_contains_answer(item, answer) for item in evidence)


def evaluate_searchqa_answer(
    prediction: str,
    *,
    gold_answer: str,
    answer_aliases: list[str],
    expected_answer_type: str,
    actual_answer_type: str,
    evidence: list[str],
) -> dict[str, Any]:
    """Evaluate SearchQA answer, type, and support without collapsing uncertainty."""

    prediction = _require_text(prediction, "prediction")
    gold_answer = _require_text(gold_answer, "gold_answer")
    if not isinstance(answer_aliases, list) or not answer_aliases or not all(
        isinstance(item, str) and item.strip() for item in answer_aliases
    ):
        raise ValueError("answer_aliases must be a non-empty list of strings")
    if not isinstance(expected_answer_type, str) or not expected_answer_type.strip():
        raise ValueError("expected_answer_type must be a non-empty string")
    if not isinstance(actual_answer_type, str) or not actual_answer_type.strip():
        raise ValueError("actual_answer_type must be a non-empty string")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) and item.strip() for item in evidence):
        raise ValueError("evidence must be a non-empty list of strings")

    aliases = [gold_answer, *answer_aliases]
    normalized_prediction = _normalize_search_text(prediction)
    normalized_aliases = {_normalize_search_text(alias) for alias in aliases}
    exact_match = normalized_prediction in normalized_aliases
    token_f1 = max(_token_f1(prediction, alias) for alias in aliases)
    ambiguous = bool(_ALTERNATIVE_RE.search(prediction))
    answer_type_correct: bool | None
    if expected_answer_type.casefold() == "unknown" or actual_answer_type.casefold() == "unknown":
        answer_type_correct = None
    else:
        answer_type_correct = expected_answer_type.casefold() == actual_answer_type.casefold()
    evidence_supported = _evidence_support(prediction, evidence)
    requires_model_judgment = ambiguous
    error_code = "ambiguous_answer" if ambiguous else None
    correct = exact_match and not ambiguous
    return {
        "evaluator_source": "deterministic_searchqa",
        "extracted_answer": prediction,
        "gold_answer": gold_answer,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "answer_type_correct": answer_type_correct,
        "evidence_supported": evidence_supported,
        "ambiguous": ambiguous,
        "requires_model_judgment": requires_model_judgment,
        "correct": correct,
        "confidence": 1.0 if correct else (0.5 if requires_model_judgment else 0.0),
        "error_code": error_code,
    }


def evaluate_domain_answer(
    *,
    domain: str,
    prediction: str,
    example: dict[str, Any],
    actual_answer_type: str | None = None,
    evidence_supported: bool | None = None,
) -> dict[str, Any]:
    """Dispatch answer-only evaluation for native and model-backed evaluator paths."""

    if domain == "math":
        return evaluate_gsm8k_answer(prediction, str(example["gold_answer"]))
    if domain == "search_qa":
        result = evaluate_searchqa_answer(
            prediction,
            gold_answer=str(example["gold_answer"]),
            answer_aliases=list(example.get("answer_aliases", example.get("answers", [example["gold_answer"]]))),
            expected_answer_type=str(example.get("answer_type", "unknown")),
            actual_answer_type=str(actual_answer_type or "unknown"),
            evidence=list(example.get("evidence", [])),
        )
        if evidence_supported is not None:
            result["model_evidence_supported"] = bool(evidence_supported)
        return result
    raise ValueError(f"unsupported evaluation domain: {domain}")
