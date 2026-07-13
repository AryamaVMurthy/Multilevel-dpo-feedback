from __future__ import annotations

import json
from dataclasses import dataclass

from text_feedback_dpo.scoring import normalize_answer


class FeedbackFormatError(ValueError):
    """Raised when teacher feedback violates the minimal-hint contract."""


@dataclass(frozen=True)
class MinimalFeedback:
    hint: str


def parse_feedback(text: str, *, gold_answer: str) -> MinimalFeedback:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FeedbackFormatError(f"invalid JSON feedback: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"hint"}:
        raise FeedbackFormatError("feedback must contain exactly one field: hint")
    hint = payload["hint"]
    if not isinstance(hint, str) or not hint.strip():
        raise FeedbackFormatError("hint must be a non-empty string")
    hint = hint.strip()
    if len(hint.split()) > 24:
        raise FeedbackFormatError("hint exceeds 24 words")
    normalized_gold = normalize_answer(gold_answer)
    normalized_hint = normalize_answer(hint)
    if normalized_gold and normalized_gold in normalized_hint:
        raise FeedbackFormatError("hint contains the gold answer")
    return MinimalFeedback(hint=hint)
