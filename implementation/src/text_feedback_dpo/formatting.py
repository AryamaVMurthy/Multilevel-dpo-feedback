from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass


class XMLFormatError(ValueError):
    """Raised when a model response violates the explicit XML contract."""


@dataclass(frozen=True)
class StudentResponse:
    answer: str
    evidence: str


@dataclass(frozen=True)
class MinimalFeedback:
    error_span: str
    hint: str
    scope: str


def _text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    if child is None or child.text is None or not child.text.strip():
        raise XMLFormatError(f"missing non-empty <{name}> element")
    return child.text.strip()


def _parse(text: str, expected_root: str) -> ET.Element:
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError as exc:
        raise XMLFormatError(f"invalid XML: {exc}") from exc
    if root.tag != expected_root:
        raise XMLFormatError(f"expected root <{expected_root}>, got <{root.tag}>")
    return root


def parse_student_response(text: str) -> StudentResponse:
    root = _parse(text, "response")
    if {child.tag for child in root} != {"answer", "evidence"}:
        raise XMLFormatError("<response> must contain exactly <answer> and <evidence>")
    return StudentResponse(answer=_text(root, "answer"), evidence=_text(root, "evidence"))


def parse_feedback(text: str, *, gold_answer: str) -> MinimalFeedback:
    root = _parse(text, "feedback")
    if {child.tag for child in root} != {"error_span", "hint", "scope"}:
        raise XMLFormatError("<feedback> must contain exactly <error_span>, <hint>, and <scope>")
    feedback = MinimalFeedback(_text(root, "error_span"), _text(root, "hint"), _text(root, "scope"))
    if len(feedback.hint.split()) > 24:
        raise XMLFormatError("hint exceeds 24 words")
    if re.search(re.escape(gold_answer.strip()), feedback.hint, flags=re.IGNORECASE):
        raise XMLFormatError("hint contains the gold answer")
    if feedback.scope not in {"entity", "relation", "evidence", "verification"}:
        raise XMLFormatError("scope must be entity, relation, evidence, or verification")
    return feedback
