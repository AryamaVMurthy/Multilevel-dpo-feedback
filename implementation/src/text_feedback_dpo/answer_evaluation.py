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
_MATH_SAFE_RE = re.compile(r"^[0-9A-Za-z\s+\-*/^=().,\[\]{}]+$")
_MATH_UNIT_RE = re.compile(r"^(.*?)(?:\s+|\\\\text\{)([A-Za-z]+)\}?$")
_MATH_INTERVAL_RE = re.compile(r"^([\[(])(.+),(.+)([\])])$")
_MATH_CHOICE_MARKER_RE = re.compile(r"\\textbf\{\(([A-E])\)\}")


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


def _last_boxed(value: str) -> str:
    marker = "\\boxed{"
    start = value.rfind(marker)
    if start < 0:
        return value.strip()
    index = start + len(marker)
    depth = 1
    for end in range(index, len(value)):
        if value[end] == "{":
            depth += 1
        elif value[end] == "}":
            depth -= 1
            if depth == 0:
                return value[index:end].strip()
    raise ValueError("unbalanced boxed MATH answer")


def _braced_group(value: str, start: int) -> tuple[str, int] | None:
    if start >= len(value) or value[start] != "{":
        return None
    depth = 1
    for index in range(start + 1, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    return None


def _replace_grouped_latex_math(value: str) -> str:
    output: list[str] = []
    index = 0
    fraction_macros = ("\\dfrac", "\\tfrac", "\\frac")
    while index < len(value):
        macro = next((name for name in fraction_macros if value.startswith(name, index)), None)
        if macro is not None:
            numerator = _braced_group(value, index + len(macro))
            if numerator is not None:
                denominator = _braced_group(value, numerator[1])
                if denominator is not None:
                    output.append(
                        f"(({_replace_grouped_latex_math(numerator[0])})/"
                        f"({_replace_grouped_latex_math(denominator[0])}))"
                    )
                    index = denominator[1]
                    continue
        if value.startswith("\\sqrt", index):
            radicand = _braced_group(value, index + len("\\sqrt"))
            if radicand is not None:
                output.append(f"sqrt({_replace_grouped_latex_math(radicand[0])})")
                index = radicand[1]
                continue
        output.append(value[index])
        index += 1
    return "".join(output)


def _latex_math(value: str) -> str:
    value = _last_boxed(value).strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\pi", "pi")
    value = value.replace("\\infty", "oo")
    value = value.replace("\\{", "{").replace("\\}", "}")
    for spacing in ("\\!", "\\,", "\\;", "\\:", "\\quad", "\\qquad"):
        value = value.replace(spacing, "")
    value = re.sub(r"\\text\{([^{}]+)\}", r" \1", value)
    value = _replace_grouped_latex_math(value)
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    return " ".join(value.split())


def _sympy_expression(value: str) -> Any | None:
    normalized = _latex_math(value)
    if not normalized or "__" in normalized or not _MATH_SAFE_RE.fullmatch(normalized):
        return None
    try:
        import sympy
    except ImportError as exc:
        raise ImportError("sympy is required for deterministic MATH answer evaluation") from exc
    symbols = {letter: sympy.Symbol(letter) for letter in "abcdefghijklmnopqrstuvwxyz"}
    locals_map = {**symbols, "pi": sympy.pi, "oo": sympy.oo, "sqrt": sympy.sqrt}
    try:
        return sympy.sympify(normalized, locals=locals_map, evaluate=True)
    except (sympy.SympifyError, TypeError, ValueError, SyntaxError):
        return None


def _unit_parts(value: str) -> tuple[str, str | None]:
    normalized = _latex_math(value)
    match = _MATH_UNIT_RE.fullmatch(normalized)
    if match is None:
        return normalized, None
    expression, unit = match.groups()
    expression = expression.strip()
    if not expression or not any(character.isdigit() for character in expression):
        return normalized, None
    return expression, unit.casefold()


def _math_interval(value: str) -> tuple[str, Any, Any, str] | None:
    normalized = _latex_math(value).replace(" ", "")
    match = _MATH_INTERVAL_RE.fullmatch(normalized)
    if match is None:
        return None
    left, start, end, right = match.groups()
    start_expression = _sympy_expression(start)
    end_expression = _sympy_expression(end)
    if start_expression is None or end_expression is None:
        return None
    try:
        import sympy
    except ImportError as exc:
        raise ImportError("sympy is required for deterministic MATH answer evaluation") from exc
    return left, sympy.nsimplify(start_expression, rational=True), sympy.nsimplify(end_expression, rational=True), right


def _math_set(value: str) -> set[str] | None:
    raw = _last_boxed(value).strip()
    if raw.startswith("\\{") and raw.endswith("\\}"):
        contents = raw[2:-2]
    elif raw.startswith("{") and raw.endswith("}"):
        contents = raw[1:-1]
    else:
        return None
    if "," not in contents:
        return None
    values = [part.strip() for part in contents.split(",")]
    if not all(values):
        return None
    canonical: set[str] = set()
    for item in values:
        expression = _sympy_expression(item)
        if expression is None:
            return None
        try:
            import sympy
        except ImportError as exc:
            raise ImportError("sympy is required for deterministic MATH answer evaluation") from exc
        canonical.add(str(sympy.nsimplify(expression, rational=True)))
    return canonical


def evaluate_math_answer(prediction: str, gold_answer: str) -> dict[str, Any]:
    """Evaluate safe numeric, symbolic, set, interval, and unit MATH answers.

    Unsupported or ambiguous formats deliberately route to the model evaluator instead of
    being heuristically accepted as correct.
    """

    prediction = _require_text(prediction, "prediction")
    gold_answer = _require_text(gold_answer, "gold_answer")
    base: dict[str, Any] = {
        "evaluator_source": "deterministic_math",
        "extracted_answer": prediction,
        "gold_answer": gold_answer,
        "correct": False,
        "confidence": 0.0,
        "ambiguous": False,
        "requires_model_judgment": False,
        "error_code": None,
    }
    if _ALTERNATIVE_RE.search(prediction) or _ALTERNATIVE_RE.search(gold_answer):
        return {
            **base,
            "ambiguous": True,
            "requires_model_judgment": True,
            "confidence": 0.5,
            "error_code": "alternative_answer",
        }
    prediction_set = _math_set(prediction)
    gold_set = _math_set(gold_answer)
    if prediction_set is not None or gold_set is not None:
        if prediction_set is None or gold_set is None:
            return {**base, "requires_model_judgment": True, "confidence": 0.5, "error_code": "set_parse_failure"}
        correct = prediction_set == gold_set
        return {**base, "correct": correct, "confidence": 1.0 if correct else 0.0, "normalized_prediction": sorted(prediction_set), "normalized_gold": sorted(gold_set)}
    prediction_interval = _math_interval(prediction)
    gold_interval = _math_interval(gold_answer)
    if prediction_interval is not None or gold_interval is not None:
        if prediction_interval is None or gold_interval is None:
            return {**base, "requires_model_judgment": True, "confidence": 0.5, "error_code": "interval_parse_failure"}
        correct = prediction_interval == gold_interval
        return {**base, "correct": correct, "confidence": 1.0 if correct else 0.0, "normalized_prediction": str(prediction_interval), "normalized_gold": str(gold_interval)}
    prediction_expression, prediction_unit = _unit_parts(prediction)
    gold_expression, gold_unit = _unit_parts(gold_answer)
    if prediction_unit != gold_unit:
        return {**base, "requires_model_judgment": True, "confidence": 0.5, "error_code": "unit_mismatch_or_ambiguity"}
    left = _sympy_expression(prediction_expression)
    right = _sympy_expression(gold_expression)
    if left is None or right is None:
        return {**base, "requires_model_judgment": True, "confidence": 0.5, "error_code": "symbolic_parse_failure"}
    try:
        import sympy

        correct = bool(sympy.simplify(left - right) == 0)
    except (TypeError, ValueError):
        return {**base, "requires_model_judgment": True, "confidence": 0.5, "error_code": "symbolic_comparison_failure"}
    return {
        **base,
        "correct": correct,
        "confidence": 1.0 if correct else 0.0,
        "normalized_prediction": str(left),
        "normalized_gold": str(right),
        "unit": prediction_unit,
    }


def _official_math_choice_answer(prediction: str, problem: str) -> tuple[str, str] | None:
    selected = _last_boxed(prediction).strip().upper()
    if re.fullmatch(r"[A-E]", selected) is None:
        return None
    markers = list(_MATH_CHOICE_MARKER_RE.finditer(problem))
    if not markers:
        return None
    choices: dict[str, str] = {}
    for index, marker in enumerate(markers):
        label = marker.group(1)
        if label in choices:
            raise ValueError(f"official MATH problem contains duplicate choice label: {label}")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(problem)
        raw_answer = problem[marker.end() : end].strip().strip("$").strip()
        mapped_answer = _latex_math(raw_answer).strip().strip("$").strip()
        if not mapped_answer:
            raise ValueError(f"official MATH problem contains an empty choice: {label}")
        choices[label] = mapped_answer
    if selected not in choices:
        raise ValueError(f"selected official MATH choice is missing from problem: {selected}")
    return selected, choices[selected]


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
        if example.get("source") == "EleutherAI/hendrycks_math" or "source_subject" in example:
            choice = _official_math_choice_answer(prediction, str(example.get("problem", "")))
            if choice is not None:
                selected, mapped_answer = choice
                result = evaluate_math_answer(mapped_answer, str(example["gold_answer"]))
                return {
                    **result,
                    "evaluator_source": "deterministic_math_multiple_choice",
                    "extracted_answer": prediction,
                    "selected_choice": selected,
                    "mapped_answer": mapped_answer,
                }
            return evaluate_math_answer(prediction, str(example["gold_answer"]))
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
