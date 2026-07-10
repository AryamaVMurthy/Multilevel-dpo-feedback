from __future__ import annotations

import re
from collections.abc import Iterable


_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
}
_EXPLICIT_OPERATIONS = {
    "add", "addition", "subtract", "subtraction", "multiply", "multiplication", "divide",
    "division", "plus", "minus", "times", "quotient", "product", "sum", "difference",
    "equation", "equations", "equals", "equal",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", text)


def _normalized_tokens(text: str) -> list[str]:
    return [word.lower().replace("’", "'") for word in _words(text)]


def _sentence_count(text: str) -> int:
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", text.strip()) if chunk.strip()]
    return len(chunks)


def _copied_evidence(hint_tokens: list[str], evidence: Iterable[str]) -> bool:
    if len(hint_tokens) < 3:
        return False
    evidence_tokens = [_normalized_tokens(item) for item in evidence]
    for source_tokens in evidence_tokens:
        if len(source_tokens) < 3:
            continue
        for index in range(len(hint_tokens) - 2):
            phrase = hint_tokens[index : index + 3]
            if any(source_tokens[offset : offset + 3] == phrase for offset in range(len(source_tokens) - 2)):
                return True
    return False


def _validate(
    guidance: str,
    *,
    problem: str,
    gold_answer: str,
    evidence: Iterable[str],
    enforce_length: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    stripped = guidance.strip()
    words = _words(stripped)
    tokens = [word.lower() for word in words]
    if not stripped:
        reasons.append("empty")
    if enforce_length:
        if not 8 <= len(words) <= 15:
            reasons.append("word_count")
        if _sentence_count(stripped) != 1:
            reasons.append("sentence_count")
    if re.search(r"\d", stripped):
        reasons.append("digits")
    if re.search(r"[=+*/<>]", stripped):
        reasons.append("equation_symbol")
    if any(token.strip("'’-") in _NUMBER_WORDS for token in tokens):
        reasons.append("quantities")
    if any(token.strip("'’-") in _EXPLICIT_OPERATIONS for token in tokens):
        reasons.append("explicit_operation")
    word_matches = list(re.finditer(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", stripped))
    for index, match in enumerate(word_matches):
        if index == 0 or not match.group(0)[:1].isupper():
            continue
        previous_text = stripped[: match.start()].rstrip()
        if previous_text.endswith((".", "!", "?")):
            continue
        reasons.append("proper_noun")
        break
    answer_tokens = _normalized_tokens(gold_answer)
    hint_tokens = _normalized_tokens(stripped)
    if answer_tokens and len(answer_tokens) >= 1:
        answer_phrase = " ".join(answer_tokens)
        if answer_phrase and answer_phrase in " ".join(hint_tokens):
            reasons.append("answer_disclosure")
        if len(answer_tokens) == 1 and len(answer_tokens[0]) == 1 and answer_tokens[0] in hint_tokens:
            reasons.append("answer_initial")
    if _copied_evidence(hint_tokens, evidence):
        reasons.append("copied_evidence")
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "word_count": len(words),
        "sentence_count": _sentence_count(stripped),
        "guidance": guidance,
        "problem_word_count": len(_words(problem)),
    }


def validate_guidance_surface(
    guidance: str,
    *,
    problem: str,
    gold_answer: str,
    evidence: Iterable[str] = (),
) -> dict[str, object]:
    return _validate(
        guidance,
        problem=problem,
        gold_answer=gold_answer,
        evidence=evidence,
        enforce_length=True,
    )


def validate_accumulated_guidance(
    guidance_history: list[str],
    *,
    problem: str,
    gold_answer: str,
    evidence: Iterable[str] = (),
) -> dict[str, object]:
    if not guidance_history:
        return {"valid": False, "reasons": ["empty_history"], "hint_count": 0, "accumulated": ""}
    individual = [
        validate_guidance_surface(
            guidance,
            problem=problem,
            gold_answer=gold_answer,
            evidence=evidence,
        )
        for guidance in guidance_history
    ]
    accumulated = " ".join(guidance_history)
    aggregate = _validate(
        accumulated,
        problem=problem,
        gold_answer=gold_answer,
        evidence=evidence,
        enforce_length=False,
    )
    reasons = sorted({reason for result in individual for reason in result["reasons"]} | set(aggregate["reasons"]))
    return {
        "valid": all(bool(result["valid"]) for result in individual) and not aggregate["reasons"],
        "reasons": reasons,
        "hint_count": len(guidance_history),
        "individual": individual,
        "accumulated": accumulated,
        "aggregate": aggregate,
    }
