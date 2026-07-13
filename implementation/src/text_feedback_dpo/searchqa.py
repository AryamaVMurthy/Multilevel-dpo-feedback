from __future__ import annotations

from collections.abc import Callable, Iterable


def _snippets(row: dict) -> list[str]:
    """Normalize official SearchQA result records without fabricating evidence.

    The official archive contains a sequence of result records, and individual
    records may have an empty snippet. Empty records are excluded; a row with
    no usable snippet fails explicitly below.
    """
    value = row.get("search_results", row.get("snippets"))
    if value is None and isinstance(row.get("context"), str):
        value = [row["context"]]
    if isinstance(value, dict):
        value = value.get("snippets")
    if not isinstance(value, list) or not value:
        raise ValueError("SearchQA row requires a non-empty snippets/search_results list")
    normalized = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("snippet"), str) and item["snippet"].strip():
            normalized.append(item["snippet"].strip())
        elif item in (None, "") or isinstance(item, dict):
            continue
        else:
            raise ValueError(f"unsupported SearchQA snippet record type: {type(item).__name__}")
    if not normalized:
        raise ValueError("SearchQA row has no usable non-empty evidence snippets")
    return normalized


def materialize_row(row: dict, *, split: str, index: int) -> dict:
    question = row.get("question")
    answer = row.get("answer", row.get("gold_answer"))
    if answer is None and isinstance(row.get("answers"), list) and row["answers"]:
        answer = row["answers"][0]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("SearchQA row requires non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("SearchQA row requires non-empty answer")
    snippets = _snippets(row)
    return {"id": f"{split}-{index}", "question": question.strip(), "gold_answer": answer.strip(), "snippets": snippets}


def pack_evidence(snippets: Iterable[str], *, max_tokens: int, token_count: Callable[[str], int]) -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    selected: list[str] = []
    used = 0
    for snippet in snippets:
        cost = token_count(snippet)
        if cost < 0:
            raise ValueError("token_count returned a negative value")
        if used + cost > max_tokens:
            break
        selected.append(snippet)
        used += cost
    if not selected:
        raise ValueError("no evidence snippet fits the configured token budget")
    return "\n".join(selected)
