from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from text_feedback_dpo.searchqa import convert_original_searchqa_row


def convert_gsm8k_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    answer = str(row.get("answer", ""))
    marker = "####"
    if marker not in answer:
        raise ValueError(f"GSM8K row {index} has no answer marker")
    gold_answer = answer.rsplit(marker, 1)[1].strip()
    if not gold_answer:
        raise ValueError(f"GSM8K row {index} has an empty marked answer")
    return {
        "id": f"gsm8k-{index}",
        "domain": "math",
        "problem": str(row["question"]),
        "gold_answer": gold_answer,
        "source": "openai/gsm8k",
    }


def convert_searchqa_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    if "search_results" in row:
        return convert_original_searchqa_row(row, split="unknown", index=index)
    answers = row.get("answers")
    if not isinstance(answers, list) or not answers or not str(answers[0]).strip():
        raise ValueError(f"SearchQA row {index} has no answer")
    context = str(row.get("context", "")).strip()
    if not context:
        raise ValueError(f"SearchQA row {index} has no controlled context")
    return {
        "id": f"searchqa-{index}",
        "domain": "search_qa",
        "problem": str(row["question"]),
        "gold_answer": str(answers[0]).strip(),
        "evidence": [context[:6000]],
        "source": "nyu-dl/SearchQA",
    }


def load_benchmark_examples(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for benchmark loading") from exc

    examples: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        split = str(spec["split"])
        count = int(spec["count"])
        if count <= 0:
            raise ValueError("benchmark count must be positive")
        config_name = spec.get("config")
        if config_name:
            dataset = load_dataset(name, str(config_name), split=split, streaming=True)
        else:
            dataset = load_dataset(name, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            if row_index >= count:
                break
            if name == "openai/gsm8k":
                examples.append(convert_gsm8k_row(row, index=len(examples)))
            elif name == "lucadiliello/searchqa":
                examples.append(convert_searchqa_row(row, index=len(examples)))
            else:
                raise ValueError(f"unsupported benchmark dataset: {name}")
    if not examples:
        raise ValueError("benchmark loading produced no examples")
    return examples
