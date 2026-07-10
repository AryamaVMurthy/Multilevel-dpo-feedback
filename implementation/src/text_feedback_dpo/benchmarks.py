from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from text_feedback_dpo.searchqa import convert_original_searchqa_row


MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def extract_math_boxed_answer(solution: str) -> str:
    """Extract the final balanced ``\\boxed{...}`` answer from an official MATH solution."""

    text = str(solution)
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("MATH solution has no boxed final answer")
    index = start + len(marker)
    depth = 1
    for end in range(index, len(text)):
        character = text[end]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                answer = text[index:end].strip()
                if not answer:
                    raise ValueError("MATH solution has an empty boxed final answer")
                return answer
    raise ValueError("MATH solution has an unbalanced boxed final answer")


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


def convert_math_row(
    row: dict[str, Any],
    *,
    subject: str,
    source_split: str,
    index: int,
) -> dict[str, Any]:
    """Normalize one official MATH row while retaining extraction provenance."""

    if subject not in MATH_SUBJECTS:
        raise ValueError(f"unsupported MATH subject: {subject}")
    problem = str(row.get("problem", "")).strip()
    solution = str(row.get("solution", "")).strip()
    if not problem:
        raise ValueError(f"MATH {subject}:{source_split}:{index} has an empty problem")
    if not solution:
        raise ValueError(f"MATH {subject}:{source_split}:{index} has an empty solution")
    level = row.get("level")
    if isinstance(level, str):
        match = re.fullmatch(r"Level\s+([1-5])", level.strip(), flags=re.IGNORECASE)
        level = int(match.group(1)) if match is not None else None
    if isinstance(level, bool) or not isinstance(level, int) or level not in {1, 2, 3, 4, 5}:
        raise ValueError(f"MATH {subject}:{source_split}:{index} has invalid level")
    declared_subject = re.sub(r"[\s-]+", "_", str(row.get("type", subject)).strip().casefold())
    if declared_subject and declared_subject != subject:
        raise ValueError(
            f"MATH {subject}:{source_split}:{index} declares mismatched subject {declared_subject!r}"
        )
    gold_answer = extract_math_boxed_answer(solution)
    return {
        "id": f"math-{subject}-{source_split}-{index}",
        "domain": "math",
        "problem": problem,
        "gold_answer": gold_answer,
        "reference_solution": solution,
        "source": "EleutherAI/hendrycks_math",
        "source_subject": subject,
        "difficulty_level": level,
        "gold_answer_extraction": {
            "method": "last_balanced_boxed",
            "source_field": "solution",
            "source_value": gold_answer,
        },
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
            elif name == "EleutherAI/hendrycks_math":
                if not config_name:
                    raise ValueError("MATH benchmark loading requires one official subject config")
                examples.append(
                    convert_math_row(
                        dict(row),
                        subject=str(config_name),
                        source_split=split,
                        index=len(examples),
                    )
                )
            elif name == "lucadiliello/searchqa":
                examples.append(convert_searchqa_row(row, index=len(examples)))
            else:
                raise ValueError(f"unsupported benchmark dataset: {name}")
    if not examples:
        raise ValueError("benchmark loading produced no examples")
    return examples
