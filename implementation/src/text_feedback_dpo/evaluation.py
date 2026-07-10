from __future__ import annotations

from collections import Counter
from typing import Any

from text_feedback_dpo.scoring import evaluate_rollout, normalize_answer


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{name} row is missing id")
        if row_id in indexed:
            raise ValueError(f"{name} contains duplicate id: {row_id}")
        indexed[row_id] = row
    return indexed


def _token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_examples(examples: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate saved model rollouts without teacher-only information in prompts."""

    rollout_by_id = _index(rollouts, "rollouts")
    if not examples:
        raise ValueError("examples must not be empty")

    per_example: list[dict[str, Any]] = []
    math_scores: list[float] = []
    search_em: list[float] = []
    search_f1: list[float] = []
    answer_types: list[float] = []
    evidence_support: list[float] = []
    scores: list[float] = []
    format_valid: list[float] = []
    verification_present: list[float] = []
    generated_tokens: list[float] = []

    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("example row is missing id")
        if example_id not in rollout_by_id:
            raise ValueError(f"missing rollout for example id: {example_id}")
        domain = example.get("domain")
        if domain not in {"math", "search_qa"}:
            raise ValueError(f"unsupported evaluation domain for {example_id}: {domain}")
        gold_answer = example.get("gold_answer")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError(f"example {example_id} is missing gold_answer")

        rollout_row = rollout_by_id[example_id]
        result = evaluate_rollout(str(rollout_row.get("rollout", "")), gold_answer)
        score = float(result["score"])
        scores.append(score)
        format_valid.append(float(result["format_valid"]))
        verification_present.append(float(result["verification_present"]))
        if "generated_tokens" in rollout_row:
            generated_tokens.append(float(rollout_row["generated_tokens"]))

        row = {"id": example_id, "domain": domain, **result}
        if domain == "math":
            math_scores.append(score)
        else:
            evidence = example.get("evidence")
            expected_type = example.get("answer_type")
            actual_type = rollout_row.get("answer_type")
            if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) for item in evidence):
                raise ValueError(f"search_qa example {example_id} is missing controlled evidence")
            if not isinstance(expected_type, str) or not expected_type:
                raise ValueError(f"search_qa example {example_id} is missing answer_type")
            if not isinstance(actual_type, str) or not actual_type:
                raise ValueError(f"search_qa rollout {example_id} is missing answer_type")
            final_answer = str(result.get("final_answer", ""))
            em = score
            f1 = _token_f1(final_answer, gold_answer)
            support = float(normalize_answer(final_answer) in normalize_answer(" ".join(evidence)))
            type_score = float(actual_type == expected_type)
            search_em.append(em)
            search_f1.append(f1)
            evidence_support.append(support)
            answer_types.append(type_score)
            row.update(
                {
                    "search_exact_match": em,
                    "search_token_f1": f1,
                    "evidence_supported": bool(support),
                    "answer_type_correct": bool(type_score),
                }
            )
        per_example.append(row)

    return {
        "common": {
            "examples": len(examples),
            "final_answer_accuracy": _mean(scores),
            "format_valid_rate": _mean(format_valid),
            "verification_present_rate": _mean(verification_present),
            "average_generated_tokens": _mean(generated_tokens),
        },
        "math": {"examples": len(math_scores), "exact_accuracy": _mean(math_scores)},
        "search_qa": {
            "examples": len(search_em),
            "exact_match": _mean(search_em),
            "token_f1": _mean(search_f1),
            "answer_type_accuracy": _mean(answer_types),
            "evidence_support_rate": _mean(evidence_support),
        },
        "per_example": per_example,
    }
