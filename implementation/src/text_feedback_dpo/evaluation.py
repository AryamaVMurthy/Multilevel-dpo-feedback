from __future__ import annotations

from collections import Counter
from typing import Any, Callable

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


def evaluate_native_examples(
    examples: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    *,
    evaluator: Callable[[dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate natural Qwen responses through the explicit evaluator contract."""

    if not examples:
        raise ValueError("examples must not be empty")
    rollout_by_id = _index(rollouts, "native rollouts")
    per_example: list[dict[str, Any]] = []
    scores: list[float] = []
    confidences: list[float] = []
    requires_judgment: list[float] = []
    nonempty: list[float] = []
    generated_tokens: list[float] = []
    math_scores: list[float] = []
    search_exact: list[float] = []
    search_f1: list[float] = []
    search_types: list[float] = []
    search_support: list[float] = []

    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("example row is missing id")
        if example_id not in rollout_by_id:
            raise ValueError(f"missing native rollout for example id: {example_id}")
        domain = example.get("domain")
        if domain not in {"math", "search_qa"}:
            raise ValueError(f"unsupported evaluation domain for {example_id}: {domain}")
        rollout_row = rollout_by_id[example_id]
        response = rollout_row.get("response")
        if not isinstance(response, str):
            raise ValueError(f"native rollout {example_id} is missing string response")
        result = evaluator(example, response)
        if not isinstance(result, dict):
            raise ValueError(f"evaluator returned non-object result for {example_id}")
        required = ("correct", "confidence", "requires_model_judgment", "deterministic")
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(f"evaluator result for {example_id} is missing: {', '.join(missing)}")
        if not isinstance(result["correct"], bool):
            raise ValueError(f"evaluator result correct must be boolean for {example_id}")
        if not isinstance(result["confidence"], (int, float)) or not 0 <= float(result["confidence"]) <= 1:
            raise ValueError(f"evaluator confidence must be between 0 and 1 for {example_id}")
        if not isinstance(result["requires_model_judgment"], bool):
            raise ValueError(f"evaluator requires_model_judgment must be boolean for {example_id}")
        deterministic = result["deterministic"]
        if not isinstance(deterministic, dict) or not isinstance(deterministic.get("correct"), bool):
            raise ValueError(f"evaluator deterministic result is invalid for {example_id}")

        score = float(result["correct"])
        scores.append(score)
        confidences.append(float(result["confidence"]))
        requires_judgment.append(float(result["requires_model_judgment"]))
        nonempty.append(float(bool(response.strip())))
        if "generated_tokens" in rollout_row:
            generated_tokens.append(float(rollout_row["generated_tokens"]))
        row = {"id": example_id, "domain": domain, "evaluator_result": result}
        if domain == "math":
            math_scores.append(score)
        else:
            for field in ("exact_match", "token_f1", "answer_type_correct", "evidence_supported"):
                if field not in deterministic:
                    raise ValueError(f"SearchQA deterministic result for {example_id} is missing {field}")
            search_exact.append(float(deterministic["exact_match"]))
            search_f1.append(float(deterministic["token_f1"]))
            if deterministic["answer_type_correct"] is not None:
                search_types.append(float(deterministic["answer_type_correct"]))
            search_support.append(float(deterministic["evidence_supported"]))
            row["search_metrics"] = {
                "exact_match": deterministic["exact_match"],
                "token_f1": deterministic["token_f1"],
                "answer_type_correct": deterministic["answer_type_correct"],
                "evidence_supported": deterministic["evidence_supported"],
            }
        per_example.append(row)

    return {
        "common": {
            "examples": len(examples),
            "final_answer_accuracy": _mean(scores),
            "evaluator_confidence": _mean(confidences),
            "requires_model_judgment_rate": _mean(requires_judgment),
            "nonempty_response_rate": _mean(nonempty),
            "average_generated_tokens": _mean(generated_tokens),
        },
        "math": {"examples": len(math_scores), "exact_accuracy": _mean(math_scores)},
        "search_qa": {
            "examples": len(search_exact),
            "exact_match": _mean(search_exact),
            "token_f1": _mean(search_f1),
            "answer_type_accuracy": _mean(search_types),
            "evidence_support_rate": _mean(search_support),
        },
        "per_example": per_example,
    }
