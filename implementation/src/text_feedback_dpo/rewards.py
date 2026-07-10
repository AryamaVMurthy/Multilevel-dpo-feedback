from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


@dataclass(frozen=True)
class RewardBreakdown:
    domain: str
    total: float
    components: dict[str, float]
    truncated: bool = False


def _bounded_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"reward field {field} must be finite numeric")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"reward field {field} must be between zero and one")
    return number


def compute_reward_from_evaluation(*, domain: str, result: Mapping[str, Any]) -> RewardBreakdown:
    if not isinstance(result, Mapping):
        raise ValueError("evaluation result must be a mapping")
    deterministic = result.get("deterministic")
    if not isinstance(deterministic, Mapping):
        raise ValueError("evaluation result is missing deterministic domain evidence")
    if domain == "math":
        exact = deterministic.get("numeric_exact_match")
        if not isinstance(exact, bool):
            raise ValueError("math evaluation is missing numeric_exact_match")
        return RewardBreakdown(domain=domain, total=float(exact), components={"numeric_exact_match": float(exact)})
    if domain == "search_qa":
        exact = _bounded_number(float(deterministic.get("exact_match")), "exact_match")
        f1 = _bounded_number(deterministic.get("token_f1"), "token_f1")
        support = float(bool(deterministic.get("evidence_supported")))
        answer_type = deterministic.get("answer_type_correct")
        type_score = 0.5 if answer_type is None else float(bool(answer_type))
        components = {
            "exact_match": exact,
            "token_f1": f1,
            "evidence_supported": support,
            "answer_type_correct": type_score,
        }
        weights = {
            "exact_match": 0.55,
            "token_f1": 0.25,
            "evidence_supported": 0.10,
            "answer_type_correct": 0.10,
        }
        total = sum(weights[key] * components[key] for key in weights)
        return RewardBreakdown(domain=domain, total=total, components=components)
    raise ValueError(f"unsupported reward domain: {domain}")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        if not completion.strip():
            raise ValueError("completion must be non-empty")
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping) and isinstance(last.get("content"), str) and last["content"].strip():
            return last["content"]
    raise ValueError("completion must be a non-empty string or chat message list")


def build_grpo_reward_function(
    *,
    examples_by_id: Mapping[str, Mapping[str, Any]],
    evaluator: Callable[[dict[str, Any], str], Mapping[str, Any]],
    domain: str,
    mask_truncated_completions: bool,
    max_completion_tokens: int = 2048,
) -> Callable[..., list[float]]:
    if not examples_by_id:
        raise ValueError("examples_by_id must not be empty")
    if domain not in {"math", "search_qa"}:
        raise ValueError("domain must be math or search_qa")

    def reward_func(
        completions: Sequence[Any],
        *,
        example_id: Sequence[str],
        truncated: Sequence[bool] | None = None,
        completion_ids: Sequence[Sequence[Any]] | None = None,
        **_: Any,
    ) -> list[float]:
        if truncated is None:
            if completion_ids is None:
                raise ValueError("GRPO reward requires truncated flags or completion_ids for truncation masking")
            truncated = [len(token_ids) >= max_completion_tokens for token_ids in completion_ids]
        if len(completions) != len(example_id) or len(completions) != len(truncated):
            raise ValueError("completion, example_id, and truncated batches must have equal length")
        rewards: list[float] = []
        for completion, row_id, is_truncated in zip(completions, example_id, truncated):
            if not isinstance(row_id, str) or row_id not in examples_by_id:
                raise ValueError(f"reward batch references unknown example id: {row_id}")
            if not isinstance(is_truncated, bool):
                raise ValueError("truncated batch values must be boolean")
            try:
                result = evaluator(dict(examples_by_id[row_id]), _completion_text(completion))
                reward = compute_reward_from_evaluation(domain=domain, result=result)
            except Exception as exc:
                raise RuntimeError(f"evaluator failure for reward example {row_id}: {exc}") from exc
            rewards.append(0.0 if mask_truncated_completions and is_truncated else reward.total)
        return rewards

    return reward_func


def validate_grpo_reward_groups(
    groups: Sequence[Sequence[float]],
    *,
    truncated_rate: float,
    evaluator_agreement: float,
) -> dict[str, float]:
    if not groups:
        raise ValueError("GRPO reward groups must not be empty")
    if not 0 <= truncated_rate <= 1:
        raise ValueError("truncated_rate must be between zero and one")
    if not 0 <= evaluator_agreement <= 1:
        raise ValueError("evaluator_agreement must be between zero and one")
    zero_variance = 0
    for index, group in enumerate(groups):
        if not group:
            raise ValueError(f"GRPO reward group {index} is empty")
        values = [float(value) for value in group]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"GRPO reward group {index} contains nonfinite reward")
        if max(values) - min(values) <= 1e-12:
            zero_variance += 1
    zero_variance_rate = zero_variance / len(groups)
    if zero_variance_rate > 0.50:
        raise ValueError(f"zero-variance reward groups exceed gate: {zero_variance_rate:.3f}")
    if truncated_rate > 0.05:
        raise ValueError(f"truncation rate exceeds gate: {truncated_rate:.3f}")
    if evaluator_agreement < 0.95:
        raise ValueError(f"evaluator agreement is below gate: {evaluator_agreement:.3f}")
    return {
        "groups": float(len(groups)),
        "zero_variance_rate": zero_variance_rate,
        "truncated_rate": truncated_rate,
        "evaluator_agreement": evaluator_agreement,
    }
