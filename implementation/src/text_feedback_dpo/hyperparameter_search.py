from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


def _candidate_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DpoCandidate:
    learning_rate: float
    beta: float
    weight_decay: float
    warmup_fraction: float
    scheduler: str
    loss_type: str = "sigmoid_norm"
    ld_alpha: float | None = None

    @property
    def candidate_id(self) -> str:
        return "dpo-" + _candidate_hash(asdict(self))


@dataclass(frozen=True)
class GrpoCandidate:
    learning_rate: float
    kl_beta: float
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    num_iterations: int = 2
    num_generations: int = 4
    loss_type: str = "grpo"

    @property
    def candidate_id(self) -> str:
        return "grpo-" + _candidate_hash(asdict(self))


def build_dpo_candidates(
    *,
    learning_rates: Iterable[float],
    betas: Iterable[float],
    weight_decay: float,
    warmup_fraction: float,
    scheduler: str,
    loss_type: str = "sigmoid_norm",
    ld_alpha: float | None = None,
) -> list[DpoCandidate]:
    if loss_type not in {"sigmoid_norm", "sigmoid"}:
        raise ValueError("DPO loss_type must be sigmoid_norm or sigmoid")
    if loss_type == "sigmoid_norm" and ld_alpha is not None:
        raise ValueError("sigmoid_norm DPO must not set ld_alpha")
    if loss_type == "sigmoid" and ld_alpha not in {0.25, 0.5, 0.75}:
        raise ValueError("length-desensitized DPO requires ld_alpha in {0.25, 0.5, 0.75}")
    candidates = [
        DpoCandidate(
            float(lr),
            float(beta),
            float(weight_decay),
            float(warmup_fraction),
            str(scheduler),
            str(loss_type),
            ld_alpha,
        )
        for lr in learning_rates
        for beta in betas
    ]
    if not candidates:
        raise ValueError("DPO candidate matrix must not be empty")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("DPO candidate matrix contains duplicate candidates")
    return candidates


def build_grpo_candidates(
    *,
    learning_rates: Iterable[float],
    kl_betas: Iterable[float],
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    num_iterations: int = 2,
    num_generations: int = 4,
    loss_type: str = "grpo",
) -> list[GrpoCandidate]:
    candidates = [
        GrpoCandidate(
            float(lr),
            float(beta),
            float(epsilon_low),
            float(epsilon_high),
            int(num_iterations),
            int(num_generations),
            str(loss_type),
        )
        for lr in learning_rates
        for beta in kl_betas
    ]
    if not candidates:
        raise ValueError("GRPO candidate matrix must not be empty")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("GRPO candidate matrix contains duplicate candidates")
    return candidates


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    if not hasattr(candidate, "candidate_id"):
        raise ValueError("candidate must expose candidate_id")
    payload = asdict(candidate)
    payload["candidate_id"] = candidate.candidate_id
    return payload


def _ledger_hash(ledger: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in ledger.items() if key not in {"frozen", "freeze_manifest"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def create_search_ledger(
    *,
    method: str,
    candidates: Iterable[Any],
    promote_counts: Iterable[int],
    dataset_manifest_hash: str,
    seed: int,
) -> dict[str, Any]:
    candidate_list = list(candidates)
    counts = [int(value) for value in promote_counts]
    if not method.strip() or not candidate_list or not counts:
        raise ValueError("search ledger requires method, candidates, and promotion stages")
    if any(value <= 0 for value in counts) or counts[-1] != 1:
        raise ValueError("promotion counts must be positive and end with one selected candidate")
    candidate_payloads = [_candidate_payload(candidate) for candidate in candidate_list]
    if any(count > len(candidate_list) for count in counts):
        raise ValueError("promotion count exceeds candidate count")
    return {
        "schema": "hyperparameter-ledger-v1",
        "method": method,
        "dataset_manifest_hash": dataset_manifest_hash,
        "seed": seed,
        "candidate_order": [item["candidate_id"] for item in candidate_payloads],
        "candidates": {item["candidate_id"]: item for item in candidate_payloads},
        "promote_counts": counts,
        "observations": {},
        "promotions": {},
        "frozen": False,
    }


def register_observation(
    ledger: dict[str, Any],
    *,
    candidate_id: str,
    stage: int,
    status: str,
    metrics: Mapping[str, Any],
    artifact_hash: str,
    failure_reason: str | None = None,
) -> None:
    if ledger.get("frozen"):
        raise RuntimeError("search ledger is frozen")
    if candidate_id not in ledger.get("candidates", {}):
        raise ValueError(f"unknown candidate id: {candidate_id}")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0 or stage >= len(ledger["promote_counts"]):
        raise ValueError("observation stage is outside the ledger stages")
    if status not in {"valid", "invalid"}:
        raise ValueError("observation status must be valid or invalid")
    if not isinstance(metrics, Mapping):
        raise ValueError("observation metrics must be a mapping")
    copied_metrics = dict(metrics)
    if status == "valid":
        score = copied_metrics.get("selection_metric")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise ValueError("valid observation requires finite selection_metric")
        if not isinstance(artifact_hash, str) or not artifact_hash:
            raise ValueError("valid observation requires artifact_hash")
    else:
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            raise ValueError("invalid observation requires explicit failure_reason")
    key = f"{stage}:{candidate_id}"
    observation = {
        "candidate_id": candidate_id,
        "candidate": dict(ledger["candidates"][candidate_id]),
        "stage": stage,
        "status": status,
        "metrics": copied_metrics,
        "artifact_hash": artifact_hash,
        "failure_reason": failure_reason,
    }
    previous = ledger["observations"].get(key)
    if previous is not None:
        if previous != observation:
            raise ValueError(f"observation already exists with different content: {key}")
        return
    ledger["observations"][key] = observation


def _stage_candidates(ledger: Mapping[str, Any], stage: int) -> list[str]:
    if stage == 0:
        return list(ledger["candidate_order"])
    previous = ledger["promotions"].get(str(stage - 1))
    if not isinstance(previous, list) or not previous:
        raise ValueError(f"stage {stage} cannot start before stage {stage - 1} promotion")
    return list(previous)


def promote_stage(ledger: dict[str, Any], *, stage: int) -> list[str]:
    if ledger.get("frozen"):
        raise RuntimeError("search ledger is frozen")
    if stage < 0 or stage >= len(ledger["promote_counts"]):
        raise ValueError("promotion stage is outside the ledger")
    previous = ledger["promotions"].get(str(stage))
    if previous is not None:
        return list(previous)
    candidates = _stage_candidates(ledger, stage)
    observations = []
    for candidate_id in candidates:
        key = f"{stage}:{candidate_id}"
        if key not in ledger["observations"]:
            raise ValueError(f"candidate is missing stage observation: {key}")
        observation = ledger["observations"][key]
        if observation["status"] == "valid":
            observations.append(observation)
    target = int(ledger["promote_counts"][stage])
    if len(observations) < target:
        raise ValueError(f"only {len(observations)} valid candidates available for stage {stage}, need {target}")

    def sort_key(observation: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
        candidate = ledger["candidates"][observation["candidate_id"]]
        regularization = float(candidate.get("beta", candidate.get("kl_beta", 0.0)))
        return (
            -float(observation["metrics"]["selection_metric"]),
            float(candidate["learning_rate"]),
            -regularization,
            float(observation["metrics"].get("gpu_hours", math.inf)),
            str(observation["candidate_id"]),
        )

    promoted = [observation["candidate_id"] for observation in sorted(observations, key=sort_key)[:target]]
    ledger["promotions"][str(stage)] = promoted
    return promoted


def freeze_selection(ledger: dict[str, Any], *, candidate_id: str, stage: int) -> dict[str, Any]:
    if ledger.get("frozen"):
        raise RuntimeError("search ledger is already frozen")
    if stage != len(ledger["promote_counts"]) - 1:
        raise ValueError("selection can only be frozen at the final promotion stage")
    promoted = ledger["promotions"].get(str(stage))
    if promoted != [candidate_id]:
        raise ValueError("candidate is not the sole final-stage promotion")
    observation = ledger["observations"].get(f"{stage}:{candidate_id}")
    if not observation or observation["status"] != "valid":
        raise ValueError("selected candidate lacks a valid final-stage observation")
    manifest = {
        "schema": "hyperparameter-freeze-v1",
        "method": ledger["method"],
        "dataset_manifest_hash": ledger["dataset_manifest_hash"],
        "seed": ledger["seed"],
        "candidate_id": candidate_id,
        "candidate": dict(ledger["candidates"][candidate_id]),
        "stage": stage,
        "selection_observation": observation,
        "ledger_hash": _ledger_hash(ledger),
    }
    ledger["frozen"] = True
    ledger["freeze_manifest"] = manifest
    return manifest
