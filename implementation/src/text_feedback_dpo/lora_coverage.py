from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


_DEFAULT_EXCLUDED = {
    "vision",
    "visual",
    "multimodal_projector",
    "multi_modal_projector",
    "projector",
    "embedding",
    "embed_tokens",
    "lm_head",
    "output_head",
}


@dataclass(frozen=True)
class LoraCoverage:
    target_modules: tuple[str, ...]
    inventory: tuple[dict[str, Any], ...]
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    estimated_lora_parameters: int
    rank: int
    coverage_hash: str


def _is_linear_module(module: Any) -> bool:
    return (
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
    )


def _is_text_projection(name: str, *, excluded: set[str]) -> bool:
    parts = {part.casefold() for part in name.split(".")}
    if parts & excluded:
        return False
    # Qwen3.5 stores projections beneath text transformer layer stacks. Requiring a layer path
    # prevents accidental targeting of tokenizer embeddings, heads, and vision projections.
    return "layers" in parts and bool(parts & {"model", "language_model", "text_model", "backbone", "transformer"})


def _parameter_counts(model: Any) -> tuple[int, int]:
    if not hasattr(model, "named_parameters"):
        raise ValueError("model must expose named_parameters for LoRA coverage accounting")
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if not hasattr(parameter, "numel") or not isinstance(parameter.requires_grad, bool):
            raise ValueError(f"model parameter {name} cannot be counted")
        count = int(parameter.numel())
        if count < 0:
            raise ValueError(f"model parameter {name} has invalid size")
        total += count
        if parameter.requires_grad:
            trainable += count
    if total <= 0:
        raise ValueError("model exposes no parameters for LoRA coverage accounting")
    return total, trainable


def discover_lora_coverage(
    model: Any,
    *,
    rank: int,
    excluded_components: Iterable[str] = (),
    expected_target_modules: Iterable[str] | None = None,
) -> LoraCoverage:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if not hasattr(model, "named_modules"):
        raise ValueError("model must expose named_modules for LoRA coverage discovery")
    excluded = _DEFAULT_EXCLUDED | {str(item).casefold() for item in excluded_components}
    inventory: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name or not _is_linear_module(module) or not _is_text_projection(name, excluded=excluded):
            continue
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"text projection {name} has invalid dimensions")
        inventory.append(
            {
                "name": name,
                "class": module.__class__.__name__,
                "shape": [out_features, in_features],
                "in_features": in_features,
                "out_features": out_features,
            }
        )
    inventory.sort(key=lambda item: str(item["name"]))
    if not inventory:
        raise ValueError("no text-backbone linear modules found for LoRA coverage")
    targets = tuple(str(item["name"]) for item in inventory)
    if expected_target_modules is not None and targets != tuple(expected_target_modules):
        raise ValueError("expected target inventory does not match discovered text projections")
    total, trainable = _parameter_counts(model)
    estimated = sum(rank * (int(item["in_features"]) + int(item["out_features"])) for item in inventory)
    payload = {
        "targets": targets,
        "inventory": inventory,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "rank": rank,
    }
    return LoraCoverage(
        target_modules=targets,
        inventory=tuple(inventory),
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_percentage=100.0 * trainable / total,
        estimated_lora_parameters=estimated,
        rank=rank,
        coverage_hash=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


def build_lora_config(*, model: Any, rank: int, alpha: int, dropout: float, excluded_components: Iterable[str] = ()) -> Any:
    coverage = discover_lora_coverage(
        model,
        rank=rank,
        excluded_components=excluded_components,
    )
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("peft is required to construct the audited LoRA configuration") from exc
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(coverage.target_modules),
    )
