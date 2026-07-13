from __future__ import annotations


def build_method_config(method: str, *, max_length: int) -> dict:
    if method not in {"sft", "dpo", "grpo", "dapo"}:
        raise ValueError(f"unknown method: {method}")
    if max_length != 4096:
        raise ValueError("max_length must remain exactly 4096")
    return {"method": method, "full_finetuning": True, "adapter": None, "max_length": max_length, "bf16": True, "gradient_checkpointing": True, "deepspeed_stage": 3}
