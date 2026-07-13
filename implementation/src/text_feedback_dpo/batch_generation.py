from __future__ import annotations

from collections.abc import Callable


def generate_batch(provider: Callable[..., list[str]], prompts: list[str], **generation_kwargs: object) -> list[dict]:
    if not prompts:
        return []
    outputs = provider(prompts, **generation_kwargs)
    if not isinstance(outputs, list) or len(outputs) != len(prompts):
        raise ValueError(f"batch generation cardinality mismatch: expected {len(prompts)}, got {len(outputs) if isinstance(outputs, list) else type(outputs).__name__}")
    return [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, outputs, strict=True)]
