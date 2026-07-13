from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class RuntimeErrorExplicit(RuntimeError):
    """Runtime failure that must never degrade to an implicit fallback."""


@dataclass(frozen=True)
class StudentGeneration:
    response: str
    scratchpad: str | None
    mode: str


def load_tokenizer(model_id: str, *, revision: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for model runtime") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision or None, trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeErrorExplicit(f"tokenizer for {model_id} has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_student(model_id: str, *, revision: str | None = None, attention_implementation: str = "sdpa", device: str = "cuda:0"):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("torch and transformers are required for student runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeErrorExplicit("CUDA is required for full-finetuning student runtime")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision or None,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        trust_remote_code=False,
    )
    model.config.use_cache = True
    return model


def load_teacher(model_id: str, *, revision: str | None = None, quantization: str, attention_implementation: str = "sdpa", device: str = "cuda:0"):
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("torch and transformers are required for teacher runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeErrorExplicit("CUDA is required for teacher inference")
    kwargs: dict[str, Any] = {
        "device_map": {"": device},
        "trust_remote_code": False,
        "attn_implementation": attention_implementation,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError("teacher quantization must be 4bit or bf16")
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision or None, **kwargs)


def generate_batch(model, tokenizer, prompts: list[str], *, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    if not prompts:
        return []
    if max_new_tokens <= 0 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")
    max_input_tokens = 4096 - max_new_tokens
    if max_input_tokens <= 0:
        raise ValueError("max_new_tokens leaves no room for an input within the 4096-token limit")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(model.device)
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    output_ids = model.generate(**encoded, **kwargs)
    input_length = encoded.input_ids.shape[1]
    return [tokenizer.decode(output[input_length:], skip_special_tokens=True).strip() for output in output_ids]


def render_teacher_prompts(tokenizer, prompts: list[str], *, enable_thinking: bool = True) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]


def extract_qwen_final_content(text: str) -> str:
    stripped = text.strip()
    has_open = "<think>" in stripped
    has_close = "</think>" in stripped
    if has_open != has_close:
        raise RuntimeErrorExplicit("unterminated or malformed Qwen thinking block")
    if has_close:
        stripped = stripped.rsplit("</think>", maxsplit=1)[1].strip()
    if not stripped:
        raise RuntimeErrorExplicit("teacher generation has no final content after private thinking")
    return stripped


def generate_student_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    mode: str,
    scratchpad_max_new_tokens: int,
    answer_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_fn: Callable[..., list[str]] | None = None,
) -> list[StudentGeneration]:
    if mode not in {"direct", "two_pass"}:
        raise ValueError("student thinking mode must be direct or two_pass")
    generate = generation_fn or generate_batch
    if mode == "direct":
        responses = generate(
            model, tokenizer, prompts, max_new_tokens=answer_max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
        return [StudentGeneration(response=response, scratchpad=None, mode=mode) for response in responses]
    if scratchpad_max_new_tokens <= 0:
        raise ValueError("scratchpad_max_new_tokens must be positive in two_pass mode")
    scratchpad_prompts = [
        f"{prompt}\n\nPrivate scratchpad: reason from the evidence before answering. This text will not be scored."
        for prompt in prompts
    ]
    scratchpads = generate(
        model, tokenizer, scratchpad_prompts, max_new_tokens=scratchpad_max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    if len(scratchpads) != len(prompts):
        raise RuntimeErrorExplicit("student scratchpad batch cardinality mismatch")
    answer_prompts = [
        f"{prompt}\n\nPrivate scratchpad (do not repeat it):\n{scratchpad}\n\nReturn only the short answer with no explanation.\nAnswer:"
        for prompt, scratchpad in zip(prompts, scratchpads, strict=True)
    ]
    responses = generate(
        model, tokenizer, answer_prompts, max_new_tokens=answer_max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    if len(responses) != len(prompts):
        raise RuntimeErrorExplicit("student answer batch cardinality mismatch")
    return [
        StudentGeneration(response=response, scratchpad=scratchpad, mode=mode)
        for response, scratchpad in zip(responses, scratchpads, strict=True)
    ]


def choose_teacher_candidate(candidates: list[str], *, quantization: str) -> dict:
    if not candidates:
        raise ValueError("teacher candidates must not be empty")
    if quantization not in {"4bit", "bf16"}:
        raise ValueError("teacher quantization must be 4bit or bf16")
    return {"model_id": candidates[0], "quantization": quantization, "selection_reason": "first_explicit_candidate_requires_runtime_fit_probe"}
