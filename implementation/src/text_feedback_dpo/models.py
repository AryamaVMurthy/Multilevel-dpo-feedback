from __future__ import annotations

from dataclasses import dataclass, field


class ModelProvider:
    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        raise NotImplementedError


@dataclass
class FakeModelProvider(ModelProvider):
    outputs: dict[str, str]
    prompts: list[tuple[str, str]] = field(default_factory=list)

    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        self.prompts.append((role, prompt))
        if role not in self.outputs:
            raise ValueError(f"missing fake output for role: {role}")
        return self.outputs[role]


class TransformersModelProvider(ModelProvider):
    def __init__(
        self,
        *,
        model_ids: dict[str, str],
        allow_cpu_for_unit_tests: bool = False,
    ) -> None:
        self.model_ids = model_ids
        self.allow_cpu_for_unit_tests = allow_cpu_for_unit_tests
        self._loaded: dict[str, tuple[object, object]] = {}

    def _load(self, role: str) -> tuple[object, object]:
        if role not in self.model_ids:
            raise ValueError(f"missing model id for role: {role}")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for TransformersModelProvider") from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for TransformersModelProvider") from exc

        if not self.allow_cpu_for_unit_tests and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for model generation; refusing hidden CPU fallback")

        if role not in self._loaded:
            model_id = self.model_ids[role]
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self._loaded[role] = (tokenizer, model)
        return self._loaded[role]

    def generate(self, role: str, prompt: str, **generation_kwargs: object) -> str:
        tokenizer, model = self._load(role)
        encoded = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(generation_kwargs.get("max_new_tokens", 128)),
            do_sample=float(generation_kwargs.get("temperature", 0.0)) > 0.0,
            temperature=float(generation_kwargs.get("temperature", 0.0)) or None,
            top_p=float(generation_kwargs.get("top_p", 1.0)),
        )
        generated = output_ids[0][encoded["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)
