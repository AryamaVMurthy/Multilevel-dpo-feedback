import unittest

from text_feedback_dpo.lora_coverage import discover_lora_coverage


class FakeParameter:
    def __init__(self, count, requires_grad=True):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self._count


class FakeLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = object()


class FakeModel:
    def __init__(self):
        self.modules = [
            ("model.language_model.layers.0.linear_attn.in_proj_qkv", FakeLinear(8, 24)),
            ("model.language_model.layers.0.linear_attn.out_proj", FakeLinear(8, 8)),
            ("model.language_model.layers.0.mlp.gate_proj", FakeLinear(8, 16)),
            ("model.language_model.layers.1.self_attn.q_proj", FakeLinear(8, 8)),
            ("model.language_model.layers.1.self_attn.o_proj", FakeLinear(8, 8)),
            ("model.visual.blocks.0.q_proj", FakeLinear(8, 8)),
            ("model.language_model.embed_tokens", FakeLinear(8, 8)),
            ("model.language_model.lm_head", FakeLinear(8, 8)),
        ]

    def named_modules(self):
        return iter([("", self), *self.modules])

    def named_parameters(self):
        return iter(
            [
                ("model.language_model.layers.0.weight", FakeParameter(100)),
                ("model.visual.blocks.0.weight", FakeParameter(50)),
                ("model.language_model.lm_head.weight", FakeParameter(25, requires_grad=False)),
            ]
        )


class LoraCoverageTest(unittest.TestCase):
    def test_discovers_all_text_projection_classes_and_excludes_non_text_components(self):
        coverage = discover_lora_coverage(FakeModel(), rank=16)

        self.assertEqual(
            coverage.target_modules,
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "model.language_model.layers.0.linear_attn.out_proj",
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.1.self_attn.o_proj",
                "model.language_model.layers.1.self_attn.q_proj",
            ),
        )
        self.assertEqual(coverage.total_parameters, 175)
        self.assertEqual(coverage.trainable_parameters, 150)
        self.assertEqual(coverage.inventory[0]["shape"], [24, 8])
        self.assertGreater(coverage.estimated_lora_parameters, 0)

    def test_empty_or_unexpected_inventory_fails_explicitly(self):
        class Empty:
            def named_modules(self):
                return iter([("", self)])

            def named_parameters(self):
                return iter([])

        with self.assertRaisesRegex(ValueError, "no text-backbone linear modules"):
            discover_lora_coverage(Empty(), rank=16)
        with self.assertRaisesRegex(ValueError, "expected target inventory"):
            discover_lora_coverage(FakeModel(), rank=16, expected_target_modules=("wrong",))


if __name__ == "__main__":
    unittest.main()
