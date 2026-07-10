import unittest

from text_feedback_dpo.training import (
    build_distillation_rows,
    build_standard_dpo_pairs,
    dpo_loss,
    response_sft_weight,
)


class TrainingDataTest(unittest.TestCase):
    def test_distillation_rows_keep_prompt_and_teacher_target(self):
        rows = build_distillation_rows(
            [{"prompt": "Solve", "completion": "answer"}],
        )
        self.assertEqual(rows, [{"text": "Solve\nanswer"}])

    def test_standard_dpo_keeps_one_initial_pair_per_example(self):
        pairs = [
            {"id": "m1", "metadata": {"failed_attempt": 0}},
            {"id": "m1", "metadata": {"failed_attempt": 1}},
            {"id": "m2", "metadata": {"failed_attempt": 0}},
        ]
        selected = build_standard_dpo_pairs(pairs)
        self.assertEqual(selected, [pairs[0], pairs[2]])

    def test_response_sft_weight_anneals_to_zero(self):
        self.assertEqual(response_sft_weight(0, 10, initial=1.0), 1.0)
        self.assertAlmostEqual(response_sft_weight(5, 10, initial=1.0), 0.5)
        self.assertEqual(response_sft_weight(10, 10, initial=1.0), 0.0)
        self.assertEqual(response_sft_weight(20, 10, initial=1.0), 0.0)

    def test_dpo_loss_has_lower_value_for_a_better_margin(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable in the local unit-test interpreter")
        weak_margin = dpo_loss(
            torch.tensor([0.1]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            beta=0.1,
        )
        strong_margin = dpo_loss(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            beta=0.1,
        )
        self.assertLess(float(strong_margin), float(weak_margin))


if __name__ == "__main__":
    unittest.main()
