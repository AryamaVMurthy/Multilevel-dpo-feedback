import unittest

from text_feedback_dpo.rewards import (
    build_grpo_reward_function,
    compute_reward_from_evaluation,
    validate_grpo_reward_groups,
)


class RewardTest(unittest.TestCase):
    def test_exact_math_reward_does_not_use_incidental_substrings(self):
        wrong = compute_reward_from_evaluation(
            domain="math",
            result={"correct": False, "deterministic": {"numeric_exact_match": False}},
        )
        right = compute_reward_from_evaluation(
            domain="math",
            result={"correct": True, "deterministic": {"numeric_exact_match": True}},
        )
        self.assertEqual(wrong.total, 0.0)
        self.assertEqual(right.total, 1.0)

    def test_searchqa_components_have_explicit_weights_and_unknown_type_is_neutral(self):
        reward = compute_reward_from_evaluation(
            domain="search_qa",
            result={
                "correct": True,
                "deterministic": {
                    "exact_match": True,
                    "token_f1": 0.8,
                    "answer_type_correct": None,
                    "evidence_supported": True,
                },
            },
        )
        self.assertAlmostEqual(reward.components["exact_match"], 1.0)
        self.assertAlmostEqual(reward.components["token_f1"], 0.8)
        self.assertAlmostEqual(reward.total, 0.55 + 0.25 * 0.8 + 0.1 + 0.05)

    def test_reward_function_masks_truncated_completions_and_fails_evaluator_errors(self):
        examples = {"m1": {"id": "m1", "domain": "math"}}
        function = build_grpo_reward_function(
            examples_by_id=examples,
            evaluator=lambda _example, response: {
                "correct": response == "right",
                "deterministic": {"numeric_exact_match": response == "right"},
            },
            domain="math",
            mask_truncated_completions=True,
        )
        rewards = function(["right", "wrong"], example_id=["m1", "m1"], truncated=[False, True])
        self.assertEqual(rewards, [1.0, 0.0])
        broken = build_grpo_reward_function(
            examples_by_id=examples,
            evaluator=lambda *_args: (_ for _ in ()).throw(ValueError("bad evaluator")),
            domain="math",
            mask_truncated_completions=True,
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator failure"):
            broken(["right"], example_id=["m1"], truncated=[False])

    def test_preflight_rejects_degenerate_groups_or_truncation(self):
        with self.assertRaisesRegex(ValueError, "zero-variance"):
            validate_grpo_reward_groups([[1.0, 1.0], [0.0, 0.0]], truncated_rate=0.0, evaluator_agreement=1.0)
        with self.assertRaisesRegex(ValueError, "truncation"):
            validate_grpo_reward_groups([[0.0, 1.0]], truncated_rate=0.06, evaluator_agreement=1.0)
        with self.assertRaisesRegex(ValueError, "agreement"):
            validate_grpo_reward_groups([[0.0, 1.0]], truncated_rate=0.0, evaluator_agreement=0.94)


if __name__ == "__main__":
    unittest.main()
