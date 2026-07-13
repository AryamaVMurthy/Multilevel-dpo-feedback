import unittest

from text_feedback_dpo.trainers import _dpo_args, _rl_args, _sft_args, searchqa_rl_reward


class TrainerContractTest(unittest.TestCase):
    def test_sft_uses_prompt_completion_and_completion_only_loss(self):
        args = _sft_args({}, "out")
        self.assertTrue(args["completion_only_loss"])
        self.assertEqual(args["max_length"], 4096)
        self.assertNotIn("dataset_text_field", args)
        self.assertFalse(args["save_only_model"])
        self.assertGreaterEqual(args["save_total_limit"], 2)

    def test_dpo_uses_plain_completion_length_contract(self):
        args = _dpo_args({}, "out")
        self.assertEqual(args["max_length"], 4096)
        self.assertEqual(args["loss_type"], ["sigmoid"])

    def test_rl_completions_are_short_and_dapo_is_explicit(self):
        self.assertEqual(_rl_args({}, "out", method="grpo")["max_completion_length"], 32)
        dapo = _rl_args({}, "out", method="dapo")
        self.assertEqual(dapo["loss_type"], "dapo")
        self.assertTrue(dapo["mask_truncated_completions"])

    def test_exact_match_dominates_bounded_f1_shaping(self):
        exact = searchqa_rl_reward("Ada Lovelace", "Ada Lovelace")
        partial = searchqa_rl_reward("Ada", "Ada Lovelace")
        wrong = searchqa_rl_reward("Grace Hopper", "Ada Lovelace")
        self.assertEqual(exact, 1.0)
        self.assertGreater(partial, wrong)
        self.assertLessEqual(partial, 0.1)
        self.assertGreaterEqual(wrong, 0.0)


if __name__ == "__main__":
    unittest.main()
