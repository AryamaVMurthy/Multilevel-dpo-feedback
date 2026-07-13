import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from text_feedback_dpo.trainers import _common_args, _dpo_args, _rl_args, _save_final, _sft_args, searchqa_rl_reward


class TrainerContractTest(unittest.TestCase):
    def test_final_save_keeps_resumable_checkpoints(self):
        class Trainer:
            def save_model(self, path):
                Path(path).mkdir(parents=True)

        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = output / "checkpoint-10"
            checkpoint.mkdir()
            (checkpoint / "optimizer.pt").write_text("state", encoding="utf-8")
            _save_final(Trainer(), output)
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((output / "final").is_dir())
    def test_common_training_args_enable_measured_ada_optimizations(self):
        args = _common_args({"max_steps": -1}, "out")
        self.assertTrue(args["bf16"])
        self.assertFalse(args["fp16"])
        self.assertTrue(args["tf32"])
        self.assertEqual(args["optim"], "adamw_torch_fused")
        self.assertNotIn("group_by_length", args)
        self.assertEqual(args["dataloader_num_workers"], 0)
        self.assertTrue(args["dataloader_pin_memory"])
        self.assertFalse(args["ddp_find_unused_parameters"])
        self.assertEqual(args["gradient_checkpointing_kwargs"], {"use_reentrant": False})
        self.assertTrue(args["include_num_input_tokens_seen"])

    def test_every_method_initializes_the_model_with_real_torch_bfloat16(self):
        config = {"max_steps": 1, "model_revision": "pinned-revision"}
        for args in (
            _sft_args(config, "out"),
            _dpo_args(config, "out"),
            _rl_args(config, "out", method="grpo"),
            _rl_args(config, "out", method="dapo"),
        ):
            self.assertEqual(args["model_init_kwargs"]["dtype"], torch.bfloat16)
            self.assertNotIn("torch_dtype", args["model_init_kwargs"])

    def test_sft_uses_prompt_completion_and_completion_only_loss(self):
        args = _sft_args({"max_steps": -1}, "out")
        self.assertTrue(args["completion_only_loss"])
        self.assertEqual(args["max_length"], 4096)
        self.assertNotIn("dataset_text_field", args)
        self.assertFalse(args["save_only_model"])
        self.assertGreaterEqual(args["save_total_limit"], 2)

    def test_dpo_uses_plain_completion_length_contract(self):
        args = _dpo_args({"max_steps": -1}, "out")
        self.assertEqual(args["max_length"], 4096)
        self.assertEqual(args["loss_type"], ["sigmoid"])
        self.assertTrue(args["precompute_ref_log_probs"])
        self.assertFalse(args["use_liger_kernel"])

    def test_rl_completions_are_short_and_dapo_is_explicit(self):
        grpo = _rl_args({"max_steps": -1}, "out", method="grpo")
        self.assertEqual(grpo["max_completion_length"], 256)
        self.assertEqual(grpo["loss_type"], "grpo")
        dapo = _rl_args({"max_steps": -1}, "out", method="dapo")
        self.assertEqual(dapo["loss_type"], "dapo")
        self.assertEqual(dapo["epsilon"], 0.2)
        self.assertEqual(dapo["epsilon_high"], 0.28)
        self.assertEqual(dapo["beta"], 0.0)
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
