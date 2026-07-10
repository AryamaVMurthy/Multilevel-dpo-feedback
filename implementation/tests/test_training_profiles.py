import unittest

from text_feedback_dpo.hyperparameter_search import build_dpo_candidates, build_grpo_candidates
from text_feedback_dpo.training import (
    build_paper_dpo_config_kwargs,
    build_paper_grpo_config_kwargs,
    build_optimizer_profile,
    materialize_warmup_steps,
)


class TrainingProfileTest(unittest.TestCase):
    def test_optimizer_profile_materializes_integer_warmup_and_fused_adamw(self):
        self.assertEqual(materialize_warmup_steps(19, 0.05), 1)
        self.assertEqual(materialize_warmup_steps(100, 0.05), 5)
        profile = build_optimizer_profile(
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_fraction=0.05,
            total_updates=100,
            scheduler="cosine",
        )
        self.assertEqual(profile["optim"], "adamw_torch_fused")
        self.assertEqual(profile["adam_beta1"], 0.9)
        self.assertEqual(profile["adam_beta2"], 0.999)
        self.assertEqual(profile["adam_epsilon"], 1e-8)
        self.assertEqual(profile["warmup_steps"], 5)
        self.assertNotIn("warmup_ratio", profile)

    def test_paper_dpo_and_grpo_profiles_use_approved_generation_and_update_contract(self):
        dpo_candidate = build_dpo_candidates(
            learning_rates=(5e-6,),
            betas=(0.1,),
            weight_decay=0.01,
            warmup_fraction=0.05,
            scheduler="cosine",
        )[0]
        dpo = build_paper_dpo_config_kwargs(
            output_dir="out",
            max_steps=10,
            candidate=dpo_candidate,
            effective_global_batch=16,
        )
        self.assertEqual(dpo["max_length"], 2048)
        self.assertEqual(dpo["gradient_accumulation_steps"], 16)
        self.assertTrue(dpo["bf16"])
        self.assertEqual(dpo["beta"], 0.1)
        self.assertEqual(dpo["optim"], "adamw_torch_fused")

        grpo_candidate = build_grpo_candidates(learning_rates=(5e-6,), kl_betas=(0.01,))[0]
        grpo = build_paper_grpo_config_kwargs(
            output_dir="out",
            max_steps=10,
            candidate=grpo_candidate,
        )
        self.assertEqual(grpo["num_generations"], 4)
        self.assertEqual(grpo["generation_batch_size"], 4)
        self.assertEqual(grpo["epsilon"], 0.2)
        self.assertEqual(grpo["loss_type"], "grpo")
        self.assertEqual(grpo["beta"], 0.01)
        self.assertTrue(grpo["mask_truncated_completions"])


if __name__ == "__main__":
    unittest.main()
