import unittest
from pathlib import Path
from unittest import mock

from text_feedback_dpo.cli import _paper_candidates, _paper_evaluator, _selection_metric
from text_feedback_dpo.experiment_config import load_paper_experiment
from text_feedback_dpo.hyperparameter_search import build_dpo_candidates, build_grpo_candidates
from text_feedback_dpo.training import (
    build_paper_dpo_config_kwargs,
    build_paper_grpo_config_kwargs,
    build_optimizer_profile,
    materialize_warmup_steps,
)


class TrainingProfileTest(unittest.TestCase):
    def test_math_primary_and_length_desensitized_ledgers_are_objectively_labeled(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))
        primary = _paper_candidates(config, "standard_dpo")
        length_desensitized = _paper_candidates(config, "ld_dpo")

        self.assertEqual(len(primary), 12)
        self.assertTrue(all(candidate.loss_type == "sigmoid_norm" and candidate.ld_alpha is None for candidate in primary))
        self.assertEqual(len(length_desensitized), 36)
        self.assertEqual({candidate.ld_alpha for candidate in length_desensitized}, {0.25, 0.5, 0.75})
        self.assertTrue(all(candidate.loss_type == "sigmoid" for candidate in length_desensitized))
        self.assertEqual(_selection_metric(config, {"math": {"exact_accuracy": 0.6}}), 0.6)

    def test_paper_evaluator_uses_the_explicit_greedy_role_profile(self):
        config = load_paper_experiment(Path("configs/paper/gsm8k.yaml"))
        with mock.patch("text_feedback_dpo.cli.TransformersModelProvider") as provider_class:
            with mock.patch("text_feedback_dpo.cli.make_model_evaluator", return_value="evaluator") as make:
                result = _paper_evaluator(config)

        self.assertEqual(result, "evaluator")
        self.assertIs(make.call_args.kwargs["generate"], provider_class.return_value.generate_result)
        self.assertEqual(
            make.call_args.kwargs["generation_kwargs"],
            {"enable_thinking": False, "do_sample": False, "max_new_tokens": 256},
        )
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
            max_length=10240,
            loss_type="sigmoid_norm",
            ld_alpha=None,
        )
        self.assertEqual(dpo["max_length"], 10240)
        self.assertEqual(dpo["gradient_accumulation_steps"], 16)
        self.assertTrue(dpo["bf16"])
        self.assertEqual(dpo["beta"], 0.1)
        self.assertEqual(dpo["optim"], "adamw_torch_fused")
        self.assertEqual(dpo["loss_type"], "sigmoid_norm")
        self.assertNotIn("ld_alpha", dpo)

        ld_candidate = build_dpo_candidates(
            learning_rates=(5e-6,),
            betas=(0.1,),
            weight_decay=0.01,
            warmup_fraction=0.05,
            scheduler="cosine",
            loss_type="sigmoid",
            ld_alpha=0.5,
        )[0]
        ld_dpo = build_paper_dpo_config_kwargs(
            output_dir="out",
            max_steps=10,
            candidate=ld_candidate,
            effective_global_batch=16,
            max_length=10240,
            loss_type="sigmoid",
            ld_alpha=0.5,
        )
        self.assertEqual(ld_dpo["loss_type"], "sigmoid")
        self.assertEqual(ld_dpo["ld_alpha"], 0.5)

        grpo_candidate = build_grpo_candidates(learning_rates=(5e-6,), kl_betas=(0.01,))[0]
        grpo = build_paper_grpo_config_kwargs(
            output_dir="out",
            max_steps=10,
            candidate=grpo_candidate,
            max_completion_length=8192,
        )
        self.assertEqual(grpo["max_completion_length"], 8192)
        self.assertEqual(grpo["num_generations"], 4)
        self.assertEqual(grpo["generation_batch_size"], 4)
        self.assertEqual(grpo["epsilon"], 0.2)
        self.assertEqual(grpo["loss_type"], "grpo")
        self.assertEqual(grpo["beta"], 0.01)
        self.assertTrue(grpo["mask_truncated_completions"])
        self.assertEqual(grpo["temperature"], 1.0)
        self.assertEqual(grpo["top_p"], 0.95)
        self.assertEqual(grpo["top_k"], 20)
        self.assertTrue(grpo["use_vllm"])
        self.assertEqual(grpo["vllm_mode"], "colocate")
        self.assertEqual(grpo["generation_kwargs"], {"presence_penalty": 1.5})


if __name__ == "__main__":
    unittest.main()
