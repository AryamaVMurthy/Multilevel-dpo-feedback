import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from text_feedback_dpo.cli import (
    _paper_candidates,
    _paper_evaluator,
    _selection_metric,
    _tuning_validation_failure,
    run_train_paper,
)
from text_feedback_dpo.experiment_config import load_paper_experiment
from text_feedback_dpo.hyperparameter_search import build_dpo_candidates, build_grpo_candidates
from text_feedback_dpo.training import (
    build_paper_dpo_config_kwargs,
    build_paper_grpo_config_kwargs,
    build_paper_sft_config_kwargs,
    build_optimizer_profile,
    materialize_warmup_steps,
)


class TrainingProfileTest(unittest.TestCase):
    def test_tuning_validation_gate_rejects_failures_and_excess_truncation(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))
        with TemporaryDirectory() as tmp:
            failures = Path(tmp) / "failures.jsonl"
            failures.write_text("", encoding="utf-8")
            self.assertIsNone(
                _tuning_validation_failure(
                    config,
                    {"common": {"truncation_rate": 0.10}},
                    failures_path=failures,
                )
            )
            self.assertIn(
                "truncation rate",
                _tuning_validation_failure(
                    config,
                    {"common": {"truncation_rate": 0.101}},
                    failures_path=failures,
                ),
            )
            failures.write_text('{"error":"parse"}\n', encoding="utf-8")
            self.assertIn(
                "failures",
                _tuning_validation_failure(
                    config,
                    {"common": {"truncation_rate": 0.0}},
                    failures_path=failures,
                ),
            )

    def test_response_sft_reuses_frozen_standard_dpo_optimizer_candidate(self):
        candidate = _paper_candidates(
            load_paper_experiment(Path("configs/paper/math.yaml")),
            "standard_dpo",
        )[0]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "response_sft.jsonl"
            data.write_text(
                json.dumps({"prompt": "Solve.", "completion": "FINAL: \\boxed{4}"}) + "\n",
                encoding="utf-8",
            )
            freeze = root / "standard-freeze.json"
            freeze.write_text(
                json.dumps({
                    "method": "standard_dpo",
                    "candidate_id": candidate.candidate_id,
                    "candidate": vars(candidate),
                }),
                encoding="utf-8",
            )
            with mock.patch(
                "text_feedback_dpo.cli.train_paper_sft",
                return_value={"method": "response_sft"},
            ) as train:
                result = run_train_paper(
                    config_path=Path("configs/paper/math.yaml"),
                    method="response_sft",
                    seed=17,
                    data_path=data,
                    freeze_manifest_path=freeze,
                    output_dir=root / "output",
                )

        self.assertEqual(result["method"], "response_sft")
        self.assertEqual(train.call_args.kwargs["candidate"], candidate)
        self.assertEqual(train.call_args.kwargs["rows"][0]["prompt"], "Solve.")

    def test_multilevel_and_matched_reuse_frozen_standard_optimizer_candidate(self):
        candidate = _paper_candidates(
            load_paper_experiment(Path("configs/paper/math.yaml")),
            "standard_dpo",
        )[0]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "pairs.jsonl"
            data.write_text(
                json.dumps({"prompt": "Solve.", "chosen": "right", "rejected": "wrong"}) + "\n",
                encoding="utf-8",
            )
            freeze = root / "standard-freeze.json"
            freeze.write_text(
                json.dumps({
                    "method": "standard_dpo",
                    "candidate_id": candidate.candidate_id,
                    "candidate": vars(candidate),
                }),
                encoding="utf-8",
            )
            for method in ("multilevel_dpo", "matched_dpo"):
                with self.subTest(method=method), mock.patch(
                    "text_feedback_dpo.cli.train_paper_dpo",
                    return_value={"method": method},
                ) as train:
                    result = run_train_paper(
                        config_path=Path("configs/paper/math.yaml"),
                        method=method,
                        seed=17,
                        data_path=data,
                        freeze_manifest_path=freeze,
                        output_dir=root / method,
                    )
                    self.assertEqual(result["method"], method)
                    self.assertEqual(train.call_args.kwargs["candidate"], candidate)

    def test_math_primary_and_length_desensitized_ledgers_are_objectively_labeled(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))
        primary = _paper_candidates(config, "standard_dpo")
        length_desensitized = _paper_candidates(config, "ld_dpo")

        self.assertEqual(len(primary), 96)
        self.assertEqual({candidate.weight_decay for candidate in primary}, {0.0, 0.01})
        self.assertEqual({candidate.warmup_fraction for candidate in primary}, {0.05, 0.1})
        self.assertEqual({candidate.scheduler for candidate in primary}, {"linear", "cosine"})
        self.assertTrue(all(candidate.loss_type == "sigmoid_norm" and candidate.ld_alpha is None for candidate in primary))
        self.assertEqual(len(length_desensitized), 288)
        self.assertEqual({candidate.ld_alpha for candidate in length_desensitized}, {0.25, 0.5, 0.75})
        self.assertTrue(all(candidate.loss_type == "sigmoid" for candidate in length_desensitized))
        self.assertEqual(_selection_metric(config, {"math": {"exact_accuracy": 0.6}}), 0.6)

    def test_grpo_and_dapo_have_independent_clipped_candidate_ledgers(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))
        grpo = _paper_candidates(config, "grpo")
        dapo = _paper_candidates(config, "dapo")

        self.assertEqual(len(grpo), 12)
        self.assertEqual(len(dapo), 12)
        self.assertTrue(all(candidate.loss_type == "grpo" for candidate in grpo))
        self.assertTrue(all(candidate.epsilon_low == 0.2 for candidate in grpo))
        self.assertTrue(all(candidate.epsilon_high == 0.2 for candidate in grpo))
        self.assertTrue(all(candidate.num_iterations == 2 for candidate in grpo))
        self.assertTrue(all(candidate.loss_type == "dapo" for candidate in dapo))
        self.assertTrue(all(candidate.epsilon_low == 0.2 for candidate in dapo))
        self.assertTrue(all(candidate.epsilon_high == 0.28 for candidate in dapo))
        self.assertNotEqual(grpo[0].candidate_id, dapo[0].candidate_id)

    def test_paper_evaluator_uses_the_explicit_greedy_role_profile(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))
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
            max_length=18432,
            loss_type="sigmoid_norm",
            ld_alpha=None,
        )
        self.assertEqual(dpo["max_length"], 18432)
        self.assertEqual(dpo["gradient_accumulation_steps"], 16)
        self.assertTrue(dpo["bf16"])
        self.assertTrue(dpo["gradient_checkpointing"])
        self.assertFalse(dpo["use_cache"])
        self.assertEqual(dpo["save_strategy"], "steps")
        self.assertEqual(dpo["save_steps"], 0.25)
        self.assertEqual(dpo["save_total_limit"], 4)
        self.assertTrue(dpo["save_only_model"])

        sft = build_paper_sft_config_kwargs(
            output_dir="out",
            max_steps=10,
            candidate=dpo_candidate,
            effective_global_batch=16,
            max_length=18432,
        )
        self.assertEqual(sft["max_length"], 18432)
        self.assertEqual(sft["gradient_accumulation_steps"], 16)
        self.assertTrue(sft["completion_only_loss"])
        self.assertTrue(sft["gradient_checkpointing"])
        self.assertTrue(sft["bf16"])
        self.assertEqual(sft["optim"], "adamw_torch_fused")
        self.assertEqual(sft["save_steps"], 0.25)
        self.assertTrue(sft["save_only_model"])

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
            max_length=18432,
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
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )
        self.assertEqual(grpo["max_completion_length"], 8192)
        self.assertEqual(grpo["vllm_max_model_length"], 10240)
        self.assertEqual(grpo["num_generations"], 4)
        self.assertEqual(grpo["generation_batch_size"], 4)
        self.assertEqual(grpo["epsilon"], 0.2)
        self.assertEqual(grpo["epsilon_high"], 0.2)
        self.assertEqual(grpo["num_iterations"], 2)
        self.assertEqual(grpo["loss_type"], "grpo")
        self.assertEqual(grpo["beta"], 0.01)
        self.assertTrue(grpo["mask_truncated_completions"])
        self.assertEqual(grpo["save_steps"], 0.25)
        self.assertTrue(grpo["save_only_model"])
        self.assertEqual(grpo["temperature"], 0.7)
        self.assertEqual(grpo["top_p"], 0.8)
        self.assertEqual(grpo["top_k"], 20)
        self.assertEqual(grpo["min_p"], 0.0)
        self.assertEqual(grpo["repetition_penalty"], 1.0)
        self.assertEqual(grpo["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(grpo["use_vllm"])
        self.assertEqual(grpo["vllm_mode"], "colocate")
        self.assertEqual(grpo["generation_kwargs"], {"presence_penalty": 0.0})

        with self.assertRaisesRegex(ValueError, "must not exceed 8192"):
            build_paper_grpo_config_kwargs(
                output_dir="out",
                max_steps=10,
                candidate=grpo_candidate,
                max_completion_length=8193,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )


if __name__ == "__main__":
    unittest.main()
