import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import yaml

from text_feedback_dpo.cli import run_validate_paper_config
from text_feedback_dpo.cli import run_materialize_dataset
from text_feedback_dpo.experiment_config import load_paper_experiment


class PaperExperimentConfigTest(unittest.TestCase):
    def _load_mapping(self, relative_path: str) -> dict:
        value = yaml.safe_load(Path(relative_path).read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _write_and_load(self, value: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.yaml"
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            return load_paper_experiment(path)

    def test_math_config_freezes_exact_qwen3_protocol_and_optimizer_search(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))

        self.assertEqual(config.schema_version, 4)
        self.assertEqual(
            config.models,
            {
                "student": {
                    "id": "Qwen/Qwen3-4B",
                    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                },
                "teacher": {
                    "id": "Qwen/Qwen3-8B",
                    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                },
                "evaluator": {
                    "id": "Qwen/Qwen3-8B",
                    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                },
            },
        )
        student = config.generation.roles["student"]
        self.assertFalse(student.enable_thinking)
        self.assertTrue(student.do_sample)
        self.assertEqual(student.max_new_tokens, 8192)
        self.assertTrue(student.stop_after_final_answer)
        self.assertEqual(
            (
                student.temperature,
                student.top_p,
                student.top_k,
                student.min_p,
                student.presence_penalty,
                student.repetition_penalty,
            ),
            (0.7, 0.8, 20, 0.0, 1.5, 1.0),
        )
        for role, max_new_tokens in {
            "teacher": 64,
            "evaluator": 256,
            "guidance_guard": 8,
            "guidance_critic": 8,
        }.items():
            profile = config.generation.roles[role]
            self.assertFalse(profile.enable_thinking)
            self.assertFalse(profile.do_sample)
            self.assertEqual(profile.max_new_tokens, max_new_tokens)
            self.assertIsNone(profile.temperature)
        self.assertEqual(config.optimizer.name, "adamw_torch_fused")
        self.assertEqual(config.optimizer.adam_betas, (0.9, 0.999))
        self.assertEqual(config.optimizer.weight_decay, 0.01)
        self.assertEqual(config.optimizer.warmup_fraction, 0.05)
        self.assertEqual(config.dpo_search.learning_rates, (2e-6, 5e-6, 1e-5))
        self.assertEqual(config.dpo_search.betas, (0.05, 0.1, 0.3, 0.5))
        self.assertEqual(config.grpo_search.kl_betas, (0.0, 0.001, 0.01, 0.04))
        self.assertEqual(config.grpo_search.epsilon_low, 0.2)
        self.assertEqual(config.grpo_search.epsilon_high, 0.2)
        self.assertEqual(config.grpo_search.dapo_epsilon_high, 0.28)
        self.assertEqual(config.grpo_search.num_iterations, 2)
        self.assertEqual(config.lora.rank, 16)
        self.assertEqual(config.lora.target_policy, "qwen3_text_linear")
        self.assertEqual(config.training["max_sequence_tokens"], 18432)
        self.assertTrue(config.evaluation["baseline_before_training"])
        self.assertEqual(config.evaluation["generation_seed"], 20260711)
        self.assertEqual(config.evaluation["max_truncation_rate"], 0.05)
        self.assertEqual(config.evaluation["minimum_evaluator_audit_agreement"], 0.95)
        self.assertEqual(config.collection["feedback_policy"], "hint_only")
        self.assertTrue(config.require_freeze_manifest_for_test)

    def test_feedback_policy_is_required_and_validated(self):
        missing = self._load_mapping("configs/paper/math.yaml")
        del missing["collection"]["feedback_policy"]
        with self.assertRaisesRegex(ValueError, r"collection\.feedback_policy.*required"):
            self._write_and_load(missing)

        invalid = self._load_mapping("configs/paper/math.yaml")
        invalid["collection"]["feedback_policy"] = "automatic"
        with self.assertRaisesRegex(ValueError, r"collection\.feedback_policy"):
            self._write_and_load(invalid)

    def test_three_math_feedback_pilots_are_paired_except_for_policy_and_id(self):
        expected = {
            "error_only": "configs/pilots/math-feedback-error-only.yaml",
            "hint_only": "configs/pilots/math-feedback-hint-only.yaml",
            "error_and_hint": "configs/pilots/math-feedback-error-and-hint.yaml",
        }
        comparable = []
        for policy, path in expected.items():
            config = load_paper_experiment(Path(path))
            self.assertEqual(config.collection["feedback_policy"], policy)
            self.assertEqual(config.experiment_id, f"qwen3-math-feedback-pilot-{policy.replace('_', '-')}")
            value = asdict(config)
            value.pop("experiment_id")
            value["collection"].pop("feedback_policy")
            comparable.append(value)
        self.assertEqual(comparable[1:], comparable[:-1])

    def test_model_ids_and_revisions_are_exact_and_post_trained(self):
        for role, field, invalid in (
            ("student", "id", "Qwen/Qwen3-4B-Base"),
            ("student", "revision", "0" * 40),
            ("teacher", "id", "Qwen/Qwen3-8B-Base"),
            ("evaluator", "revision", "f" * 40),
        ):
            value = self._load_mapping("configs/paper/math.yaml")
            value["models"][role][field] = invalid
            with self.assertRaisesRegex(ValueError, rf"models\.{role}\.{field}.*exact frozen Qwen3"):
                self._write_and_load(value)

    def test_searchqa_config_has_disjoint_auxiliary_hparam_roles(self):
        config = load_paper_experiment(Path("configs/paper/searchqa8k.yaml"))

        self.assertEqual(config.dataset.source_counts, {"train": 99820, "validation": 13393, "test": 27248})
        self.assertEqual(config.dataset.splits, {"train": 5000, "validation": 1000, "test": 2000})
        self.assertEqual(config.dataset.auxiliary_hparam, {"train": 2000, "validation": 500})
        self.assertFalse(config.generation.roles["student"].stop_after_final_answer)
        self.assertEqual(config.collection["prompt_protocol"], "qwen3-nonthinking-r1")

    def test_math_config_pins_all_subjects_and_derives_primary_split_counts(self):
        config = load_paper_experiment(Path("configs/paper/math.yaml"))

        self.assertEqual(config.dataset.source, "EleutherAI/hendrycks_math")
        self.assertEqual(config.dataset.revision, "21a5633873b6a120296cce3e2df9d5550074f4a3")
        self.assertEqual(config.dataset.source_counts, {"train": 7500, "validation": 0, "test": 5000})
        self.assertEqual(config.dataset.splits, {"train": 0, "validation": 0, "test": 5000})
        self.assertEqual(config.dataset.primary_levels, (4, 5))
        self.assertEqual(config.dataset.train_fraction, 0.9)
        self.assertEqual(config.dataset.validation_tune_fraction, 2 / 3)
        self.assertEqual(len(config.dataset.subjects), 7)
        self.assertTrue(config.generation.roles["student"].stop_after_final_answer)
        self.assertEqual(config.collection["prompt_protocol"], "qwen3-nonthinking-final-r1")

    def test_math_config_rejects_noncanonical_subject_order_or_primary_levels(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["dataset"]["subjects"] = list(reversed(value["dataset"]["subjects"]))
        with self.assertRaisesRegex(ValueError, "official MATH subjects"):
            self._write_and_load(value)

        value = self._load_mapping("configs/paper/math.yaml")
        value["dataset"]["primary_levels"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "primary_levels"):
            self._write_and_load(value)

    def test_unknown_nested_optimizer_key_fails_with_field_path(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["optimizer"]["silent_magic"] = True

        with self.assertRaisesRegex(ValueError, r"optimizer\.silent_magic"):
            self._write_and_load(value)

    def test_missing_dataset_revision_fails_with_remediation(self):
        value = self._load_mapping("configs/paper/math.yaml")
        del value["dataset"]["revision"]

        with self.assertRaisesRegex(ValueError, r"dataset\.revision.*required"):
            self._write_and_load(value)

    def test_searchqa_mrqa_mirror_is_rejected(self):
        value = self._load_mapping("configs/paper/searchqa8k.yaml")
        value["dataset"]["source"] = "lucadiliello/searchqa"

        with self.assertRaisesRegex(ValueError, "original SearchQA"):
            self._write_and_load(value)

    def test_deprecated_warmup_ratio_fails_explicitly(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["optimizer"]["warmup_ratio"] = value["optimizer"].pop("warmup_fraction")

        with self.assertRaisesRegex(ValueError, r"optimizer\.warmup_ratio.*deprecated"):
            self._write_and_load(value)

    def test_non_8192_student_completion_budget_fails(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["student"]["max_new_tokens"] = 2048

        with self.assertRaisesRegex(ValueError, r"generation\.student\.max_new_tokens.*8192"):
            self._write_and_load(value)

    def test_every_role_rejects_thinking_or_generation_above_ceiling(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["teacher"]["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, r"generation\.teacher.*enable_thinking=false"):
            self._write_and_load(value)

        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["evaluator"]["max_new_tokens"] = 8193
        with self.assertRaisesRegex(ValueError, r"generation\.evaluator\.max_new_tokens.*8192"):
            self._write_and_load(value)

    def test_math_requires_final_answer_stopping(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["student"]["stop_after_final_answer"] = False
        with self.assertRaisesRegex(ValueError, "final-answer stopping"):
            self._write_and_load(value)

    def test_paper_student_rejects_thinking_mode_or_old_sampling_profile(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["student"]["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "non-thinking sampled"):
            self._write_and_load(value)

        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["student"]["top_p"] = 0.95
        with self.assertRaisesRegex(ValueError, "top_p=0.8"):
            self._write_and_load(value)

    def test_primary_dpo_objective_and_length_ablation_grid_are_frozen(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["dpo_search"]["loss_type"] = "sigmoid"
        with self.assertRaisesRegex(ValueError, "sigmoid_norm"):
            self._write_and_load(value)

        value = self._load_mapping("configs/paper/math.yaml")
        value["dpo_search"]["ld_alpha_values"] = [0.5]
        with self.assertRaisesRegex(ValueError, "ld_alpha_values"):
            self._write_and_load(value)

    def test_missing_role_generation_profile_fails_explicitly(self):
        value = self._load_mapping("configs/paper/math.yaml")
        del value["generation"]["guidance_critic"]

        with self.assertRaisesRegex(ValueError, r"generation\.guidance_critic.*required"):
            self._write_and_load(value)

    def test_greedy_role_rejects_silent_sampling_parameters(self):
        value = self._load_mapping("configs/paper/math.yaml")
        value["generation"]["teacher"]["temperature"] = 1.0

        with self.assertRaisesRegex(ValueError, r"generation\.teacher\.temperature.*unknown"):
            self._write_and_load(value)

    def test_overlap_prone_searchqa_auxiliary_counts_fail(self):
        value = self._load_mapping("configs/paper/searchqa8k.yaml")
        value["dataset"]["auxiliary_hparam"]["train"] = 96000

        with self.assertRaisesRegex(ValueError, r"dataset\.auxiliary_hparam\.train.*unused"):
            self._write_and_load(value)

    def test_cli_validation_returns_identity_and_frozen_protocol(self):
        result = run_validate_paper_config(Path("configs/paper/math.yaml"))

        self.assertEqual(result["experiment_id"], "qwen3-paper-math-level45")
        self.assertEqual(result["dataset"], "math")
        self.assertTrue(result["require_freeze_manifest_for_test"])

    def test_materialize_cli_validates_config_before_delegating(self):
        materialized = {
            "output_dir": "/tmp/math-output",
            "manifest": {
                "schema": "paper-dataset-manifest-v1",
                "metadata": {"dataset": "math"},
                "roles": {"train": 10, "validation": 2, "test": 3},
                "nested_roles": {"tune": 1, "confirm": 1},
                "content_sha256": "a" * 64,
            },
        }
        with mock.patch("text_feedback_dpo.cli.materialize_paper_dataset", return_value=materialized) as materialize:
            result = run_materialize_dataset(
                Path("configs/paper/math.yaml"),
                Path("/tmp/math-source"),
                Path("/tmp/math-output"),
            )

        self.assertEqual(result["schema"], "paper-dataset-materialization-summary-v1")
        self.assertEqual(result["dataset"], "math")
        self.assertEqual(result["roles"], {"train": 10, "validation": 2, "test": 3})
        self.assertEqual(result["content_sha256"], "a" * 64)
        self.assertEqual(materialize.call_args.args[1:], (Path("/tmp/math-source"), Path("/tmp/math-output")))


if __name__ == "__main__":
    unittest.main()
