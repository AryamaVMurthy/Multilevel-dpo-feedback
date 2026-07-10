import tempfile
import unittest
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

    def test_gsm8k_config_freezes_optimizer_search_and_nested_validation(self):
        config = load_paper_experiment(Path("configs/paper/gsm8k.yaml"))

        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.dataset.revision, "740312add88f781978c0658806c59bc2815b9866")
        self.assertEqual(config.dataset.splits, {"train": 6726, "validation": 747, "test": 1319})
        self.assertEqual(config.dataset.validation_roles, {"tune": 500, "confirm": 247})
        student = config.generation.roles["student"]
        self.assertTrue(student.enable_thinking)
        self.assertTrue(student.do_sample)
        self.assertEqual(student.max_new_tokens, 8192)
        self.assertEqual(
            (student.temperature, student.top_p, student.top_k, student.presence_penalty),
            (1.0, 0.95, 20, 1.5),
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
        self.assertEqual(config.lora.rank, 16)
        self.assertEqual(config.lora.target_policy, "qwen35_text_linear")
        self.assertEqual(config.training["max_sequence_tokens"], 10240)
        self.assertTrue(config.evaluation["baseline_before_training"])
        self.assertEqual(config.evaluation["generation_seed"], 20260710)
        self.assertEqual(config.evaluation["max_truncation_rate"], 0.05)
        self.assertEqual(config.evaluation["minimum_evaluator_audit_agreement"], 0.95)
        self.assertTrue(config.require_freeze_manifest_for_test)

    def test_searchqa_config_has_disjoint_auxiliary_hparam_roles(self):
        config = load_paper_experiment(Path("configs/paper/searchqa8k.yaml"))

        self.assertEqual(config.dataset.source_counts, {"train": 99820, "validation": 13393, "test": 27248})
        self.assertEqual(config.dataset.splits, {"train": 5000, "validation": 1000, "test": 2000})
        self.assertEqual(config.dataset.auxiliary_hparam, {"train": 2000, "validation": 500})

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
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        value["optimizer"]["silent_magic"] = True

        with self.assertRaisesRegex(ValueError, r"optimizer\.silent_magic"):
            self._write_and_load(value)

    def test_missing_dataset_revision_fails_with_remediation(self):
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        del value["dataset"]["revision"]

        with self.assertRaisesRegex(ValueError, r"dataset\.revision.*required"):
            self._write_and_load(value)

    def test_searchqa_mrqa_mirror_is_rejected(self):
        value = self._load_mapping("configs/paper/searchqa8k.yaml")
        value["dataset"]["source"] = "lucadiliello/searchqa"

        with self.assertRaisesRegex(ValueError, "original SearchQA"):
            self._write_and_load(value)

    def test_deprecated_warmup_ratio_fails_explicitly(self):
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        value["optimizer"]["warmup_ratio"] = value["optimizer"].pop("warmup_fraction")

        with self.assertRaisesRegex(ValueError, r"optimizer\.warmup_ratio.*deprecated"):
            self._write_and_load(value)

    def test_non_8192_student_completion_budget_fails(self):
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        value["generation"]["student"]["max_new_tokens"] = 2048

        with self.assertRaisesRegex(ValueError, r"generation\.student\.max_new_tokens.*8192"):
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
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        del value["generation"]["guidance_critic"]

        with self.assertRaisesRegex(ValueError, r"generation\.guidance_critic.*required"):
            self._write_and_load(value)

    def test_greedy_role_rejects_silent_sampling_parameters(self):
        value = self._load_mapping("configs/paper/gsm8k.yaml")
        value["generation"]["teacher"]["temperature"] = 1.0

        with self.assertRaisesRegex(ValueError, r"generation\.teacher\.temperature.*unknown"):
            self._write_and_load(value)

    def test_overlap_prone_searchqa_auxiliary_counts_fail(self):
        value = self._load_mapping("configs/paper/searchqa8k.yaml")
        value["dataset"]["auxiliary_hparam"]["train"] = 96000

        with self.assertRaisesRegex(ValueError, r"dataset\.auxiliary_hparam\.train.*unused"):
            self._write_and_load(value)

    def test_cli_validation_returns_identity_and_frozen_protocol(self):
        result = run_validate_paper_config(Path("configs/paper/gsm8k.yaml"))

        self.assertEqual(result["experiment_id"], "qwen35-paper-gsm8k")
        self.assertEqual(result["dataset"], "gsm8k")
        self.assertTrue(result["require_freeze_manifest_for_test"])

    def test_materialize_cli_validates_config_before_delegating(self):
        with mock.patch("text_feedback_dpo.cli.materialize_paper_dataset", return_value={"ok": True}) as materialize:
            result = run_materialize_dataset(
                Path("configs/paper/gsm8k.yaml"),
                Path("/tmp/gsm8k-source"),
                Path("/tmp/gsm8k-output"),
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(materialize.call_args.args[1:], (Path("/tmp/gsm8k-source"), Path("/tmp/gsm8k-output")))


if __name__ == "__main__":
    unittest.main()
