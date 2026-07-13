import unittest
from pathlib import Path

from text_feedback_dpo.config import load_config


class TuringRuntimeTest(unittest.TestCase):
    def test_primary_config_is_searchqa_4b_full_finetune_with_quantized_teacher_candidates(self):
        config = load_config(Path("configs/searchqa.yaml"))
        self.assertEqual(config["student_model"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen3-32B")
        self.assertEqual(config["training"]["teacher_fallback_model"], "Qwen/Qwen3-14B")
        self.assertEqual(config["training"]["teacher_quantization"], "4bit")
        self.assertTrue(config["training"]["full_finetuning"])

    def test_turing_scripts_fail_fast_and_use_no_hidden_fallback(self):
        for path in Path("scripts").glob("*.sh"):
            text = path.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", text)
            self.assertNotIn("|| true", text)

    def test_generation_is_plain_short_answer_with_explicit_thinking_mode(self):
        text = Path("scripts/turing_generate.sh").read_text(encoding="utf-8")
        self.assertIn("--max-new-tokens 32", text)
        self.assertIn("STUDENT_THINKING_MODE", text)
        self.assertNotIn("--max-new-tokens 512", text)

    def test_collection_uses_one_explicit_teacher_identity_and_complete_cache_key(self):
        text = Path("scripts/turing_collect.sh").read_text(encoding="utf-8")
        for value in ("STUDENT_REVISION", "DATASET_REVISION", "PROMPT_VERSION", "SEED", "--teacher-thinking"):
            self.assertIn(value, text)
        self.assertNotIn("TEACHER_FALLBACK", text)

    def test_gpu_scripts_use_conservative_uv_and_verify_source_root(self):
        for name in ("turing_generate.sh", "turing_collect.sh", "turing_train.sh", "turing_preflight.sh"):
            text = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("UV_CONCURRENT_DOWNLOADS=1", text)
            self.assertIn("src/text_feedback_dpo", text)

    def test_retirement_script_is_scoped_and_archive_first(self):
        text = Path("scripts/turing_retire_invalid_run.sh").read_text(encoding="utf-8")
        self.assertIn("EXPECTED_INVALID_RUN_ROOT", text)
        self.assertIn("realpath", text)
        self.assertIn("ARCHIVE_DIR", text)
        self.assertIn('$actual/baseline/shards', text)
        self.assertNotIn("rm -rf", text)

    def test_prompt_preflight_compares_direct_and_two_pass(self):
        text = Path("scripts/turing_prompt_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("direct", text)
        self.assertIn("two_pass", text)
        self.assertIn("preflight-quality", text)
        self.assertIn("select-thinking-mode", text)

    def test_model_preflight_actually_runs_role_specific_probe(self):
        text = Path("scripts/turing_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL_ROLE", text)
        self.assertIn("probe-model", text)
        self.assertIn("--teacher-quantization 4bit", text)

    def test_training_uses_configurable_multi_gpu_with_fixed_effective_batch(self):
        text = Path("scripts/turing_train.sh").read_text(encoding="utf-8")
        self.assertIn('TRAIN_GPUS" != "2" && "$TRAIN_GPUS" != "4"', text)
        self.assertIn("SLURM_GPUS_ON_NODE", text)
        self.assertIn("EFFECTIVE_BATCH_SIZE", text)
        self.assertIn("GRADIENT_ACCUMULATION_STEPS", text)
        self.assertIn('--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"', text)
        self.assertIn('--nproc_per_node="$TRAIN_GPUS"', text)


if __name__ == "__main__":
    unittest.main()
