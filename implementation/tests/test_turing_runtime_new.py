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


if __name__ == "__main__":
    unittest.main()
