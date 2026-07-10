import tempfile
import unittest
from pathlib import Path

from text_feedback_dpo.config import load_config


class ConfigTest(unittest.TestCase):
    def test_missing_student_model_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("run_id: bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "student_model"):
                load_config(path)

    def test_unknown_top_level_key_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "\n".join(
                    [
                        "run_id: bad",
                        "student_model: Qwen/Qwen3.5-2B",
                        "teacher_model: Qwen/Qwen3.5-9B",
                        "teacher_mode: stronger_model",
                        "max_examples: 5",
                        "output_dir: runs/test",
                        "generation: {}",
                        "teacher_generation: {}",
                        "slurm: {}",
                        "surprise: true",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)

    def test_smoke_config_loads(self):
        config = load_config(Path("configs/basic_smoke.yaml"))
        self.assertEqual(config["student_model"], "Qwen/Qwen3.5-2B")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(config["max_examples"], 5)
        for section in ("generation", "teacher_generation"):
            self.assertEqual(config[section]["temperature"], 1.0)
            self.assertEqual(config[section]["top_p"], 0.95)
            self.assertEqual(config[section]["top_k"], 20)
            self.assertEqual(config[section]["presence_penalty"], 1.5)

    def test_one_example_pair_validation_config_is_explicitly_bounded(self):
        config = load_config(Path("configs/pair_validation_1x.yaml"))
        self.assertEqual(config["run_id"], "qwen35-pair-validation-1x-r4")
        self.assertEqual(config["student_model"], "Qwen/Qwen3.5-2B")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(config["teacher_mode"], "stronger_model")
        self.assertEqual(config["max_examples"], 1)
        self.assertEqual(config["generation"]["max_new_tokens"], 2048)
        self.assertEqual(config["teacher_generation"]["max_new_tokens"], 4096)
        for section in ("generation", "teacher_generation"):
            self.assertEqual(config[section]["temperature"], 1.0)
            self.assertEqual(config[section]["top_p"], 0.95)
            self.assertEqual(config[section]["top_k"], 20)
            self.assertEqual(config[section]["presence_penalty"], 1.5)


if __name__ == "__main__":
    unittest.main()
