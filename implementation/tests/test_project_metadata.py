import unittest
from pathlib import Path


class ProjectMetadataTest(unittest.TestCase):
    def test_pyproject_declares_cli_and_python_version(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.12"', text)
        self.assertIn('tfdpo = "text_feedback_dpo.cli:main"', text)

    def test_grpo_vllm_environment_is_isolated_from_the_main_lock(self):
        main = Path("pyproject.toml").read_text(encoding="utf-8")
        grpo = Path("environments/grpo/pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn('"vllm', main)
        self.assertIn('"vllm>=0.24.0"', grpo)
        self.assertIn('text-feedback-dpo = { path = "../..", editable = true }', grpo)
        self.assertTrue(Path("environments/grpo/uv.lock").exists())


if __name__ == "__main__":
    unittest.main()
