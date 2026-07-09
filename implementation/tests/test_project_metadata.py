import unittest
from pathlib import Path


class ProjectMetadataTest(unittest.TestCase):
    def test_pyproject_declares_cli_and_python_version(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.12"', text)
        self.assertIn('tfdpo = "text_feedback_dpo.cli:main"', text)


if __name__ == "__main__":
    unittest.main()
