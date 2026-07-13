import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.artifacts import validate_artifacts
from text_feedback_dpo.dataset import build_sft_rows


class ArtifactTest(unittest.TestCase):
    def test_sft_rows_use_plain_completion_and_teacher_free_prompt(self):
        rows = build_sft_rows([{"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["completion"], "Ada")
        self.assertTrue(rows[0]["prompt"].endswith("Answer:"))
        self.assertNotIn("<response>", rows[0]["prompt"] + rows[0]["completion"])
        self.assertNotIn("gold_answer", rows[0]["prompt"])

    def test_validate_artifacts_fails_when_required_manifest_is_missing(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_artifacts(Path(tmp))


if __name__ == "__main__":
    unittest.main()
