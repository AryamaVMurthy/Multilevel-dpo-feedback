import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.artifacts import validate_artifacts
from text_feedback_dpo.dataset import build_sft_rows


class ArtifactTest(unittest.TestCase):
    def test_sft_rows_use_xml_completion_and_teacher_free_prompt(self):
        rows = build_sft_rows([{"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"}])
        self.assertEqual(len(rows), 1)
        self.assertIn("<response>", rows[0]["text"])
        self.assertIn("<answer>Ada</answer>", rows[0]["text"])
        self.assertNotIn("gold_answer", rows[0]["text"])

    def test_validate_artifacts_fails_when_required_manifest_is_missing(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_artifacts(Path(tmp))


if __name__ == "__main__":
    unittest.main()
