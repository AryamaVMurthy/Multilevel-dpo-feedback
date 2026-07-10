import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.evaluation_audit import audit_checkpoint_evaluation
from text_feedback_dpo.io import write_jsonl


class EvaluationAuditTest(unittest.TestCase):
    def test_teacher_free_exact_metadata_and_manual_agreement_gate(self):
        predictions = [
            {
                "id": "m1",
                "teacher_free": True,
                "prompt_tokens": 10,
                "generated_tokens": 20,
                "terminated": True,
                "truncated": False,
                "finish_reason": "eos",
                "evaluator_result": {"correct": True},
            },
            {
                "id": "m2",
                "teacher_free": True,
                "prompt_tokens": 11,
                "generated_tokens": 21,
                "terminated": True,
                "truncated": False,
                "finish_reason": "eos",
                "evaluator_result": {"correct": False},
            },
        ]
        labels = [
            {"id": "m1", "manual_correct": True, "notes": "Exact answer."},
            {"id": "m2", "manual_correct": False, "notes": "Wrong answer."},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "predictions.jsonl", predictions)
            write_jsonl(root / "labels.jsonl", labels)
            result = audit_checkpoint_evaluation(
                predictions_path=root / "predictions.jsonl",
                labels_path=root / "labels.jsonl",
                output_dir=root / "audit",
                minimum_labels=2,
                minimum_agreement=0.95,
                max_truncation_rate=0.05,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["manual_agreement"], 1.0)
            self.assertEqual(result["truncation_rate"], 0.0)
            self.assertTrue((root / "audit" / "report.html").exists())

    def test_disagreement_or_missing_generation_metadata_fails_gate(self):
        predictions = [
            {
                "id": "m1",
                "teacher_free": True,
                "prompt_tokens": None,
                "generated_tokens": 20,
                "terminated": False,
                "truncated": True,
                "finish_reason": "length",
                "evaluator_result": {"correct": True},
            }
        ]
        labels = [{"id": "m1", "manual_correct": False, "notes": "Incomplete."}]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "predictions.jsonl", predictions)
            write_jsonl(root / "labels.jsonl", labels)
            result = audit_checkpoint_evaluation(
                predictions_path=root / "predictions.jsonl",
                labels_path=root / "labels.jsonl",
                output_dir=root / "audit",
                minimum_labels=1,
                minimum_agreement=0.95,
                max_truncation_rate=0.05,
            )
            self.assertFalse(result["passed"])
            self.assertIn("manual_agreement", result["failed_gates"])
            self.assertIn("generation_metadata", result["failed_gates"])
            self.assertIn("truncation_rate", result["failed_gates"])
            disagreements = [json.loads(line) for line in (root / "audit" / "disagreements.jsonl").read_text().splitlines()]
            self.assertEqual(disagreements[0]["id"], "m1")


if __name__ == "__main__":
    unittest.main()
