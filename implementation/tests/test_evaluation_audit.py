import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.evaluation_audit import (
    audit_checkpoint_evaluation,
    rescore_checkpoint_evaluation,
)
from text_feedback_dpo.io import write_jsonl


class EvaluationAuditTest(unittest.TestCase):
    def test_rescore_preserves_inputs_and_repairs_deterministic_decision(self):
        predictions = [
            {
                "id": "m1",
                "teacher_free": True,
                "prompt_tokens": 10,
                "generated_tokens": 20,
                "terminated": True,
                "truncated": False,
                "finish_reason": "final_answer",
                "evaluator_result": {
                    "answer": "\\sqrt{3}-3",
                    "correct": True,
                    "model_correct": True,
                },
            },
            {
                "id": "m2",
                "teacher_free": True,
                "prompt_tokens": 11,
                "generated_tokens": 21,
                "terminated": True,
                "truncated": False,
                "finish_reason": "final_answer",
                "evaluator_result": {
                    "answer": "45/1024",
                    "correct": True,
                    "model_correct": True,
                },
            },
        ]
        examples = [
            {
                "id": "m1",
                "domain": "math",
                "source": "EleutherAI/hendrycks_math",
                "gold_answer": "\\frac{1}{2^{98}}",
            },
            {
                "id": "m2",
                "domain": "math",
                "source": "EleutherAI/hendrycks_math",
                "gold_answer": "\\dfrac{45}{1024}",
            },
        ]
        labels = [
            {"id": "m1", "manual_correct": False, "notes": "Wrong magnitude."},
            {"id": "m2", "manual_correct": True, "notes": "Equivalent fraction."},
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "predictions.jsonl", predictions)
            write_jsonl(root / "examples.jsonl", examples)
            write_jsonl(root / "labels.jsonl", labels)
            result = rescore_checkpoint_evaluation(
                predictions_path=root / "predictions.jsonl",
                examples_path=root / "examples.jsonl",
                output_dir=root / "rescore",
                source_commit="a" * 40,
            )
            self.assertEqual(result["original_correct"], 2)
            self.assertEqual(result["rescored_correct"], 1)
            self.assertEqual(result["changed_ids"], ["m1"])
            self.assertEqual(result["requires_model_judgment"], 0)
            rescored = [
                json.loads(line)
                for line in (root / "rescore" / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertFalse(rescored[0]["evaluator_result"]["correct"])
            self.assertEqual(rescored[0]["rescore_source_commit"], "a" * 40)
            audit = audit_checkpoint_evaluation(
                predictions_path=root / "rescore" / "predictions.jsonl",
                labels_path=root / "labels.jsonl",
                output_dir=root / "audit",
                minimum_labels=2,
                minimum_agreement=0.95,
                max_truncation_rate=0.05,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["manual_agreement"], 1.0)

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

    def test_rescore_uses_deterministic_truth_when_model_disagrees(self):
        predictions = [
            {
                "id": "m1",
                "teacher_free": True,
                "prompt_tokens": 10,
                "generated_tokens": 20,
                "terminated": True,
                "truncated": False,
                "finish_reason": "final_answer",
                "evaluator_result": {
                    "answer": "6",
                    "correct": False,
                    "model_correct": False,
                },
            }
        ]
        examples = [
            {
                "id": "m1",
                "domain": "math",
                "source": "EleutherAI/hendrycks_math",
                "gold_answer": "6",
            }
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "predictions.jsonl", predictions)
            write_jsonl(root / "examples.jsonl", examples)
            result = rescore_checkpoint_evaluation(
                predictions_path=root / "predictions.jsonl",
                examples_path=root / "examples.jsonl",
                output_dir=root / "rescore",
                source_commit="b" * 40,
            )
            self.assertEqual(result["rescored_correct"], 1)
            self.assertEqual(result["changed_ids"], ["m1"])
            rescored = [
                json.loads(line)
                for line in (root / "rescore" / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertTrue(rescored[0]["evaluator_result"]["correct"])

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
