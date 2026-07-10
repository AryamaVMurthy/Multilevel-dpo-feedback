import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.heldout import evaluate_checkpoint, validate_adapter_compatibility


class HeldoutTest(unittest.TestCase):
    def _evaluator(self, example, response):
        correct = response == "right"
        return {
            "correct": correct,
            "answer": "4" if correct else "5",
            "confidence": 0.9,
            "requires_model_judgment": False,
            "deterministic": {"numeric_exact_match": correct, "correct": correct},
        }

    def test_base_checkpoint_is_teacher_free_and_persists_raw_predictions(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute this value.", "gold_answer": "4"}]
        prompts = []
        with TemporaryDirectory() as tmp:
            result = evaluate_checkpoint(
                examples=examples,
                generate=lambda prompt: (prompts.append(prompt) or "right"),
                evaluator=self._evaluator,
                output_dir=Path(tmp),
                checkpoint_kind="base",
                base_model_revision="base-rev",
                seed=17,
                test=False,
            )
            self.assertEqual(result["common"]["final_answer_accuracy"], 1.0)
            self.assertNotIn("gold answer", prompts[0].lower())
            predictions = json.loads((Path(tmp) / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(predictions["response"], "right")
            self.assertTrue(predictions["teacher_free"])

    def test_test_evaluation_requires_freeze_and_cannot_repeat(self):
        examples = [{"id": "m1", "domain": "math", "problem": "Compute this value.", "gold_answer": "4"}]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({"schema": "hyperparameter-freeze-v1"}), encoding="utf-8")
            evaluate_checkpoint(
                examples=examples,
                generate=lambda _prompt: "right",
                evaluator=self._evaluator,
                output_dir=root / "test",
                checkpoint_kind="adapter",
                base_model_revision="base-rev",
                seed=17,
                test=True,
                freeze_manifest=freeze,
            )
            with self.assertRaisesRegex(FileExistsError, "test marker"):
                evaluate_checkpoint(
                    examples=examples,
                    generate=lambda _prompt: "right",
                    evaluator=self._evaluator,
                    output_dir=root / "test",
                    checkpoint_kind="adapter",
                    base_model_revision="base-rev",
                    seed=17,
                    test=True,
                    freeze_manifest=freeze,
                )

    def test_adapter_manifest_mismatch_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "base_model_revision": "base-rev",
                        "lora_coverage_hash": "coverage-a",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base_model_revision"):
                validate_adapter_compatibility(
                    adapter,
                    base_model_revision="other-rev",
                    lora_coverage_hash="coverage-a",
                )


if __name__ == "__main__":
    unittest.main()
