import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import run_native_pipeline


class ScriptedProvider:
    def __init__(self):
        self.student_outputs = iter(["wrong answer", "wrong answer", "correct answer"])
        self.teacher_outputs = iter(["Check the arithmetic operation.", "Recompute the operation."])
        self.calls = []

    def generate(self, role, prompt, **kwargs):
        self.calls.append((role, prompt, kwargs))
        if role == "student":
            return next(self.student_outputs)
        if role == "teacher":
            return next(self.teacher_outputs)
        raise AssertionError(f"unexpected role: {role}")


class NativeCliTest(unittest.TestCase):
    def test_native_pipeline_persists_attempts_and_multilevel_pairs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples_path = root / "examples.jsonl"
            examples_path.write_text(
                json.dumps(
                    {
                        "id": "m1",
                        "domain": "math",
                        "problem": "What is 2 + 2?",
                        "gold_answer": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run_id: native-test",
                        "student_model: student",
                        "teacher_model: teacher",
                        "evaluator_model: evaluator",
                        "teacher_mode: stronger_model",
                        f"examples_path: {examples_path}",
                        "max_examples: 1",
                        "output_dir: runs/native-test",
                        "max_guidance_steps: 3",
                        "max_guidance_regenerations: 1",
                        "generation: {}",
                        "teacher_generation: {}",
                        "evaluator_generation: {}",
                        "slurm: {}",
                        "allow_missing_slurm_account_for_local: true",
                    ]
                ),
                encoding="utf-8",
            )

            provider = ScriptedProvider()
            results = iter(
                [
                    {"correct": False, "answer": "5", "confidence": 0.9, "reason": "wrong"},
                    {"correct": False, "answer": "5", "confidence": 0.9, "reason": "wrong"},
                    {"correct": True, "answer": "4", "confidence": 0.9, "reason": "correct"},
                ]
            )

            result = run_native_pipeline(
                config_path=config_path,
                output_dir=root / "run",
                model_provider=provider,
                evaluator=lambda _example, _response: next(results),
                guidance_guard=lambda _example, _guidance, _result, _attempt: {
                    "safe": True,
                    "confidence": 0.9,
                    "reason": "hint only",
                },
            )

            self.assertEqual(result["accepted_pairs"], 2)
            self.assertEqual(len((root / "run" / "attempts.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len((root / "run" / "pairs.jsonl").read_text().splitlines()), 2)
            self.assertTrue((root / "run" / "metrics.json").exists())
            self.assertTrue(any(role == "teacher" for role, _, _ in provider.calls))


if __name__ == "__main__":
    unittest.main()
