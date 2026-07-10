import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.validation import validate_observability_manifest, validate_run


class RunValidationTest(unittest.TestCase):
    def _write_valid_native_run(self, output: Path) -> None:
        output.mkdir(parents=True)
        (output / "events.jsonl").write_text(
            json.dumps({"event_name": "run_start", "run_id": "native-1"}) + "\n"
            + json.dumps({"event_name": "run_end", "run_id": "native-1"})
            + "\n",
            encoding="utf-8",
        )
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": "native-1",
                    "method": "native_iterative_guidance_dpo",
                    "student_model": "Qwen/Qwen3.5-2B",
                    "teacher_model": "Qwen/Qwen3.5-9B",
                    "accepted_pairs": 1,
                }
            ),
            encoding="utf-8",
        )
        example = {"id": "m1", "domain": "math", "problem": "2+2", "gold_answer": "4"}
        attempt = {"id": "m1", "attempt": 0, "response": "5", "result": {"correct": False}}
        pair = {"id": "m1", "prompt": "2+2", "chosen": "4", "rejected": "5", "metadata": {}}
        for name, rows in {
            "examples.jsonl": [example],
            "attempts.jsonl": [attempt],
            "guidance.jsonl": [],
            "generation_events.jsonl": [],
            "pairs.jsonl": [pair],
            "response_sft.jsonl": [],
            "failures.jsonl": [],
        }.items():
            (output / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        (output / "gpu-1.csv").write_text("timestamp,memory.used\n", encoding="utf-8")

    def _write_valid_run(self, output: Path) -> None:
        output.mkdir(parents=True)
        (output / "events.jsonl").write_text(
            json.dumps({"event_name": "run_start", "run_id": "validation-1"}) + "\n"
            + json.dumps({"event_name": "run_end", "run_id": "validation-1"})
            + "\n",
            encoding="utf-8",
        )
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": "validation-1",
                    "student_model": "Qwen/Qwen3.5-2B",
                    "teacher_model": "Qwen/Qwen3.5-9B",
                    "accepted_pairs": 1,
                }
            ),
            encoding="utf-8",
        )
        pair = {
            "id": "example-1",
            "prompt": "Solve 2 + 2 using the required format.",
            "chosen": "<final>4</final>",
            "rejected": "<final>5</final>",
            "metadata": {"feedback": "Recompute the arithmetic."},
        }
        for name, rows in {
            "examples.jsonl": [{"id": "example-1", "gold_answer": "4"}],
            "rollouts.jsonl": [{"id": "example-1"}],
            "corrections.jsonl": [{"id": "example-1"}],
            "pairs.jsonl": [pair],
            "rejections.jsonl": [],
        }.items():
            (output / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        (output / "gpu-1.csv").write_text("timestamp,memory.used\n", encoding="utf-8")

    def test_valid_complete_run_writes_validation_summary(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self._write_valid_run(output)

            result = validate_run(output)

            self.assertTrue(result["valid"])
            self.assertTrue((output / "validation.json").exists())
            self.assertEqual(result["accepted_pairs"], 1)

    def test_run_with_teacher_feedback_in_prompt_fails_validation(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self._write_valid_run(output)
            pair_path = output / "pairs.jsonl"
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            pair["prompt"] = "Solve 2 + 2. Recompute the arithmetic. Gold answer: 4"
            pair_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "leaks teacher feedback or gold answer"):
                validate_run(output)

    def test_run_with_zero_accepted_pairs_fails_validation(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self._write_valid_run(output)
            (output / "metrics.json").write_text(
                json.dumps(
                    {
                        "run_id": "validation-1",
                        "student_model": "Qwen/Qwen3.5-2B",
                        "teacher_model": "Qwen/Qwen3.5-9B",
                        "accepted_pairs": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "accepted_pairs must be positive"):
                validate_run(output)

    def test_validate_run_cli_writes_validation_summary(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self._write_valid_run(output)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "text_feedback_dpo.cli",
                    "validate-run",
                    "--output-dir",
                    str(output),
                ],
                env={**os.environ, "PYTHONPATH": "src"},
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "validation.json").exists())

    def test_valid_native_run_writes_validation_summary(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self._write_valid_native_run(output)

            result = validate_run(output)

            self.assertTrue(result["valid"])
            self.assertEqual(result["schema"], "native_iterative_guidance")

    def test_observability_manifest_validation_requires_failure_ledger(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            path.write_text(json.dumps({"git_commit": "abc"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run manifest"):
                validate_observability_manifest(path)


if __name__ == "__main__":
    unittest.main()
