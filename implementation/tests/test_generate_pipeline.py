import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import run_generate_pipeline
from text_feedback_dpo.models import FakeModelProvider


STUDENT_WRONG = """<plan>
Solve with one arithmetic branch.
</plan>
<think branch="A">
2 + 2 = 5.
</think>
<reflect>
Branch comparison: one branch.
Evidence / derivation check: the arithmetic should be checked.
Verification: recalculating 2 + 2 does not support 5.
Decision: answer
</reflect>
<final>
5
</final>"""


TEACHER_CORRECTION = """<feedback>
The student should recompute the arithmetic and verify before final.
</feedback>

<corrected_rollout>
<plan>
Solve with one arithmetic branch and verify before final.
</plan>
<think branch="A">
2 + 2 = 4.
</think>
<reflect>
Branch comparison: one branch is sufficient.
Evidence / derivation check: direct addition gives 4.
Verification: recalculating 2 + 2 gives 4.
Decision: answer
</reflect>
<final>
4
</final>
</corrected_rollout>"""


class GeneratePipelineTest(unittest.TestCase):
    def test_fake_pipeline_writes_rollouts_corrections_pairs_and_report(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            provider = FakeModelProvider({"student": STUDENT_WRONG, "teacher": TEACHER_CORRECTION})

            result = run_generate_pipeline(
                config_path=Path("configs/basic_smoke.yaml"),
                output_dir=out,
                model_provider=provider,
            )

            self.assertEqual(result["accepted_pairs"], 1)
            self.assertTrue((out / "events.jsonl").exists())
            self.assertTrue((out / "rollouts.jsonl").exists())
            self.assertTrue((out / "corrections.jsonl").exists())
            self.assertTrue((out / "pairs.jsonl").exists())
            self.assertTrue((out / "rejections.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "report.html").exists())

            pair = json.loads((out / "pairs.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(pair["prompt"], "What is 2 + 2?")
            self.assertNotIn("The student should recompute", pair["prompt"])

            events = [
                json.loads(line)
                for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event["event_name"] == "student_generated" for event in events))
            self.assertTrue(any(event["event_name"] == "teacher_corrected" for event in events))

    def test_cli_fake_smoke_mode_writes_artifacts_without_transformers(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            exit_code = __import__("subprocess").run(
                [
                    "python3",
                    "-m",
                    "text_feedback_dpo.cli",
                    "generate-pipeline",
                    "--config",
                    "configs/basic_smoke.yaml",
                    "--output-dir",
                    str(out),
                    "--fake-smoke",
                ],
                env={"PYTHONPATH": "src"},
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            self.assertEqual(exit_code, 0)
            self.assertTrue((out / "pairs.jsonl").exists())

    def test_malformed_teacher_output_is_logged_and_persisted_before_failure(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            provider = FakeModelProvider(
                {"student": STUDENT_WRONG, "teacher": "missing required structured blocks"}
            )

            with self.assertRaisesRegex(ValueError, "missing <feedback> block"):
                run_generate_pipeline(
                    config_path=Path("configs/basic_smoke.yaml"),
                    output_dir=out,
                    model_provider=provider,
                )

            failure = json.loads((out / "teacher_failures.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(failure["error_code"], "teacher_output_parse_failed")
            self.assertEqual(failure["raw_teacher_output"], "missing required structured blocks")
            events = [
                json.loads(line)
                for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event["event_name"] == "failure" for event in events))

    def test_placeholder_teacher_blocks_fail_explicitly(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            provider = FakeModelProvider(
                {
                    "student": STUDENT_WRONG,
                    "teacher": "<feedback>...</feedback><corrected_rollout>...</corrected_rollout>",
                }
            )

            with self.assertRaisesRegex(ValueError, "placeholder"):
                run_generate_pipeline(
                    config_path=Path("configs/basic_smoke.yaml"),
                    output_dir=out,
                    model_provider=provider,
                )


if __name__ == "__main__":
    unittest.main()
