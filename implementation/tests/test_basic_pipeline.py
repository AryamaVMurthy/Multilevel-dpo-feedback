import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import run_basic_pipeline


VALID_ORIGINAL = """<plan>
Solve directly.
</plan>
<think branch="A">
2 + 2 = 5.
</think>
<reflect>
Branch comparison: one branch.
Evidence / derivation check: arithmetic is inconsistent.
Verification: 2 + 2 should equal 4, not 5.
Decision: answer
</reflect>
<final>
5
</final>"""


VALID_CORRECTED = """<plan>
Solve the arithmetic and verify before final.
</plan>
<think branch="A">
2 + 2 = 4.
</think>
<reflect>
Branch comparison: one branch is sufficient.
Evidence / derivation check: addition is direct.
Verification: recomputing 2 + 2 gives 4.
Decision: answer
</reflect>
<final>
4
</final>"""


BAD_CORRECTED = """<plan>
Solve the arithmetic.
</plan>
<think branch="A">
3 + 3 = 6.
</think>
<reflect>
Branch comparison: one branch.
Evidence / derivation check: direct addition.
Decision: answer
</reflect>
<final>
6
</final>"""


class BasicPipelineTest(unittest.TestCase):
    def test_basic_pipeline_writes_observable_artifacts_and_filters_pairs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "examples.jsonl"
            rollouts = root / "rollouts.jsonl"
            corrections = root / "corrections.jsonl"
            output_dir = root / "run"

            examples.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "math-1",
                                "domain": "math",
                                "problem": "What is 2 + 2?",
                                "gold_answer": "4",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "math-2",
                                "domain": "math",
                                "problem": "What is 3 + 3?",
                                "gold_answer": "6",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rollouts.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "math-1", "rollout": VALID_ORIGINAL}),
                        json.dumps({"id": "math-2", "rollout": VALID_ORIGINAL}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            corrections.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "math-1",
                                "feedback": "Verify the arithmetic before final.",
                                "corrected_rollout": VALID_CORRECTED,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "math-2",
                                "feedback": "The correction must include verification.",
                                "corrected_rollout": BAD_CORRECTED,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_basic_pipeline(
                examples_path=examples,
                rollouts_path=rollouts,
                corrections_path=corrections,
                output_dir=output_dir,
                run_id="basic-fixture",
            )

            self.assertEqual(result["accepted_pairs"], 1)
            self.assertEqual(result["rejected_examples"], 1)
            self.assertTrue((output_dir / "events.jsonl").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "pairs.jsonl").exists())
            self.assertTrue((output_dir / "rejections.jsonl").exists())
            self.assertTrue((output_dir / "report.html").exists())

            events = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["event_name"], "run_start")
            self.assertEqual(events[-1]["event_name"], "run_end")
            self.assertTrue(all("elapsed_ms" in event for event in events))

            pair = json.loads((output_dir / "pairs.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(pair["prompt"], "What is 2 + 2?")
            self.assertEqual(pair["chosen"], VALID_CORRECTED)
            self.assertEqual(pair["rejected"], VALID_ORIGINAL)
            self.assertNotIn("Verify the arithmetic", pair["prompt"])

            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["examples_total"], 2)
            self.assertEqual(metrics["accepted_pairs"], 1)
            self.assertEqual(metrics["rejected_examples"], 1)
            self.assertEqual(metrics["verification_missing_rejections"], 1)


if __name__ == "__main__":
    unittest.main()
