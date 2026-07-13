import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.audit import audit_trajectories, write_trajectory_audit
from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.collection import collect_dataset_batchwise
from text_feedback_dpo.runtime import GeneratedText


def _example() -> dict:
    return {
        "id": "q1",
        "question": "Who wrote the first algorithm?",
        "gold_answer": "Ada Lovelace",
        "sources": [{
            "source_id": "S001",
            "original_rank": 1,
            "title": "Algorithm history",
            "url": "https://example.test/algorithm",
            "snippet": "Ada Lovelace wrote the first algorithm.",
        }],
    }


def _artifact(*, hints=(), correct=False) -> dict:
    response = (
        "Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001"
        if correct
        else "Answer: Grace Hopper\nReasoning: The source names another person [S001].\nSources: S001"
    )
    return run_fixed_retrieval_pipeline(
        [_example()],
        query_generate_batch=lambda _prompts: [GeneratedText("first algorithm author", False)],
        response_generate_batch=lambda _prompts: [GeneratedText(response, False)],
        policy_hash="policy-v1",
        hints_by_id={"q1": list(hints)},
    )[0]


def _trajectory() -> dict:
    def student(requests, **_kwargs):
        return [_artifact(hints=request["hints"], correct=bool(request["hints"])) for request in requests]

    def siblings(requests, **_kwargs):
        return [_artifact(correct=request["seed"] == 101) for request in requests]

    return collect_dataset_batchwise(
        examples=[_example()],
        student_generate_batch=student,
        teacher_generate_batch=lambda _prompts, **_kwargs: ['{"hint":"Recheck the associated person."}'],
        max_interventions=1,
        sibling_generate_batch=siblings,
        sibling_seeds=(101, 102),
        student_seed=7,
    )[0]


class TrajectoryAuditTest(unittest.TestCase):
    def test_emits_exact_attempt_sibling_and_eligibility_evidence(self):
        result = audit_trajectories([_example()], [_trajectory()], sibling_seeds=(101, 102))
        self.assertEqual(result["summary"]["trajectories"], 1)
        self.assertEqual(result["summary"]["resolved"], 1)
        self.assertEqual(result["summary"]["teacher_leakage"], 0)
        self.assertEqual(result["summary"]["sft_eligible"], 1)
        self.assertEqual(result["summary"]["preference_eligible"], 1)
        row = result["rows"][0]
        for field in (
            "question", "gold_answer", "raw_query", "top_sources", "raw_response",
            "error_code", "teacher_hint", "retry_raw_response", "siblings",
            "teacher_leakage", "latency_seconds", "teacher_prompt_token_count",
            "sft_eligible", "preference_eligible",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["attempt_index"], 0)
        self.assertEqual(row["retry_raw_response"], _trajectory()["attempts"][1]["response"])
        self.assertIsNone(row["latency_seconds"])
        self.assertIn("latency_seconds", result["summary"]["unavailable_observability"])

    def test_writes_json_jsonl_csv_and_escaped_html(self):
        example = _example()
        example["question"] = "Who wrote <script>alert(1)</script>?"
        trajectory = _trajectory()
        # Rebuild because canonical validation binds the question bytes.
        example = _example()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_trajectory_audit(
                audit_trajectories([example], [trajectory], sibling_seeds=(101, 102)),
                output_prefix=root / "smoke-audit",
            )
            self.assertEqual(set(paths), {"json", "jsonl", "csv", "html"})
            self.assertEqual(json.loads(paths["json"].read_text())["summary"]["trajectories"], 1)
            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("Trajectory audit", html)
            self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
