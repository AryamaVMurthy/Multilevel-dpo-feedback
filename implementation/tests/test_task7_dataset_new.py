import hashlib
import json
import unittest

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.dataset import SFTDataGateError, build_sft_rows_from_trajectories
from text_feedback_dpo.runtime import GeneratedText


def _hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate(**overrides):
    candidate = run_fixed_retrieval_pipeline(
        [_example()],
        query_generate_batch=lambda prompts: [GeneratedText("Ada algorithm", False) for _ in prompts],
        response_generate_batch=lambda prompts: [GeneratedText("Answer: Ada Lovelace\nReasoning: The source identifies Ada Lovelace [S001].\nSources: S001", False) for _ in prompts],
        policy_hash="policy-v1",
    )[0]
    candidate["verified_no_hint_success"] = True
    candidate["future_sibling_gain"] = 1.0
    candidate.update(overrides)
    return candidate


def _example():
    return {
        "id": "q1",
        "question": "Who wrote the first algorithm?",
        "gold_answer": "Ada Lovelace",
        "sources": [{
            "source_id": "S001", "original_rank": 1, "title": "Ada biography",
            "url": "https://example.test/ada", "snippet": "Ada Lovelace wrote the first algorithm.",
        }],
    }


def _trajectory(candidate=None, **overrides):
    candidate = candidate or _candidate()
    trajectory = {
        "id": "q1",
        "training_eligible": True,
        "query_prompt": candidate["query_prompt"],
        "query_prompt_hash": candidate["query_prompt_hash"],
        "no_hint_siblings": [candidate],
    }
    trajectory.update(overrides)
    return trajectory


class Task7DatasetTest(unittest.TestCase):
    def test_builds_separate_student_query_and_visible_response_rows(self):
        rows, report = build_sft_rows_from_trajectories([_trajectory()], examples={"q1": _example()})

        self.assertEqual([row["task"] for row in rows], ["query", "response"])
        self.assertEqual(rows[0]["prompt"], _candidate()["query_prompt"])
        self.assertEqual(rows[0]["completion"], " Ada algorithm")
        self.assertEqual(rows[1]["prompt"], _candidate()["response_prompt"])
        self.assertEqual(rows[1]["completion"].startswith(" Answer:"), True)
        self.assertEqual(rows[1]["visible_response"], _candidate()["raw_response"])
        self.assertEqual(report["query_rows"], 1)
        self.assertEqual(report["response_rows"], 1)
        self.assertEqual(report["exclusion_counts"], {})
        self.assertEqual(rows[0]["metadata"]["provenance"], "student")
        self.assertTrue(rows[0]["metadata"]["verified_no_hint_success"])

    def test_rejects_teacher_hint_fabrication_and_forged_context_with_exact_reasons(self):
        cases = [
            ("teacher_provenance", {"provenance": "teacher"}),
            ("hinted_prompt", {"response_prompt": "Hints:\n- look again\n\nResponse:"}),
            ("fabricated_target", {"fabricated": True}),
            ("unverified_no_hint_success", {"verified_no_hint_success": False}),
            ("response_prompt_hash_mismatch", {"response_prompt_hash": "forged"}),
            ("retrieval_context_hash_mismatch", {"retrieval_context_hash": "forged"}),
        ]
        for reason, overrides in cases:
            with self.subTest(reason=reason):
                rows, report = build_sft_rows_from_trajectories([_trajectory(_candidate(**overrides))], examples={"q1": _example()})
                self.assertEqual(rows, [])
                expected_count = 1 if reason in {"hinted_prompt", "response_prompt_hash_mismatch", "retrieval_context_hash_mismatch"} else 2
                self.assertEqual(report["exclusion_counts"].get(reason), expected_count)
                self.assertEqual({item["reason"] for item in report["exclusions"]}, {reason})

    def test_length_gate_never_silently_truncates_and_reports_coverage(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(4097))

        with self.assertRaises(SFTDataGateError) as context:
            build_sft_rows_from_trajectories(
                [_trajectory()], examples={"q1": _example()}, tokenizer=Tokenizer(), min_coverage=1.0,
            )
        self.assertIn("combined_token_length_exceeds_max_length", context.exception.report["exclusion_counts"])
        self.assertEqual(context.exception.report["query_coverage"], 0.0)
        self.assertEqual(context.exception.report["response_coverage"], 0.0)

    def test_canonical_validator_rejects_forged_raw_response_and_retrieval(self):
        forged_ranked = [dict(_candidate()["canonical_ranked_search_results"][0], title="forged")]
        forged_ranked_hash = _hash(forged_ranked)
        for overrides in (
            {"raw_response": "Answer: Ada Lovelace\nReasoning: forged [S001].\nSources: S001"},
            {"canonical_ranked_search_results": forged_ranked, "retrieval_context_hash": forged_ranked_hash},
        ):
            with self.subTest(overrides=overrides):
                rows, report = build_sft_rows_from_trajectories(
                    [_trajectory(_candidate(**overrides))], examples={"q1": _example()}
                )
                self.assertEqual(rows, [])
                self.assertEqual(report["exclusion_counts"].get("canonical_artifact_validation_failed"), 2)


if __name__ == "__main__":
    unittest.main()
