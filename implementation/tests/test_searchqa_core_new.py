import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.config import load_config
from text_feedback_dpo.preferences import build_preference_rows, build_query_preference_rows, build_response_preference_rows
from text_feedback_dpo.prompts import build_student_prompt
from text_feedback_dpo.scoring import score_searchqa
from text_feedback_dpo.trajectories import collect_trajectory


class SearchQACoreContractTest(unittest.TestCase):
    def test_searchqa_scoring_reports_exact_match_f1_and_evidence_support(self):
        result = score_searchqa(
            response="Ada Lovelace",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["evidence_support"], 1.0)
        self.assertTrue(result["correct"])

    def test_searchqa_scoring_does_not_accept_unrelated_answer(self):
        result = score_searchqa(
            response="Grace Hopper",
            gold_answer="Ada Lovelace",
            packed_evidence="Ada Lovelace wrote the first algorithm.",
        )
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertFalse(result["correct"])

    def test_trajectory_stops_at_first_correct_without_teacher_written_answer(self):
        source = {"source_id": "S001", "original_rank": 1, "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada wrote the first algorithm."}

        def artifact(response, correct, no_hint):
            return {
                "id": "sq-1", "raw_query": "algorithm author", "ranked_search_results": [source],
                "raw_response": response, "truncation": {"query": False, "response": False},
                "cited_score": {"correct": correct, "parse_valid": True, "answer_correct": correct, "lexical_cited_answer_support": 1.0 if correct else 0.0},
                "policy_hash": "p", "prompt_version": "v", "response_schema_version": 1,
                "query_prompt_hash": "q", "response_prompt_hash": "r", "evaluator_version": "e", "no_hint": no_hint,
            }
        outputs = iter([artifact("Answer: Grace", False, True), artifact("Answer: Ada Lovelace", True, False)])

        def student(_prompt, _attempt):
            return next(outputs)

        def teacher(_request):
            return '{"hint":"Recheck the person associated with the algorithm."}'

        trajectory = collect_trajectory(
            example={
                "id": "sq-1",
                "question": "Who wrote the first algorithm?",
                "gold_answer": "Ada Lovelace",
                "sources": [source],
            },
            student_generate=student,
            teacher_generate=teacher,
            max_interventions=4,
        )
        self.assertTrue(trajectory["resolved"])
        self.assertEqual(len(trajectory["attempts"]), 2)
        self.assertEqual(trajectory["chosen"]["raw_response"], "Answer: Ada Lovelace")
        self.assertEqual(len(trajectory["interventions"]), 1)
        self.assertNotIn("Ada Lovelace", trajectory["interventions"][0]["hint"])
        self.assertEqual(trajectory["interventions"][0]["level"], 1)

    def test_preference_builder_excludes_hints_from_prompt_and_keeps_all_failures(self):
        self.assertEqual(build_preference_rows({"id": "sq-1", "resolved": True, "training_eligible": False}), [])

    def test_config_requires_searchqa_and_full_finetuning(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "searchqa-test",
                        "student_model": "Qwen/Qwen3-4B-Base",
                        "teacher_model": "Qwen/Qwen3-14B",
                        "student_revision": "student-rev",
                        "teacher_revision": "teacher-rev",
                        "retrieval": {"backend": "fixed_bm25", "top_k": 8, "k1": 1.2, "b": 0.75, "schema_version": 1},
                        "dataset": {"name": "searchqa", "source": "kyunghyuncho/search_qa", "revision": "data-rev", "max_length": 4096},
                        "training": {"full_finetuning": True, "method": "dpo"},
                        "slurm": {"partition": "u22", "gpus": 2},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["dataset"]["name"], "searchqa")
            self.assertTrue(config["training"]["full_finetuning"])

    def test_prompt_uses_plain_answer_contract_without_markup(self):
        prompt = build_student_prompt(
            {"question": "Who <wrote> it?", "packed_evidence": "A & B", "gold_answer": "Ada"},
            [],
        )
        self.assertIn("Who <wrote> it?", prompt)
        self.assertIn("A & B", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertIn("Think through the evidence", prompt)
        self.assertIn("plain text only", prompt)
        self.assertIn("Do not use XML", prompt)
        self.assertIn("at most 8 words", prompt)
        self.assertIn("noun phrase", prompt)
        self.assertIn("Never restate", prompt)
        self.assertNotIn("<response>", prompt)
        self.assertNotIn("<student_task>", prompt)

    def test_preference_builders_require_same_no_hint_context_and_student_provenance(self):
        def sibling(answer, query, response_hash="response-hash"):
            return {
                "provenance": "student", "no_hint": True, "raw_query": query,
                "raw_response": answer, "query_prompt_hash": "query-prompt-hash",
                "response_prompt_hash": response_hash, "retrieval_context_hash": "retrieval-hash",
                "policy_hash": "policy-v1", "evaluator_version": "evaluator-v1", "response_prompt": "same response prompt",
            "cited_score": {"correct": answer == "Ada", "parse_valid": True, "lexical_cited_answer_support": 1.0, "answer_correct": answer == "Ada"},
            "future_sibling_gain": 1.0 if answer == "Ada" else 0.0,
            "verified_no_hint_success": answer == "Ada",
            }
        trajectory = {
            "id": "q1", "resolved": True, "training_eligible": True,
            "attempts": [], "chosen": sibling("Ada", "writer ada"),
            "no_hint_siblings": [sibling("Ada", "writer ada"), sibling("Grace", "writer grace")],
            "interventions": [{"level": 1, "hint": "Recheck the associated person.", "future_sibling_gain": 1.0}],
            "query_prompt": "Generate one query.", "query_prompt_hash": "query-prompt-hash", "response_prompt_hash": "response-hash",
            "retrieval_context_hash": "retrieval-hash", "policy_hash": "policy-v1",
        }
        query_rows = build_query_preference_rows(trajectory)
        response_rows = build_response_preference_rows(trajectory)
        self.assertEqual(len(query_rows), 1)
        self.assertEqual(query_rows[0]["chosen"], " writer ada")
        self.assertEqual(query_rows[0]["rejected"], " writer grace")
        self.assertEqual(len(response_rows), 1)
        self.assertEqual(response_rows[0]["chosen"], " Ada")
        self.assertEqual(response_rows[0]["rejected"], " Grace")
        for row in query_rows + response_rows:
            self.assertTrue(row["metadata"]["no_hint"])
            self.assertEqual(row["metadata"]["provenance"], "student")
        with self.assertRaisesRegex(ValueError, "context"):
            build_response_preference_rows({**trajectory, "no_hint_siblings": [sibling("Ada", "writer ada", "other-hash"), sibling("Grace", "writer grace")]})


if __name__ == "__main__":
    unittest.main()
