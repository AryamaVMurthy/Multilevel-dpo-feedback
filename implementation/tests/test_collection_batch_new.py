import unittest

from text_feedback_dpo.collection import collect_dataset_batchwise


class CollectionBatchTest(unittest.TestCase):
    def test_empty_student_response_uses_explicit_teacher_error_sentinel(self):
        with self.assertRaisesRegex(ValueError, "complete source records"):
            collect_dataset_batchwise(
                examples=[{"id": "q1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"}],
                student_generate_batch=lambda _prompts, **_kwargs: [],
                teacher_generate_batch=lambda _prompts, **_kwargs: [],
                max_interventions=1,
            )

    def test_collection_batches_each_attempt_and_preserves_first_correct(self):
        with self.assertRaisesRegex(ValueError, "complete source records"):
            collect_dataset_batchwise(
                examples=[{"id": "1", "question": "Who?", "gold_answer": "Ada"}],
                student_generate_batch=lambda _prompts, **_kwargs: [],
                teacher_generate_batch=lambda _prompts, **_kwargs: [],
                max_interventions=2,
            )

    def test_collection_uses_active_artifacts_teacher_once_and_verifies_no_hint_siblings(self):
        source = {
            "source_id": "S001", "original_rank": 1, "retrieval_rank": 1,
            "title": "Ada", "url": "https://example.test/ada", "snippet": "Ada wrote the algorithm.",
            "query_hash": "query-hash", "corpus_hash": "corpus-hash", "requested_top_k": 8,
            "effective_top_k": 1, "source_count": 1, "bm25_score": 1.0,
        }

        def artifact(response, *, correct, prompt_hash="response-prompt", query="writer", no_hint=True):
            return {
                "id": "q1", "raw_query": query, "ranked_search_results": [source],
                "raw_response": response, "response_prompt_hash": prompt_hash,
                "query_prompt_hash": "query-prompt", "prompt_version": "fixed-retrieval-cited-v1",
                "response_schema_version": 1, "policy_hash": "policy-v1", "evaluator_version": "evaluator-v1",
                "truncation": {"query": False, "response": False},
                "cited_score": {
                    "parse_valid": True, "answer_correct": correct, "correct": correct,
                    "lexical_cited_answer_support": 1.0 if correct else 0.0,
                    "citation_precision": 1.0 if correct else 0.0,
                    "error_code": None if correct else "answer_mismatch",
                },
                "parsed_response": {"answer": "Ada" if correct else "Grace", "reasoning": "Source [S001].", "source_ids": ["S001"]},
                "provenance": "student", "no_hint": no_hint,
                "response_prompt": "Answer using retrieved sources.",
            }

        student_calls = []
        teacher_calls = []
        sibling_calls = []

        def student(prompts, **_kwargs):
            student_calls.append(prompts)
            if len(student_calls) == 1:
                return [artifact("Answer: Grace", correct=False, no_hint=True)]
            return [artifact("Answer: Ada", correct=True, no_hint=False)]

        def teacher(prompts, **_kwargs):
            teacher_calls.append(prompts)
            self.assertIn("complete_source_records", prompts[0])
            self.assertIn("retrieved_records", prompts[0])
            self.assertIn("deterministic_diagnostics", prompts[0])
            return ['{"hint":"Recheck the associated person."}']

        def siblings(requests, **kwargs):
            sibling_calls.append((requests, kwargs))
            self.assertEqual([request["seed"] for request in requests], [101, 102])
            return [artifact("Answer: Ada", correct=True, query="writer ada", no_hint=True), artifact("Answer: Grace", correct=False, query="writer grace", no_hint=True)]

        rows = collect_dataset_batchwise(
            examples=[{"id": "q1", "question": "Who?", "gold_answer": "Ada", "sources": [source]}],
            student_generate_batch=student,
            teacher_generate_batch=teacher,
            max_interventions=1,
            sibling_generate_batch=siblings,
            sibling_seeds=(101, 102),
        )
        row = rows[0]
        self.assertEqual([len(call) for call in student_calls], [1, 1])
        self.assertEqual([len(call) for call in teacher_calls], [1])
        self.assertEqual(len(sibling_calls), 1)
        self.assertTrue(row["training_eligible"])
        self.assertEqual(row["interventions"][0]["responsible_region"], "answer")
        self.assertGreater(row["interventions"][0]["hint_token_count"], 0)
        self.assertGreater(row["interventions"][0]["repair_scope_cost"], 0)
        self.assertGreater(row["interventions"][0]["efficiency_score"], 0)
        self.assertNotIn("Ada", row["interventions"][0]["hint"])


if __name__ == "__main__":
    unittest.main()
