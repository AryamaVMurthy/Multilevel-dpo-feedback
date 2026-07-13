import unittest

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.collection import collect_dataset_batchwise
from text_feedback_dpo.runtime import GeneratedText


def _example():
    return {
        "id": "q1", "question": "Who?", "gold_answer": "Ada",
        "sources": [{
            "source_id": "S001", "original_rank": 1, "title": "Ada",
            "url": "https://example.test/ada", "snippet": "Ada wrote the algorithm.",
        }],
    }


def _artifact(*, hints=(), query="writer", correct=False):
    response = (
        "Answer: Ada\nReasoning: Source identifies Ada [S001].\nSources: S001"
        if correct else
        "Answer: Grace\nReasoning: Source identifies another person [S001].\nSources: S001"
    )
    return run_fixed_retrieval_pipeline(
        [_example()],
        query_generate_batch=lambda _prompts: [GeneratedText(query, False)],
        response_generate_batch=lambda _prompts: [GeneratedText(response, False)],
        policy_hash="policy-v1", hints_by_id={"q1": list(hints)},
    )[0]


class CollectionBatchTest(unittest.TestCase):
    def test_collection_requires_complete_sources_before_generation(self):
        for incomplete in (
            {"id": "q1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"},
            {"id": "1", "question": "Who?", "gold_answer": "Ada"},
        ):
            with self.subTest(incomplete=incomplete), self.assertRaisesRegex(ValueError, "complete source records"):
                collect_dataset_batchwise(
                    examples=[incomplete], student_generate_batch=lambda _requests, **_kwargs: [],
                    teacher_generate_batch=lambda _prompts, **_kwargs: [], max_interventions=1,
                    student_seed=7,
                )

    def test_collection_uses_active_artifacts_teacher_once_and_verifies_no_hint_siblings(self):
        student_calls = []
        teacher_calls = []
        sibling_calls = []

        def student(requests, **_kwargs):
            student_calls.append(requests)
            hints = tuple(requests[0]["hints"])
            return [_artifact(hints=hints, correct=bool(hints))]

        def teacher(prompts, **_kwargs):
            teacher_calls.append(prompts)
            self.assertIn("complete_source_records", prompts[0])
            self.assertIn("retrieved_records", prompts[0])
            self.assertIn("deterministic_diagnostics", prompts[0])
            return ['{"hint":"Recheck the associated person."}']

        def siblings(requests, **kwargs):
            sibling_calls.append((requests, kwargs))
            self.assertEqual([request["seed"] for request in requests], [101, 102])
            return [
                _artifact(query="writer ada", correct=True),
                _artifact(query="writer grace", correct=False),
            ]

        rows = collect_dataset_batchwise(
            examples=[_example()], student_generate_batch=student, teacher_generate_batch=teacher,
            max_interventions=1, sibling_generate_batch=siblings, sibling_seeds=(101, 102),
            student_seed=7,
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
        self.assertEqual(row["ranked_interventions"], row["interventions"])
        self.assertNotIn("Ada", row["interventions"][0]["hint"])


if __name__ == "__main__":
    unittest.main()
