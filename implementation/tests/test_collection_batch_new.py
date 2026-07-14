import copy
import unittest

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.collection import collect_dataset_batchwise
from text_feedback_dpo.runtime import GeneratedText
from text_feedback_dpo.trajectories import revalidate_cached_trajectory


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
    def test_collection_checkpoints_after_round_and_resumes_without_replaying_teacher(self):
        student_calls = []
        teacher_calls = []
        checkpoints = []

        def student(requests, **_kwargs):
            student_calls.append(requests)
            return [_artifact(hints=requests[0]["hints"], correct=bool(requests[0]["hints"]))]

        def teacher(prompts, **_kwargs):
            teacher_calls.append(prompts)
            return ['{"hint":"Recheck the associated person."}']

        collect_dataset_batchwise(
            examples=[_example()], student_generate_batch=student, teacher_generate_batch=teacher,
            max_interventions=1, sibling_generate_batch=None, sibling_seeds=(101, 102),
            student_seed=7, checkpoint_callback=checkpoints.append,
        )
        self.assertEqual(len(checkpoints), 2)
        checkpoint = checkpoints[0]
        self.assertEqual(checkpoint["schema_version"], 1)
        self.assertEqual(checkpoint["next_attempt_index"], 1)
        self.assertEqual(checkpoint["active_ids"], ["q1"])
        self.assertEqual(len(checkpoint["states"]["q1"]["interventions"]), 1)

        resumed_student_calls = []
        resumed_teacher_calls = []
        rows = collect_dataset_batchwise(
            examples=[_example()],
            student_generate_batch=lambda requests, **_kwargs: (
                resumed_student_calls.append(requests) or [_artifact(hints=requests[0]["hints"], correct=True)]
            ),
            teacher_generate_batch=lambda prompts, **_kwargs: (
                resumed_teacher_calls.append(prompts) or []
            ),
            max_interventions=1, sibling_generate_batch=None, sibling_seeds=(101, 102),
            student_seed=7, resume_checkpoint=checkpoint,
        )
        self.assertEqual(len(resumed_student_calls), 1)
        self.assertEqual(resumed_teacher_calls, [])
        self.assertTrue(rows[0]["resolved"])

    def test_malformed_canonical_source_fails_before_any_model_call(self):
        malformed = _example()
        malformed["sources"][0]["source_id"] = "not-canonical"
        student_calls = []
        teacher_calls = []

        with self.assertRaisesRegex(ValueError, "source_id"):
            collect_dataset_batchwise(
                examples=[malformed],
                student_generate_batch=lambda requests, **_kwargs: student_calls.append(requests),
                teacher_generate_batch=lambda prompts, **_kwargs: teacher_calls.append(prompts),
                max_interventions=1,
                sibling_generate_batch=None,
                sibling_seeds=(101, 102),
                student_seed=7,
            )
        self.assertEqual(student_calls, [])
        self.assertEqual(teacher_calls, [])

    def test_collection_requires_complete_sources_before_generation(self):
        for incomplete in (
            {"id": "q1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"},
            {"id": "1", "question": "Who?", "gold_answer": "Ada"},
        ):
            with self.subTest(incomplete=incomplete), self.assertRaisesRegex(ValueError, "invalid canonical sources"):
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
            self.assertNotIn("complete_source_records", prompts[0])
            self.assertIn("available_source_count", prompts[0])
            self.assertIn("retrieved_records", prompts[0])
            self.assertIn("deterministic_diagnostics", prompts[0])
            self.assertEqual(_kwargs["max_new_tokens"], 1024)
            self.assertEqual(_kwargs["gold_answers"], ["Ada"])
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
        intervention = row["interventions"][0]
        self.assertEqual(intervention["raw_teacher_response"], '{"hint":"Recheck the associated person."}')
        self.assertIn("Private request:", intervention["teacher_prompt"])
        for field in ("raw_teacher_response_hash", "teacher_prompt_hash", "hint_hash", "supervision_hash"):
            self.assertRegex(intervention[field], r"^[0-9a-f]{64}$")
        for attempt in row["attempts"]:
            self.assertRegex(attempt["supervision_hash"], r"^[0-9a-f]{64}$")

        no_attempts = copy.deepcopy(row)
        no_attempts["attempts"] = []
        with self.assertRaisesRegex(ValueError, "nonempty.*attempt 0"):
            revalidate_cached_trajectory(
                no_attempts, example=_example(), expected_sibling_seeds=(101, 102)
            )

        forged = copy.deepcopy(row)
        forged["interventions"][0]["raw_teacher_response"] = '{"hint":"Inspect the chronology."}'
        with self.assertRaisesRegex(ValueError, "teacher feedback|hint"):
            revalidate_cached_trajectory(
                forged, example=_example(), expected_sibling_seeds=(101, 102)
            )
        leaked = copy.deepcopy(row)
        leaked["interventions"][0]["raw_teacher_response"] = '{"hint":"Recheck Ada directly."}'
        leaked["interventions"][0]["hint"] = "Recheck Ada directly."
        with self.assertRaisesRegex(ValueError, "teacher feedback.*gold answer"):
            revalidate_cached_trajectory(
                leaked, example=_example(), expected_sibling_seeds=(101, 102)
            )


if __name__ == "__main__":
    unittest.main()
