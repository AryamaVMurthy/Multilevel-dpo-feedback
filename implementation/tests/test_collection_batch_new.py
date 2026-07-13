import unittest

from text_feedback_dpo.collection import collect_dataset_batchwise


class CollectionBatchTest(unittest.TestCase):
    def test_empty_student_response_uses_explicit_teacher_error_sentinel(self):
        examples = [{
            "id": "q1",
            "question": "Who wrote the first algorithm?",
            "gold_answer": "Ada Lovelace",
            "packed_evidence": "Ada Lovelace wrote the first algorithm.",
        }]

        def student(prompts, **_kwargs):
            return [""] if not any("<hint>" in prompt for prompt in prompts) else [
                "<response><answer>Ada Lovelace</answer><evidence>Ada evidence</evidence></response>"
            ]

        def teacher(prompts, **_kwargs):
            self.assertIn("__EMPTY_RESPONSE__", prompts[0])
            return ["<feedback><error_span>__EMPTY_RESPONSE__</error_span><hint>Produce a complete XML response.</hint><scope>verification</scope></feedback>"]

        rows = collect_dataset_batchwise(
            examples=examples,
            student_generate_batch=student,
            teacher_generate_batch=teacher,
            max_interventions=1,
        )
        self.assertTrue(rows[0]["resolved"])
        self.assertEqual(rows[0]["interventions"][0]["error_span"], "__EMPTY_RESPONSE__")
        self.assertEqual(rows[0]["preference_rows"][0]["rejected"], "__EMPTY_RESPONSE__")

    def test_collection_batches_each_attempt_and_preserves_first_correct(self):
        student_calls = []
        teacher_calls = []

        def student(prompts, **_kwargs):
            student_calls.append(len(prompts))
            if len(student_calls) == 1:
                return [
                    "<response><answer>Wrong</answer><evidence>x</evidence></response>",
                    "<response><answer>Ada</answer><evidence>Ada evidence</evidence></response>",
                ]
            return ["<response><answer>Ada</answer><evidence>Ada evidence</evidence></response>"]

        def teacher(prompts, **_kwargs):
            teacher_calls.append(len(prompts))
            return ["<feedback><error_span>Wrong</error_span><hint>Recheck the entity.</hint><scope>entity</scope></feedback>"]

        rows = collect_dataset_batchwise(
            examples=[
                {"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"},
                {"id": "2", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"},
            ],
            student_generate_batch=student,
            teacher_generate_batch=teacher,
            max_interventions=2,
        )
        self.assertEqual(student_calls, [2, 1])
        self.assertEqual(teacher_calls, [1])
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["resolved"])
        self.assertTrue(rows[1]["resolved"])
        self.assertEqual(len(rows[0]["attempts"]), 2)
        self.assertEqual(len(rows[1]["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
