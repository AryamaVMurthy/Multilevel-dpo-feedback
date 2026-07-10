import unittest

from text_feedback_dpo.evaluators import parse_evaluator_output
from text_feedback_dpo.methods import build_native_iterative_guidance_pairs
from text_feedback_dpo.prompts import (
    build_native_student_prompt,
    build_privileged_guidance_prompt,
)


WRONG = "I calculate the result as 5."
RIGHT = "The result is 4."


class NativePipelineTest(unittest.TestCase):
    def test_native_student_prompt_allows_model_native_reasoning(self):
        prompt = build_native_student_prompt(
            problem="What is 2 + 2?",
            domain="math",
        )
        self.assertIn("reason in the style that is natural for you", prompt.lower())
        self.assertIn("final answer", prompt.lower())
        self.assertNotIn("<think", prompt)
        self.assertNotIn("<reflect", prompt)
        self.assertNotIn("must use", prompt.lower())

    def test_guidance_prompt_has_privileged_answer_but_forbids_disclosure(self):
        prompt = build_privileged_guidance_prompt(
            problem="What is 2 + 2?",
            gold_answer="4",
            rollout=WRONG,
            result={"correct": False, "reason": "arithmetic error"},
            domain="math",
        )
        self.assertIn("Gold answer (teacher-only):\n4", prompt)
        self.assertIn("must never reveal", prompt.lower())
        self.assertIn(WRONG, prompt)

    def test_evaluator_output_requires_explicit_correctness_and_confidence(self):
        parsed = parse_evaluator_output(
            '{"correct": true, "answer": "4", "confidence": 0.98, "reason": "matches"}'
        )
        self.assertTrue(parsed["correct"])
        self.assertEqual(parsed["answer"], "4")
        self.assertAlmostEqual(parsed["confidence"], 0.98)

        with self.assertRaisesRegex(ValueError, "correct"):
            parse_evaluator_output('{"answer": "4"}')

    def test_evaluator_accepts_native_reasoning_around_final_json(self):
        parsed = parse_evaluator_output(
            'I checked the response carefully.\n'
            '{"correct": true, "answer": "4", "confidence": 0.98, "reason": "matches"}\n'
            'This judgment is final.'
        )
        self.assertTrue(parsed["correct"])

    def test_native_collector_pairs_all_wrong_attempts_with_first_correct(self):
        examples = [
            {
                "id": "m1",
                "domain": "math",
                "problem": "What is 2 + 2?",
                "gold_answer": "4",
            }
        ]
        outputs = iter([WRONG, WRONG, RIGHT])
        guidance_calls = []

        def evaluate(_example, response):
            return {"correct": response == RIGHT, "confidence": 0.99, "reason": "test"}

        result = build_native_iterative_guidance_pairs(
            examples=examples,
            base_prompt_builder=lambda example: f"Solve naturally: {example['problem']}",
            retry_prompt_builder=lambda base, guidance: f"{base}\nHint:\n{guidance}",
            student_generate=lambda prompt: next(outputs),
            evaluate=evaluate,
            teacher_guidance=lambda _example, _rollout, _result, attempt: (
                guidance_calls.append(attempt) or f"Recheck step {attempt}."
            ),
            guidance_guard=lambda _example, guidance, _result, _attempt: {
                "safe": True,
                "reason": "does not reveal answer",
                "confidence": 0.99,
                "guidance": guidance,
            },
            max_guidance_steps=3,
            max_guidance_regenerations=1,
        )

        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(all(pair["chosen"] == RIGHT for pair in result["pairs"]))
        self.assertEqual(result["metrics"]["first_correct_attempt"], {"m1": 2})
        self.assertEqual(result["metrics"]["success_by_attempt"]["2"], 1)
        self.assertEqual(guidance_calls, [1, 2])
        self.assertEqual(len(result["attempts"]), 3)

    def test_native_collector_does_not_create_pairs_after_unsafe_guidance(self):
        examples = [
            {
                "id": "m1",
                "domain": "math",
                "problem": "What is 2 + 2?",
                "gold_answer": "4",
            }
        ]

        result = build_native_iterative_guidance_pairs(
            examples=examples,
            base_prompt_builder=lambda example: example["problem"],
            retry_prompt_builder=lambda base, guidance: f"{base}\n{guidance}",
            student_generate=lambda _prompt: WRONG,
            evaluate=lambda _example, _response: {"correct": False, "confidence": 1.0},
            teacher_guidance=lambda *_args: "The answer is 4.",
            guidance_guard=lambda *_args: {
                "safe": False,
                "reason": "direct answer disclosure",
                "confidence": 1.0,
            },
            max_guidance_steps=2,
            max_guidance_regenerations=1,
        )

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["metrics"]["unresolved_examples"], 1)
        self.assertEqual(result["failures"][0]["error_code"], "unsafe_guidance")
        self.assertEqual(result["failures"][0]["guidance_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
