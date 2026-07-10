import unittest

from text_feedback_dpo.methods import build_exact_dpo_pairs, build_iterative_guidance_dpo_pairs


WRONG = "<plan>x</plan><reflect>Verification: no</reflect><final>5</final>"
RIGHT = "<plan>x</plan><reflect>Verification: 2 + 2 = 4</reflect><final>4</final>"


class MethodPairBuilderTest(unittest.TestCase):
    def test_exact_pdf_method_keeps_only_teacher_corrected_improvements(self):
        examples = [{"id": "m1", "domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"}]

        result = build_exact_dpo_pairs(
            examples=examples,
            student_generate=lambda _prompt: WRONG,
            teacher_correct=lambda _example, _rollout, _result: {
                "feedback": "Check the addition before final.",
                "corrected_rollout": RIGHT,
            },
        )

        self.assertEqual(len(result["pairs"]), 1)
        pair = result["pairs"][0]
        self.assertEqual(pair["chosen"], RIGHT)
        self.assertEqual(pair["rejected"], WRONG)
        self.assertNotIn("Check the addition", pair["prompt"])
        self.assertEqual(result["format_sft"][0]["completion"], RIGHT)

    def test_iterative_guidance_pairs_every_wrong_attempt_with_first_correct_rollout(self):
        examples = [{"id": "m1", "domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"}]
        outputs = iter([WRONG, WRONG, RIGHT])

        result = build_iterative_guidance_dpo_pairs(
            examples=examples,
            student_generate=lambda _prompt: next(outputs),
            teacher_guidance=lambda _example, _rollout, _result, attempt: f"Guidance {attempt}: recheck arithmetic.",
            max_guidance_steps=2,
        )

        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(all(pair["chosen"] == RIGHT for pair in result["pairs"]))
        self.assertTrue(all(pair["rejected"] == WRONG for pair in result["pairs"]))
        self.assertTrue(all(pair["prompt"] == result["base_prompts"]["m1"] for pair in result["pairs"]))
        self.assertEqual(result["metrics"]["first_correct_examples"], 1)
        self.assertEqual(result["metrics"]["wrong_attempts"], 2)

    def test_iterative_guidance_fails_if_no_correct_rollout_is_reached(self):
        examples = [{"id": "m1", "domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"}]

        result = build_iterative_guidance_dpo_pairs(
            examples=examples,
            student_generate=lambda _prompt: WRONG,
            teacher_guidance=lambda _example, _rollout, _result, attempt: f"Guidance {attempt}",
            max_guidance_steps=2,
        )

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["metrics"]["unresolved_examples"], 1)
        self.assertEqual(result["failures"][0]["error_code"], "no_correct_rollout_within_guidance_budget")


if __name__ == "__main__":
    unittest.main()
