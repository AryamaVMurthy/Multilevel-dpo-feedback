import unittest

from text_feedback_dpo.prompts import build_student_prompt, build_teacher_prompt


class PromptTest(unittest.TestCase):
    def test_math_student_prompt_requires_structure_and_math_verification(self):
        prompt = build_student_prompt("What is 2 + 2?", "math")
        self.assertIn("<plan>", prompt)
        self.assertIn("<reflect>", prompt)
        self.assertIn("<final>", prompt)
        self.assertIn("arithmetic", prompt.lower())
        self.assertIn("substitution", prompt.lower())
        self.assertIn("constraint", prompt.lower())
        self.assertIn("at most 2 branches", prompt.lower())

    def test_search_qa_student_prompt_requires_evidence_verification(self):
        prompt = build_student_prompt("Who wrote Hamlet?", "search_qa")
        self.assertIn("entity", prompt.lower())
        self.assertIn("relation", prompt.lower())
        self.assertIn("evidence", prompt.lower())
        self.assertIn("answer type", prompt.lower())
        self.assertIn("at most 3 branches", prompt.lower())

    def test_teacher_prompt_contains_correction_contract(self):
        prompt = build_teacher_prompt(
            problem="What is 2 + 2?",
            gold_answer="4",
            student_rollout="<final>5</final>",
            result={"score": 0.0},
            domain="math",
            teacher_mode="stronger_model",
        )
        self.assertIn("What is 2 + 2?", prompt)
        self.assertIn("Gold answer:\n4", prompt)
        self.assertIn("<final>5</final>", prompt)
        self.assertIn('"score": 0.0', prompt)
        self.assertIn("<feedback>", prompt)
        self.assertIn("<corrected_rollout>", prompt)
        self.assertIn("<think branch=\"A\">", prompt)
        self.assertIn("do not use <thinking>", prompt)
        self.assertIn("Verification:", prompt)

    def test_privileged_teacher_prompt_marks_training_only_context(self):
        prompt = build_teacher_prompt(
            problem="What is 2 + 2?",
            gold_answer="4",
            student_rollout="<final>5</final>",
            result={"score": 0.0},
            domain="math",
            teacher_mode="same_model_privileged",
        )
        self.assertIn("privileged training-only", prompt.lower())
        self.assertIn("never available during student evaluation", prompt.lower())

    def test_unknown_teacher_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "teacher_mode"):
            build_teacher_prompt(
                problem="x",
                gold_answer="y",
                student_rollout="z",
                result={},
                domain="math",
                teacher_mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
