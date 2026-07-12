import unittest

from text_feedback_dpo.prompts import (
    build_native_student_prompt,
    build_privileged_guidance_prompt,
    build_student_prompt,
    build_teacher_prompt,
    trajectory_format_instructions,
)


class PromptTest(unittest.TestCase):
    def test_feedback_policies_are_distinct_answer_free_and_thread_agnostic(self):
        prompts = {
            policy: build_privileged_guidance_prompt(
                problem="What is 2 + 2?",
                gold_answer="4",
                reference_solution="2 + 2 = 4.",
                rollout="The answer is 5.",
                result={"correct": False},
                domain="math",
                feedback_policy=policy,
            )
            for policy in ("error_only", "hint_only", "error_and_hint")
        }

        self.assertIn("identify the general error", prompts["error_only"].lower())
        self.assertIn("do not give a next-step hint", prompts["error_only"].lower())
        self.assertIn("one slight directional hint", prompts["hint_only"].lower())
        self.assertIn("do not diagnose", prompts["hint_only"].lower())
        self.assertIn("identify the general error", prompts["error_and_hint"].lower())
        self.assertIn("one slight directional hint", prompts["error_and_hint"].lower())
        for prompt in prompts.values():
            self.assertIn("Reference solution (teacher-only):\n2 + 2 = 4.", prompt)
            self.assertIn("must not reveal", prompt.lower())
            self.assertIn("equivalent expression", prompt.lower())
            self.assertIn("decisive intermediate", prompt.lower())
            self.assertIn("standalone", prompt.lower())
            self.assertIn("not refer to a previous turn", prompt.lower())
            self.assertIn("<student_feedback>", prompt)

    def test_feedback_policy_must_be_explicit_and_known(self):
        with self.assertRaisesRegex(ValueError, "feedback_policy"):
            build_privileged_guidance_prompt(
                problem="p",
                gold_answer="g",
                reference_solution="Reference derivation.",
                rollout="r",
                result={"correct": False},
                domain="math",
                feedback_policy="unknown",
            )

    def test_retry_prompt_presents_feedback_as_general_advice(self):
        prompt = build_native_student_prompt(
            problem="What is 2 + 2?",
            domain="math",
            guidance="Check the operation used to combine the quantities.",
        )
        self.assertIn("General problem-solving advice", prompt)
        self.assertNotIn("earlier attempt", prompt.lower())
        self.assertNotIn("teacher", prompt.lower())
        self.assertNotIn("retry", prompt.lower())

    def test_math_student_prompt_requires_structure_and_math_verification(self):
        prompt = build_student_prompt("What is 2 + 2?", "math")
        self.assertIn("<plan>", prompt)
        self.assertIn("<reflect>", prompt)
        self.assertIn("<final>", prompt)
        self.assertIn("arithmetic", prompt.lower())
        self.assertIn("substitution", prompt.lower())
        self.assertIn("constraint", prompt.lower())
        self.assertIn("at most 2 branches", prompt.lower())
        self.assertIn("do not use <thinking>", prompt.lower())
        self.assertNotIn("...", prompt)

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
        self.assertIn("do not use <thinking>", prompt.lower())
        self.assertIn("Verification:", prompt)
        self.assertNotIn("...", prompt)

    def test_student_and_teacher_share_the_same_trajectory_contract(self):
        contract = trajectory_format_instructions("search_qa")
        student_prompt = build_student_prompt("Who wrote Hamlet?", "search_qa")
        teacher_prompt = build_teacher_prompt(
            problem="Who wrote Hamlet?",
            gold_answer="William Shakespeare",
            student_rollout="student",
            result={"score": 0.0},
            domain="search_qa",
            teacher_mode="stronger_model",
        )
        self.assertIn(contract, student_prompt)
        self.assertIn(contract, teacher_prompt)

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
