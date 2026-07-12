import unittest

from text_feedback_dpo import evaluators
from text_feedback_dpo.evaluators import (
    ModelOutputParseError,
    build_evaluator_prompt,
    build_evaluator_repair_prompt,
    build_guidance_critic_prompt,
    build_guidance_guard_prompt,
    make_model_guidance_critic,
    make_model_guidance_guard,
    make_model_evaluator,
    parse_guidance_critic_output,
    parse_guidance_guard_output,
    parse_evaluator_output,
)
from text_feedback_dpo.methods import build_native_iterative_guidance_pairs
from text_feedback_dpo.models import ModelGeneration
from text_feedback_dpo.prompts import (
    build_native_student_prompt,
    build_privileged_guidance_prompt,
)


WRONG = "I calculate the result as 5."
RIGHT = "The result is 4."


class NativePipelineTest(unittest.TestCase):
    def test_student_feedback_output_accepts_one_multiline_tagged_block(self):
        parsed = evaluators.parse_student_feedback_output(
            "<student_feedback>\nCheck whether the selected relation applies before substituting values.\n"
            "Keep the verification independent of the target value.\n</student_feedback>"
        )
        self.assertEqual(
            parsed,
            "Check whether the selected relation applies before substituting values.\n"
            "Keep the verification independent of the target value.",
        )

    def test_student_feedback_output_rejects_malformed_blocks(self):
        invalid_outputs = (
            "plain feedback",
            "<student_feedback></student_feedback>",
            "prefix<student_feedback>Check the relation.</student_feedback>",
            "<student_feedback>Check.</student_feedback>suffix",
            "<student_feedback>One.</student_feedback><student_feedback>Two.</student_feedback>",
            "<student_feedback><student_feedback>Nested.</student_feedback></student_feedback>",
        )
        for raw in invalid_outputs:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evaluators.parse_student_feedback_output(raw)

    def test_first_attempt_correct_response_is_retained_for_response_sft(self):
        result = build_native_iterative_guidance_pairs(
            examples=[{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}],
            base_prompt_builder=lambda example: example["problem"],
            retry_prompt_builder=lambda base, guidance: f"{base}\n{guidance}",
            student_generate=lambda _prompt: RIGHT,
            evaluate=lambda _example, _response: {"correct": True, "answer": "4"},
            teacher_guidance=lambda *_args: "unused",
            guidance_guard=lambda *_args: {"safe": True},
            max_guidance_steps=1,
            max_guidance_regenerations=0,
        )

        self.assertEqual(len(result["response_sft"]), 1)
        self.assertEqual(result["response_sft"][0]["completion"], RIGHT)
        self.assertEqual(result["pairs"], [])

    def test_guidance_critic_uses_an_exact_token_contract(self):
        self.assertTrue(parse_guidance_critic_output("VALID")["valid"])
        self.assertFalse(parse_guidance_critic_output("INVALID")["valid"])
        with self.assertRaisesRegex(ValueError, "VALID or INVALID"):
            parse_guidance_critic_output("MAYBE")

        prompt = build_guidance_critic_prompt(
            example={"domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"},
            response="The answer is 5.",
            result={"correct": False, "answer": "5"},
            guidance="Recheck the final arithmetic operation.",
        )
        self.assertIn("directionally correct", prompt.lower())
        self.assertIn("exactly one token: valid or invalid", prompt.lower())

    def test_model_guidance_critic_preserves_its_generation_metadata(self):
        roles = []
        critic = make_model_guidance_critic(
            generate=lambda role, *_args, **_kwargs: (roles.append(role) or ModelGeneration(
                text="VALID",
                prompt_tokens=120,
                generated_tokens=1,
                terminated=True,
                truncated=False,
                finish_reason="eos",
            )),
            generation_kwargs={},
        )
        result = critic(
            {"domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"},
            "Recheck the final arithmetic operation.",
            {"correct": False, "answer": "5", "response": "The answer is 5."},
            1,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["generation"]["generated_tokens"], 1)
        self.assertEqual(roles, ["guidance_critic"])

    def test_model_guidance_guard_uses_its_distinct_role(self):
        roles = []
        guard = make_model_guidance_guard(
            generate=lambda role, *_args, **_kwargs: (roles.append(role) or "SAFE"),
            generation_kwargs={},
        )
        result = guard(
            {"domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"},
            "Recheck the final operation before answering.",
            {"correct": False},
            1,
        )
        self.assertTrue(result["safe"])
        self.assertEqual(roles, ["guidance_guard"])

    def test_native_collector_rejects_directionally_wrong_guidance(self):
        result = build_native_iterative_guidance_pairs(
            examples=[{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}],
            base_prompt_builder=lambda example: example["problem"],
            retry_prompt_builder=lambda base, guidance: f"{base}\n{guidance}",
            student_generate=lambda _prompt: WRONG,
            evaluate=lambda _example, _response: {"correct": False, "answer": "5"},
            teacher_guidance=lambda *_args: "Recheck whether the wrong relation should remain unchanged.",
            guidance_guard=lambda *_args: {"safe": True},
            guidance_critic=lambda *_args: {"valid": False},
            max_guidance_steps=1,
            max_guidance_regenerations=0,
        )

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["failures"][0]["error_code"], "invalid_guidance")
        self.assertFalse(result["failures"][0]["guidance_records"][0]["critic"]["valid"])

    def test_truncated_student_generation_is_never_a_correct_rollout(self):
        result = build_native_iterative_guidance_pairs(
            examples=[{"id": "m1", "domain": "math", "problem": "Compute.", "gold_answer": "4"}],
            base_prompt_builder=lambda example: example["problem"],
            retry_prompt_builder=lambda base, guidance: f"{base}\n{guidance}",
            student_generate=lambda _prompt: ModelGeneration(
                text="The answer mentioned in unfinished reasoning is 4",
                prompt_tokens=10,
                generated_tokens=8192,
                terminated=False,
                truncated=True,
                finish_reason="length",
            ),
            evaluate=lambda _example, _response: {"correct": True, "answer": "4"},
            teacher_guidance=lambda *_args: "Recheck how the quantities relate before answering fully.",
            guidance_guard=lambda *_args: {"safe": False},
            max_guidance_steps=1,
            max_guidance_regenerations=0,
        )

        self.assertEqual(result["pairs"], [])
        self.assertFalse(result["attempts"][0]["result"]["correct"])
        self.assertTrue(result["attempts"][0]["result"]["student_truncation_override"])
        self.assertEqual(result["attempts"][0]["generation"]["generated_tokens"], 8192)

    def test_model_evaluator_preserves_exact_generation_metadata(self):
        evaluator = make_model_evaluator(
            generate=lambda *_args, **_kwargs: ModelGeneration(
                text="<verdict>CORRECT</verdict>\n<evaluated_answer>4</evaluated_answer>",
                prompt_tokens=90,
                generated_tokens=18,
                terminated=True,
                truncated=False,
                finish_reason="eos",
            ),
            generation_kwargs={},
        )

        result = evaluator(
            {"domain": "math", "problem": "Compute.", "gold_answer": "4"},
            "The answer is 4.",
        )

        self.assertEqual(result["evaluator_generations"][0]["prompt_tokens"], 90)
        self.assertEqual(result["evaluator_generations"][0]["generated_tokens"], 18)
        self.assertEqual(result["evaluator_generations"][0]["finish_reason"], "eos")
        self.assertNotIn("generated_tokens_estimate", result)

    def test_guidance_guard_omits_estimate_when_exact_token_count_exists(self):
        guard = make_model_guidance_guard(
            generate=lambda *_args, **_kwargs: ModelGeneration(
                text="SAFE",
                prompt_tokens=80,
                generated_tokens=1,
                terminated=True,
                truncated=False,
                finish_reason="eos",
            ),
            generation_kwargs={},
        )

        result = guard(
            {"domain": "math", "problem": "Compute.", "gold_answer": "4"},
            "Recheck the calculation.",
            {},
            0,
        )

        self.assertEqual(result["generated_tokens"], 1)
        self.assertNotIn("generated_tokens_estimate", result)

    def test_native_student_prompt_allows_model_native_reasoning(self):
        prompt = build_native_student_prompt(
            problem="What is 2 + 2?",
            domain="math",
        )
        self.assertIn("at most 6 numbered steps", prompt.lower())
        self.assertIn("final:", prompt.lower())
        self.assertIn("FINAL: \\boxed{answer}", prompt)
        self.assertIn("do not output anything after", prompt.lower())
        self.assertNotIn("<think", prompt)
        self.assertNotIn("<reflect", prompt)
        self.assertNotIn("must use", prompt.lower())
        self.assertNotIn("evaluation process", prompt.lower())

    def test_guidance_prompt_has_privileged_answer_but_forbids_disclosure(self):
        prompt = build_privileged_guidance_prompt(
            problem="What is 2 + 2?",
            gold_answer="4",
            reference_solution="2 + 2 = 4.",
            rollout=WRONG,
            result={"correct": False, "reason": "arithmetic error"},
            domain="math",
            feedback_policy="hint_only",
        )
        self.assertIn("Gold answer (teacher-only):\n4", prompt)
        self.assertIn("Reference solution (teacher-only):\n2 + 2 = 4.", prompt)
        self.assertIn("must not reveal", prompt.lower())
        self.assertIn("slight directional hint", prompt.lower())
        self.assertIn("understand the mathematical issue", prompt.lower())
        self.assertNotIn("do not use digits", prompt.lower())
        self.assertNotIn("proper nouns", prompt.lower())
        self.assertIn(WRONG, prompt)

    def test_guidance_regeneration_prompt_includes_prior_review_without_answer(self):
        prompt = build_privileged_guidance_prompt(
            problem="What is 2 + 2?",
            gold_answer="4",
            reference_solution="2 + 2 = 4.",
            rollout=WRONG,
            result={"correct": False},
            domain="math",
            feedback_policy="hint_only",
            prior_reviews=[
                {
                    "guidance": "The answer is 4.",
                    "surface": {"valid": False, "reasons": ["answer_disclosure"]},
                    "critic": None,
                    "guard": None,
                }
            ],
        )

        self.assertIn("Previous rejected hint", prompt)
        self.assertIn("answer_disclosure", prompt)
        self.assertIn("write different standalone feedback", prompt.lower())

    def test_guidance_prompt_requires_reference_solution(self):
        with self.assertRaisesRegex(ValueError, "reference_solution"):
            build_privileged_guidance_prompt(
                problem="p",
                gold_answer="g",
                reference_solution="",
                rollout="r",
                result={"correct": False},
                domain="math",
                feedback_policy="hint_only",
            )

    def test_evaluator_output_uses_tagged_verdict_and_evaluated_answer(self):
        parsed = parse_evaluator_output(
            "<verdict>CORRECT</verdict>\n<evaluated_answer>12/\\sqrt{3}</evaluated_answer>"
        )
        self.assertTrue(parsed["correct"])
        self.assertEqual(parsed["answer"], "12/\\sqrt{3}")

        with self.assertRaisesRegex(ValueError, "verdict"):
            parse_evaluator_output("<evaluated_answer>4</evaluated_answer>")

    def test_evaluator_output_rejects_ambiguous_or_surrounding_content(self):
        invalid_outputs = (
            "CORRECT",
            "<verdict>MAYBE</verdict><evaluated_answer>4</evaluated_answer>",
            "prefix<verdict>CORRECT</verdict><evaluated_answer>4</evaluated_answer>",
            "<verdict>CORRECT</verdict><evaluated_answer>4</evaluated_answer>suffix",
            "<verdict>CORRECT</verdict><verdict>WRONG</verdict><evaluated_answer>4</evaluated_answer>",
            "<verdict>CORRECT</verdict><evaluated_answer></evaluated_answer>",
        )
        for raw in invalid_outputs:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_evaluator_output(raw)

    def test_guidance_guard_accepts_explicit_single_token_contract(self):
        parsed = parse_guidance_guard_output("SAFE")
        self.assertTrue(parsed["safe"])
        self.assertIsNone(parsed["confidence"])
        self.assertIn("explicit", parsed["reason"])

        with self.assertRaisesRegex(ValueError, "SAFE or UNSAFE"):
            parse_guidance_guard_output("MAYBE")

    def test_model_evaluator_attaches_domain_checks_without_hiding_model_judgment(self):
        evaluator = make_model_evaluator(
            generate=lambda *_args, **_kwargs: (
                "<verdict>CORRECT</verdict>\n<evaluated_answer>4</evaluated_answer>"
            ),
            generation_kwargs={},
        )
        result = evaluator(
            {"domain": "math", "problem": "What is 2 + 2?", "gold_answer": "4"},
            "The result is four.",
        )

        self.assertTrue(result["correct"])
        self.assertTrue(result["model_correct"])
        self.assertTrue(result["deterministic"]["numeric_exact_match"])
        self.assertEqual(result["deterministic"]["evaluator_source"], "deterministic_numeric")

    def test_model_evaluator_repairs_malformed_serialization_and_preserves_every_attempt(self):
        outputs = iter(
            [
                '{false,"$74","reason","high"}',
                "<verdict>WRONG</verdict>\n<evaluated_answer>$74</evaluated_answer>",
            ]
        )
        evaluator = make_model_evaluator(
            generate=lambda *_args, **_kwargs: next(outputs),
            generation_kwargs={},
            max_regenerations=1,
        )
        result = evaluator(
            {"domain": "math", "problem": "Compute the value.", "gold_answer": "75"},
            "The result is 74.",
        )

        self.assertFalse(result["correct"])
        self.assertEqual(result["evaluator_regenerations"], 1)
        self.assertEqual(len(result["raw_evaluator_outputs"]), 2)
        self.assertEqual(len(result["evaluator_parse_failures"]), 1)

    def test_model_evaluator_exhaustion_exposes_all_raw_outputs(self):
        outputs = iter(["invalid one", "invalid two"])
        evaluator = make_model_evaluator(
            generate=lambda *_args, **_kwargs: next(outputs),
            generation_kwargs={},
            max_regenerations=1,
        )
        with self.assertRaises(ModelOutputParseError) as caught:
            evaluator(
                {"domain": "math", "problem": "Compute.", "gold_answer": "4"},
                "five",
            )
        self.assertEqual(caught.exception.raw_outputs, ["invalid one", "invalid two"])

    def test_structured_role_prompts_require_tagged_evaluator_output(self):
        example = {
            "domain": "math",
            "problem": "What is 2 + 2?",
            "gold_answer": "4",
        }
        evaluator_prompt = build_evaluator_prompt(example=example, response=WRONG)
        guard_prompt = build_guidance_guard_prompt(example=example, guidance="Recheck the operation.")
        self.assertIn("<verdict>", evaluator_prompt.lower())
        self.assertIn("<evaluated_answer>", evaluator_prompt.lower())
        self.assertIn("do not provide analysis", evaluator_prompt.lower())
        self.assertIn("normal mathematical notation", evaluator_prompt.lower())
        repair_prompt = build_evaluator_repair_prompt(
            original_prompt=evaluator_prompt,
            raw='{"answer":"\\boxed{4}"}',
            error="missing verdict tag",
        )
        self.assertIn("exactly two tagged fields", repair_prompt.lower())
        self.assertIn("exactly one token: safe or unsafe", guard_prompt.lower())
        self.assertIn("safe example", guard_prompt.lower())
        self.assertIn("unsafe example", guard_prompt.lower())
        self.assertIn("broad relation-level correction", guard_prompt.lower())

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
        guard_contexts = []

        def evaluate(_example, response):
            return {"correct": response == RIGHT, "confidence": 0.99, "reason": "test"}

        result = build_native_iterative_guidance_pairs(
            examples=examples,
            base_prompt_builder=lambda example: f"Solve naturally: {example['problem']}",
            retry_prompt_builder=lambda base, guidance: f"{base}\nHint:\n{guidance}",
            student_generate=lambda prompt: next(outputs),
            evaluate=evaluate,
            teacher_guidance=lambda _example, _rollout, _result, attempt, _regeneration, _prior: (
                guidance_calls.append(attempt)
                or "Recheck how the quantities relate before answering fully."
            ),
            guidance_guard=lambda _example, guidance, _result, _attempt: (
                guard_contexts.append(guidance)
                or {
                    "safe": True,
                    "reason": "does not reveal answer",
                    "confidence": 0.99,
                    "guidance": guidance,
                }
            ),
            max_guidance_steps=3,
            max_guidance_regenerations=1,
        )

        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(all(pair["chosen"] == RIGHT for pair in result["pairs"]))
        self.assertEqual(result["metrics"]["first_correct_attempt"], {"m1": 2})
        self.assertEqual(result["metrics"]["success_by_attempt"]["2"], 1)
        self.assertEqual(guidance_calls, [1, 2])
        self.assertEqual(guard_contexts, [
            "Recheck how the quantities relate before answering fully.",
            "Recheck how the quantities relate before answering fully. Recheck how the quantities relate before answering fully.",
        ])
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
        self.assertEqual(result["failures"][0]["error_code"], "invalid_guidance_surface")
        self.assertEqual(result["failures"][0]["guidance_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
