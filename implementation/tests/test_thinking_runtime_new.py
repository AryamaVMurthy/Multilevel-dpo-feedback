import unittest

from text_feedback_dpo.runtime import (
    RuntimeErrorExplicit,
    GeneratedText,
    decode_generated_records,
    extract_qwen_final_content,
    generate_student_batch,
    render_teacher_prompts,
    validate_teacher_identity,
)


class FakeTeacherTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered teacher prompt"


class ThinkingRuntimeTest(unittest.TestCase):
    def test_teacher_identity_is_pinned_qwen3_instruct_with_explicit_fallback(self):
        self.assertEqual(
            validate_teacher_identity("Qwen/Qwen3-32B", revision="teacher-rev", quantization="4bit", fallback_reason=None),
            "primary_qwen3_32b_4bit",
        )
        with self.assertRaisesRegex(ValueError, "Qwen3-32B"):
            validate_teacher_identity("other/model", revision="rev", quantization="4bit", fallback_reason=None)
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            validate_teacher_identity("Qwen/Qwen3-32B", revision="", quantization="4bit", fallback_reason=None)
        with self.assertRaisesRegex(ValueError, "4bit"):
            validate_teacher_identity("Qwen/Qwen3-32B", revision="rev", quantization="bf16", fallback_reason=None)
        with self.assertRaisesRegex(ValueError, "fallback reason"):
            validate_teacher_identity("Qwen/Qwen3-14B", revision="rev", quantization="4bit", fallback_reason=None)
        self.assertEqual(
            validate_teacher_identity(
                "Qwen/Qwen3-14B", revision="teacher-rev", quantization="4bit",
                fallback_reason="32B does not fit the measured allocation",
            ),
            "fallback_qwen3_14b_4bit",
        )
    def test_direct_student_generation_requires_exact_batch_cardinality(self):
        with self.assertRaisesRegex(RuntimeError, "answer batch cardinality"):
            generate_student_batch(
                object(), object(), ["one", "two"], mode="direct",
                scratchpad_max_new_tokens=8, answer_max_new_tokens=8,
                temperature=0.0, top_p=1.0,
                generation_fn=lambda *_args, **_kwargs: [GeneratedText("one", False)],
            )

    def test_generation_refuses_input_truncation_with_explicit_total_budget(self):
        from text_feedback_dpo.runtime import generate_batch_records

        class Encoded:
            input_ids = type("Ids", (), {"shape": (1, 4000)})()

            def to(self, _device):
                return self

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __init__(self):
                self.calls = []

            def __call__(self, prompts, **kwargs):
                self.calls.append((prompts, kwargs))
                return Encoded()

        tokenizer = Tokenizer()
        model = type("Model", (), {"device": "cpu", "generate": lambda *_args, **_kwargs: []})()
        with self.assertRaisesRegex(RuntimeErrorExplicit, "truncation|4096"):
            generate_batch_records(model, tokenizer, ["long prompt"], max_new_tokens=200, temperature=0.0, top_p=1.0)
        self.assertFalse(tokenizer.calls[0][1]["truncation"])

    def test_generation_records_true_length_cap_truncation(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            @staticmethod
            def decode(ids, **_kwargs):
                return " ".join(str(item) for item in ids)

        records = decode_generated_records(
            Tokenizer(),
            [[99, 10, 2, 0], [99, 10, 11, 12]],
            input_length=1,
            max_new_tokens=3,
        )
        self.assertEqual(records[0].text, "10")
        self.assertFalse(records[0].truncated)
        self.assertEqual(records[1].text, "10 11 12")
        self.assertTrue(records[1].truncated)

    def test_teacher_uses_native_qwen_thinking_chat_template(self):
        tokenizer = FakeTeacherTokenizer()
        rendered = render_teacher_prompts(tokenizer, ["Give one hint."])
        self.assertEqual(rendered, ["rendered teacher prompt"])
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "Give one hint."}])
        self.assertTrue(kwargs["enable_thinking"])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["tokenize"])

    def test_teacher_private_thinking_is_removed_before_hint_parsing(self):
        self.assertEqual(
            extract_qwen_final_content('<think>private reasoning</think>\n{"hint":"Recheck the entity."}'),
            '{"hint":"Recheck the entity."}',
        )
        with self.assertRaisesRegex(RuntimeErrorExplicit, "unterminated"):
            extract_qwen_final_content("<think>private reasoning")

    def test_two_pass_student_keeps_bounded_scratchpad_out_of_response(self):
        calls = []

        def generate(_model, _tokenizer, prompts, **kwargs):
            calls.append((prompts, kwargs))
            text = "The evidence points to the algorithm's author." if len(calls) == 1 else "Ada Lovelace"
            return [GeneratedText(text=text, truncated=False)]

        results = generate_student_batch(
            object(),
            object(),
            ["Evidence...\nQuestion: Who?\nAnswer:"],
            mode="two_pass",
            scratchpad_max_new_tokens=128,
            answer_max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            generation_fn=generate,
        )
        self.assertEqual(calls[0][1]["max_new_tokens"], 128)
        self.assertEqual(calls[1][1]["max_new_tokens"], 32)
        self.assertIn("The evidence points", calls[1][0][0])
        self.assertIn("Do not use XML", calls[1][0][0])
        self.assertIn("at most 8 words", calls[1][0][0])
        self.assertIn("best short guess", calls[1][0][0])
        self.assertEqual(results[0].response, "Ada Lovelace")
        self.assertEqual(results[0].scratchpad, "The evidence points to the algorithm's author.")
        self.assertNotIn("evidence points", results[0].response)

    def test_two_pass_accepts_stage_specific_scratchpad_instruction(self):
        calls = []

        def generate(_model, _tokenizer, prompts, **_kwargs):
            calls.append(prompts)
            return [GeneratedText("scratch", False)] if len(calls) == 1 else [GeneratedText("visible", False)]

        generate_student_batch(
            object(), object(), ["Search query:"], mode="two_pass", scratchpad_max_new_tokens=8,
            answer_max_new_tokens=8, temperature=0.0, top_p=1.0, generation_fn=generate,
            scratchpad_instruction="Privately formulate retrieval terms only; do not solve the question.",
            visible_instruction="Return one search query.",
        )
        self.assertIn("retrieval terms only", calls[0][0])
        self.assertIn("Return one search query", calls[1][0])

    def test_direct_student_has_no_private_scratchpad(self):
        def generate(_model, _tokenizer, _prompts, **_kwargs):
            return [GeneratedText(text="Ada Lovelace", truncated=False)]

        results = generate_student_batch(
            object(), object(), ["prompt"], mode="direct", scratchpad_max_new_tokens=128,
            answer_max_new_tokens=32, temperature=0.0, top_p=1.0, generation_fn=generate,
        )
        self.assertEqual(results[0].response, "Ada Lovelace")
        self.assertIsNone(results[0].scratchpad)


if __name__ == "__main__":
    unittest.main()
