import unittest

from text_feedback_dpo.runtime import (
    RuntimeErrorExplicit,
    GeneratedText,
    decode_generated_records,
    extract_qwen_final_content,
    generate_student_batch,
    render_teacher_prompts,
)


class FakeTeacherTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered teacher prompt"


class ThinkingRuntimeTest(unittest.TestCase):
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
        self.assertEqual(results[0].response, "Ada Lovelace")
        self.assertEqual(results[0].scratchpad, "The evidence points to the algorithm's author.")
        self.assertNotIn("evidence points", results[0].response)

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
