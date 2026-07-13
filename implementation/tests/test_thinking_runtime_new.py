import unittest

from text_feedback_dpo.runtime import (
    RuntimeErrorExplicit,
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
            return ["The evidence points to the algorithm's author."] if len(calls) == 1 else ["Ada Lovelace"]

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
            return ["Ada Lovelace"]

        results = generate_student_batch(
            object(), object(), ["prompt"], mode="direct", scratchpad_max_new_tokens=128,
            answer_max_new_tokens=32, temperature=0.0, top_p=1.0, generation_fn=generate,
        )
        self.assertEqual(results[0].response, "Ada Lovelace")
        self.assertIsNone(results[0].scratchpad)


if __name__ == "__main__":
    unittest.main()
