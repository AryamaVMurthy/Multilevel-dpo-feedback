import unittest

from text_feedback_dpo.dataset import build_sft_rows
from text_feedback_dpo.searchqa import materialize_row, pack_evidence


class SearchQADataTest(unittest.TestCase):
    def test_materialize_row_requires_question_answer_and_snippets(self):
        row = materialize_row(
            {"question": "Who?", "answer": "Ada", "search_results": ["Ada was a writer.", "Other text"]},
            split="train",
            index=3,
        )
        self.assertEqual(row["id"], "train-3")
        self.assertEqual(row["question"], "Who?")
        self.assertEqual(row["gold_answer"], "Ada")
        self.assertEqual(row["snippets"], ["Ada was a writer.", "Other text"])

    def test_materialize_row_fails_explicitly_on_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "snippets"):
            materialize_row({"question": "Who?", "answer": "Ada"}, split="train", index=0)

    def test_materialize_row_accepts_official_search_results_sequence_mapping(self):
        row = materialize_row(
            {"question": "Who?", "answer": "Ada", "search_results": {"snippets": ["Ada evidence"]}},
            split="train",
            index=1,
        )
        self.assertEqual(row["snippets"], ["Ada evidence"])

    def test_materialize_row_skips_empty_official_snippets_but_requires_evidence(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": [{"snippet": ""}, {"snippet": "Ada evidence"}, None],
            },
            split="validation",
            index=0,
        )
        self.assertEqual(row["snippets"], ["Ada evidence"])
        with self.assertRaisesRegex(ValueError, "no usable"):
            materialize_row({"question": "Who?", "answer": "Ada", "search_results": [{"snippet": ""}]}, split="validation", index=1)

    def test_materialize_row_accepts_searchqa_parquet_mirror_schema(self):
        row = materialize_row(
            {"question": "Who?", "answers": ["Ada"], "context": "Ada evidence"},
            split="validation",
            index=2,
        )
        self.assertEqual(row["gold_answer"], "Ada")
        self.assertEqual(row["snippets"], ["Ada evidence"])

    def test_pack_evidence_is_deterministic_and_never_exceeds_budget(self):
        snippets = ["one two", "three four", "five six"]
        packed = pack_evidence(snippets, max_tokens=4, token_count=lambda text: len(text.split()))
        self.assertEqual(packed, "one two\nthree four")

    def test_sft_target_is_plain_prompt_completion(self):
        row = {
            "id": "train-0",
            "question": "Who?",
            "gold_answer": "Ada",
            "packed_evidence": "prefix " * 500 + "Ada evidence near the answer " + "suffix " * 500,
        }
        result = build_sft_rows([row])[0]
        self.assertEqual(result["id"], "train-0")
        self.assertEqual(result["completion"], "Ada")
        self.assertTrue(result["prompt"].endswith("Answer:"))
        self.assertIn("Ada evidence near the answer", result["prompt"])
        self.assertNotIn("<response>", result["prompt"] + result["completion"])


if __name__ == "__main__":
    unittest.main()
