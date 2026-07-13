import unittest

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.preflight import assess_preflight, select_preflight_rows, select_thinking_mode, summarize_response_quality
from text_feedback_dpo.runtime import GeneratedText


class PreflightQualityTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada wrote it."},
            {"id": "2", "question": "Where?", "gold_answer": "California", "packed_evidence": "The valley is in California."},
        ]

    def test_train_dev_selection_is_deterministic_and_order_independent(self):
        first = select_preflight_rows(self.examples, sample_size=1, seed=7)
        second = select_preflight_rows(list(reversed(self.examples)), sample_size=1, seed=7)
        self.assertEqual(first, second)

    def test_quality_summary_reports_plain_answer_failures_and_lengths(self):
        metrics = summarize_response_quality(
            self.examples,
            [
                {"id": "1", "response": "Ada", "truncated": False},
                {"id": "2", "response": "<answer></answer>", "truncated": True},
            ],
        )
        self.assertEqual(metrics["examples"], 2)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["nonempty_rate"], 1.0)
        self.assertEqual(metrics["truncation_rate"], 0.5)
        self.assertEqual(metrics["markup_rate"], 0.5)
        self.assertEqual(metrics["verbose_rate"], 0.0)
        self.assertEqual(metrics["answer_words"]["max"], 2)

    def test_missing_truncation_metadata_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            summarize_response_quality(self.examples, [{"id": "1", "response": "Ada"}, {"id": "2", "response": "California"}])

    def test_gate_rejects_structurally_broken_outputs(self):
        decision = assess_preflight({"nonempty_rate": 0.5, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.1, "verbose_rate": 0.1})
        self.assertFalse(decision["promote"])
        self.assertIn("nonempty_rate", decision["failures"])
        self.assertIn("markup_rate", decision["failures"])
        self.assertIn("verbose_rate", decision["failures"])

    def test_thinking_mode_selection_uses_accuracy_after_quality_gate(self):
        direct = {"exact_match": 0.25, "f1": 0.30, "nonempty_rate": 1.0, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.0, "verbose_rate": 0.0}
        two_pass = {"exact_match": 0.35, "f1": 0.40, "nonempty_rate": 1.0, "copying_rate": 0.0, "truncation_rate": 0.0, "markup_rate": 0.0, "verbose_rate": 0.0}
        self.assertEqual(select_thinking_mode({"direct": direct, "two_pass": two_pass})["selected"], "two_pass")

    def test_cited_summary_and_gate_cover_query_retrieval_format_citation_support_and_rendering(self):
        examples = [{"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada"}]
        predictions = [{
            "id": "1",
            "raw_query": "Ada author",
            "ranked_search_results": [{"source_id": "S001", "title": "Ada", "url": "https://example.test/1", "snippet": "Ada", "original_rank": 1, "retrieval_rank": 1, "bm25_score": 1.0, "query_hash": "q", "corpus_hash": "c"}],
            "raw_response": "Answer: Ada\nReasoning: Source says Ada [S001].\nSources: S001",
            "parsed_response": {"answer": "Ada", "reasoning": "Source says Ada [S001].", "source_ids": ["S001"]},
            "rendered_visible_response": "Answer: Ada\nReasoning: Source says Ada [S001].\nSources:\n[S001] Ada — https://example.test/1",
            "truncated": False,
            "truncation": {"query": False, "response": False},
            "retrieval_metrics": {"recall@8": 1.0},
        }]
        metrics = summarize_response_quality(examples, predictions, protocol="active-search")
        self.assertEqual(metrics["valid_format_rate"], 1.0)
        self.assertEqual(metrics["valid_citation_rate"], 1.0)
        self.assertEqual(metrics["lexical_cited_answer_support_rate"], 1.0)
        self.assertNotIn("cited_answer_support_rate", metrics)
        self.assertEqual(metrics["retrieval_recall@8"], 1.0)
        self.assertEqual(metrics["rendered_visible_rate"], 1.0)
        self.assertTrue(assess_preflight(metrics)["promote"])

    def test_active_preflight_uses_lexical_support_label_and_rejects_duplicate_example_ids(self):
        duplicate_examples = [self.examples[0], dict(self.examples[0])]
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_response_quality(duplicate_examples, [{"id": "1", "response": "Ada", "truncated": False}])

    def test_active_preflight_aggregates_invalid_and_truncated_model_outputs(self):
        def sources(label):
            return [{"source_id": f"S{index:03d}", "original_rank": index, "title": f"{label} {index}", "url": f"https://example.test/{label}/{index}", "snippet": f"{label} evidence {index}"} for index in range(1, 9)]

        examples = [
            {"id": "invalid", "question": "Invalid?", "gold_answer": "invalid", "sources": sources("invalid")},
            {"id": "query-truncated", "question": "Query?", "gold_answer": "query", "sources": sources("query")},
            {"id": "malformed", "question": "Malformed?", "gold_answer": "malformed", "sources": sources("malformed")},
            {"id": "response-truncated", "question": "Response?", "gold_answer": "response", "sources": sources("response")},
        ]
        results = run_fixed_retrieval_pipeline(
            examples,
            query_generate_batch=lambda _prompts: [GeneratedText("bad\nquery", False), GeneratedText("query", True), GeneratedText("malformed", False), GeneratedText("response", False)],
            response_generate_batch=lambda prompts: [GeneratedText("bad", False), GeneratedText("Answer: response\nReasoning: Source [S001].\nSources: S001", True)][: len(prompts)],
            policy_hash="policy-v1",
        )
        metrics = summarize_response_quality(examples, results, protocol="active-search")
        self.assertEqual(metrics["query_truncation_rate"], 0.25)
        self.assertEqual(metrics["response_truncation_rate"], 0.25)
        self.assertEqual(metrics["malformed_rate"], 1.0)
        self.assertFalse(assess_preflight(metrics)["promote"])


if __name__ == "__main__":
    unittest.main()
