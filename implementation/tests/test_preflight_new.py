import unittest
import copy

from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
from text_feedback_dpo.preflight import assess_preflight, select_preflight_rows, select_thinking_mode, summarize_response_quality
from text_feedback_dpo.runtime import GeneratedText


class PreflightQualityTest(unittest.TestCase):
    @staticmethod
    def _active_fixture():
        sources = [
            {"source_id": f"S{index:03d}", "original_rank": index, "title": f"Ada {index}",
             "url": f"https://example.test/{index}", "snippet": f"Ada evidence {index}"}
            for index in range(1, 4)
        ]
        example = {"id": "1", "question": "Who?", "gold_answer": "Ada", "sources": sources}
        prediction = run_fixed_retrieval_pipeline(
            [example],
            query_generate_batch=lambda _prompts: [GeneratedText("Ada author", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("Answer: Ada\nReasoning: Source says Ada [S001].\nSources: S001", False)],
            policy_hash="policy-v1",
        )[0]
        return example, prediction

    def test_active_preflight_recomputes_dataset_owned_retrieval_and_rejects_tampering(self):
        example, prediction = self._active_fixture()
        for field, value in (("bm25_score", 999.0), ("url", "https://evil.test"), ("requested_top_k", 7)):
            tampered = copy.deepcopy(prediction)
            tampered["ranked_search_results"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ranked retrieval"):
                summarize_response_quality([example], [tampered], protocol="active-search")
        tampered_metrics = copy.deepcopy(prediction)
        tampered_metrics["retrieval_metrics"]["recall@8"] = 0.0
        with self.assertRaisesRegex(ValueError, "retrieval_metrics"):
            summarize_response_quality([example], [tampered_metrics], protocol="active-search")

    def test_active_preflight_rejects_every_forged_persisted_canonical_field(self):
        example, prediction = self._active_fixture()
        mutations = {
            "cited_score": {**prediction["cited_score"], "citation_recall": 0.25},
            "parsed_response": {**prediction["parsed_response"], "answer": "forged"},
            "rendered_visible_response": "forged",
            "error_code": "forged",
            "query_prompt": prediction["query_prompt"] + " ",
            "query_prompt_hash": "0" * 64,
            "response_prompt": prediction["response_prompt"] + " ",
            "response_prompt_hash": "1" * 64,
            "retrieval_context_hash": "2" * 64,
            "canonical_ranked_search_results": [],
        }
        for field, value in mutations.items():
            forged = copy.deepcopy(prediction)
            forged[field] = value
            expected = "canonical ranked retrieval" if field == "canonical_ranked_search_results" else field
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                summarize_response_quality([example], [forged], protocol="active-search")

        metrics = summarize_response_quality([example], [prediction], protocol="active-search")
        self.assertEqual(metrics["answer_capability_exact_match"], 1.0)
        self.assertEqual(metrics["protocol_exact_match"], 1.0)
        self.assertEqual(metrics["retrieval_recall@1"], 1.0)
        self.assertEqual(metrics["retrieval_recall@3"], 1.0)
        self.assertEqual(metrics["retrieval_recall@5"], 1.0)
        self.assertEqual(metrics["retrieval_recall@8"], 1.0)
        self.assertEqual(metrics["retrieval_mrr"], 1.0)
        self.assertEqual(metrics["empty_query_rate"], 0.0)
        self.assertEqual(metrics["invalid_query_rate"], 0.0)
        self.assertEqual(metrics["duplicate_citation_rate"], 0.0)
        self.assertIn("query_words", metrics)
        self.assertIn("timing_ms", metrics)
        self.assertIn("throughput_examples_per_second", metrics)

    def test_active_preflight_allows_null_response_only_for_canonical_query_stage_errors(self):
        example, prediction = self._active_fixture()
        prediction["raw_response"] = None
        prediction["error_code"] = "line_count"
        with self.assertRaisesRegex(ValueError, "raw_response text"):
            summarize_response_quality([example], [prediction], protocol="active-search")

    def test_active_preflight_rejects_contradictory_stage_truncation_flags(self):
        example, prediction = self._active_fixture()
        prediction["query_truncated"] = True
        with self.assertRaisesRegex(ValueError, "query_truncated.*mismatch"):
            summarize_response_quality([example], [prediction], protocol="active-search")

    def test_archival_preflight_rejects_any_active_only_schema_field(self):
        predictions = [
            {"id": "1", "response": "Ada", "truncated": False, "raw_response": "active"},
            {"id": "2", "response": "California", "truncated": False},
        ]
        with self.assertRaisesRegex(ValueError, "active-search fields"):
            summarize_response_quality(self.examples, predictions, protocol="archival")

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
        example, prediction = self._active_fixture()
        metrics = summarize_response_quality([example], [prediction], protocol="active-search")
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
        self.assertEqual(metrics["any_stage_truncation_rate"], 0.5)
        self.assertFalse(assess_preflight(metrics)["promote"])


if __name__ == "__main__":
    unittest.main()
