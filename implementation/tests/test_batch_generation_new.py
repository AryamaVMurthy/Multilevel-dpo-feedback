import unittest
from unittest.mock import patch

from text_feedback_dpo.batch_generation import _hash, _record_output, generate_batch, parse_search_query, run_fixed_retrieval_pipeline
from text_feedback_dpo.runtime import GeneratedText, StudentGeneration


def source_records(prefix: str) -> list[dict]:
    return [
        {
            "source_id": f"S{index:03d}",
            "original_rank": index,
            "title": f"{prefix} source {index}",
            "url": f"https://example.test/{prefix}/{index}",
            "snippet": f"{prefix} evidence source {index} contains the relevant fact.",
        }
        for index in range(1, 9)
    ]


class BatchGenerationTest(unittest.TestCase):
    def test_search_query_parser_is_strict_without_rejecting_legitimate_math(self):
        self.assertEqual(parse_search_query("prove 3 > 2"), "prove 3 > 2")
        for invalid in (
            "!!!", "two\nlines", "```query```", '{"query":"Ada"}',
            "<query>Ada</query>", "Answer: Ada", "Reasoning: Ada", "Sources: S001",
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen",
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "query_invalid_format"):
                parse_search_query(invalid)

    def test_batch_generation_calls_provider_once_and_preserves_order(self):
        calls = []

        def provider(prompts, **kwargs):
            calls.append((prompts, kwargs))
            return prompts

        outputs = generate_batch(provider, ["a", "b"], max_new_tokens=32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outputs[0]["response"], "a")
        self.assertEqual(outputs[1]["response"], "b")

    def test_batch_generation_rejects_wrong_output_cardinality(self):
        with self.assertRaisesRegex(ValueError, "cardinality"):
            generate_batch(lambda prompts, **kwargs: ["one"], ["a", "b"])

    def test_active_search_rebuilds_prompts_and_emits_ordered_cited_artifacts(self):
        rows = [
            {"id": "row-2", "question": "What is beta?", "gold_answer": "beta", "sources": source_records("beta"), "prompt": "STALE"},
            {"id": "row-1", "question": "What is alpha?", "gold_answer": "alpha", "sources": source_records("alpha"), "prompt": "STALE"},
        ]
        query_calls = []
        response_calls = []

        def query_provider(prompts, **_kwargs):
            query_calls.append(prompts)
            return [GeneratedText("beta fact", False), GeneratedText("alpha fact", False)]

        def response_provider(prompts, **_kwargs):
            response_calls.append(prompts)
            return [
                GeneratedText(
                    f"Answer: {'beta' if 'beta?' in prompt else 'alpha'}\n"
                    f"Reasoning: The source states {'beta' if 'beta?' in prompt else 'alpha'} [S001].\n"
                    "Sources: S001",
                    False,
                )
                for prompt in prompts
            ]

        results = run_fixed_retrieval_pipeline(
            rows,
            query_generate_batch=query_provider,
            response_generate_batch=response_provider,
            query_batch_size=2,
            response_batch_size=1,
            policy_hash="policy-v1",
        )

        self.assertEqual([row["id"] for row in results], ["row-2", "row-1"])
        self.assertEqual(len(query_calls), 1)
        self.assertEqual(len(response_calls), 2)
        self.assertTrue(all("STALE" not in prompt for prompt in query_calls[0]))
        self.assertTrue(all("What is beta?" in prompt or "What is alpha?" in prompt for prompt in query_calls[0]))
        self.assertEqual(results[0]["raw_query"], "beta fact")
        self.assertEqual(results[0]["ranked_search_results"][0]["retrieval_rank"], 1)
        self.assertEqual(results[0]["parsed_response"]["answer"], "beta")
        self.assertIn("https://example.test/beta/1", results[0]["rendered_visible_response"])
        self.assertNotIn("<", results[0]["rendered_visible_response"])
        self.assertEqual(results[0]["prompt_version"], "fixed-retrieval-cited-v1")
        self.assertEqual(results[0]["policy_hash"], "policy-v1")
        self.assertEqual(set(results[0]["timings_ms"]), {
            "query_generation_batch_wall_ms", "query_generation_amortized_per_item_ms",
            "retrieval_individual_ms", "response_generation_batch_wall_ms",
            "response_generation_amortized_per_item_ms", "pipeline_wall_ms",
        })
        self.assertNotIn("total_example_ms", results[0]["timings_ms"])
        self.assertEqual(results[0]["timings_ms"]["pipeline_wall_ms"], results[1]["timings_ms"]["pipeline_wall_ms"])
        self.assertFalse(results[0]["truncation"]["query"])
        self.assertFalse(results[0]["truncation"]["response"])

    def test_active_search_preserves_requested_and_effective_retrieval_shape_for_sparse_rows(self):
        row = {"id": "sparse", "question": "What is sparse?", "gold_answer": "sparse", "sources": source_records("sparse")[:3]}
        result = run_fixed_retrieval_pipeline(
            [row],
            query_generate_batch=lambda _prompts: [GeneratedText("sparse evidence", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("Answer: sparse\nReasoning: Source states sparse [S001].\nSources: S001", False)],
            policy_hash="policy-v1",
        )[0]
        ranked = result["ranked_search_results"]
        self.assertEqual(len(ranked), 3)
        self.assertTrue(all(item["requested_top_k"] == 8 for item in ranked))
        self.assertTrue(all(item["effective_top_k"] == 3 for item in ranked))
        self.assertTrue(all(item["source_count"] == 3 for item in ranked))

    def test_active_pipeline_accepts_real_related_links_shape(self):
        sources = source_records("real")[:3]
        sources[0]["related_links"] = None
        sources[1]["related_links"] = "https://related.test/one"
        sources[2]["related_links"] = ["https://related.test/two"]
        result = run_fixed_retrieval_pipeline(
            [{"id": "real", "question": "What is real?", "gold_answer": "real", "sources": sources}],
            query_generate_batch=lambda _prompts: [GeneratedText("real evidence", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("Answer: real\nReasoning: Source says real [S001].\nSources: S001", False)],
            policy_hash="policy-v1",
        )[0]
        self.assertEqual(result["ranked_search_results"][2]["related_links"], ["https://related.test/two"])

    def test_active_search_compacts_invalid_query_examples_without_reordering_outputs(self):
        rows = [
            {"id": "bad", "question": "Bad?", "gold_answer": "bad", "sources": source_records("bad")},
            {"id": "good", "question": "Good?", "gold_answer": "good", "sources": source_records("good")},
        ]
        response_prompts = []

        def query_provider(_prompts, **_kwargs):
            return [GeneratedText("two lines\nnot allowed", False), GeneratedText("good fact", False)]

        def response_provider(prompts, **_kwargs):
            response_prompts.extend(prompts)
            return [GeneratedText("Answer: good\nReasoning: Source states good [S001].\nSources: S001", False)]

        results = run_fixed_retrieval_pipeline(
            rows,
            query_generate_batch=query_provider,
            response_generate_batch=response_provider,
            query_batch_size=2,
            response_batch_size=2,
            policy_hash="policy-v1",
        )

        self.assertEqual([row["id"] for row in results], ["bad", "good"])
        self.assertEqual(results[0]["error_code"], "query_invalid_format")
        self.assertIsNone(results[0]["rendered_visible_response"])
        self.assertEqual(len(response_prompts), 1)
        self.assertIn("Good?", response_prompts[0])

    def test_active_search_preserves_private_scratchpad_metadata_without_showing_it(self):
        row = {"id": "row", "question": "What?", "gold_answer": "answer", "sources": source_records("answer")}

        def query_provider(_prompts, **_kwargs):
            return [StudentGeneration("answer fact", "private query work", "two_pass", False, False)]

        def response_provider(_prompts, **_kwargs):
            return [StudentGeneration("Answer: answer\nReasoning: Source states answer [S001].\nSources: S001", "private response work", "two_pass", False, False)]

        result = run_fixed_retrieval_pipeline(
            [row], query_generate_batch=query_provider, response_generate_batch=response_provider, policy_hash="policy-v1"
        )[0]
        self.assertEqual(result["private_scratchpad"]["query"], "private query work")
        self.assertEqual(result["private_scratchpad"]["response"], "private response work")
        self.assertNotIn("private query work", result["rendered_visible_response"])

    def test_active_search_requires_explicit_truncation_metadata_and_rejects_truncated_response(self):
        row = {"id": "row", "question": "What?", "gold_answer": "answer", "sources": source_records("answer")}
        with self.assertRaisesRegex(ValueError, "truncated metadata"):
            run_fixed_retrieval_pipeline(
                [row], query_generate_batch=lambda _prompts: ["answer fact"],
                response_generate_batch=lambda _prompts: [], policy_hash="policy-v1",
            )

        result = run_fixed_retrieval_pipeline(
            [row],
            query_generate_batch=lambda _prompts: [GeneratedText("answer fact", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("Answer: answer\nReasoning: Source says answer [S001].\nSources: S001", True)],
            policy_hash="policy-v1",
        )[0]
        self.assertEqual(result["error_code"], "response_truncated")
        self.assertFalse(result["cited_score"]["parse_valid"])
        self.assertIsNone(result["parsed_response"])
        self.assertIsNone(result["rendered_visible_response"])
        self.assertEqual(result["cited_score"]["answer_capability_exact_match"], 1.0)
        self.assertEqual(result["cited_score"]["protocol_exact_match"], 0.0)
        self.assertFalse(result["cited_score"]["correct"])

    def test_all_model_failures_have_zero_retrieval_and_cited_diagnostics(self):
        rows = [
            {"id": "invalid-query", "question": "Invalid?", "gold_answer": "invalid", "sources": source_records("invalid")},
            {"id": "truncated-query", "question": "Truncated?", "gold_answer": "truncated", "sources": source_records("truncated")},
            {"id": "malformed-response", "question": "Malformed?", "gold_answer": "malformed", "sources": source_records("malformed")},
            {"id": "truncated-response", "question": "Response?", "gold_answer": "response", "sources": source_records("response")},
        ]
        query_outputs = [
            GeneratedText("not one line\ninvalid", False),
            GeneratedText("truncated query", True),
            GeneratedText("malformed fact", False),
            GeneratedText("response fact", False),
        ]

        def query_provider(_prompts):
            return query_outputs

        def response_provider(prompts):
            return [
                GeneratedText("not a three-line response", False),
                GeneratedText("Answer: response\nReasoning: Source says response [S001].\nSources: S001", True),
            ][: len(prompts)]

        results = run_fixed_retrieval_pipeline(
            rows,
            query_generate_batch=query_provider,
            response_generate_batch=response_provider,
            policy_hash="policy-v1",
        )
        self.assertEqual([row["error_code"] for row in results], ["query_invalid_format", "query_truncated", "line_count", "response_truncated"])
        for row in results:
            self.assertIsInstance(row["cited_score"], dict)
            self.assertIsInstance(row["retrieval_metrics"], dict)
            self.assertEqual(set(row["retrieval_metrics"]), {"recall@1", "recall@3", "recall@5", "recall@8", "reciprocal_rank", "mrr", "first_answer_rank"})
            if row["error_code"].startswith("query_"):
                self.assertEqual(row["cited_score"]["lexical_cited_answer_support"], 0.0)
                self.assertEqual(row["retrieval_metrics"]["mrr"], 0.0)
            else:
                self.assertFalse(row["cited_score"]["parse_valid"])

    def test_pipeline_passes_response_truncation_to_cited_scorer(self):
        row = {"id": "row", "question": "What?", "gold_answer": "answer", "sources": source_records("answer")}
        with patch("text_feedback_dpo.batch_generation.score_cited_response", wraps=__import__("text_feedback_dpo.scoring", fromlist=["score_cited_response"]).score_cited_response) as score:
            run_fixed_retrieval_pipeline(
                [row],
                query_generate_batch=lambda _prompts: [GeneratedText("answer fact", False)],
                response_generate_batch=lambda _prompts: [GeneratedText("bad", True)],
                policy_hash="policy-v1",
            )
        self.assertTrue(score.call_args.kwargs["truncated"])

    def test_provider_objects_and_hashes_have_no_hidden_defaults(self):
        class MissingTruncation:
            text = "query"

        with self.assertRaisesRegex(ValueError, "truncated"):
            _record_output(MissingTruncation())
        with self.assertRaises(TypeError):
            _hash({"not_json": object()})



if __name__ == "__main__":
    unittest.main()
