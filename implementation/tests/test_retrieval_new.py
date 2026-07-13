import copy
import inspect
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import text_feedback_dpo.retrieval as retrieval
from text_feedback_dpo import scoring
from text_feedback_dpo.config import load_config
from text_feedback_dpo.retrieval import FixedBM25Retriever, retrieval_metrics, tokenize_query


def source(source_id: str, original_rank: int, snippet: str, *, title: str | None = None) -> dict:
    return {
        "source_id": source_id,
        "original_rank": original_rank,
        "title": title or f"Title {source_id}",
        "url": f"https://example.test/{source_id}",
        "snippet": snippet,
        "related_links": None,
    }


class FixedBM25RetrievalTest(unittest.TestCase):
    def test_related_links_accept_real_searchqa_shapes_and_affect_canonical_hash(self):
        shaped = [
            {**source("S001", 1, "alpha evidence"), "related_links": None},
            {**source("S002", 2, "beta evidence"), "related_links": " https://related.test/one "},
            {**source("S003", 3, "gamma evidence"), "related_links": [" https://related.test/two ", ""]},
        ]

        results = FixedBM25Retriever(shaped).search("evidence", top_k=8)

        self.assertEqual(results[0]["related_links"], None)
        self.assertEqual(results[1]["related_links"], "https://related.test/one")
        self.assertEqual(results[2]["related_links"], ["https://related.test/two", ""])
        changed = copy.deepcopy(shaped)
        changed[2]["related_links"] = ["https://related.test/different"]
        self.assertNotEqual(
            results[0]["corpus_hash"],
            FixedBM25Retriever(changed).search("evidence", top_k=8)[0]["corpus_hash"],
        )
        malformed = copy.deepcopy(shaped)
        malformed[2]["related_links"] = ["valid", 3]
        with self.assertRaisesRegex(ValueError, "only strings"):
            FixedBM25Retriever(malformed)

    def test_default_k1_is_frozen_at_one_point_two(self):
        self.assertEqual(FixedBM25Retriever([source("S001", 1, "evidence")]).k1, 1.2)

    def test_query_tokenization_is_explicit_and_retriever_has_no_gold_answer_input(self):
        self.assertEqual(tokenize_query(" Ada-Lovelace's algorithm! "), ("ada", "lovelace", "s", "algorithm"))
        self.assertNotIn("gold_answer", inspect.signature(FixedBM25Retriever.search).parameters)

    def test_query_tokenization_normalizes_unicode_before_casefolding(self):
        self.assertEqual(tokenize_query("Caf\u00e9"), tokenize_query("Cafe\u0301"))
        self.assertEqual(tokenize_query("Cafe\u0301"), ("caf\u00e9",))

    def test_bm25_ranks_matching_sources_and_returns_hashes_and_canonical_metadata(self):
        sources = [
            source("S001", 1, "Ada Lovelace wrote the first algorithm."),
            source("S002", 2, "The history of algorithms includes many contributors."),
        ]

        results = FixedBM25Retriever(sources, k1=1.5, b=0.75).search("Who wrote the algorithm?", top_k=2)

        self.assertEqual([row["source_id"] for row in results], ["S001", "S002"])
        self.assertEqual(results[0]["title"], "Title S001")
        self.assertEqual(results[0]["url"], "https://example.test/S001")
        self.assertEqual(results[0]["snippet"], sources[0]["snippet"])
        self.assertEqual(results[0]["retrieval_rank"], 1)
        self.assertGreater(results[0]["bm25_score"], results[1]["bm25_score"])
        self.assertNotIn("score", results[0])
        self.assertEqual(results[0]["matched_query_terms"], ["wrote", "the", "algorithm"])
        self.assertEqual(results[0]["query_hash"], results[1]["query_hash"])
        self.assertEqual(results[0]["corpus_hash"], results[1]["corpus_hash"])
        self.assertEqual(len(results[0]["query_hash"]), 64)
        self.assertEqual(len(results[0]["corpus_hash"]), 64)

    def test_zero_score_results_are_explicit_and_keep_original_order(self):
        sources = [
            source("S003", 3, "Gamma evidence."),
            source("S001", 1, "Alpha evidence."),
            source("S002", 2, "Beta evidence."),
        ]

        results = FixedBM25Retriever(sources).search("unseen-token", top_k=3)

        self.assertEqual([row["source_id"] for row in results], ["S001", "S002", "S003"])
        self.assertEqual([row["bm25_score"] for row in results], [0.0, 0.0, 0.0])
        self.assertEqual([row["matched_query_terms"] for row in results], [[], [], []])
        self.assertEqual([row["retrieval_rank"] for row in results], [1, 2, 3])

    def test_score_ties_use_original_rank(self):
        sources = [
            source("S002", 2, "same text"),
            source("S001", 1, "same text"),
        ]

        results = FixedBM25Retriever(sources).search("same", top_k=2)

        self.assertEqual([row["source_id"] for row in results], ["S001", "S002"])

    def test_sources_reject_duplicate_original_rank(self):
        with self.assertRaisesRegex(ValueError, "original_rank"):
            FixedBM25Retriever([source("S001", 1, "alpha"), source("S002", 1, "beta")])

    def test_corpus_identity_and_results_are_independent_of_input_order(self):
        sources = [
            source("S003", 3, "gamma evidence"),
            source("S001", 1, "alpha evidence"),
            source("S002", 2, "beta evidence"),
        ]

        forward = FixedBM25Retriever(sources).search("evidence", top_k=3)
        reversed_input = FixedBM25Retriever(list(reversed(sources))).search("evidence", top_k=3)

        self.assertEqual(forward, reversed_input)

    def test_invalid_query_sources_parameters_and_bm25_parameters_fail_explicitly(self):
        valid_sources = [source("S001", 1, "some evidence")]
        with self.assertRaisesRegex(ValueError, "query"):
            FixedBM25Retriever(valid_sources).search("!!!", top_k=1)
        with self.assertRaisesRegex(ValueError, "top_k"):
            FixedBM25Retriever(valid_sources).search("evidence", top_k=0)
        with self.assertRaisesRegex(ValueError, "source"):
            FixedBM25Retriever([])
        with self.assertRaisesRegex(ValueError, "k1"):
            FixedBM25Retriever(valid_sources, k1=0)
        with self.assertRaisesRegex(ValueError, "b"):
            FixedBM25Retriever(valid_sources, b=1.1)

    def test_requested_top_k_returns_all_available_sources_with_explicit_effective_k(self):
        results = FixedBM25Retriever([source("S001", 1, "evidence")]).search("evidence", top_k=8)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["requested_top_k"], 8)
        self.assertEqual(results[0]["effective_top_k"], 1)
        self.assertEqual(results[0]["source_count"], 1)

    def test_retrieval_metrics_use_normalized_answer_matching_and_validate_inputs(self):
        sources = [
            source("S001", 1, "This is unrelated author evidence."),
            source("S002", 2, "A biography identifies ADA-LOVELACE as the author."),
            source("S003", 3, "Another unrelated source."),
            source("S004", 4, "Further context."),
            source("S005", 5, "Last source."),
        ]
        ranked = FixedBM25Retriever(sources).search("author evidence", top_k=5)

        metrics = retrieval_metrics(ranked, "The Ada Lovelace", ks=(1, 3, 5))

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["first_answer_rank"], 2)
        self.assertNotIn("answer_bearing_recall@1", metrics)
        self.assertNotIn("first_answer_bearing_rank", metrics)
        with self.assertRaisesRegex(ValueError, "gold"):
            retrieval_metrics(ranked, "", ks=(1,))
        with self.assertRaisesRegex(ValueError, "k"):
            retrieval_metrics(ranked, "Ada Lovelace", ks=(0,))

    def test_retrieval_metrics_reuse_the_shared_answer_normalizer(self):
        self.assertIs(retrieval.normalize_answer, scoring.normalize_answer)
        self.assertFalse(hasattr(retrieval, "_normalize_answer"))

    def test_retrieval_metrics_reject_malformed_ranked_results(self):
        ranked = FixedBM25Retriever(
            [source("S001", 1, "alpha evidence"), source("S002", 2, "Ada Lovelace evidence")]
        ).search("evidence", top_k=2)
        corruptions = [
            ("missing retrieval rank", 0, "retrieval_rank", None, "retrieval_rank", True),
            ("wrong retrieval rank", 1, "retrieval_rank", 1, "retrieval_rank", False),
            ("empty source id", 0, "source_id", "", "source_id", False),
            ("duplicate source id", 1, "source_id", "S001", "source_id", False),
            ("nonpositive original rank", 0, "original_rank", 0, "original_rank", False),
            ("duplicate original rank", 1, "original_rank", 1, "original_rank", False),
            ("missing bm25 score", 0, "bm25_score", None, "bm25_score", True),
            ("negative bm25 score", 0, "bm25_score", -0.1, "bm25_score", False),
            ("nonfinite bm25 score", 0, "bm25_score", math.inf, "bm25_score", False),
            ("empty query hash", 0, "query_hash", "", "query_hash", False),
            ("mismatched query hash", 1, "query_hash", "different", "query_hash", False),
            ("empty corpus hash", 0, "corpus_hash", "", "corpus_hash", False),
            ("mismatched corpus hash", 1, "corpus_hash", "different", "corpus_hash", False),
            ("missing title", 0, "title", None, "title", True),
            ("empty title", 0, "title", "", "title", False),
            ("missing snippet", 0, "snippet", None, "snippet", True),
            ("empty snippet", 0, "snippet", "", "snippet", False),
        ]
        for label, index, field, value, expected_error, remove in corruptions:
            malformed = copy.deepcopy(ranked)
            if remove:
                malformed[index].pop(field)
            else:
                malformed[index][field] = value
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, expected_error):
                retrieval_metrics(malformed, "Ada Lovelace", ks=(1, 2))

    def test_searchqa_config_requires_fixed_bm25_retrieval_settings(self):
        config = load_config(Path("configs/searchqa.yaml"))
        self.assertEqual(
            config["retrieval"],
            {"backend": "fixed_bm25", "top_k": 8, "k1": 1.2, "b": 0.75, "schema_version": 1},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            altered = dict(config)
            altered["retrieval"] = dict(config["retrieval"], backend="web_search")
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "backend"):
                load_config(path)

    def test_searchqa_config_freezes_top_k_and_rejects_nonfinite_bm25_parameters(self):
        config = load_config(Path("configs/searchqa.yaml"))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for field, value in (("top_k", 7), ("k1", math.inf), ("b", math.nan)):
                altered = dict(config)
                altered["retrieval"] = dict(config["retrieval"], **{field: value})
                path.write_text(yaml.safe_dump(altered), encoding="utf-8")
                expected = "exactly 8" if field == "top_k" else field
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                    load_config(path)

    def test_searchqa_config_schema_version_requires_integer_one_not_bool(self):
        config = load_config(Path("configs/searchqa.yaml"))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            altered = dict(config)
            altered["retrieval"] = dict(config["retrieval"], schema_version=True)
            path.write_text(yaml.safe_dump(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
