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
    def test_default_k1_is_frozen_at_one_point_two(self):
        self.assertEqual(FixedBM25Retriever([source("S001", 1, "evidence")]).k1, 1.2)

    def test_query_tokenization_is_explicit_and_retriever_has_no_gold_answer_input(self):
        self.assertEqual(tokenize_query(" Ada-Lovelace's algorithm! "), ("ada", "lovelace", "s", "algorithm"))
        self.assertNotIn("gold_answer", inspect.signature(FixedBM25Retriever.search).parameters)

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

    def test_ties_use_original_rank_then_source_id(self):
        sources = [
            source("S002", 1, "same text"),
            source("S001", 1, "same text"),
            source("S003", 2, "different text"),
        ]

        results = FixedBM25Retriever(sources).search("same", top_k=3)

        self.assertEqual([row["source_id"] for row in results], ["S001", "S002", "S003"])

    def test_invalid_query_sources_parameters_and_bm25_parameters_fail_explicitly(self):
        valid_sources = [source("S001", 1, "some evidence")]
        with self.assertRaisesRegex(ValueError, "query"):
            FixedBM25Retriever(valid_sources).search("!!!", top_k=1)
        with self.assertRaisesRegex(ValueError, "top_k"):
            FixedBM25Retriever(valid_sources).search("evidence", top_k=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            FixedBM25Retriever(valid_sources).search("evidence", top_k=2)
        with self.assertRaisesRegex(ValueError, "source"):
            FixedBM25Retriever([])
        with self.assertRaisesRegex(ValueError, "k1"):
            FixedBM25Retriever(valid_sources, k1=0)
        with self.assertRaisesRegex(ValueError, "b"):
            FixedBM25Retriever(valid_sources, b=1.1)

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


if __name__ == "__main__":
    unittest.main()
