import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

from text_feedback_dpo.dataset import build_sft_rows, dataset_fingerprint, load_searchqa_split_with_stats
from text_feedback_dpo.searchqa import materialize_row, pack_evidence


class SearchQADataTest(unittest.TestCase):
    def test_materialize_row_preserves_aligned_official_source_records(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["Ada was a writer.", "Other text"],
                    "titles": ["Ada biography", "Other source"],
                    "urls": ["https://example.test/ada", "https://example.test/other"],
                    "related_links": ["https://example.test/related", ""],
                },
            },
            split="train",
            index=3,
        )
        self.assertEqual(row["id"], "train-3")
        self.assertEqual(row["question"], "Who?")
        self.assertEqual(row["gold_answer"], "Ada")
        self.assertEqual(row["snippets"], ["Ada was a writer.", "Other text"])
        self.assertEqual(
            row["sources"],
            [
                {
                    "source_id": "S001",
                    "original_rank": 1,
                    "title": "Ada biography",
                    "url": "https://example.test/ada",
                    "snippet": "Ada was a writer.",
                    "related_links": "https://example.test/related",
                },
                {
                    "source_id": "S002",
                    "original_rank": 2,
                    "title": "Other source",
                    "url": "https://example.test/other",
                    "snippet": "Other text",
                    "related_links": "",
                },
            ],
        )

    def test_materialize_row_filters_blank_snippets_without_shifting_source_metadata(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["first", "  ", "third"],
                    "titles": ["First title", "Blank title", "Third title"],
                    "urls": ["https://example.test/first", "https://example.test/blank", "https://example.test/third"],
                    "related_links": ["first-related", "blank-related", "third-related"],
                },
            },
            split="validation",
            index=0,
        )
        self.assertEqual([source["source_id"] for source in row["sources"]], ["S001", "S003"])
        self.assertEqual([source["original_rank"] for source in row["sources"]], [1, 3])
        self.assertEqual([source["title"] for source in row["sources"]], ["First title", "Third title"])
        self.assertEqual([source["url"] for source in row["sources"]], ["https://example.test/first", "https://example.test/third"])
        self.assertEqual(row["snippets"], ["first", "third"])

    def test_materialize_row_accepts_official_raw_records_with_provenance(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": [
                    {"snippet": "Ada evidence", "title": "Ada", "url": "https://example.test/ada", "related_links": ""},
                ],
            },
            split="train",
            index=1,
        )
        self.assertEqual(row["sources"][0]["source_id"], "S001")
        self.assertEqual(row["sources"][0]["url"], "https://example.test/ada")

    def test_materialize_row_accepts_real_official_related_links_shapes(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": [
                    {
                        "snippet": "Ada evidence",
                        "title": "Ada",
                        "url": "https://example.test/ada",
                        "related_links": [],
                    },
                    {
                        "snippet": "Other evidence",
                        "title": "Other",
                        "url": "https://example.test/other",
                        "related_links": ["https://example.test/related"],
                    },
                ],
            },
            split="validation",
            index=1,
        )
        self.assertEqual(row["sources"][0]["related_links"], [])
        self.assertEqual(row["sources"][1]["related_links"], ["https://example.test/related"])

    def test_materialize_row_rejects_malformed_related_links_entries(self):
        with self.assertRaisesRegex(ValueError, "related_links"):
            materialize_row(
                {
                    "question": "Who?",
                    "answer": "Ada",
                    "search_results": [
                        {
                            "snippet": "Ada evidence",
                            "title": "Ada",
                            "url": "https://example.test/ada",
                            "related_links": [7],
                        },
                    ],
                },
                split="validation",
                index=1,
            )

    def test_materialize_row_fails_explicitly_on_source_array_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            materialize_row(
                {
                    "question": "Who?",
                    "answer": "Ada",
                    "search_results": {
                        "snippets": ["Ada evidence", "Other evidence"],
                        "titles": ["Ada"],
                        "urls": ["https://example.test/ada", "https://example.test/other"],
                    },
                },
                split="train",
                index=0,
            )

    def test_materialize_row_filters_unusable_metadata_without_shifting_ranks(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["missing title", "usable", "missing url", "  "],
                    "titles": ["  ", "Usable title", "Missing URL", "Blank snippet"],
                    "urls": ["https://example.test/one", "https://example.test/two", "  ", "https://example.test/four"],
                },
            },
            split="train",
            index=0,
        )
        self.assertEqual([source["source_id"] for source in row["sources"]], ["S002"])
        self.assertEqual(row["source_filter_stats"], {
            "input_records": 4,
            "usable_records": 1,
            "dropped_records": 3,
            "drop_reasons": {"blank_snippet": 1, "missing_title": 1, "missing_url": 1},
        })

    def test_materialize_row_filters_nonempty_tokenless_snippets_for_bm25(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["!!!", "Ada evidence", "—"],
                    "titles": ["Punctuation one", "Ada title", "Punctuation two"],
                    "urls": [
                        "https://example.test/one",
                        "https://example.test/ada",
                        "https://example.test/two",
                    ],
                },
            },
            split="validation",
            index=0,
        )
        self.assertEqual([source["source_id"] for source in row["sources"]], ["S002"])
        self.assertEqual(row["source_filter_stats"], {
            "input_records": 3,
            "usable_records": 1,
            "dropped_records": 2,
            "drop_reasons": {"no_tokens_snippet": 2},
        })

    def test_materialize_row_fails_only_when_no_source_has_required_metadata(self):
        with self.assertRaisesRegex(ValueError, "no usable"):
            materialize_row(
                {
                    "question": "Who?",
                    "answer": "Ada",
                    "search_results": {
                        "snippets": ["Ada evidence"],
                        "titles": ["  "],
                        "urls": ["https://example.test/ada"],
                    },
                },
                split="train",
                index=0,
            )

    def test_materialize_row_fails_clearly_for_source_less_or_unsupported_schemas(self):
        for search_results in (["Ada evidence"], {"snippets": ["Ada evidence"]}):
            with self.subTest(search_results=search_results):
                with self.assertRaisesRegex(ValueError, "source provenance"):
                    materialize_row({"question": "Who?", "answer": "Ada", "search_results": search_results}, split="train", index=0)

    def test_materialize_row_requires_usable_source_and_allows_missing_optional_related_links(self):
        with self.assertRaisesRegex(ValueError, "no usable"):
            materialize_row(
                {
                    "question": "Who?",
                    "answer": "Ada",
                    "search_results": {
                        "snippets": ["  "],
                        "titles": ["Ada"],
                        "urls": ["https://example.test/ada"],
                    },
                },
                split="train",
                index=0,
            )
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["Ada evidence"],
                    "titles": ["Ada"],
                    "urls": ["https://example.test/ada"],
                },
            },
            split="train",
            index=0,
        )
        self.assertIsNone(row["sources"][0]["related_links"])

    def test_official_loader_reports_source_schema_and_dropped_rows(self):
        from text_feedback_dpo.dataset import _load_official_searchqa_zip

        with TemporaryDirectory() as directory:
            archive_path = Path(directory) / "train.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "000-valid.json",
                    '{"question":"Who?","answer":"Ada","search_results":[{"snippet":"Ada evidence","title":"Ada","url":"https://example.test/ada","related_links":""}]}'
                )
                archive.writestr(
                    "001-empty.json",
                    '{"question":"Who?","answer":"Ada","search_results":[{"snippet":"","title":"Ada","url":"https://example.test/ada","related_links":""}]}'
                )
            with patch("huggingface_hub.hf_hub_download", return_value=str(archive_path)):
                rows, stats = _load_official_searchqa_zip("train", "revision", None)
        self.assertEqual(rows[0]["sources"][0]["source_id"], "S001")
        self.assertEqual(stats["source_schema"], "searchqa.search_results.v1")
        self.assertEqual(stats["source_schema_version"], 1)
        self.assertEqual(stats["drop_reasons"], {"no_usable_evidence": 1})
        self.assertEqual(stats["source_records"], {
            "input_records": 2,
            "usable_records": 1,
            "dropped_records": 1,
            "drop_reasons": {"blank_snippet": 1},
        })

    def test_official_loader_counts_empty_raw_search_results_as_dropped(self):
        from text_feedback_dpo.dataset import _load_official_searchqa_zip

        with TemporaryDirectory() as directory:
            archive_path = Path(directory) / "train.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "000-empty-list.json",
                    '{"question":"Who?","answer":"Ada","search_results":[]}'
                )
                archive.writestr(
                    "001-valid.json",
                    '{"question":"Who?","answer":"Ada","search_results":[{"snippet":"Ada evidence","title":"Ada","url":"https://example.test/ada","related_links":""}]}'
                )
            with patch("huggingface_hub.hf_hub_download", return_value=str(archive_path)):
                rows, stats = _load_official_searchqa_zip("train", "revision", 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "train-1")
        self.assertEqual(stats["source_rows"], 2)
        self.assertEqual(stats["dropped_rows"], 1)
        self.assertEqual(stats["drop_reasons"], {"no_usable_evidence": 1})

    def test_generic_loader_limit_counts_successfully_materialized_rows(self):
        raw_rows = [
            {"question": "Who?", "answer": "Ada", "search_results": []},
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["Ada evidence"],
                    "titles": ["Ada"],
                    "urls": ["https://example.test/ada"],
                },
            },
        ]
        with patch("datasets.load_dataset", return_value=raw_rows) as load_dataset:
            rows, stats = load_searchqa_split_with_stats("mock/searchqa", "train", revision="revision", limit=1)
        load_dataset.assert_called_once_with("mock/searchqa", split="train", revision="revision")
        self.assertEqual([row["id"] for row in rows], ["train-1"])
        self.assertEqual(stats["source_rows"], 2)
        self.assertEqual(stats["materialized_rows"], 1)
        self.assertEqual(stats["dropped_rows"], 1)
        self.assertEqual(stats["drop_reasons"], {"no_usable_evidence": 1})

    def test_stream_loader_updates_stats_as_rows_are_consumed(self):
        from text_feedback_dpo.dataset import stream_searchqa_split_with_stats

        raw_rows = [
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["Ada evidence"],
                    "titles": ["Ada"],
                    "urls": ["https://example.test/ada"],
                },
            },
            {
                "question": "Who else?",
                "answer": "Grace",
                "search_results": {
                    "snippets": ["Grace evidence"],
                    "titles": ["Grace"],
                    "urls": ["https://example.test/grace"],
                },
            },
        ]
        with patch("datasets.load_dataset", return_value=raw_rows) as load_dataset:
            rows, stats = stream_searchqa_split_with_stats("mock/searchqa", "train", revision="revision")
            self.assertEqual(stats["materialized_rows"], 0)
            self.assertEqual(next(rows)["id"], "train-0")
            self.assertEqual(stats["materialized_rows"], 1)
            self.assertEqual([row["id"] for row in rows], ["train-1"])
            self.assertEqual(stats["materialized_rows"], 2)
        load_dataset.assert_called_once_with("mock/searchqa", split="train", revision="revision")

    def test_dataset_fingerprint_is_deterministic_for_materialized_sources(self):
        row = materialize_row(
            {
                "question": "Who?",
                "answer": "Ada",
                "search_results": {
                    "snippets": ["Ada evidence"],
                    "titles": ["Ada"],
                    "urls": ["https://example.test/ada"],
                },
            },
            split="train",
            index=0,
        )
        self.assertEqual(dataset_fingerprint([row]), dataset_fingerprint([dict(row)]))
        changed = {**row, "sources": [{**row["sources"][0], "title": "Changed"}]}
        self.assertNotEqual(dataset_fingerprint([row]), dataset_fingerprint([changed]))

    def test_pack_evidence_is_deterministic_and_never_exceeds_budget(self):
        snippets = ["one two", "three four", "five six"]
        packed = pack_evidence(snippets, max_tokens=4, token_count=lambda text: len(text.split()))
        self.assertEqual(packed, "one two\nthree four")

    def test_legacy_gold_answer_sft_target_is_rejected(self):
        row = {
            "id": "train-0",
            "question": "Who?",
            "gold_answer": "Ada",
            "packed_evidence": "prefix " * 500 + "Ada evidence near the answer " + "suffix " * 500,
        }
        with self.assertRaisesRegex(RuntimeError, "removed unsafe SFT path"):
            build_sft_rows([row])


if __name__ == "__main__":
    unittest.main()
