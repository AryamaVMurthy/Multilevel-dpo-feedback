import json
import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.searchqa import load_original_searchqa


class SearchQALoaderTest(unittest.TestCase):
    def _row(self, question: str, answer: str) -> dict:
        return {
            "question": question,
            "answer": answer,
            "search_results": [
                {
                    "snippets": f"Evidence supports {answer}.",
                    "titles": "Title",
                    "urls": "https://example.test/source",
                }
            ],
        }

    def _write_archive(self, root: Path) -> Path:
        archive = root / "searchqa.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for split, rows in {
                "train": [self._row("Who is A?", "A")],
                "validation": [self._row("Who is B?", "B")],
                "test": [self._row("Who is C?", "C")],
            }.items():
                handle.writestr(f"data/{split}.json", json.dumps(rows))
        return archive

    def test_loads_original_split_files_and_preserves_controlled_evidence(self):
        with TemporaryDirectory() as tmp:
            result = load_original_searchqa(self._write_archive(Path(tmp)))

        self.assertEqual({split: len(rows) for split, rows in result["splits"].items()}, {
            "train": 1,
            "validation": 1,
            "test": 1,
        })
        row = result["splits"]["train"][0]
        self.assertEqual(row["domain"], "search_qa")
        self.assertEqual(row["gold_answer"], "A")
        self.assertEqual(row["evidence"], ["Evidence supports A."])
        self.assertEqual(row["source"], "nyu-dl/SearchQA")
        self.assertEqual(row["source_key"], "train:0")
        self.assertEqual(row["answer_aliases"], ["A"])
        self.assertIn("air_date", row["source_metadata"])
        self.assertEqual(len(result["artifact_sha256"]), 64)

    def test_missing_split_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "searchqa.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train.json", json.dumps([self._row("A", "B")]))

            with self.assertRaisesRegex(ValueError, "validation.*test"):
                load_original_searchqa(archive)

    def test_missing_evidence_fails_without_silent_context_fallback(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "searchqa.zip"
            row = {"question": "A", "answer": "B", "search_results": []}
            with zipfile.ZipFile(archive, "w") as handle:
                for split in ("train", "validation", "test"):
                    handle.writestr(f"{split}.json", json.dumps([row]))

            with self.assertRaisesRegex(ValueError, "controlled evidence"):
                load_original_searchqa(archive)


if __name__ == "__main__":
    unittest.main()
