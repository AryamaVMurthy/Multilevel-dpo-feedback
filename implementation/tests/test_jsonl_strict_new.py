import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.cli import iter_jsonl as iter_cli_jsonl
from text_feedback_dpo.io import read_jsonl
from text_feedback_dpo.training import (
    build_reference_manifest,
    dataset_identity_hash,
    load_precomputed_reference_log_probs,
    write_precomputed_reference_log_probs,
)


class StrictJSONLTest(unittest.TestCase):
    def test_readers_preserve_single_trailing_newline_but_reject_blank_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id":"one"}\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"id": "one"}])
            self.assertEqual(list(iter_cli_jsonl(path)), [{"id": "one"}])

            path.write_text('{"id":"one"}\n  \t\n{"id":"two"}\n', encoding="utf-8")
            for reader in (read_jsonl, lambda value: list(iter_cli_jsonl(value))):
                with self.subTest(reader=reader), self.assertRaisesRegex(
                    ValueError, rf"blank JSONL record.*{path}:2"
                ):
                    reader(path)

    def test_reference_artifact_rejects_blank_record_with_path_and_line(self):
        row = {"id": "r1", "prompt": "p", "chosen": " c", "rejected": " r"}
        manifest = build_reference_manifest(
            model="Qwen/Qwen3-4B-Base",
            model_revision="model-rev",
            reference_checkpoint_hash="a" * 64,
            tokenizer="Qwen/Qwen3-4B-Base",
            tokenizer_revision="tok-rev",
            data_hash=dataset_identity_hash([row]),
            prompt_context_schema={"schema": 1},
            max_length=4096,
            trl_version="0.0.test",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "refs.jsonl"
            write_precomputed_reference_log_probs(
                path,
                [{**row, "ref_chosen_logps": -1.0, "ref_rejected_logps": -2.0}],
                manifest,
            )
            path.write_text(path.read_text(encoding="utf-8") + " \t\n", encoding="utf-8")
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_manifest["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(stored_manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, rf"blank JSONL record.*{path}:2"):
                load_precomputed_reference_log_probs(path, manifest)


if __name__ == "__main__":
    unittest.main()
