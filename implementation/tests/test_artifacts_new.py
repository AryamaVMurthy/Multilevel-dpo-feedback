import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.artifacts import validate_artifacts
from text_feedback_dpo.dataset import build_sft_rows
from text_feedback_dpo.preferences import build_preference_rows


class ArtifactTest(unittest.TestCase):
    @staticmethod
    def _identity_hash(value):
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_legacy_gold_answer_sft_builder_is_removed(self):
        with self.assertRaisesRegex(RuntimeError, "removed unsafe SFT path"):
            build_sft_rows([
                {"id": "1", "question": "Who?", "gold_answer": "Ada", "packed_evidence": "Ada evidence"}
            ])

    def test_active_preference_builder_rejects_archival_plain_answer_schema(self):
        with self.assertRaisesRegex(ValueError, "active trajectory"):
            build_preference_rows({
                "id": "q1", "resolved": True, "prompt": "Question: Who?\n\nAnswer:",
                "chosen": "Ada", "attempts": [{"attempt_index": 0, "response": "Grace", "correct": False}],
            })

    def test_validate_artifacts_fails_when_required_manifest_is_missing(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_artifacts(Path(tmp))

    def test_active_manifest_requires_complete_identities_and_matching_artifact_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "predictions.jsonl"
            artifact.write_text('{"id":"1"}\n', encoding="utf-8")
            source_identity = {"identity": "searchqa.search_results.v1", "version": 1}
            prompt_identity = {"identity": "fixed-retrieval-cited-v1"}
            response_identity = {"identity": "cited-response", "schema_version": 1}
            manifest = {
                "command": "generate-searchqa", "max_length": 4096, "rows": 1,
                "model": {"identity": "model", "revision": "rev", "policy_hash": "policy"},
                "dataset": {"source": "searchqa", "revision": "data-rev", "sha256": "a" * 64},
                "source_schema": {**source_identity, "sha256": self._identity_hash(source_identity)},
                "retrieval": {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75},
                "prompt": {**prompt_identity, "sha256": self._identity_hash(prompt_identity)},
                "response": {**response_identity, "sha256": self._identity_hash(response_identity)},
                "generation": {"context_budget": 4096, "query_max_new_tokens": 32, "response_max_new_tokens": 256},
                "timing": {"pipeline_wall_ms": 1.0},
                "required_files": [artifact.name],
                "artifacts": [{"path": artifact.name, "format": "jsonl", "rows": 1, "bytes": artifact.stat().st_size,
                               "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(validate_artifacts(root)["valid"])
            manifest["artifacts"][0]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                validate_artifacts(root)
            manifest["artifacts"][0]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest["model"]["revision"] = ""
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model.revision"):
                validate_artifacts(root)
            manifest["model"]["revision"] = "rev"
            manifest["source_schema"]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_schema identity hash"):
                validate_artifacts(root)
            manifest["source_schema"]["sha256"] = self._identity_hash(source_identity)
            manifest["rows"] = 999
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest rows.*artifact row count"):
                validate_artifacts(root)

    def test_active_manifest_rejects_unknown_format_and_self_consistent_wrong_protocol_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "predictions.jsonl"
            artifact.write_text('{"id":"1"}\n', encoding="utf-8")
            source_identity = {"identity": "searchqa.search_results.v1", "version": 1}
            prompt_identity = {"identity": "fixed-retrieval-cited-v1"}
            response_identity = {"identity": "cited-response", "schema_version": 1}
            manifest = {
                "command": "generate-searchqa", "max_length": 4096, "rows": 1,
                "model": {"identity": "model", "revision": "rev", "policy_hash": "policy"},
                "dataset": {"source": "searchqa", "revision": "data-rev", "sha256": "a" * 64},
                "source_schema": {**source_identity, "sha256": self._identity_hash(source_identity)},
                "retrieval": {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75},
                "prompt": {**prompt_identity, "sha256": self._identity_hash(prompt_identity)},
                "response": {**response_identity, "sha256": self._identity_hash(response_identity)},
                "generation": {"context_budget": 4096, "query_max_new_tokens": 32, "response_max_new_tokens": 256},
                "timing": {"pipeline_wall_ms": 1.0}, "required_files": [artifact.name],
                "artifacts": [{"path": artifact.name, "format": "jsonl", "rows": 1, "bytes": artifact.stat().st_size,
                               "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
            }
            for section, identity in (
                ("prompt", {"identity": "self-consistent-but-wrong"}),
                ("response", {"identity": "wrong-response", "schema_version": 1}),
            ):
                with self.subTest(section=section):
                    changed = json.loads(json.dumps(manifest))
                    changed[section] = {**identity, "sha256": self._identity_hash(identity)}
                    (root / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"{section} identity"):
                        validate_artifacts(root)

            changed = json.loads(json.dumps(manifest))
            changed["artifacts"][0]["format"] = "text"
            changed["artifacts"][0]["rows"] = 999
            (root / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supported format"):
                validate_artifacts(root)

    def test_active_manifest_parses_jsonl_objects_and_rejects_blank_or_duplicate_ids(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "predictions.jsonl"
            source_identity = {"identity": "searchqa.search_results.v1", "version": 1}
            prompt_identity = {"identity": "fixed-retrieval-cited-v1"}
            response_identity = {"identity": "cited-response", "schema_version": 1}

            def write_case(text, rows):
                artifact.write_text(text, encoding="utf-8")
                manifest = {
                    "command": "generate-searchqa", "max_length": 4096, "rows": rows,
                    "model": {"identity": "model", "revision": "rev", "policy_hash": "a" * 64},
                    "dataset": {"source": "searchqa", "revision": "data-rev", "sha256": "b" * 64},
                    "source_schema": {**source_identity, "sha256": self._identity_hash(source_identity)},
                    "retrieval": {"identity": "fixed_bm25", "schema_version": 1, "requested_top_k": 8, "k1": 1.2, "b": 0.75},
                    "prompt": {**prompt_identity, "sha256": self._identity_hash(prompt_identity)},
                    "response": {**response_identity, "sha256": self._identity_hash(response_identity)},
                    "generation": {"context_budget": 4096, "query_max_new_tokens": 32, "response_max_new_tokens": 256},
                    "timing": {"pipeline_wall_ms": 1.0}, "required_files": [artifact.name],
                    "artifacts": [{"path": artifact.name, "format": "jsonl", "rows": rows,
                                   "bytes": artifact.stat().st_size,
                                   "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
                }
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            for text, rows, message in (
                ('{"id":"1"}\n\n', 1, "blank"),
                ('{"id":"1"\n', 1, "invalid JSON"),
                ('[1, 2]\n', 1, "JSON object"),
                ('{"id":"1"}\n{"id":"1"}\n', 2, "duplicate.*id"),
            ):
                with self.subTest(text=text):
                    write_case(text, rows)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_artifacts(root)

    def test_collect_manifest_parses_every_required_jsonl_including_cache_artifacts(self):
        for filename, text, message in (
            ("trajectories.jsonl", '{"id":"1"}\n\n', "blank"),
            ("trajectory-cache.jsonl", '{"id":"1"\n', "invalid JSON"),
            ("trajectory-cache.jsonl", '[{"id":"1"}]\n', "JSON object"),
        ):
            with self.subTest(filename=filename, message=message), TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifact = root / filename
                artifact.write_text(text, encoding="utf-8")
                (root / "manifest.json").write_text(json.dumps({
                    "command": "collect", "max_length": 4096,
                    "required_files": [filename],
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifacts(root)


if __name__ == "__main__":
    unittest.main()
