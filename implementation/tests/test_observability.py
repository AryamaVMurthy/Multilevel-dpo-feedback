import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.observability import JsonlLogger, validate_run_manifest, write_run_manifest


class ObservabilityTest(unittest.TestCase):
    def test_event_schema_includes_timestamp_status_and_elapsed_ms(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = JsonlLogger(path, run_id="run-1")
            logger.event("example_evaluated", stage="evaluate")

            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["event_name"], "example_evaluated")
            self.assertEqual(event["run_id"], "run-1")
            self.assertEqual(event["status"], "ok")
            self.assertIn("timestamp", event)
            self.assertIn("elapsed_ms", event)
            self.assertEqual(event["stage"], "evaluate")

    def test_failure_event_includes_required_error_context(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = JsonlLogger(path, run_id="run-1")
            logger.failure(stage="parse", error_code="bad_tag", message="missing reflect")

            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["event_name"], "failure")
            self.assertEqual(event["status"], "error")
            self.assertEqual(event["stage"], "parse")
            self.assertEqual(event["error_code"], "bad_tag")
            self.assertEqual(event["message"], "missing reflect")

    def test_run_manifest_requires_paper_observability_fields(self):
        required = {
            "git_commit": "abc",
            "config_hash": "config",
            "dataset_manifest_hash": "dataset",
            "seed": 17,
            "source_revision": "source",
            "model_revisions": {"student": "student", "teacher": "teacher"},
            "package_versions": {"torch": "x"},
            "slurm": {"job_id": "1"},
            "gpu_telemetry": {"peak_memory_bytes": 10},
            "token_counts": {"input": 1, "output": 2},
            "latency_ms": {"mean": 1},
            "throughput": {"examples_per_second": 1},
            "peak_memory_bytes": 10,
            "pair_metrics": {"pairs": 2},
            "evaluator_metrics": {"confidence": 0.9},
            "training_metrics": {"loss": 0.1},
            "architecture": {"coverage_hash": "coverage"},
            "optimizer": {"name": "adamw_torch_fused"},
            "candidate_id": "candidate",
            "promotion_stage": 1,
            "selection_evidence": {"metric": 0.8},
            "search_ledger_hash": "ledger",
            "failure_ledger": [],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            write_run_manifest(path, required)
            loaded = validate_run_manifest(path)
            self.assertEqual(loaded["git_commit"], "abc")
            with self.assertRaisesRegex(ValueError, "training_metrics"):
                broken = dict(required)
                broken.pop("training_metrics")
                write_run_manifest(path, broken)


if __name__ == "__main__":
    unittest.main()
