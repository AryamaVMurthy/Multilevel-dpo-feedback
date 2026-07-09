import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from text_feedback_dpo.observability import JsonlLogger


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


if __name__ == "__main__":
    unittest.main()
