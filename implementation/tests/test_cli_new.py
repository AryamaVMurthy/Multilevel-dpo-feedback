import unittest
import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from text_feedback_dpo.cli import build_parser
from text_feedback_dpo.prompts import prompt_builder_identity


class CLITest(unittest.TestCase):
    def test_prepare_searchqa_emits_exact_active_source_schema(self):
        class Tokenizer:
            @staticmethod
            def encode(text, *, add_special_tokens):
                self.assertFalse(add_special_tokens)
                return text.split()

        prepared_row = {
            "id": "train-0", "question": "Who?", "gold_answer": "Ada",
            "snippets": ["Ada evidence"],
            "sources": [{
                "source_id": "S001", "original_rank": 1, "title": "Ada",
                "url": "https://example.test/ada", "snippet": "Ada evidence",
                "related_links": ["https://example.test/related"],
            }],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.jsonl"
            args = build_parser().parse_args([
                "prepare-searchqa", "--source", "source", "--split", "train",
                "--tokenizer-model", "model", "--tokenizer-revision", "tok-rev",
                "--revision", "data-rev", "--output", str(output), "--max-evidence-tokens", "100",
            ])
            with patch("text_feedback_dpo.runtime.load_tokenizer", return_value=Tokenizer()), patch(
                "text_feedback_dpo.cli.load_searchqa_split_with_stats",
                return_value=([prepared_row], {"loaded": 1}),
            ):
                args.func(args)

            source = json.loads(output.read_text(encoding="utf-8"))["sources"][0]
            self.assertEqual(set(source), {"source_id", "original_rank", "title", "url", "snippet"})

    def test_prepare_searchqa_streams_official_rows_and_writes_manifest_after_success(self):
        class Tokenizer:
            @staticmethod
            def encode(text, *, add_special_tokens):
                del text
                self.assertFalse(add_special_tokens)
                return [1]

        prepared_row = {
            "id": "train-0", "question": "Who?", "gold_answer": "Ada",
            "snippets": ["Ada evidence"],
            "sources": [{
                "source_id": "S001", "original_rank": 1, "title": "Ada",
                "url": "https://example.test/ada", "snippet": "Ada evidence",
                "related_links": None,
            }],
        }
        stream_stats = {
            "source_rows": 1, "materialized_rows": 1, "dropped_rows": 0,
            "drop_reasons": {},
            "source_records": {
                "input_records": 1, "usable_records": 1, "dropped_records": 0,
                "drop_reasons": {},
            },
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.jsonl"
            args = build_parser().parse_args([
                "prepare-searchqa", "--source", "kyunghyuncho/search_qa", "--split", "train",
                "--tokenizer-model", "model", "--tokenizer-revision", "tok-rev",
                "--revision", "data-rev", "--output", str(output), "--max-evidence-tokens", "100",
            ])
            with patch("text_feedback_dpo.runtime.load_tokenizer", return_value=Tokenizer()), patch(
                "text_feedback_dpo.cli.stream_searchqa_split_with_stats",
                return_value=(iter([prepared_row]), stream_stats),
            ):
                args.func(args)

            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            manifest = json.loads((Path(directory) / "prepared.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["load_stats"]["materialized_rows"], 1)

    def test_exposes_only_searchqa_training_commands(self):
        parser = build_parser()
        for command in ("prepare-searchqa", "shard-jsonl", "merge-predictions", "probe-model", "collect", "build-preferences", "build-sft-data", "precompute-dpo-ref-log-probs", "generate", "evaluate", "preflight-quality", "select-thinking-mode", "report", "validate-run", "train-sft", "train-dpo", "train-grpo", "train-dapo"):
            parsed = parser.parse_args([command] + self._required_args(command))
            self.assertEqual(parsed.command, command)

    def test_generation_and_collection_expose_explicit_thinking_controls(self):
        parser = build_parser()
        generated = parser.parse_args(["generate"] + self._required_args("generate"))
        self.assertEqual(generated.student_thinking_mode, "direct")
        self.assertEqual(generated.scratchpad_max_new_tokens, 256)
        self.assertEqual(generated.max_new_tokens, 32)
        collected = parser.parse_args(["collect"] + self._required_args("collect"))
        self.assertEqual(collected.student_thinking_mode, "direct")
        self.assertTrue(collected.teacher_thinking)
        self.assertEqual(collected.query_max_new_tokens, 32)
        self.assertEqual(collected.response_max_new_tokens, 256)
        self.assertEqual(collected.teacher_max_new_tokens, 1024)
        self.assertEqual(collected.student_batch_size, 32)
        self.assertEqual(collected.teacher_batch_size, 8)
        self.assertEqual(collected.dataset_revision, "data-rev")
        self.assertEqual(collected.prompt_version, "fixed-retrieval-cited-v1")
        self.assertEqual(collected.sibling_count, 2)
        self.assertEqual(collected.sibling_seeds, [101, 102])
        self.assertEqual(collected.seed, 7)
        teacher_probe = parser.parse_args(["probe-model"] + self._required_args("probe-model"))
        self.assertEqual(teacher_probe.teacher_max_new_tokens, 1024)

    def test_teacher_probe_supplies_bounded_retrieved_context_to_private_prompt(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "teacher-probe.json"
            args = build_parser().parse_args([
                "probe-model", "--role", "teacher", "--model", "model", "--model-revision", "rev",
                "--teacher-quantization", "4bit", "--output", str(output),
            ])
            with patch("text_feedback_dpo.runtime.load_tokenizer", return_value=object()), patch(
                "text_feedback_dpo.runtime.load_teacher", return_value=object()
            ), patch(
                "text_feedback_dpo.runtime.render_teacher_prompts",
                side_effect=lambda _tokenizer, prompts, enable_thinking=True: prompts,
            ), patch(
                "text_feedback_dpo.runtime.generate_batch",
                return_value=['{"hint":"Focus on the associated writer."}'],
            ) as generate_batch:
                args.func(args)

            prompt = generate_batch.call_args.args[2][0]
            self.assertNotIn('"complete_source_records"', prompt)
            self.assertIn('"available_source_count"', prompt)
            self.assertIn('"retrieved_records"', prompt)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["role"], "teacher")
            self.assertEqual(result["hint"], "Focus on the associated writer.")

    def test_generate_searchqa_is_explicit_and_has_independent_batch_defaults(self):
        parser = build_parser()
        parsed = parser.parse_args(["generate-searchqa", "--data", "x.jsonl", "--output", "y.jsonl", "--model", "model", "--model-revision", "model-rev", "--dataset-source", "searchqa", "--dataset-revision", "data-rev", "--attention-implementation", "sdpa", "--policy-hash", "p1"])
        self.assertEqual(parsed.command, "generate-searchqa")
        self.assertEqual(parsed.query_batch_size, 4)
        self.assertEqual(parsed.response_batch_size, 4)
        self.assertEqual(parsed.query_max_new_tokens, 32)
        self.assertEqual(parsed.response_max_new_tokens, 256)
        self.assertEqual(parsed.top_k, 8)
        self.assertEqual(parsed.k1, 1.2)
        self.assertEqual(parsed.b, 0.75)
        self.assertEqual(parsed.context_budget, 4096)

    def test_evaluate_and_preflight_require_explicit_protocol(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["evaluate", "--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["preflight-quality", "--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json", "--samples", "s.jsonl", "--split-name", "train-dev"])

    def test_active_protocol_rejects_mixed_prediction_schema_and_duplicate_ids(self):
        from text_feedback_dpo.cli import _cmd_evaluate_active_search

        example = {"id": "1", "gold_answer": "Ada"}
        with self.assertRaisesRegex(ValueError, "raw_query"):
            _cmd_evaluate_active_search(object(), {"1": example}, [{"id": "1", "response": "Ada"}])

    def test_archival_evaluator_rejects_active_only_fields(self):
        from text_feedback_dpo.cli import cmd_evaluate

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            predictions = root / "predictions.jsonl"
            data.write_text(json.dumps({"id": "1", "gold_answer": "Ada", "packed_evidence": "Ada"}) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps({"id": "1", "response": "Ada", "raw_response": "active"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "active-search fields"):
                cmd_evaluate(Namespace(data=data, predictions=predictions, output=root / "out.json", protocol="archival"))

    def test_active_evaluator_aggregates_all_model_failure_categories(self):
        from text_feedback_dpo.cli import _cmd_evaluate_active_search
        from text_feedback_dpo.batch_generation import run_fixed_retrieval_pipeline
        from text_feedback_dpo.runtime import GeneratedText

        names = ("invalid", "query-truncated", "malformed", "response-truncated")
        examples = {
            name: {
                "id": name, "question": f"{name}?", "gold_answer": "response",
                "sources": [{"source_id": "S001", "original_rank": 1, "title": "Response source",
                             "url": "https://example.test/response", "snippet": "response evidence"}],
            }
            for name in names
        }
        predictions = run_fixed_retrieval_pipeline(
            list(examples.values()),
            query_generate_batch=lambda _prompts: [GeneratedText("two\nlines", False), GeneratedText("query", True), GeneratedText("query", False), GeneratedText("query", False)],
            response_generate_batch=lambda _prompts: [GeneratedText("bad", False), GeneratedText("Answer: response\nReasoning: Source [S001].\nSources: S001", True)],
            policy_hash="policy-v1",
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            _cmd_evaluate_active_search(Namespace(output=output), examples, predictions)
            summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["query_truncation_rate"], 0.25)
        self.assertEqual(summary["response_truncation_rate"], 0.25)
        self.assertEqual(summary["malformed_rate"], 1.0)
        self.assertEqual(summary["lexical_cited_answer_support_rate"], 0.0)

    def test_generate_searchqa_command_writes_scored_rendered_trajectory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [
                {
                    "source_id": f"S{index:03d}",
                    "original_rank": index,
                    "title": f"Ada source {index}",
                    "url": f"https://example.test/ada/{index}",
                    "snippet": f"Ada Lovelace source evidence number {index}.",
                }
                for index in range(1, 9)
            ]
            data = root / "data.jsonl"
            data.write_text(json.dumps({"id": "row-1", "question": "Who?", "gold_answer": "Ada Lovelace", "sources": sources}) + "\n", encoding="utf-8")
            output = root / "predictions.jsonl"
            args = build_parser().parse_args([
                "generate-searchqa", "--data", str(data), "--output", str(output), "--model", "model",
                "--model-revision", "model-rev", "--dataset-source", "searchqa",
                "--dataset-revision", "data-rev", "--attention-implementation", "sdpa", "--policy-hash", "policy-v1",
            ])

            def fake_records(_model, _tokenizer, prompts, **_kwargs):
                if prompts[0].endswith("Search query:"):
                    return [{"text": "Ada Lovelace author", "truncated": False}]
                return [{"text": "Answer: Ada Lovelace\nReasoning: Source identifies Ada Lovelace [S001].\nSources: S001", "truncated": False}]

            with patch("text_feedback_dpo.runtime.load_tokenizer", return_value=object()), patch("text_feedback_dpo.runtime.load_student", return_value=object()), patch("text_feedback_dpo.runtime.generate_batch_records", side_effect=fake_records):
                args.func(args)

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["id"], "row-1")
            self.assertEqual(row["retrieval_metrics"]["recall@8"], 1.0)
            self.assertEqual(row["ranked_search_results"][0]["requested_top_k"], 8)
            self.assertEqual(row["ranked_search_results"][0]["effective_top_k"], 8)
            self.assertEqual(row["ranked_search_results"][0]["source_count"], 8)
            self.assertTrue(row["cited_score"]["parse_valid"])
            self.assertIn("[S001] Ada source 1 — https://example.test/ada/1", row["rendered_visible_response"])
            self.assertFalse(row["truncation"]["query"])
            self.assertFalse(row["truncation"]["response"])

            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["max_length"], 4096)
            self.assertEqual(manifest["model"]["revision"], "model-rev")
            self.assertEqual(manifest["dataset"]["source"], "searchqa")
            self.assertEqual(manifest["dataset"]["revision"], "data-rev")
            self.assertEqual(manifest["source_schema"]["identity"], "searchqa.search_results.v1")
            self.assertEqual(manifest["retrieval"]["requested_top_k"], 8)
            self.assertEqual(manifest["prompt"]["builders"], prompt_builder_identity())
            self.assertNotIn("query_builder", manifest["prompt"])
            self.assertEqual(manifest["artifacts"][0]["path"], output.name)
            self.assertIn("sha256", manifest["artifacts"][0])
            validated = root / "validated.json"
            validate_args = build_parser().parse_args(["validate-run", "--directory", str(root), "--output", str(validated)])
            validate_args.func(validate_args)
            self.assertTrue(json.loads(validated.read_text(encoding="utf-8"))["valid"])

            evaluated = root / "evaluated.json"
            evaluate_args = build_parser().parse_args([
                "evaluate", "--data", str(data), "--predictions", str(output), "--output", str(evaluated), "--protocol", "active-search",
            ])
            evaluate_args.func(evaluate_args)
            summary = json.loads(evaluated.read_text(encoding="utf-8"))
            self.assertEqual(summary["exact_match"], 1.0)
            self.assertEqual(summary["retrieval_recall@8"], 1.0)
            self.assertEqual(summary["query_truncation_rate"], 0.0)
            self.assertEqual(summary["response_truncation_rate"], 0.0)

            preflight = root / "preflight.json"
            samples = root / "samples.jsonl"
            preflight_args = build_parser().parse_args([
                "preflight-quality", "--data", str(data), "--predictions", str(output),
                "--output", str(preflight), "--samples", str(samples), "--split-name", "train-dev",
                "--protocol", "active-search", "--sample-size", "1",
            ])
            preflight_args.func(preflight_args)
            preflight_summary = json.loads(preflight.read_text(encoding="utf-8"))
            sample = json.loads(samples.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(preflight_summary["protocol_exact_match"], 1.0)
            self.assertEqual(preflight_summary["retrieval_recall@8"], 1.0)
            self.assertIn("https://example.test/ada/1", sample["rendered_visible_response"])
            self.assertEqual(sample["ranked_search_results"], row["ranked_search_results"])

    def test_generate_searchqa_validates_every_source_before_model_or_tokenizer_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            malformed_source = {
                "source_id": "S001", "original_rank": 1, "title": "Ada",
                "url": "https://example.test/ada", "snippet": "Ada evidence",
                "unknown": "must fail",
            }
            valid_source = {key: value for key, value in malformed_source.items() if key != "unknown"}
            rows = [
                {"id": "row-1", "question": "Who?", "gold_answer": "Ada", "sources": [valid_source]},
                {"id": "row-2", "question": "Who else?", "gold_answer": "Ada", "sources": [malformed_source]},
            ]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = build_parser().parse_args([
                "generate-searchqa", "--data", str(data), "--output", str(root / "out.jsonl"),
                "--model", "model", "--model-revision", "model-rev", "--dataset-source", "searchqa",
                "--dataset-revision", "data-rev", "--attention-implementation", "sdpa", "--policy-hash", "policy-v1",
            ])
            with patch("text_feedback_dpo.runtime.load_tokenizer") as load_tokenizer, patch(
                "text_feedback_dpo.runtime.load_student"
            ) as load_student, self.assertRaisesRegex(ValueError, "unknown fields"):
                args.func(args)
            load_tokenizer.assert_not_called()
            load_student.assert_not_called()

    def test_generate_searchqa_validates_all_row_fields_before_runtime_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "source_id": "S001", "original_rank": 1, "title": "Ada",
                "url": "https://example.test/ada", "snippet": "Ada evidence",
            }
            rows = [
                {"id": "row-1", "question": "Who?", "gold_answer": "Ada", "sources": [source]},
                {"id": "row-2", "question": " ", "gold_answer": "Ada", "sources": [source]},
            ]
            data = root / "data.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = build_parser().parse_args([
                "generate-searchqa", "--data", str(data), "--output", str(root / "out.jsonl"),
                "--model", "model", "--model-revision", "model-rev", "--dataset-source", "searchqa",
                "--dataset-revision", "data-rev", "--attention-implementation", "sdpa", "--policy-hash", "policy-v1",
            ])
            with patch("text_feedback_dpo.runtime.load_tokenizer") as load_tokenizer, patch(
                "text_feedback_dpo.runtime.load_student"
            ) as load_student, self.assertRaisesRegex(ValueError, "non-empty question"):
                args.func(args)
            load_tokenizer.assert_not_called()
            load_student.assert_not_called()

    @staticmethod
    def _required_args(command):
        if command == "prepare-searchqa":
            return ["--source", "source", "--split", "train", "--tokenizer-model", "model", "--tokenizer-revision", "tok-rev", "--revision", "data-rev", "--output", "x.jsonl", "--max-evidence-tokens", "100"]
        if command == "probe-model":
            return ["--role", "teacher", "--model", "model", "--model-revision", "rev", "--teacher-quantization", "4bit", "--output", "probe.json"]
        if command == "shard-jsonl":
            return ["--input", "data.jsonl", "--output-dir", "shards", "--shards", "4"]
        if command == "merge-predictions":
            return ["--shard-dir", "shards", "--output", "merged.jsonl", "--shards", "4"]
        if command == "collect":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--student-model", "Qwen/Qwen3-4B-Base", "--teacher-model", "Qwen/Qwen3-32B", "--student-revision", "student-rev", "--teacher-revision", "teacher-rev", "--dataset-revision", "data-rev", "--prompt-version", "fixed-retrieval-cited-v1", "--policy-version", "sft-v1", "--seed", "7", "--teacher-quantization", "4bit", "--attention-implementation", "sdpa", "--student-device", "cuda:1", "--teacher-device", "cuda:0", "--trajectory-cache", "cache.jsonl", "--policy-hash", "p1", "--sibling-count", "2", "--sibling-seeds", "101", "102"]
        if command == "build-preferences":
            return ["--data", "data.jsonl", "--trajectories", "x.jsonl", "--output", "y.jsonl"]
        if command == "build-sft-data":
            return ["--config", "config.yaml", "--data", "x.jsonl", "--trajectories", "t.jsonl", "--output", "y.jsonl", "--report", "report.json", "--min-coverage", "0.5", "--min-rows", "2"]
        if command == "precompute-dpo-ref-log-probs":
            return ["--config", "config.yaml", "--data", "x.jsonl", "--output", "refs.jsonl", "--reference-checkpoint-hash", "a" * 64, "--prompt-context-schema", "schema.json", "--attention-implementation", "sdpa", "--device", "cuda:0"]
        if command == "evaluate":
            return ["--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json", "--protocol", "archival"]
        if command == "preflight-quality":
            return ["--data", "x.jsonl", "--predictions", "y.jsonl", "--output", "z.json", "--samples", "samples.jsonl", "--split-name", "train-dev", "--protocol", "archival"]
        if command == "select-thinking-mode":
            return ["--direct", "direct.json", "--two-pass", "two.json", "--output", "choice.json"]
        if command == "generate":
            return ["--data", "x.jsonl", "--output", "y.jsonl", "--model", "model", "--attention-implementation", "sdpa"]
        if command == "report":
            return ["--metrics", "x.json", "--output", "y.html"]
        if command == "validate-run":
            return ["--directory", "run", "--output", "y.json"]
        args = [
            "--config", "config.yaml", "--train", "train.jsonl", "--eval", "eval.jsonl", "--output", "out",
            "--max-steps", "1", "--max-length", "4096", "--per-device-train-batch-size", "1",
            "--per-device-eval-batch-size", "4", "--dataloader-num-workers", "2",
            "--gradient-accumulation-steps", "8", "--attention-implementation", "sdpa",
        ]
        if command == "train-dpo":
            args += ["--ref-log-probs", "train.refs.jsonl", "--eval-ref-log-probs", "eval.refs.jsonl", "--reference-checkpoint-hash", "a" * 64, "--prompt-context-schema", "schema.json"]
        return args


if __name__ == "__main__":
    unittest.main()
