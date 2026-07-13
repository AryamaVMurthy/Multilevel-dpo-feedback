import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROBE_RUNNER = SCRIPTS / "turing_probe_runner.py"


def valid_probe_result(*, throughput: float = 10.0, compile_: bool = False) -> dict:
    token_ids = [[1, 2, 3]]
    decoded_outputs = ["decoded"]
    return {
        "schema_version": 1,
        "probe_name": "probe",
        "status": "ok",
        "fallback_reason": "none",
        "output_hash": hashlib.sha256(json.dumps(token_ids, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "decoded_output_hash": hashlib.sha256(json.dumps(decoded_outputs, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "output_token_ids": token_ids,
        "decoded_outputs": decoded_outputs,
        "examples_per_second": throughput / 2,
        "tokens_per_second": throughput,
        "peak_gpu_memory_mb": 1024.0,
        "gpu_utilization": {
            "sample_count": 3,
            "utilization_mean_percent": 70.0,
            "utilization_peak_percent": 90.0,
            "nvidia_smi_peak_memory_mb": 1100.0,
            "power_peak_watts": 200.0,
            "temperature_peak_c": 70.0,
        },
        "gpu_hardware": {
            "count": 1,
            "devices": [{
                "index": 0,
                "name": "nvidia rtx 6000 ada generation",
                "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                "total_memory_bytes": 51527024640,
                "compute_capability": "8.9",
            }],
        },
        "package_versions": {
            "torch": "2.13.0+cu126",
            "transformers": "5.13.0",
            "trl": "1.8.0",
            "deepspeed": "0.19.2",
            "bitsandbytes": "0.49.0",
            "flash-attn": "missing",
            "liger-kernel": "missing",
        },
        "config": {
            "probe_kind": "generation",
            "attention_implementation": "sdpa",
            "generation_batch_size": 4,
            "static_cache": False,
            "compile": compile_,
            "train_microbatch": 1,
            "gradient_accumulation_steps": 32,
            "dataloader_workers": 0,
            "packing": False,
            "padding_free": False,
            "use_liger_kernel": False,
        },
        "identities": {
            "commit_hash": "commit",
            "config_sha256": "config",
            "model": "model",
            "model_revision": "revision",
            "dataset_source": "source",
            "dataset_revision": "dataset-revision",
            "dataset_sha256": "dataset",
            "prompt_sha256": "prompt",
            "retrieval_sha256": "retrieval",
            "source_schema_sha256": "schema",
        },
    }


def freeze_probe(tmp_path: Path, baseline_data: dict, candidate_data: dict | None = None):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(baseline_data), encoding="utf-8")
    command = [
        sys.executable, str(PROBE_RUNNER), "freeze-decision", "--baseline", str(baseline),
        "--output", str(tmp_path / "decision.json"), "--query-max-new-tokens", "32",
        "--response-max-new-tokens", "256", "--student-thinking-mode", "direct",
        "--scratchpad-max-new-tokens", "128", "--query-temperature", "0",
        "--response-temperature", "0", "--top-p", "1", "--top-k", "8",
        "--k1", "1.2", "--b", "0.75",
    ]
    if candidate_data is not None:
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps(candidate_data), encoding="utf-8")
        command.extend(("--candidate", str(candidate)))
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def write_realistic_checkpoint(path: Path, step: int) -> None:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step + 1, "log_history": [{"step": step, "loss": 1.0}]}),
        encoding="utf-8",
    )
    torch.save({"model.embed_tokens.weight": torch.ones(4, 4)}, path / "pytorch_model.bin")
    torch.save({
        "state": {0: {"step": torch.tensor(step), "exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}},
        "param_groups": [{"params": [0], "lr": 1e-6, "step": step}],
    }, path / "optimizer.pt")
    torch.save({"last_epoch": step, "_step_count": step + 1}, path / "scheduler.pt")
    torch.save({
        "python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.arange(16, dtype=torch.uint8),
    }, path / "rng_state.pth")


def write_identity_decision(path: Path, identities: dict[str, str]) -> str:
    write_decision(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identities"] = identities
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_smoke_manifest(tmp_path: Path, name: str, identities: dict[str, str], decision_sha: str, method: str) -> tuple[Path, str]:
    initial = tmp_path / f"{name}-initial" / "checkpoint-1"
    resumed = tmp_path / f"{name}-resumed" / "checkpoint-2"
    write_realistic_checkpoint(initial, 1)
    write_realistic_checkpoint(resumed, 2)
    manifest = tmp_path / f"{name}-smoke.json"
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(manifest), "--commit-hash", identities["commit_hash"],
         "--config-sha256", identities["config_sha256"], "--model", identities["model"],
         "--model-revision", identities["model_revision"], "--dataset-source", identities["dataset_source"],
         "--dataset-revision", identities["dataset_revision"], "--dataset-sha256", identities["dataset_sha256"],
         "--eval-dataset-sha256", identities.get("eval_dataset_sha256", "not-applicable"),
         "--prompt-sha256", identities["prompt_sha256"], "--retrieval-sha256", identities["retrieval_sha256"],
         "--source-schema-sha256", identities["source_schema_sha256"], "--optimization-decision-sha256", decision_sha,
         "--method", method],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def run_script(name: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / name)],
        cwd=ROOT,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        check=False,
    )


def required_generate_env(tmp_path: Path) -> dict[str, str]:
    data = tmp_path / "shard.jsonl"
    data.write_text('{"id":"one","question":"q","search_results":[]}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    return {
        "TURING_ACCOUNT": "test",
        "PROJECT_DIR": str(ROOT),
        "DATA": str(data),
        "OUTPUT": str(tmp_path / "predictions.jsonl"),
        "MODEL": "model",
        "MODEL_REVISION": "revision",
        "DATASET_SOURCE": "kyunghyuncho/search_qa",
        "DATASET_REVISION": "dataset-revision",
        "CONFIG": str(config),
        "PROMPT_VERSION": "fixed-retrieval-cited-v1",
        "PROMPT_HASH": "prompt",
        "RETRIEVAL_HASH": "retrieval",
        "SOURCE_SCHEMA_HASH": "schema",
        "POLICY_HASH": "policy",
        "STUDENT_THINKING_MODE": "direct",
        "SCRATCHPAD_MAX_NEW_TOKENS": "128",
        "QUERY_TEMPERATURE": "0.0",
        "RESPONSE_TEMPERATURE": "0.0",
        "TOP_P": "1.0",
        "TOP_K": "8",
        "BM25_K1": "1.2",
        "BM25_B": "0.75",
        "SHARD_INDEX": "0",
        "SHARD_COUNT": "1",
        "SHARD_INPUT_SHA256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "MERGE_ID": "eval-shards-v1",
        "SLURM_JOB_ID": "123",
        "SLURM_NNODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1",
    }


def write_decision(path: Path) -> str:
    decision = {
        "schema_version": 1,
        "status": "frozen",
        "fallback_reason": "sdpa_baseline_selected",
        "selected": {
            "attention_implementation": "sdpa",
            "generation": {
                "query_batch_size": 4,
                "response_batch_size": 4,
                "query_max_new_tokens": 32,
                "response_max_new_tokens": 256,
                "static_cache": False,
                "compile": False,
                "student_thinking_mode": "direct",
                "scratchpad_max_new_tokens": 128,
                "query_temperature": 0.0,
                "response_temperature": 0.0,
                "top_p": 1.0,
                "top_k": 8,
                "k1": 1.2,
                "b": 0.75,
            },
            "training": {
                "microbatch": 1,
                "gradient_accumulation_steps": 32,
                "dataloader_workers": 0,
                "packing": False,
                "padding_free": False,
                "use_liger_kernel": False,
            },
        },
        "identities": {
            "commit_hash": "commit",
            "config_sha256": "config",
            "model": "model",
            "model_revision": "revision",
            "dataset_source": "kyunghyuncho/search_qa",
            "dataset_revision": "dataset-revision",
            "dataset_sha256": "dataset",
            "prompt_sha256": "prompt",
            "retrieval_sha256": "retrieval",
            "source_schema_sha256": "schema",
        },
        "baseline_result_sha256": "a" * 64,
        "selected_result_sha256": "a" * 64,
    }
    path.write_text(json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_rejects_two_nodes_before_launch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SLURM_NNODES"] = "2"
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "requires exactly one node" in result.stderr
    assert "multi_node_generation_forbidden" in result.stderr


def test_generation_rejects_non_exact_single_gpu(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SLURM_GPUS_ON_NODE"] = "2"
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "requires exactly one allocated GPU" in result.stderr


def test_generation_rejects_shard_input_identity_mismatch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env["SHARD_INPUT_SHA256"] = "0" * 64
    result = run_script("turing_generate.sh", env)
    assert result.returncode == 2
    assert "SHARD_INPUT_SHA256" in result.stderr
    assert "shard_input_identity_mismatch" in result.stderr


def test_probe_runner_validates_frozen_decision_hash_and_emits_generation_contract(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    digest = write_decision(decision)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "validate-decision",
            "--decision",
            str(decision),
            "--expected-sha256",
            digest,
            "--purpose",
            "generation",
            "--commit-hash", "commit",
            "--config-sha256", "config",
            "--model", "model",
            "--model-revision", "revision",
            "--dataset-source", "kyunghyuncho/search_qa",
            "--dataset-revision", "dataset-revision",
            "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt",
            "--retrieval-sha256", "retrieval",
            "--source-schema-sha256", "schema",
            "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128",
            "--query-temperature", "0", "--response-temperature", "0", "--top-p", "1",
            "--top-k", "8", "--k1", "1.2", "--b", "0.75",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract["status"] == "validated"
    assert contract["decision_sha256"] == digest
    assert contract["attention_implementation"] == "sdpa"
    assert contract["query_batch_size"] == 4
    assert contract["fallback_reason"] == "sdpa_baseline_selected"


def test_probe_runner_rejects_cross_model_or_dataset_decision_reuse(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    digest = write_decision(decision)
    common = [
        sys.executable, str(PROBE_RUNNER), "validate-decision", "--decision", str(decision),
        "--expected-sha256", digest, "--purpose", "generation", "--commit-hash", "commit", "--config-sha256", "config",
        "--model-revision", "revision", "--dataset-source", "kyunghyuncho/search_qa",
        "--dataset-revision", "dataset-revision", "--prompt-sha256", "prompt",
        "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128",
        "--query-temperature", "0", "--response-temperature", "0", "--top-p", "1",
        "--top-k", "8", "--k1", "1.2", "--b", "0.75",
    ]
    for mismatched_args in (("--model", "other-model", "--dataset-sha256", "dataset"), ("--model", "model", "--dataset-sha256", "other-dataset")):
        result = subprocess.run([*common, *mismatched_args], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 2
        assert json.loads(result.stderr)["fallback_reason"] == "decision_identity_mismatch"


def test_freeze_excludes_faster_candidate_that_launch_validator_rejects(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    unsupported = tmp_path / "unsupported.json"
    baseline.write_text(json.dumps(valid_probe_result()), encoding="utf-8")
    unsupported.write_text(json.dumps(valid_probe_result(throughput=100.0, compile_=True)), encoding="utf-8")
    decision = tmp_path / "decision.json"
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "freeze-decision", "--baseline", str(baseline), "--candidate", str(unsupported),
         "--output", str(decision), "--query-max-new-tokens", "32", "--response-max-new-tokens", "256",
         "--student-thinking-mode", "direct", "--scratchpad-max-new-tokens", "128", "--query-temperature", "0",
         "--response-temperature", "0", "--top-p", "1", "--top-k", "8", "--k1", "1.2", "--b", "0.75"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    frozen = json.loads(decision.read_text(encoding="utf-8"))
    assert frozen["selected_result"] == str(baseline.resolve())
    assert frozen["rejected_candidates"][0]["fallback_reason"] == "launch_unsupported_compile"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_hash", None),
        ("decoded_output_hash", "fake"),
        ("examples_per_second", 0),
        ("tokens_per_second", float("nan")),
        ("peak_gpu_memory_mb", -1),
        ("gpu_utilization", {}),
        ("package_versions", {}),
        ("config", {}),
        ("identities", {}),
        ("output_token_ids", [[9, 9]]),
        ("gpu_hardware", {}),
        ("gpu_utilization", {
            "sample_count": 1, "utilization_mean_percent": 0.0, "utilization_peak_percent": 1.0,
            "nvidia_smi_peak_memory_mb": 1.0, "power_peak_watts": 1.0, "temperature_peak_c": 1.0,
        }),
    ],
)
def test_freeze_rejects_incomplete_or_fake_baseline_probe_artifacts(tmp_path: Path, field: str, value):
    baseline = valid_probe_result()
    if value is None:
        baseline.pop(field)
    else:
        baseline[field] = value
    result = freeze_probe(tmp_path, baseline)
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "baseline_probe_invalid"


@pytest.mark.parametrize("mismatch", ("identities", "package_versions", "config_sha256", "gpu_hardware"))
def test_freeze_never_accepts_candidate_with_baseline_parity_mismatch(tmp_path: Path, mismatch: str):
    baseline = valid_probe_result()
    candidate = valid_probe_result(throughput=20.0)
    if mismatch == "identities":
        candidate["identities"]["model"] = "other-model"
    elif mismatch == "package_versions":
        candidate["package_versions"]["torch"] = "other-version"
    else:
        if mismatch == "config_sha256":
            candidate["identities"]["config_sha256"] = "other-config"
        else:
            candidate["gpu_hardware"]["devices"][0]["uuid"] = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    result = freeze_probe(tmp_path, baseline, candidate)
    assert result.returncode == 0, result.stderr
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["selected_result"] == str((tmp_path / "baseline.json").resolve())
    assert "parity_mismatch" in decision["rejected_candidates"][0]["fallback_reason"]


def test_generation_missing_explicit_thinking_mode_fails_before_launch(tmp_path: Path):
    env = required_generate_env(tmp_path)
    env.pop("STUDENT_THINKING_MODE")
    result = run_script("turing_generate.sh", env)
    assert result.returncode != 0
    assert "STUDENT_THINKING_MODE" in result.stderr


def test_probe_runner_rejects_decision_hash_mismatch(tmp_path: Path):
    decision = tmp_path / "optimization-decision.json"
    write_decision(decision)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "validate-decision",
            "--decision",
            str(decision),
            "--expected-sha256",
            "0" * 64,
                "--purpose",
                "training",
                "--commit-hash", "commit",
            "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
            "--dataset-source", "kyunghyuncho/search_qa", "--dataset-revision", "dataset-revision",
            "--dataset-sha256", "dataset", "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval",
            "--source-schema-sha256", "schema",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    error = json.loads(result.stderr)
    assert error["status"] == "rejected"
    assert error["fallback_reason"] == "decision_hash_mismatch"


def test_fake_step_only_checkpoints_are_rejected(tmp_path: Path):
    initial = tmp_path / "checkpoint-10"
    resumed = tmp_path / "checkpoint-20"
    initial.mkdir()
    resumed.mkdir()
    (initial / "trainer_state.json").write_text('{"global_step":10}\n', encoding="utf-8")
    (resumed / "trainer_state.json").write_text('{"global_step":20}\n', encoding="utf-8")
    output = tmp_path / "smoke-manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "create-smoke-manifest",
            "--initial-checkpoint",
            str(initial),
            "--resumed-checkpoint",
            str(resumed),
            "--output", str(output), "--commit-hash", "commit", "--config-sha256", "config",
            "--model", "model", "--model-revision", "revision", "--dataset-source", "source",
            "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
            "--optimization-decision-sha256", "decision", "--method", "dpo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "checkpoint_model_state_missing"


def test_checkpoint_smoke_script_has_real_bounded_save_resume_contract():
    text = (SCRIPTS / "turing_checkpoint_smoke.sh").read_text(encoding="utf-8")
    assert "--max-steps" in text
    assert "--dataloader-workers" in text
    assert "--per-device-train-batch-size" in text
    assert "task7_checkpoint_smoke_cli_missing" in text
    assert "fallback_reason" in text
    assert text.count("torch.distributed.run") >= 2
    assert "--resume-from-checkpoint" in text
    assert "create-smoke-manifest" in text


def test_smoke_manifest_requires_checkpoint_lineage_and_matching_identity(tmp_path: Path):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    for checkpoint, step in ((initial, 1), (resumed, 2)):
        write_realistic_checkpoint(checkpoint, step)
    manifest = tmp_path / "smoke.json"
    identities = [
        "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
        "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "dataset-revision",
        "--dataset-sha256", "dataset", "--prompt-sha256", "prompt",
        "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--optimization-decision-sha256", "decision", "--method", "dpo",
    ]
    create = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(manifest), *identities],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert create.returncode == 0, create.stderr
    created_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert created_manifest["package_versions"]["torch"]
    assert len(created_manifest["training_metrics_lineage_sha256"]) == 64
    assert len(created_manifest["initial_checkpoint"]["training_metrics_sha256"]) == 64
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    valid = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", digest, *identities], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert valid.returncode == 0, valid.stderr
    wrong_identities = identities.copy()
    wrong_identities[wrong_identities.index("config")]= "other-config"
    invalid = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", digest, *wrong_identities], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert invalid.returncode == 2
    assert json.loads(invalid.stderr)["fallback_reason"] == "smoke_identity_mismatch"


def test_smoke_validation_rejects_package_identity_tampering(tmp_path: Path):
    identities = {
        "commit_hash": "commit", "config_sha256": "config", "model": "model", "model_revision": "revision",
        "dataset_source": "source", "dataset_revision": "dataset-revision", "dataset_sha256": "dataset",
        "prompt_sha256": "prompt", "retrieval_sha256": "retrieval", "source_schema_sha256": "schema",
    }
    manifest, _ = write_smoke_manifest(tmp_path, "package", identities, "decision", "dpo")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["package_versions"]["torch"] = "forged"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "validate-checkpoints", "--smoke-manifest", str(manifest),
         "--expected-sha256", hashlib.sha256(manifest.read_bytes()).hexdigest(), "--commit-hash", "commit",
         "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
         "--dataset-source", "source", "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
         "--eval-dataset-sha256", "not-applicable",
         "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
         "--optimization-decision-sha256", "decision", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "smoke_package_identity_mismatch"


def test_checkpoint_gate_rejects_large_padded_arbitrary_mappings(tmp_path: Path):
    for root, step in ((tmp_path / "checkpoint-1", 1), (tmp_path / "checkpoint-2", 2)):
        root.mkdir()
        (root / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": step + 1, "log_history": [{"step": step, "loss": 1.0}]}),
            encoding="utf-8",
        )
        torch.save({"model.embed_tokens.weight": "x" * 8192}, root / "pytorch_model.bin")
        torch.save({"state": {0: {"padding": "x" * 8192}}, "param_groups": [{"params": [0]}]}, root / "optimizer.pt")
        torch.save({"padding": "x" * 8192}, root / "scheduler.pt")
        torch.save({"padding": "x" * 8192}, root / "rng_state.pth")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(tmp_path / "checkpoint-1"),
         "--resumed-checkpoint", str(tmp_path / "checkpoint-2"), "--output", str(tmp_path / "smoke.json"),
         "--commit-hash", "commit", "--config-sha256", "config", "--model", "model", "--model-revision", "revision",
         "--dataset-source", "source", "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
         "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
         "--optimization-decision-sha256", "decision", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "checkpoint_model_semantics_invalid"


def test_checkpoint_gate_rejects_tiny_named_placeholders(tmp_path: Path):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    for checkpoint, step in ((initial, 1), (resumed, 2)):
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"placeholder")
    result = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(tmp_path / "smoke.json"),
         "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
         "--model-revision", "revision", "--dataset-source", "source", "--dataset-revision", "dataset-revision",
         "--dataset-sha256", "dataset", "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval",
         "--source-schema-sha256", "schema", "--optimization-decision-sha256", "decision", "--method", "dpo"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] in {"trainer_state_invalid", "checkpoint_state_too_small"}


def test_owned_checkpoint_gate_rejects_no_progress(tmp_path: Path):
    initial = tmp_path / "checkpoint-10"
    resumed = tmp_path / "resumed" / "checkpoint-10"
    write_realistic_checkpoint(initial, 10)
    write_realistic_checkpoint(resumed, 10)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_RUNNER),
            "create-smoke-manifest",
            "--initial-checkpoint",
            str(initial),
            "--resumed-checkpoint",
            str(resumed),
            "--output", str(tmp_path / "gate.json"), "--commit-hash", "commit", "--config-sha256", "config",
            "--model", "model", "--model-revision", "revision", "--dataset-source", "source",
            "--dataset-revision", "dataset-revision", "--dataset-sha256", "dataset",
            "--prompt-sha256", "prompt", "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
            "--optimization-decision-sha256", "decision", "--method", "dpo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["fallback_reason"] == "resume_step_not_advanced"


def test_comparisons_reaches_launch_only_after_all_decisions_and_smokes_validate(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    sft_train = tmp_path / "sft.jsonl"
    sft_train.write_text('{"id":"sft"}\n', encoding="utf-8")
    sft_eval = tmp_path / "sft-eval.jsonl"
    sft_eval.write_text('{"id":"eval"}\n', encoding="utf-8")
    rl_data = tmp_path / "rl.jsonl"
    rl_data.write_text('{"id":"rl"}\n', encoding="utf-8")
    rl_eval = tmp_path / "rl-eval.jsonl"
    rl_eval.write_text('{"id":"rl-eval"}\n', encoding="utf-8")
    val_data = tmp_path / "val.jsonl"
    val_data.write_text('{"id":"val"}\n', encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    common = {"commit_hash": commit, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "dataset_source": "source", "dataset_revision": "dataset-revision", "prompt_sha256": "prompt",
              "retrieval_sha256": "retrieval", "source_schema_sha256": "schema"}
    output_root = tmp_path / "output"
    specs = {
        "SFT_TRAIN": ("base", "base-revision", sft_train, sft_eval, "sft"),
        "GRPO_TRAIN": ("rl-start", "rl-revision", rl_data, rl_eval, "grpo"),
        "DAPO_TRAIN": ("rl-start", "rl-revision", rl_data, rl_eval, "dapo"),
        "SFT_GENERATION": (str(output_root / "sft/final"), "sft-output-revision", val_data, None, None),
        "GRPO_GENERATION": (str(output_root / "grpo/final"), "grpo-output-revision", val_data, None, None),
        "DAPO_GENERATION": (str(output_root / "dapo/final"), "dapo-output-revision", val_data, None, None),
    }
    env: dict[str, str] = {}
    for name, (model, revision, data, eval_data, method) in specs.items():
        identities = {**common, "model": model, "model_revision": revision, "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        if eval_data is not None:
            identities["eval_dataset_sha256"] = hashlib.sha256(eval_data.read_bytes()).hexdigest()
        decision = tmp_path / f"{name.lower()}-decision.json"
        decision_sha = write_identity_decision(decision, identities)
        env[f"{name}_DECISION"] = str(decision)
        env[f"{name}_DECISION_SHA256"] = decision_sha
        if method:
            manifest, manifest_sha = write_smoke_manifest(tmp_path, name.lower(), identities, decision_sha, method)
            env[f"{name.split('_')[0]}_SMOKE_MANIFEST"] = str(manifest)
            env[f"{name.split('_')[0]}_SMOKE_MANIFEST_SHA256"] = manifest_sha
    fake_bin = tmp_path / "home/.local/bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "module").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "uv").write_text(
        "#!/bin/bash\ncase \" $* \" in *\" --help \"*) echo '--eval --dataloader-workers --per-device-train-batch-size'; exit 0;; "
        "*\" torch.distributed.run \"*) echo mocked-training-launch >&2; exit 91;; esac\n"
        f"if [[ \"${{1:-}}\" == run && \"${{2:-}}\" == --frozen && \"${{3:-}}\" == python && \"${{4:-}}\" == \"{PROBE_RUNNER}\" ]]; then exec \"{sys.executable}\" \"${{@:4}}\"; fi\n"
        "echo 'torch=x;transformers=x;trl=x;deepspeed=x;bitsandbytes=x'\n",
        encoding="utf-8",
    )
    for item in fake_bin.iterdir():
        item.chmod(0o755)
    env.update({
        "HOME": str(tmp_path / "home"), "PATH": f"{fake_bin}:{os.environ['PATH']}", "HF_CACHE_ROOT": str(tmp_path / "hf"), "TURING_ACCOUNT": "account",
        "PROJECT_DIR": str(ROOT), "CONFIG": str(config), "BASE_MODEL": "base", "BASE_REVISION": "base-revision",
        "RL_START_MODEL": "rl-start", "RL_START_REVISION": "rl-revision", "SFT_TRAIN": str(sft_train),
        "SFT_EVAL": str(sft_eval), "RL_DATA": str(rl_data), "RL_EVAL": str(rl_eval), "VAL_DATA": str(val_data), "OUTPUT_ROOT": str(output_root),
        "TRAIN_GPUS": "4", "DATASET_SOURCE": "source", "DATASET_REVISION": "dataset-revision", "PROMPT_HASH": "prompt",
        "RETRIEVAL_HASH": "retrieval", "SOURCE_SCHEMA_HASH": "schema", "POLICY_HASH": "policy", "LEARNING_RATE": "1e-6",
        "EPOCHS": "1", "SAVE_STEPS": "100", "EVAL_STEPS": "100", "SFT_OUTPUT_REVISION": "sft-output-revision",
        "GRPO_OUTPUT_REVISION": "grpo-output-revision", "DAPO_OUTPUT_REVISION": "dapo-output-revision",
        "STUDENT_THINKING_MODE": "direct", "SCRATCHPAD_MAX_NEW_TOKENS": "128", "QUERY_TEMPERATURE": "0",
        "RESPONSE_TEMPERATURE": "0", "TOP_P": "1", "TOP_K": "8", "BM25_K1": "1.2", "BM25_B": "0.75",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_JOB_ID": "123",
    })
    result = run_script("turing_comparisons.sh", env)
    assert result.returncode == 91, result.stderr
    assert "event=launch_contract_validated" in result.stdout
    assert "mocked-training-launch" in result.stderr


def test_build_preferences_fails_without_explicit_canonical_data(tmp_path: Path):
    result = run_script("turing_build_preferences.sh", {
        "TURING_ACCOUNT": "account", "PROJECT_DIR": str(ROOT), "TRAJECTORIES": str(tmp_path / "trajectories.jsonl"),
        "OUTPUT": str(tmp_path / "preferences.jsonl"),
    })
    assert result.returncode != 0
    assert "PREFERENCE_DATA" in result.stderr
