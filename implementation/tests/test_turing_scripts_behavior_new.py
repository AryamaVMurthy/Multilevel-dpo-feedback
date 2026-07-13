import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROBE_RUNNER = SCRIPTS / "turing_probe_runner.py"


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
    identities = {
        "commit_hash": "commit",
        "config_sha256": "config", "model": "model", "model_revision": "revision",
        "dataset_source": "source", "dataset_revision": "revision", "dataset_sha256": "dataset",
        "prompt_sha256": "prompt", "retrieval_sha256": "retrieval", "source_schema_sha256": "schema",
    }
    base_config = {
        "probe_kind": "generation", "attention_implementation": "sdpa", "generation_batch_size": 4,
        "static_cache": False, "compile": False, "train_microbatch": 1,
        "gradient_accumulation_steps": 32, "dataloader_workers": 0, "packing": False,
        "padding_free": False, "use_liger_kernel": False,
    }
    baseline = tmp_path / "baseline.json"
    unsupported = tmp_path / "unsupported.json"
    baseline.write_text(json.dumps({"status": "ok", "output_hash": "same", "decoded_output_hash": "same", "tokens_per_second": 10.0, "config": base_config, "identities": identities, "package_versions": {}}), encoding="utf-8")
    unsupported.write_text(json.dumps({"status": "ok", "output_hash": "same", "decoded_output_hash": "same", "tokens_per_second": 100.0, "config": {**base_config, "compile": True}, "identities": identities, "package_versions": {}}), encoding="utf-8")
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
            "--model", "model", "--model-revision", "revision", "--dataset-sha256", "dataset",
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
    assert "task7_max_steps_cli_missing" in text
    assert text.count("torch.distributed.run") >= 2
    assert "--resume-from-checkpoint" in text
    assert "create-smoke-manifest" in text


def test_smoke_manifest_requires_checkpoint_lineage_and_matching_identity(tmp_path: Path):
    initial = tmp_path / "checkpoint-1"
    resumed = tmp_path / "checkpoint-2"
    for checkpoint, step in ((initial, 1), (resumed, 2)):
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(f"{name}-{step}".encode())
    manifest = tmp_path / "smoke.json"
    identities = [
        "--commit-hash", "commit", "--config-sha256", "config", "--model", "model",
        "--model-revision", "revision", "--dataset-sha256", "dataset", "--prompt-sha256", "prompt",
        "--retrieval-sha256", "retrieval", "--source-schema-sha256", "schema",
        "--optimization-decision-sha256", "decision", "--method", "dpo",
    ]
    create = subprocess.run(
        [sys.executable, str(PROBE_RUNNER), "create-smoke-manifest", "--initial-checkpoint", str(initial),
         "--resumed-checkpoint", str(resumed), "--output", str(manifest), *identities],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert create.returncode == 0, create.stderr
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


def test_owned_checkpoint_gate_rejects_no_progress(tmp_path: Path):
    initial = tmp_path / "checkpoint-10"
    resumed = tmp_path / "resumed" / "checkpoint-10"
    initial.mkdir()
    resumed.mkdir(parents=True)
    (initial / "trainer_state.json").write_text('{"global_step":10}\n', encoding="utf-8")
    (resumed / "trainer_state.json").write_text('{"global_step":10}\n', encoding="utf-8")
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
            "--model", "model", "--model-revision", "revision", "--dataset-sha256", "dataset",
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
