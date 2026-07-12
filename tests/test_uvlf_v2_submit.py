from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "submit" / "submit_uvlf_v2.py"
JOB_RUNNER = PROJECT_ROOT / "scripts" / "submit" / "run_uvlf_v2_job.py"
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"
LEGACY_SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "submit" / "submit_uvlf_imf_compare.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_submit_dry_run_renders_exact_resources_command_logs_and_policy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SUBMIT_SCRIPT),
            "--config",
            str(PRODUCTION_CONFIG),
            "--cpus",
            "40",
            "--mem",
            "96G",
            "--time",
            "18:00:00",
            "--partition",
            "amd",
            "--nodelist",
            "amd1",
            "--resume",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = completed.stdout
    assert "--cpus-per-task=40" in rendered
    assert "--mem=96G" in rendered
    assert "--time=18:00:00" in rendered
    assert "--partition=amd" in rendered
    assert "--nodelist=amd1" in rendered
    assert str(JOB_RUNNER) in rendered
    assert "--resume" in rendered
    assert "_%j.out" in rendered and "_%j.err" in rendered
    assert "dry_run=true" in rendered


def test_submit_rejects_cpu_request_smaller_than_config_workers() -> None:
    module = _load(SUBMIT_SCRIPT, "submit_uvlf_v2_cpu_test")
    args = argparse.Namespace(
        config=PRODUCTION_CONFIG,
        job_name="test",
        cpus=31,
        mem="64G",
        time="12:00:00",
        partition=None,
        nodelist=None,
        dry_run=True,
        resume=False,
        overwrite=False,
    )
    with pytest.raises(ValueError, match="smaller than config"):
        module.build_sbatch_command(args)


def test_legacy_submit_entry_is_disabled_with_migration_message() -> None:
    completed = subprocess.run(
        [sys.executable, str(LEGACY_SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "legacy UVLF submit entry point is disabled" in completed.stderr


def test_job_execution_manifest_is_atomic_no_clobber_and_strict_json(tmp_path: Path) -> None:
    module = _load(JOB_RUNNER, "run_uvlf_v2_job_manifest_test")
    path = tmp_path / "execution.json"
    payload = {
        "schema_version": "auroralf.slurm_execution.v1",
        "job_id": "12345",
        "requested_cpus": 8,
        "requested_memory": "32G",
        "requested_time": "02:00:00",
        "config_sha256": "a" * 64,
        "command": ["python", "run_uvlf_v2.py"],
        "stdout_path": "job.out",
        "stderr_path": "job.err",
        "exit_code": 0,
        "final_artifact_validated": True,
    }

    assert module._write_json_no_clobber(path, payload) == path
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        module._write_json_no_clobber(path, payload)
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".*.tmp")) == []
