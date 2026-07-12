#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf import UVLFRunConfig
from auroralf.io import canonical_config_sha256, read_uvlf_artifact


RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one submitted AuroraLF v2 SLURM job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--requested-cpus", type=int, required=True)
    parser.add_argument("--requested-memory", required=True)
    parser.add_argument("--requested-time", required=True)
    parser.add_argument("--stdout-pattern", type=Path, required=True)
    parser.add_argument("--stderr-pattern", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_json_no_clobber(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        content = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"execution manifest already exists: {path}") from None
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def execute_job(args: argparse.Namespace) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError("run_uvlf_v2_job.py requires SLURM_JOB_ID")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if allocated != int(args.requested_cpus) or allocated <= 0:
        raise RuntimeError("SLURM_CPUS_PER_TASK does not match requested CPUs")
    config_path = args.config.expanduser().resolve(strict=True)
    config = UVLFRunConfig.from_toml(config_path)
    if config.sampling.workers > allocated:
        raise ValueError("config workers exceed the submitted CPU allocation")
    stdout_path = Path(str(args.stdout_pattern).replace("%j", job_id)).resolve()
    stderr_path = Path(str(args.stderr_pattern).replace("%j", job_id)).resolve()
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(RUN_SCRIPT), "--config", str(config_path)]
    if args.resume:
        command.append("--resume")
    elif args.overwrite:
        command.append("--overwrite")
    manifest = (PROJECT_ROOT / "outputs" / f"uvlf_v2_execution_{job_id}.json").resolve()
    payload: dict[str, object] = {
        "schema_version": "auroralf.slurm_execution.v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "job_id": job_id,
        "requested_cpus": int(args.requested_cpus),
        "requested_memory": str(args.requested_memory),
        "requested_time": str(args.requested_time),
        "config_path": str(config_path),
        "config_sha256": canonical_config_sha256(config),
        "command": command,
        "command_shell": shlex.join(command),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code": 0,
        "final_artifact_validated": True,
        "publication_invariant": (
            "The final artifact is published only if the command exits zero and strict "
            "read_uvlf_artifact validation succeeds."
        ),
    }
    _write_json_no_clobber(manifest, payload)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["AURORALF_EXECUTION_MANIFEST"] = str(manifest)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"UVLF v2 command failed with exit code {completed.returncode}")
    artifact = read_uvlf_artifact(config.output.artifact_path, load_samples=False)
    if artifact.provenance.source_checksums[-1].label != "slurm_execution":
        raise RuntimeError("final artifact is missing SLURM execution provenance")
    if artifact.provenance.source_checksums[-1].path != manifest:
        raise RuntimeError("final artifact references the wrong SLURM execution manifest")
    artifact.provenance.verify_sources()
    print(f"validated_uvlf_v2_artifact={config.output.artifact_path}", flush=True)
    return config.output.artifact_path


def main(argv: list[str] | None = None) -> None:
    execute_job(_parse_args(argv))


if __name__ == "__main__":
    main()
