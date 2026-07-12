#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf import UVLFRunConfig


JOB_RUNNER = PROJECT_ROOT / "scripts" / "submit" / "run_uvlf_v2_job.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the AuroraLF v2 production CLI to SLURM.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job-name", default="auroralf_uvlf_v2")
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--nodelist", default=None)
    parser.add_argument("--dry-run", action="store_true")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_sbatch_command(args: argparse.Namespace) -> tuple[list[str], UVLFRunConfig]:
    config_path = args.config.expanduser().resolve(strict=True)
    config = UVLFRunConfig.from_toml(config_path)
    cpus = config.sampling.workers if args.cpus is None else int(args.cpus)
    if cpus < config.sampling.workers:
        raise ValueError(
            f"--cpus {cpus} is smaller than config sampling.workers={config.sampling.workers}"
        )
    if cpus <= 0:
        raise ValueError("--cpus must be positive")
    if not str(args.mem).strip():
        raise ValueError("--mem must be non-empty")
    if not str(args.time).strip():
        raise ValueError("--time must be non-empty")
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    stdout = outputs / f"{args.job_name}_%j.out"
    stderr = outputs / f"{args.job_name}_%j.err"
    runner_command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(JOB_RUNNER),
        "--config",
        str(config_path),
        "--requested-cpus",
        str(cpus),
        "--requested-memory",
        str(args.mem),
        "--requested-time",
        str(args.time),
        "--stdout-pattern",
        str(stdout),
        "--stderr-pattern",
        str(stderr),
    ]
    if args.resume:
        runner_command.append("--resume")
    elif args.overwrite:
        runner_command.append("--overwrite")
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={args.job_name}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        f"--output={stdout}",
        f"--error={stderr}",
    ]
    if args.partition is not None:
        command.append(f"--partition={args.partition}")
    if args.nodelist is not None:
        command.append(f"--nodelist={args.nodelist}")
    command.append(f"--wrap={shlex.join(runner_command)}")
    return command, config


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    command, config = build_sbatch_command(args)
    print(f"config={args.config.expanduser().resolve(strict=True)}")
    print(f"artifact={config.output.artifact_path}")
    print(f"sbatch_command={shlex.join(command)}")
    if args.dry_run:
        print("dry_run=true")
        return
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse sbatch job id: {completed.stdout!r}")
    print(f"submitted_job_id={job_id}")


if __name__ == "__main__":
    main()
