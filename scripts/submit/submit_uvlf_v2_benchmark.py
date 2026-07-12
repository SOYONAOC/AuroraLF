#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TARGET = _PROJECT_ROOT / "scripts" / "analysis" / "benchmark_uvlf_v2_streaming.py"
_DEFAULT_REPORT = _PROJECT_ROOT / "outputs" / "uvlf_v2_streaming_benchmark.json"
_DEFAULT_STDOUT = _PROJECT_ROOT / "outputs" / "uvlf_v2_benchmark-%j.out"
_DEFAULT_STDERR = _PROJECT_ROOT / "outputs" / "uvlf_v2_benchmark-%j.err"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the reduced AuroraLF v2 streaming benchmark to SLURM. "
            "Put benchmark-script arguments after '--'."
        )
    )
    parser.add_argument("--job-name", default="uvlf_v2_benchmark")
    parser.add_argument("--target-script", default=str(_DEFAULT_TARGET))
    parser.add_argument("--report", default=str(_DEFAULT_REPORT))
    parser.add_argument("--cpus-per-task", type=_positive_int, default=2)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--child-timeout-seconds", type=_positive_float, default=480.0)
    parser.add_argument("--output", default=str(_DEFAULT_STDOUT))
    parser.add_argument("--error", default=str(_DEFAULT_STDERR))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        args.script_args = [*unknown, *args.script_args]
    return args


def _target_args(raw_args: list[str]) -> list[str]:
    if not raw_args:
        return []
    if raw_args[0] != "--":
        raise ValueError("Put benchmark-script arguments after '--'.")
    return raw_args[1:]


def _resolve_project_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        (_PROJECT_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _build_sbatch_command(args: argparse.Namespace) -> list[str]:
    target = _resolve_project_path(args.target_script)
    report = _resolve_project_path(args.report)
    stdout = _resolve_project_path(args.output)
    stderr = _resolve_project_path(args.error)
    project_python = _PROJECT_ROOT / ".venv" / "bin" / "python"
    if not target.is_file():
        raise FileNotFoundError(f"benchmark target script not found: {target}")
    if not project_python.is_file():
        raise FileNotFoundError(f"project Python interpreter not found: {project_python}")
    if report.suffix != ".json":
        raise ValueError("benchmark report path must have suffix .json")
    if type(args.mem) is not str or not args.mem.strip():
        raise ValueError("--mem must be a non-empty SLURM memory value")
    if type(args.time) is not str or not args.time.strip():
        raise ValueError("--time must be a non-empty SLURM time value")
    if type(args.job_name) is not str or not args.job_name.strip():
        raise ValueError("--job-name must be non-empty")

    benchmark_command = [
        str(project_python),
        str(target),
        "--report",
        str(report),
        "--child-timeout-seconds",
        str(args.child_timeout_seconds),
        *_target_args(list(args.script_args)),
    ]
    wrapped = (
        f"cd {shlex.quote(str(_PROJECT_ROOT))} && "
        "PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        f"{shlex.join(benchmark_command)}"
    )
    command = [
        "sbatch",
        "--job-name",
        args.job_name,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem",
        args.mem,
        "--time",
        args.time,
        "--output",
        str(stdout),
        "--error",
        str(stderr),
    ]
    if args.partition is not None:
        command.extend(["--partition", str(args.partition)])
    command.extend(["--wrap", wrapped])
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    command = _build_sbatch_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=_PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
