#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SUBMITTER = Path.home() / ".codex" / "skills" / "dmde-compute" / "scripts" / "submit_python_job.py"
DEFAULT_TARGET_SCRIPT = "scripts/run/run_uvlf_compare_imf_no_delay_all_z.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the top-heavy Pop II IMF UVLF comparison through the project "
            "SLURM auto-node selector. Put target-script arguments after '--'."
        )
    )
    parser.add_argument("--job-name", default="uvlf_imf_compare")
    parser.add_argument("--submitter", default=str(DEFAULT_SUBMITTER))
    parser.add_argument("--target-script", default=DEFAULT_TARGET_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-fraction", type=float, default=None)
    parser.add_argument("--min-available-mem-gib", type=float, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--node-memory-check", choices=("auto", "always", "never"), default=None)
    parser.add_argument("--no-align-parallelism", action="store_true")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _target_args(raw_args: list[str]) -> list[str]:
    if not raw_args:
        return []
    if raw_args[0] != "--":
        raise ValueError("Put target-script arguments after '--'.")
    return raw_args[1:]


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    submitter = Path(args.submitter).expanduser().resolve()
    target_script = (project_root / args.target_script).resolve()
    project_python = project_root / ".venv" / "bin" / "python"

    if not submitter.exists():
        raise FileNotFoundError(f"SLURM submitter not found: {submitter}")
    if not target_script.exists():
        raise FileNotFoundError(f"Target script not found: {target_script}")
    if not project_python.exists():
        raise FileNotFoundError(f"Project Python interpreter not found: {project_python}")

    command = [
        sys.executable,
        str(submitter),
        "--job-name",
        str(args.job_name),
        "--project-root",
        str(project_root),
        "--python",
        str(project_python),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.cpu_fraction is not None:
        command.extend(["--cpu-fraction", str(args.cpu_fraction)])
    if args.min_available_mem_gib is not None:
        command.extend(["--min-available-mem-gib", str(args.min_available_mem_gib)])
    if args.time is not None:
        command.extend(["--time", str(args.time)])
    if args.node_memory_check is not None:
        command.extend(["--node-memory-check", str(args.node_memory_check)])
    if args.no_align_parallelism:
        command.append("--no-align-parallelism")

    command.append(str(target_script))
    command.extend(["--", *_target_args(list(args.script_args))])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
