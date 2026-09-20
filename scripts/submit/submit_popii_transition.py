"""Freeze and submit independent UVLF/rate jobs to cp6, with no preflight job."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from auroralf.experiments.artifacts import digest
from auroralf.experiments.deployment import freeze_release, verification_command, verify_release
from auroralf.experiments.transition_workers import validate_variants

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = "configs/uvlf/popii_popiii.json"


def ssh(command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "sc", command], text=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path(DEFAULT_PLAN))
    p.add_argument("--release", required=True, help="Unique frozen release directory name")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument(
        "--dry-run", action="store_true", help="Default: print commands without submitting"
    )
    args = p.parse_args()
    if not args.release or Path(args.release).name != args.release or args.release in (".", ".."):
        raise ValueError("release must be one directory name")
    plan_name = str(args.plan.resolve(strict=True).relative_to(ROOT))
    plan = json.loads((ROOT / plan_name).read_text())
    validate_variants(plan["variants"])
    remote = "/fs2/home/xuelei/zhuhr/AuroraLF/releases/" + args.release
    out = ROOT / "outputs" / args.release
    out.mkdir(parents=True, exist_ok=True)
    release = ROOT / "outputs/deployments" / args.release
    if args.prepare:
        files = [str(p.relative_to(ROOT)) for p in (ROOT / "auroralf").rglob("*.py")]
        files += [
            plan_name,
            "scripts/run/run_popii_transition.py",
            "pyproject.toml",
            "uv.lock",
            plan["model"]["popii_ssp"],
            plan["model"]["popiii_ssp"],
            plan["popiii_uv"],
        ]
        freeze_release(ROOT, release, files)
        ssh("mkdir " + shlex.quote(remote))
        subprocess.run(["rsync", "-a", str(release) + "/", "sc:" + remote + "/"], check=True)
        ssh(
            "ln -s /fs2/home/xuelei/zhuhr/AuroraLF/releases/atomic-memory-20260907-01/.venv "
            + shlex.quote(remote + "/.venv")
        )
        ssh("mkdir " + shlex.quote(remote + "/outputs"))
        print("Prepared", remote)
        return
    if args.apply:
        verify_release(release)
        if digest(release / plan_name) != digest(ROOT / plan_name):
            raise ValueError("Requested plan differs from frozen release; prepare a new release")
    snapshot = dict(
        nodes=ssh('sinfo -N -p cp6 -h -o "%N|%t|%C"'),
        jobs=ssh('squeue -u xuelei -h -o "%i|%j|%T|%D"'),
    )
    if not snapshot["nodes"].strip():
        raise RuntimeError("cp6 node snapshot empty")
    commands = []
    for kind in ("uvlf", "rates"):
        wrap = (
            "set -eu; export PYTHONPATH=.; "
            + verification_command()
            + "; "
            + shlex.join(
                [
                    ".venv/bin/python",
                    "scripts/run/run_popii_transition.py",
                    "--plan",
                    plan_name,
                    "--kind",
                    kind,
                ]
            )
        )
        commands.append(
            [
                "sbatch",
                "--parsable",
                "--partition=cp6",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=56",
                "--exclusive",
                "--job-name=AUR-transition-" + kind,
                "--chdir=" + remote,
                "--output=" + remote + "/outputs/" + kind + "_%j.out",
                "--error=" + remote + "/outputs/" + kind + "_%j.err",
                "--wrap",
                wrap,
            ]
        )
    record = dict(remote=remote, release=str(release), commands=commands, scheduler=snapshot)
    if not args.apply:
        (out / "dry-run.json").write_text(json.dumps(record, indent=2) + "\n")
        for cmd in commands:
            print(shlex.join(cmd))
        return
    path = out / "submission.json"
    if path.exists():
        raise FileExistsError(path)
    record["jobs"] = {}
    path.write_text(json.dumps(record, indent=2) + "\n")
    for kind, cmd in zip(("uvlf", "rates"), commands, strict=True):
        answer = ssh(shlex.join(cmd)).strip()
        if not answer.isdigit():
            raise RuntimeError("Ambiguous sbatch response: " + answer)
        record["jobs"][kind] = answer
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(kind, answer, flush=True)


if __name__ == "__main__":
    main()
