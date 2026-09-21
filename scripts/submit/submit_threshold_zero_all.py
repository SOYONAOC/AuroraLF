"""Submit the frozen all-observable rerun to cp6; dry-run by default."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from auroralf.experiments.deployment import verification_command

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/threshold_zero_all_20260918"
p = argparse.ArgumentParser()
p.add_argument("--prepare", action="store_true")
p.add_argument("--apply", action="store_true")
a = p.parse_args()
plan = json.loads((OUT / "tasks.json").read_text())
remote = plan["remote"]


def ssh(s):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "sc", s], text=True
    )


if a.prepare:
    ssh("mkdir " + shlex.quote(remote))
    subprocess.run(
        [
            "rsync",
            "-a",
            str(ROOT / "outputs/deployments" / plan["release"]) + "/",
            "sc:" + remote + "/",
        ],
        check=True,
    )
    ssh(
        "ln -s /fs2/home/xuelei/zhuhr/AuroraLF/releases/atomic-memory-20260907-01/.venv "
        + shlex.quote(remote + "/.venv")
        + "; mkdir "
        + shlex.quote(remote + "/outputs")
    )
    print("Prepared", remote)
    raise SystemExit
nodes = ssh('sinfo -N -p cp6 -h -o "%N|%t|%c|%C"')
jobs = ssh('squeue -r -u xuelei -h -o "%i|%j|%T|%D"')
usable = [
    r.split("|")
    for r in nodes.splitlines()
    if not any(s in r.split("|")[1] for s in ["down", "drain", "drng", "maint", "inval"])
]
if not usable or any(int(r[2]) != 56 for r in usable):
    raise RuntimeError("Unexpected cp6 hardware")
reserved = sum(int(r.split("|")[3]) for r in jobs.splitlines())
concurrency = 10 - reserved
if concurrency < 1:
    raise RuntimeError("User ten-node ceiling reached")
cases = " ".join(f"{i}) {cmd} ;;" for i, cmd in enumerate(plan["commands"]))
wrap = (
    "set -eu; export PYTHONPATH=.; export MPLBACKEND=Agg; "
    + verification_command()
    + '; case "$SLURM_ARRAY_TASK_ID" in '
    + cases
    + " *) exit 64 ;; esac"
)
cmd = [
    "sbatch",
    "--parsable",
    "--partition=cp6",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=56",
    "--exclusive",
    f"--array=0-{len(plan['commands']) - 1}%{concurrency}",
    "--job-name=AUR-mu0-all",
    "--chdir=" + remote,
    "--output=" + remote + "/outputs/batch_%A_%a.out",
    "--error=" + remote + "/outputs/batch_%A_%a.err",
    "--wrap",
    wrap,
]
record = dict(nodes=nodes, jobs=jobs, command=cmd, remote=remote)
if a.apply:
    path = OUT / "submission.json"
    if path.exists():
        raise FileExistsError(path)
    response = ssh(shlex.join(cmd)).strip()
    record["response"] = response
    path.write_text(json.dumps(record, indent=2) + "\n")
    if not response.isdigit():
        raise RuntimeError("Ambiguous scheduler response: " + response)
    print("Submitted", response)
else:
    (OUT / "dry-run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(shlex.join(cmd))
