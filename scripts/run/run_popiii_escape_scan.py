"""Run independent Pop III escape fractions sequentially in one GPU allocation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("A non-debug SLURM allocation is required")
    plan = json.loads(args.plan.read_text())
    for path, expected in plan["input_sha256"].items():
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen scan input changed: {path}")
    small = ROOT.parent / "SmallScale21cm"
    for case in plan["cases"]:
        print(f"START fesc_popiii={case['fesc_popiii']}", flush=True)
        subprocess.run(
            [
                str(small / "packages/EoRCaLC/.venv/bin/python"),
                str(small / "scripts/run/run_aurora_rates.py"),
                "--source-table",
                case["source"],
                "--density-root",
                plan["density_root"],
                "--redshift-list",
                plan["redshift_list"],
                "--output",
                case["run"],
                "--cases",
                "popii_popiii",
                "--density-provenance-status",
                "legacy-cosmology-unverified",
            ],
            cwd=small,
            check=True,
        )
        print(f"COMPLETE fesc_popiii={case['fesc_popiii']}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analysis/analyze_popiii_escape_scan.py"),
            "--plan",
            str(args.plan.resolve()),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
