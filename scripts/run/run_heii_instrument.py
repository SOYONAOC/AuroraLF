"""Run bounded instrument post-processing in one allocation or local workstation."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_redshift(z):
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    log = ROOT / f"outputs/heii_instrument_20260918/response_z{z:g}.log"
    with log.open("w") as handle:
        subprocess.run(
            [sys.executable, "-u", "scripts/analysis/heii_instrument_response.py", "--z", str(z)],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"z={z}: response grid complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-workers", type=int)
    args = parser.parse_args()
    if args.local_workers is not None:
        if not 1 <= args.local_workers <= 4:
            raise ValueError("Bounded workstation analysis supports 1--4 workers")
        workers = args.local_workers
    else:
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Specify --local-workers or use a SLURM allocation")
        workers = min(4, int(os.environ["SLURM_CPUS_PER_TASK"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_redshift, [6.639, 8.1623, 10.6, 12.342, 13.86]))
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, "scripts/analysis/summarize_heii_instrument.py"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print("INSTRUMENT_GRID_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
