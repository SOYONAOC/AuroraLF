"""Increase paired low-redshift sampling for the existing instantaneous-rate audit."""

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from audit_ionizing_rates import sha

    from auroralf.experiments.ionizing_audit import audit_cell
    from auroralf.experiments.ionizing_sources import SourceConfig, initialize_worker

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tracks", type=int, default=4096)
    a = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("non-debug SLURM allocation required")
    original = json.loads((a.reference / "summary.json").read_text())
    for name, expected in original["products_sha256"].items():
        if sha(a.reference / name) != expected:
            raise ValueError(f"reference changed: {name}")
    cfg = replace(SourceConfig(**original["model"]), n_tracks=a.tracks)
    a.output.mkdir(parents=True, exist_ok=False)
    with np.load(a.reference / "cells.npz") as data:
        indices = np.flatnonzero(data["redshifts"] <= 8)
        zs, mass = data["redshifts"][indices], data["mass_msun"]
    with np.load(a.reference / "weights.npz") as data:
        weights, nh = data["weight_per_h"][indices], float(data["n_h_mpc3"])
    means = np.empty((len(zs), len(mass), 9))
    cov = np.empty((len(zs), len(mass), 2, 2))
    tasks = [(iz, im, float(z), float(m)) for iz, z in enumerate(zs) for im, m in enumerate(mass)]
    with ProcessPoolExecutor(
        max_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(cfg,),
    ) as pool:
        for n, (iz, im, values, covariance) in enumerate(pool.map(audit_cell, tasks), 1):
            means[iz, im], cov[iz, im] = values, covariance
            if n % 20 == 0:
                print(f"{n}/{len(tasks)} refined cells", flush=True)
    np.savez_compressed(
        a.output / "cells.npz",
        redshifts=zs,
        mass_msun=mass,
        means=means,
        rate_covariance_of_mean=cov,
        weight_per_h=weights,
        n_h_mpc3=nh,
    )
    rows = []
    for i, z in enumerate(zs):
        total = np.sum(weights[i, :, None] * means[i], axis=0)
        c = np.sum(weights[i, :, None, None] ** 2 * cov[i], axis=0)
        q2, q3 = total[2:4]
        gradient = np.array([-q3, q2]) / (q2 + q3) ** 2
        row = dict(
            z=float(z),
            cumulative_photons_per_h=total[:2].tolist(),
            current_escaped_rate_s_mpc3=(total[2:4] * nh).tolist(),
            current_popiii_share=float(q3 / (q2 + q3)),
            current_popiii_share_mc_se=float(np.sqrt(gradient @ c @ gradient)),
            cumulative_popiii_share=float(total[1] / total[:2].sum()),
            popiii_cumulative_with_prior_popii=float(total[4] / total[1]),
            popiii_current_with_prior_popii=float(total[5] / q3),
            popiii_cumulative_q_gt_1=float(total[6] / total[1]),
            popiii_current_q_gt_1=float(total[7] / q3),
            popii_quadrature_relative_check=float(total[8] / q2),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        status="complete",
        job=os.environ["SLURM_JOB_ID"],
        tracks=cfg.n_tracks,
        rows=rows,
        reference=str(a.reference.resolve()),
        reference_sha256=sha(a.reference / "summary.json"),
        source_sha256=original["source_sha256"],
        audit_code_sha256={
            str(p): sha(p)
            for p in [Path(__file__).resolve(), ROOT / "auroralf/experiments/ionizing_audit.py"]
        },
        products_sha256={"cells.npz": sha(a.output / "cells.npz")},
        interpretation="same physics and time resolution; larger paired sample, contains original 256 draws, not independent repeats",
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
