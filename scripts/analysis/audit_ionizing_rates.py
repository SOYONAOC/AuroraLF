"""Reproduce existing source histories and diagnose current emission and ordering.

Runs in an AuroraLF SLURM allocation. An explicit subprocess uses SmallScale's
own environment for the exact original HMF; neither environment is modified.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Each spawned worker gets one BLAS thread; configure before importing NumPy.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export_weights(a):
    from astropy import units as u
    from eorcalc import MassFunctionConfig, PowerSpectrumConfig
    from eorcalc.powerspec import create_mass_functions

    manifest = json.loads((a.run / "manifest.json").read_text())
    cosmo = create_mass_functions(
        PowerSpectrumConfig(**manifest["power"]),
        MassFunctionConfig(hmf_model="Reed07"),
        cache_dir=a.output / "hmf_cache",
    )
    with np.load(a.output / "cells.npz") as data:
        mass, zs = data["mass_msun"], data["redshifts"]
    lm = np.log10(mass)
    width = np.r_[np.diff(lm)[0] / 2, (lm[2:] - lm[:-2]) / 2, np.diff(lm)[-1] / 2]
    nh = cosmo.nHu.to_value(u.Mpc**-3)
    weights = [cosmo.dndmst(mass, z) * mass * np.log(10) * width / nh for z in zs]
    np.savez(a.output / "weights.npz", weight_per_h=weights, n_h_mpc3=nh)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights-only", action="store_true")
    a = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("non-debug SLURM allocation required")
    if a.weights_only:
        export_weights(a)
        return
    from auroralf.experiments.ionizing_audit import audit_cell
    from auroralf.experiments.ionizing_sources import SourceConfig, initialize_worker

    original = json.loads((a.source / "manifest.json").read_text())
    if (
        original["status"] != "complete"
        or sha(a.source / "sources.npz") != original["product_sha256"]
    ):
        raise ValueError("source table is incomplete or changed")
    for path, expected in original["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"original source implementation changed: {path}")
    for path, expected in original["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"original SSP changed: {path}")
    cfg = SourceConfig(**original["resolved_model"])
    with np.load(a.source / "sources.npz") as old:
        mass = old["mass_msun"]
        zs = old["redshifts"][old["redshifts"] <= 20]
        reference = old["mean_photons"][: len(zs), :, :2]
    a.output.mkdir(parents=True, exist_ok=False)
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
            if n % 50 == 0:
                print(f"{n}/{len(tasks)} cells audited", flush=True)
    np.testing.assert_allclose(means[:, :, :2], reference, rtol=1e-12, atol=0)
    names = [
        "NII",
        "NIII",
        "QII",
        "QIII",
        "NIII_prior_popii",
        "QIII_prior_popii",
        "NIII_q_gt_1",
        "QIII_q_gt_1",
        "QII_quadrature_absolute_change",
    ]
    np.savez_compressed(
        a.output / "cells.npz",
        redshifts=zs,
        mass_msun=mass,
        channel_names=names,
        means=means,
        rate_covariance_of_mean=cov,
    )
    smallscale = a.run.resolve().parents[2]
    subprocess.run(
        [
            str(smallscale / "packages/EoRCaLC/.venv/bin/python"),
            str(Path(__file__).resolve()),
            "--source",
            str(a.source.resolve()),
            "--run",
            str(a.run.resolve()),
            "--output",
            str(a.output.resolve()),
            "--weights-only",
        ],
        check=True,
    )
    with np.load(a.output / "weights.npz") as data:
        weights, nh = data["weight_per_h"], float(data["n_h_mpc3"])
    budget = json.loads((a.run / "photon_budget.json").read_text())
    rows = []
    for i, z in enumerate(zs):
        total = np.sum(weights[i, :, None] * means[i], axis=0)
        old = next(row for row in budget if row["z"] == z)
        np.testing.assert_allclose(total[:2], old["photons_per_h"][:2], rtol=1e-10, atol=0)
        c = np.sum(weights[i, :, None, None] ** 2 * cov[i], axis=0)
        r2, r3 = total[2:4]
        gradient = np.array([-r3, r2]) / (r2 + r3) ** 2
        row = dict(
            z=float(z),
            cumulative_photons_per_h=total[:2].tolist(),
            current_escaped_rate_s_mpc3=(total[2:4] * nh).tolist(),
            current_popiii_share=float(r3 / (r2 + r3)),
            current_popiii_share_mc_se=float(np.sqrt(gradient @ c @ gradient)),
            cumulative_popiii_share=float(total[1] / total[:2].sum()),
            popiii_cumulative_with_prior_popii=float(total[4] / total[1]),
            popiii_current_with_prior_popii=float(total[5] / r3),
            popiii_cumulative_q_gt_1=float(total[6] / total[1]),
            popiii_current_q_gt_1=float(total[7] / r3),
            popii_quadrature_relative_check=float(total[8] / r2),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        status="complete",
        job=os.environ["SLURM_JOB_ID"],
        rows=rows,
        cumulative_source_reproduction="all 666 mass-redshift cells match rtol 1e-12; global budgets match rtol 1e-10",
        interpretation="current rates computed directly from SSP and same birth histories, not differences of cumulative tables; pre-start unknown excluded",
        prior_popii="positive integral of model Pop II SFR before Pop III burst; diagnostic, not proof all gas was enriched",
        model=original["resolved_model"],
        source_sha256=original["product_sha256"],
        audit_code_sha256={
            str(p): sha(p)
            for p in [Path(__file__).resolve(), ROOT / "auroralf/experiments/ionizing_audit.py"]
        },
        products_sha256={p.name: sha(p) for p in a.output.glob("*.npz")},
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
