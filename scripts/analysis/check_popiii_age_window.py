"""Paired Pop III age-window convergence using the production MAH seeds.

Only the Pop III rate is recalculated. Complete first-crossing histories and
the source model are retained; no Pop II or spatial evolution is rerun.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WINDOWS = np.array([2, 3, 4, 5, 5.25, 6, 7, 8, 10, 15, 20, 30, 50, 75, 100.0])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(model):
    from astropy.cosmology import FlatLambdaCDM

    from auroralf.experiments.ionizing_rates import RateConfig
    from auroralf.ssp.ionizing import load_popiii_ionizing_kernel

    global STATE
    cfg = RateConfig(**model)
    cosmo = cfg.cosmology()
    astro = FlatLambdaCDM(H0=100 * cfg.h, Om0=cfg.omega_m, Ob0=cfg.omega_b)
    STATE = cfg, cosmo, astro, load_popiii_ionizing_kernel(cfg.popiii_ssp)


def cell(task):
    from auroralf.cooling import compute_atomic_cooling_mass_msun
    from auroralf.experiments.ionizing_rates import threshold_averaged_rate
    from auroralf.mah import generate_halo_histories
    from auroralf.seeding import derive_pipeline_random_seeds

    iz, im, z, mass_final = task
    cfg, cosmo, astro, kernel = STATE
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    samples, uppers, checks = [], [], []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        block_seed = int(
            np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(block_seed, redshift=z, mass_index=im)
        history = generate_halo_histories(
            n_tracks=cfg.track_chunk,
            z_final=z,
            Mh_final=mass_final,
            z_start_max=cfg.z_start,
            cosmology=cosmo,
            random_seed=seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt,
            store_inactive_history=True,
            sampler="mcbride",
        ).tracks
        t, mass, redshift = [history[k].reshape(cfg.track_chunk, -1) for k in ("t_gyr", "Mh", "z")]
        cooling = compute_atomic_cooling_mass_msun(redshift, cosmology=cosmo)
        args = (t, mass, cooling, kernel, cfg.q_log10_mean, cfg.q_log10_sigma)
        result = [threshold_averaged_rate(*args, order=32, max_age_myr=float(w)) for w in WINDOWS]
        samples.append(np.stack([r[0] for r in result], axis=1))
        uppers.append(np.stack([r[1] for r in result], axis=1))
        checks.append(
            np.stack(
                [
                    threshold_averaged_rate(*args, order=16, max_age_myr=100)[0],
                    threshold_averaged_rate(*args, order=64, max_age_myr=100)[0],
                    threshold_averaged_rate(*args, order=64, max_age_myr=6)[0],
                ],
                axis=1,
            )
        )
    factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
    sample = factor * np.concatenate(samples)
    if not np.isfinite(sample).all() or np.any(sample < 0):
        raise ValueError("invalid Pop III rates")
    return (
        iz,
        im,
        sample.mean(0),
        np.cov(sample, rowvar=False) / cfg.n_tracks,
        factor * np.concatenate(uppers).mean(0),
        factor * np.concatenate(checks).mean(0),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data_save/ionizing_sources/instantaneous_100myr_v1"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data_save/ionizing_sources/popiii_age_window_v1"),
    )
    a = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("non-debug SLURM allocation required")
    original = json.loads((a.source / "manifest.json").read_text())
    if (
        original["status"] != "complete"
        or sha(a.source / "sources.npz") != original["product_sha256"]
    ):
        raise ValueError("invalid reference product")
    for path, expected in original["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"SSP changed: {path}")
    with np.load(a.source / "sources.npz") as data:
        original_z = data["redshifts"]
        zi = np.unique(
            [np.argmin(abs(original_z - z)) for z in [6, 8, 10, 12.5, 15, 20, 25, 30, 40, 49]]
        )
        zs, mass = original_z[zi], data["mass_msun"]
        reference = data["mean_rate"][zi, :, 1]
    a.output.mkdir(parents=True, exist_ok=False)
    shape = (len(zs), len(mass), len(WINDOWS))
    mean, upper = np.empty(shape), np.empty(shape)
    covariance = np.empty(shape + (len(WINDOWS),))
    checks = np.empty(shape[:2] + (3,))
    tasks = [(iz, im, float(z), float(m)) for iz, z in enumerate(zs) for im, m in enumerate(mass)]
    with ProcessPoolExecutor(
        max_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(original["resolved_model"],),
    ) as pool:
        futures = [pool.submit(cell, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            iz, im, mean[iz, im], covariance[iz, im], upper[iz, im], checks[iz, im] = (
                future.result()
            )
            if n % 25 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} paired cells", flush=True)
    np.testing.assert_allclose(checks[:, :, 0], reference, rtol=2e-10, atol=0)
    np.savez_compressed(
        a.output / "cells.npz",
        redshifts=zs,
        mass_msun=mass,
        windows_myr=WINDOWS,
        mean=mean,
        covariance_of_mean=covariance,
        censored_upper=upper,
        checks=checks,
        reference=reference,
    )
    result = dict(
        status="complete",
        job_id=os.environ["SLURM_JOB_ID"],
        model=original["resolved_model"],
        reference=str(a.source.resolve()),
        reference_sha256=sha(a.source / "manifest.json"),
        product_sha256=sha(a.output / "cells.npz"),
        redshifts=zs.tolist(),
        windows_myr=WINDOWS.tolist(),
        quadrature_order=32,
        check_columns=[
            "100 Myr order 16 (production reproduction)",
            "100 Myr order 64",
            "6 Myr order 64",
        ],
        source_sha256={
            str(path): sha(path)
            for path in [
                Path(__file__),
                ROOT / "auroralf/experiments/ionizing_rates.py",
                ROOT / "auroralf/ssp/ionizing.py",
            ]
        },
    )
    (a.output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
