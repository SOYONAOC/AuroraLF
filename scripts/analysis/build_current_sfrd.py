"""Current Pop II and first-burst Pop III SFRD inputs, without SSP/fesc.

Average newly formed mass over trailing 1, 5, 10 and 20 Myr. Full first-passage
history is retained; a previous crossing cannot trigger a second burst.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402
from astropy.cosmology import FlatLambdaCDM  # noqa: E402

from auroralf.cooling import compute_atomic_cooling_mass_msun  # noqa: E402
from auroralf.experiments.ionizing_rates import RateConfig  # noqa: E402
from auroralf.mah import generate_halo_histories  # noqa: E402
from auroralf.seeding import derive_pipeline_random_seeds  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = np.array([1.0, 5.0, 10.0, 20.0])


def initialize(model):
    global STATE
    cfg = RateConfig(**model)
    STATE = cfg, cfg.cosmology(), FlatLambdaCDM(H0=100 * cfg.h, Om0=cfg.omega_m, Ob0=cfg.omega_b)


def burst_mass(t, mass, cooling, mean, sigma, window, order=16):
    """Expected halo mass at first crossing in a trailing interval, per MAH."""
    ratio = np.log10(mass / cooling)
    record = np.maximum.accumulate(ratio, axis=1)
    rows, left = np.nonzero(ratio[:, 1:] > record[:, :-1])
    den = ratio[rows, left + 1] - ratio[rows, left]
    fraction = np.clip(
        (t[rows, -1] - window / 1000 - t[rows, left]) / (t[rows, left + 1] - t[rows, left]), 0, 1
    )
    lo = np.maximum(record[rows, left], ratio[rows, left] + fraction * den)
    hi = ratio[rows, left + 1]
    good = hi > lo
    rows, left, den, lo, hi = [a[good] for a in (rows, left, den, lo, hi)]
    value = np.zeros(len(rows))
    lm = np.log(mass)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for node, weight in zip(nodes, weights, strict=True):
        x = lo + (node + 1) * (hi - lo) / 2
        f = (x - ratio[rows, left]) / den
        mb = np.exp(lm[rows, left] + f * (lm[rows, left + 1] - lm[rows, left]))
        pdf = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        value += weight * (hi - lo) / 2 * pdf * mb
    return np.bincount(rows, weights=value, minlength=len(t))


def recent_sfr(t, sfr, active, window):
    """Exact average of the production piecewise-linear masked SFR."""
    s = np.where(active, sfr, 0.0)
    dt = np.diff(t, axis=1)
    lo = np.clip((t[:, -1, None] - window / 1000 - t[:, :-1]) / dt, 0, 1)
    at_lo = s[:, :-1] + lo * np.diff(s, axis=1)
    return np.sum(0.5 * (at_lo + s[:, 1:]) * dt * (1 - lo), axis=1) / (window / 1000)


def cell(task):
    iz, im, z, mf = task
    cfg, cosmo, astro = STATE
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    samples, checks = [], []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        seed = int(
            np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(seed, redshift=z, mass_index=im)
        history = generate_halo_histories(
            n_tracks=cfg.track_chunk,
            z_final=z,
            Mh_final=mf,
            z_start_max=cfg.z_start,
            cosmology=cosmo,
            random_seed=seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt,
            store_inactive_history=True,
            sampler="mcbride",
        )
        h = compute_sfr_from_tracks(
            history.tracks,
            cosmology=cosmo,
            enable_time_delay=True,
            regular_convolution_backend="direct",
        )
        t, m, zgrid, sfr, active = [
            h[k].reshape(cfg.track_chunk, -1) for k in ["t_gyr", "Mh", "z", "SFR", "active_flag"]
        ]
        assert (t[:, -1] - t[:, 0]).min() * 1000 > WINDOWS.max()
        cooling = compute_atomic_cooling_mass_msun(zgrid, cosmology=cosmo)
        factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m
        s2 = np.stack([recent_sfr(t, sfr, active, w) for w in WINDOWS], axis=1)
        s3 = np.stack(
            [
                factor
                * burst_mass(t, m, cooling, cfg.q_log10_mean, cfg.q_log10_sigma, w)
                / (w * 1e6)
                for w in WINDOWS
            ],
            axis=1,
        )
        reference = (
            factor
            * burst_mass(t, m, cooling, cfg.q_log10_mean, cfg.q_log10_sigma, 10.0, order=32)
            / 1e7
        )
        samples.append(np.stack([s2, s3], axis=-1))
        checks.append(abs(reference - s3[:, 2]))
    s = np.concatenate(samples)
    assert np.isfinite(s).all() and np.all(s >= 0)
    flat = s.reshape(len(s), -1)
    return (
        iz,
        im,
        s.mean(0),
        np.cov(flat, rowvar=False) / len(s),
        float(np.concatenate(checks).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data_save/ionizing_sources/sfrd_v1")
    parser.add_argument("--redshifts", type=float, nargs="+")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data_save/ionizing_sources/instantaneous_100myr_v1"
    )
    parser.add_argument("--q-log10-mean", type=float)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("non-debug SLURM allocation required")
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (
        hashlib.sha256((source / "sources.npz").read_bytes()).hexdigest()
        == manifest["product_sha256"]
    )
    with np.load(source / "sources.npz") as d:
        original_z, mass = d["redshifts"], d["mass_msun"]
    zs = original_z[
        np.unique(
            [
                np.argmin(abs(original_z - z))
                for z in [6, 7, 8, 9, 10, 11, 12, 12.5, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 30]
            ]
        )
    ]
    if args.redshifts is not None:
        zs = np.unique(args.redshifts)
        if (
            not np.isfinite(zs).all()
            or np.any(zs < 0)
            or np.any(zs >= manifest["resolved_model"]["z_start"])
        ):
            raise ValueError("redshifts must be finite, non-negative and below z_start")
    if args.q_log10_mean is not None:
        if not np.isfinite(args.q_log10_mean):
            raise ValueError("Non-finite threshold mean")
        manifest["resolved_model"]["q_log10_mean"] = args.q_log10_mean
    mean = np.empty((len(zs), len(mass), len(WINDOWS), 2))
    cov = np.empty((len(zs), len(mass), 8, 8))
    error = np.empty((len(zs), len(mass)))
    tasks = [(i, j, float(z), float(m)) for i, z in enumerate(zs) for j, m in enumerate(mass)]
    with ProcessPoolExecutor(
        max_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(manifest["resolved_model"],),
    ) as pool:
        futures = [pool.submit(cell, task) for task in tasks]
        for n, f in enumerate(as_completed(futures), 1):
            i, j, mean[i, j], cov[i, j], error[i, j] = f.result()
            if n % 50 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} SFR cells", flush=True)
    np.savez_compressed(
        out / "cells.npz",
        redshifts=zs,
        mass_msun=mass,
        windows_myr=WINDOWS,
        mean_sfr=mean,
        covariance_of_mean=cov,
        popiii_quadrature_error=error,
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                job_id=os.environ["SLURM_JOB_ID"],
                model=manifest["resolved_model"],
                quantity="Newly formed initial stellar mass / trailing time interval per descendant main branch",
                units="Msun/yr/halo",
                populations=["popii", "popiii"],
                windows_myr=WINDOWS.tolist(),
                limitations="Main branches only; no enrichment or pristine gas gate; no fesc/SSP weighting",
                source_manifest_sha256=hashlib.sha256(
                    (source / "manifest.json").read_bytes()
                ).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                product_sha256=hashlib.sha256((out / "cells.npz").read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
