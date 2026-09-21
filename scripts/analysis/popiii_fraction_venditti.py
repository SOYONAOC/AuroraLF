"""Conditional young Pop III mass fractions, with an explicit formed-mass proxy.

The Pop II denominator is main-branch integrated initial stellar mass, with no
stellar mass return or accreted stellar component. This is not a reproduction
of Venditti's surviving stellar-particle mass or stellar-mass-selected sample.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np  # noqa: E402
from astropy.cosmology import FlatLambdaCDM  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from auroralf.cooling import compute_atomic_cooling_mass_msun  # noqa: E402
from auroralf.experiments.ionizing_rates import RateConfig  # noqa: E402
from auroralf.mah import generate_halo_histories  # noqa: E402
from auroralf.seeding import derive_pipeline_random_seeds  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_save/popiii_fraction_venditti_20260917"
REFERENCE = ROOT / "data_save/ionizing_sources/sfrd_v1"


def integrate(t, mass, ratio, m2, cfg, order):
    """Integrate the full q distribution over first passages in the last 3 Myr."""
    record = np.maximum.accumulate(ratio, axis=1)
    rows, left = np.nonzero(ratio[:, 1:] > record[:, :-1])
    den = ratio[rows, left + 1] - ratio[rows, left]
    f0 = np.clip(
        (t[rows, -1] - 0.003 - t[rows, left]) / (t[rows, left + 1] - t[rows, left]),
        0,
        1,
    )
    lo = np.maximum(record[rows, left], ratio[rows, left] + f0 * den)
    hi = ratio[rows, left + 1]
    keep = hi > lo
    rows, left, den, lo, hi = [a[keep] for a in (rows, left, den, lo, hi)]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros((len(t), 3))
    lm = np.log(mass)
    for node, weight in zip(nodes, weights, strict=True):
        q = lo + (node + 1) * (hi - lo) / 2
        f = (q - ratio[rows, left]) / den
        m3 = (
            cfg.epsilon_b
            * cfg.omega_b
            / cfg.omega_m
            * np.exp(lm[rows, left] + f * (lm[rows, left + 1] - lm[rows, left]))
        )
        pdf = np.exp(-0.5 * ((q - cfg.q_log10_mean) / cfg.q_log10_sigma) ** 2)
        prob = weight * (hi - lo) / 2 * pdf / (cfg.q_log10_sigma * np.sqrt(2 * np.pi))
        for j, value in enumerate((prob, prob * m3, prob * m3 / (m2[rows] + m3))):
            result[:, j] += np.bincount(rows, weights=value, minlength=len(t))
    # Independent CDF check of the first-passage event probability.
    cutoff = t[0, -1] - 0.003
    j = np.searchsorted(t[0], cutoff, side="right") - 1
    f = (cutoff - t[:, j]) / (t[:, j + 1] - t[:, j])
    level = np.maximum(record[:, j], ratio[:, j] + f * (ratio[:, j + 1] - ratio[:, j]))
    expected = ndtr((record[:, -1] - cfg.q_log10_mean) / cfg.q_log10_sigma)
    expected -= ndtr((level - cfg.q_log10_mean) / cfg.q_log10_sigma)
    np.testing.assert_allclose(result[:, 0], expected, atol=2e-15, rtol=1e-8)
    return result


def cell(task):
    z, mf, cfg_dict, im = task
    cfg = RateConfig(**cfg_dict)
    cosmo = cfg.cosmology()
    astro = FlatLambdaCDM(H0=100 * cfg.h, Om0=cfg.omega_m, Ob0=cfg.omega_b)
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    outputs, stars, errors = [], [], []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        seed = int(
            np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(seed, redshift=z, mass_index=im)
        h = generate_halo_histories(
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
            h.tracks, cosmology=cosmo, enable_time_delay=True, regular_convolution_backend="direct"
        )
        t, m, zz, sfr, active = [
            h[k].reshape(cfg.track_chunk, -1) for k in ("t_gyr", "Mh", "z", "SFR", "active_flag")
        ]
        np.testing.assert_allclose(t, np.broadcast_to(t[0], t.shape))
        m2 = np.trapezoid(np.where(active, sfr, 0), t, axis=1) * 1e9
        ratio = np.log10(m / compute_atomic_cooling_mass_msun(zz, cosmology=cosmo))
        a, b = [integrate(t, m, ratio, m2, cfg, n) for n in (16, 32)]
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
        outputs.append(b)
        stars.append(m2)
        errors.append(float(np.max(abs(a - b) / np.maximum(abs(b), 1e-30))))
    values, m2 = np.concatenate(outputs), np.concatenate(stars)
    prob, young_mass, fraction = values.mean(0)
    covariance = np.cov(values, rowvar=False) / len(values)
    gradient = np.array([-fraction / prob**2, 0, 1 / prob])
    return dict(
        z=z,
        halo_mass_msun=mf,
        host_probability=prob,
        host_probability_se=float(np.sqrt(covariance[0, 0])),
        conditional_mean_popiii_mass_msun=young_mass / prob,
        conditional_mean_mass_fraction_proxy=fraction / prob,
        conditional_mean_mass_fraction_proxy_se=float(np.sqrt(gradient @ covariance @ gradient)),
        all_halos_mean_mass_fraction_proxy=fraction,
        conditional_mean_popii_formed_mass_msun=float(np.average(m2, weights=values[:, 0])),
        popii_formed_mass_p16_p50_p84_msun=np.percentile(m2, [16, 50, 84]).tolist(),
        max_quadrature_relative_error=max(errors),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redshifts", type=float, nargs="+", default=[6.0, 6.7])
    parser.add_argument("--masses", type=float, nargs="+", default=[1e10, 1e11, 1e12])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("A non-debug SLURM allocation is required")
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    if any(
        not np.isfinite(z) or z < 0 or z >= manifest["model"]["z_start"] for z in args.redshifts
    ):
        raise ValueError("Redshifts must be finite and between zero and z_start")
    assert (
        hashlib.sha256((REFERENCE / "cells.npz").read_bytes()).hexdigest()
        == manifest["product_sha256"]
    )
    with np.load(REFERENCE / "cells.npz") as d:
        masses = d["mass_msun"]
    tasks = []
    for z in args.redshifts:
        for mf in args.masses:
            indices = np.flatnonzero(np.isclose(masses, mf, rtol=1e-10, atol=0))
            if len(indices) != 1:
                raise ValueError(f"Requested mass {mf} does not match one source-grid mass")
            tasks.append((z, mf, manifest["model"], int(indices[0])))
    with ProcessPoolExecutor(max_workers=min(6, int(os.environ["SLURM_CPUS_PER_TASK"]))) as pool:
        rows = list(pool.map(cell, tasks))
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="complete",
        job_id=os.environ["SLURM_JOB_ID"],
        model=manifest["model"],
        age_window_myr=3,
        mass_fraction_definition="MIII,initial(age<3Myr)/(MII,formed,mainbranch + MIII,initial(age<3Myr)); expectation conditional on recent burst",
        limitations=__doc__,
        rows=rows,
        requested_redshifts=args.redshifts,
        requested_masses_msun=args.masses,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "mass_proxy.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
