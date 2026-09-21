"""Diagnose recent first bursts with production seeds and existing rate tables.

Probabilities average over MAHs and the full random-threshold distribution,
including thresholds never crossed. This is a small local diagnostic, not a
production source-table or map rerun.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import brentq
from scipy.special import ndtr

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.experiments.ionizing_rates import RateConfig
from auroralf.mah import generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.ssp.ionizing import load_popiii_ionizing_kernel


def main():
    root = Path(__file__).resolve().parents[2]
    run = root / "data_save/ionizing_sources/popiii_age_window_v1"
    reference = root / "data_save/ionizing_sources/instantaneous_100myr_v1"
    out = root / "data_save/ionizing_sources/recent_bursts_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    original = json.loads((reference / "manifest.json").read_text())
    for folder, meta, name in [(run, manifest, "cells.npz"), (reference, original, "sources.npz")]:
        assert meta["status"] == "complete"
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == meta["product_sha256"]
    cfg = RateConfig(**manifest["model"])
    cosmo = cfg.cosmology()
    astro = FlatLambdaCDM(H0=100 * cfg.h, Om0=cfg.omega_m, Ob0=cfg.omega_b)
    kernel = load_popiii_ionizing_kernel(cfg.popiii_ssp)
    with np.load(run / "cells.npz") as d:
        zs, mass, windows, rates = (d[k] for k in ["redshifts", "mass_msun", "windows_myr", "mean"])
    with np.load(run / "weights.npz") as d:
        np.testing.assert_array_equal(d["redshifts"], zs)
        np.testing.assert_array_equal(d["mass_msun"], mass)
        weights = d["weight_mpc3"]
    with np.load(reference / "sources.npz") as d:
        zi = [int(np.flatnonzero(d["redshifts"] == z)[0]) for z in zs]
        np.testing.assert_array_equal(d["mass_msun"], mass)
        popii = d["mean_rate"][zi, :, 0]
    j6 = int(np.flatnonzero(windows == 6)[0])
    globals_ = []
    for iz in range(6):
        n2 = float(weights[iz] @ popii[iz])
        n3 = float(weights[iz] @ rates[iz, :, j6])
        contributions = weights[iz] * rates[iz, :, j6]
        cumulative = np.cumsum(contributions) / n3
        globals_.append(
            dict(
                z=float(zs[iz]),
                popii_rate=n2,
                popiii_rate=n3,
                popiii_fraction=n3 / (n2 + n3),
                popiii_mass_median_grid=float(mass[np.searchsorted(cumulative, 0.5)]),
                popiii_fraction_from_below_1e10=float(contributions[mass < 1e10].sum() / n3),
            )
        )
    cells = []
    for z in [6.0, 10.0, 15.0]:
        for mf in [1e8, 1e10, 1e12]:
            im = int(np.flatnonzero(np.isclose(mass, mf, rtol=1e-10))[0])
            dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
            records, ratios, times = [], [], []
            for start in range(0, cfg.n_tracks, cfg.track_chunk):
                block_seed = int(
                    np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                        1, dtype=np.uint64
                    )[0]
                )
                seeds = derive_pipeline_random_seeds(block_seed, redshift=z, mass_index=im)
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
                ).tracks
                t, m, zz = [h[k].reshape(cfg.track_chunk, -1) for k in ["t_gyr", "Mh", "z"]]
                ratio = np.log10(m / compute_atomic_cooling_mass_msun(zz, cosmology=cosmo))
                records.append(np.maximum.accumulate(ratio, axis=1))
                ratios.append(ratio)
                times.append(t)
            rec, ratio, t = map(np.concatenate, [records, ratios, times])

            def cdf(r):
                return ndtr((r - cfg.q_log10_mean) / cfg.q_log10_sigma)

            pre = cdf(rec[:, 0])
            crossed = cdf(rec[:, -1])
            resolved = crossed - pre

            def recent(age):
                # First passages use linearly interpolated log(M/Mcool), while
                # older record maxima are retained even on descending segments.
                cutoff = t[:, -1] - age / 1000
                j = np.array(
                    [np.searchsorted(tt, cut, side="right") - 1 for tt, cut in zip(t, cutoff)]
                )
                j = np.clip(j, 0, t.shape[1] - 2)
                row = np.arange(len(t))
                f = np.clip((cutoff - t[row, j]) / (t[row, j + 1] - t[row, j]), 0, 1)
                level = np.maximum(
                    rec[row, j], ratio[row, j] + f * (ratio[row, j + 1] - ratio[row, j])
                )
                return crossed - cdf(level)

            recent6 = recent(6)
            np.testing.assert_allclose(pre + resolved + 1 - crossed, 1, atol=1e-14)
            assert np.all(recent6 >= -1e-14) and np.all(recent6 <= resolved + 1e-14)
            median = brentq(
                lambda age: recent(age).mean() - 0.5 * resolved.mean(),
                0,
                float((t[0, -1] - t[0, 0]) * 1000),
            )
            iz = int(np.flatnonzero(zs == z)[0])
            n2, n3 = float(popii[iz, im]), float(rates[iz, im, j6])
            cells.append(
                dict(
                    z=z,
                    mass_msun=mf,
                    recent6_probability=float(recent6.mean()),
                    recent6_probability_se=float(recent6.std(ddof=1) / np.sqrt(len(t))),
                    resolved_burst_probability=float(resolved.mean()),
                    pre_start_probability=float(pre.mean()),
                    not_yet_probability=float((1 - crossed).mean()),
                    resolved_median_age_myr=median,
                    popii_rate=n2,
                    popiii_rate=n3,
                    popiii_fraction=n3 / (n2 + n3),
                )
            )
            print(json.dumps(cells[-1]), flush=True)
    ages = np.array([1.0, 3.0, 6.0, 10.0, 30.0, 100.0])
    age_example = float((astro.age(6) - astro.age(10)).value * 1000)
    result = dict(
        status="complete",
        tracks_per_cell=cfg.n_tracks,
        probability_denominator="All MAH/threshold realizations at fixed final mass and redshift; one first burst per main branch",
        global_rate_units="photons/s/cMpc^3",
        single_halo_rate_units="photons/s/halo",
        settings=dict(popii_age_myr=100, popiii_age_myr=6, fesc_both=0.2, epsilon_b=0.03),
        global_rates=globals_,
        cells=cells,
        ssp=[
            dict(
                age_myr=float(a),
                rate_per_msun=float(kernel.rate(a)),
                relative_to_1myr=float(kernel.rate(a) / kernel.rate(1)),
            )
            for a in ages
        ],
        example=dict(
            birth_z=10,
            observation_z=6,
            age_myr=age_example,
            intrinsic_rate_relative_to_1myr=float(kernel.rate(age_example) / kernel.rate(1)),
        ),
        limitations="No pristine-gas/metal-enrichment gate; main branch only. Source diagnostics, not a reionization-map rerun.",
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "cells.npz",
                run / "weights.npz",
                reference / "sources.npz",
                Path(cfg.popiii_ssp),
                Path(__file__),
            ]
        },
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(global_rates=globals_, ssp=result["ssp"], example=result["example"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
