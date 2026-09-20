"""Read-only diagnostics of the existing main-branch ionizing source model."""

import numpy as np

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.experiments import ionizing_sources as source
from auroralf.experiments.random_q import draw_logq, first_crossing
from auroralf.mah import generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import compute_sfr_from_tracks
from auroralf.ssp.ionizing import cumulative_photons_from_sfh


def rate_from_sfh(t, sfr, active, kernel, order=16, *, max_age_myr=None):
    """Instantaneous intrinsic photons/s from piecewise-linear birth SFR.

    Integrates SFR [Msun/yr] times SSP rate [photons/s/Msun] over birth time.
    No derivative of independently resampled cumulative source tables is used.
    A finite max_age_myr clips birth-time intervals at the exact age boundary;
    it does not truncate the history used to generate the SFR.
    """
    t, sfr, active = np.asarray(t), np.asarray(sfr), np.asarray(active)
    if (
        t.ndim != 2
        or t.shape != sfr.shape
        or t.shape != active.shape
        or t.shape[1] < 2
        or active.dtype != bool
        or not np.isfinite(t).all()
        or not np.isfinite(sfr).all()
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(sfr < 0)
    ):
        raise ValueError("invalid SFH for instantaneous source audit")
    s = np.where(active, sfr, 0.0)
    left, right = t[:, :-1], t[:, 1:]
    if max_age_myr is not None:
        if not np.isfinite(max_age_myr) or max_age_myr <= 0:
            raise ValueError("max_age_myr must be finite and positive or None")
        left = np.minimum(np.maximum(left, t[:, -1, None] - max_age_myr / 1000), right)
    original_width = np.diff(t, axis=1)
    slope = np.diff(s, axis=1) / original_width
    left_sfr = s[:, :-1] + slope * (left - t[:, :-1])
    width = right - left
    answer = np.zeros(len(t))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for node, weight in zip(nodes, weights, strict=True):
        f = (node + 1) / 2
        # Empty old intervals have no stars to evaluate in the SSP table.
        age = np.where(width > 0, (t[:, -1, None] - (left + f * width)) * 1000, 0.0)
        answer += np.sum(
            (left_sfr + f * width * slope) * kernel.rate(age) * width * (weight / 2) * 1e9,
            axis=1,
        )
    return answer


def audit_cell(task):
    """Regenerate the exact production histories and add rates/order diagnostics."""
    iz, im, z, mass_final = task
    cfg, cosmo, astro, k2, k3 = source.STATE
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    logq = draw_logq(cfg.seed, im, cfg.n_tracks, cfg.q_log10_mean, cfg.q_log10_sigma)
    pieces = []
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
            Mh_final=mass_final,
            z_start_max=cfg.z_start,
            cosmology=cosmo,
            random_seed=seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt,
            store_inactive_history=True,
            sampler="mcbride",
        )
        tracks = compute_sfr_from_tracks(
            h.tracks, cosmology=cosmo, enable_time_delay=True, regular_convolution_backend="direct"
        )
        shape = (cfg.track_chunk, -1)
        t, mass, active, sfr = [
            tracks[k].reshape(shape) for k in ("t_gyr", "Mh", "active_flag", "SFR")
        ]
        cooling = compute_atomic_cooling_mass_msun(tracks["z"].reshape(shape), cosmology=cosmo)
        q = logq[start : start + cfg.track_chunk]
        status, tb, mb = first_crossing(t, mass, cooling, q)
        valid = status == 1
        n2 = cfg.fesc_popii * cumulative_photons_from_sfh(t, sfr, active, k2)
        n3, r3 = np.zeros(len(t)), np.zeros(len(t))
        factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
        age = (t[valid, -1] - tb[valid]) * 1000
        n3[valid] = factor * mb[valid] * k3.yield_photons(age)
        r3[valid] = factor * mb[valid] * k3.rate(age)
        r2 = cfg.fesc_popii * rate_from_sfh(t, sfr, active, k2, order=16)
        r2_check = cfg.fesc_popii * rate_from_sfh(t, sfr, active, k2, order=8)
        prior = np.zeros(len(t), dtype=bool)
        width = np.diff(t[valid], axis=1)
        f = np.clip((tb[valid, None] - t[valid, :-1]) / width, 0, 1)
        s = np.where(active[valid], sfr[valid], 0)
        prior_mass = np.sum(width * (s[:, :-1] * f + 0.5 * np.diff(s, axis=1) * f**2), axis=1) * 1e9
        prior[valid] = prior_mass > 0
        pieces.append(
            np.stack(
                [
                    n2,
                    n3,
                    r2,
                    r3,
                    n3 * prior,
                    r3 * prior,
                    n3 * (q > 0),
                    r3 * (q > 0),
                    abs(r2 - r2_check),
                ],
                axis=-1,
            )
        )
    sample = np.concatenate(pieces)
    if not np.isfinite(sample).all() or np.any(sample < 0):
        raise ValueError("invalid source audit samples")
    return iz, im, sample.mean(axis=0), np.cov(sample[:, 2:4], rowvar=False, ddof=1) / cfg.n_tracks
