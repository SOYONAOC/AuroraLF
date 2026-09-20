"""Age-dependent photon rates with the random first-crossing threshold marginalized.

The q distribution and first-passage interpolation are unchanged. Integrating
over q removes rare-young-burst sampling noise; MAH sampling remains explicit.
No integrated stellar photon yield is used by this module.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.experiments import ionizing_sources as source
from auroralf.experiments.ionizing_audit import rate_from_sfh
from auroralf.mah import generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import compute_sfr_from_tracks


@dataclass(frozen=True)
class RateConfig(source.SourceConfig):
    """Retain full MAHs; cap Pop II at 100 Myr and Pop III at 6 Myr."""

    max_lookback_myr: float = 100.0
    popiii_max_age_myr: float = 6.0

    def __post_init__(self):
        super().__post_init__()
        for key in ("max_lookback_myr", "popiii_max_age_myr"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


def threshold_averaged_rate(t, mass, cooling, kernel, mean, sigma, order=8, *, max_age_myr=None):
    """Expected Mburst*qH(age) for resolved first passages, per track.

    The integration measure is the Normal PDF of log10(q). Only newly reached
    record levels of log10(M/Mcool) may trigger a first crossing. Unknown
    pre-start events are excluded from this resolved contribution. An age limit
    clips crossing intervals after finding records along the entire history:
    an old burst cannot be retriggered by a later crossing inside the window.
    """
    t, mass, cooling = (np.asarray(v, dtype=float) for v in (t, mass, cooling))
    if (
        t.ndim != 2
        or mass.shape != t.shape
        or cooling.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, mass, cooling))
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(mass <= 0)
        or np.any(cooling <= 0)
        or not np.isfinite([mean, sigma]).all()
        or sigma <= 0
        or order < 2
    ):
        raise ValueError("invalid first-passage rate inputs")
    r = np.log10(mass / cooling)
    record = np.maximum.accumulate(r, axis=1)
    rows, left = np.nonzero(r[:, 1:] > record[:, :-1])
    lo, hi = record[rows, left], r[rows, left + 1]
    denominator = r[rows, left + 1] - r[rows, left]
    if max_age_myr is not None:
        if not np.isfinite(max_age_myr) or max_age_myr <= 0:
            raise ValueError("max_age_myr must be finite and positive or None")
        fraction = np.clip(
            (t[rows, -1] - max_age_myr / 1000 - t[rows, left])
            / (t[rows, left + 1] - t[rows, left]),
            0,
            1,
        )
        lo = np.maximum(lo, r[rows, left] + fraction * denominator)
        # Test the time intersection explicitly: r_left + (r_right-r_left)
        # can round below r_right, leaving a spurious threshold interval for
        # a birth segment wholly older than the stellar-age window.
        keep = (hi > lo) & (t[rows, left + 1] > t[rows, -1] - max_age_myr / 1000)
        rows, left, lo, hi, denominator = (v[keep] for v in (rows, left, lo, hi, denominator))
    width = hi - lo
    nodes, weights = np.polynomial.legendre.leggauss(order)
    values = np.zeros(len(rows))
    logmass = np.log(mass)
    for node, weight in zip(nodes, weights, strict=True):
        logq = lo + (node + 1) * width / 2
        fraction = (logq - r[rows, left]) / denominator
        birth = t[rows, left] + fraction * (t[rows, left + 1] - t[rows, left])
        mb = np.exp(
            logmass[rows, left] + fraction * (logmass[rows, left + 1] - logmass[rows, left])
        )
        pdf = np.exp(-0.5 * ((logq - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        values += weight * width / 2 * pdf * mb * kernel.rate((t[rows, -1] - birth) * 1000)
    resolved = np.bincount(rows, weights=values, minlength=len(t))
    # Censored burst ages are in [tfinal-tstart, tfinal]. Bound the instantaneous
    # rate over this interval, rather than reusing a lifetime photon yield.
    age_min = (t[:, -1] - t[:, 0]) * 1000
    age_max = t[:, -1] * 1000
    if max_age_myr is not None:
        age_max = np.minimum(age_max, max_age_myr)
    allowed = age_min <= age_max
    upper_rate = np.zeros(len(t))
    upper_rate[allowed] = np.maximum(kernel.rate(age_min[allowed]), kernel.rate(age_max[allowed]))
    for age, rate in zip(kernel.age_myr, kernel.rate_per_msun, strict=True):
        upper_rate = np.where(
            allowed & (age >= age_min) & (age <= age_max),
            np.maximum(upper_rate, rate),
            upper_rate,
        )
    shift = sigma**2 * np.log(10)
    q_truncated_mean = np.exp(mean * np.log(10) + 0.5 * (sigma * np.log(10)) ** 2) * ndtr(
        (r[:, 0] - mean - shift) / sigma
    )
    upper_extra = cooling[:, 0] * q_truncated_mean * upper_rate
    return resolved, upper_extra


def rate_cell(task):
    iz, im, z, mass_final = task
    cfg, cosmo, astro, k2, k3 = source.STATE
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    chunks, errors = [], []
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
        t, m, active, sfr = [
            tracks[k].reshape(shape) for k in ("t_gyr", "Mh", "active_flag", "SFR")
        ]
        cooling = compute_atomic_cooling_mass_msun(tracks["z"].reshape(shape), cosmology=cosmo)
        r2 = cfg.fesc_popii * rate_from_sfh(
            t, sfr, active, k2, order=16, max_age_myr=cfg.max_lookback_myr
        )
        r3, upper = threshold_averaged_rate(
            t,
            m,
            cooling,
            k3,
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            order=16,
            max_age_myr=cfg.popiii_max_age_myr,
        )
        check, _ = threshold_averaged_rate(
            t,
            m,
            cooling,
            k3,
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            order=8,
            max_age_myr=cfg.popiii_max_age_myr,
        )
        factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
        chunks.append(np.stack([r2, factor * r3, factor * upper], axis=-1))
        errors.append(factor * abs(r3 - check))
    sample = np.concatenate(chunks)
    if not np.isfinite(sample).all() or np.any(sample < 0):
        raise FloatingPointError("invalid instantaneous source rates")
    return (
        iz,
        im,
        dict(
            mean_rate=sample.mean(axis=0),
            se_rate=sample.std(axis=0, ddof=1) / np.sqrt(cfg.n_tracks),
            rate_covariance_of_mean=np.cov(sample, rowvar=False) / cfg.n_tracks,
            quadrature_error=np.concatenate(errors).mean(),
        ),
    )
