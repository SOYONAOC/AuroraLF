"""Separate cumulative Pop II/III source budgets from current McBride histories.

This first adapter uses a final-halo main-branch closure. Other progenitors
are not added or inferred from the HMF. Left-censored Pop III events have
explicit lower/upper photon budgets, not invented burst times.
"""

from dataclasses import dataclass

import numpy as np

from auroralf.constants import KM_PER_MPC, SECONDS_PER_GYR
from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import compute_sfr_from_tracks
from auroralf.ssp.ionizing import (
    cumulative_photons_from_sfh,
    load_bpass_ionizing_kernel,
    load_popiii_ionizing_kernel,
)

from .random_q import draw_logq, first_crossing


@dataclass(frozen=True)
class SourceConfig:
    n_tracks: int
    track_chunk: int
    n_grid: int
    seed: int
    z_start: float
    epsilon_b: float
    fesc_popii: float
    fesc_popiii: float
    q_log10_mean: float
    q_log10_sigma: float
    h: float
    omega_m: float
    omega_b: float
    popii_ssp: str
    popiii_ssp: str

    def __post_init__(self):
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) < 1
            for key in ("n_tracks", "track_chunk", "n_grid", "seed")
        ):
            raise ValueError("invalid source sampling dimensions")
        if self.n_tracks < 2 or self.n_grid < 2 or self.n_tracks % self.track_chunk:
            raise ValueError("invalid track chunks")
        for key in ("epsilon_b", "fesc_popii", "fesc_popiii"):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key)
        if (
            not np.isfinite(self.q_log10_mean)
            or not np.isfinite(self.q_log10_sigma)
            or self.q_log10_sigma <= 0
        ):
            raise ValueError("invalid q distribution")
        if not np.isfinite(self.z_start) or self.z_start <= 0:
            raise ValueError("invalid start redshift")
        self.cosmology()

    def cosmology(self):
        return Cosmology(
            h0=100 * self.h * SECONDS_PER_GYR / KM_PER_MPC,
            omega_m=self.omega_m,
            omega_b=self.omega_b,
            omega_lambda=1 - self.omega_m,
        )


def initialize_worker(config):
    global STATE
    from astropy.cosmology import FlatLambdaCDM

    cosmo = config.cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    STATE = (
        config,
        cosmo,
        astro,
        load_bpass_ionizing_kernel(config.popii_ssp),
        load_popiii_ionizing_kernel(config.popiii_ssp),
    )


def sample_mass_redshift(task):
    """Return unweighted all-history means and sampling diagnostics at fixed M,z."""
    iz, im, z, mass_final = task
    cfg, cosmo, astro, k2, k3 = STATE
    if (
        not np.isfinite(z)
        or not 0 <= z < cfg.z_start
        or not np.isfinite(mass_final)
        or mass_final <= 0
    ):
        raise ValueError("invalid source-table coordinate")
    time_final = astro.age(z).value
    dt = (time_final - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
    # This also ensures the left-censored upper bound is supported by the SSP.
    full_age_yield = k3.yield_photons(time_final * 1000)
    qlog = draw_logq(cfg.seed, im, cfg.n_tracks, cfg.q_log10_mean, cfg.q_log10_sigma)
    samples, statuses, initial_active = [], [], []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        block_seed = int(
            np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(block_seed, redshift=z, mass_index=im)
        histories = generate_halo_histories(
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
            histories.tracks,
            cosmology=cosmo,
            enable_time_delay=True,
            regular_convolution_backend="direct",
        )
        shape = (cfg.track_chunk, -1)
        t, m, active = (tracks[key].reshape(shape) for key in ("t_gyr", "Mh", "active_flag"))
        cooling = compute_atomic_cooling_mass_msun(tracks["z"].reshape(shape), cosmology=cosmo)
        q = qlog[start : start + cfg.track_chunk]
        status, tb, mb = first_crossing(t, m, cooling, q)
        p2 = cfg.fesc_popii * cumulative_photons_from_sfh(
            t, tracks["SFR"].reshape(shape), active, k2
        )
        p3 = np.zeros(cfg.track_chunk)
        resolved = status == 1
        factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
        p3[resolved] = (
            factor * mb[resolved] * k3.yield_photons((t[resolved, -1] - tb[resolved]) * 1000)
        )
        upper_extra = np.zeros(cfg.track_chunk)
        censored = status == 2
        # At a continuous threshold crossing Mburst=q*Mcool(zburst). Atomic
        # cooling mass decreases towards earlier z, so q*Mcool(z_start) is
        # an upper bound even for a nonmonotonic McBride mass history.
        upper_extra[censored] = factor * 10.0 ** q[censored] * cooling[censored, 0] * full_age_yield
        samples.append(np.stack([p2, p3, upper_extra], axis=-1))
        statuses.append(status)
        initial_active.append(active[:, 0])
    sample = np.concatenate(samples)
    status = np.concatenate(statuses)
    if not np.isfinite(sample).all() or np.any(sample < 0):
        raise FloatingPointError("invalid source sample")
    return (
        iz,
        im,
        dict(
            mean_photons=sample.mean(axis=0),
            se_photons=sample.std(axis=0, ddof=1) / np.sqrt(cfg.n_tracks),
            half_means_photons=np.stack([s.mean(axis=0) for s in np.array_split(sample, 2)]),
            status_counts=np.bincount(status, minlength=3),
            initial_popii_active_count=int(np.count_nonzero(np.concatenate(initial_active))),
        ),
    )
