"""Shared UVLF and ionizing-source workers for Pop III -> II onset models."""

import numpy as np
from scipy.special import ndtr, ndtri

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.experiments import ionizing_sources as source
from auroralf.experiments.ionizing_audit import rate_from_sfh
from auroralf.experiments.ionizing_rates import RateConfig, threshold_averaged_rate
from auroralf.experiments.popii_transition import BirthIntegral, averaged_popii, record_at
from auroralf.experiments.random_q import burst_light
from auroralf.mah import generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import compute_sfr_from_tracks
from auroralf.ssp import load_popiii_uv_luminosity_table, load_uv1600_table
from auroralf.ssp.ionizing import IonizingKernel
from auroralf.uvlf import uv_luminosity_to_muv

VARIANTS = ("baseline", "delay0", "delay30")
DELAYS_MYR = {"baseline": None, "delay0": 0.0, "delay30": 30.0}


def validate_variants(variants):
    if (
        not isinstance(variants, (list, tuple))
        or not variants
        or any(not isinstance(v, str) or v not in DELAYS_MYR for v in variants)
        or len(set(variants)) != len(variants)
    ):
        raise ValueError("variants must be unique selections from baseline, delay0, delay30")
    return tuple(variants)


def initialize(model, popiii_uv=None, variants=("delay0",)):
    global UV, SELECTED
    SELECTED = validate_variants(variants)
    source.initialize_worker(RateConfig(**model))
    if popiii_uv is not None:
        if model["max_lookback_myr"] != 100 or model["popiii_max_age_myr"] != 100:
            raise ValueError("UV onset strata require the validated 100 Myr SSP windows")
        a2, k2 = load_uv1600_table(model["popii_ssp"], wavelength_a=1500)
        a3, k3 = load_popiii_uv_luminosity_table(popiii_uv)
        UV = IonizingKernel(a2, k2), a3, k3


def histories(z, im, final_mass):
    cfg, cosmo, astro, _, _ = source.STATE
    dt = (astro.age(z).value - astro.age(cfg.z_start).value) / (cfg.n_grid - 1)
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
            Mh_final=final_mass,
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
        t, m, active, sfr, zg = [
            tracks[k].reshape(cfg.track_chunk, -1)
            for k in ("t_gyr", "Mh", "active_flag", "SFR", "z")
        ]
        yield start, t, m, active, sfr, compute_atomic_cooling_mass_msun(zg, cosmology=cosmo)


def rate_cell(task):
    iz, im, z, final_mass = task
    cfg, cosmo, _, k2, k3 = source.STATE
    pieces, extras, errors = [], [], []
    for _, t, m, active, sfr, cool in histories(z, im, final_mass):
        ii, bounds, qerr = [], [], []
        for name in SELECTED:
            delay = DELAYS_MYR[name]
            if delay is None:
                ii.append(rate_from_sfh(t, sfr, active, k2, max_age_myr=cfg.max_lookback_myr))
                bounds.append(np.zeros(len(t)))
                coarse = rate_from_sfh(
                    t, sfr, active, k2, max_age_myr=cfg.max_lookback_myr, order=8
                )
                qerr.append(abs(ii[-1] - coarse))
                continue
            fine = averaged_popii(
                t,
                sfr,
                active,
                m,
                cool,
                k2,
                delay,
                cfg.q_log10_mean,
                cfg.q_log10_sigma,
                max_age_myr=cfg.max_lookback_myr,
            )
            coarse = averaged_popii(
                t,
                sfr,
                active,
                m,
                cool,
                k2,
                delay,
                cfg.q_log10_mean,
                cfg.q_log10_sigma,
                max_age_myr=cfg.max_lookback_myr,
                order=8,
            )
            ii.append(fine[:, 0])
            bounds.append(fine[:, 1] - fine[:, 0])
            qerr.append(abs(fine[:, 0] - coarse[:, 0]))
        iii, upper = threshold_averaged_rate(
            t,
            m,
            cool,
            k3,
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            order=16,
            max_age_myr=cfg.popiii_max_age_myr,
        )
        coarse, _ = threshold_averaged_rate(
            t,
            m,
            cool,
            k3,
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            order=8,
            max_age_myr=cfg.popiii_max_age_myr,
        )
        factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
        pieces.append(
            np.stack(
                [np.stack((cfg.fesc_popii * v, factor * iii, factor * upper), axis=-1) for v in ii],
                axis=1,
            )
        )
        extras.append(cfg.fesc_popii * np.stack(bounds, axis=-1))
        errors.append(
            np.column_stack((factor * abs(iii - coarse), cfg.fesc_popii * np.stack(qerr, axis=-1)))
        )
    samples, extra, error = np.concatenate(pieces), np.concatenate(extras), np.concatenate(errors)
    if not np.isfinite(samples).all() or np.any(samples < 0):
        raise FloatingPointError("invalid gated source samples")
    return (
        iz,
        im,
        dict(
            mean_rate=samples.mean(0),
            se_rate=samples.std(0, ddof=1) / np.sqrt(cfg.n_tracks),
            rate_covariance_of_mean=np.stack(
                [
                    np.cov(samples[:, i, :], rowvar=False) / cfg.n_tracks
                    for i in range(len(SELECTED))
                ]
            ),
            quadrature_error=error[:, 0].mean(),
            popii_quadrature_error=error[:, 1:].mean(0),
            popii_censored_upper_extra=extra.mean(0),
        ),
    )


def uv_proposals(t, m, cool, uniforms, mean, sigma, a3, k3, fb, epsilon):
    """Eight disjoint strata, separating old, censored and untriggered bursts."""
    ratio = np.log10(m / cool)
    ages = np.array([0.0, 3.0, 10.0, 30.0, 100.0, 130.0])
    if np.any((t[:, -1] - t[:, 0]) * 1000 <= ages[-1]):
        raise ValueError("UV histories must resolve the 130 Myr transition window")
    cdf = ndtr((record_at(t, ratio, t[:, -1, None] - ages[None, :] / 1000) - mean) / sigma)
    initial = ndtr((ratio[:, 0] - mean) / sigma)
    # Recent five strata, old resolved, left censored, not yet triggered.
    lo = np.column_stack((cdf[:, 1:], initial, np.zeros(len(t)), cdf[:, 0]))
    hi = np.column_stack((cdf[:, :-1], cdf[:, -1], initial, np.ones(len(t))))
    p = hi - lo
    if np.any(p < 0) or uniforms.shape != p.shape or np.any((uniforms <= 0) | (uniforms >= 1)):
        raise ValueError("invalid conditional UV proposals")
    light, birth = np.zeros_like(p), np.full_like(p, np.inf)
    censored = np.zeros(p.shape, bool)
    censored[:, 6] = True
    for j in range(6):
        use = p[:, j] > 0
        quantile = lo[use, j] + uniforms[use, j] * p[use, j]
        quantile = np.clip(quantile, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        logq = mean + sigma * ndtri(quantile)
        event = burst_light(t[use], m[use], cool[use], logq, a3, k3, fb, 100.0)
        if np.any(event["status"] != 1):
            raise FloatingPointError("conditional crossing unresolved")
        light[use, j] = epsilon * event["popiii_per_efficiency"]
        birth[use, j] = event["burst_time_gyr"]
    birth[:, 6] = t[:, 0]  # Latest possible date, explicitly a lower-bound gate.
    np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=2e-15)
    return light, p, birth, censored


def uv_cell(task):
    im, z, final_mass, edges = task
    cfg, cosmo, _, _, _ = source.STATE
    k2, a3, k3 = UV
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0x5452414E, im]))
    uniforms = rng.uniform(np.nextafter(0.0, 1.0), 1.0, (cfg.n_tracks, 8))
    hist = np.zeros((len(SELECTED), 3, len(edges) - 1))
    diagnostics = np.zeros(len(SELECTED) + 2)
    for start, t, m, active, sfr, cool in histories(z, im, final_mass):
        p3, probability, birth, _ = uv_proposals(
            t,
            m,
            cool,
            uniforms[start : start + cfg.track_chunk],
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            a3,
            k3,
            cosmo.omega_b / cosmo.omega_m,
            cfg.epsilon_b,
        )
        integral = BirthIntegral(t, sfr, active, k2)
        base = np.broadcast_to(integral.after(np.full((len(t), 1), -np.inf)), birth.shape)
        p2s = [
            base if DELAYS_MYR[name] is None else integral.after(birth + DELAYS_MYR[name] / 1000)
            for name in SELECTED
        ]
        for v, p2 in enumerate(p2s):
            if np.any(p2 > base * (1 + 1e-12)):
                raise FloatingPointError("gate increased Pop II light")
            for c, lum in enumerate((p2, p3, p2 + p3)):
                hist[v, c] += np.histogram(
                    uv_luminosity_to_muv(lum).ravel(), edges, weights=probability.ravel()
                )[0]
        diagnostics[: len(SELECTED)] += np.array([(v * probability).sum() for v in p2s])
        diagnostics[-2] += (p3 * probability).sum()
        diagnostics[-1] += probability[:, 6].sum()
    return im, hist / cfg.n_tracks, diagnostics / cfg.n_tracks
