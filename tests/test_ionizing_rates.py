import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from auroralf.experiments.ionizing_audit import rate_from_sfh
from auroralf.experiments.ionizing_rates import threshold_averaged_rate
from auroralf.experiments.random_q import first_crossing
from auroralf.ssp.ionizing import IonizingKernel


def test_integrated_first_passage_matches_independent_quadrature():
    t = np.array([[0.05, 0.06, 0.07, 0.08]])
    m = np.array([[1.0, 4.0, 2.0, 8.0]])
    c = np.ones_like(m)
    kernel = IonizingKernel(np.array([1.0, 10.0, 100.0]), np.array([8.0, 1.0, 0.0]))
    got, upper = threshold_averaged_rate(t, m, c, kernel, 0.5, 1.5, order=64)

    def integrand(q):
        status, birth, mass = first_crossing(t, m, c, np.array([q]))
        assert status[0] == 1
        return (
            mass[0]
            * kernel.rate((t[0, -1] - birth[0]) * 1000)
            * np.exp(-0.5 * ((q - 0.5) / 1.5) ** 2)
            / (1.5 * np.sqrt(2 * np.pi))
        )

    expected = quad(integrand, 0.0, np.log10(8), points=[np.log10(4)], epsabs=1e-8)[0]
    np.testing.assert_allclose(got[0], expected, rtol=2e-4)
    assert 0 <= upper[0] < 8


def test_constant_mass_constant_rate_probability_and_no_repeat_crossing():
    t = np.array([[0.05, 0.06, 0.07, 0.08]])
    m = np.full_like(t, 10.0)
    c = np.array([[10.0, 1.0, 10.0, 1.0]])
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    got, _ = threshold_averaged_rate(t, m, c, k, 0.0, 1.0, order=16)
    np.testing.assert_allclose(got, 30 * (ndtr(1) - ndtr(0)), rtol=1e-12)


def test_old_censored_sources_age_out_instead_of_using_lifetime_yield():
    t = np.array([[0.05, 0.1, 0.2]])
    m = np.ones_like(t)
    k = IonizingKernel(np.array([1.0, 5.0, 10.0, 300.0]), np.array([10.0, 1.0, 0.0, 0.0]))
    resolved, upper = threshold_averaged_rate(t, m, m, k, 0.0, 1.0)
    assert resolved[0] == 0 and upper[0] == 0


def test_popii_window_clips_a_birth_interval_without_evaluating_old_ssp():
    t = np.array([[0.0, 0.15, 0.2]])
    sfr = 10 * t
    # Old stars exceed this kernel's support and must never be evaluated.
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    got = rate_from_sfh(t, sfr, np.ones_like(t, dtype=bool), k, max_age_myr=100)
    # Linear SFR has mean 1.5 Msun/yr over the final 100 Myr.
    np.testing.assert_allclose(got, 1.5 * 1e8 * 3, rtol=1e-14)


@pytest.mark.parametrize("window,logq_min", [(100.0, 1.0), (75.0, 1.25)])
def test_popiii_window_clips_threshold_probability_at_exact_birth_time(window, logq_min):
    t = np.array([[0.01, 0.11, 0.21]])
    m = np.full_like(t, 10.0)
    c = np.array([[10.0, 1.0, 0.1]])
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    resolved, upper = threshold_averaged_rate(t, m, c, k, 0, 1, order=32, max_age_myr=window)
    np.testing.assert_allclose(resolved, 30 * (ndtr(2) - ndtr(logq_min)), rtol=1e-13)
    assert upper[0] == 0


def test_old_popiii_burst_is_not_retriggered_inside_recent_window():
    t = np.array([[0.01, 0.06, 0.2, 0.21]])
    m = np.full_like(t, 10.0)
    c = np.array([[10.0, 1.0, 10.0, 1.0]])
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    resolved, upper = threshold_averaged_rate(t, m, c, k, 0, 1, max_age_myr=100)
    np.testing.assert_array_equal(resolved, [0])
    np.testing.assert_array_equal(upper, [0])


def test_old_crossing_roundoff_cannot_leave_a_spurious_young_interval():
    t = np.array([[0.01, 0.02, 1.1]])
    m = 10.0 ** np.array([[-10.0, -0.9, -0.9]])
    k = IonizingKernel(np.array([1.0, 100.0]), np.ones(2))
    resolved, upper = threshold_averaged_rate(t, m, np.ones_like(m), k, 0, 1, max_age_myr=100)
    np.testing.assert_array_equal(resolved, [0])
    np.testing.assert_array_equal(upper, [0])


def test_window_longer_than_universe_preserves_resolved_and_censored_rates():
    t = np.array([[0.01, 0.02, 0.03]])
    m = np.array([[1.0, 2.0, 3.0]])
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    got = threshold_averaged_rate(t, m, np.ones_like(m), k, 0, 1, max_age_myr=100)
    expected = threshold_averaged_rate(t, m, np.ones_like(m), k, 0, 1)
    np.testing.assert_allclose(got, expected, rtol=1e-14)


@pytest.mark.parametrize("window", [0, -1, np.nan, np.inf])
def test_invalid_age_window_rejected(window):
    t = np.array([[0.01, 0.02]])
    m = np.ones_like(t)
    k = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    with pytest.raises(ValueError, match="max_age_myr"):
        threshold_averaged_rate(t, m, m, k, 0, 1, max_age_myr=window)
    with pytest.raises(ValueError, match="max_age_myr"):
        rate_from_sfh(t, m, m.astype(bool), k, max_age_myr=window)


def production_config(**overrides):
    import tomllib
    from pathlib import Path

    from auroralf.experiments.ionizing_rates import RateConfig

    raw = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/experiments/ionizing_rates.toml"
        ).read_text()
    )
    return RateConfig(**(raw["model"] | overrides))


@pytest.mark.parametrize("field", ["max_lookback_myr", "popiii_max_age_myr"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_population_age_limits_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        production_config(**{field: value})


def test_production_popiii_window_changes_only_popiii_rate(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from astropy.cosmology import FlatLambdaCDM

    from auroralf.experiments import ionizing_rates as rates

    cfg = production_config(n_tracks=2, track_chunk=2, n_grid=3)
    assert cfg.max_lookback_myr == 100
    assert cfg.popiii_max_age_myr == 6
    t = np.tile([0.01, 0.015, 0.03], (2, 1))
    mass = np.full_like(t, 10.0)
    cooling = np.tile([10.0, 1.0, 0.1], (2, 1))
    history = dict(
        t_gyr=t.ravel(),
        Mh=mass.ravel(),
        z=np.zeros(t.size),
        active_flag=np.ones(t.size, dtype=bool),
        SFR=np.ones(t.size),
    )
    kernel = IonizingKernel(np.array([1.0, 100.0]), np.array([3.0, 3.0]))
    monkeypatch.setattr(
        rates,
        "generate_halo_histories",
        lambda **kwargs: SimpleNamespace(tracks=history),
    )
    monkeypatch.setattr(rates, "compute_sfr_from_tracks", lambda tracks, **kwargs: tracks)
    monkeypatch.setattr(rates, "compute_atomic_cooling_mass_msun", lambda *args, **kwargs: cooling)
    cosmo = cfg.cosmology()
    astro = FlatLambdaCDM(H0=100 * cfg.h, Om0=cfg.omega_m, Ob0=cfg.omega_b)
    monkeypatch.setattr(rates.source, "STATE", (cfg, cosmo, astro, kernel, kernel), raising=False)
    short = rates.rate_cell((0, 0, 6.0, 10.0))[2]
    monkeypatch.setattr(
        rates.source,
        "STATE",
        (replace(cfg, popiii_max_age_myr=100), cosmo, astro, kernel, kernel),
    )
    long = rates.rate_cell((0, 0, 6.0, 10.0))[2]
    np.testing.assert_equal(short["mean_rate"][0], long["mean_rate"][0])
    # The last interval crosses log10(q)=1..2 over 15 Myr. Its final 6 Myr
    # therefore integrates only log10(q)=1.6..2, preserving the full-history records.
    factor = cfg.epsilon_b * cosmo.omega_b / cosmo.omega_m * cfg.fesc_popiii
    expected = (
        factor
        * 30
        * (
            ndtr((2 - cfg.q_log10_mean) / cfg.q_log10_sigma)
            - ndtr((1.6 - cfg.q_log10_mean) / cfg.q_log10_sigma)
        )
    )
    np.testing.assert_allclose(short["mean_rate"][1], expected, rtol=1e-12)
    assert 0 < short["mean_rate"][1] < long["mean_rate"][1]
    assert short["mean_rate"][2] == 0 < long["mean_rate"][2]
    assert short["quadrature_error"] < 1e-12
