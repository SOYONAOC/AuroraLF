"""Analytic and independent-integration checks of Pop II transition gates."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from auroralf.experiments.ionizing_audit import rate_from_sfh
from auroralf.experiments.popii_transition import (
    BirthIntegral,
    averaged_popii,
    record_at,
    transition_probability,
)
from auroralf.experiments.transition_workers import uv_proposals
from auroralf.ssp.ionizing import IonizingKernel


def setup_history():
    t = np.linspace(0.1, 0.3, 21)[None, :]
    sfr = np.ones_like(t) * 2
    active = np.ones_like(t, dtype=bool)
    kernel = IonizingKernel(np.array([0.001, 1000.0]), np.array([3.0, 3.0]))
    return t, sfr, active, kernel


def test_exact_onset_and_no_renormalization():
    t, sfr, active, kernel = setup_history()
    integral = BirthIntegral(t, sfr, active, kernel)
    onset = np.array([[-np.inf, 0.2037, 0.2471, 0.3, np.inf]])
    expected = 6e9 * np.maximum(0, 0.3 - np.clip(onset, 0.2, 0.3))
    np.testing.assert_allclose(integral.after(onset), expected, rtol=1e-14)
    np.testing.assert_allclose(
        integral.after(onset)[:, 0], rate_from_sfh(t, sfr, active, kernel, max_age_myr=100)
    )


def test_linear_sfr_exact_partial_integral():
    t, sfr, active, kernel = setup_history()
    sfr[:] = t
    onset = np.array([[0.2123, 0.2874]])
    expected = 3e9 * (0.3**2 - onset**2) / 2
    np.testing.assert_allclose(
        BirthIntegral(t, sfr, active, kernel).after(onset), expected, rtol=1e-13
    )


def test_running_record_not_later_recrossing():
    t = np.array([[0.1, 0.2, 0.3, 0.4]])
    ratio = np.array([[0.0, 2.0, 0.0, 3.0]])
    q = np.array([[0.15, 0.25, 0.35, 0.39]])
    np.testing.assert_allclose(record_at(t, ratio, q), [[1.0, 2.0, 2.0, 2.7]])


def test_censored_delay_bounds_and_no_pre_start_burst_date():
    t, sfr, active, kernel = setup_history()
    ratio = np.zeros_like(t)
    birth = np.array([[0.11, 0.129, 0.131, 0.2]])
    lower, upper = transition_probability(t, ratio, birth, 30, 0, 1)
    np.testing.assert_allclose(lower, [[0, 0, 0.5, 0.5]])
    np.testing.assert_allclose(upper, [[0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(*transition_probability(t, ratio, birth, 0, 0, 1))


@pytest.mark.parametrize("delay", [0.0, 30.0, 500.0])
def test_average_matches_independent_normal_cdf_integral(delay):
    t, sfr, active, kernel = setup_history()
    mass = 10 ** ((t - 0.1) * 20 - 2)
    cool = np.ones_like(t)
    result = averaged_popii(t, sfr, active, mass, cool, kernel, delay, 0, 1)

    def f(birth, upper):
        cutoff = birth - delay / 1000
        if cutoff < 0 or (cutoff < 0.1 and not upper):
            return 0
        r = max(-2.0, (cutoff - 0.1) * 20 - 2)
        return 6e9 * ndtr(r)

    expected = [
        quad(
            lambda b: f(b, upper),
            0.2,
            0.3,
            epsabs=1e-3,
            points=[0.1 + delay / 1000] if 0.2 < 0.1 + delay / 1000 < 0.3 else None,
        )[0]
        for upper in (False, True)
    ]
    np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=1e-6)


def test_gate_decreases_light_and_delay_decreases_again():
    t, sfr, active, kernel = setup_history()
    mass = 10 ** ((t - 0.1) * 20 - 2)
    cool = np.ones_like(t)
    base = rate_from_sfh(t, sfr, active, kernel, max_age_myr=100)
    v0 = averaged_popii(t, sfr, active, mass, cool, kernel, 0, 0, 1)
    v30 = averaged_popii(t, sfr, active, mass, cool, kernel, 30, 0, 1)
    assert np.all(v30 <= v0) and np.all(v0 <= base[:, None])


def test_all_uv_strata_normalized_and_untriggered_dark():
    t, sfr, active, kernel = setup_history()
    mass = 10 ** ((t - 0.1) * 20 - 2)
    cool = np.ones_like(t)
    p3, p, birth, censored = uv_proposals(
        t, mass, cool, np.full((1, 8), 0.5), 0, 1, np.array([0.001, 100.0]), np.ones(2), 0.16, 0.03
    )
    assert np.all(p > 0)
    np.testing.assert_allclose(p.sum(1), 1)
    assert np.isinf(birth[0, -1]) and p3[0, -1] == 0
    assert censored[0, 6] and birth[0, 6] == t[0, 0]
    assert np.all(p3[:, 4:] == 0)
    gated = BirthIntegral(t, sfr, active, kernel).after(birth)
    assert gated[0, -1] == 0


def test_invalid_delay_and_nan_onset_rejected():
    t, sfr, active, kernel = setup_history()
    with pytest.raises(ValueError):
        transition_probability(t, t, t, -1, 0, 1)
    with pytest.raises(ValueError):
        BirthIntegral(t, sfr, active, kernel).after(np.array([[np.nan]]))
