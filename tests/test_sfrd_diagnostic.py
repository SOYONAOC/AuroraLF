"""Analytic mass-formation checks for the SFRD diagnostic."""

import numpy as np
from scipy.special import ndtr

from scripts.analysis.build_current_sfrd import burst_mass, recent_sfr


def test_linear_sfr_average_in_clipped_interval():
    t = np.array([[0.0, 0.05, 0.1]])
    # SFR=100*t; the average over t=0.09..0.1 is 9.5 Msun/yr.
    np.testing.assert_allclose(recent_sfr(t, 100 * t, np.ones_like(t, dtype=bool), 10), [9.5])


def test_burst_mass_matches_gaussian_probability():
    t = np.array([[0.0, 0.05, 0.1]])
    m = np.full_like(t, 10.0)
    cooling = m / 10.0 ** np.array([[-2.0, 0.0, 2.0]])
    expected = 10 * (ndtr(2.0) - ndtr(1.6))
    np.testing.assert_allclose(burst_mass(t, m, cooling, 0.0, 1.0, 10.0), [expected], rtol=1e-12)


def test_old_burst_not_retriggered_in_recent_window():
    t = np.array([[0.0, 0.05, 0.1]])
    m = np.full_like(t, 10.0)
    cooling = m / 10.0 ** np.array([[-2.0, 2.0, 1.0]])
    np.testing.assert_array_equal(burst_mass(t, m, cooling, 0.0, 1.0, 10.0), [0.0])
