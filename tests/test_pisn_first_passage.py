"""Independent integral and first-crossing regression for rare-event rates."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from auroralf.experiments.pisn import load_marigo_kernel
from auroralf.experiments.pisn_first_passage import threshold_averaged_pisn_rate


def test_narrow_delay_resolved_on_coarse_grid():
    kernel = load_marigo_kernel("external_data/pisn/marigo2003_nonrotating_lifetimes.csv")
    t = np.array([[0.0, 0.010]])
    m = np.full_like(t, 1e8)
    c = 1e8 / 10 ** np.array([[-1.0, 1.0]])
    got = threshold_averaged_pisn_rate(t, m, c, kernel)[0]
    # Direct age integration: logq = 1-age/5, |dlogq/dage| = 1/5.
    knots = np.sort(kernel.lifetime(np.array([140.0, 250.0, 260.0])))
    expected = sum(
        quad(
            lambda a: 1e8 * kernel.rate(a) * norm.pdf(1 - a / 5, scale=1.5) / 5,
            lo,
            hi,
            epsabs=1e-13,
        )[0]
        for lo, hi in zip(knots[:-1], knots[1:])
    )
    assert got == pytest.approx(expected, rel=1e-10)


def test_prior_record_prevents_second_burst():
    kernel = load_marigo_kernel("external_data/pisn/marigo2003_nonrotating_lifetimes.csv")
    t = np.array([[0.0, 0.005, 0.006, 0.010]])
    m = np.full_like(t, 1e8)
    c = 1e8 / 10 ** np.array([[-1.0, 2.0, -1.0, 1.0]])
    assert threshold_averaged_pisn_rate(t, m, c, kernel)[0] == 0


def test_short_history_is_not_silently_accepted():
    kernel = load_marigo_kernel("external_data/pisn/marigo2003_nonrotating_lifetimes.csv")
    t = np.array([[0.0, 0.002]])
    with pytest.raises(ValueError, match="pre-start"):
        threshold_averaged_pisn_rate(t, np.ones_like(t), np.ones_like(t), kernel)
