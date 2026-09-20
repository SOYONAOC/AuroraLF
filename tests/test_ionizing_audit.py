import numpy as np
import pytest

from auroralf.experiments.ionizing_audit import rate_from_sfh
from auroralf.ssp.ionizing import IonizingKernel


def test_current_rate_counts_formed_mass_and_seconds_correctly():
    # 3 Msun/yr over 10 Myr with 2 photons/s/Msun gives 6e7 photons/s.
    t = np.array([[0.0, 0.002, 0.01]])
    kernel = IonizingKernel(np.array([1.0, 100.0]), np.array([2.0, 2.0]))
    answer = rate_from_sfh(t, np.full_like(t, 3.0), np.ones_like(t, bool), kernel)
    np.testing.assert_allclose(answer, [6e7], rtol=1e-14)
    np.testing.assert_array_equal(rate_from_sfh(t, t * 0 + 3, np.zeros_like(t, bool), kernel), [0])


def test_current_rate_agrees_with_age_integral_for_constant_sfr():
    kernel = IonizingKernel(np.array([1.0, 3.0, 10.0, 100.0]), np.array([5.0, 3.0, 1.0, 0.0]))
    # Put the SSP age knots at segment boundaries; exact age integral is independent.
    t = np.array([[0.0, 0.007, 0.009, 0.01]])
    result = rate_from_sfh(t, t * 0 + 2, np.ones_like(t, bool), kernel)
    expected = 2 * kernel.yield_photons(10) / (365.25 * 86400)
    np.testing.assert_allclose(result, [expected], rtol=1e-11)
    with pytest.raises(ValueError):
        rate_from_sfh(t[:, ::-1], t * 0 + 2, np.ones_like(t, bool), kernel)
