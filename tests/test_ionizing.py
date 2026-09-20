"""Photon units, cumulative-age integration, and SFH accounting."""

import numpy as np
import pytest
from scipy.integrate import quad

from auroralf.ssp.ionizing import (
    SECONDS_PER_MYR,
    IonizingKernel,
    cumulative_photons_from_sfh,
    load_bpass_ionizing_kernel,
    load_popiii_ionizing_kernel,
)


def test_constant_rate_counts_seconds_and_birth_mass():
    kernel = IonizingKernel([1, 100], [2e47, 2e47])
    np.testing.assert_allclose(
        kernel.yield_photons([0, 0.1, 1, 50]),
        np.array([0, 0.1, 1, 50]) * SECONDS_PER_MYR * 2e47,
    )
    # Constant 3 Msun/yr from 0 to 10 Myr: mean stellar age is 5 Myr.
    t = np.array([[0, 0.001, 0.01]])
    photons = cumulative_photons_from_sfh(
        t, np.full_like(t, 3), np.ones_like(t, bool), kernel
    )
    np.testing.assert_allclose(
        photons, 3 * 1e7 * 5 * SECONDS_PER_MYR * 2e47, rtol=1e-14
    )


def test_exact_log_age_integral_matches_independent_quadrature():
    kernel = IonizingKernel([0.01, 1, 2, 7, 100], [8e47, 6e47, 5e47, 1e44, 0])
    for age in [0, 0.003, 0.01, 0.7, 1, 1.5, 7, 80, 100]:
        integral = quad(
            lambda x: kernel.rate(x) / 1e47,
            0,
            age,
            points=kernel.age_myr[kernel.age_myr < age],
            epsabs=1e-10,
        )[0]
        np.testing.assert_allclose(
            kernel.yield_photons(age), integral * 1e47 * SECONDS_PER_MYR, rtol=2e-12
        )
    assert np.all(np.diff(kernel.yield_photons(np.linspace(0, 100, 10001))) >= 0)


def test_invalid_ages_and_inactive_sfh():
    kernel = IonizingKernel([1, 100], [1e47, 0])
    for age in [-1, np.nan, 101]:
        with pytest.raises(ValueError, match="coverage"):
            kernel.yield_photons(age)
    t = np.array([[0, 0.01]])
    np.testing.assert_array_equal(
        cumulative_photons_from_sfh(t, t + 1, t < 0, kernel), [0]
    )
    with pytest.raises(ValueError, match="history"):
        cumulative_photons_from_sfh(t, t - 1, t > -1, kernel)


def test_real_matching_ssps():
    # These real source files are required project inputs; never skip or synthesize them.
    base = "external_data/ssp_spectra/"
    k2 = load_bpass_ionizing_kernel(
        base + "bpass_byrne23_imf135_300/BASEL/"
        "spectra-bin-imf135_300.BASEL.z001.a+00.dat"
    )
    k3 = load_popiii_ionizing_kernel(
        base + "schaerer2010_pop3/pop3_ge0_logE_500_001_is5.20"
    )
    line = np.loadtxt(base + "schaerer2010_pop3/pop3_ge0_logE_500_001_is5.22")
    np.testing.assert_allclose(k3.rate_per_msun, 10.0 ** line[:, 1], rtol=0, atol=1e-50)
    assert k2.age_myr[0] == 1 and k3.age_myr[0] == 0.01
    assert k2.yield_photons(100) > 0 and k3.yield_photons(100) > 0
