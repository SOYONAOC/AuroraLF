"""Normalization, delayed-event conservation and observer-frame rate checks."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad

from auroralf.experiments.pisn import (
    LifetimeKernel,
    LognormalIMF,
    load_marigo_kernel,
    observer_rate_per_deg2,
)


@pytest.fixture
def kernel():
    return load_marigo_kernel("external_data/pisn/marigo2003_nonrotating_lifetimes.csv")


def test_imf_mass_normalization_and_number_yield():
    imf = LognormalIMF()
    assert quad(lambda m: m * imf.number_density(m), 1, 500)[0] == pytest.approx(1)
    eta = quad(imf.number_density, 140, 260)[0]
    assert imf.yield_per_msun() == pytest.approx(eta, rel=1e-9)
    # The 1/M Jacobian matters; integrating the log-space number shape as dN/dM
    # would produce a substantially different progenitor number distribution.
    assert imf.moment(1, 500) == pytest.approx(
        quad(lambda m: np.exp(-(np.log(m / 60) ** 2) / 2) / m, 1, 500)[0]
    )


@pytest.mark.parametrize("window", [0, 0.1, 0.25, 0.5])
def test_delayed_kernel_conserves_number(kernel, window):
    points = sorted(
        set(
            [
                *kernel.delay_bounds_myr,
                *kernel.lifetime_myr,
                *(kernel.delay_bounds_myr + window),
                *(kernel.lifetime_myr + window),
            ]
        )
    )
    integrated = quad(lambda a: kernel.rate(a, window) * 1e6, 0, 10, points=points, epsabs=1e-12)[0]
    assert integrated == pytest.approx(kernel.imf.yield_per_msun(), rel=1e-7)
    assert kernel.cumulative(-1) == pytest.approx(0, abs=1e-15)
    assert kernel.cumulative(10) == pytest.approx(kernel.imf.yield_per_msun())


def test_delays_include_helium_burning(kernel):
    assert kernel.lifetime(250) == pytest.approx(2.4471)
    assert np.all(kernel.rate([0, 1, 4, 100]) == 0)
    hydrogen = load_marigo_kernel(
        "external_data/pisn/marigo2003_nonrotating_lifetimes.csv", hydrogen_only=True
    )
    assert np.all(kernel.delay_bounds_myr > hydrogen.delay_bounds_myr)
    ages = np.linspace(*kernel.delay_bounds_myr, 50)[1:-1]
    derivative = (kernel.cumulative(ages + 1e-7) - kernel.cumulative(ages - 1e-7)) / 0.2
    assert np.allclose(derivative, kernel.rate(ages), rtol=1e-5)


def test_observer_volume_and_time_dilation():
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.30966)
    z = 12.5
    from astropy import units as u

    expected = 2e-5 * cosmo.differential_comoving_volume(z).to_value(u.Mpc**3 / u.deg**2) / (1 + z)
    assert observer_rate_per_deg2(2e-5, z, cosmo) == pytest.approx(expected)


def test_invalid_inputs_fail(kernel):
    with pytest.raises(ValueError):
        LognormalIMF(sigma=0)
    with pytest.raises(ValueError):
        LifetimeKernel(np.array([120, 250, 500]), np.array([3, 2, 4]))
    with pytest.raises(ValueError):
        kernel.lifetime(1000)
    with pytest.raises(ValueError):
        kernel.rate(np.nan)
    with pytest.raises(ValueError):
        kernel.rate(3, -1)
