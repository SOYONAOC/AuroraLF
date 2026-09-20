"""Physical and numerical checks for the reionization optical-depth integral."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c, m_p, sigma_T
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.reionization_calibration import neutral_residuals, thomson_depth

COSMO = dict(h=0.6766, omega_m=0.30966, omega_b=0.04897)


def test_instantaneous_hydrogen_reionization_analytic_integral():
    # Neutral above z=8, fully ionized below; a negligible transition interval.
    z = [20, 8.000001, 8, 6]
    out = thomson_depth(z, [0, 0, 1, 1], hydrogen_mass_fraction=1, dz=0.0002, **COSMO)
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966, Ob0=0.04897)
    nh = cosmo.critical_density0 * cosmo.Ob0 / m_p
    expected = (c * sigma_T * nh / cosmo.H0).to_value(u.dimensionless_unscaled)
    expected *= 2 / (3 * cosmo.Om0) * (np.sqrt(cosmo.Om0 * 9**3 + 1 - cosmo.Om0) - 1)
    np.testing.assert_allclose(out["total"], expected, rtol=1e-6)


def test_helium_and_completion_increase_depth_and_grid_converges():
    z, q = [30, 15, 10, 8, 6.01], [0, 0.01, 0.15, 0.4, 0.93]
    central = thomson_depth(z, q, **COSMO)
    early = thomson_depth(z, q, completion_redshift=6.01, **COSMO)
    early_fine = thomson_depth(z, q, completion_redshift=6.01, dz=0.001, **COSMO)
    late = thomson_depth(z, q, completion_redshift=5, **COSMO)
    fine = thomson_depth(z, q, dz=0.001, **COSMO)
    assert late["total"] < central["total"] < early["total"]
    assert np.all(np.diff(central["tau"]) >= 0)
    np.testing.assert_allclose(central["total"], fine["total"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(early["total"], early_fine["total"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(central["electrons_per_h"][0], 1 + 2 / 12, atol=1e-6)


def test_mass_weighting_matters():
    z = [30, 15, 10, 8, 6.01]
    volume = thomson_depth(z, [0, 0.005, 0.1, 0.4, 1], **COSMO)
    mass = thomson_depth(z, [0, 0.01, 0.2, 0.6, 1], **COSMO)
    assert mass["total"] > volume["total"]


@pytest.mark.parametrize("q", [[0.1, 0.5, 1], [0, np.nan, 1], [0, 0.5, 1.1]])
def test_invalid_history_is_not_silently_completed(q):
    with pytest.raises(ValueError):
        thomson_depth([20, 10, 6], q, **COSMO)


def test_neutral_calibration_excludes_wide_bins_and_outside_simulation():
    point = dict(id="narrow", z=7, z_min=7, z_max=7, xhi=0.4, lower=0.3, upper=0.6, kind="interval")
    wide = dict(point, id="wide", z_min=6.5, z_max=8)
    outside = dict(point, id="outside", z=5.9, z_min=5.9, z_max=5.9)
    score, rows = neutral_residuals([20, 7, 6.01], [0, 0.5, 0.99], [point, wide, outside])
    # Model xHI=.5 uses upper uncertainty .2; wide bin contributes no score.
    np.testing.assert_allclose(score, 0.25)
    assert rows[0]["used"] and not rows[1]["used"]
    assert rows[2]["model_xhi"] is None and not rows[2]["used"]
