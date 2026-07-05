from __future__ import annotations

import numpy as np

from scripts.plot.plot_popiii_mup_pisn_proxy import (
    _integrated_lnu_per_sfr,
    _pisn_events_per_stellar_mass,
)


def test_pisn_events_per_stellar_mass_for_salpeter_popiii_imf() -> None:
    eta = _pisn_events_per_stellar_mass(
        imf_slope=2.35,
        imf_min_msun=50.0,
        imf_max_msun=500.0,
        pisn_min_msun=140.0,
        pisn_max_msun=260.0,
    )

    np.testing.assert_allclose(eta, 1.3221293019964493e-3)


def test_integrated_lnu_per_sfr_has_year_units() -> None:
    ages_myr = np.array([0.01, 1.0, 2.0], dtype=float)
    lnu_per_msun = np.full_like(ages_myr, 2.0e20, dtype=float)

    lnu_per_sfr = _integrated_lnu_per_sfr(
        ages_myr=ages_myr,
        lnu_per_msun=lnu_per_msun,
        visibility_myr=2.0,
    )

    np.testing.assert_allclose(lnu_per_sfr, 2.0e20 * 1.99e6)
