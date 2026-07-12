from __future__ import annotations

import numpy as np
import pytest


def test_integrate_lookback_mass_uses_years_per_gyr() -> None:
    from scripts.analysis.plot_popiii_sfh_vs_popii import integrate_mass_within_lookback

    lookback_myr = np.array([10.0, 5.0, 0.0])
    sfr = np.array([2.0, 2.0, 2.0])

    mass = integrate_mass_within_lookback(lookback_myr=lookback_myr, sfr_msun_yr=sfr, max_lookback_myr=5.0)

    assert mass == pytest.approx(1.0e7)


def test_prepare_log_series_masks_nonpositive_values() -> None:
    from scripts.analysis.plot_popiii_sfh_vs_popii import positive_for_log_plot

    values = positive_for_log_plot(np.array([1.0, 0.0, -1.0, 2.0]))

    np.testing.assert_allclose(values[[0, 3]], np.array([1.0, 2.0]))
    assert np.isnan(values[1])
    assert np.isnan(values[2])


def test_popiii_sfh_comparison_defaults_target_brightest_heii_track(monkeypatch) -> None:
    from scripts.analysis import plot_popiii_sfh_vs_popii

    monkeypatch.setattr("sys.argv", ["plot_popiii_sfh_vs_popii.py"])
    args = plot_popiii_sfh_vs_popii._parse_args()

    assert args.z == pytest.approx(10.583)
    assert args.logMh == pytest.approx(8.272727272727273)
    assert args.random_seed == 111
    assert args.track_index == 4
    assert args.output_prefix.name == "popiii_sfh_vs_popii_brightest_heii_track"


def test_popiii_sfh_comparison_exposes_visbal2015_arguments(monkeypatch) -> None:
    from scripts.analysis import plot_popiii_sfh_vs_popii

    monkeypatch.setattr("sys.argv", ["plot_popiii_sfh_vs_popii.py"])
    args = plot_popiii_sfh_vs_popii._parse_args()

    assert args.include_visbal2015_sfh is False
    assert args.visbal_fstar == pytest.approx(0.1)
    assert args.eta_duty_values == "1.0,0.1,0.01"

    values = plot_popiii_sfh_vs_popii.parse_eta_duty_values("1,0.25,0.01")

    np.testing.assert_allclose(values, np.array([1.0, 0.25, 0.01]))
