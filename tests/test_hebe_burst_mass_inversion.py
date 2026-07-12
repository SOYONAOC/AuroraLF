from __future__ import annotations

import numpy as np
import pytest


def test_required_burst_mass_from_heii_luminosity_uses_q_per_mass() -> None:
    from scripts.analysis.plot_hebe_burst_mass_inversion import required_burst_mass_msun

    mass = required_burst_mass_msun(
        heii_luminosity_erg_s=np.array([8.54e40]),
        q_heplus_per_msun=np.array([3.721321209361762e46]),
        caseb_erg_per_photon=5.7e-12,
    )

    np.testing.assert_allclose(mass, np.array([4.02611204393197e5]))


def test_caseb_heii_to_hgamma_ratio_uses_hbeta_and_caseb_balmer_ratio() -> None:
    from scripts.analysis.plot_hebe_burst_mass_inversion import caseb_heii_to_hgamma_ratio

    ratio = caseb_heii_to_hgamma_ratio(
        heii1640_luminosity_per_msun=np.array([6.0]),
        hbeta_luminosity_per_msun=np.array([10.0]),
        hgamma_to_hbeta=0.47,
    )

    np.testing.assert_allclose(ratio, np.array([6.0 / 4.7]))


def test_hbeta_loader_reconstructs_nonmonotonic_schaerer_is5_age_grid(tmp_path) -> None:
    from scripts.analysis.plot_hebe_burst_mass_inversion import _load_hbeta_luminosity_table

    table = tmp_path / "pop3_test_is5.22"
    table.write_text(
        "\n".join(
            [
                " 4.000E+00  47.0 46.0 45.0 1.00E+34",
                " 6.000E+00  47.0 46.0 45.0 2.00E+34",
                " 5.000E+00  47.0 46.0 45.0 3.00E+34",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ages_myr, hbeta = _load_hbeta_luminosity_table(table)

    np.testing.assert_allclose(ages_myr, np.array([1.0e-2, 1.0, 2.0]))
    np.testing.assert_allclose(hbeta, np.array([1.0e34, 2.0e34, 3.0e34]))


def test_burst_mass_inversion_rejects_nonpositive_photon_rates() -> None:
    from scripts.analysis.plot_hebe_burst_mass_inversion import required_burst_mass_msun

    with pytest.raises(ValueError, match="q_heplus_per_msun must be positive"):
        required_burst_mass_msun(
            heii_luminosity_erg_s=np.array([1.0]),
            q_heplus_per_msun=np.array([0.0]),
        )


def test_burst_mass_inversion_defaults_use_existing_observation_table(monkeypatch) -> None:
    from scripts.analysis import plot_hebe_burst_mass_inversion

    monkeypatch.setattr("sys.argv", ["plot_hebe_burst_mass_inversion.py"])
    args = plot_hebe_burst_mass_inversion._parse_args()

    assert args.observation_file.name == "maiolino_rusta_2026_heii_constraints.csv"
    assert args.output_prefix.name == "hebe_burst_mass_inversion"
    assert args.hgamma_to_hbeta == pytest.approx(0.47)
