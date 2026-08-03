from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_HISTORY = PROJECT_ROOT / "data_save" / "zeus21_popiii_fiducial.csv"
MASS_DISTRIBUTION = PROJECT_ROOT / "data_save" / "zeus21_popiii_mass_distribution.npz"
MASS_BIN_COMPOSITION = (
    PROJECT_ROOT / "data_save" / "zeus21_popii_popiii_mass_bin_fractions.csv"
)


def _load_global_columns() -> dict[str, np.ndarray]:
    with GLOBAL_HISTORY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"tracked Zeus21 global history is empty: {GLOBAL_HISTORY}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def test_zeus21_mass_distribution_closes_to_tracked_global_history() -> None:
    global_columns = _load_global_columns()
    with np.load(MASS_DISTRIBUTION, allow_pickle=False) as payload:
        assert str(payload["schema_version"]) == "auroralf.zeus21_popiii_mass_distribution.v2"
        redshift = np.asarray(payload["redshift"], dtype=float)
        halo_mass = np.asarray(payload["halo_mass_msun"], dtype=float)
        hmf_dndm = np.asarray(payload["hmf_dndm_mpc3_msun"], dtype=float)
        sfr_popii = np.asarray(payload["popii_sfr_per_halo_msun_yr"], dtype=float)
        sfr_popiii = np.asarray(payload["popiii_sfr_per_halo_msun_yr"], dtype=float)
        popii = np.asarray(payload["dsfrd_dlog10m_popii_msun_yr_mpc3"], dtype=float)
        popiii = np.asarray(payload["dsfrd_dlog10m_popiii_msun_yr_mpc3"], dtype=float)

    np.testing.assert_allclose(
        redshift,
        global_columns["redshift"],
        rtol=1.0e-12,
        atol=0.0,
    )
    assert halo_mass.ndim == 1
    assert np.all(np.diff(halo_mass) > 0.0)
    assert hmf_dndm.shape == sfr_popii.shape == sfr_popiii.shape == popii.shape == popiii.shape
    assert popiii.shape == (redshift.size, halo_mass.size)
    for values in (hmf_dndm, sfr_popii, sfr_popiii, popii, popiii):
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    np.testing.assert_allclose(
        popii,
        hmf_dndm * sfr_popii * halo_mass[None, :] * np.log(10.0),
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        popiii,
        hmf_dndm * sfr_popiii * halo_mass[None, :] * np.log(10.0),
        rtol=2.0e-12,
        atol=0.0,
    )
    integrated_popii = np.trapezoid(popii, np.log10(halo_mass), axis=1)
    integrated_popiii = np.trapezoid(popiii, np.log10(halo_mass), axis=1)
    np.testing.assert_allclose(
        integrated_popii,
        global_columns["sfrd_popii_msun_yr_mpc3"],
        rtol=5.0e-3,
        atol=0.0,
    )
    np.testing.assert_allclose(
        integrated_popiii,
        global_columns["sfrd_popiii_msun_yr_mpc3"],
        rtol=5.0e-3,
        atol=0.0,
    )
    global_total = (
        global_columns["sfrd_popii_msun_yr_mpc3"]
        + global_columns["sfrd_popiii_msun_yr_mpc3"]
    )
    np.testing.assert_allclose(
        global_columns["popiii_fraction_of_total_sfrd"],
        global_columns["sfrd_popiii_msun_yr_mpc3"] / global_total,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        global_columns["popii_fraction_of_total_sfrd"]
        + global_columns["popiii_fraction_of_total_sfrd"],
        1.0,
        rtol=1.0e-12,
    )


def test_zeus21_popiii_mass_summary_matches_resolved_distribution() -> None:
    global_columns = _load_global_columns()
    with np.load(MASS_DISTRIBUTION, allow_pickle=False) as payload:
        halo_mass = np.asarray(payload["halo_mass_msun"], dtype=float)
        contribution = np.asarray(
            payload["dsfrd_dlog10m_popiii_msun_yr_mpc3"],
            dtype=float,
        )

    peak = global_columns["popiii_sfrd_peak_halo_mass_msun"]
    p16 = global_columns["popiii_sfrd_p16_halo_mass_msun"]
    median = global_columns["popiii_sfrd_median_halo_mass_msun"]
    p84 = global_columns["popiii_sfrd_p84_halo_mass_msun"]
    np.testing.assert_allclose(peak, halo_mass[np.argmax(contribution, axis=1)])
    assert np.all(p16 <= median)
    assert np.all(median <= p84)
    assert np.all((p16 >= halo_mass[0]) & (p84 <= halo_mass[-1]))


def test_zeus21_instantaneous_population_fraction_and_transitions() -> None:
    with np.load(MASS_DISTRIBUTION, allow_pickle=False) as payload:
        redshift = np.asarray(payload["redshift"], dtype=float)
        halo_mass = np.asarray(payload["composition_halo_mass_msun"], dtype=float)
        sfr_popii = np.asarray(
            payload["composition_sfr_popii_per_halo_msun_yr"],
            dtype=float,
        )
        sfr_popiii = np.asarray(
            payload["composition_sfr_popiii_per_halo_msun_yr"],
            dtype=float,
        )
        fraction = np.asarray(
            payload["popiii_instantaneous_sfr_fraction"],
            dtype=float,
        )
        levels = np.asarray(payload["popiii_fraction_levels"], dtype=float)
        transition_mass = np.asarray(
            payload["popiii_fraction_transition_halo_mass_msun"],
            dtype=float,
        )

    assert halo_mass.ndim == 1
    assert fraction.shape == sfr_popii.shape == sfr_popiii.shape
    assert fraction.shape == (redshift.size, halo_mass.size)
    np.testing.assert_allclose(
        fraction,
        sfr_popiii / (sfr_popii + sfr_popiii),
        rtol=2.0e-12,
        atol=0.0,
    )
    assert np.all(np.diff(fraction, axis=1) <= 1.0e-10)
    assert transition_mass.shape == (redshift.size, levels.size)
    assert np.all(transition_mass[:, 0] < transition_mass[:, 1])
    assert np.all(transition_mass[:, 1] < transition_mass[:, 2])
    assert np.all(np.diff(transition_mass[:, 1]) < 0.0)

    log_mass = np.log10(halo_mass)
    for redshift_index, row in enumerate(fraction):
        reconstructed_levels = np.asarray(
            [
                np.interp(np.log10(mass), log_mass, row)
                for mass in transition_mass[redshift_index]
            ]
        )
        np.testing.assert_allclose(reconstructed_levels, levels, atol=2.0e-12)


def test_zeus21_mass_bin_population_fractions_are_normalized() -> None:
    global_columns = _load_global_columns()
    with np.load(MASS_DISTRIBUTION, allow_pickle=False) as payload:
        redshift = np.asarray(payload["redshift"], dtype=float)
        mass_bin_edges = np.asarray(payload["mass_bin_edges_msun"], dtype=float)
        popii = np.asarray(
            payload["sfrd_popii_by_mass_bin_msun_yr_mpc3"],
            dtype=float,
        )
        popiii = np.asarray(
            payload["sfrd_popiii_by_mass_bin_msun_yr_mpc3"],
            dtype=float,
        )
        fraction = np.asarray(payload["popiii_sfr_fraction_by_mass_bin"], dtype=float)
        total_share = np.asarray(
            payload["total_sfrd_fraction_by_mass_bin"],
            dtype=float,
        )

    expected_shape = (redshift.size, mass_bin_edges.size - 1)
    assert popii.shape == popiii.shape == fraction.shape == total_share.shape
    assert popii.shape == expected_shape
    np.testing.assert_allclose(fraction, popiii / (popii + popiii), rtol=2.0e-12)
    np.testing.assert_allclose(np.sum(total_share, axis=1), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(
        np.sum(popii, axis=1),
        global_columns["sfrd_popii_msun_yr_mpc3"],
        rtol=5.0e-3,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sum(popiii, axis=1),
        global_columns["sfrd_popiii_msun_yr_mpc3"],
        rtol=5.0e-3,
        atol=0.0,
    )
    assert np.all(fraction[:, 1] > 0.98)
    assert np.all((fraction[:, 2] > 0.35) & (fraction[:, 2] < 0.50))
    assert np.all(fraction[:, 3] < 0.02)

    with MASS_BIN_COMPOSITION.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == redshift.size * (mass_bin_edges.size - 1)
    saved_fraction = np.asarray(
        [float(row["popiii_fraction_of_bin_sfr"]) for row in rows],
        dtype=float,
    ).reshape(expected_shape)
    saved_total_share = np.asarray(
        [float(row["bin_fraction_of_total_sfrd"]) for row in rows],
        dtype=float,
    ).reshape(expected_shape)
    np.testing.assert_allclose(saved_fraction, fraction, rtol=1.0e-12)
    np.testing.assert_allclose(saved_total_share, total_share, rtol=1.0e-12)
