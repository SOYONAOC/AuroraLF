from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_HISTORY = PROJECT_ROOT / "data_save" / "zeus21_popiii_fiducial.csv"
MASS_DISTRIBUTION = PROJECT_ROOT / "data_save" / "zeus21_popiii_mass_distribution.npz"


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
        assert str(payload["schema_version"]) == "auroralf.zeus21_popiii_mass_distribution.v1"
        redshift = np.asarray(payload["redshift"], dtype=float)
        halo_mass = np.asarray(payload["halo_mass_msun"], dtype=float)
        hmf_dndm = np.asarray(payload["hmf_dndm_mpc3_msun"], dtype=float)
        sfr_per_halo = np.asarray(payload["popiii_sfr_per_halo_msun_yr"], dtype=float)
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
    assert hmf_dndm.shape == sfr_per_halo.shape == popii.shape == popiii.shape
    assert popiii.shape == (redshift.size, halo_mass.size)
    for values in (hmf_dndm, sfr_per_halo, popii, popiii):
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    np.testing.assert_allclose(
        popiii,
        hmf_dndm * sfr_per_halo * halo_mass[None, :] * np.log(10.0),
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
