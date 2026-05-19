from __future__ import annotations

import numpy as np

from auroralf.chemistry import summarize_metallicity_history


def test_metallicity_history_summary_uses_active_and_starforming_masks() -> None:
    z_grid = np.array([[14.0, 13.0, 12.5], [14.0, 13.0, 12.5]], dtype=float)
    gas_metallicity = np.array([[0.0, 0.2, 0.4], [0.8, 1.0, 1.2]], dtype=float)
    birth_metallicity = np.array([[0.0, 0.1, 0.3], [0.7, 0.9, 1.1]], dtype=float)
    active_grid = np.array([[False, True, True], [True, False, True]], dtype=bool)
    starforming_grid = np.array([[False, True, True], [True, False, False]], dtype=bool)
    topheavy_source_grid = np.array([[False, True, False], [True, False, False]], dtype=bool)

    summary = summarize_metallicity_history(
        z_grid=z_grid,
        gas_metallicity_zsun_grid=gas_metallicity,
        birth_metallicity_zsun_grid=birth_metallicity,
        active_grid=active_grid,
        starforming_grid=starforming_grid,
        topheavy_source_grid=topheavy_source_grid,
    )

    np.testing.assert_allclose(summary["z"], [14.0, 13.0, 12.5])
    np.testing.assert_allclose(summary["gas_median"], [0.8, 0.2, 0.8])
    np.testing.assert_allclose(summary["birth_median"], [0.7, 0.1, 0.3])
    np.testing.assert_array_equal(summary["active_count"], [1, 1, 2])
    np.testing.assert_array_equal(summary["starforming_count"], [1, 1, 1])
    np.testing.assert_allclose(summary["topheavy_source_fraction"], [1.0, 1.0, 0.0])


def test_metallicity_history_summary_returns_nan_when_no_valid_sources() -> None:
    z_grid = np.array([[14.0, 13.0]], dtype=float)
    metallicity = np.array([[0.2, 0.3]], dtype=float)
    inactive = np.array([[False, False]], dtype=bool)

    summary = summarize_metallicity_history(
        z_grid=z_grid,
        gas_metallicity_zsun_grid=metallicity,
        birth_metallicity_zsun_grid=metallicity,
        active_grid=inactive,
        starforming_grid=inactive,
    )

    assert np.all(np.isnan(summary["gas_median"]))
    assert np.all(np.isnan(summary["birth_median"]))
    assert np.all(np.isnan(summary["topheavy_source_fraction"]))
