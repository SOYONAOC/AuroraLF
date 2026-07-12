from __future__ import annotations

from pathlib import Path

import numpy as np

from auroralf.uvlf import compute_dust_attenuated_uvlf
from scripts.plot.plot_popiii_mup_comparison_uvlf import _load_observation_table_for_comparison
from scripts.plot.plot_popiii_mup_comparison_uvlf import _apply_optional_dust_to_component
from scripts.plot.plot_popiii_mup_comparison_uvlf import _resolve_log_y_limits


def test_popiii_mup_comparison_loads_legacy_z6_observation_table() -> None:
    observation = _load_observation_table_for_comparison(
        Path("external_data/observations/uvlf/redshift_6/bowler_uvlf_z6.npz")
    )

    assert observation["label"] == "Bowler+15"
    assert observation["source"] == "bowler_uvlf_z6.npz"
    assert observation["z_note"] == ""
    assert np.asarray(observation["muv"], dtype=float).size == 5
    assert not np.any(np.asarray(observation["upper_limit"], dtype=bool))


def test_popiii_mup_comparison_applies_project_dust_transform() -> None:
    centers = np.array([-23.0, -22.0, -21.0, -20.0, -19.0], dtype=float)
    phi = np.array([1.0e-7, 2.0e-6, 1.0e-5, 5.0e-5, 2.0e-4], dtype=float)

    observed = _apply_optional_dust_to_component(centers, phi, z_obs=6.0, apply_dust=True)
    expected = np.asarray(compute_dust_attenuated_uvlf(centers, phi, 6.0, muv_obs=centers)["phi_obs"])

    np.testing.assert_allclose(observed, expected)


def test_popiii_mup_comparison_keeps_intrinsic_component_without_dust() -> None:
    centers = np.array([-23.0, -22.0, -21.0, -20.0, -19.0], dtype=float)
    phi = np.array([1.0e-7, 2.0e-6, 1.0e-5, 5.0e-5, 2.0e-4], dtype=float)

    intrinsic = _apply_optional_dust_to_component(centers, phi, z_obs=6.0, apply_dust=False)

    np.testing.assert_array_equal(intrinsic, phi)
    assert intrinsic is not phi


def test_popiii_mup_comparison_uses_data_driven_y_limit_without_fixed_phi_min() -> None:
    lower, upper = _resolve_log_y_limits(
        all_positive=np.array([1.0e-12, 1.0e-4]),
        comparison_positive=np.array([2.0e-7, 1.0e-4]),
        fixed_phi_min=None,
    )

    assert np.isclose(lower, 8.0e-8)
    assert np.isclose(upper, 3.0e-4)


def test_popiii_mup_comparison_honors_explicit_fixed_phi_min() -> None:
    lower, upper = _resolve_log_y_limits(
        all_positive=np.array([1.0e-12, 1.0e-4]),
        comparison_positive=np.array([2.0e-7, 1.0e-4]),
        fixed_phi_min=1.0e-9,
    )

    assert lower == 1.0e-9
    assert np.isclose(upper, 3.0e-4)


def test_popiii_mup_comparison_limits_default_lower_bound_to_observation_floor() -> None:
    lower, upper = _resolve_log_y_limits(
        all_positive=np.array([1.0e-15, 1.0e-4]),
        comparison_positive=np.array([1.0e-15, 1.0e-4]),
        observation_positive=np.array([1.0e-6]),
        fixed_phi_min=None,
    )

    assert np.isclose(lower, 1.0e-8)
    assert np.isclose(upper, 3.0e-4)
