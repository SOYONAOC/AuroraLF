from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from auroralf.uvlf import compute_dust_attenuated_uvlf
from scripts.plot.plot_popiii_mup_comparison_uvlf import _load_observation_table_for_comparison
from scripts.plot.plot_popiii_mup_comparison_uvlf import _apply_optional_dust_to_component
from scripts.plot.plot_popiii_mup_comparison_uvlf import _resolve_log_y_limits
from scripts.plot.plot_popiii_mup_comparison_uvlf import _save_npz
from scripts.plot.plot_popiii_mup_comparison_uvlf import Scenario
from scripts.plot.plot_popiii_mup_comparison_uvlf import ScenarioResult


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


def test_popiii_mup_summary_npz_omits_per_halo_samples(tmp_path: Path) -> None:
    values = np.array([1.0, 2.0])
    components = {
        name: {
            "phi": values,
            "phi_sigma": 0.1 * values,
            "raw_counts": np.array([10, 20]),
        }
        for name in ("popii", "popiii", "total", "popiii_burst", "total_burst")
    }
    result = ScenarioResult(
        scenario=Scenario(
            key="fixed_mup1e10",
            title="fixed",
            upper_mass_mode="fixed",
            upper_mass_msun=1.0e10,
        ),
        components=components,
        plot_data={name: (values, values) for name in components},
        plot_columns={name: values for name in components},
        total_luminosity=values,
        popii_luminosity=values,
        popiii_luminosity=values,
        scattered_popiii_luminosity=values,
        scattered_total_luminosity=values,
        scattered_sample_weight=values,
        sample_weight=values,
        sample_mh=values,
        sample_stellar_channel=np.array(["popii", "popiii"]),
        popiii_upper_mass_msun=1.0e10,
    )
    args = argparse.Namespace(
        z=6.0,
        logM_max=13.0,
        N_mass=5062,
        n_tracks=1000,
        n_grid=240,
        random_seed=42,
        smooth_sigma_mag=0.6,
        phi_min=None,
        popiii_burst_sigma_mag=2.0,
        popiii_burst_quadrature_order=31,
        plot_min_raw_counts=10,
        lw_background_j21=0.0,
        apply_dust=True,
        popiii_ssp_label="test",
        hmf_dlog10m=0.02,
        epsilon_0=0.12,
        fstar_characteristic_mass_msun=10.0**11.7,
        fstar_beta=0.66,
        fstar_gamma=0.65,
    )
    output = tmp_path / "summary.npz"

    _save_npz(
        output,
        args=args,
        bin_edges=np.array([-21.0, -20.0, -19.0]),
        centers=np.array([-20.5, -19.5]),
        popiii_minimum_mass_msun=1.0e6,
        atomic_mass_msun=1.0e8,
        logm_min=6.0,
        popiii_ssp_file=tmp_path / "ssp.dat",
        results=[result],
        include_samples=False,
    )

    with np.load(output, allow_pickle=False) as payload:
        assert not bool(payload["samples_included"][0])
        assert payload["mass_function_model"][0] == "hmf_reed07"
        assert "fixed_mup1e10_phi_total_burst" in payload.files
        assert "fixed_mup1e10_total_luminosity" not in payload.files
