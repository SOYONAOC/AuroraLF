from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.seeding import derive_hmf_mass_seed, derive_pipeline_random_seeds
from scripts.analysis import plot_sfrd_lw_background_from_model as lw_script

from scripts.analysis.plot_sfrd_lw_background_from_model import (
    _compute_lw_proxy,
    _integrate_sfrd_at_z,
    _parse_fixed_lw_j21_values,
    _resolve_log_ylim,
    _scenario_label,
)


def test_sfrd_integration_has_no_independent_mass_sampling_lw_parameter() -> None:
    signature = inspect.signature(_integrate_sfrd_at_z)

    assert "mass_sampling_lw_background_j21" not in signature.parameters


def test_sfrd_integration_uses_base_seed_keyed_by_redshift() -> None:
    signature = inspect.signature(_integrate_sfrd_at_z)

    assert "base_seed" in signature.parameters
    assert "random_seed" not in signature.parameters


def test_parse_fixed_lw_j21_values_rejects_negative_entries() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _parse_fixed_lw_j21_values("0.0,-0.1")


def test_parse_fixed_lw_j21_values_accepts_comma_separated_values() -> None:
    values = _parse_fixed_lw_j21_values("0,0.01,0.1")

    np.testing.assert_allclose(values, np.array([0.0, 0.01, 0.1]))


def test_lw_proxy_is_local_to_the_lw_horizon() -> None:
    support_z = np.array([10.0, 11.0, 12.0, 20.0, 32.0], dtype=float)
    rho_sfrd = np.array([1.0, 1.0, 1.0, 1000.0, 1000.0], dtype=float)
    evaluation_z = np.array([10.0, 11.0, 12.0, 20.0], dtype=float)

    cosmology = Cosmology()
    local_proxy = _compute_lw_proxy(
        support_z,
        rho_sfrd,
        evaluation_z=evaluation_z,
        cosmology=cosmology,
        horizon_fraction=0.1,
    )
    wide_proxy = _compute_lw_proxy(
        support_z,
        rho_sfrd,
        evaluation_z=evaluation_z,
        cosmology=cosmology,
        horizon_fraction=0.5,
    )

    assert local_proxy.shape == evaluation_z.shape
    assert np.all(np.isfinite(local_proxy))
    assert np.all(local_proxy > 0.0)
    assert wide_proxy[0] > 100.0 * local_proxy[0]


def test_lw_proxy_uses_complete_horizon_at_requested_endpoint() -> None:
    evaluation_z = np.array([10.0, 11.0, 12.0], dtype=float)
    support_z = np.array([10.0, 11.0, 12.0, 13.0, 14.6], dtype=float)

    proxy = _compute_lw_proxy(
        support_z,
        np.ones_like(support_z),
        evaluation_z=evaluation_z,
        cosmology=Cosmology(),
        horizon_fraction=0.2,
    )

    assert proxy.shape == evaluation_z.shape
    assert np.all(proxy > 0.0)


def test_lw_proxy_rejects_support_that_does_not_cover_complete_horizon() -> None:
    with pytest.raises(
        ValueError,
        match=r"LW support is insufficient: required zmax=14\.6, provided zmax=12",
    ):
        _compute_lw_proxy(
            np.array([10.0, 11.0, 12.0]),
            np.ones(3),
            evaluation_z=np.array([10.0, 11.0, 12.0]),
            cosmology=Cosmology(),
            horizon_fraction=0.2,
        )


@pytest.mark.parametrize(
    ("support_z", "evaluation_z", "message"),
    [
        (
            np.array([10.0, 12.0, 11.0, 20.0]),
            np.array([10.0, 11.0]),
            "support_z must be strictly increasing",
        ),
        (
            np.array([10.0, 11.0, 11.0, 20.0]),
            np.array([10.0, 11.0]),
            "support_z must be strictly increasing",
        ),
        (np.array([10.0, np.nan, 20.0]), np.array([10.0, 11.0]), "support_z must be finite"),
        (np.array([-1.0, 10.0, 20.0]), np.array([10.0, 11.0]), "support_z must be finite"),
        (
            np.array([10.0, 11.0, 20.0]),
            np.array([11.0, 10.0]),
            "evaluation_z must be strictly increasing",
        ),
        (
            np.array([10.0, 11.0, 20.0]),
            np.array([10.0, 10.0]),
            "evaluation_z must be strictly increasing",
        ),
        (np.array([10.0, 11.0, 20.0]), np.array([-1.0, 10.0]), "evaluation_z must be finite"),
    ],
)
def test_lw_proxy_rejects_invalid_redshift_grids(
    support_z: np.ndarray,
    evaluation_z: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _compute_lw_proxy(
            support_z,
            np.ones_like(support_z),
            evaluation_z=evaluation_z,
            cosmology=Cosmology(),
            horizon_fraction=0.2,
        )


@pytest.mark.parametrize("rho_sfrd", [np.array([1.0, np.nan, 1.0]), np.array([1.0, -1.0, 1.0])])
def test_lw_proxy_rejects_nonfinite_or_negative_sfrd(rho_sfrd: np.ndarray) -> None:
    with pytest.raises(ValueError, match="rho_sfrd must be finite and non-negative"):
        _compute_lw_proxy(
            np.array([10.0, 11.0, 14.6]),
            rho_sfrd,
            evaluation_z=np.array([10.0, 11.0]),
            cosmology=Cosmology(),
            horizon_fraction=0.2,
        )


def test_lw_proxy_cosmology_grids_include_exact_light_cone_endpoints() -> None:
    calls: list[np.ndarray] = []

    class RecordingCosmology:
        @staticmethod
        def hubble(redshift: np.ndarray) -> np.ndarray:
            calls.append(np.asarray(redshift).copy())
            return np.ones_like(redshift, dtype=float)

    evaluation_z = np.array([10.0, 12.0])
    _compute_lw_proxy(
        np.array([10.0, 11.0, 12.0, 13.0, 14.6]),
        np.ones(5),
        evaluation_z=evaluation_z,
        cosmology=RecordingCosmology(),  # type: ignore[arg-type]
        horizon_fraction=0.2,
        dense_size=128,
    )

    for integration_z, z_now in zip(calls, evaluation_z, strict=True):
        assert integration_z[0] == pytest.approx(z_now)
        assert integration_z[-1] == pytest.approx(z_now + 0.2 * (1.0 + z_now))


def test_lw_proxy_has_formed_mass_units_and_dense_grid_converges() -> None:
    class ConstantHubbleCosmology:
        @staticmethod
        def hubble(redshift: np.ndarray) -> np.ndarray:
            return np.full_like(redshift, 0.1, dtype=float)

    rho_sfrd = 2.5
    evaluation_z = np.array([10.0, 12.0])
    support_z = np.array([10.0, 12.0, 14.6])
    expected = (
        rho_sfrd
        * 1.0e9
        / 0.1
        * np.log(
            (1.0 + evaluation_z + 0.2 * (1.0 + evaluation_z))
            / (1.0 + evaluation_z)
        )
    )

    default_proxy = _compute_lw_proxy(
        support_z,
        np.full_like(support_z, rho_sfrd),
        evaluation_z=evaluation_z,
        cosmology=ConstantHubbleCosmology(),  # type: ignore[arg-type]
        horizon_fraction=0.2,
    )
    coarse_proxy = _compute_lw_proxy(
        support_z,
        np.full_like(support_z, rho_sfrd),
        evaluation_z=evaluation_z,
        cosmology=ConstantHubbleCosmology(),  # type: ignore[arg-type]
        horizon_fraction=0.2,
        dense_size=128,
    )

    default_relative_error = np.abs(default_proxy - expected) / expected
    coarse_relative_error = np.abs(coarse_proxy - expected) / expected
    assert np.all(default_relative_error < 1.0e-7)
    assert np.all(coarse_relative_error < 2.0e-3)
    assert np.all(default_relative_error < coarse_relative_error)


def test_lw_proxy_integrates_narrow_features_at_support_knots() -> None:
    class ConstantHubbleCosmology:
        @staticmethod
        def hubble(redshift: np.ndarray) -> np.ndarray:
            return np.full_like(redshift, 0.1, dtype=float)

    support_z = np.array([10.0, 10.000001, 10.000002, 12.0, 14.6])
    rho_sfrd = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    reference_z = np.linspace(10.0, 10.000002, 20_001)
    reference_rho = np.interp(reference_z, support_z, rho_sfrd)
    reference = 1.0e9 * np.trapezoid(
        reference_rho / ((1.0 + reference_z) * 0.1),
        x=reference_z,
    )

    proxy = _compute_lw_proxy(
        support_z,
        rho_sfrd,
        evaluation_z=np.array([10.0]),
        cosmology=ConstantHubbleCosmology(),  # type: ignore[arg-type]
        horizon_fraction=0.2,
    )

    assert proxy[0] > 0.0
    assert proxy[0] == pytest.approx(reference, rel=1.0e-8)


def test_lw_support_grid_covers_horizon_independently_of_request_order() -> None:
    requested = np.array([12.0, 10.0, 11.0], dtype=float)

    support = lw_script._build_lw_support_grid(
        requested,
        horizon_fraction=0.2,
        z_start_max=20.0,
    )
    reversed_support = lw_script._build_lw_support_grid(
        requested[::-1],
        horizon_fraction=0.2,
        z_start_max=20.0,
    )

    np.testing.assert_array_equal(support, reversed_support)
    np.testing.assert_allclose(support, np.array([10.0, 11.0, 12.0, 13.0, 14.0, 14.6]))
    assert np.all(np.diff(support) > 0.0)
    assert all(value in support for value in requested)
    assert support[-1] == pytest.approx(12.0 + 0.2 * (1.0 + 12.0))


def test_lw_support_grid_rejects_horizon_at_mah_start_limit() -> None:
    with pytest.raises(
        ValueError,
        match=r"LW support requires zmax=14\.6 below z_start_max=14\.6",
    ):
        lw_script._build_lw_support_grid(
            np.array([10.0, 11.0, 12.0]),
            horizon_fraction=0.2,
            z_start_max=14.6,
        )


def test_lw_support_grid_rejects_unrepresentable_positive_horizon() -> None:
    with pytest.raises(
        ValueError,
        match="LW horizon endpoint must be strictly above the maximum requested redshift",
    ):
        lw_script._build_lw_support_grid(
            np.array([10.0, 11.0, 12.0]),
            horizon_fraction=float(np.nextafter(0.0, 1.0)),
            z_start_max=20.0,
        )


def test_lw_support_grid_fails_before_allocating_pathological_default_spacing() -> None:
    requested_z = np.array([10.0, 10.000001, 12.0])

    with pytest.raises(
        ValueError,
        match=r"LW support grid requires .* points, exceeding max_support_points=512; .*--lw-support-dz",
    ):
        lw_script._build_lw_support_grid(
            requested_z,
            horizon_fraction=0.2,
            z_start_max=20.0,
            max_support_points=512,
        )


def test_lw_support_grid_rejects_subnormal_spacing_without_overflow() -> None:
    with pytest.raises(
        ValueError,
        match=r"LW support grid requires more than 512 points; .*--lw-support-dz",
    ):
        lw_script._build_lw_support_grid(
            np.array([10.0, 11.0, 12.0]),
            horizon_fraction=0.2,
            z_start_max=20.0,
            support_dz=float(np.nextafter(0.0, 1.0)),
            max_support_points=512,
        )


def test_lw_support_grid_accepts_explicit_coarse_spacing() -> None:
    requested_z = np.array([10.0, 10.000001, 12.0])

    support_z = lw_script._build_lw_support_grid(
        requested_z,
        horizon_fraction=0.2,
        z_start_max=20.0,
        support_dz=0.5,
        max_support_points=512,
    )

    assert np.all(np.diff(support_z) > 0.0)
    np.testing.assert_array_equal(support_z[: requested_z.size], requested_z)
    assert support_z[-1] == pytest.approx(14.6)


@pytest.mark.parametrize("support_dz", [0.0, -0.1, np.nan, np.inf])
def test_lw_support_grid_rejects_invalid_explicit_spacing(support_dz: float) -> None:
    with pytest.raises(ValueError, match="support_dz must be finite and positive"):
        lw_script._build_lw_support_grid(
            np.array([10.0, 11.0, 12.0]),
            horizon_fraction=0.2,
            z_start_max=20.0,
            support_dz=support_dz,
        )


@pytest.mark.parametrize("max_support_points", [0, -1, 1.5, True])
def test_lw_support_grid_rejects_invalid_point_cap(max_support_points: object) -> None:
    with pytest.raises(ValueError, match="max_support_points must be a positive integer"):
        lw_script._build_lw_support_grid(
            np.array([10.0, 11.0, 12.0]),
            horizon_fraction=0.2,
            z_start_max=20.0,
            max_support_points=max_support_points,  # type: ignore[arg-type]
        )


def test_z_values_override_does_not_validate_against_unused_z_max() -> None:
    args = argparse.Namespace(
        z_values="10,11,12",
        z_min=5.0,
        z_max=35.0,
        n_z=13,
        N_mass=2,
        n_tracks=1,
        n_grid=2,
        logM_max=12.0,
        z_start_max=20.0,
        lw_horizon_fraction=0.2,
        lw_proxy_dense_size=128,
        lw_support_dz=None,
        lw_support_max_points=512,
    )

    lw_script._validate_args(args)
    requested_z = lw_script._parse_z_values(args)
    support_z = lw_script._build_lw_support_grid(
        requested_z,
        horizon_fraction=args.lw_horizon_fraction,
        z_start_max=args.z_start_max,
    )

    assert support_z[-1] == pytest.approx(14.6)
    with pytest.raises(ValueError, match=r"LW support requires zmax=14\.6"):
        lw_script._build_lw_support_grid(
            requested_z,
            horizon_fraction=args.lw_horizon_fraction,
            z_start_max=14.6,
        )


def test_parse_args_exposes_bounded_lw_support_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_sfrd_lw_background_from_model.py"])

    defaults = lw_script._parse_args()

    assert defaults.lw_support_dz is None
    assert defaults.lw_support_max_points == 512

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_sfrd_lw_background_from_model.py",
            "--lw-support-dz",
            "0.5",
            "--lw-support-max-points",
            "64",
        ],
    )
    explicit = lw_script._parse_args()

    assert explicit.lw_support_dz == pytest.approx(0.5)
    assert explicit.lw_support_max_points == 64


def test_main_computes_extended_support_but_outputs_requested_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_z = np.array([10.0, 11.0, 12.0], dtype=float)
    args = argparse.Namespace(
        z_values="10,11,12",
        z_min=10.0,
        z_max=12.0,
        n_z=3,
        N_mass=2,
        n_tracks=1,
        n_grid=2,
        logM_max=12.0,
        z_start_max=20.0,
        plot_z_min=None,
        plot_z_max=None,
        sfrd_ymin=None,
        sfrd_ymax=None,
        lw_ymin=None,
        lw_ymax=None,
        random_seed=123,
        disable_time_delay=False,
        fixed_lw_j21_values="0.1",
        lw_horizon_fraction=0.2,
        lw_proxy_dense_size=128,
        lw_support_dz=None,
        lw_support_max_points=512,
        output_prefix=tmp_path / "lw",
        slide_output=tmp_path / "slide.pdf",
        no_slide_output=True,
    )
    integration_calls: list[tuple[float, int]] = []
    written_points: list[lw_script.SFRDPoint] = []
    saved_arrays: dict[str, np.ndarray] = {}
    plotted_z: list[np.ndarray] = []
    csv_provenance: list[lw_script.SFRDRunProvenance] = []

    def fake_integrate(
        *,
        z: float,
        base_seed: int,
        popiii_sfr_parameters: object,
        **_: object,
    ) -> lw_script.SFRDPoint:
        integration_calls.append((float(z), int(base_seed)))
        lw_value = float(getattr(popiii_sfr_parameters, "lw_background_j21"))
        scenario = lw_script._scenario_key(lw_value)
        return lw_script.SFRDPoint(
            z=float(z),
            scenario=scenario,
            lw_background_j21=lw_value,
            rho_sfr_popii=1.0 + float(z),
            rho_sfr_popiii=0.1 + float(z),
            popiii_minimum_mass_msun=1.0e6,
        )

    monkeypatch.setattr(lw_script, "_parse_args", lambda: args)
    monkeypatch.setattr(lw_script, "_integrate_sfrd_at_z", fake_integrate)
    monkeypatch.setattr(
        lw_script,
        "_write_csv",
        lambda _path, points, _lw, *, provenance: (
            written_points.extend(points),
            csv_provenance.append(provenance),
        ),
    )
    monkeypatch.setattr(
        lw_script.np,
        "savez",
        lambda _path, **arrays: saved_arrays.update(
            {name: np.asarray(values).copy() for name, values in arrays.items()}
        ),
    )
    monkeypatch.setattr(
        lw_script,
        "_plot",
        lambda **kwargs: plotted_z.append(np.asarray(kwargs["z"]).copy()),
    )

    lw_script.main()

    support_z = lw_script._build_lw_support_grid(
        requested_z,
        horizon_fraction=0.2,
        z_start_max=20.0,
    )
    assert {z for z, _seed in integration_calls} == set(support_z)
    assert len(integration_calls) == 2 * support_z.size
    assert {seed for _z, seed in integration_calls} == {123}
    assert {point.z for point in written_points} == set(requested_z)
    np.testing.assert_array_equal(saved_arrays["requested_z"], requested_z)
    np.testing.assert_array_equal(saved_arrays["lw_support_z"], support_z)
    np.testing.assert_array_equal(
        saved_arrays["popii_support_sfrd_msun_yr_mpc3"],
        1.0 + support_z,
    )
    assert saved_arrays["schema_version"].item() == "auroralf_sfrd_lw_proxy_v1"
    assert saved_arrays["provenance_base_seed"].item() == 123
    assert csv_provenance[0].base_seed == 123
    assert csv_provenance[0].lw_support_dz == pytest.approx(1.0)
    np.testing.assert_array_equal(plotted_z[0], requested_z)


def test_sfrd_integration_derives_hmf_and_mah_seeds_from_redshift_and_mass_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_halo_mass: list[np.ndarray] = []
    captured_mah_seeds: list[int] = []

    monkeypatch.setattr(
        lw_script,
        "compute_popiii_lw_minimum_mass_msun",
        lambda *_args, **_kwargs: 1.0e6,
    )
    monkeypatch.setattr(
        lw_script,
        "compute_halo_mass_function_dndm",
        lambda halo_mass, *_args, **_kwargs: captured_halo_mass.append(np.asarray(halo_mass).copy())
        or np.ones_like(halo_mass),
    )
    monkeypatch.setattr(
        lw_script,
        "cosmo_age_gyr",
        lambda redshift, _cosmology: 0.0 if float(redshift) == 20.0 else 1.0,
    )

    def fake_histories(**kwargs: object) -> SimpleNamespace:
        captured_mah_seeds.append(int(kwargs["random_seed"]))
        return SimpleNamespace(metadata={"grid_size": 2}, tracks={})

    monkeypatch.setattr(lw_script, "generate_halo_histories", fake_histories)
    monkeypatch.setattr(
        lw_script,
        "compute_sfr_from_tracks",
        lambda *_args, **_kwargs: {
            "Mh": np.ones(2),
            "dMh_dt_sfr": np.ones(2),
            "z": np.array([20.0, 10.0]),
            "active_flag": np.ones(2, dtype=bool),
            "SFR": np.ones(2),
        },
    )
    monkeypatch.setattr(
        lw_script,
        "compute_popiii_sfr_from_grids",
        lambda **_kwargs: SimpleNamespace(sfr_grid=np.ones((1, 2))),
    )

    base_seed = 123
    _integrate_sfrd_at_z(
        cosmology=Cosmology(),
        z=10.0,
        popiii_sfr_parameters=lw_script.PopIIISFRParameters(lw_background_j21=0.0),
        N_mass=2,
        n_tracks=1,
        n_grid=2,
        logM_max=8.0,
        z_start_max=20.0,
        base_seed=base_seed,
        enable_time_delay=False,
        mass_function_model=lw_script.DEFAULT_MASS_FUNCTION_MODEL,
        hmf_dlog10m=lw_script.DEFAULT_HMF_DLOG10M,
    )

    expected_rng = np.random.default_rng(derive_hmf_mass_seed(base_seed, 10.0))
    expected_halo_mass = np.power(10.0, expected_rng.uniform(6.0, 8.0, size=2))
    np.testing.assert_allclose(captured_halo_mass[0], expected_halo_mass)
    assert captured_mah_seeds == [
        derive_pipeline_random_seeds(base_seed, redshift=10.0, mass_index=index).mah
        for index in range(2)
    ]


def test_sfrd_artifacts_record_support_arrays_units_and_run_provenance(tmp_path: Path) -> None:
    assert hasattr(lw_script, "SFRDRunProvenance")
    assert hasattr(lw_script, "_build_npz_payload")
    baseline_scenario = lw_script._scenario_key(0.0)
    scenario = lw_script._scenario_key(0.1)
    requested_z = np.array([10.0, 11.0])
    support_z = np.array([10.0, 11.0, 12.0, 13.4])
    popii_requested = np.array([1.0, 2.0])
    popii_support = np.array([1.0, 2.0, 3.0, 4.0])
    popiii_requested = {
        baseline_scenario: np.array([0.05, 0.06]),
        scenario: np.array([0.1, 0.2]),
    }
    popiii_support = {
        baseline_scenario: np.array([0.05, 0.06, 0.07, 0.08]),
        scenario: np.array([0.1, 0.2, 0.3, 0.4]),
    }
    proxies = {
        "popii": np.array([1.0e8, 2.0e8]),
        baseline_scenario: np.array([5.0e6, 6.0e6]),
        scenario: np.array([1.0e7, 2.0e7]),
    }
    provenance = lw_script.SFRDRunProvenance(
        schema_version="auroralf_sfrd_lw_proxy_v1",
        base_seed=123,
        N_mass=2,
        n_tracks=1,
        n_grid=80,
        logM_max=12.0,
        z_start_max=20.0,
        lw_horizon_fraction=0.2,
        lw_proxy_dense_size=4096,
        lw_support_dz=1.0,
        lw_support_max_points=512,
        cosmology_h0_gyr_inv=0.068,
        cosmology_h0_km_s_mpc=67.4,
        cosmology_omega_m=0.315,
        cosmology_omega_b=0.04897,
        cosmology_omega_lambda=0.685,
        mass_function_model="Reed07",
        hmf_dlog10m=0.01,
        fixed_lw_j21_values=(0.1,),
        enable_time_delay=True,
    )
    payload = lw_script._build_npz_payload(
        requested_z=requested_z,
        support_z=support_z,
        popii_requested_sfrd=popii_requested,
        popii_support_sfrd=popii_support,
        popiii_requested_sfrd_by_scenario=popiii_requested,
        popiii_support_sfrd_by_scenario=popiii_support,
        lw_proxy_by_series=proxies,
        provenance=provenance,
    )
    npz_path = tmp_path / "sfrd.npz"
    np.savez(npz_path, **payload)

    with np.load(npz_path, allow_pickle=False) as loaded:
        assert all(loaded[name].dtype != object for name in loaded.files)
        expected_provenance_keys = {
            "provenance_base_seed",
            "provenance_N_mass",
            "provenance_n_tracks",
            "provenance_n_grid",
            "provenance_logM_max",
            "provenance_z_start_max",
            "provenance_lw_horizon_fraction",
            "provenance_lw_proxy_dense_size",
            "provenance_lw_support_dz",
            "provenance_lw_support_max_points",
            "provenance_cosmology_h0_gyr_inv",
            "provenance_cosmology_h0_km_s_mpc",
            "provenance_cosmology_omega_m",
            "provenance_cosmology_omega_b",
            "provenance_cosmology_omega_lambda",
            "provenance_mass_function_model",
            "provenance_hmf_dlog10m",
            "provenance_fixed_lw_background_j21",
            "provenance_enable_time_delay",
        }
        assert expected_provenance_keys <= set(loaded.files)
        assert loaded["schema_version"].item() == "auroralf_sfrd_lw_proxy_v1"
        np.testing.assert_array_equal(loaded["requested_z"], requested_z)
        np.testing.assert_array_equal(loaded["lw_support_z"], support_z)
        np.testing.assert_array_equal(
            loaded["popii_support_sfrd_msun_yr_mpc3"],
            popii_support,
        )
        np.testing.assert_array_equal(
            loaded[f"{scenario}_support_sfrd_msun_yr_mpc3"],
            popiii_support[scenario],
        )
        assert loaded["provenance_base_seed"].item() == 123
        assert loaded["provenance_enable_time_delay"].item()
        np.testing.assert_array_equal(
            loaded["provenance_fixed_lw_background_j21"],
            np.array([0.1]),
        )

    points = [
        lw_script.SFRDPoint(10.0, "popiii_no_external_lw", 0.0, 1.0, 0.05, 1.0e6),
        lw_script.SFRDPoint(11.0, "popiii_no_external_lw", 0.0, 2.0, 0.06, 1.1e6),
        lw_script.SFRDPoint(10.0, scenario, 0.1, 1.0, 0.1, 2.0e6),
        lw_script.SFRDPoint(11.0, scenario, 0.1, 2.0, 0.2, 2.1e6),
    ]
    csv_path = tmp_path / "sfrd.csv"
    lw_script._write_csv(csv_path, points, proxies, provenance=provenance)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    expected_csv_provenance_columns = {
        "schema_version",
        "base_seed",
        "N_mass",
        "n_tracks",
        "n_grid",
        "logM_max",
        "z_start_max",
        "lw_horizon_fraction",
        "lw_proxy_dense_size",
        "lw_support_dz",
        "lw_support_max_points",
        "cosmology_h0_gyr_inv",
        "cosmology_h0_km_s_mpc",
        "cosmology_omega_m",
        "cosmology_omega_b",
        "cosmology_omega_lambda",
        "mass_function_model",
        "hmf_dlog10m",
        "fixed_lw_background_j21",
        "enable_time_delay",
    }
    assert expected_csv_provenance_columns <= set(rows[0])
    assert "cumulative_formed_mass_proxy_msun_Mpc^-3" in rows[0]
    assert "lw_proxy_msun_Mpc^-3" not in rows[0]
    assert {row["schema_version"] for row in rows} == {"auroralf_sfrd_lw_proxy_v1"}
    assert {row["base_seed"] for row in rows} == {"123"}
    assert {row["mass_function_model"] for row in rows} == {"Reed07"}


@pytest.mark.parametrize(
    ("requested_z", "support_z", "message"),
    [
        (
            np.array([10.0, 11.0]),
            np.array([10.0, 10.5, 13.4]),
            "requested_z must be an exact float64 subset of support_z",
        ),
        (
            np.array([10.0, 11.0]),
            np.array([10.0, 11.0, 12.0, 13.3]),
            r"LW support is insufficient for artifact provenance: required zmax=13\.4, provided zmax=13\.3",
        ),
    ],
)
def test_npz_payload_rejects_support_inconsistent_with_requested_horizon(
    requested_z: np.ndarray,
    support_z: np.ndarray,
    message: str,
) -> None:
    baseline = lw_script._scenario_key(0.0)
    provenance = lw_script.SFRDRunProvenance(
        schema_version="auroralf_sfrd_lw_proxy_v1",
        base_seed=123,
        N_mass=2,
        n_tracks=1,
        n_grid=80,
        logM_max=12.0,
        z_start_max=20.0,
        lw_horizon_fraction=0.2,
        lw_proxy_dense_size=4096,
        lw_support_dz=1.0,
        lw_support_max_points=512,
        cosmology_h0_gyr_inv=0.068,
        cosmology_h0_km_s_mpc=67.4,
        cosmology_omega_m=0.315,
        cosmology_omega_b=0.04897,
        cosmology_omega_lambda=0.685,
        mass_function_model="Reed07",
        hmf_dlog10m=0.01,
        fixed_lw_j21_values=(0.1,),
        enable_time_delay=True,
    )

    with pytest.raises(ValueError, match=message):
        lw_script._build_npz_payload(
            requested_z=requested_z,
            support_z=support_z,
            popii_requested_sfrd=np.ones(requested_z.size),
            popii_support_sfrd=np.ones(support_z.size),
            popiii_requested_sfrd_by_scenario={baseline: np.ones(requested_z.size)},
            popiii_support_sfrd_by_scenario={baseline: np.ones(support_z.size)},
            lw_proxy_by_series={
                "popii": np.ones(requested_z.size),
                baseline: np.ones(requested_z.size),
            },
            provenance=provenance,
        )


def test_scenario_label_formats_fixed_lw_value() -> None:
    assert _scenario_label("popiii_fixed_lw_j21_0.1") == r"Pop III, fixed $J_{\rm LW,21}=0.1$"


def test_scenario_key_roundtrips_nearby_lw_values_without_collision() -> None:
    assert hasattr(lw_script, "_scenario_key")
    first_value = 0.1
    second_value = float(np.nextafter(first_value, np.inf))

    first_key = lw_script._scenario_key(first_value)
    second_key = lw_script._scenario_key(second_value)

    assert first_key != second_key
    prefix = "popiii_fixed_lw_j21_"
    assert float(first_key.removeprefix(prefix)) == first_value
    assert float(second_key.removeprefix(prefix)) == second_value


def test_resolve_log_ylim_uses_explicit_bounds() -> None:
    assert _resolve_log_ylim((1.0e-5, 1.0e-1), [np.array([1.0e-8, 1.0e-2])]) == pytest.approx(
        (1.0e-5, 1.0e-1)
    )
