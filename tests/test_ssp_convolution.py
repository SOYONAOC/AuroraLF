from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

import numpy as np
import pytest

from auroralf.ssp import convolution


def test_uv1600_npz_cache_reloads_same_path_atomic_replacement_and_hits_when_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auroralf.ssp import uv1600

    path = tmp_path / "canonical.npz"
    replacement = tmp_path / "replacement.npz"
    np.savez(
        path,
        ages_myr=np.array([1.0, 2.0]),
        luminosity_per_msun=np.array([10.0, 20.0]),
    )
    uv1600._load_uv1600_table_cached.cache_clear()
    real_load = uv1600._load_uv1600_table_from_npz
    reads: list[Path] = []

    def read_spy(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        reads.append(Path(file_path))
        return real_load(file_path)

    monkeypatch.setattr(uv1600, "_load_uv1600_table_from_npz", read_spy)
    _, first = uv1600.load_uv1600_table(path)
    _, unchanged = uv1600.load_uv1600_table(path)
    assert len(reads) == 1
    np.testing.assert_array_equal(unchanged, first)
    original_mtime_ns = path.stat().st_mtime_ns

    np.savez(
        replacement,
        ages_myr=np.array([1.0, 2.0]),
        luminosity_per_msun=np.array([30.0, 40.0]),
    )
    assert replacement.stat().st_size == path.stat().st_size
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, path)
    os.utime(path, ns=(original_mtime_ns, original_mtime_ns))

    _, replaced = uv1600.load_uv1600_table(path)

    assert len(reads) == 2
    np.testing.assert_array_equal(replaced, np.array([30.0, 40.0]))


def _valid_inputs() -> dict[str, object]:
    return {
        "t_grid_gyr": np.array([[0.0, 1.0e-3, 2.0e-3]]),
        "sfr_grid": np.ones((1, 3), dtype=float),
        "active_grid": np.ones((1, 3), dtype=bool),
        "ssp_age_myr": np.array([1.0e-2, 1.0, 2.0]),
        "ssp_observable_per_msun": np.ones(3, dtype=float),
        "lookback_max_myr": 10.0,
    }


def test_common_ssp_convolution_is_exported_from_public_ssp_api() -> None:
    import auroralf.ssp as ssp

    assert (
        ssp.compute_final_ssp_observable_from_sfr_grid
        is convolution.compute_final_ssp_observable_from_sfr_grid
    )


@pytest.mark.parametrize(
    ("observable_name", "kernel_scale"),
    [("uv", 2.0), ("heii1640", 3.0e34), ("q2", 4.0e45)],
)
def test_common_ssp_convolution_integrates_constant_kernels(
    observable_name: str,
    kernel_scale: float,
) -> None:
    assert hasattr(convolution, "compute_final_ssp_observable_from_sfr_grid")

    result = convolution.compute_final_ssp_observable_from_sfr_grid(
        **{
            **_valid_inputs(),
            "ssp_observable_per_msun": np.full(3, kernel_scale),
        }
    )

    assert observable_name
    np.testing.assert_allclose(result, np.array([2.0e6 * kernel_scale]))


def test_common_ssp_convolution_preserves_internal_zero_and_single_active_source_bin() -> None:
    result = convolution.compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=np.array([[0.0, 1.0e-3, 2.0e-3], [0.0, 1.0e-3, 2.0e-3]]),
        sfr_grid=np.array([[1.0, 1.0, 1.0], [5.0, 2.0, 5.0]]),
        active_grid=np.array([[True, False, True], [False, True, False]]),
        ssp_age_myr=np.array([1.0e-2, 2.0]),
        ssp_observable_per_msun=np.ones(2),
        lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(result, np.array([1.0e6, 2.0e6]))


def test_common_ssp_convolution_converts_gyr_lookback_age_to_myr_once() -> None:
    result = convolution.compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=np.array([[0.0, 1.0e-3, 2.0e-3]]),
        sfr_grid=np.ones((1, 3)),
        active_grid=np.ones((1, 3), dtype=bool),
        ssp_age_myr=np.array([1.0e-2, 1.0, 2.0]),
        ssp_observable_per_msun=np.array([10.0, 20.0, 30.0]),
        lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(result, np.array([4.0e7]))


def test_common_ssp_convolution_inserts_exact_lookback_boundary() -> None:
    result = convolution.compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=np.array([[0.0, 1.0e-3, 3.0e-3]]),
        sfr_grid=np.array([[0.0, 1.0, 1.0]]),
        active_grid=np.ones((1, 3), dtype=bool),
        ssp_age_myr=np.array([1.0e-2, 1.0]),
        ssp_observable_per_msun=np.ones(2),
        lookback_max_myr=1.0,
    )

    np.testing.assert_allclose(result, np.array([1.0e6]))


def _replace(name: str, value: object) -> Callable[[dict[str, object]], None]:
    def apply(values: dict[str, object]) -> None:
        values[name] = value

    return apply


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_replace("t_grid_gyr", np.empty((1, 0))), "non-empty 2D"),
        (_replace("sfr_grid", np.ones((1, 2))), "identical shapes"),
        (_replace("active_grid", np.ones((1, 3))), "boolean dtype"),
        (_replace("active_grid", np.array([[True, np.nan, False]], dtype=object)), "boolean dtype"),
        (_replace("t_grid_gyr", np.array([[0.0, np.nan, 1.0]])), "finite"),
        (_replace("t_grid_gyr", np.array([[0.0, 1.0, 1.0]])), "strictly increasing"),
        (_replace("sfr_grid", np.array([[1.0, np.nan, 1.0]])), "finite"),
        (_replace("sfr_grid", np.array([[1.0, -1.0, 1.0]])), "non-negative"),
        (_replace("ssp_age_myr", np.array([])), "non-empty 1D"),
        (_replace("ssp_age_myr", np.array([[1.0, 2.0]])), "non-empty 1D"),
        (_replace("ssp_observable_per_msun", np.ones(2)), "same length"),
        (_replace("ssp_age_myr", np.array([1.0e-2, np.nan, 2.0])), "finite"),
        (_replace("ssp_age_myr", np.array([0.0, 1.0, 2.0])), "strictly positive"),
        (_replace("ssp_age_myr", np.array([1.0e-2, 1.0, 1.0])), "strictly increasing"),
        (_replace("ssp_observable_per_msun", np.array([1.0, np.nan, 1.0])), "finite"),
        (_replace("ssp_observable_per_msun", np.array([1.0, -1.0, 1.0])), "non-negative"),
        (_replace("lookback_max_myr", 0.0), "lookback_max_myr must be finite and positive"),
        (_replace("lookback_max_myr", np.nan), "lookback_max_myr must be finite and positive"),
        (_replace("time_unit_in_years", 0.0), "time_unit_in_years must be finite and positive"),
        (_replace("time_unit_in_years", np.nan), "time_unit_in_years must be finite and positive"),
    ],
)
def test_common_ssp_convolution_rejects_invalid_inputs(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    values = _valid_inputs()
    mutate(values)

    with pytest.raises(ValueError, match=message):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)


def test_common_ssp_convolution_rejects_nonfinite_result() -> None:
    values = _valid_inputs()
    values["sfr_grid"] = np.full((1, 3), 1.0e308)
    values["ssp_observable_per_msun"] = np.full(3, 1.0e308)

    with pytest.raises(RuntimeError, match="final SSP observable must be finite and non-negative"):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("t_grid_gyr", [[0.0, True, 2.0e-3]]),
        ("sfr_grid", [[1.0, False, 1.0]]),
        ("ssp_age_myr", [1.0e-2, True, 2.0]),
        ("ssp_observable_per_msun", [1.0, np.bool_(True), 1.0]),
    ],
)
def test_common_ssp_convolution_rejects_boolean_members(
    name: str,
    value: object,
) -> None:
    values = _valid_inputs()
    values[name] = value

    with pytest.raises(ValueError, match=f"{name} must not contain boolean values"):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lookback_max_myr", True),
        ("lookback_max_myr", np.bool_(False)),
        ("time_unit_in_years", True),
        ("time_unit_in_years", np.bool_(False)),
    ],
)
def test_common_ssp_convolution_rejects_boolean_scalars(name: str, value: object) -> None:
    values = _valid_inputs()
    values[name] = value

    with pytest.raises(ValueError, match=f"{name} must not be boolean"):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)


def test_common_ssp_convolution_rejects_unrepresentable_lookback_conversion() -> None:
    values = _valid_inputs()
    values["lookback_max_myr"] = float(np.nextafter(0.0, 1.0))
    with pytest.raises(
        ValueError,
        match="lookback_max_myr cannot be represented as a positive Gyr interval",
    ):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)

    values = _valid_inputs()
    values["t_grid_gyr"] = np.array([[1.0e16 - 4.0, 1.0e16 - 2.0, 1.0e16]])
    values["lookback_max_myr"] = 1.0
    with pytest.raises(
        ValueError,
        match="lookback boundary is not representable at every observation time",
    ):
        convolution.compute_final_ssp_observable_from_sfr_grid(**values)


def test_common_ssp_convolution_accepts_short_representable_lookback() -> None:
    result = convolution.compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=np.array([[0.0, 5.0e-7, 1.0e-6]]),
        sfr_grid=np.ones((1, 3)),
        active_grid=np.ones((1, 3), dtype=bool),
        ssp_age_myr=np.array([1.0e-6, 1.0e-3]),
        ssp_observable_per_msun=np.ones(2),
        lookback_max_myr=1.0e-3,
    )

    np.testing.assert_allclose(result, np.array([1.0e3]))


def test_legacy_halo_uv_convolution_delegates_to_common_engine_with_myr_ssp_ages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_common_engine(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.array([123.0])

    monkeypatch.setattr(
        convolution,
        "compute_final_ssp_observable_from_sfr_grid",
        fake_common_engine,
    )

    details = convolution.compute_halo_uv_luminosity(
        t_obs=2.0e-3,
        t_history=np.array([0.0, 1.0e-3, 2.0e-3]),
        mh_history=np.ones(3),
        sfr_history=np.ones(3),
        ssp_age_grid=np.array([1.0e-5, 2.0e-3]),
        ssp_luv_grid=np.array([10.0, 20.0]),
        M_min=0.0,
        t_z50=0.0,
        ssp_lookback_max_myr=10.0,
        return_details=True,
    )

    assert details["L_uv_halo"] == pytest.approx(123.0)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["ssp_age_myr"], np.array([1.0e-2, 2.0]))
    np.testing.assert_array_equal(calls[0]["active_grid"], np.ones((1, 3), dtype=bool))


def test_legacy_halo_uv_convolution_preserves_mass_crossing_and_details() -> None:
    details = convolution.compute_halo_uv_luminosity(
        t_obs=3.0e-3,
        t_history=np.array([0.0, 1.0e-3, 2.0e-3, 3.0e-3]),
        mh_history=np.array([0.0, 0.0, 2.0, 3.0]),
        sfr_history=np.ones(4),
        ssp_age_grid=np.array([1.0e-5, 1.0e-2]),
        ssp_luv_grid=np.ones(2),
        M_min=1.0,
        t_z50=5.0e-4,
        ssp_lookback_max_myr=10.0,
        return_details=True,
    )

    assert details["t_cross_Mmin"] == pytest.approx(1.5e-3)
    assert details["ti"] == pytest.approx(1.5e-3)
    assert details["L_uv_halo"] == pytest.approx(1.5e6)
    np.testing.assert_allclose(details["t_used"], np.array([1.5e-3, 2.0e-3, 3.0e-3]))


def test_legacy_boundaries_are_exact_and_details_integrate_to_common_result() -> None:
    lower = 1.000005
    upper = 1.001995
    t_history = np.array([1.0, lower, 1.001, upper, 1.002])
    sfr_history = np.array([9.0, 1.0, 2.0, 3.0, 9.0])

    t_used_direct, _ = convolution._augment_with_boundaries(
        t_history,
        sfr_history,
        lower=lower,
        upper=upper,
    )
    np.testing.assert_array_equal(t_used_direct, np.array([lower, 1.001, upper]))

    details = convolution.compute_halo_uv_luminosity(
        t_obs=upper,
        t_history=t_history,
        mh_history=np.ones(t_history.size),
        sfr_history=sfr_history,
        ssp_age_grid=np.array([1.0e-8, 1.0e-3, 2.0e-3]),
        ssp_luv_grid=np.array([1.0, 3.0, 7.0]),
        M_min=0.0,
        t_z50=lower,
        ssp_lookback_max_myr=10.0,
        return_details=True,
    )

    np.testing.assert_array_equal(details["t_used"], np.array([lower, 1.001, upper]))
    assert details["t_used"][-1] == upper
    details_integral = np.trapezoid(
        details["integrand_used"],
        x=details["t_used"] * 1.0e9,
    )
    assert details["L_uv_halo"] == pytest.approx(details_integral)


def _valid_legacy_inputs() -> dict[str, object]:
    return {
        "t_obs": 2.0e-3,
        "t_history": np.array([0.0, 1.0e-3, 2.0e-3]),
        "mh_history": np.ones(3),
        "sfr_history": np.ones(3),
        "ssp_age_grid": np.array([1.0e-5, 1.0e-3, 2.0e-3]),
        "ssp_luv_grid": np.ones(3),
        "M_min": 0.0,
        "t_z50": 0.0,
        "time_unit_in_years": 1.0e9,
        "ssp_lookback_max_myr": 10.0,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("t_history", [0.0, True, 2.0e-3]),
        ("mh_history", [1.0, False, 1.0]),
        ("sfr_history", [1.0, True, 1.0]),
        ("ssp_age_grid", [1.0e-5, np.bool_(True), 2.0e-3]),
        ("ssp_luv_grid", [1.0, False, 1.0]),
        ("t_obs", True),
        ("M_min", np.bool_(True)),
        ("t_z50", False),
        ("time_unit_in_years", True),
        ("ssp_lookback_max_myr", np.bool_(False)),
    ],
)
def test_legacy_halo_uv_convolution_rejects_boolean_numeric_inputs(
    name: str,
    value: object,
) -> None:
    values = _valid_legacy_inputs()
    values[name] = value

    with pytest.raises(ValueError, match="boolean"):
        convolution.compute_halo_uv_luminosity(**values)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("t_history", np.array([0.0, np.nan, 2.0e-3]), "t_history.*finite"),
        ("mh_history", np.array([1.0, np.inf, 1.0]), "mh_history.*finite"),
        ("sfr_history", np.array([1.0, np.nan, 1.0]), "sfr_history.*finite"),
        ("sfr_history", np.array([1.0, -1.0, 1.0]), "sfr_history.*non-negative"),
        ("ssp_age_grid", np.array([1.0e-5, np.nan, 2.0e-3]), "ssp_age_grid.*finite"),
        ("ssp_age_grid", np.array([0.0, 1.0e-3, 2.0e-3]), "ssp_age_grid.*positive"),
        ("ssp_age_grid", np.array([1.0e-5, 1.0e-5, 2.0e-3]), "ssp_age_grid.*increasing"),
        ("ssp_luv_grid", np.array([1.0, np.nan, 1.0]), "ssp_luv_grid.*finite"),
        ("ssp_luv_grid", np.array([1.0, -1.0, 1.0]), "ssp_luv_grid.*non-negative"),
        ("t_obs", np.nan, "t_obs.*finite"),
        ("M_min", np.nan, "M_min.*finite"),
        ("t_z50", np.inf, "t_z50.*finite"),
        ("time_unit_in_years", np.nan, "time_unit_in_years.*finite and positive"),
        ("ssp_lookback_max_myr", np.nan, "ssp_lookback_max_myr.*finite and positive"),
    ],
)
def test_legacy_halo_uv_convolution_rejects_invalid_physical_inputs(
    name: str,
    value: object,
    message: str,
) -> None:
    values = _valid_legacy_inputs()
    values[name] = value

    with pytest.raises(ValueError, match=message):
        convolution.compute_halo_uv_luminosity(**values)


def test_legacy_halo_uv_convolution_rejects_unrepresentable_lookbacks_before_early_return() -> None:
    values = _valid_legacy_inputs()
    values["ssp_lookback_max_myr"] = float(np.nextafter(0.0, 1.0))
    values["t_z50"] = values["t_obs"]
    with pytest.raises(
        ValueError,
        match="ssp_lookback_max_myr cannot be represented as a positive Gyr interval",
    ):
        convolution.compute_halo_uv_luminosity(**values)

    values = _valid_legacy_inputs()
    values["t_history"] = np.array([[1.0e16 - 4.0, 1.0e16 - 2.0, 1.0e16]]).ravel()
    values["t_obs"] = 1.0e16
    values["t_z50"] = 1.0e16
    values["ssp_lookback_max_myr"] = 1.0
    with pytest.raises(
        ValueError,
        match="ssp_lookback_max_myr lookback boundary is not representable",
    ):
        convolution.compute_halo_uv_luminosity(**values)


def test_parallel_uv_convolution_single_worker_keeps_legacy_ssp_age_units() -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    result = compute_uv_luminosities_parallel(
        t_grid=np.array([0.0, 1.0e-3, 2.0e-3]),
        mh_grid=np.ones((1, 3)),
        sfr_grid=np.ones((1, 3)),
        active_grid=np.ones((1, 3), dtype=bool),
        ssp_age_grid=np.array([1.0e-5, 2.0e-3]),
        ssp_luv_grid=np.ones(2),
        n_workers=1,
        ssp_lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(result, np.array([2.0e6]))


@pytest.mark.parametrize("n_workers", [1, 2])
def test_parallel_uv_convolution_preserves_inactive_knots_and_single_source_bin(
    n_workers: int,
) -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    t_grid = np.array([0.0, 1.0e-3, 2.0e-3])
    sfr_grid = np.array([[5.0, 2.0, 5.0], [5.0, 2.0, 5.0]])
    active_grid = np.array([[False, True, False], [False, True, False]])
    expected = convolution.compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=np.broadcast_to(t_grid, sfr_grid.shape),
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_myr=np.array([1.0e-2, 2.0]),
        ssp_observable_per_msun=np.ones(2),
        lookback_max_myr=10.0,
    )

    result = compute_uv_luminosities_parallel(
        t_grid=t_grid,
        mh_grid=np.ones_like(sfr_grid),
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_grid=np.array([1.0e-5, 2.0e-3]),
        ssp_luv_grid=np.ones(2),
        n_workers=n_workers,
        ssp_lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(expected, np.array([2.0e6, 2.0e6]))
    np.testing.assert_allclose(result, expected)


def test_parallel_uv_convolution_accepts_per_row_time_grids_without_compression() -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    t_grid = np.array([[0.0, 1.0e-3, 2.0e-3], [0.0, 2.0e-3, 4.0e-3]])
    sfr_grid = np.array([[5.0, 2.0, 5.0], [5.0, 2.0, 5.0]])
    active_grid = np.array([[False, True, False], [False, True, False]])

    result = compute_uv_luminosities_parallel(
        t_grid=t_grid,
        mh_grid=np.ones_like(sfr_grid),
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_grid=np.array([1.0e-5, 1.0e-2]),
        ssp_luv_grid=np.ones(2),
        n_workers=2,
        ssp_lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(result, np.array([2.0e6, 4.0e6]))


@pytest.mark.parametrize(
    "t_grid",
    [
        np.array([0.0, 1.0e-3]),
        np.ones((1, 3)),
        np.ones((2, 2, 2)),
    ],
)
def test_parallel_uv_convolution_rejects_invalid_shared_or_row_time_shape(
    t_grid: np.ndarray,
) -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    with pytest.raises(ValueError, match="t_grid must be shared 1D or match the 2D SFR grid"):
        compute_uv_luminosities_parallel(
            t_grid=t_grid,
            mh_grid=np.ones((2, 3)),
            sfr_grid=np.ones((2, 3)),
            active_grid=np.ones((2, 3), dtype=bool),
            ssp_age_grid=np.array([1.0e-5, 2.0e-3]),
            ssp_luv_grid=np.ones(2),
            n_workers=1,
            ssp_lookback_max_myr=10.0,
        )


@pytest.mark.parametrize("n_workers", [0, -1, True, np.bool_(False), 1.5])
def test_parallel_uv_convolution_requires_positive_nonboolean_integer_workers(
    n_workers: object,
) -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    with pytest.raises(ValueError, match="n_workers must be a positive non-boolean integer"):
        compute_uv_luminosities_parallel(
            t_grid=np.array([0.0, 1.0e-3]),
            mh_grid=np.ones((1, 2)),
            sfr_grid=np.ones((1, 2)),
            active_grid=np.ones((1, 2), dtype=bool),
            ssp_age_grid=np.array([1.0e-5, 1.0e-3]),
            ssp_luv_grid=np.ones(2),
            n_workers=n_workers,  # type: ignore[arg-type]
            ssp_lookback_max_myr=10.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("t_grid", [0.0, True, 2.0e-3]),
        ("mh_grid", [[1.0, False, 1.0]]),
        ("sfr_grid", [[1.0, True, 1.0]]),
        ("ssp_age_grid", [1.0e-5, np.bool_(True)]),
        ("ssp_luv_grid", [1.0, False]),
        ("ssp_lookback_max_myr", True),
    ],
)
def test_parallel_uv_convolution_rejects_boolean_numeric_inputs_before_cast(
    name: str,
    value: object,
) -> None:
    from auroralf.uvlf.pipeline import compute_uv_luminosities_parallel

    inputs: dict[str, object] = {
        "t_grid": np.array([0.0, 1.0e-3, 2.0e-3]),
        "mh_grid": np.ones((1, 3)),
        "sfr_grid": np.ones((1, 3)),
        "active_grid": np.ones((1, 3), dtype=bool),
        "ssp_age_grid": np.array([1.0e-5, 2.0e-3]),
        "ssp_luv_grid": np.ones(2),
        "n_workers": 1,
        "ssp_lookback_max_myr": 10.0,
    }
    inputs[name] = value

    with pytest.raises(ValueError, match="boolean"):
        compute_uv_luminosities_parallel(**inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("t_grid", [[0.0, True, 2.0e-3]]),
        ("mh_grid", [[1.0, False, 1.0]]),
        ("sfr_grid", [[1.0, True, 1.0]]),
        ("ssp_age_grid", [1.0e-5, np.bool_(True)]),
        ("ssp_luv_grid", [1.0, False]),
        ("ssp_lookback_max_myr", np.bool_(True)),
    ],
)
def test_uv_chunk_direct_entry_rejects_boolean_numeric_inputs(
    name: str,
    value: object,
) -> None:
    from auroralf.uvlf import pipeline

    values: dict[str, object] = {
        "t_grid": np.array([[0.0, 1.0e-3, 2.0e-3]]),
        "mh_grid": np.ones((1, 3)),
        "sfr_grid": np.ones((1, 3)),
        "active_grid": np.ones((1, 3), dtype=bool),
        "ssp_age_grid": np.array([1.0e-5, 2.0e-3]),
        "ssp_luv_grid": np.ones(2),
        "ssp_lookback_max_myr": 10.0,
    }
    values[name] = value
    pipeline._UV_WORKER_STATE["ssp_luv_grid"] = values["ssp_luv_grid"]
    args = (
        values["t_grid"],
        values["mh_grid"],
        values["sfr_grid"],
        values["active_grid"],
        values["ssp_age_grid"],
        values["ssp_lookback_max_myr"],
    )

    with pytest.raises(ValueError, match="boolean"):
        pipeline._compute_uv_chunk(args)  # type: ignore[arg-type]
