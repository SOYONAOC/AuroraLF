from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auroralf.uvlf import pipeline as uv_pipeline
from auroralf.uvlf.imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMFTransitionParameters,
    compute_topheavy_source_flags,
    validate_imf_mode,
)
from auroralf.uvlf.pipeline import _compute_final_uv_luminosity_components_vectorized


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else PROJECT_ROOT / parsed


def test_default_ssp_paths_exist_in_repo() -> None:
    assert _repo_path(DEFAULT_CANONICAL_SSP_FILE).is_file()
    assert _repo_path(DEFAULT_MILD_TOPHEAVY_SSP_FILE).is_file()


def test_validate_imf_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="imf_mode"):
        validate_imf_mode("global_extreme_topheavy")


def test_legacy_z_mode_defaults_to_active_sources_without_redshift_gate() -> None:
    z_grid = np.array([[12.0, 10.0, 9.9, 12.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e11)
    active_grid = np.array([[True, True, True, False]])

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=None,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, True, True, False]]))


def test_historical_source_redshift_gate_can_be_enabled() -> None:
    z_grid = np.array([[12.0, 10.0, 9.9, 12.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e11)
    active_grid = np.array([[True, True, True, False]])

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            source_redshift_gate_enabled=True,
            metallicity_topheavy_max_zsun=None,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, True, False, False]]))


def test_mah_burst_mode_requires_fast_growth_time() -> None:
    z_grid = np.array([[12.0, 12.0, 12.0, 8.0]])
    mh_grid = np.array([[1.0e10, 1.0e10, 1.0e10, 1.0e10]])
    dmhdt_grid = np.array([[1.0e11, 2.0e10, 0.0, 1.0e11]])
    active_grid = np.ones_like(z_grid, dtype=bool)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            growth_time_threshold_myr=200.0,
            metallicity_topheavy_max_zsun=None,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, False, False, True]]))


def test_canonical_mode_returns_no_topheavy_flags() -> None:
    z_grid = np.array([[20.0, 15.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e12)
    active_grid = np.ones_like(z_grid, dtype=bool)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_CANONICAL,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
    )

    np.testing.assert_array_equal(flags, np.zeros_like(active_grid, dtype=bool))


@pytest.mark.parametrize("invalid_rate", [np.nan, np.inf, -np.inf])
def test_topheavy_flags_reject_nonfinite_effective_accretion_rate(invalid_rate: float) -> None:
    with pytest.raises(ValueError, match="dmhdt_sfr_grid.*finite"):
        compute_topheavy_source_flags(
            imf_mode=IMF_MODE_CANONICAL,
            z_grid=np.array([[10.0]]),
            mh_grid=np.array([[1.0e10]]),
            dmhdt_sfr_grid=np.array([[invalid_rate]]),
            active_grid=np.array([[True]]),
        )


def test_z_gated_mode_requires_low_birth_metallicity_when_configured() -> None:
    z_grid = np.array([[12.0, 12.0, 12.0, 9.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e11)
    active_grid = np.ones_like(z_grid, dtype=bool)
    birth_metallicity = np.array([[0.01, 0.05, 0.051, 0.01]], dtype=float)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
        birth_metallicity_zsun_grid=birth_metallicity,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, True, False, True]]))


def test_mah_burst_mode_requires_growth_and_low_birth_metallicity() -> None:
    z_grid = np.array([[12.0, 12.0, 12.0, 12.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.array([[1.0e11, 1.0e11, 2.0e10, 1.0e11]])
    active_grid = np.ones_like(z_grid, dtype=bool)
    birth_metallicity = np.array([[0.01, 0.08, 0.01, 0.05]], dtype=float)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        active_grid=active_grid,
        birth_metallicity_zsun_grid=birth_metallicity,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            growth_time_threshold_myr=200.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, False, False, True]]))


def test_metallicity_gate_requires_birth_metallicity_grid() -> None:
    z_grid = np.array([[12.0, 12.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e11)
    active_grid = np.ones_like(z_grid, dtype=bool)

    with pytest.raises(ValueError, match="birth_metallicity_zsun_grid"):
        compute_topheavy_source_flags(
            imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
            z_grid=z_grid,
            mh_grid=mh_grid,
            dmhdt_sfr_grid=dmhdt_grid,
            active_grid=active_grid,
            transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        )


def test_variable_imf_convolution_separates_canonical_and_topheavy_components() -> None:
    t_grid = np.array([[0.0, 0.05, 0.1]])
    sfr_grid = np.array([[1.0, 1.0, 1.0]])
    active_grid = np.ones_like(sfr_grid, dtype=bool)
    ssp_age_grid = np.array([1.0e-3, 1000.0])
    canonical_luv_grid = np.array([1.0, 1.0])
    topheavy_luv_grid = np.array([10.0, 10.0])

    canonical_only, topheavy_only = _compute_final_uv_luminosity_components_vectorized(
        t_grid=t_grid,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        topheavy_source_flag_grid=np.zeros_like(active_grid, dtype=bool),
        ssp_age_grid=ssp_age_grid,
        ssp_luv_grid=canonical_luv_grid,
        topheavy_ssp_age_grid=ssp_age_grid,
        topheavy_ssp_luv_grid=topheavy_luv_grid,
        ssp_lookback_max_myr=1000.0,
    )
    np.testing.assert_allclose(canonical_only, np.array([1.0e8]))
    np.testing.assert_allclose(topheavy_only, np.array([0.0]))

    canonical_only, topheavy_only = _compute_final_uv_luminosity_components_vectorized(
        t_grid=t_grid,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        topheavy_source_flag_grid=np.ones_like(active_grid, dtype=bool),
        ssp_age_grid=ssp_age_grid,
        ssp_luv_grid=canonical_luv_grid,
        topheavy_ssp_age_grid=ssp_age_grid,
        topheavy_ssp_luv_grid=topheavy_luv_grid,
        ssp_lookback_max_myr=1000.0,
    )
    np.testing.assert_allclose(canonical_only, np.array([0.0]))
    np.testing.assert_allclose(topheavy_only, np.array([1.0e9]))


def test_variable_imf_uv_helper_delegates_component_masks_to_common_engine(monkeypatch) -> None:
    assert hasattr(uv_pipeline, "compute_final_ssp_observable_from_sfr_grid")
    calls: list[dict[str, object]] = []

    def fake_common_engine(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.array([float(len(calls))])

    monkeypatch.setattr(
        uv_pipeline,
        "compute_final_ssp_observable_from_sfr_grid",
        fake_common_engine,
    )
    active_grid = np.ones((1, 3), dtype=bool)
    topheavy_grid = np.array([[False, True, False]])
    canonical_age_myr = np.array([1.0e-2, 100.0])
    topheavy_age_myr = np.array([2.0e-2, 80.0])

    canonical, topheavy = _compute_final_uv_luminosity_components_vectorized(
        t_grid=np.array([[0.0, 0.05, 0.1]]),
        sfr_grid=np.ones((1, 3)),
        active_grid=active_grid,
        topheavy_source_flag_grid=topheavy_grid,
        ssp_age_grid=canonical_age_myr,
        ssp_luv_grid=np.array([1.0, 2.0]),
        topheavy_ssp_age_grid=topheavy_age_myr,
        topheavy_ssp_luv_grid=np.array([10.0, 20.0]),
        ssp_lookback_max_myr=100.0,
    )

    np.testing.assert_array_equal(canonical, np.array([1.0]))
    np.testing.assert_array_equal(topheavy, np.array([2.0]))
    np.testing.assert_array_equal(
        calls[0]["active_grid"],
        np.array([[True, False, True]]),
    )
    np.testing.assert_array_equal(
        calls[1]["active_grid"],
        np.array([[False, True, False]]),
    )
    np.testing.assert_array_equal(calls[0]["ssp_age_myr"], canonical_age_myr)
    np.testing.assert_array_equal(calls[1]["ssp_age_myr"], topheavy_age_myr)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("t_grid", [[0.0, True, 0.1]]),
        ("sfr_grid", [[1.0, False, 1.0]]),
        ("ssp_age_grid", [1.0e-2, np.bool_(True)]),
        ("ssp_luv_grid", [1.0, False]),
        ("topheavy_ssp_age_grid", [2.0e-2, True]),
        ("topheavy_ssp_luv_grid", [10.0, np.bool_(False)]),
        ("ssp_lookback_max_myr", True),
    ],
)
def test_variable_imf_uv_helper_rejects_boolean_numeric_inputs_before_cast(
    name: str,
    value: object,
) -> None:
    inputs: dict[str, object] = {
        "t_grid": np.array([[0.0, 0.05, 0.1]]),
        "sfr_grid": np.ones((1, 3)),
        "active_grid": np.ones((1, 3), dtype=bool),
        "topheavy_source_flag_grid": np.array([[False, True, False]]),
        "ssp_age_grid": np.array([1.0e-2, 100.0]),
        "ssp_luv_grid": np.array([1.0, 2.0]),
        "topheavy_ssp_age_grid": np.array([2.0e-2, 80.0]),
        "topheavy_ssp_luv_grid": np.array([10.0, 20.0]),
        "ssp_lookback_max_myr": 100.0,
    }
    inputs[name] = value

    with pytest.raises(ValueError, match="boolean"):
        _compute_final_uv_luminosity_components_vectorized(**inputs)  # type: ignore[arg-type]
