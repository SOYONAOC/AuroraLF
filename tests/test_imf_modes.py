from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_z_gated_mode_flags_only_active_high_redshift_sources() -> None:
    z_grid = np.array([[12.0, 10.0, 9.9, 12.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e11)
    active_grid = np.array([[True, True, True, False]])

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        active_grid=active_grid,
        transition_parameters=IMFTransitionParameters(z_topheavy_min=10.0),
    )

    np.testing.assert_array_equal(flags, np.array([[True, True, False, False]]))


def test_mah_burst_mode_requires_fast_growth_time() -> None:
    z_grid = np.array([[12.0, 12.0, 12.0, 8.0]])
    mh_grid = np.array([[1.0e10, 1.0e10, 1.0e10, 1.0e10]])
    dmhdt_grid = np.array([[1.0e11, 2.0e10, -1.0e11, 1.0e11]])
    active_grid = np.ones_like(z_grid, dtype=bool)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        active_grid=active_grid,
        transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            growth_time_threshold_myr=200.0,
        ),
    )

    np.testing.assert_array_equal(flags, np.array([[True, False, False, False]]))


def test_canonical_mode_returns_no_topheavy_flags() -> None:
    z_grid = np.array([[20.0, 15.0]])
    mh_grid = np.full_like(z_grid, 1.0e10)
    dmhdt_grid = np.full_like(z_grid, 1.0e12)
    active_grid = np.ones_like(z_grid, dtype=bool)

    flags = compute_topheavy_source_flags(
        imf_mode=IMF_MODE_CANONICAL,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        active_grid=active_grid,
    )

    np.testing.assert_array_equal(flags, np.zeros_like(active_grid, dtype=bool))


def test_variable_imf_convolution_separates_canonical_and_topheavy_components() -> None:
    t_grid = np.array([[0.0, 0.05, 0.1]])
    sfr_grid = np.array([[1.0, 1.0, 1.0]])
    active_grid = np.ones_like(sfr_grid, dtype=bool)
    ssp_age_grid = np.array([1.0e-6, 1.0])
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
