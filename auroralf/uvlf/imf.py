"""Archived Pop II IMF-gate implementation.

The SFRIIIII production configuration uses only the canonical IMF. These modes
and selectors remain importable for exact historical reproduction; high-level
run configuration requires an explicit ``enable_archived_imf_gate`` opt-in.

The gate thresholds and Boolean combinations are AuroraLF historical
heuristics, not direct equations from a paper.  The canonical SSP is BPASS
(Eldridge et al. 2017, DOI: 10.1017/pasa.2017.51; Stanway & Eldridge 2018,
DOI: 10.1093/mnras/sty1353; Byrne et al. 2022, DOI:
10.1093/mnras/stac807).  Those SSP references do not validate the archived
source-time gate itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from auroralf.model_options import (
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN,
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMF_MODES,
    IMFTransitionParameters,
    validate_imf_mode,
)

DEFAULT_CANONICAL_SSP_FILE = (
    "external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/"
    "spectra-bin-imf135_300.BASEL.z001.a+00.dat"
)
DEFAULT_MILD_TOPHEAVY_SSP_FILE = (
    "external_data/ssp_spectra/bpass_v2_2_1/imf100_300/"
    "SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5"
)
DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY = 0.05
def resolve_ssp_path(file_path: str | Path) -> Path:
    return Path(file_path).expanduser().resolve()


def requires_topheavy_ssp(imf_mode: str) -> bool:
    return validate_imf_mode(imf_mode) != IMF_MODE_CANONICAL


def compute_topheavy_source_flags(
    *,
    imf_mode: str,
    z_grid: np.ndarray,
    mh_grid: np.ndarray,
    dmhdt_sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    birth_metallicity_zsun_grid: np.ndarray | None = None,
    transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
) -> np.ndarray:
    """Return the archived AuroraLF heuristic top-heavy source-time mask."""

    mode = validate_imf_mode(imf_mode)
    z = np.asarray(z_grid, dtype=float)
    mh = np.asarray(mh_grid, dtype=float)
    dmhdt_sfr = np.asarray(dmhdt_sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)

    if mh.shape != dmhdt_sfr.shape or mh.shape != active.shape:
        raise ValueError("mh_grid, dmhdt_sfr_grid, and active_grid must have identical shapes")
    if not np.all(np.isfinite(dmhdt_sfr)):
        raise ValueError("dmhdt_sfr_grid must contain only finite values")
    if np.any(dmhdt_sfr < 0.0):
        raise ValueError("dmhdt_sfr_grid must be non-negative")
    if z.ndim == 1:
        if z.size != mh.shape[1]:
            raise ValueError("1D z_grid length must match the time axis of the halo grids")
        z = np.broadcast_to(z[None, :], mh.shape)
    elif z.shape != mh.shape:
        raise ValueError("z_grid must either be 1D over time or match mh_grid shape")

    if mode == IMF_MODE_CANONICAL:
        return np.zeros_like(active, dtype=bool)

    metallicity_max = transition_parameters.metallicity_topheavy_max_zsun
    if metallicity_max is None:
        metallicity_gate = np.ones_like(active, dtype=bool)
    else:
        if float(metallicity_max) <= 0.0:
            raise ValueError("metallicity_topheavy_max_zsun must be positive when provided")
        if birth_metallicity_zsun_grid is None:
            raise ValueError(
                "birth_metallicity_zsun_grid must be provided when metallicity_topheavy_max_zsun is set"
            )
        birth_metallicity = np.asarray(birth_metallicity_zsun_grid, dtype=float)
        if birth_metallicity.shape != mh.shape:
            raise ValueError("birth_metallicity_zsun_grid must match mh_grid shape")
        invalid_active_metallicity = active & (~np.isfinite(birth_metallicity) | (birth_metallicity < 0.0))
        if np.any(invalid_active_metallicity):
            raise ValueError("birth_metallicity_zsun_grid must be finite and non-negative for active sources")
        metallicity_gate = birth_metallicity <= float(metallicity_max)

    source_gate = active
    if transition_parameters.source_redshift_gate_enabled:
        source_gate = active & np.isfinite(z) & (z >= float(transition_parameters.z_topheavy_min))
    # Historical production gate, kept here for provenance but not used by default:
    # source_gate = active & np.isfinite(z) & (z >= float(transition_parameters.z_topheavy_min))
    if mode == IMF_MODE_Z_GATED_MILD_TOPHEAVY:
        return source_gate & metallicity_gate

    threshold_gyr = float(transition_parameters.growth_time_threshold_myr) / 1.0e3
    if threshold_gyr <= 0.0:
        raise ValueError("growth_time_threshold_myr must be positive")

    growth_time_gyr = np.full_like(mh, np.inf, dtype=float)
    positive_growth = np.isfinite(mh) & np.isfinite(dmhdt_sfr) & (mh > 0.0) & (dmhdt_sfr > 0.0)
    growth_time_gyr[positive_growth] = mh[positive_growth] / dmhdt_sfr[positive_growth]
    return source_gate & (growth_time_gyr <= threshold_gyr) & metallicity_gate
