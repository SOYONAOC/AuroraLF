"""Utilities for reading and interpolating SSP spectra."""

from .convolution import (
    SSP_UV_LOOKBACK_MAX_MYR,
    compute_final_ssp_observable_from_sfr_grid,
    compute_halo_uv_luminosity,
    interpolate_ssp_luminosity,
)
from .uv1600 import (
    DEFAULT_POPIII_UV_SSP_FILE,
    interpolate_uv1600_luminosity_per_msun,
    load_popiii_uv_luminosity_table,
    load_uv1600_table,
)

__all__ = [
    "SSP_UV_LOOKBACK_MAX_MYR",
    "compute_final_ssp_observable_from_sfr_grid",
    "compute_halo_uv_luminosity",
    "DEFAULT_POPIII_UV_SSP_FILE",
    "interpolate_ssp_luminosity",
    "interpolate_uv1600_luminosity_per_msun",
    "load_popiii_uv_luminosity_table",
    "load_uv1600_table",
]
