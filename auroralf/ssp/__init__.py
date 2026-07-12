"""Utilities for reading and interpolating SSP spectra."""

from .convolution import (
    SSP_UV_LOOKBACK_MAX_MYR,
    compute_final_ssp_observable_from_sfr_grid,
    compute_halo_uv_luminosity,
    interpolate_ssp_luminosity,
)
from .heii1640 import (
    DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON,
    DEFAULT_POPIII_HEII1640_SSP_FILE,
    compute_final_ssp_heplus_rate_from_sfr_grid,
    compute_final_ssp_line_luminosity_from_sfr_grid,
    heii1640_luminosity_from_heplus_rate,
    load_popiii_heii1640_luminosity_table,
    load_popiii_heplus_ionizing_photon_table,
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
    "compute_final_ssp_heplus_rate_from_sfr_grid",
    "compute_final_ssp_line_luminosity_from_sfr_grid",
    "DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON",
    "DEFAULT_POPIII_HEII1640_SSP_FILE",
    "DEFAULT_POPIII_UV_SSP_FILE",
    "heii1640_luminosity_from_heplus_rate",
    "interpolate_ssp_luminosity",
    "interpolate_uv1600_luminosity_per_msun",
    "load_popiii_heii1640_luminosity_table",
    "load_popiii_heplus_ionizing_photon_table",
    "load_popiii_uv_luminosity_table",
    "load_uv1600_table",
]
