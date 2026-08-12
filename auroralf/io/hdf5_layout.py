"""Canonical HDF5 field names and units for AuroraLF UVLF artifacts."""

_ROOT_GROUPS = {"config", "provenance", "axes", "results", "diagnostics"}
_ROOT_ATTRS = {"schema_name", "schema_version"}
_RESULT_FIELDS = (
    "bin_edges_muv",
    "bin_centers_muv",
    "bin_width_mag",
    "raw_counts",
    "weighted_counts_per_mpc3",
    "weight_squared_counts_per_mpc6",
    "weighted_count_sigma_per_mpc3",
    "effective_counts",
    "phi_intrinsic_per_mpc3_per_mag",
    "phi_intrinsic_sigma_per_mpc3_per_mag",
    "phi_observed_per_mpc3_per_mag",
    "phi_observed_sigma_per_mpc3_per_mag",
)
_RESULT_UNITS = {
    "bin_edges_muv": "mag",
    "bin_centers_muv": "mag",
    "bin_width_mag": "mag",
    "raw_counts": "count",
    "weighted_counts_per_mpc3": "Mpc^-3",
    "weight_squared_counts_per_mpc6": "Mpc^-6",
    "weighted_count_sigma_per_mpc3": "Mpc^-3",
    "effective_counts": "count",
    "phi_intrinsic_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_intrinsic_sigma_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_observed_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_observed_sigma_per_mpc3_per_mag": "Mpc^-3 mag^-1",
}
_DIAGNOSTIC_FIELDS = (
    "sampling_seconds",
    "sample_count",
    "valid_sample_count",
    "topheavy_source_fraction",
    "popiii_source_fraction",
    "sfrd_msun_per_yr_per_mpc3",
    "popiii_sfrd_msun_per_yr_per_mpc3",
)
_DIAGNOSTIC_UNITS = {
    "sampling_seconds": "s",
    "sample_count": "count",
    "valid_sample_count": "count",
    "topheavy_source_fraction": "dimensionless",
    "popiii_source_fraction": "dimensionless",
    "sfrd_msun_per_yr_per_mpc3": "Msun yr^-1 Mpc^-3",
    "popiii_sfrd_msun_per_yr_per_mpc3": "Msun yr^-1 Mpc^-3",
}
_SAMPLE_FIELDS = (
    "mass_index",
    "track_index",
    "halo_mass_msun",
    "mass_weight_per_mpc3",
    "uv_luminosity_erg_per_s_hz",
    "muv",
    "sfr_msun_per_yr",
    "popiii_sfr_msun_per_yr",
)
_SAMPLE_UNITS = {
    "mass_index": "index",
    "track_index": "index",
    "halo_mass_msun": "Msun",
    "mass_weight_per_mpc3": "Mpc^-3",
    "uv_luminosity_erg_per_s_hz": "erg s^-1 Hz^-1",
    "muv": "mag",
    "sfr_msun_per_yr": "Msun yr^-1",
    "popiii_sfr_msun_per_yr": "Msun yr^-1",
}

__all__ = []
