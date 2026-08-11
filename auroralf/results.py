from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf._array_utils import (
    immutable_array as _immutable_array,
    validate_real_array_members as _validate_real_array_members,
)
from auroralf.uvlf.imf import validate_imf_mode


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and boolean values are not allowed")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and boolean values are not allowed")
    return int(value)


def _validate_integer_array_members(name: str, value: object) -> None:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            raise TypeError(f"{name} must contain integer non-boolean values")
        if np.issubdtype(value.dtype, np.integer):
            return
        if value.dtype == np.dtype(object):
            for item in value.flat:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
                    raise TypeError(f"{name} must contain integer non-boolean values")
            return
        raise TypeError(f"{name} must contain integer non-boolean values")
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
                raise TypeError(f"{name} must contain integer non-boolean values")
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain integer non-boolean values")


def _validate_boolean_array_members(name: str, value: object) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(bool):
            raise TypeError(f"{name} must have boolean dtype and boolean members")
        return
    if isinstance(value, (list, tuple)):
        if any(not isinstance(item, (bool, np.bool_)) for item in value):
            raise TypeError(f"{name} must have boolean dtype and boolean members")
        return
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must have boolean dtype and boolean members")


def _working_float_1d(name: str, value: object) -> np.ndarray:
    _validate_real_array_members(name, value)
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return array


def _working_sigma_1d(name: str, value: object) -> np.ndarray:
    array = _working_float_1d(name, value)
    if np.any(np.isinf(array)) or np.any(array[np.isfinite(array)] < 0.0):
        raise ValueError(f"{name} may contain NaN but must not contain infinity or negative finite values")
    return array


def _readonly_float_1d(name: str, value: object) -> np.ndarray:
    return _immutable_array(_working_float_1d(name, value))


def _readonly_int_1d(name: str, value: object) -> np.ndarray:
    _validate_integer_array_members(name, value)
    array = np.array(value, dtype=np.int64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return _immutable_array(array)


def _readonly_bool_1d(name: str, value: object) -> np.ndarray:
    _validate_boolean_array_members(name, value)
    array = np.array(value, dtype=bool, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return _immutable_array(array)


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _require_nonnegative(name: str, array: np.ndarray) -> None:
    _require_finite(name, array)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")


def _readonly_sigma_1d(name: str, value: object) -> np.ndarray:
    return _immutable_array(_working_sigma_1d(name, value))


@dataclass(frozen=True, slots=True)
class HaloTrackResult:
    halo_id: np.ndarray
    time_gyr: np.ndarray
    redshift: np.ndarray
    halo_mass_msun: np.ndarray
    dmh_dt_raw_msun_per_gyr: np.ndarray
    dmh_dt_sfr_msun_per_gyr: np.ndarray
    dmh_dt_clipped: np.ndarray
    sfr_msun_per_yr: np.ndarray
    active: np.ndarray
    birth_metallicity_zsun: np.ndarray | None
    gas_metallicity_zsun: np.ndarray | None

    def __post_init__(self) -> None:
        halo_id = _readonly_int_1d("halo_id", self.halo_id)
        time = _readonly_float_1d("time_gyr", self.time_gyr)
        redshift = _readonly_float_1d("redshift", self.redshift)
        halo_mass = _readonly_float_1d("halo_mass_msun", self.halo_mass_msun)
        raw_rate = _readonly_float_1d(
            "dmh_dt_raw_msun_per_gyr", self.dmh_dt_raw_msun_per_gyr
        )
        sfr_rate = _readonly_float_1d(
            "dmh_dt_sfr_msun_per_gyr", self.dmh_dt_sfr_msun_per_gyr
        )
        clipped = _readonly_bool_1d("dmh_dt_clipped", self.dmh_dt_clipped)
        sfr = _readonly_float_1d("sfr_msun_per_yr", self.sfr_msun_per_yr)
        active = _readonly_bool_1d("active", self.active)
        birth = (
            None
            if self.birth_metallicity_zsun is None
            else _readonly_float_1d(
                "birth_metallicity_zsun", self.birth_metallicity_zsun
            )
        )
        gas = (
            None
            if self.gas_metallicity_zsun is None
            else _readonly_float_1d("gas_metallicity_zsun", self.gas_metallicity_zsun)
        )
        arrays = [halo_id, time, redshift, halo_mass, raw_rate, sfr_rate, clipped, sfr, active]
        if birth is not None:
            arrays.append(birth)
        if gas is not None:
            arrays.append(gas)
        if any(array.size != time.size for array in arrays):
            raise ValueError("all HaloTrackResult arrays must have the same length")
        if np.any(halo_id < 0):
            raise ValueError("halo_id must be non-negative")
        _require_finite("time_gyr", time)
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time_gyr must be strictly increasing")
        _require_nonnegative("redshift", redshift)
        _require_finite("halo_mass_msun", halo_mass)
        if np.any(halo_mass <= 0.0):
            raise ValueError("halo_mass_msun must be positive")
        _require_finite("dmh_dt_raw_msun_per_gyr", raw_rate)
        _require_nonnegative("dmh_dt_sfr_msun_per_gyr", sfr_rate)
        expected_sfr_rate = np.maximum(raw_rate, 0.0)
        if not np.array_equal(sfr_rate, expected_sfr_rate):
            raise ValueError(
                "dmh_dt_sfr_msun_per_gyr must exactly equal "
                "maximum(dmh_dt_raw_msun_per_gyr, 0)"
            )
        _require_nonnegative("sfr_msun_per_yr", sfr)
        if birth is not None:
            _require_nonnegative("birth_metallicity_zsun", birth)
        if gas is not None:
            _require_nonnegative("gas_metallicity_zsun", gas)
        if not np.array_equal(clipped, raw_rate < 0.0):
            raise ValueError("dmh_dt_clipped must equal (dmh_dt_raw_msun_per_gyr < 0)")
        for name, value in (
            ("halo_id", halo_id),
            ("time_gyr", time),
            ("redshift", redshift),
            ("halo_mass_msun", halo_mass),
            ("dmh_dt_raw_msun_per_gyr", raw_rate),
            ("dmh_dt_sfr_msun_per_gyr", sfr_rate),
            ("dmh_dt_clipped", clipped),
            ("sfr_msun_per_yr", sfr),
            ("active", active),
            ("birth_metallicity_zsun", birth),
            ("gas_metallicity_zsun", gas),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class IMFModeResult:
    imf_mode: str
    bin_edges_muv: np.ndarray
    bin_centers_muv: np.ndarray
    bin_width_mag: np.ndarray
    raw_counts: np.ndarray
    weighted_counts_per_mpc3: np.ndarray
    weight_squared_counts_per_mpc6: np.ndarray
    weighted_count_sigma_per_mpc3: np.ndarray
    effective_counts: np.ndarray
    phi_intrinsic_per_mpc3_per_mag: np.ndarray
    phi_intrinsic_sigma_per_mpc3_per_mag: np.ndarray
    phi_observed_per_mpc3_per_mag: np.ndarray
    phi_observed_sigma_per_mpc3_per_mag: np.ndarray
    halo_tracks: tuple[HaloTrackResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        edges = _readonly_float_1d("bin_edges_muv", self.bin_edges_muv)
        centers = _readonly_float_1d("bin_centers_muv", self.bin_centers_muv)
        widths = _readonly_float_1d("bin_width_mag", self.bin_width_mag)
        raw_counts = _readonly_int_1d("raw_counts", self.raw_counts)
        weighted_counts = _readonly_float_1d(
            "weighted_counts_per_mpc3", self.weighted_counts_per_mpc3
        )
        weight_squared = _readonly_float_1d(
            "weight_squared_counts_per_mpc6", self.weight_squared_counts_per_mpc6
        )
        weighted_sigma = _readonly_sigma_1d(
            "weighted_count_sigma_per_mpc3", self.weighted_count_sigma_per_mpc3
        )
        effective = _readonly_float_1d("effective_counts", self.effective_counts)
        intrinsic_phi = _readonly_float_1d(
            "phi_intrinsic_per_mpc3_per_mag", self.phi_intrinsic_per_mpc3_per_mag
        )
        intrinsic_sigma = _readonly_sigma_1d(
            "phi_intrinsic_sigma_per_mpc3_per_mag",
            self.phi_intrinsic_sigma_per_mpc3_per_mag,
        )
        observed_phi = _readonly_float_1d(
            "phi_observed_per_mpc3_per_mag", self.phi_observed_per_mpc3_per_mag
        )
        observed_sigma = _readonly_sigma_1d(
            "phi_observed_sigma_per_mpc3_per_mag",
            self.phi_observed_sigma_per_mpc3_per_mag,
        )
        _require_finite("bin_edges_muv", edges)
        if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("bin_edges_muv must contain at least two strictly increasing edges")
        bin_count = edges.size - 1
        arrays = (
            centers,
            widths,
            raw_counts,
            weighted_counts,
            weight_squared,
            weighted_sigma,
            effective,
            intrinsic_phi,
            intrinsic_sigma,
            observed_phi,
            observed_sigma,
        )
        if any(array.size != bin_count for array in arrays):
            raise ValueError("all IMFModeResult bin arrays must match the bin count")
        expected_centers = 0.5 * (edges[:-1] + edges[1:])
        expected_widths = np.diff(edges)
        if not np.array_equal(centers, expected_centers):
            raise ValueError("bin_centers_muv must equal adjacent-edge midpoints")
        if not np.array_equal(widths, expected_widths):
            raise ValueError("bin_width_mag must equal np.diff(bin_edges_muv)")
        if np.any(raw_counts < 0):
            raise ValueError("raw_counts must be non-negative")
        for name, array in (
            ("weighted_counts_per_mpc3", weighted_counts),
            ("weight_squared_counts_per_mpc6", weight_squared),
            ("effective_counts", effective),
            ("phi_intrinsic_per_mpc3_per_mag", intrinsic_phi),
            ("phi_observed_per_mpc3_per_mag", observed_phi),
        ):
            _require_nonnegative(name, array)
        if type(self.halo_tracks) is not tuple:
            raise TypeError("halo_tracks must be a tuple")
        for track in self.halo_tracks:
            if type(track) is not HaloTrackResult:
                raise TypeError("halo_tracks entries must be exactly HaloTrackResult")
        for name, value in (
            ("imf_mode", mode),
            ("bin_edges_muv", edges),
            ("bin_centers_muv", centers),
            ("bin_width_mag", widths),
            ("raw_counts", raw_counts),
            ("weighted_counts_per_mpc3", weighted_counts),
            ("weight_squared_counts_per_mpc6", weight_squared),
            ("weighted_count_sigma_per_mpc3", weighted_sigma),
            ("effective_counts", effective),
            ("phi_intrinsic_per_mpc3_per_mag", intrinsic_phi),
            ("phi_intrinsic_sigma_per_mpc3_per_mag", intrinsic_sigma),
            ("phi_observed_per_mpc3_per_mag", observed_phi),
            ("phi_observed_sigma_per_mpc3_per_mag", observed_sigma),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class RedshiftResult:
    redshift: float
    imf_modes: tuple[IMFModeResult, ...]

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_modes) is not tuple:
            raise TypeError("imf_modes must be a tuple")
        if not self.imf_modes:
            raise ValueError("imf_modes must be non-empty")
        for result in self.imf_modes:
            if type(result) is not IMFModeResult:
                raise TypeError("imf_modes entries must be exactly IMFModeResult")
        names = tuple(result.imf_mode for result in self.imf_modes)
        if len(set(names)) != len(names):
            raise ValueError("imf_modes must contain unique mode names")
        object.__setattr__(self, "redshift", redshift)

    def for_mode(self, name: str) -> IMFModeResult:
        if type(name) is not str:
            raise TypeError("name must be a string")
        for result in self.imf_modes:
            if result.imf_mode == name:
                return result
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class ModeRunDiagnostics:
    redshift: float
    imf_mode: str
    sampling_seconds: float
    sample_count: int
    valid_sample_count: int
    topheavy_source_fraction: float
    popiii_source_fraction: float
    sfrd_msun_per_yr_per_mpc3: float
    popiii_sfrd_msun_per_yr_per_mpc3: float

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        seconds = _strict_float("sampling_seconds", self.sampling_seconds)
        sample_count = _strict_int("sample_count", self.sample_count)
        valid_count = _strict_int("valid_sample_count", self.valid_sample_count)
        topheavy_fraction = _strict_float(
            "topheavy_source_fraction", self.topheavy_source_fraction
        )
        popiii_fraction = _strict_float("popiii_source_fraction", self.popiii_source_fraction)
        sfrd = _strict_float("sfrd_msun_per_yr_per_mpc3", self.sfrd_msun_per_yr_per_mpc3)
        popiii_sfrd = _strict_float(
            "popiii_sfrd_msun_per_yr_per_mpc3",
            self.popiii_sfrd_msun_per_yr_per_mpc3,
        )
        if seconds < 0.0:
            raise ValueError("sampling_seconds must be non-negative")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not 0 <= valid_count <= sample_count:
            raise ValueError("valid_sample_count must lie in [0, sample_count]")
        if not 0.0 <= topheavy_fraction <= 1.0:
            raise ValueError("topheavy_source_fraction must lie in [0, 1]")
        if not 0.0 <= popiii_fraction <= 1.0:
            raise ValueError("popiii_source_fraction must lie in [0, 1]")
        if sfrd < 0.0:
            raise ValueError("sfrd_msun_per_yr_per_mpc3 must be non-negative")
        if popiii_sfrd < 0.0:
            raise ValueError("popiii_sfrd_msun_per_yr_per_mpc3 must be non-negative")
        for name, value in (
            ("redshift", redshift),
            ("imf_mode", mode),
            ("sampling_seconds", seconds),
            ("sample_count", sample_count),
            ("valid_sample_count", valid_count),
            ("topheavy_source_fraction", topheavy_fraction),
            ("popiii_source_fraction", popiii_fraction),
            ("sfrd_msun_per_yr_per_mpc3", sfrd),
            ("popiii_sfrd_msun_per_yr_per_mpc3", popiii_sfrd),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    total_seconds: float
    mode_runs: tuple[ModeRunDiagnostics, ...]

    def __post_init__(self) -> None:
        total = _strict_float("total_seconds", self.total_seconds)
        if total < 0.0:
            raise ValueError("total_seconds must be non-negative")
        if type(self.mode_runs) is not tuple:
            raise TypeError("mode_runs must be a tuple")
        for result in self.mode_runs:
            if type(result) is not ModeRunDiagnostics:
                raise TypeError("mode_runs entries must be exactly ModeRunDiagnostics")
        axes = tuple((result.redshift, result.imf_mode) for result in self.mode_runs)
        if len(set(axes)) != len(axes):
            raise ValueError("mode_runs must contain unique (redshift, imf_mode) pairs")
        object.__setattr__(self, "total_seconds", total)


@dataclass(frozen=True, slots=True)
class UVLFRunResult:
    config: UVLFRunConfig
    redshifts: tuple[RedshiftResult, ...]
    diagnostics: RunDiagnostics

    def __post_init__(self) -> None:
        if type(self.config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
        if type(self.redshifts) is not tuple:
            raise TypeError("redshifts must be a tuple")
        if type(self.diagnostics) is not RunDiagnostics:
            raise TypeError("diagnostics must be exactly RunDiagnostics")
        if len(self.redshifts) != len(self.config.redshifts):
            raise ValueError("result redshift count must match config.redshifts")
        for expected_redshift, redshift_result in zip(
            self.config.redshifts,
            self.redshifts,
            strict=True,
        ):
            if type(redshift_result) is not RedshiftResult:
                raise TypeError("redshifts entries must be exactly RedshiftResult")
            if redshift_result.redshift != expected_redshift:
                raise ValueError("result redshifts must exactly match config.redshifts")
            names = tuple(result.imf_mode for result in redshift_result.imf_modes)
            if names != self.config.stellar_population.imf_modes:
                raise ValueError("result imf_modes must exactly match configured imf_modes")
            configured_edges = np.asarray(self.config.sampling.muv_bin_edges, dtype=float)
            for mode_result in redshift_result.imf_modes:
                if not np.array_equal(mode_result.bin_edges_muv, configured_edges):
                    raise ValueError(
                        "result bin_edges_muv must exactly match "
                        "config.sampling.muv_bin_edges"
                    )
        expected_axes = tuple(
            (redshift, mode)
            for redshift in self.config.redshifts
            for mode in self.config.stellar_population.imf_modes
        )
        diagnostic_axes = tuple(
            (result.redshift, result.imf_mode) for result in self.diagnostics.mode_runs
        )
        if diagnostic_axes != expected_axes:
            raise ValueError("diagnostics mode runs must exactly match configured redshift/mode axes")

    def for_redshift(self, redshift: float) -> RedshiftResult:
        requested = _strict_float("redshift", redshift)
        for result in self.redshifts:
            if result.redshift == requested:
                return result
        raise KeyError(redshift)


__all__ = [
    "HaloTrackResult",
    "IMFModeResult",
    "ModeRunDiagnostics",
    "RedshiftResult",
    "RunDiagnostics",
    "UVLFRunResult",
]
