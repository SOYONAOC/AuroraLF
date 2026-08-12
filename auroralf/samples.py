"""Immutable in-memory halo sample records shared by compute and I/O layers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from auroralf.model_options import validate_imf_mode


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean value")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer non-boolean value")
    return int(value)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.flags.writeable = False
    return result


def _strict_float_array(name: str, value: object, *, allow_nan: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype == np.dtype(bool) or np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real non-boolean values")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinity")
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_array(array)


def _strict_int_array(name: str, value: object) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype == np.dtype(bool) or not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{name} must contain integer non-boolean values")
    array = np.array(value, dtype=np.int64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return _immutable_array(array)


@dataclass(frozen=True, slots=True)
class HaloSampleTable:
    redshift: float
    imf_mode: str
    mass_index: np.ndarray
    track_index: np.ndarray
    halo_mass_msun: np.ndarray
    mass_weight_per_mpc3: np.ndarray
    uv_luminosity_erg_per_s_hz: np.ndarray
    muv: np.ndarray
    sfr_msun_per_yr: np.ndarray
    popiii_sfr_msun_per_yr: np.ndarray

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        mass_index = _strict_int_array("mass_index", self.mass_index)
        track_index = _strict_int_array("track_index", self.track_index)
        halo_mass = _strict_float_array("halo_mass_msun", self.halo_mass_msun)
        mass_weight = _strict_float_array("mass_weight_per_mpc3", self.mass_weight_per_mpc3)
        luminosity = _strict_float_array(
            "uv_luminosity_erg_per_s_hz", self.uv_luminosity_erg_per_s_hz
        )
        muv = _strict_float_array("muv", self.muv, allow_nan=True)
        sfr = _strict_float_array("sfr_msun_per_yr", self.sfr_msun_per_yr)
        popiii_sfr = _strict_float_array(
            "popiii_sfr_msun_per_yr", self.popiii_sfr_msun_per_yr
        )
        arrays = (track_index, halo_mass, mass_weight, luminosity, muv, sfr, popiii_sfr)
        if any(array.size != mass_index.size for array in arrays):
            raise ValueError("all HaloSampleTable arrays must have the same length")
        if np.any(mass_index < 0) or np.any(track_index < 0):
            raise ValueError("sample indices must be non-negative")
        if np.any(halo_mass <= 0.0):
            raise ValueError("halo_mass_msun must be positive")
        for name, array in (
            ("mass_weight_per_mpc3", mass_weight),
            ("uv_luminosity_erg_per_s_hz", luminosity),
            ("sfr_msun_per_yr", sfr),
            ("popiii_sfr_msun_per_yr", popiii_sfr),
        ):
            if np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("redshift", redshift),
            ("imf_mode", mode),
            ("mass_index", mass_index),
            ("track_index", track_index),
            ("halo_mass_msun", halo_mass),
            ("mass_weight_per_mpc3", mass_weight),
            ("uv_luminosity_erg_per_s_hz", luminosity),
            ("muv", muv),
            ("sfr_msun_per_yr", sfr),
            ("popiii_sfr_msun_per_yr", popiii_sfr),
        ):
            object.__setattr__(self, name, value)

    @property
    def key(self) -> tuple[float, str]:
        return (self.redshift, self.imf_mode)


@dataclass(frozen=True, slots=True)
class HaloSampleDescriptor:
    redshift: float
    imf_mode: str
    sample_count: int

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        sample_count = _strict_int("sample_count", self.sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        object.__setattr__(self, "redshift", redshift)
        object.__setattr__(self, "imf_mode", mode)
        object.__setattr__(self, "sample_count", sample_count)

    @property
    def key(self) -> tuple[float, str]:
        return (self.redshift, self.imf_mode)


__all__ = ["HaloSampleDescriptor", "HaloSampleTable"]
