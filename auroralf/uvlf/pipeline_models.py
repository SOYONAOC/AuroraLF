"""Validated immutable data models passed between UVLF pipeline stages."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from auroralf._array_utils import (
    immutable_array as _immutable_array,
    validate_real_array_members as _validate_real_array_members,
)
from auroralf.mah import HaloHistoryResult
from auroralf.model_options import IMF_MODE_CANONICAL, validate_imf_mode


@dataclass(frozen=True)
class HaloUVPipelineResult:
    histories: HaloHistoryResult
    sfr_tracks: dict[str, np.ndarray]
    uv_luminosities: np.ndarray
    uv_luminosities_canonical: np.ndarray
    uv_luminosities_topheavy: np.ndarray
    uv_luminosities_popiii: np.ndarray
    redshift_grid: np.ndarray
    floor_mass: np.ndarray
    active_grid: np.ndarray
    imf_topheavy_source_grid: np.ndarray
    popiii_source_grid: np.ndarray
    metadata: dict[str, Any]
    gas_metallicity_zsun_grid: np.ndarray | None = None
    birth_metallicity_zsun_grid: np.ndarray | None = None
    metal_mass_grid: np.ndarray | None = None
    gas_mass_grid: np.ndarray | None = None


def _readonly_view(values: np.ndarray) -> np.ndarray:
    readonly = np.asarray(values).view()
    readonly.flags.writeable = False
    return readonly


def _readonly_grid(
    name: str,
    values: object,
    *,
    boolean: bool = False,
    immutable: bool = True,
) -> np.ndarray:
    if not boolean:
        _validate_real_array_members(name, values)
    array = np.asarray(values, dtype=bool if boolean else float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if boolean and np.asarray(values).dtype != np.dtype(bool):
        raise TypeError(f"{name} must have exact boolean dtype")
    return _immutable_array(array) if immutable else array


def _readonly_vector(
    name: str,
    values: object,
    *,
    boolean: bool = False,
    immutable: bool = True,
) -> np.ndarray:
    if not boolean:
        _validate_real_array_members(name, values)
    array = np.asarray(values, dtype=bool if boolean else float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if boolean and np.asarray(values).dtype != np.dtype(bool):
        raise TypeError(f"{name} must have exact boolean dtype")
    return _immutable_array(array) if immutable else array


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SharedHaloBatch:
    popiii_enabled: bool
    redshift_grid: np.ndarray
    floor_mass_msun: np.ndarray
    time_gyr_grid: np.ndarray
    redshift_history_grid: np.ndarray
    halo_mass_msun_grid: np.ndarray
    dmh_dt_sfr_msun_per_gyr_grid: np.ndarray
    sfr_msun_per_yr_grid: np.ndarray
    active_grid: np.ndarray
    starforming_grid: np.ndarray
    popiii_sfr_msun_per_yr_grid: np.ndarray
    popiii_source_grid: np.ndarray
    popiii_fstar_grid: np.ndarray
    popiii_duty_cycle_grid: np.ndarray
    popiii_lower_mass_msun_grid: np.ndarray
    popiii_upper_mass_msun_grid: np.ndarray
    burst_sfr_multiplier_grid: np.ndarray
    birth_metallicity_zsun_grid: np.ndarray | None
    gas_metallicity_zsun_grid: np.ndarray | None
    metal_mass_msun_grid: np.ndarray | None
    gas_mass_msun_grid: np.ndarray | None
    metallicity_source: str
    timing_mah_generation_seconds: float
    timing_sfr_and_chemistry_seconds: float
    _array_policy: InitVar[str] = "copy"

    def __post_init__(self, _array_policy: str) -> None:
        if type(_array_policy) is not str:
            raise TypeError("_array_policy must be exactly str")
        if _array_policy not in ("copy", "view", "mutable"):
            raise ValueError("_array_policy must be one of ('copy', 'view', 'mutable')")
        if type(self.popiii_enabled) is not bool:
            raise TypeError("popiii_enabled must be exactly bool")
        time_grid = _readonly_grid(
            "time_gyr_grid",
            self.time_gyr_grid,
            immutable=False,
        )
        shape = time_grid.shape
        float_grids: dict[str, np.ndarray] = {}
        for name in (
            "redshift_history_grid",
            "halo_mass_msun_grid",
            "dmh_dt_sfr_msun_per_gyr_grid",
            "sfr_msun_per_yr_grid",
            "popiii_sfr_msun_per_yr_grid",
            "popiii_fstar_grid",
            "popiii_duty_cycle_grid",
            "popiii_lower_mass_msun_grid",
            "popiii_upper_mass_msun_grid",
            "burst_sfr_multiplier_grid",
        ):
            grid = _readonly_grid(
                name,
                getattr(self, name),
                immutable=False,
            )
            if grid.shape != shape:
                raise ValueError(f"{name} must match time_gyr_grid shape {shape}")
            float_grids[name] = grid
        bool_grids: dict[str, np.ndarray] = {}
        for name in ("active_grid", "starforming_grid", "popiii_source_grid"):
            grid = _readonly_grid(
                name,
                getattr(self, name),
                boolean=True,
                immutable=False,
            )
            if grid.shape != shape:
                raise ValueError(f"{name} must match time_gyr_grid shape {shape}")
            bool_grids[name] = grid
        redshift_grid = _readonly_vector(
            "redshift_grid",
            self.redshift_grid,
            immutable=False,
        )
        floor_mass = _readonly_vector(
            "floor_mass_msun",
            self.floor_mass_msun,
            immutable=False,
        )
        if redshift_grid.size != shape[1] or floor_mass.size != shape[1]:
            raise ValueError("redshift_grid and floor_mass_msun must match the history time axis")
        if not np.all(np.isfinite(time_grid)) or np.any(np.diff(time_grid, axis=1) <= 0.0):
            raise ValueError("time_gyr_grid must be finite and strictly increasing per halo")
        if not np.all(np.isfinite(redshift_grid)) or np.any(redshift_grid < 0.0):
            raise ValueError("redshift_grid must be finite and non-negative")
        if not np.all(np.isfinite(floor_mass)) or np.any(floor_mass < 0.0):
            raise ValueError("floor_mass_msun must be finite and non-negative")
        for name in (
            "redshift_history_grid",
            "halo_mass_msun_grid",
            "dmh_dt_sfr_msun_per_gyr_grid",
            "sfr_msun_per_yr_grid",
            "popiii_sfr_msun_per_yr_grid",
            "popiii_fstar_grid",
            "popiii_duty_cycle_grid",
            "burst_sfr_multiplier_grid",
        ):
            grid = float_grids[name]
            if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("popiii_lower_mass_msun_grid", "popiii_upper_mass_msun_grid"):
            finite = float_grids[name][np.isfinite(float_grids[name])]
            if np.any(np.isinf(float_grids[name])) or np.any(finite < 0.0):
                raise ValueError(f"{name} may contain NaN but finite values must be non-negative")
        expected_starforming = (
            bool_grids["active_grid"]
            & np.isfinite(float_grids["sfr_msun_per_yr_grid"])
            & (float_grids["sfr_msun_per_yr_grid"] > 0.0)
        )
        if not np.array_equal(bool_grids["starforming_grid"], expected_starforming):
            raise ValueError("starforming_grid must exactly match active positive finite canonical SFR")
        popiii_sfr = float_grids["popiii_sfr_msun_per_yr_grid"]
        popiii_source = bool_grids["popiii_source_grid"]
        expected_popiii_source = bool_grids["active_grid"] & (popiii_sfr > 0.0)
        if not np.array_equal(popiii_source, expected_popiii_source):
            raise ValueError(
                "popiii_source_grid must exactly match active positive Pop III SFR"
            )
        popiii_fstar = float_grids["popiii_fstar_grid"]
        popiii_duty = float_grids["popiii_duty_cycle_grid"]
        if np.any(popiii_fstar > 1.0):
            raise ValueError("popiii_fstar_grid must lie in [0, 1]")
        if np.any(popiii_duty > 1.0):
            raise ValueError("popiii_duty_cycle_grid must lie in [0, 1]")
        if self.popiii_enabled:
            if np.any(popiii_sfr[~bool_grids["active_grid"]] != 0.0):
                raise ValueError("Pop III SFR must be zero outside active cells")
            if np.any(popiii_fstar[popiii_source] <= 0.0):
                raise ValueError("Pop III source cells require positive fstar")
            if np.any(popiii_duty[popiii_source] <= 0.0):
                raise ValueError("Pop III source cells require positive duty cycle")
            lower = float_grids["popiii_lower_mass_msun_grid"]
            upper = float_grids["popiii_upper_mass_msun_grid"]
            source_bounds_valid = (
                np.isfinite(lower[popiii_source])
                & np.isfinite(upper[popiii_source])
                & (lower[popiii_source] > 0.0)
                & (upper[popiii_source] > 0.0)
            )
            if not np.all(source_bounds_valid):
                raise ValueError(
                    "Pop III source duty-cycle mass scales must be finite and positive"
                )
        else:
            disabled_zero = (
                np.all(popiii_sfr == 0.0)
                and not np.any(popiii_source)
                and np.all(popiii_fstar == 0.0)
                and np.all(popiii_duty == 0.0)
            )
            if not disabled_zero:
                raise ValueError(
                    "disabled Pop III requires SFR, source, fstar, and duty to be zero"
                )
            if not (
                np.all(np.isnan(float_grids["popiii_lower_mass_msun_grid"]))
                and np.all(np.isnan(float_grids["popiii_upper_mass_msun_grid"]))
            ):
                raise ValueError("disabled Pop III requires all bounds to be NaN")
        optional_grids: dict[str, np.ndarray | None] = {}
        for name in (
            "birth_metallicity_zsun_grid",
            "gas_metallicity_zsun_grid",
            "metal_mass_msun_grid",
            "gas_mass_msun_grid",
        ):
            value = getattr(self, name)
            if value is None:
                optional_grids[name] = None
                continue
            grid = _readonly_grid(name, value, immutable=False)
            if grid.shape != shape or not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
                raise ValueError(f"{name} must match shape {shape} and be finite non-negative")
            optional_grids[name] = grid
        if self.metallicity_source not in ("none", "mzr", "regulator"):
            raise ValueError("metallicity_source must be one of ('none', 'mzr', 'regulator')")
        present_optional = {
            name for name, value in optional_grids.items() if value is not None
        }
        expected_optional = {
            "none": set(),
            "mzr": {"birth_metallicity_zsun_grid"},
            "regulator": set(optional_grids),
        }[self.metallicity_source]
        if present_optional != expected_optional:
            raise ValueError(
                "metallicity_source must agree with the supplied metallicity grids"
            )
        for name in ("timing_mah_generation_seconds", "timing_sfr_and_chemistry_seconds"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Real):
                raise TypeError(f"{name} must be a real non-boolean value")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if _array_policy == "copy":
            time_grid = _immutable_array(time_grid)
            redshift_grid = _immutable_array(redshift_grid)
            floor_mass = _immutable_array(floor_mass)
            float_grids = {
                name: _immutable_array(value) for name, value in float_grids.items()
            }
            bool_grids = {
                name: _immutable_array(value) for name, value in bool_grids.items()
            }
            optional_grids = {
                name: None if value is None else _immutable_array(value)
                for name, value in optional_grids.items()
            }
        elif _array_policy == "view":
            time_grid = _readonly_view(time_grid)
            redshift_grid = _readonly_view(redshift_grid)
            floor_mass = _readonly_view(floor_mass)
            float_grids = {
                name: _readonly_view(value) for name, value in float_grids.items()
            }
            bool_grids = {
                name: _readonly_view(value) for name, value in bool_grids.items()
            }
            optional_grids = {
                name: None if value is None else _readonly_view(value)
                for name, value in optional_grids.items()
            }
        object.__setattr__(self, "time_gyr_grid", time_grid)
        object.__setattr__(self, "redshift_grid", redshift_grid)
        object.__setattr__(self, "floor_mass_msun", floor_mass)
        for name, value in float_grids.items():
            object.__setattr__(self, name, value)
        for name, value in bool_grids.items():
            object.__setattr__(self, name, value)
        for name, value in optional_grids.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class LoadedSSPKernels:
    canonical_age_myr: np.ndarray
    canonical_luminosity_per_msun: np.ndarray
    topheavy_age_myr: np.ndarray | None
    topheavy_luminosity_per_msun: np.ndarray | None
    popiii_age_myr: np.ndarray | None
    popiii_luminosity_per_msun: np.ndarray | None
    canonical_ssp_path: Path
    topheavy_ssp_path: Path
    popiii_ssp_path: Path
    topheavy_ssp_template_metallicity_zsun: float | None

    def __post_init__(self) -> None:
        pairs = (
            ("canonical", self.canonical_age_myr, self.canonical_luminosity_per_msun),
            ("topheavy", self.topheavy_age_myr, self.topheavy_luminosity_per_msun),
            ("popiii", self.popiii_age_myr, self.popiii_luminosity_per_msun),
        )
        for prefix, age_values, luminosity_values in pairs:
            if (age_values is None) != (luminosity_values is None):
                raise ValueError(f"{prefix} SSP age and luminosity must be provided together")
            if age_values is None or luminosity_values is None:
                continue
            ages = _readonly_vector(f"{prefix}_age_myr", age_values)
            luminosity = _readonly_vector(
                f"{prefix}_luminosity_per_msun",
                luminosity_values,
            )
            if ages.size != luminosity.size:
                raise ValueError(f"{prefix} SSP age and luminosity arrays must have equal length")
            if not np.all(np.isfinite(ages)) or np.any(ages <= 0.0) or np.any(np.diff(ages) <= 0.0):
                raise ValueError(f"{prefix} SSP ages must be finite positive and strictly increasing")
            if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0.0):
                raise ValueError(f"{prefix} SSP luminosity must be finite and non-negative")
            object.__setattr__(self, f"{prefix}_age_myr", ages)
            object.__setattr__(self, f"{prefix}_luminosity_per_msun", luminosity)
        if self.canonical_age_myr is None or self.canonical_luminosity_per_msun is None:
            raise ValueError("canonical SSP kernels are required")
        for name in ("canonical_ssp_path", "topheavy_ssp_path", "popiii_ssp_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
        metallicity = self.topheavy_ssp_template_metallicity_zsun
        if metallicity is not None:
            if isinstance(metallicity, (bool, np.bool_)) or not isinstance(
                metallicity,
                Real,
            ):
                raise TypeError(
                    "topheavy_ssp_template_metallicity_zsun must be a real non-boolean value"
                )
            metallicity = float(metallicity)
            if not np.isfinite(metallicity) or metallicity <= 0.0:
                raise ValueError(
                    "topheavy_ssp_template_metallicity_zsun must be finite and positive"
                )
            object.__setattr__(
                self,
                "topheavy_ssp_template_metallicity_zsun",
                metallicity,
            )

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        return (
            _rebuild_loaded_ssp_kernels,
            (
                self.canonical_age_myr,
                self.canonical_luminosity_per_msun,
                self.topheavy_age_myr,
                self.topheavy_luminosity_per_msun,
                self.popiii_age_myr,
                self.popiii_luminosity_per_msun,
                self.canonical_ssp_path,
                self.topheavy_ssp_path,
                self.popiii_ssp_path,
                self.topheavy_ssp_template_metallicity_zsun,
            ),
        )


def _rebuild_loaded_ssp_kernels(
    canonical_age_myr: object,
    canonical_luminosity_per_msun: object,
    topheavy_age_myr: object,
    topheavy_luminosity_per_msun: object,
    popiii_age_myr: object,
    popiii_luminosity_per_msun: object,
    canonical_ssp_path: object,
    topheavy_ssp_path: object,
    popiii_ssp_path: object,
    topheavy_ssp_template_metallicity_zsun: object,
) -> LoadedSSPKernels:
    return LoadedSSPKernels(
        canonical_age_myr=canonical_age_myr,
        canonical_luminosity_per_msun=canonical_luminosity_per_msun,
        topheavy_age_myr=topheavy_age_myr,
        topheavy_luminosity_per_msun=topheavy_luminosity_per_msun,
        popiii_age_myr=popiii_age_myr,
        popiii_luminosity_per_msun=popiii_luminosity_per_msun,
        canonical_ssp_path=canonical_ssp_path,
        topheavy_ssp_path=topheavy_ssp_path,
        popiii_ssp_path=popiii_ssp_path,
        topheavy_ssp_template_metallicity_zsun=(
            topheavy_ssp_template_metallicity_zsun
        ),
    )


@dataclass(frozen=True, slots=True)
class HaloModeEvaluation:
    imf_mode: str
    uv_luminosity_erg_per_s_hz: np.ndarray
    canonical_uv_luminosity_erg_per_s_hz: np.ndarray
    topheavy_uv_luminosity_erg_per_s_hz: np.ndarray
    popiii_uv_luminosity_erg_per_s_hz: np.ndarray
    topheavy_source_grid: np.ndarray
    candidate_topheavy_source_grid: np.ndarray
    topheavy_light_fraction: np.ndarray
    popiii_light_fraction: np.ndarray
    topheavy_source_count: int
    starforming_source_count: int
    popiii_source_count: int
    active_source_count: int
    uv_convolution_seconds: float

    def __post_init__(self) -> None:
        mode = validate_imf_mode(self.imf_mode)
        vectors: dict[str, np.ndarray] = {}
        for name in (
            "uv_luminosity_erg_per_s_hz",
            "canonical_uv_luminosity_erg_per_s_hz",
            "topheavy_uv_luminosity_erg_per_s_hz",
            "popiii_uv_luminosity_erg_per_s_hz",
            "topheavy_light_fraction",
            "popiii_light_fraction",
        ):
            vector = _readonly_vector(name, getattr(self, name))
            if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
            vectors[name] = vector
        if len({vector.size for vector in vectors.values()}) != 1:
            raise ValueError("all HaloModeEvaluation luminosity/fraction vectors must match")
        total = vectors["uv_luminosity_erg_per_s_hz"]
        canonical = vectors["canonical_uv_luminosity_erg_per_s_hz"]
        topheavy = vectors["topheavy_uv_luminosity_erg_per_s_hz"]
        popiii = vectors["popiii_uv_luminosity_erg_per_s_hz"]
        if not np.array_equal(total, canonical + topheavy + popiii):
            raise ValueError(
                "total UV luminosity must exactly equal the sum of canonical, "
                "top-heavy, and Pop III components"
            )
        positive = total > 0.0
        expected_topheavy_fraction = np.zeros_like(total)
        expected_popiii_fraction = np.zeros_like(total)
        expected_topheavy_fraction[positive] = topheavy[positive] / total[positive]
        expected_popiii_fraction[positive] = popiii[positive] / total[positive]
        if not np.array_equal(
            vectors["topheavy_light_fraction"],
            expected_topheavy_fraction,
        ):
            raise ValueError(
                "topheavy_light_fraction must exactly match top-heavy / total light"
            )
        if not np.array_equal(
            vectors["popiii_light_fraction"],
            expected_popiii_fraction,
        ):
            raise ValueError(
                "popiii_light_fraction must exactly match Pop III / total light"
            )
        for name in ("topheavy_light_fraction", "popiii_light_fraction"):
            if np.any(vectors[name] > 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        grids: dict[str, np.ndarray] = {}
        for name in ("topheavy_source_grid", "candidate_topheavy_source_grid"):
            grid = _readonly_grid(name, getattr(self, name), boolean=True)
            if grid.shape[0] != vectors["uv_luminosity_erg_per_s_hz"].size:
                raise ValueError(f"{name} halo axis must match luminosity vectors")
            grids[name] = grid
        if grids["topheavy_source_grid"].shape != grids["candidate_topheavy_source_grid"].shape:
            raise ValueError("topheavy source grids must have identical shapes")
        if np.any(
            grids["topheavy_source_grid"]
            & ~grids["candidate_topheavy_source_grid"]
        ):
            raise ValueError("topheavy_source_grid must be a subset of candidate mask")
        for name in (
            "topheavy_source_count",
            "starforming_source_count",
            "popiii_source_count",
            "active_source_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer non-boolean count")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        if self.topheavy_source_count != int(
            np.count_nonzero(grids["topheavy_source_grid"])
        ):
            raise ValueError("topheavy_source_count must equal the source mask count")
        cell_count = int(grids["topheavy_source_grid"].size)
        if self.active_source_count > cell_count:
            raise ValueError(
                f"active_source_count ({self.active_source_count}) must not exceed "
                f"the top-heavy grid cell count ({cell_count})"
            )
        if self.starforming_source_count > self.active_source_count:
            raise ValueError(
                f"starforming_source_count ({self.starforming_source_count}) must not "
                f"exceed active_source_count ({self.active_source_count})"
            )
        candidate_count = int(
            np.count_nonzero(grids["candidate_topheavy_source_grid"])
        )
        if candidate_count > self.starforming_source_count:
            raise ValueError(
                f"candidate source count ({candidate_count}) must not exceed "
                f"starforming_source_count ({self.starforming_source_count})"
            )
        if self.topheavy_source_count > candidate_count:
            raise ValueError(
                f"topheavy_source_count ({self.topheavy_source_count}) must not exceed "
                f"candidate source count ({candidate_count})"
            )
        if self.popiii_source_count > self.active_source_count:
            raise ValueError("popiii_source_count must not exceed active_source_count")
        if mode == IMF_MODE_CANONICAL and (
            np.any(topheavy != 0.0)
            or np.any(vectors["topheavy_light_fraction"] != 0.0)
            or np.any(grids["topheavy_source_grid"])
            or np.any(grids["candidate_topheavy_source_grid"])
            or self.topheavy_source_count != 0
        ):
            raise ValueError(
                "canonical mode requires all top-heavy components, fractions, masks, "
                "candidate masks, and counts to be zero"
            )
        if isinstance(self.uv_convolution_seconds, (bool, np.bool_)) or not isinstance(
            self.uv_convolution_seconds,
            Real,
        ):
            raise TypeError("uv_convolution_seconds must be a real non-boolean value")
        seconds = float(self.uv_convolution_seconds)
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("uv_convolution_seconds must be finite and non-negative")
        object.__setattr__(self, "imf_mode", mode)
        object.__setattr__(self, "uv_convolution_seconds", seconds)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        for name, value in grids.items():
            object.__setattr__(self, name, value)
