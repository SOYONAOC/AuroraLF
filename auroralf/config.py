from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

import astropy.units as u
import numpy as np

from auroralf.constants import (
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_B,
    PLANCK18_OMEGA_M,
)
from auroralf.model_options import (
    IMF_MODE_CANONICAL,
    IMFTransitionParameters,
    validate_imf_mode,
    validate_mass_function_model,
)
from auroralf.chemistry import MZRBirthMetallicityParameters, RegulatorMetallicityParameters
from auroralf.chemistry.mzr import MZR_RELATIONS
from auroralf.mah import (
    Cosmology,
    validate_mah_backend,
    validate_thesan_time_grid_mode,
    validate_tng_time_grid_mode,
)
from auroralf.mah.sampling import validate_parameter_sampler
from auroralf.sfr import PopIIISFRParameters, SFRModelParameters
from auroralf.sfr.popiii import POPIII_UPPER_MASS_MODES


CONFIG_SCHEMA_VERSION = "2.2.0"
_UINT64_MAX = 2**64 - 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and boolean/string values are not allowed")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and boolean/float/string values are not allowed")
    return int(value)


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _strict_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _strict_optional_float(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _strict_float(name, value)


def _strict_optional_path(name: str, value: object) -> Path | None:
    if value is None:
        return None
    return _strict_absolute_path(name, value)


def _strict_absolute_path(name: str, value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    if not value.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return value


def _normalize_float_tuple(name: str, values: object, *, nonnegative: bool = False) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    normalized = tuple(_strict_float(f"{name}[{index}]", value) for index, value in enumerate(values))
    if nonnegative and any(value < 0.0 for value in normalized):
        raise ValueError(f"{name} must contain only non-negative values")
    return normalized


@dataclass(frozen=True, slots=True)
class CosmologyConfig:
    h0_km_s_mpc: float = PLANCK18_H0_KM_S_MPC
    omega_m: float = PLANCK18_OMEGA_M
    omega_b: float = PLANCK18_OMEGA_B

    def __post_init__(self) -> None:
        h0 = _strict_float("h0_km_s_mpc", self.h0_km_s_mpc)
        omega_m = _strict_float("omega_m", self.omega_m)
        omega_b = _strict_float("omega_b", self.omega_b)
        if h0 <= 0.0:
            raise ValueError("h0_km_s_mpc must be positive")
        if not 0.0 < omega_m <= 1.0:
            raise ValueError("omega_m must lie in (0, 1]")
        if omega_b <= 0.0 or omega_b > omega_m:
            raise ValueError("omega_b must be positive and not exceed omega_m")
        object.__setattr__(self, "h0_km_s_mpc", h0)
        object.__setattr__(self, "omega_m", omega_m)
        object.__setattr__(self, "omega_b", omega_b)

    def to_model(self) -> Cosmology:
        return Cosmology(
            h0=(self.h0_km_s_mpc * u.km / (u.s * u.Mpc)).to_value(u.Gyr**-1),
            omega_m=self.omega_m,
            omega_b=self.omega_b,
            omega_lambda=1.0 - self.omega_m,
        )


@dataclass(frozen=True, slots=True)
class MAHConfig:
    backend: str = "mcbride"
    sampler: str = "mcbride"
    z_start_max: float = 50.0
    n_time_steps: int = 240
    tng_cache_path: Path | None = None
    tng_mass_bin_width_dex: float = 0.15
    tng_min_candidates: int = 5
    tng_smoothing_myr: float = 0.0
    tng_time_grid_mode: str = "snapshot"
    thesan_cache_path: Path | None = None
    thesan_mass_bin_width_dex: float = 0.15
    thesan_min_candidates: int = 5
    thesan_smoothing_myr: float = 0.0
    thesan_time_grid_mode: str = "snapshot"

    def __post_init__(self) -> None:
        backend = validate_mah_backend(_strict_string("backend", self.backend))
        sampler = validate_parameter_sampler(_strict_string("sampler", self.sampler))
        z_start_max = _strict_float("z_start_max", self.z_start_max)
        n_time_steps = _strict_int("n_time_steps", self.n_time_steps)
        tng_cache = _strict_optional_path("tng_cache_path", self.tng_cache_path)
        thesan_cache = _strict_optional_path("thesan_cache_path", self.thesan_cache_path)
        tng_width = _strict_float("tng_mass_bin_width_dex", self.tng_mass_bin_width_dex)
        tng_min = _strict_int("tng_min_candidates", self.tng_min_candidates)
        tng_smoothing = _strict_float("tng_smoothing_myr", self.tng_smoothing_myr)
        thesan_width = _strict_float("thesan_mass_bin_width_dex", self.thesan_mass_bin_width_dex)
        thesan_min = _strict_int("thesan_min_candidates", self.thesan_min_candidates)
        thesan_smoothing = _strict_float("thesan_smoothing_myr", self.thesan_smoothing_myr)
        tng_mode = validate_tng_time_grid_mode(
            _strict_string("tng_time_grid_mode", self.tng_time_grid_mode)
        )
        thesan_mode = validate_thesan_time_grid_mode(
            _strict_string("thesan_time_grid_mode", self.thesan_time_grid_mode)
        )
        if z_start_max <= 0.0:
            raise ValueError("z_start_max must be positive")
        if n_time_steps < 2:
            raise ValueError("n_time_steps must be at least 2")
        if tng_width <= 0.0:
            raise ValueError("tng_mass_bin_width_dex must be positive")
        if tng_min <= 0:
            raise ValueError("tng_min_candidates must be positive")
        if tng_smoothing < 0.0:
            raise ValueError("tng_smoothing_myr must be non-negative")
        if thesan_width <= 0.0:
            raise ValueError("thesan_mass_bin_width_dex must be positive")
        if thesan_min <= 0:
            raise ValueError("thesan_min_candidates must be positive")
        if thesan_smoothing < 0.0:
            raise ValueError("thesan_smoothing_myr must be non-negative")
        if backend == "tng" and tng_cache is None:
            raise ValueError("tng_cache_path is required when backend='tng'")
        if backend == "thesan" and thesan_cache is None:
            raise ValueError("thesan_cache_path is required when backend='thesan'")
        for name, value in (
            ("backend", backend),
            ("sampler", sampler),
            ("z_start_max", z_start_max),
            ("n_time_steps", n_time_steps),
            ("tng_cache_path", tng_cache),
            ("tng_mass_bin_width_dex", tng_width),
            ("tng_min_candidates", tng_min),
            ("tng_smoothing_myr", tng_smoothing),
            ("tng_time_grid_mode", tng_mode),
            ("thesan_cache_path", thesan_cache),
            ("thesan_mass_bin_width_dex", thesan_width),
            ("thesan_min_candidates", thesan_min),
            ("thesan_smoothing_myr", thesan_smoothing),
            ("thesan_time_grid_mode", thesan_mode),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MZRConfig:
    relation: str = "fire2_highz"
    returned_fraction: float = 0.4
    scatter_dex: float = 0.0
    stellar_mass_floor_msun: float = 1.0e6

    def __post_init__(self) -> None:
        relation = _strict_string("relation", self.relation)
        returned = _strict_float("returned_fraction", self.returned_fraction)
        scatter = _strict_float("scatter_dex", self.scatter_dex)
        floor = _strict_float("stellar_mass_floor_msun", self.stellar_mass_floor_msun)
        if relation not in MZR_RELATIONS:
            raise ValueError(f"relation must be one of {MZR_RELATIONS}")
        if not 0.0 <= returned < 1.0:
            raise ValueError("returned_fraction must lie in [0, 1)")
        if scatter < 0.0:
            raise ValueError("scatter_dex must be non-negative")
        if floor <= 0.0:
            raise ValueError("stellar_mass_floor_msun must be positive")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "returned_fraction", returned)
        object.__setattr__(self, "scatter_dex", scatter)
        object.__setattr__(self, "stellar_mass_floor_msun", floor)

    def to_model(self) -> MZRBirthMetallicityParameters:
        return MZRBirthMetallicityParameters(
            relation=self.relation,
            returned_fraction=self.returned_fraction,
            scatter_dex=self.scatter_dex,
            stellar_mass_floor_msun=self.stellar_mass_floor_msun,
        )


@dataclass(frozen=True, slots=True)
class RegulatorConfig:
    solar_metallicity_mass_fraction: float = 0.0142
    gas_fraction_norm: float = 0.02
    gas_fraction_mass_scale_msun: float = 1.0e10
    gas_fraction_mass_slope: float = 0.0
    gas_fraction_redshift_scale: float = 10.0
    gas_fraction_redshift_slope: float = 0.0
    returned_fraction: float = 0.4
    metal_yield: float = 0.01
    inflow_metallicity_zsun: float = 0.0
    metal_loading_norm: float = 20.0
    metal_loading_mass_scale_msun: float = 1.0e10
    metal_loading_mass_slope: float = -0.5
    metal_loading_redshift_scale: float = 10.0
    metal_loading_redshift_slope: float = 0.0
    stellar_mass_floor_msun: float = 0.0
    metallicity_scatter_dex: float = 0.0

    def __post_init__(self) -> None:
        normalized = {
            field.name: _strict_float(field.name, getattr(self, field.name))
            for field in fields(self)
        }
        positive = (
            "solar_metallicity_mass_fraction",
            "gas_fraction_norm",
            "gas_fraction_mass_scale_msun",
            "gas_fraction_redshift_scale",
            "metal_loading_mass_scale_msun",
            "metal_loading_redshift_scale",
        )
        for name in positive:
            if normalized[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= normalized["returned_fraction"] < 1.0:
            raise ValueError("returned_fraction must lie in [0, 1)")
        for name in (
            "metal_yield",
            "inflow_metallicity_zsun",
            "metal_loading_norm",
            "stellar_mass_floor_msun",
            "metallicity_scatter_dex",
        ):
            if normalized[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def to_model(self) -> RegulatorMetallicityParameters:
        return RegulatorMetallicityParameters(
            **{field.name: getattr(self, field.name) for field in fields(self)}
        )


@dataclass(frozen=True, slots=True)
class StarFormationConfig:
    enable_time_delay: bool = True
    efficiency_normalization: float = 0.12
    characteristic_halo_mass_msun: float = 10.0**11.7
    low_mass_slope: float = 0.66
    high_mass_slope: float = 0.65
    enable_archived_burst_scatter: bool = False
    burst_scatter_dex: float = 0.0
    burst_scatter_correlation_timescale_myr: float = 20.0
    burst_scatter_mass_conserving: bool = True
    enable_archived_metallicity: bool = False
    metallicity_source: str = "none"
    mzr: MZRConfig | None = None
    regulator: RegulatorConfig | None = None

    def __post_init__(self) -> None:
        enable_delay = _strict_bool("enable_time_delay", self.enable_time_delay)
        archive_burst = _strict_bool(
            "enable_archived_burst_scatter", self.enable_archived_burst_scatter
        )
        preserve_mass = _strict_bool(
            "burst_scatter_mass_conserving", self.burst_scatter_mass_conserving
        )
        archive_metallicity = _strict_bool(
            "enable_archived_metallicity", self.enable_archived_metallicity
        )
        efficiency = _strict_float("efficiency_normalization", self.efficiency_normalization)
        mass = _strict_float(
            "characteristic_halo_mass_msun", self.characteristic_halo_mass_msun
        )
        low_slope = _strict_float("low_mass_slope", self.low_mass_slope)
        high_slope = _strict_float("high_mass_slope", self.high_mass_slope)
        scatter = _strict_float("burst_scatter_dex", self.burst_scatter_dex)
        timescale = _strict_float(
            "burst_scatter_correlation_timescale_myr",
            self.burst_scatter_correlation_timescale_myr,
        )
        source = _strict_string("metallicity_source", self.metallicity_source)
        if source not in ("none", "mzr", "regulator"):
            raise ValueError("metallicity_source must be one of ('none', 'mzr', 'regulator')")
        if self.mzr is not None and type(self.mzr) is not MZRConfig:
            raise TypeError("mzr must be exactly MZRConfig or None")
        if self.regulator is not None and type(self.regulator) is not RegulatorConfig:
            raise TypeError("regulator must be exactly RegulatorConfig or None")
        expected = (
            (source == "none" and self.mzr is None and self.regulator is None)
            or (source == "mzr" and self.mzr is not None and self.regulator is None)
            or (source == "regulator" and self.mzr is None and self.regulator is not None)
        )
        if not expected:
            raise ValueError("metallicity_source must match exactly one nested metallicity config")
        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("efficiency_normalization must lie in [0, 1]")
        if mass <= 0.0:
            raise ValueError("characteristic_halo_mass_msun must be positive")
        if low_slope < 0.0:
            raise ValueError("low_mass_slope must be non-negative")
        if high_slope < 0.0:
            raise ValueError("high_mass_slope must be non-negative")
        if scatter < 0.0:
            raise ValueError("burst_scatter_dex must be non-negative")
        if timescale <= 0.0:
            raise ValueError("burst_scatter_correlation_timescale_myr must be positive")
        if scatter > 0.0 and not archive_burst:
            raise ValueError(
                "burst scatter is archived; set "
                "star_formation.enable_archived_burst_scatter=true only for "
                "explicit historical reproduction"
            )
        if source != "none" and not archive_metallicity:
            raise ValueError(
                "metallicity models are archived; set "
                "star_formation.enable_archived_metallicity=true only for "
                "explicit historical reproduction"
            )
        for name, value in (
            ("enable_time_delay", enable_delay),
            ("efficiency_normalization", efficiency),
            ("characteristic_halo_mass_msun", mass),
            ("low_mass_slope", low_slope),
            ("high_mass_slope", high_slope),
            ("enable_archived_burst_scatter", archive_burst),
            ("burst_scatter_dex", scatter),
            ("burst_scatter_correlation_timescale_myr", timescale),
            ("burst_scatter_mass_conserving", preserve_mass),
            ("enable_archived_metallicity", archive_metallicity),
            ("metallicity_source", source),
        ):
            object.__setattr__(self, name, value)

    def to_model(self) -> SFRModelParameters:
        return SFRModelParameters(
            epsilon_0=self.efficiency_normalization,
            characteristic_mass=self.characteristic_halo_mass_msun,
            beta_star=self.low_mass_slope,
            gamma_star=self.high_mass_slope,
        )


@dataclass(frozen=True, slots=True)
class StellarPopulationConfig:
    imf_modes: tuple[str, ...] = (IMF_MODE_CANONICAL,)
    enable_archived_imf_gate: bool = False
    canonical_ssp_path: Path = _PROJECT_ROOT / "external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
    topheavy_ssp_path: Path = _PROJECT_ROOT / "external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5"
    topheavy_ssp_template_metallicity_zsun: float | None = 0.05
    historical_topheavy_redshift_min: float = 10.0
    source_redshift_gate_enabled: bool = False
    growth_time_threshold_myr: float = 50.0
    birth_metallicity_topheavy_max_zsun: float | None = 0.05
    enable_popiii: bool = False
    popiii_ssp_path: Path = _PROJECT_ROOT / "external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_sal_500_001_is5.25"
    popiii_efficiency: float = 1.0e-3
    popiii_pivot_halo_mass_msun: float = 1.0e7
    popiii_low_mass_slope: float = 0.0
    popiii_high_mass_slope: float = 0.0
    lw_background_j21: float = 0.0
    popiii_upper_mass_mode: str = "atomic"
    popiii_upper_mass_msun: float | None = None

    def __post_init__(self) -> None:
        if type(self.imf_modes) is not tuple:
            raise TypeError("imf_modes must be a tuple")
        if not self.imf_modes:
            raise ValueError("imf_modes must be non-empty")
        modes: list[str] = []
        for index, mode_value in enumerate(self.imf_modes):
            mode = _strict_string(f"imf_modes[{index}]", mode_value)
            modes.append(validate_imf_mode(mode))
        normalized_modes = tuple(modes)
        if normalized_modes[0] != IMF_MODE_CANONICAL:
            raise ValueError("canonical IMF mode must be first")
        if len(set(normalized_modes)) != len(normalized_modes):
            raise ValueError("imf_modes must be unique")
        archive_gate = _strict_bool(
            "enable_archived_imf_gate", self.enable_archived_imf_gate
        )
        canonical_path = _strict_absolute_path("canonical_ssp_path", self.canonical_ssp_path)
        topheavy_path = _strict_absolute_path("topheavy_ssp_path", self.topheavy_ssp_path)
        popiii_path = _strict_absolute_path("popiii_ssp_path", self.popiii_ssp_path)
        template_z = _strict_optional_float(
            "topheavy_ssp_template_metallicity_zsun",
            self.topheavy_ssp_template_metallicity_zsun,
        )
        historical_z = _strict_float(
            "historical_topheavy_redshift_min", self.historical_topheavy_redshift_min
        )
        source_gate = _strict_bool(
            "source_redshift_gate_enabled", self.source_redshift_gate_enabled
        )
        growth_time = _strict_float("growth_time_threshold_myr", self.growth_time_threshold_myr)
        birth_gate = _strict_optional_float(
            "birth_metallicity_topheavy_max_zsun",
            self.birth_metallicity_topheavy_max_zsun,
        )
        enable_popiii = _strict_bool("enable_popiii", self.enable_popiii)
        popiii_efficiency = _strict_float("popiii_efficiency", self.popiii_efficiency)
        pivot_mass = _strict_float(
            "popiii_pivot_halo_mass_msun", self.popiii_pivot_halo_mass_msun
        )
        low_slope = _strict_float("popiii_low_mass_slope", self.popiii_low_mass_slope)
        high_slope = _strict_float("popiii_high_mass_slope", self.popiii_high_mass_slope)
        lw = _strict_float("lw_background_j21", self.lw_background_j21)
        upper_mode = _strict_string("popiii_upper_mass_mode", self.popiii_upper_mass_mode)
        upper_mass = _strict_optional_float("popiii_upper_mass_msun", self.popiii_upper_mass_msun)
        if historical_z < 0.0:
            raise ValueError("historical_topheavy_redshift_min must be non-negative")
        if growth_time <= 0.0:
            raise ValueError("growth_time_threshold_myr must be positive")
        if birth_gate is not None and birth_gate <= 0.0:
            raise ValueError("birth_metallicity_topheavy_max_zsun must be positive when provided")
        has_variant = len(normalized_modes) > 1
        if has_variant and (template_z is None or template_z <= 0.0):
            raise ValueError(
                "topheavy_ssp_template_metallicity_zsun must be positive for variant IMF modes"
            )
        if template_z is not None and template_z <= 0.0:
            raise ValueError("topheavy_ssp_template_metallicity_zsun must be positive when provided")
        if not 0.0 <= popiii_efficiency <= 1.0:
            raise ValueError("popiii_efficiency must lie in [0, 1]")
        if pivot_mass <= 0.0:
            raise ValueError("popiii_pivot_halo_mass_msun must be positive")
        if lw < 0.0:
            raise ValueError("lw_background_j21 must be non-negative")
        if upper_mode not in POPIII_UPPER_MASS_MODES:
            raise ValueError(f"popiii_upper_mass_mode must be one of {POPIII_UPPER_MASS_MODES}")
        if upper_mode == "fixed" and (upper_mass is None or upper_mass <= 0.0):
            raise ValueError("popiii_upper_mass_msun must be positive when mode is fixed")
        if upper_mode == "atomic" and upper_mass is not None:
            raise ValueError("popiii_upper_mass_msun must be None when mode is atomic")
        if upper_mass is not None and upper_mass <= 0.0:
            raise ValueError("popiii_upper_mass_msun must be positive when provided")
        for name, value in (
            ("imf_modes", normalized_modes),
            ("enable_archived_imf_gate", archive_gate),
            ("canonical_ssp_path", canonical_path),
            ("topheavy_ssp_path", topheavy_path),
            ("topheavy_ssp_template_metallicity_zsun", template_z),
            ("historical_topheavy_redshift_min", historical_z),
            ("source_redshift_gate_enabled", source_gate),
            ("growth_time_threshold_myr", growth_time),
            ("birth_metallicity_topheavy_max_zsun", birth_gate),
            ("enable_popiii", enable_popiii),
            ("popiii_ssp_path", popiii_path),
            ("popiii_efficiency", popiii_efficiency),
            ("popiii_pivot_halo_mass_msun", pivot_mass),
            ("popiii_low_mass_slope", low_slope),
            ("popiii_high_mass_slope", high_slope),
            ("lw_background_j21", lw),
            ("popiii_upper_mass_mode", upper_mode),
            ("popiii_upper_mass_msun", upper_mass),
        ):
            object.__setattr__(self, name, value)

    def to_imf_transition_model(self) -> IMFTransitionParameters:
        return IMFTransitionParameters(
            z_topheavy_min=self.historical_topheavy_redshift_min,
            source_redshift_gate_enabled=self.source_redshift_gate_enabled,
            growth_time_threshold_myr=self.growth_time_threshold_myr,
            metallicity_topheavy_max_zsun=self.birth_metallicity_topheavy_max_zsun,
        )

    def to_popiii_model(self) -> PopIIISFRParameters:
        return PopIIISFRParameters(
            epsilon_star=self.popiii_efficiency,
            pivot_mass_msun=self.popiii_pivot_halo_mass_msun,
            alpha_star=self.popiii_low_mass_slope,
            beta_star=self.popiii_high_mass_slope,
            lw_background_j21=self.lw_background_j21,
            upper_mass_mode=self.popiii_upper_mass_mode,
            upper_mass_msun=self.popiii_upper_mass_msun,
        )


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    mass_batch_size: int
    n_halo_mass_samples: int = 3000
    n_tracks_per_halo_mass: int = 1000
    log10_halo_mass_min_msun: float = 9.0
    log10_halo_mass_max_msun: float = 13.0
    muv_bin_edges: tuple[float, ...] = tuple(np.linspace(-24.0, -12.0, 41))
    workers: int = 1
    mass_function_model: str = "hmf_reed07"
    hmf_dlog10m: float = 0.005
    apply_dust: bool = True

    def __post_init__(self) -> None:
        mass_batch_size = _strict_int("mass_batch_size", self.mass_batch_size)
        n_mass = _strict_int("n_halo_mass_samples", self.n_halo_mass_samples)
        n_tracks = _strict_int("n_tracks_per_halo_mass", self.n_tracks_per_halo_mass)
        log_min = _strict_float("log10_halo_mass_min_msun", self.log10_halo_mass_min_msun)
        log_max = _strict_float("log10_halo_mass_max_msun", self.log10_halo_mass_max_msun)
        edges = _normalize_float_tuple("muv_bin_edges", self.muv_bin_edges)
        workers = _strict_int("workers", self.workers)
        model_name = _strict_string("mass_function_model", self.mass_function_model)
        try:
            model = validate_mass_function_model(model_name)
        except ValueError as error:
            raise ValueError(f"mass_function_model: {error}") from error
        hmf_step = _strict_float("hmf_dlog10m", self.hmf_dlog10m)
        apply_dust = _strict_bool("apply_dust", self.apply_dust)
        if mass_batch_size <= 0:
            raise ValueError("mass_batch_size must be positive")
        if n_mass <= 0:
            raise ValueError("n_halo_mass_samples must be positive")
        if n_tracks <= 0:
            raise ValueError("n_tracks_per_halo_mass must be positive")
        if log_max <= log_min:
            raise ValueError(
                "log10_halo_mass_max_msun must exceed log10_halo_mass_min_msun"
            )
        if len(edges) < 2:
            raise ValueError("muv_bin_edges must contain at least two values")
        if any(right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)):
            raise ValueError("muv_bin_edges must be strictly increasing")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if hmf_step <= 0.0:
            raise ValueError("hmf_dlog10m must be positive")
        for name, value in (
            ("mass_batch_size", mass_batch_size),
            ("n_halo_mass_samples", n_mass),
            ("n_tracks_per_halo_mass", n_tracks),
            ("log10_halo_mass_min_msun", log_min),
            ("log10_halo_mass_max_msun", log_max),
            ("muv_bin_edges", edges),
            ("workers", workers),
            ("mass_function_model", model),
            ("hmf_dlog10m", hmf_step),
            ("apply_dust", apply_dust),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    artifact_path: Path = _PROJECT_ROOT / "outputs/auroralf_v2.h5"

    def __post_init__(self) -> None:
        path = _strict_absolute_path("artifact_path", self.artifact_path)
        if path.suffix != ".h5":
            raise ValueError("artifact_path must have suffix .h5")
        object.__setattr__(self, "artifact_path", path)


@dataclass(frozen=True, slots=True)
class UVLFRunConfig:
    schema_version: str
    run_id: str
    redshifts: tuple[float, ...]
    base_seed: int
    cosmology: CosmologyConfig
    mah: MAHConfig
    star_formation: StarFormationConfig
    stellar_population: StellarPopulationConfig
    sampling: SamplingConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        schema = _strict_string("schema_version", self.schema_version)
        run_id = _strict_string("run_id", self.run_id)
        redshifts = _normalize_float_tuple("redshifts", self.redshifts, nonnegative=True)
        base_seed = _strict_int("base_seed", self.base_seed)
        if schema != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {CONFIG_SCHEMA_VERSION!r}")
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        if not redshifts:
            raise ValueError("redshifts must be non-empty")
        if any(right <= left for left, right in zip(redshifts[:-1], redshifts[1:], strict=True)):
            raise ValueError("redshifts must be strictly increasing")
        if not 0 <= base_seed <= _UINT64_MAX:
            raise ValueError("base_seed must lie in the uint64 range [0, 2**64 - 1]")
        for name, value, expected in (
            ("cosmology", self.cosmology, CosmologyConfig),
            ("mah", self.mah, MAHConfig),
            ("star_formation", self.star_formation, StarFormationConfig),
            ("stellar_population", self.stellar_population, StellarPopulationConfig),
            ("sampling", self.sampling, SamplingConfig),
            ("output", self.output, OutputConfig),
        ):
            if type(value) is not expected:
                raise TypeError(f"{name} must be exactly {expected.__name__}")
        has_variant = len(self.stellar_population.imf_modes) > 1
        if has_variant and not self.stellar_population.enable_archived_imf_gate:
            raise ValueError(
                "variant IMF gate modes are archived; set "
                "stellar_population.enable_archived_imf_gate=true only for "
                "explicit historical reproduction"
            )
        has_birth_gate = self.stellar_population.birth_metallicity_topheavy_max_zsun is not None
        if has_variant and has_birth_gate and self.star_formation.metallicity_source == "none":
            raise ValueError(
                "star_formation.metallicity_source must not be 'none' when a variant IMF birth-metallicity gate is enabled"
            )
        if any(redshift >= self.mah.z_start_max for redshift in redshifts):
            raise ValueError("all redshifts must be strictly below mah.z_start_max")
        object.__setattr__(self, "schema_version", schema)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "redshifts", redshifts)
        object.__setattr__(self, "base_seed", base_seed)

    @classmethod
    def from_toml(cls, path: str | Path) -> UVLFRunConfig:
        source_path = Path(path).expanduser().resolve()
        with source_path.open("rb") as handle:
            root = tomllib.load(handle)
        if not isinstance(root, dict):
            raise ValueError("TOML root must be a table")
        return _decode_run_config(root, source_path.parent)


_ROOT_REQUIRED = {
    "schema_version",
    "run_id",
    "redshifts",
    "base_seed",
    "cosmology",
    "mah",
    "star_formation",
    "stellar_population",
    "sampling",
    "output",
}
_COSMOLOGY_REQUIRED = {"h0_km_s_mpc", "omega_m", "omega_b"}
_MAH_REQUIRED = {
    "backend",
    "sampler",
    "z_start_max",
    "n_time_steps",
    "tng_mass_bin_width_dex",
    "tng_min_candidates",
    "tng_smoothing_myr",
    "tng_time_grid_mode",
    "thesan_mass_bin_width_dex",
    "thesan_min_candidates",
    "thesan_smoothing_myr",
    "thesan_time_grid_mode",
}
_MAH_OPTIONAL = {"tng_cache_path", "thesan_cache_path"}
_MZR_REQUIRED = {"relation", "returned_fraction", "scatter_dex", "stellar_mass_floor_msun"}
_REGULATOR_REQUIRED = {field.name for field in fields(RegulatorConfig)}
_STAR_FORMATION_REQUIRED = {
    "enable_time_delay",
    "efficiency_normalization",
    "characteristic_halo_mass_msun",
    "low_mass_slope",
    "high_mass_slope",
    "enable_archived_burst_scatter",
    "burst_scatter_dex",
    "burst_scatter_correlation_timescale_myr",
    "burst_scatter_mass_conserving",
    "enable_archived_metallicity",
    "metallicity_source",
}
_STAR_FORMATION_OPTIONAL = {"mzr", "regulator"}
_STELLAR_POPULATION_REQUIRED = {
    "imf_modes",
    "enable_archived_imf_gate",
    "canonical_ssp_path",
    "topheavy_ssp_path",
    "historical_topheavy_redshift_min",
    "source_redshift_gate_enabled",
    "growth_time_threshold_myr",
    "enable_popiii",
    "popiii_ssp_path",
    "popiii_efficiency",
    "popiii_pivot_halo_mass_msun",
    "popiii_low_mass_slope",
    "popiii_high_mass_slope",
    "lw_background_j21",
    "popiii_upper_mass_mode",
}
_STELLAR_POPULATION_OPTIONAL = {
    "topheavy_ssp_template_metallicity_zsun",
    "birth_metallicity_topheavy_max_zsun",
    "popiii_upper_mass_msun",
}
_SAMPLING_REQUIRED = {
    "mass_batch_size",
    "n_halo_mass_samples",
    "n_tracks_per_halo_mass",
    "log10_halo_mass_min_msun",
    "log10_halo_mass_max_msun",
    "muv_bin_edges",
    "workers",
    "mass_function_model",
    "hmf_dlog10m",
    "apply_dust",
}
_OUTPUT_REQUIRED = {"artifact_path"}


def _validate_table(
    value: object,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    allowed = required if optional is None else required | optional
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown TOML key: {name}.{unknown[0]}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"missing required TOML key: {name}.{missing[0]}")
    return value


def _resolve_toml_path(name: str, value: object, base_directory: Path) -> Path:
    path_text = _strict_string(name, value)
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _decode_mzr(value: object) -> MZRConfig:
    table = _validate_table(value, name="star_formation.mzr", required=_MZR_REQUIRED)
    return MZRConfig(
        relation=table["relation"],
        returned_fraction=table["returned_fraction"],
        scatter_dex=table["scatter_dex"],
        stellar_mass_floor_msun=table["stellar_mass_floor_msun"],
    )


def _decode_regulator(value: object) -> RegulatorConfig:
    table = _validate_table(
        value,
        name="star_formation.regulator",
        required=_REGULATOR_REQUIRED,
    )
    return RegulatorConfig(**{name: table[name] for name in _REGULATOR_REQUIRED})


def _decode_run_config(root: Mapping[str, Any], base_directory: Path) -> UVLFRunConfig:
    root_table = _validate_table(root, name="root", required=_ROOT_REQUIRED)
    cosmology_table = _validate_table(
        root_table["cosmology"],
        name="cosmology",
        required=_COSMOLOGY_REQUIRED,
    )
    mah_table = _validate_table(
        root_table["mah"],
        name="mah",
        required=_MAH_REQUIRED,
        optional=_MAH_OPTIONAL,
    )
    star_table = _validate_table(
        root_table["star_formation"],
        name="star_formation",
        required=_STAR_FORMATION_REQUIRED,
        optional=_STAR_FORMATION_OPTIONAL,
    )
    stellar_table = _validate_table(
        root_table["stellar_population"],
        name="stellar_population",
        required=_STELLAR_POPULATION_REQUIRED,
        optional=_STELLAR_POPULATION_OPTIONAL,
    )
    sampling_table = _validate_table(
        root_table["sampling"],
        name="sampling",
        required=_SAMPLING_REQUIRED,
    )
    output_table = _validate_table(
        root_table["output"],
        name="output",
        required=_OUTPUT_REQUIRED,
    )

    tng_cache = (
        _resolve_toml_path("mah.tng_cache_path", mah_table["tng_cache_path"], base_directory)
        if "tng_cache_path" in mah_table
        else None
    )
    thesan_cache = (
        _resolve_toml_path(
            "mah.thesan_cache_path",
            mah_table["thesan_cache_path"],
            base_directory,
        )
        if "thesan_cache_path" in mah_table
        else None
    )
    mzr = _decode_mzr(star_table["mzr"]) if "mzr" in star_table else None
    regulator = (
        _decode_regulator(star_table["regulator"])
        if "regulator" in star_table
        else None
    )
    template_metallicity = (
        stellar_table["topheavy_ssp_template_metallicity_zsun"]
        if "topheavy_ssp_template_metallicity_zsun" in stellar_table
        else None
    )
    birth_gate = (
        stellar_table["birth_metallicity_topheavy_max_zsun"]
        if "birth_metallicity_topheavy_max_zsun" in stellar_table
        else None
    )
    upper_mass = (
        stellar_table["popiii_upper_mass_msun"]
        if "popiii_upper_mass_msun" in stellar_table
        else None
    )

    redshift_values = root_table["redshifts"]
    if type(redshift_values) is not list:
        raise TypeError("redshifts must be a TOML array")
    imf_mode_values = stellar_table["imf_modes"]
    if type(imf_mode_values) is not list:
        raise TypeError("stellar_population.imf_modes must be a TOML array")
    bin_edge_values = sampling_table["muv_bin_edges"]
    if type(bin_edge_values) is not list:
        raise TypeError("sampling.muv_bin_edges must be a TOML array")

    return UVLFRunConfig(
        schema_version=root_table["schema_version"],
        run_id=root_table["run_id"],
        redshifts=tuple(redshift_values),
        base_seed=root_table["base_seed"],
        cosmology=CosmologyConfig(
            h0_km_s_mpc=cosmology_table["h0_km_s_mpc"],
            omega_m=cosmology_table["omega_m"],
            omega_b=cosmology_table["omega_b"],
        ),
        mah=MAHConfig(
            backend=mah_table["backend"],
            sampler=mah_table["sampler"],
            z_start_max=mah_table["z_start_max"],
            n_time_steps=mah_table["n_time_steps"],
            tng_cache_path=tng_cache,
            tng_mass_bin_width_dex=mah_table["tng_mass_bin_width_dex"],
            tng_min_candidates=mah_table["tng_min_candidates"],
            tng_smoothing_myr=mah_table["tng_smoothing_myr"],
            tng_time_grid_mode=mah_table["tng_time_grid_mode"],
            thesan_cache_path=thesan_cache,
            thesan_mass_bin_width_dex=mah_table["thesan_mass_bin_width_dex"],
            thesan_min_candidates=mah_table["thesan_min_candidates"],
            thesan_smoothing_myr=mah_table["thesan_smoothing_myr"],
            thesan_time_grid_mode=mah_table["thesan_time_grid_mode"],
        ),
        star_formation=StarFormationConfig(
            enable_time_delay=star_table["enable_time_delay"],
            efficiency_normalization=star_table["efficiency_normalization"],
            characteristic_halo_mass_msun=star_table["characteristic_halo_mass_msun"],
            low_mass_slope=star_table["low_mass_slope"],
            high_mass_slope=star_table["high_mass_slope"],
            enable_archived_burst_scatter=star_table[
                "enable_archived_burst_scatter"
            ],
            burst_scatter_dex=star_table["burst_scatter_dex"],
            burst_scatter_correlation_timescale_myr=star_table[
                "burst_scatter_correlation_timescale_myr"
            ],
            burst_scatter_mass_conserving=star_table["burst_scatter_mass_conserving"],
            enable_archived_metallicity=star_table["enable_archived_metallicity"],
            metallicity_source=star_table["metallicity_source"],
            mzr=mzr,
            regulator=regulator,
        ),
        stellar_population=StellarPopulationConfig(
            imf_modes=tuple(imf_mode_values),
            enable_archived_imf_gate=stellar_table["enable_archived_imf_gate"],
            canonical_ssp_path=_resolve_toml_path(
                "stellar_population.canonical_ssp_path",
                stellar_table["canonical_ssp_path"],
                base_directory,
            ),
            topheavy_ssp_path=_resolve_toml_path(
                "stellar_population.topheavy_ssp_path",
                stellar_table["topheavy_ssp_path"],
                base_directory,
            ),
            topheavy_ssp_template_metallicity_zsun=template_metallicity,
            historical_topheavy_redshift_min=stellar_table[
                "historical_topheavy_redshift_min"
            ],
            source_redshift_gate_enabled=stellar_table["source_redshift_gate_enabled"],
            growth_time_threshold_myr=stellar_table["growth_time_threshold_myr"],
            birth_metallicity_topheavy_max_zsun=birth_gate,
            enable_popiii=stellar_table["enable_popiii"],
            popiii_ssp_path=_resolve_toml_path(
                "stellar_population.popiii_ssp_path",
                stellar_table["popiii_ssp_path"],
                base_directory,
            ),
            popiii_efficiency=stellar_table["popiii_efficiency"],
            popiii_pivot_halo_mass_msun=stellar_table["popiii_pivot_halo_mass_msun"],
            popiii_low_mass_slope=stellar_table["popiii_low_mass_slope"],
            popiii_high_mass_slope=stellar_table["popiii_high_mass_slope"],
            lw_background_j21=stellar_table["lw_background_j21"],
            popiii_upper_mass_mode=stellar_table["popiii_upper_mass_mode"],
            popiii_upper_mass_msun=upper_mass,
        ),
        sampling=SamplingConfig(
            mass_batch_size=sampling_table["mass_batch_size"],
            n_halo_mass_samples=sampling_table["n_halo_mass_samples"],
            n_tracks_per_halo_mass=sampling_table["n_tracks_per_halo_mass"],
            log10_halo_mass_min_msun=sampling_table["log10_halo_mass_min_msun"],
            log10_halo_mass_max_msun=sampling_table["log10_halo_mass_max_msun"],
            muv_bin_edges=tuple(bin_edge_values),
            workers=sampling_table["workers"],
            mass_function_model=sampling_table["mass_function_model"],
            hmf_dlog10m=sampling_table["hmf_dlog10m"],
            apply_dust=sampling_table["apply_dust"],
        ),
        output=OutputConfig(
            artifact_path=_resolve_toml_path(
                "output.artifact_path",
                output_table["artifact_path"],
                base_directory,
            )
        ),
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CosmologyConfig",
    "MAHConfig",
    "MZRConfig",
    "OutputConfig",
    "RegulatorConfig",
    "SamplingConfig",
    "StarFormationConfig",
    "StellarPopulationConfig",
    "UVLFRunConfig",
]
