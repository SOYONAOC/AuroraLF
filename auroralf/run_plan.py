"""Immutable execution contract for one AuroraLF UVLF run.

The public composition root lives in :mod:`auroralf.driver`.  This module is
dependency-light data only: it records the exact module choices and resolved
objects consumed by UVLF workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from auroralf.chemistry import (
    MZRBirthMetallicityParameters,
    RegulatorMetallicityParameters,
)
from auroralf.config import UVLFRunConfig
from auroralf.mah import Cosmology
from auroralf.model_options import IMFTransitionParameters
from auroralf.sfr import PopIIISFRParameters, SFRModelParameters
from auroralf.uvlf.pipeline_models import LoadedSSPKernels


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


@dataclass(frozen=True, slots=True)
class UVLFModuleSwitches:
    """Normalized module selectors exposed by one run configuration."""

    mah_backend: str
    parameter_sampler: str
    enable_time_delay: bool
    enable_popiii: bool
    imf_modes: tuple[str, ...]
    enable_archived_imf_gate: bool
    enable_archived_burst_scatter: bool
    enable_archived_metallicity: bool
    metallicity_source: str
    mass_function_model: str
    apply_dust: bool
    include_samples: bool

    @classmethod
    def from_config(
        cls,
        config: UVLFRunConfig,
        *,
        include_samples: bool,
    ) -> UVLFModuleSwitches:
        if type(config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
        if type(include_samples) is not bool:
            raise TypeError("include_samples must be exactly bool")
        return cls(
            mah_backend=config.mah.backend,
            parameter_sampler=config.mah.sampler,
            enable_time_delay=config.star_formation.enable_time_delay,
            enable_popiii=config.stellar_population.enable_popiii,
            imf_modes=config.stellar_population.imf_modes,
            enable_archived_imf_gate=(
                config.stellar_population.enable_archived_imf_gate
            ),
            enable_archived_burst_scatter=(
                config.star_formation.enable_archived_burst_scatter
            ),
            enable_archived_metallicity=(
                config.star_formation.enable_archived_metallicity
            ),
            metallicity_source=config.star_formation.metallicity_source,
            mass_function_model=config.sampling.mass_function_model,
            apply_dust=config.sampling.apply_dust,
            include_samples=include_samples,
        )


@dataclass(frozen=True, slots=True)
class UVLFRunPlan:
    """Fully resolved, immutable module assembly consumed by UVLF workers."""

    config: UVLFRunConfig
    switches: UVLFModuleSwitches
    cosmology: Cosmology
    kernels: LoadedSSPKernels
    sfr_parameters: SFRModelParameters
    popiii_parameters: PopIIISFRParameters
    transition_parameters: IMFTransitionParameters
    mzr_parameters: MZRBirthMetallicityParameters | None
    regulator_parameters: RegulatorMetallicityParameters | None
    resolved_simulation_cache_paths: tuple[tuple[float, Path], ...]
    context_token: str
    initialization_load_count: int

    def __post_init__(self) -> None:
        if type(self.config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
        if type(self.switches) is not UVLFModuleSwitches:
            raise TypeError("switches must be exactly UVLFModuleSwitches")
        expected_switches = UVLFModuleSwitches.from_config(
            self.config,
            include_samples=self.switches.include_samples,
        )
        if self.switches != expected_switches:
            raise ValueError("switches must exactly match config module selections")
        for name, value, expected in (
            ("cosmology", self.cosmology, Cosmology),
            ("kernels", self.kernels, LoadedSSPKernels),
            ("sfr_parameters", self.sfr_parameters, SFRModelParameters),
            ("popiii_parameters", self.popiii_parameters, PopIIISFRParameters),
            (
                "transition_parameters",
                self.transition_parameters,
                IMFTransitionParameters,
            ),
        ):
            if type(value) is not expected:
                raise TypeError(f"{name} must be exactly {expected.__name__}")
        if self.cosmology != self.config.cosmology.to_model():
            raise ValueError("cosmology must exactly match config.cosmology")
        if self.sfr_parameters != self.config.star_formation.to_model():
            raise ValueError("sfr_parameters must exactly match config.star_formation")
        if self.popiii_parameters != self.config.stellar_population.to_popiii_model():
            raise ValueError(
                "popiii_parameters must exactly match config.stellar_population"
            )
        if (
            self.transition_parameters
            != self.config.stellar_population.to_imf_transition_model()
        ):
            raise ValueError(
                "transition_parameters must exactly match config.stellar_population"
            )
        if self.mzr_parameters is not None and type(
            self.mzr_parameters
        ) is not MZRBirthMetallicityParameters:
            raise TypeError("mzr_parameters must be exactly MZRBirthMetallicityParameters")
        if self.regulator_parameters is not None and type(
            self.regulator_parameters
        ) is not RegulatorMetallicityParameters:
            raise TypeError(
                "regulator_parameters must be exactly RegulatorMetallicityParameters"
            )
        expected_mzr = (
            self.config.star_formation.mzr.to_model()
            if self.config.star_formation.mzr is not None
            else None
        )
        expected_regulator = (
            self.config.star_formation.regulator.to_model()
            if self.config.star_formation.regulator is not None
            else None
        )
        if self.mzr_parameters != expected_mzr:
            raise ValueError("mzr_parameters must exactly match config")
        if self.regulator_parameters != expected_regulator:
            raise ValueError("regulator_parameters must exactly match config")
        population = self.config.stellar_population
        expected_paths = (
            population.canonical_ssp_path,
            population.topheavy_ssp_path,
            population.popiii_ssp_path,
        )
        actual_paths = (
            self.kernels.canonical_ssp_path,
            self.kernels.topheavy_ssp_path,
            self.kernels.popiii_ssp_path,
        )
        if actual_paths != expected_paths:
            raise ValueError("SSP kernel paths must exactly match config")
        has_topheavy_kernels = self.kernels.topheavy_age_myr is not None
        if len(self.switches.imf_modes) > 1 and not has_topheavy_kernels:
            raise ValueError(
                "variant IMF modes require loaded top-heavy SSP kernels"
            )
        has_popiii_kernels = self.kernels.popiii_age_myr is not None
        if self.switches.enable_popiii and not has_popiii_kernels:
            raise ValueError("enabled Pop III requires loaded Pop III SSP kernels")
        if type(self.resolved_simulation_cache_paths) is not tuple:
            raise TypeError("resolved_simulation_cache_paths must be a tuple")
        seen_redshifts: set[float] = set()
        normalized_paths: list[tuple[float, Path]] = []
        for item in self.resolved_simulation_cache_paths:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("resolved simulation cache entries must be (redshift, Path)")
            redshift = _strict_float("resolved cache redshift", item[0])
            path = item[1]
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("resolved simulation cache path must be an absolute Path")
            if redshift in seen_redshifts:
                raise ValueError("resolved simulation cache redshifts must be unique")
            seen_redshifts.add(redshift)
            normalized_paths.append((redshift, path))
        expected_cache_redshifts = (
            set(self.config.redshifts)
            if self.switches.mah_backend in ("tng", "thesan")
            else set()
        )
        if seen_redshifts != expected_cache_redshifts:
            raise ValueError(
                "resolved simulation cache paths must cover every configured redshift "
                "exactly for the active backend"
            )
        if type(self.context_token) is not str or not self.context_token:
            raise TypeError("context_token must be a non-empty string")
        load_count = _strict_int(
            "initialization_load_count",
            self.initialization_load_count,
        )
        if load_count != 1:
            raise ValueError("initialization_load_count must equal 1")
        object.__setattr__(
            self,
            "resolved_simulation_cache_paths",
            tuple(normalized_paths),
        )
        object.__setattr__(self, "initialization_load_count", load_count)

    @property
    def include_samples(self) -> bool:
        return self.switches.include_samples

    def simulation_cache_path_for(self, redshift: float) -> Path | None:
        for cache_redshift, path in self.resolved_simulation_cache_paths:
            if cache_redshift == redshift:
                return path
        return None

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        return (
            _rebuild_uvlf_run_plan,
            (
                self.config,
                self.switches,
                self.cosmology,
                self.kernels,
                self.sfr_parameters,
                self.popiii_parameters,
                self.transition_parameters,
                self.mzr_parameters,
                self.regulator_parameters,
                self.resolved_simulation_cache_paths,
                self.context_token,
                self.initialization_load_count,
            ),
        )


def _rebuild_uvlf_run_plan(
    config: object,
    switches: object,
    cosmology: object,
    kernels: object,
    sfr_parameters: object,
    popiii_parameters: object,
    transition_parameters: object,
    mzr_parameters: object,
    regulator_parameters: object,
    resolved_simulation_cache_paths: object,
    context_token: object,
    initialization_load_count: object,
) -> UVLFRunPlan:
    return UVLFRunPlan(
        config=config,
        switches=switches,
        cosmology=cosmology,
        kernels=kernels,
        sfr_parameters=sfr_parameters,
        popiii_parameters=popiii_parameters,
        transition_parameters=transition_parameters,
        mzr_parameters=mzr_parameters,
        regulator_parameters=regulator_parameters,
        resolved_simulation_cache_paths=resolved_simulation_cache_paths,
        context_token=context_token,
        initialization_load_count=initialization_load_count,
    )


__all__ = ["UVLFModuleSwitches", "UVLFRunPlan"]
