from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
import multiprocessing as mp
from numbers import Integral, Real
import os
from pathlib import Path
import time

import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.io.schema import HaloSampleTable
from auroralf.chemistry import (
    MZRBirthMetallicityParameters,
    RegulatorMetallicityParameters,
)
from auroralf.mah import Cosmology
from auroralf.mah.thesan import preload_thesan_mah_cache
from auroralf.mah.tng import preload_tng_mah_cache
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
    _require_nonnegative,
    _working_float_1d,
)
from auroralf.seeding import derive_hmf_mass_seed, derive_pipeline_random_seeds
from auroralf.sfr import PopIIISFRParameters, SFRModelParameters
from .dust import compute_dust_attenuated_uvlf
from .hmf_sampling import prepare_reed07_hmf_interpolator, uv_luminosity_to_muv
from .imf import IMFTransitionParameters, validate_imf_mode
from .pipeline import (
    HaloModeEvaluation,
    LoadedSSPKernels,
    SharedHaloBatch,
    evaluate_shared_halo_batch,
    load_ssp_kernels,
    prepare_shared_halo_batch,
)
from .streaming import WeightedHistogramAccumulator


def _strict_float_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int_scalar(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer non-boolean value")
    return int(value)


def _immutable_float_vector(name: str, value: object) -> np.ndarray:
    array = _working_float_1d(name, value)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class _MassTaskSpec:
    redshift: float
    mass_index: int
    halo_mass_msun: float
    mass_weight_per_mpc3: float

    def __post_init__(self) -> None:
        redshift = _strict_float_scalar("redshift", self.redshift)
        mass_index = _strict_int_scalar("mass_index", self.mass_index)
        halo_mass = _strict_float_scalar("halo_mass_msun", self.halo_mass_msun)
        mass_weight = _strict_float_scalar(
            "mass_weight_per_mpc3",
            self.mass_weight_per_mpc3,
        )
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if mass_index < 0:
            raise ValueError("mass_index must be non-negative")
        if halo_mass <= 0.0:
            raise ValueError("halo_mass_msun must be positive")
        if mass_weight < 0.0:
            raise ValueError("mass_weight_per_mpc3 must be non-negative")
        for name, normalized in (
            ("redshift", redshift),
            ("mass_index", mass_index),
            ("halo_mass_msun", halo_mass),
            ("mass_weight_per_mpc3", mass_weight),
        ):
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True, slots=True)
class _MassModeTaskResult:
    imf_mode: str
    uv_luminosity_erg_per_s_hz: np.ndarray
    topheavy_source_count: int
    starforming_source_count: int
    popiii_source_count: int
    active_source_count: int
    evaluation_seconds: float

    def __post_init__(self) -> None:
        mode = validate_imf_mode(self.imf_mode)
        luminosity = _immutable_float_vector(
            "uv_luminosity_erg_per_s_hz",
            self.uv_luminosity_erg_per_s_hz,
        )
        counts: dict[str, int] = {}
        for name in (
            "topheavy_source_count",
            "starforming_source_count",
            "popiii_source_count",
            "active_source_count",
        ):
            count = _strict_int_scalar(name, getattr(self, name))
            if count < 0:
                raise ValueError(f"{name} must be non-negative")
            counts[name] = count
        if counts["topheavy_source_count"] > counts["starforming_source_count"]:
            raise ValueError("topheavy_source_count must not exceed starforming_source_count")
        if counts["starforming_source_count"] > counts["active_source_count"]:
            raise ValueError("starforming_source_count must not exceed active_source_count")
        if counts["popiii_source_count"] > counts["active_source_count"]:
            raise ValueError("popiii_source_count must not exceed active_source_count")
        seconds = _strict_float_scalar("evaluation_seconds", self.evaluation_seconds)
        if seconds < 0.0:
            raise ValueError("evaluation_seconds must be non-negative")
        object.__setattr__(self, "imf_mode", mode)
        object.__setattr__(self, "uv_luminosity_erg_per_s_hz", luminosity)
        object.__setattr__(self, "evaluation_seconds", seconds)
        for name, count in counts.items():
            object.__setattr__(self, name, count)

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        return (
            _rebuild_mass_mode_task_result,
            (
                self.imf_mode,
                self.uv_luminosity_erg_per_s_hz,
                self.topheavy_source_count,
                self.starforming_source_count,
                self.popiii_source_count,
                self.active_source_count,
                self.evaluation_seconds,
            ),
        )


def _rebuild_mass_mode_task_result(
    imf_mode: str,
    uv_luminosity_erg_per_s_hz: object,
    topheavy_source_count: object,
    starforming_source_count: object,
    popiii_source_count: object,
    active_source_count: object,
    evaluation_seconds: object,
) -> _MassModeTaskResult:
    return _MassModeTaskResult(
        imf_mode=imf_mode,
        uv_luminosity_erg_per_s_hz=uv_luminosity_erg_per_s_hz,
        topheavy_source_count=topheavy_source_count,
        starforming_source_count=starforming_source_count,
        popiii_source_count=popiii_source_count,
        active_source_count=active_source_count,
        evaluation_seconds=evaluation_seconds,
    )


@dataclass(frozen=True, slots=True)
class _MassTaskResult:
    redshift: float
    mass_index: int
    halo_mass_msun: float
    mass_weight_per_mpc3: float
    final_sfr_mean_msun_per_yr: float
    final_popiii_sfr_mean_msun_per_yr: float
    mode_results: tuple[_MassModeTaskResult, ...]
    shared_preparation_seconds: float
    worker_pid: int
    worker_context_token: str
    worker_initialization_load_count: int
    final_sfr_msun_per_yr: np.ndarray | None = None
    final_popiii_sfr_msun_per_yr: np.ndarray | None = None

    def __post_init__(self) -> None:
        spec = _MassTaskSpec(
            redshift=self.redshift,
            mass_index=self.mass_index,
            halo_mass_msun=self.halo_mass_msun,
            mass_weight_per_mpc3=self.mass_weight_per_mpc3,
        )
        final_sfr = _strict_float_scalar(
            "final_sfr_mean_msun_per_yr",
            self.final_sfr_mean_msun_per_yr,
        )
        final_popiii_sfr = _strict_float_scalar(
            "final_popiii_sfr_mean_msun_per_yr",
            self.final_popiii_sfr_mean_msun_per_yr,
        )
        if final_sfr < 0.0 or final_popiii_sfr < 0.0:
            raise ValueError("final SFR means must be non-negative")
        if type(self.mode_results) is not tuple or not self.mode_results:
            raise TypeError("mode_results must be a non-empty tuple")
        if any(type(result) is not _MassModeTaskResult for result in self.mode_results):
            raise TypeError("mode_results entries must be exactly _MassModeTaskResult")
        normalized_mode_results = tuple(
            _rebuild_mass_mode_task_result(
                result.imf_mode,
                result.uv_luminosity_erg_per_s_hz,
                result.topheavy_source_count,
                result.starforming_source_count,
                result.popiii_source_count,
                result.active_source_count,
                result.evaluation_seconds,
            )
            for result in self.mode_results
        )
        modes = tuple(result.imf_mode for result in normalized_mode_results)
        if len(set(modes)) != len(modes):
            raise ValueError("mode_results must contain unique IMF modes")
        sample_count = normalized_mode_results[0].uv_luminosity_erg_per_s_hz.size
        if any(
            result.uv_luminosity_erg_per_s_hz.size != sample_count
            for result in normalized_mode_results
        ):
            raise ValueError("mode_results luminosity arrays must have equal lengths")
        if (self.final_sfr_msun_per_yr is None) != (
            self.final_popiii_sfr_msun_per_yr is None
        ):
            raise ValueError("per-track final SFR arrays must be both present or both absent")
        final_sfr_samples: np.ndarray | None = None
        final_popiii_sfr_samples: np.ndarray | None = None
        if self.final_sfr_msun_per_yr is not None:
            final_sfr_samples = _immutable_float_vector(
                "final_sfr_msun_per_yr",
                self.final_sfr_msun_per_yr,
            )
            final_popiii_sfr_samples = _immutable_float_vector(
                "final_popiii_sfr_msun_per_yr",
                self.final_popiii_sfr_msun_per_yr,
            )
            if (
                final_sfr_samples.size != sample_count
                or final_popiii_sfr_samples.size != sample_count
            ):
                raise ValueError(
                    "per-track final SFR arrays must match luminosity sample count"
                )
        seconds = _strict_float_scalar(
            "shared_preparation_seconds",
            self.shared_preparation_seconds,
        )
        if seconds < 0.0:
            raise ValueError("shared_preparation_seconds must be non-negative")
        worker_pid = _strict_int_scalar("worker_pid", self.worker_pid)
        initialization_count = _strict_int_scalar(
            "worker_initialization_load_count",
            self.worker_initialization_load_count,
        )
        if worker_pid <= 0:
            raise ValueError("worker_pid must be positive")
        if type(self.worker_context_token) is not str or not self.worker_context_token:
            raise TypeError("worker_context_token must be a non-empty string")
        if initialization_count != 1:
            raise ValueError("worker_initialization_load_count must equal 1")
        for name in (
            "redshift",
            "mass_index",
            "halo_mass_msun",
            "mass_weight_per_mpc3",
        ):
            object.__setattr__(self, name, getattr(spec, name))
        object.__setattr__(self, "final_sfr_mean_msun_per_yr", final_sfr)
        object.__setattr__(
            self,
            "final_popiii_sfr_mean_msun_per_yr",
            final_popiii_sfr,
        )
        object.__setattr__(self, "shared_preparation_seconds", seconds)
        object.__setattr__(self, "worker_pid", worker_pid)
        object.__setattr__(self, "mode_results", normalized_mode_results)
        object.__setattr__(self, "final_sfr_msun_per_yr", final_sfr_samples)
        object.__setattr__(
            self,
            "final_popiii_sfr_msun_per_yr",
            final_popiii_sfr_samples,
        )
        object.__setattr__(
            self,
            "worker_initialization_load_count",
            initialization_count,
        )

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        return (
            _rebuild_mass_task_result,
            (
                self.redshift,
                self.mass_index,
                self.halo_mass_msun,
                self.mass_weight_per_mpc3,
                self.final_sfr_mean_msun_per_yr,
                self.final_popiii_sfr_mean_msun_per_yr,
                self.mode_results,
                self.shared_preparation_seconds,
                self.worker_pid,
                self.worker_context_token,
                self.worker_initialization_load_count,
                self.final_sfr_msun_per_yr,
                self.final_popiii_sfr_msun_per_yr,
            ),
        )


def _rebuild_mass_task_result(
    redshift: object,
    mass_index: object,
    halo_mass_msun: object,
    mass_weight_per_mpc3: object,
    final_sfr_mean_msun_per_yr: object,
    final_popiii_sfr_mean_msun_per_yr: object,
    mode_results: object,
    shared_preparation_seconds: object,
    worker_pid: object,
    worker_context_token: object,
    worker_initialization_load_count: object,
    final_sfr_msun_per_yr: object = None,
    final_popiii_sfr_msun_per_yr: object = None,
) -> _MassTaskResult:
    return _MassTaskResult(
        redshift=redshift,
        mass_index=mass_index,
        halo_mass_msun=halo_mass_msun,
        mass_weight_per_mpc3=mass_weight_per_mpc3,
        final_sfr_mean_msun_per_yr=final_sfr_mean_msun_per_yr,
        final_popiii_sfr_mean_msun_per_yr=final_popiii_sfr_mean_msun_per_yr,
        mode_results=mode_results,
        shared_preparation_seconds=shared_preparation_seconds,
        worker_pid=worker_pid,
        worker_context_token=worker_context_token,
        worker_initialization_load_count=worker_initialization_load_count,
        final_sfr_msun_per_yr=final_sfr_msun_per_yr,
        final_popiii_sfr_msun_per_yr=final_popiii_sfr_msun_per_yr,
    )


@dataclass(frozen=True, slots=True)
class _SchedulingSnapshot:
    running_count: int
    completed_waiting_count: int
    total_occupancy: int
    submitted_count: int
    consumed_count: int


@dataclass(frozen=True, slots=True)
class _WorkerContext:
    config: UVLFRunConfig
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
    include_samples: bool

    def __post_init__(self) -> None:
        if type(self.config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
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
        if type(self.resolved_simulation_cache_paths) is not tuple:
            raise TypeError("resolved_simulation_cache_paths must be a tuple")
        seen_redshifts: set[float] = set()
        normalized_paths: list[tuple[float, Path]] = []
        for item in self.resolved_simulation_cache_paths:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("resolved simulation cache entries must be (redshift, Path)")
            redshift = _strict_float_scalar("resolved cache redshift", item[0])
            path = item[1]
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("resolved simulation cache path must be an absolute Path")
            if redshift in seen_redshifts:
                raise ValueError("resolved simulation cache redshifts must be unique")
            seen_redshifts.add(redshift)
            normalized_paths.append((redshift, path))
        expected_cache_redshifts = (
            set(self.config.redshifts)
            if self.config.mah.backend in ("tng", "thesan")
            else set()
        )
        if seen_redshifts != expected_cache_redshifts:
            raise ValueError(
                "resolved simulation cache paths must cover every configured redshift "
                "exactly for the active backend"
            )
        if type(self.context_token) is not str or not self.context_token:
            raise TypeError("context_token must be a non-empty string")
        if type(self.include_samples) is not bool:
            raise TypeError("include_samples must be exactly bool")
        load_count = _strict_int_scalar(
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

    def simulation_cache_path_for(self, redshift: float) -> Path | None:
        for cache_redshift, path in self.resolved_simulation_cache_paths:
            if cache_redshift == redshift:
                return path
        return None

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...]]:
        del protocol
        return (
            _rebuild_worker_context,
            (
                self.config,
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
                self.include_samples,
            ),
        )


def _rebuild_worker_context(
    config: object,
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
    include_samples: object,
) -> _WorkerContext:
    return _WorkerContext(
        config=config,
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
        include_samples=include_samples,
    )


_WORKER_CONTEXT: _WorkerContext | None = None


def _build_worker_context(
    config: UVLFRunConfig,
    *,
    include_samples: bool = False,
) -> _WorkerContext:
    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if type(include_samples) is not bool:
        raise TypeError("include_samples must be exactly bool")
    cosmology = config.cosmology.to_model()
    resolved_cache_paths: list[tuple[float, Path]] = []
    if config.mah.backend == "tng":
        if config.mah.tng_cache_path is None:
            raise RuntimeError("validated TNG config has no cache path")
        for redshift in config.redshifts:
            resolved_cache_paths.append(
                (
                    redshift,
                    preload_tng_mah_cache(
                        config.mah.tng_cache_path,
                        z_final=redshift,
                        cosmology=cosmology,
                    ),
                )
            )
    elif config.mah.backend == "thesan":
        if config.mah.thesan_cache_path is None:
            raise RuntimeError("validated THESAN config has no cache path")
        for redshift in config.redshifts:
            resolved_cache_paths.append(
                (
                    redshift,
                    preload_thesan_mah_cache(
                        config.mah.thesan_cache_path,
                        z_final=redshift,
                        cosmology=cosmology,
                    ),
                )
            )
    kernels = load_ssp_kernels(
        ssp_file=config.stellar_population.canonical_ssp_path,
        imf_modes=config.stellar_population.imf_modes,
        topheavy_ssp_file=config.stellar_population.topheavy_ssp_path,
        topheavy_ssp_metallicity=(
            config.stellar_population.topheavy_ssp_template_metallicity_zsun
        ),
        enable_popiii=config.stellar_population.enable_popiii,
        popiii_ssp_file=config.stellar_population.popiii_ssp_path,
    )
    return _WorkerContext(
        config=config,
        cosmology=cosmology,
        kernels=kernels,
        sfr_parameters=config.star_formation.to_model(),
        popiii_parameters=config.stellar_population.to_popiii_model(),
        transition_parameters=config.stellar_population.to_imf_transition_model(),
        mzr_parameters=(
            config.star_formation.mzr.to_model()
            if config.star_formation.mzr is not None
            else None
        ),
        regulator_parameters=(
            config.star_formation.regulator.to_model()
            if config.star_formation.regulator is not None
            else None
        ),
        resolved_simulation_cache_paths=tuple(resolved_cache_paths),
        context_token=f"{os.getpid()}-{time.time_ns()}-{id(config)}",
        initialization_load_count=1,
        include_samples=include_samples,
    )


def _initialize_worker(config: UVLFRunConfig, include_samples: bool = False) -> None:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        raise RuntimeError("worker context is already initialized")
    _WORKER_CONTEXT = _build_worker_context(
        config,
        include_samples=include_samples,
    )


def _clear_worker_context_for_tests() -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = None


def _run_mass_task(spec: _MassTaskSpec) -> _MassTaskResult:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("worker context is not initialized")
    if type(spec) is not _MassTaskSpec:
        raise TypeError("spec must be exactly _MassTaskSpec")
    config = context.config
    if spec.redshift not in config.redshifts:
        raise ValueError("mass task redshift is not configured in worker context")
    resolved_cache_path = context.simulation_cache_path_for(spec.redshift)
    prepare_started = time.perf_counter()
    shared = prepare_shared_halo_batch(
        n_tracks=config.sampling.n_tracks_per_halo_mass,
        z_final=spec.redshift,
        Mh_final=spec.halo_mass_msun,
        cosmology=context.cosmology,
        random_seeds=derive_pipeline_random_seeds(
            config.base_seed,
            redshift=spec.redshift,
            mass_index=spec.mass_index,
        ),
        mass_index=spec.mass_index,
        z_start_max=config.mah.z_start_max,
        n_grid=config.mah.n_time_steps,
        enable_popiii=config.stellar_population.enable_popiii,
        popiii_sfr_parameters=context.popiii_parameters,
        sampler=config.mah.sampler,
        mah_backend=config.mah.backend,
        tng_mah_cache_path=(
            resolved_cache_path
            if config.mah.backend == "tng"
            else config.mah.tng_cache_path
        ),
        tng_mass_bin_width_dex=config.mah.tng_mass_bin_width_dex,
        tng_min_candidates=config.mah.tng_min_candidates,
        tng_smoothing_myr=config.mah.tng_smoothing_myr,
        tng_time_grid_mode=config.mah.tng_time_grid_mode,
        thesan_mah_cache_path=(
            resolved_cache_path
            if config.mah.backend == "thesan"
            else config.mah.thesan_cache_path
        ),
        thesan_mass_bin_width_dex=config.mah.thesan_mass_bin_width_dex,
        thesan_min_candidates=config.mah.thesan_min_candidates,
        thesan_smoothing_myr=config.mah.thesan_smoothing_myr,
        thesan_time_grid_mode=config.mah.thesan_time_grid_mode,
        enable_time_delay=config.star_formation.enable_time_delay,
        sfr_model_parameters=context.sfr_parameters,
        mzr_metallicity_parameters=context.mzr_parameters,
        regulator_metallicity_parameters=context.regulator_parameters,
        burst_scatter_dex=config.star_formation.burst_scatter_dex,
        burst_scatter_timescale_myr=(
            config.star_formation.burst_scatter_correlation_timescale_myr
        ),
        burst_scatter_preserve_mean=(
            config.star_formation.burst_scatter_mass_conserving
        ),
    )
    shared_seconds = time.perf_counter() - prepare_started
    final_sfr_mean = float(np.mean(shared.sfr_msun_per_yr_grid[:, -1]))
    final_popiii_sfr_mean = float(
        np.mean(shared.popiii_sfr_msun_per_yr_grid[:, -1])
    )
    final_sfr_samples = (
        shared.sfr_msun_per_yr_grid[:, -1]
        if context.include_samples
        else None
    )
    final_popiii_sfr_samples = (
        shared.popiii_sfr_msun_per_yr_grid[:, -1]
        if context.include_samples
        else None
    )
    mode_results: list[_MassModeTaskResult] = []
    for mode in config.stellar_population.imf_modes:
        evaluation_started = time.perf_counter()
        evaluation = evaluate_shared_halo_batch(
            shared,
            imf_mode=mode,
            transition_parameters=context.transition_parameters,
            kernels=context.kernels,
            ssp_lookback_max_myr=100.0,
        )
        mode_results.append(
            _MassModeTaskResult(
                imf_mode=mode,
                uv_luminosity_erg_per_s_hz=(
                    evaluation.uv_luminosity_erg_per_s_hz
                ),
                topheavy_source_count=evaluation.topheavy_source_count,
                starforming_source_count=evaluation.starforming_source_count,
                popiii_source_count=evaluation.popiii_source_count,
                active_source_count=evaluation.active_source_count,
                evaluation_seconds=time.perf_counter() - evaluation_started,
            )
        )
        del evaluation
    del shared
    return _MassTaskResult(
        redshift=spec.redshift,
        mass_index=spec.mass_index,
        halo_mass_msun=spec.halo_mass_msun,
        mass_weight_per_mpc3=spec.mass_weight_per_mpc3,
        final_sfr_mean_msun_per_yr=final_sfr_mean,
        final_popiii_sfr_mean_msun_per_yr=final_popiii_sfr_mean,
        mode_results=tuple(mode_results),
        shared_preparation_seconds=shared_seconds,
        worker_pid=os.getpid(),
        worker_context_token=context.context_token,
        worker_initialization_load_count=context.initialization_load_count,
        final_sfr_msun_per_yr=final_sfr_samples,
        final_popiii_sfr_msun_per_yr=final_popiii_sfr_samples,
    )


def _validate_scheduled_result(
    spec: _MassTaskSpec,
    result: object,
) -> _MassTaskResult:
    if type(spec) is not _MassTaskSpec:
        raise TypeError("mass task spec must be exactly _MassTaskSpec")
    normalized_spec = _MassTaskSpec(
        redshift=spec.redshift,
        mass_index=spec.mass_index,
        halo_mass_msun=spec.halo_mass_msun,
        mass_weight_per_mpc3=spec.mass_weight_per_mpc3,
    )
    if type(result) is not _MassTaskResult:
        raise TypeError("mass task must return exactly _MassTaskResult")
    normalized_result = _rebuild_mass_task_result(
        result.redshift,
        result.mass_index,
        result.halo_mass_msun,
        result.mass_weight_per_mpc3,
        result.final_sfr_mean_msun_per_yr,
        result.final_popiii_sfr_mean_msun_per_yr,
        result.mode_results,
        result.shared_preparation_seconds,
        result.worker_pid,
        result.worker_context_token,
        result.worker_initialization_load_count,
        result.final_sfr_msun_per_yr,
        result.final_popiii_sfr_msun_per_yr,
    )
    if normalized_result.mass_index != normalized_spec.mass_index:
        raise RuntimeError(
            f"mass task result index {normalized_result.mass_index} does not match submitted "
            f"index {normalized_spec.mass_index}"
        )
    if normalized_result.redshift != normalized_spec.redshift:
        raise RuntimeError("mass task result redshift does not match submitted spec")
    if normalized_result.halo_mass_msun != normalized_spec.halo_mass_msun:
        raise RuntimeError("mass task result halo mass does not match submitted spec")
    if normalized_result.mass_weight_per_mpc3 != normalized_spec.mass_weight_per_mpc3:
        raise RuntimeError("mass task result mass weight does not match submitted spec")
    return normalized_result


def _ordered_parallel_results(
    specs: Iterable[_MassTaskSpec],
    *,
    executor: Executor,
    max_workers: int,
    scheduling_observer: Callable[[_SchedulingSnapshot], None] | None = None,
) -> Iterator[_MassTaskResult]:
    worker_count = _strict_int_scalar("max_workers", max_workers)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    window = 2 * worker_count
    spec_iterator = iter(specs)
    running: dict[Future[_MassTaskResult], _MassTaskSpec] = {}
    completed: dict[int, tuple[_MassTaskSpec, _MassTaskResult]] = {}
    submitted_indices: set[int] = set()
    submitted_count = 0
    consumed_count = 0
    expected_index: int | None = None
    last_submitted_index: int | None = None
    exhausted = False

    def notify() -> None:
        if scheduling_observer is None:
            return
        scheduling_observer(
            _SchedulingSnapshot(
                running_count=len(running),
                completed_waiting_count=len(completed),
                total_occupancy=len(running) + len(completed),
                submitted_count=submitted_count,
                consumed_count=consumed_count,
            )
        )

    def fill_window() -> None:
        nonlocal exhausted, expected_index, last_submitted_index, submitted_count
        while not exhausted and len(running) + len(completed) < window:
            try:
                spec = next(spec_iterator)
            except StopIteration:
                exhausted = True
                break
            if type(spec) is not _MassTaskSpec:
                raise TypeError("mass spec iterator must yield exactly _MassTaskSpec")
            if spec.mass_index in submitted_indices:
                raise RuntimeError(f"duplicate mass task index {spec.mass_index}")
            if last_submitted_index is not None and spec.mass_index != last_submitted_index + 1:
                raise RuntimeError(
                    f"mass task indices must be contiguous; got {spec.mass_index} after "
                    f"{last_submitted_index}"
                )
            if expected_index is None:
                expected_index = spec.mass_index
            future = executor.submit(_run_mass_task, spec)
            running[future] = spec
            submitted_indices.add(spec.mass_index)
            last_submitted_index = spec.mass_index
            submitted_count += 1
            notify()

    try:
        fill_window()
        while running or completed:
            while expected_index is not None and expected_index in completed:
                spec, result = completed.pop(expected_index)
                consumed_count += 1
                notify()
                yield _validate_scheduled_result(spec, result)
                expected_index += 1
                fill_window()
            if not running:
                if completed:
                    raise RuntimeError(
                        f"missing mass task result for expected index {expected_index}"
                    )
                break
            done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            if not done:
                raise RuntimeError("parallel mass scheduler wait returned no completed futures")
            for future in done:
                spec = running.pop(future)
                result = _validate_scheduled_result(spec, future.result())
                if result.mass_index in completed:
                    raise RuntimeError(f"duplicate mass task result index {result.mass_index}")
                completed[result.mass_index] = (spec, result)
            notify()
    except BaseException:
        for future in running:
            future.cancel()
        raise


@dataclass
class _ModeAccumulatorState:
    histogram: WeightedHistogramAccumulator
    sample_count: int = 0
    valid_sample_count: int = 0
    topheavy_source_count: int = 0
    starforming_source_count: int = 0
    popiii_source_count: int = 0
    active_source_count: int = 0
    sfrd_msun_per_yr_per_mpc3: float = 0.0
    popiii_sfrd_msun_per_yr_per_mpc3: float = 0.0
    evaluation_seconds: float = 0.0


def _strict_nonnegative_float_1d(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = _working_float_1d(name, value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    _require_nonnegative(name, array)
    return array


def _observed_uvlf(
    *,
    centers: np.ndarray,
    intrinsic_phi: np.ndarray,
    intrinsic_sigma: np.ndarray,
    redshift: float,
    apply_dust: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not apply_dust:
        return intrinsic_phi.copy(), intrinsic_sigma.copy()
    dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=centers,
        intrinsic_phi=intrinsic_phi,
        z=redshift,
        muv_obs=centers,
    )
    observed = _strict_nonnegative_float_1d(
        "dust['phi_obs']",
        dust["phi_obs"],
        expected_shape=intrinsic_phi.shape,
    )
    fractional_sigma = np.divide(
        intrinsic_sigma,
        intrinsic_phi,
        out=np.full_like(intrinsic_sigma, np.nan),
        where=intrinsic_phi > 0.0,
    )
    return observed, observed * fractional_sigma


def _iter_mass_task_specs(
    config: UVLFRunConfig,
    *,
    redshift: float,
    cosmology: Cosmology,
    hmf_seconds: list[float],
) -> Iterator[_MassTaskSpec]:
    hmf_started = time.perf_counter()
    interpolator = prepare_reed07_hmf_interpolator(
        log10_halo_mass_min_msun=config.sampling.log10_halo_mass_min_msun,
        log10_halo_mass_max_msun=config.sampling.log10_halo_mass_max_msun,
        z_obs=redshift,
        cosmology=cosmology,
        hmf_dlog10m=config.sampling.hmf_dlog10m,
    )
    hmf_seconds[0] += time.perf_counter() - hmf_started
    mass_rng = np.random.default_rng(derive_hmf_mass_seed(config.base_seed, redshift))
    for chunk_start in range(
        0,
        config.sampling.n_halo_mass_samples,
        config.sampling.mass_batch_size,
    ):
        chunk_stop = min(
            chunk_start + config.sampling.mass_batch_size,
            config.sampling.n_halo_mass_samples,
        )
        chunk_size = chunk_stop - chunk_start
        hmf_started = time.perf_counter()
        log_mass = mass_rng.uniform(
            config.sampling.log10_halo_mass_min_msun,
            config.sampling.log10_halo_mass_max_msun,
            size=chunk_size,
        )
        mass_msun = np.power(10.0, log_mass)
        dndm = _strict_nonnegative_float_1d(
            "HMF dndm",
            interpolator.evaluate(mass_msun),
            expected_shape=mass_msun.shape,
        )
        dndlogm = mass_msun * np.log(10.0) * dndm
        mass_weight = (
            config.sampling.log10_halo_mass_max_msun
            - config.sampling.log10_halo_mass_min_msun
        ) * dndlogm / config.sampling.n_halo_mass_samples
        hmf_seconds[0] += time.perf_counter() - hmf_started
        for local_index in range(chunk_size):
            mass_index = chunk_start + local_index
            yield _MassTaskSpec(
                redshift=redshift,
                mass_index=mass_index,
                halo_mass_msun=float(mass_msun[local_index]),
                mass_weight_per_mpc3=float(mass_weight[local_index]),
            )
        del log_mass, mass_msun, dndm, dndlogm, mass_weight


def _consume_mass_task_result(
    result: _MassTaskResult,
    *,
    config: UVLFRunConfig,
    states: dict[str, _ModeAccumulatorState],
) -> float:
    expected_modes = config.stellar_population.imf_modes
    actual_modes = tuple(mode_result.imf_mode for mode_result in result.mode_results)
    if actual_modes != expected_modes:
        raise RuntimeError(
            f"mass task IMF mode order {actual_modes} does not match config {expected_modes}"
        )
    track_weight = (
        result.mass_weight_per_mpc3 / config.sampling.n_tracks_per_halo_mass
    )
    for mode_result in result.mode_results:
        if (
            mode_result.uv_luminosity_erg_per_s_hz.size
            != config.sampling.n_tracks_per_halo_mass
        ):
            raise RuntimeError(
                "mass task luminosity count does not match n_tracks_per_halo_mass"
            )
        state = states[mode_result.imf_mode]
        muv = np.asarray(
            uv_luminosity_to_muv(mode_result.uv_luminosity_erg_per_s_hz),
            dtype=float,
        )
        weights = np.full(muv.shape, track_weight)
        state.histogram.update(muv, weights)
        state.sample_count += int(muv.size)
        state.valid_sample_count += int(np.count_nonzero(np.isfinite(muv)))
        state.topheavy_source_count += mode_result.topheavy_source_count
        state.starforming_source_count += mode_result.starforming_source_count
        state.popiii_source_count += mode_result.popiii_source_count
        state.active_source_count += mode_result.active_source_count
        state.sfrd_msun_per_yr_per_mpc3 += (
            result.final_sfr_mean_msun_per_yr * result.mass_weight_per_mpc3
        )
        state.popiii_sfrd_msun_per_yr_per_mpc3 += (
            result.final_popiii_sfr_mean_msun_per_yr
            * result.mass_weight_per_mpc3
        )
        state.evaluation_seconds += mode_result.evaluation_seconds
    return result.shared_preparation_seconds


def _observe_halo_samples(
    result: _MassTaskResult,
    *,
    config: UVLFRunConfig,
    observer: Callable[[HaloSampleTable], None],
) -> None:
    final_sfr = result.final_sfr_msun_per_yr
    final_popiii_sfr = result.final_popiii_sfr_msun_per_yr
    if final_sfr is None or final_popiii_sfr is None:
        raise RuntimeError("sample-enabled mass result is missing per-track final SFR arrays")
    track_count = config.sampling.n_tracks_per_halo_mass
    if final_sfr.size != track_count or final_popiii_sfr.size != track_count:
        raise RuntimeError("per-track final SFR count does not match config")
    track_weight = result.mass_weight_per_mpc3 / track_count
    for mode_result in result.mode_results:
        luminosity = mode_result.uv_luminosity_erg_per_s_hz
        if luminosity.size != track_count:
            raise RuntimeError(
                "mass task luminosity count does not match n_tracks_per_halo_mass"
            )
        observer(
            HaloSampleTable(
                redshift=result.redshift,
                imf_mode=mode_result.imf_mode,
                mass_index=np.full(track_count, result.mass_index, dtype=np.int64),
                track_index=np.arange(track_count, dtype=np.int64),
                halo_mass_msun=np.full(track_count, result.halo_mass_msun),
                mass_weight_per_mpc3=np.full(track_count, track_weight),
                uv_luminosity_erg_per_s_hz=luminosity,
                muv=np.asarray(uv_luminosity_to_muv(luminosity), dtype=float),
                sfr_msun_per_yr=final_sfr,
                popiii_sfr_msun_per_yr=final_popiii_sfr,
            )
        )


def run_uvlf_streaming(
    config: UVLFRunConfig,
    *,
    _mass_result_observer: Callable[[_MassTaskResult], None] | None = None,
    _scheduling_observer: Callable[[_SchedulingSnapshot], None] | None = None,
    _halo_sample_observer: Callable[[HaloSampleTable], None] | None = None,
) -> UVLFRunResult:
    """Run the bounded shared-batch UVLF core without retaining halo histories."""

    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if _halo_sample_observer is not None and not callable(_halo_sample_observer):
        raise TypeError("_halo_sample_observer must be callable or None")
    started = time.perf_counter()
    cosmology = config.cosmology.to_model()
    redshift_results: list[RedshiftResult] = []
    all_diagnostics: list[ModeRunDiagnostics] = []
    mode_count = len(config.stellar_population.imf_modes)
    worker_count = min(
        config.sampling.workers,
        config.sampling.n_halo_mass_samples,
    )
    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, _halo_sample_observer is not None),
        )
    else:
        if _WORKER_CONTEXT is not None:
            raise RuntimeError("serial worker context was unexpectedly already initialized")
        _initialize_worker(config, _halo_sample_observer is not None)

    try:
        for redshift in config.redshifts:
            states = {
                mode: _ModeAccumulatorState(
                    histogram=WeightedHistogramAccumulator(
                        np.asarray(config.sampling.muv_bin_edges, dtype=float)
                    )
                )
                for mode in config.stellar_population.imf_modes
            }
            hmf_seconds = [0.0]
            specs = _iter_mass_task_specs(
                config,
                redshift=redshift,
                cosmology=cosmology,
                hmf_seconds=hmf_seconds,
            )
            if executor is None:
                task_results: Iterable[_MassTaskResult] = (
                    _validate_scheduled_result(spec, _run_mass_task(spec))
                    for spec in specs
                )
            else:
                task_results = _ordered_parallel_results(
                    specs,
                    executor=executor,
                    max_workers=worker_count,
                    scheduling_observer=_scheduling_observer,
                )
            shared_seconds = 0.0
            consumed_count = 0
            for expected_index, result in enumerate(task_results):
                if result.mass_index != expected_index:
                    raise RuntimeError(
                        f"mass task results must be consumed in order; expected "
                        f"{expected_index}, got {result.mass_index}"
                    )
                if _mass_result_observer is not None:
                    _mass_result_observer(result)
                if _halo_sample_observer is not None:
                    _observe_halo_samples(
                        result,
                        config=config,
                        observer=_halo_sample_observer,
                    )
                shared_seconds += _consume_mass_task_result(
                    result,
                    config=config,
                    states=states,
                )
                consumed_count += 1
            if consumed_count != config.sampling.n_halo_mass_samples:
                raise RuntimeError(
                    f"missing mass task results: expected "
                    f"{config.sampling.n_halo_mass_samples}, got {consumed_count}"
                )
            shared_seconds += hmf_seconds[0]

            mode_results: list[IMFModeResult] = []
            for mode in config.stellar_population.imf_modes:
                state = states[mode]
                histogram = state.histogram.finalize()
                observed_phi, observed_sigma = _observed_uvlf(
                    centers=np.array(histogram.centers, copy=True),
                    intrinsic_phi=np.array(
                        histogram.phi_per_mpc3_per_unit,
                        copy=True,
                    ),
                    intrinsic_sigma=np.array(
                        histogram.phi_sigma_per_mpc3_per_unit,
                        copy=True,
                    ),
                    redshift=redshift,
                    apply_dust=config.sampling.apply_dust,
                )
                mode_results.append(
                    IMFModeResult(
                        imf_mode=mode,
                        bin_edges_muv=histogram.edges,
                        bin_centers_muv=histogram.centers,
                        bin_width_mag=histogram.width,
                        raw_counts=histogram.raw_counts,
                        weighted_counts_per_mpc3=(
                            histogram.weighted_counts_per_mpc3
                        ),
                        weight_squared_counts_per_mpc6=(
                            histogram.weight_squared_counts_per_mpc6
                        ),
                        weighted_count_sigma_per_mpc3=(
                            histogram.weighted_count_sigma_per_mpc3
                        ),
                        effective_counts=histogram.effective_counts,
                        phi_intrinsic_per_mpc3_per_mag=(
                            histogram.phi_per_mpc3_per_unit
                        ),
                        phi_intrinsic_sigma_per_mpc3_per_mag=(
                            histogram.phi_sigma_per_mpc3_per_unit
                        ),
                        phi_observed_per_mpc3_per_mag=observed_phi,
                        phi_observed_sigma_per_mpc3_per_mag=observed_sigma,
                        halo_tracks=(),
                    )
                )
                topheavy_fraction = (
                    state.topheavy_source_count / state.starforming_source_count
                    if state.starforming_source_count > 0
                    else 0.0
                )
                popiii_fraction = (
                    state.popiii_source_count / state.active_source_count
                    if state.active_source_count > 0
                    else 0.0
                )
                all_diagnostics.append(
                    ModeRunDiagnostics(
                        redshift=redshift,
                        imf_mode=mode,
                        sampling_seconds=(
                            state.evaluation_seconds + shared_seconds / mode_count
                        ),
                        sample_count=state.sample_count,
                        valid_sample_count=state.valid_sample_count,
                        topheavy_source_fraction=topheavy_fraction,
                        popiii_source_fraction=popiii_fraction,
                        sfrd_msun_per_yr_per_mpc3=(
                            state.sfrd_msun_per_yr_per_mpc3
                        ),
                        popiii_sfrd_msun_per_yr_per_mpc3=(
                            state.popiii_sfrd_msun_per_yr_per_mpc3
                        ),
                    )
                )
            redshift_results.append(
                RedshiftResult(
                    redshift=redshift,
                    imf_modes=tuple(mode_results),
                )
            )
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        raise
    finally:
        if worker_count == 1:
            _clear_worker_context_for_tests()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
    return UVLFRunResult(
        config=config,
        redshifts=tuple(redshift_results),
        diagnostics=RunDiagnostics(
            total_seconds=time.perf_counter() - started,
            mode_runs=tuple(all_diagnostics),
        ),
    )


__all__ = ["run_uvlf_streaming"]
