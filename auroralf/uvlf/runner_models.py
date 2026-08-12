"""Immutable worker/task messages used by the UVLF scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np

from auroralf.results import _working_float_1d
from .imf import validate_imf_mode


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


def _normalize_mass_mode_metadata(
    *,
    imf_mode: str,
    topheavy_source_count: object,
    starforming_source_count: object,
    popiii_source_count: object,
    active_source_count: object,
    evaluation_seconds: object,
) -> tuple[str, dict[str, int], float]:
    mode = validate_imf_mode(imf_mode)
    counts = {
        name: _strict_int_scalar(name, value)
        for name, value in (
            ("topheavy_source_count", topheavy_source_count),
            ("starforming_source_count", starforming_source_count),
            ("popiii_source_count", popiii_source_count),
            ("active_source_count", active_source_count),
        )
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError("mode source counts must be non-negative")
    if counts["topheavy_source_count"] > counts["starforming_source_count"]:
        raise ValueError("topheavy_source_count must not exceed starforming_source_count")
    if counts["starforming_source_count"] > counts["active_source_count"]:
        raise ValueError("starforming_source_count must not exceed active_source_count")
    if counts["popiii_source_count"] > counts["active_source_count"]:
        raise ValueError("popiii_source_count must not exceed active_source_count")
    seconds = _strict_float_scalar("evaluation_seconds", evaluation_seconds)
    if seconds < 0.0:
        raise ValueError("evaluation_seconds must be non-negative")
    return mode, counts, seconds


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
        mode, counts, seconds = _normalize_mass_mode_metadata(
            imf_mode=self.imf_mode,
            topheavy_source_count=self.topheavy_source_count,
            starforming_source_count=self.starforming_source_count,
            popiii_source_count=self.popiii_source_count,
            active_source_count=self.active_source_count,
            evaluation_seconds=self.evaluation_seconds,
        )
        luminosity = _immutable_float_vector(
            "uv_luminosity_erg_per_s_hz",
            self.uv_luminosity_erg_per_s_hz,
        )
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


def _validate_mass_mode_task_result_integrity(
    result: _MassModeTaskResult,
) -> None:
    if type(result) is not _MassModeTaskResult:
        raise TypeError("mode_results entries must be exactly _MassModeTaskResult")
    _normalize_mass_mode_metadata(
        imf_mode=result.imf_mode,
        topheavy_source_count=result.topheavy_source_count,
        starforming_source_count=result.starforming_source_count,
        popiii_source_count=result.popiii_source_count,
        active_source_count=result.active_source_count,
        evaluation_seconds=result.evaluation_seconds,
    )
    luminosity = result.uv_luminosity_erg_per_s_hz
    if type(luminosity) is not np.ndarray or luminosity.dtype != np.dtype(float):
        raise TypeError("uv_luminosity_erg_per_s_hz must be a normalized float array")
    if luminosity.ndim != 1 or luminosity.size == 0:
        raise ValueError("uv_luminosity_erg_per_s_hz must be a non-empty 1D array")
    if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0.0):
        raise ValueError("uv_luminosity_erg_per_s_hz must be finite and non-negative")
    current: object = luminosity
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            raise ValueError("uv_luminosity_erg_per_s_hz must be immutable")
        current = current.base
    if isinstance(current, memoryview) and not current.readonly:
        raise ValueError("uv_luminosity_erg_per_s_hz must be immutable")

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
        for result in self.mode_results:
            _validate_mass_mode_task_result_integrity(result)
        modes = tuple(result.imf_mode for result in self.mode_results)
        if len(set(modes)) != len(modes):
            raise ValueError("mode_results must contain unique IMF modes")
        sample_count = self.mode_results[0].uv_luminosity_erg_per_s_hz.size
        if any(
            result.uv_luminosity_erg_per_s_hz.size != sample_count
            for result in self.mode_results
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
