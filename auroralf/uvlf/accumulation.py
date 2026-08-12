"""UVLF histogram accumulation and optional halo-sample observation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.results import _require_nonnegative, _working_float_1d
from auroralf.samples import HaloSampleTable

from .dust import compute_dust_attenuated_uvlf
from .hmf_sampling import uv_luminosity_to_muv
from .streaming import WeightedHistogramAccumulator


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
    dust_transform: Callable[..., dict[str, np.ndarray]] = compute_dust_attenuated_uvlf,
) -> tuple[np.ndarray, np.ndarray]:
    if not apply_dust:
        return intrinsic_phi.copy(), intrinsic_sigma.copy()
    dust = dust_transform(
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


def _consume_mass_task_result(
    result: Any,
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
    track_weight = result.mass_weight_per_mpc3 / config.sampling.n_tracks_per_halo_mass
    for mode_result in result.mode_results:
        if mode_result.uv_luminosity_erg_per_s_hz.size != config.sampling.n_tracks_per_halo_mass:
            raise RuntimeError(
                "mass task luminosity count does not match n_tracks_per_halo_mass"
            )
        state = states[mode_result.imf_mode]
        muv = np.asarray(
            uv_luminosity_to_muv(mode_result.uv_luminosity_erg_per_s_hz),
            dtype=float,
        )
        state.histogram.update(muv, np.full(muv.shape, track_weight))
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
            result.final_popiii_sfr_mean_msun_per_yr * result.mass_weight_per_mpc3
        )
        state.evaluation_seconds += mode_result.evaluation_seconds
    return result.shared_preparation_seconds


def _observe_halo_samples(
    result: Any,
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


__all__ = []
