from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.mah import Cosmology, HaloHistoryResult, generate_halo_histories
from auroralf.sfr import (
    DEFAULT_SFR_MODEL_PARAMETERS,
    EXTENDED_BURST_LOOKBACK_MAX_MYR,
    SFRModelParameters,
    compute_sfr_from_tracks,
)
from auroralf.ssp import SSP_UV_LOOKBACK_MAX_MYR, compute_halo_uv_luminosity, interpolate_ssp_luminosity, load_uv1600_table
from .imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY,
    IMFTransitionParameters,
    compute_topheavy_source_flags,
    requires_topheavy_ssp,
    resolve_ssp_path,
    validate_imf_mode,
)


DEFAULT_SSP_FILE = DEFAULT_CANONICAL_SSP_FILE
DEFAULT_TOPHEAVY_SSP_FILE = DEFAULT_MILD_TOPHEAVY_SSP_FILE
DEFAULT_TOPHEAVY_SSP_METALLICITY = DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY
YEARS_PER_GYR = 1.0e9


@dataclass(frozen=True)
class HaloUVPipelineResult:
    histories: HaloHistoryResult
    sfr_tracks: dict[str, np.ndarray]
    uv_luminosities: np.ndarray
    uv_luminosities_canonical: np.ndarray
    uv_luminosities_topheavy: np.ndarray
    redshift_grid: np.ndarray
    floor_mass: np.ndarray
    active_grid: np.ndarray
    imf_topheavy_source_grid: np.ndarray
    metadata: dict[str, Any]


_UV_WORKER_STATE: dict[str, np.ndarray] = {}


def _build_astropy_cosmology(cosmology: Cosmology) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)


def _init_uv_worker(ssp_luv_grid: np.ndarray) -> None:
    _UV_WORKER_STATE["ssp_luv_grid"] = np.asarray(ssp_luv_grid, dtype=float)


def _compute_uv_chunk(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid, ssp_lookback_max_myr = args
    ssp_luv_grid = _UV_WORKER_STATE["ssp_luv_grid"]

    result = np.empty(mh_chunk.shape[0], dtype=float)
    for row_index in range(mh_chunk.shape[0]):
        active = np.asarray(active_chunk[row_index], dtype=bool)
        if not np.any(active):
            result[row_index] = 0.0
            continue

        t_used = np.asarray(t_grid[active], dtype=float)
        mh_used = np.asarray(mh_chunk[row_index][active], dtype=float)
        sfr_used = np.asarray(sfr_chunk[row_index][active], dtype=float)

        result[row_index] = compute_halo_uv_luminosity(
            t_obs=float(t_used[-1]),
            t_history=t_used,
            mh_history=mh_used,
            sfr_history=sfr_used,
            ssp_age_grid=ssp_age_grid,
            ssp_luv_grid=ssp_luv_grid,
            M_min=0.0,
            t_z50=float(t_used[0]),
            time_unit_in_years=YEARS_PER_GYR,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
    return result


def compute_uv_luminosities_parallel(
    t_grid: np.ndarray,
    mh_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    n_workers: int,
    ssp_lookback_max_myr: float,
) -> np.ndarray:
    if n_workers <= 1:
        _init_uv_worker(ssp_luv_grid)
        return _compute_uv_chunk((t_grid, mh_grid, sfr_grid, active_grid, ssp_age_grid, ssp_lookback_max_myr))

    chunk_count = min(n_workers, mh_grid.shape[0])
    mh_chunks = np.array_split(mh_grid, chunk_count, axis=0)
    sfr_chunks = np.array_split(sfr_grid, chunk_count, axis=0)
    active_chunks = np.array_split(active_grid, chunk_count, axis=0)
    tasks = [
        (t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid, ssp_lookback_max_myr)
        for mh_chunk, sfr_chunk, active_chunk in zip(mh_chunks, sfr_chunks, active_chunks, strict=True)
    ]

    outputs: list[np.ndarray] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_uv_worker,
        initargs=(np.asarray(ssp_luv_grid, dtype=float),),
    ) as executor:
        for chunk_output in executor.map(_compute_uv_chunk, tasks):
            outputs.append(np.asarray(chunk_output, dtype=float))
    return np.concatenate(outputs)


def default_worker_count() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def _resolve_regular_time_grid(t_grid: np.ndarray) -> np.ndarray | None:
    if t_grid.ndim != 2 or t_grid.shape[0] == 0:
        return None
    time_row = np.asarray(t_grid[0], dtype=float)
    if not np.all(np.isfinite(time_row)):
        return None
    if not np.allclose(t_grid, time_row[None, :], rtol=0.0, atol=0.0):
        return None
    return time_row


def _integrate_final_uv_components_single_halo_regular_grid(
    time_row: np.ndarray,
    sfr_row: np.ndarray,
    active_row: np.ndarray,
    topheavy_source_flag_row: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    topheavy_ssp_age_grid: np.ndarray | None,
    topheavy_ssp_luv_grid: np.ndarray | None,
    ssp_lookback_max_myr: float,
) -> tuple[float, float]:
    active = np.asarray(active_row, dtype=bool)
    if not np.any(active):
        return 0.0, 0.0

    t_obs = float(time_row[-1])
    max_lookback_gyr = float(ssp_lookback_max_myr) / 1.0e3
    first_active = int(np.argmax(active))
    lower = max(float(time_row[first_active]), t_obs - max_lookback_gyr)
    if lower >= t_obs:
        return 0.0, 0.0

    start = int(np.searchsorted(time_row, lower, side="left"))
    t_used = np.asarray(time_row[start:], dtype=float)
    sfr_used = np.asarray(sfr_row[start:], dtype=float)
    active_used = np.asarray(active[start:], dtype=bool)
    topheavy_used = np.asarray(topheavy_source_flag_row[start:], dtype=bool)

    if t_used.size == 0:
        return 0.0, 0.0

    if lower < float(t_used[0]):
        left = start - 1
        right = start
        t_left = float(time_row[left])
        t_right = float(time_row[right])
        sfr_left = float(sfr_row[left])
        sfr_right = float(sfr_row[right])
        weight = 0.0 if t_right <= t_left else (lower - t_left) / (t_right - t_left)
        sfr_lower = sfr_left + weight * (sfr_right - sfr_left)
        t_used = np.concatenate((np.array([lower], dtype=float), t_used))
        sfr_used = np.concatenate((np.array([sfr_lower], dtype=float), sfr_used))
        active_used = np.concatenate((np.array([True], dtype=bool), active_used))
        topheavy_lower = bool(topheavy_source_flag_row[left])
        topheavy_used = np.concatenate((np.array([topheavy_lower], dtype=bool), topheavy_used))

    if np.count_nonzero(active_used) < 2:
        return 0.0, 0.0

    age_used = np.maximum(t_obs - t_used, 0.0)
    canonical_kernel = np.asarray(
        interpolate_ssp_luminosity(age_used, ssp_age_grid=ssp_age_grid, ssp_luv_grid=ssp_luv_grid),
        dtype=float,
    )
    if topheavy_ssp_age_grid is None or topheavy_ssp_luv_grid is None:
        topheavy_used = np.zeros_like(active_used, dtype=bool)
        topheavy_kernel = canonical_kernel
    else:
        topheavy_kernel = np.asarray(
            interpolate_ssp_luminosity(
                age_used,
                ssp_age_grid=topheavy_ssp_age_grid,
                ssp_luv_grid=topheavy_ssp_luv_grid,
            ),
            dtype=float,
        )

    source_rate = np.where(active_used, sfr_used, 0.0)
    canonical_integrand = np.where(topheavy_used, 0.0, source_rate * canonical_kernel)
    topheavy_integrand = np.where(topheavy_used, source_rate * topheavy_kernel, 0.0)
    x_years = t_used * YEARS_PER_GYR
    canonical_luv = float(np.trapezoid(canonical_integrand, x=x_years))
    topheavy_luv = float(np.trapezoid(topheavy_integrand, x=x_years))
    return canonical_luv, topheavy_luv


def _compute_final_uv_luminosity_components_vectorized(
    t_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    topheavy_source_flag_grid: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    topheavy_ssp_age_grid: np.ndarray | None,
    topheavy_ssp_luv_grid: np.ndarray | None,
    ssp_lookback_max_myr: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_row = _resolve_regular_time_grid(t_grid)
    if time_row is None:
        raise ValueError("vectorized final UV convolution requires a shared regular time grid")
    canonical_result = np.empty(sfr_grid.shape[0], dtype=float)
    topheavy_result = np.empty(sfr_grid.shape[0], dtype=float)
    for halo_index in range(sfr_grid.shape[0]):
        canonical_luv, topheavy_luv = _integrate_final_uv_components_single_halo_regular_grid(
            time_row=time_row,
            sfr_row=np.asarray(sfr_grid[halo_index], dtype=float),
            active_row=np.asarray(active_grid[halo_index], dtype=bool),
            topheavy_source_flag_row=np.asarray(topheavy_source_flag_grid[halo_index], dtype=bool),
            ssp_age_grid=ssp_age_grid,
            ssp_luv_grid=ssp_luv_grid,
            topheavy_ssp_age_grid=topheavy_ssp_age_grid,
            topheavy_ssp_luv_grid=topheavy_ssp_luv_grid,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
        canonical_result[halo_index] = canonical_luv
        topheavy_result[halo_index] = topheavy_luv
    return canonical_result, topheavy_result


def run_halo_uv_pipeline(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    ssp_file: str | Path = DEFAULT_SSP_FILE,
    topheavy_ssp_file: str | Path = DEFAULT_TOPHEAVY_SSP_FILE,
    topheavy_ssp_metallicity: float | None = DEFAULT_TOPHEAVY_SSP_METALLICITY,
    imf_mode: str = "canonical",
    imf_transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
    cosmology: Cosmology | None = None,
    random_seed: int | None = 42,
    sampler: str = "mcbride",
    enable_time_delay: bool = False,
    workers: int | None = None,
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
) -> HaloUVPipelineResult:
    """Run the main mah -> sfr -> UV pipeline and return per-halo UV luminosities."""

    imf_mode = validate_imf_mode(imf_mode)
    cosmology = Cosmology() if cosmology is None else cosmology
    workers = default_worker_count() if workers is None else int(workers)
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    astro = _build_astropy_cosmology(cosmology)
    t_start_gyr = float(astro.age(z_start_max).value)
    t_end_gyr = float(astro.age(z_final).value)
    dt_gyr = (t_end_gyr - t_start_gyr) / float(int(n_grid) - 1)

    t0 = time.perf_counter()
    histories = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=Mh_final,
        z_start_max=z_start_max,
        cosmology=cosmology,
        random_seed=random_seed,
        time_grid_mode="uniform_in_t",
        dt=dt_gyr,
        store_inactive_history=True,
        sampler=sampler,
    )
    t1 = time.perf_counter()
    redshift_grid = np.unique(np.asarray(histories.tracks["z"], dtype=float))[::-1]

    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        enable_time_delay=enable_time_delay,
        burst_lookback_max_myr=burst_lookback_max_myr,
        model_parameters=sfr_model_parameters,
    )
    t2 = time.perf_counter()

    canonical_ssp_path = resolve_ssp_path(ssp_file)
    topheavy_ssp_path = resolve_ssp_path(topheavy_ssp_file)
    ages_myr, luv_per_msun = load_uv1600_table(canonical_ssp_path)
    ssp_age_grid_gyr = ages_myr / 1.0e3
    if requires_topheavy_ssp(imf_mode):
        topheavy_ages_myr, topheavy_luv_per_msun = load_uv1600_table(
            topheavy_ssp_path,
            metallicity=topheavy_ssp_metallicity,
        )
        topheavy_ssp_age_grid_gyr = topheavy_ages_myr / 1.0e3
    else:
        topheavy_luv_per_msun = None
        topheavy_ssp_age_grid_gyr = None

    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    n_halos = np.unique(halo_ids).size
    steps_per_halo = redshift_grid.size
    t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, steps_per_halo)
    mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, steps_per_halo)
    dmhdt_grid = np.asarray(sfr_tracks["dMh_dt"], dtype=float).reshape(n_halos, steps_per_halo)
    sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, steps_per_halo)
    active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, steps_per_halo)
    z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(n_halos, steps_per_halo)
    starforming_grid = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
    topheavy_source_grid = compute_topheavy_source_flags(
        imf_mode=imf_mode,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        active_grid=starforming_grid,
        transition_parameters=imf_transition_parameters,
    )

    floor_mass = np.zeros_like(redshift_grid, dtype=float)
    active_flat = active_grid.reshape(-1)
    if np.any(active_flat):
        active_mh = np.asarray(sfr_tracks["Mh"], dtype=float)[active_flat]
        active_z = np.asarray(sfr_tracks["z"], dtype=float)[active_flat]
        for index, z_value in enumerate(redshift_grid):
            mask = np.isclose(active_z, z_value)
            if np.any(mask):
                floor_mass[index] = float(np.min(active_mh[mask]))
    positive_floor = floor_mass[floor_mass > 0.0]
    if positive_floor.size == 0:
        raise RuntimeError("could not infer an effective M_min(z) floor from active histories")

    time_row = _resolve_regular_time_grid(t_grid)
    if time_row is None:
        raise ValueError("run_halo_uv_pipeline requires histories on a shared regular time grid")
    uv_luminosities_canonical, uv_luminosities_topheavy = _compute_final_uv_luminosity_components_vectorized(
        t_grid=t_grid,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        topheavy_source_flag_grid=topheavy_source_grid,
        ssp_age_grid=ssp_age_grid_gyr,
        ssp_luv_grid=luv_per_msun,
        topheavy_ssp_age_grid=topheavy_ssp_age_grid_gyr,
        topheavy_ssp_luv_grid=topheavy_luv_per_msun,
        ssp_lookback_max_myr=ssp_lookback_max_myr,
    )
    uv_luminosities = uv_luminosities_canonical + uv_luminosities_topheavy
    uv_convolution_method = "vectorized_final_time_variable_imf"
    t3 = time.perf_counter()
    total_light = np.asarray(uv_luminosities, dtype=float)
    positive_light = total_light > 0.0
    topheavy_light_fraction = np.zeros_like(total_light, dtype=float)
    topheavy_light_fraction[positive_light] = uv_luminosities_topheavy[positive_light] / total_light[positive_light]

    metadata = {
        "n_tracks": n_halos,
        "steps_per_halo": steps_per_halo,
        "workers": max(1, workers),
        "ssp_file": str(canonical_ssp_path),
        "canonical_ssp_file": str(canonical_ssp_path),
        "topheavy_ssp_file": str(topheavy_ssp_path),
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "imf_mode": imf_mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "growth_time_threshold_myr": float(imf_transition_parameters.growth_time_threshold_myr),
        },
        "topheavy_source_fraction": float(np.mean(topheavy_source_grid[starforming_grid]))
        if np.any(starforming_grid)
        else 0.0,
        "topheavy_source_count": int(np.count_nonzero(topheavy_source_grid & starforming_grid)),
        "starforming_source_count": int(np.count_nonzero(starforming_grid)),
        "topheavy_light_fraction_median": float(np.median(topheavy_light_fraction[positive_light]))
        if np.any(positive_light)
        else 0.0,
        "enable_time_delay": enable_time_delay,
        "time_grid_mode": "uniform_in_t",
        "dt_gyr": float(dt_gyr),
        "burst_lookback_max_myr": float(burst_lookback_max_myr),
        "ssp_lookback_max_myr": float(ssp_lookback_max_myr),
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "timing_seconds": {
            "mah_generation": t1 - t0,
            "sfr": t2 - t1,
            "uv_convolution": t3 - t2,
            "total_without_plotting": t3 - t0,
        },
        "uv_convolution_method": uv_convolution_method,
    }

    return HaloUVPipelineResult(
        histories=histories,
        sfr_tracks=sfr_tracks,
        uv_luminosities=np.asarray(uv_luminosities, dtype=float),
        uv_luminosities_canonical=np.asarray(uv_luminosities_canonical, dtype=float),
        uv_luminosities_topheavy=np.asarray(uv_luminosities_topheavy, dtype=float),
        redshift_grid=redshift_grid,
        floor_mass=floor_mass,
        active_grid=active_grid,
        imf_topheavy_source_grid=topheavy_source_grid,
        metadata=metadata,
    )
