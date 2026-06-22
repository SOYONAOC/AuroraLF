from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.chemistry import (
    MZRBirthMetallicityParameters,
    MZRBirthMetallicityResult,
    RegulatorMetallicityParameters,
    RegulatorMetallicityResult,
    compute_mzr_birth_metallicity,
    compute_regulator_metallicity,
)
from auroralf.cooling import compute_popiii_lw_minimum_mass_msun
from auroralf.mah import (
    MAH_BACKEND_MCBRIDE,
    MAH_BACKEND_THESAN,
    MAH_BACKEND_TNG,
    THESAN_TIME_GRID_SNAPSHOT,
    THESAN_TIME_GRID_UNIFORM_IN_T,
    TNG_TIME_GRID_SNAPSHOT,
    TNG_TIME_GRID_UNIFORM_IN_T,
    Cosmology,
    HaloHistoryResult,
    generate_halo_histories,
    generate_thesan_halo_histories,
    generate_tng_halo_histories,
    validate_mah_backend,
)
from auroralf.sfr import (
    DEFAULT_POPIII_SFR_PARAMETERS,
    DEFAULT_SFR_MODEL_PARAMETERS,
    EXTENDED_BURST_LOOKBACK_MAX_MYR,
    PopIIISFRParameters,
    SFRModelParameters,
    compute_popiii_sfr_from_grids,
    compute_sfr_from_tracks,
)
from auroralf.ssp import (
    DEFAULT_POPIII_UV_SSP_FILE,
    SSP_UV_LOOKBACK_MAX_MYR,
    compute_halo_uv_luminosity,
    interpolate_ssp_luminosity,
    load_popiii_uv_luminosity_table,
    load_uv1600_table,
)
from .imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY,
    IMF_MODE_CANONICAL,
    IMFTransitionParameters,
    compute_topheavy_source_flags,
    requires_topheavy_ssp,
    resolve_ssp_path,
    validate_imf_mode,
)


DEFAULT_SSP_FILE = DEFAULT_CANONICAL_SSP_FILE
DEFAULT_TOPHEAVY_SSP_FILE = DEFAULT_MILD_TOPHEAVY_SSP_FILE
DEFAULT_TOPHEAVY_SSP_METALLICITY = DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY
DEFAULT_POPIII_SSP_FILE = DEFAULT_POPIII_UV_SSP_FILE
YEARS_PER_GYR = 1.0e9
DEFAULT_BURST_SCATTER_TIMESCALE_MYR = 20.0


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


def _lognormal_unit_mean_shift_dex(scatter_dex: float) -> float:
    return -0.5 * np.log(10.0) * float(scatter_dex) ** 2


def _draw_burst_multiplier_for_segments(
    *,
    rng: np.random.Generator,
    segment_ids: np.ndarray,
    scatter_dex: float,
    preserve_mean: bool,
) -> np.ndarray:
    unique_segments, inverse = np.unique(segment_ids, return_inverse=True)
    loc = _lognormal_unit_mean_shift_dex(scatter_dex) if preserve_mean else 0.0
    delta_dex = rng.normal(loc=loc, scale=float(scatter_dex), size=unique_segments.size)
    return np.power(10.0, delta_dex[inverse])


def _apply_burst_scatter_to_sfr_grid(
    *,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    t_grid: np.ndarray,
    scatter_dex: float,
    correlation_timescale_myr: float,
    random_seed: int | None,
    preserve_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scatter_dex = float(scatter_dex)
    correlation_timescale_myr = float(correlation_timescale_myr)
    if scatter_dex < 0.0:
        raise ValueError("burst_scatter_dex must be non-negative")
    if correlation_timescale_myr <= 0.0:
        raise ValueError("burst_scatter_timescale_myr must be positive")

    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    time = np.asarray(t_grid, dtype=float)
    if sfr.shape != active.shape or sfr.shape != time.shape:
        raise ValueError("sfr_grid, active_grid, and t_grid must have the same shape")

    multiplier = np.ones_like(sfr, dtype=float)
    if scatter_dex == 0.0:
        return sfr.copy(), multiplier

    rng = np.random.default_rng(random_seed)
    correlation_gyr = correlation_timescale_myr / 1.0e3
    burst_sfr = sfr.copy()
    source_grid = active & np.isfinite(sfr) & (sfr > 0.0) & np.isfinite(time)
    for halo_index in range(sfr.shape[0]):
        source = source_grid[halo_index]
        if not np.any(source):
            continue
        first_time = float(time[halo_index, np.flatnonzero(source)[0]])
        segment_ids = np.floor((time[halo_index, source] - first_time) / correlation_gyr).astype(np.int64)
        row_multiplier = _draw_burst_multiplier_for_segments(
            rng=rng,
            segment_ids=segment_ids,
            scatter_dex=scatter_dex,
            preserve_mean=preserve_mean,
        )
        if preserve_mean and row_multiplier.size >= 2:
            source_time = time[halo_index, source]
            if np.any(np.diff(source_time) <= 0.0):
                raise ValueError("t_grid values for active SFR bins must be strictly increasing")
            original_mass = float(np.trapezoid(sfr[halo_index, source], source_time))
            burst_mass = float(np.trapezoid(sfr[halo_index, source] * row_multiplier, source_time))
            if burst_mass <= 0.0:
                raise RuntimeError("burst SFR normalization integral must be positive")
            row_multiplier = row_multiplier * (original_mass / burst_mass)
        multiplier[halo_index, source] = row_multiplier
        burst_sfr[halo_index, source] = sfr[halo_index, source] * row_multiplier

    return burst_sfr, multiplier


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
    enable_popiii: bool = False,
    popiii_sfr_parameters: PopIIISFRParameters = DEFAULT_POPIII_SFR_PARAMETERS,
    popiii_ssp_file: str | Path = DEFAULT_POPIII_SSP_FILE,
    imf_mode: str = "canonical",
    imf_transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
    cosmology: Cosmology | None = None,
    random_seed: int | None = 42,
    sampler: str = "mcbride",
    mah_backend: str = MAH_BACKEND_MCBRIDE,
    tng_mah_cache_path: str | Path | None = None,
    tng_mass_bin_width_dex: float = 0.15,
    tng_min_candidates: int = 5,
    tng_smoothing_myr: float = 0.0,
    tng_time_grid_mode: str = TNG_TIME_GRID_SNAPSHOT,
    thesan_mah_cache_path: str | Path | None = None,
    thesan_mass_bin_width_dex: float = 0.15,
    thesan_min_candidates: int = 5,
    thesan_smoothing_myr: float = 0.0,
    thesan_time_grid_mode: str = THESAN_TIME_GRID_SNAPSHOT,
    enable_time_delay: bool = False,
    workers: int | None = None,
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None = None,
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None = None,
    metallicity_random_seed: int | None = None,
    burst_scatter_dex: float = 0.0,
    burst_scatter_timescale_myr: float = DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    burst_scatter_random_seed: int | None = None,
    burst_scatter_preserve_mean: bool = True,
) -> HaloUVPipelineResult:
    """Run the main mah -> sfr -> UV pipeline and return per-halo UV luminosities."""

    imf_mode = validate_imf_mode(imf_mode)
    mah_backend = validate_mah_backend(mah_backend)
    cosmology = Cosmology() if cosmology is None else cosmology
    workers = default_worker_count() if workers is None else int(workers)
    if float(burst_scatter_dex) < 0.0:
        raise ValueError("burst_scatter_dex must be non-negative")
    if float(burst_scatter_timescale_myr) <= 0.0:
        raise ValueError("burst_scatter_timescale_myr must be positive")
    metallicity_topheavy_max_zsun = imf_transition_parameters.metallicity_topheavy_max_zsun
    if metallicity_topheavy_max_zsun is not None and float(metallicity_topheavy_max_zsun) <= 0.0:
        raise ValueError("metallicity_topheavy_max_zsun must be positive when provided")
    source_count = sum(
        source is not None
        for source in (
            mzr_metallicity_parameters,
            regulator_metallicity_parameters,
        )
    )
    if source_count > 1:
        raise ValueError(
            "provide only one birth metallicity source: "
            "mzr_metallicity_parameters or regulator_metallicity_parameters"
        )
    birth_metallicity_source_enabled = source_count == 1
    if imf_mode != IMF_MODE_CANONICAL and metallicity_topheavy_max_zsun is not None and not birth_metallicity_source_enabled:
        raise ValueError(
            "a birth metallicity source must be provided when metallicity_topheavy_max_zsun is set"
        )
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    astro = _build_astropy_cosmology(cosmology)
    t_start_gyr = float(astro.age(z_start_max).value)
    t_end_gyr = float(astro.age(z_final).value)
    dt_gyr = (t_end_gyr - t_start_gyr) / float(int(n_grid) - 1)

    t0 = time.perf_counter()
    if mah_backend == MAH_BACKEND_MCBRIDE:
        mah_mass_floor = None
        if enable_popiii:
            mah_mass_floor = lambda redshift: compute_popiii_lw_minimum_mass_msun(
                redshift,
                lw_background_j21=float(popiii_sfr_parameters.lw_background_j21),
            )
        histories = generate_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            z_start_max=z_start_max,
            M_min=mah_mass_floor,
            cosmology=cosmology,
            random_seed=random_seed,
            time_grid_mode="uniform_in_t",
            dt=dt_gyr,
            store_inactive_history=True,
            sampler=sampler,
        )
    elif mah_backend == MAH_BACKEND_TNG:
        if tng_mah_cache_path is None:
            raise ValueError("tng_mah_cache_path is required when mah_backend='tng'")
        histories = generate_tng_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            cache_path=tng_mah_cache_path,
            z_start_max=z_start_max,
            mass_bin_width_dex=tng_mass_bin_width_dex,
            min_candidates=tng_min_candidates,
            smoothing_myr=tng_smoothing_myr,
            random_seed=random_seed,
            time_grid_mode=tng_time_grid_mode,
            target_n_grid=int(n_grid)
            if str(tng_time_grid_mode).strip().lower() == TNG_TIME_GRID_UNIFORM_IN_T
            else None,
        )
    elif mah_backend == MAH_BACKEND_THESAN:
        if thesan_mah_cache_path is None:
            raise ValueError("thesan_mah_cache_path is required when mah_backend='thesan'")
        histories = generate_thesan_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            cache_path=thesan_mah_cache_path,
            z_start_max=z_start_max,
            mass_bin_width_dex=thesan_mass_bin_width_dex,
            min_candidates=thesan_min_candidates,
            smoothing_myr=thesan_smoothing_myr,
            random_seed=random_seed,
            time_grid_mode=thesan_time_grid_mode,
            target_n_grid=int(n_grid)
            if str(thesan_time_grid_mode).strip().lower() == THESAN_TIME_GRID_UNIFORM_IN_T
            else None,
        )
    else:  # pragma: no cover - guarded by validate_mah_backend
        raise RuntimeError(f"unsupported mah_backend after validation: {mah_backend}")
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
    popiii_ssp_path = resolve_ssp_path(popiii_ssp_file)
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
    if enable_popiii:
        popiii_ages_myr, popiii_luv_per_msun = load_popiii_uv_luminosity_table(popiii_ssp_path)
        popiii_ssp_age_grid_gyr = popiii_ages_myr / 1.0e3
    else:
        popiii_luv_per_msun = None
        popiii_ssp_age_grid_gyr = None

    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    n_halos = np.unique(halo_ids).size
    steps_per_halo = redshift_grid.size
    t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, steps_per_halo)
    mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, steps_per_halo)
    dmhdt_grid = np.asarray(sfr_tracks["dMh_dt"], dtype=float).reshape(n_halos, steps_per_halo)
    sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, steps_per_halo)
    active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, steps_per_halo)
    z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(n_halos, steps_per_halo)
    burst_scatter_seed_used = (
        burst_scatter_random_seed
        if burst_scatter_random_seed is not None
        else random_seed
    )
    sfr_grid, burst_sfr_multiplier_grid = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=float(burst_scatter_dex),
        correlation_timescale_myr=float(burst_scatter_timescale_myr),
        random_seed=None if burst_scatter_seed_used is None else int(burst_scatter_seed_used),
        preserve_mean=bool(burst_scatter_preserve_mean),
    )
    sfr_tracks["SFR"] = sfr_grid.reshape(-1)
    if enable_popiii:
        popiii_sfr_result = compute_popiii_sfr_from_grids(
            mh_grid=mh_grid,
            dmhdt_grid=dmhdt_grid,
            z_grid=z_grid,
            active_grid=active_grid,
            baryon_fraction=cosmology.omega_b / cosmology.omega_m,
            parameters=popiii_sfr_parameters,
        )
        sfr_popiii_grid = np.asarray(popiii_sfr_result.sfr_grid, dtype=float)
        popiii_source_grid = np.asarray(popiii_sfr_result.starforming_grid, dtype=bool)
        popiii_fstar_grid = np.asarray(popiii_sfr_result.fstar_grid, dtype=float)
        popiii_duty_cycle_grid = np.asarray(popiii_sfr_result.duty_cycle_grid, dtype=float)
        popiii_lower_mass_grid = np.asarray(popiii_sfr_result.lower_mass_msun_grid, dtype=float)
        popiii_upper_mass_grid = np.asarray(popiii_sfr_result.upper_mass_msun_grid, dtype=float)
    else:
        sfr_popiii_grid = np.zeros_like(sfr_grid, dtype=float)
        popiii_source_grid = np.zeros_like(active_grid, dtype=bool)
        popiii_fstar_grid = np.zeros_like(sfr_grid, dtype=float)
        popiii_duty_cycle_grid = np.zeros_like(sfr_grid, dtype=float)
        popiii_lower_mass_grid = np.full_like(sfr_grid, np.nan, dtype=float)
        popiii_upper_mass_grid = np.full_like(sfr_grid, np.nan, dtype=float)
    sfr_tracks["SFR_popiii"] = sfr_popiii_grid.reshape(-1)
    sfr_tracks["fstar_popiii"] = popiii_fstar_grid.reshape(-1)
    sfr_tracks["popiii_duty_cycle"] = popiii_duty_cycle_grid.reshape(-1)
    sfr_tracks["popiii_lower_mass_msun"] = popiii_lower_mass_grid.reshape(-1)
    sfr_tracks["popiii_upper_mass_msun"] = popiii_upper_mass_grid.reshape(-1)
    starforming_grid = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
    candidate_transition_parameters = replace(imf_transition_parameters, metallicity_topheavy_max_zsun=None)
    candidate_topheavy_source_grid = compute_topheavy_source_flags(
        imf_mode=imf_mode,
        z_grid=z_grid,
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        active_grid=starforming_grid,
        transition_parameters=candidate_transition_parameters,
    )
    topheavy_source_grid = candidate_topheavy_source_grid

    mzr_metallicity_result: MZRBirthMetallicityResult | None = None
    regulator_metallicity_result: RegulatorMetallicityResult | None = None
    birth_metallicity_zsun_grid: np.ndarray | None = None
    gas_metallicity_zsun_grid: np.ndarray | None = None
    metal_mass_grid: np.ndarray | None = None
    gas_mass_grid: np.ndarray | None = None
    metallicity_source = "none"
    if mzr_metallicity_parameters is not None:
        mzr_metallicity_result = compute_mzr_birth_metallicity(
            t_grid_gyr=t_grid,
            z_grid=z_grid,
            sfr_grid=sfr_grid,
            active_grid=starforming_grid,
            parameters=mzr_metallicity_parameters,
            random_seed=metallicity_random_seed,
        )
        birth_metallicity_zsun_grid = np.asarray(mzr_metallicity_result.birth_metallicity_zsun_grid, dtype=float)
        topheavy_source_grid = compute_topheavy_source_flags(
            imf_mode=imf_mode,
            z_grid=z_grid,
            mh_grid=mh_grid,
            dmhdt_grid=dmhdt_grid,
            active_grid=starforming_grid,
            birth_metallicity_zsun_grid=birth_metallicity_zsun_grid,
            transition_parameters=imf_transition_parameters,
        )
        metallicity_source = "mzr"
    elif regulator_metallicity_parameters is not None:
        regulator_metallicity_result = compute_regulator_metallicity(
            t_grid_gyr=t_grid,
            z_grid=z_grid,
            mh_grid=mh_grid,
            sfr_grid=sfr_grid,
            active_grid=starforming_grid,
            baryon_fraction=cosmology.omega_b / cosmology.omega_m,
            parameters=regulator_metallicity_parameters,
            random_seed=metallicity_random_seed,
        )
        birth_metallicity_zsun_grid = np.asarray(regulator_metallicity_result.birth_metallicity_zsun_grid, dtype=float)
        gas_metallicity_zsun_grid = np.asarray(regulator_metallicity_result.gas_metallicity_zsun_grid, dtype=float)
        metal_mass_grid = np.asarray(regulator_metallicity_result.metal_mass_grid, dtype=float)
        gas_mass_grid = np.asarray(regulator_metallicity_result.gas_mass_grid, dtype=float)
        topheavy_source_grid = compute_topheavy_source_flags(
            imf_mode=imf_mode,
            z_grid=z_grid,
            mh_grid=mh_grid,
            dmhdt_grid=dmhdt_grid,
            active_grid=starforming_grid,
            birth_metallicity_zsun_grid=birth_metallicity_zsun_grid,
            transition_parameters=imf_transition_parameters,
        )
        metallicity_source = "regulator"

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
    if enable_popiii:
        if popiii_ssp_age_grid_gyr is None or popiii_luv_per_msun is None:
            raise RuntimeError("Pop III SSP grid was not loaded despite enable_popiii=True")
        uv_luminosities_popiii, _ = _compute_final_uv_luminosity_components_vectorized(
            t_grid=t_grid,
            sfr_grid=sfr_popiii_grid,
            active_grid=popiii_source_grid,
            topheavy_source_flag_grid=np.zeros_like(popiii_source_grid, dtype=bool),
            ssp_age_grid=popiii_ssp_age_grid_gyr,
            ssp_luv_grid=popiii_luv_per_msun,
            topheavy_ssp_age_grid=None,
            topheavy_ssp_luv_grid=None,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
    else:
        uv_luminosities_popiii = np.zeros_like(uv_luminosities_canonical, dtype=float)
    uv_luminosities = uv_luminosities_canonical + uv_luminosities_topheavy + uv_luminosities_popiii
    uv_convolution_method = "vectorized_final_time_variable_imf_with_optional_popiii"
    t3 = time.perf_counter()
    total_light = np.asarray(uv_luminosities, dtype=float)
    positive_light = total_light > 0.0
    topheavy_light_fraction = np.zeros_like(total_light, dtype=float)
    topheavy_light_fraction[positive_light] = uv_luminosities_topheavy[positive_light] / total_light[positive_light]
    popiii_light_fraction = np.zeros_like(total_light, dtype=float)
    popiii_light_fraction[positive_light] = uv_luminosities_popiii[positive_light] / total_light[positive_light]

    metadata = {
        "n_tracks": n_halos,
        "steps_per_halo": steps_per_halo,
        "workers": max(1, workers),
        "ssp_file": str(canonical_ssp_path),
        "canonical_ssp_file": str(canonical_ssp_path),
        "topheavy_ssp_file": str(topheavy_ssp_path),
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "popiii_enabled": bool(enable_popiii),
        "popiii_ssp_file": str(popiii_ssp_path),
        "popiii_sfr_parameters": popiii_sfr_parameters.as_metadata(),
        "imf_mode": imf_mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "source_redshift_gate_enabled": bool(imf_transition_parameters.source_redshift_gate_enabled),
            "growth_time_threshold_myr": float(imf_transition_parameters.growth_time_threshold_myr),
            "metallicity_topheavy_max_zsun": None
            if imf_transition_parameters.metallicity_topheavy_max_zsun is None
            else float(imf_transition_parameters.metallicity_topheavy_max_zsun),
        },
        "metallicity_topheavy_gate_applied": imf_mode != IMF_MODE_CANONICAL and metallicity_topheavy_max_zsun is not None,
        "topheavy_candidate_source_fraction": float(np.mean(candidate_topheavy_source_grid[starforming_grid]))
        if np.any(starforming_grid)
        else 0.0,
        "topheavy_candidate_source_count": int(np.count_nonzero(candidate_topheavy_source_grid & starforming_grid)),
        "topheavy_source_fraction": float(np.mean(topheavy_source_grid[starforming_grid]))
        if np.any(starforming_grid)
        else 0.0,
        "topheavy_source_count": int(np.count_nonzero(topheavy_source_grid & starforming_grid)),
        "starforming_source_count": int(np.count_nonzero(starforming_grid)),
        "topheavy_light_fraction_median": float(np.median(topheavy_light_fraction[positive_light]))
        if np.any(positive_light)
        else 0.0,
        "popiii_source_fraction": float(np.mean(popiii_source_grid[active_grid]))
        if np.any(active_grid)
        else 0.0,
        "popiii_source_count": int(np.count_nonzero(popiii_source_grid)),
        "active_source_count": int(np.count_nonzero(active_grid)),
        "popiii_light_fraction_median": float(np.median(popiii_light_fraction[positive_light]))
        if np.any(positive_light)
        else 0.0,
        "popiii_luminosity_median": float(np.median(uv_luminosities_popiii[np.isfinite(uv_luminosities_popiii)]))
        if np.any(np.isfinite(uv_luminosities_popiii))
        else 0.0,
        "metallicity_source": metallicity_source,
        "mah_backend": mah_backend,
        "sampler": sampler,
        "tng_mah_cache_path": histories.metadata.get("tng_mah_cache_path"),
        "tng_source_simulation": histories.metadata.get("source_simulation") if mah_backend == MAH_BACKEND_TNG else None,
        "tng_mass_bin_width_dex": None if mah_backend != MAH_BACKEND_TNG else float(tng_mass_bin_width_dex),
        "tng_min_candidates": None if mah_backend != MAH_BACKEND_TNG else int(tng_min_candidates),
        "tng_candidate_count": histories.metadata.get("candidate_count") if mah_backend == MAH_BACKEND_TNG else None,
        "tng_smoothing_myr": None if mah_backend != MAH_BACKEND_TNG else float(tng_smoothing_myr),
        "tng_time_grid_mode": None if mah_backend != MAH_BACKEND_TNG else histories.metadata.get("tng_time_grid_mode"),
        "tng_negative_dmhdt_clip_count": histories.metadata.get("negative_dmhdt_clip_count")
        if mah_backend == MAH_BACKEND_TNG
        else None,
        "tng_negative_dmhdt_clip_fraction": histories.metadata.get("negative_dmhdt_clip_fraction")
        if mah_backend == MAH_BACKEND_TNG
        else None,
        "thesan_mah_cache_path": histories.metadata.get("thesan_mah_cache_path"),
        "thesan_source_simulation": histories.metadata.get("source_simulation")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_source_tree": histories.metadata.get("source_tree") if mah_backend == MAH_BACKEND_THESAN else None,
        "thesan_mass_bin_width_dex": None
        if mah_backend != MAH_BACKEND_THESAN
        else float(thesan_mass_bin_width_dex),
        "thesan_min_candidates": None if mah_backend != MAH_BACKEND_THESAN else int(thesan_min_candidates),
        "thesan_candidate_count": histories.metadata.get("candidate_count")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_smoothing_myr": None if mah_backend != MAH_BACKEND_THESAN else float(thesan_smoothing_myr),
        "thesan_time_grid_mode": None
        if mah_backend != MAH_BACKEND_THESAN
        else histories.metadata.get("thesan_time_grid_mode"),
        "thesan_negative_dmhdt_clip_count": histories.metadata.get("negative_dmhdt_clip_count")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_negative_dmhdt_clip_fraction": histories.metadata.get("negative_dmhdt_clip_fraction")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "mzr_metallicity_enabled": mzr_metallicity_result is not None,
        "regulator_metallicity_enabled": regulator_metallicity_result is not None,
        "metallicity_random_seed": metallicity_random_seed,
        "mzr_metallicity_parameters": mzr_metallicity_parameters.as_metadata()
        if mzr_metallicity_parameters is not None
        else None,
        "regulator_metallicity_parameters": regulator_metallicity_parameters.as_metadata()
        if regulator_metallicity_parameters is not None
        else None,
        "final_gas_metallicity_zsun_median": float(
            np.nanmedian(gas_metallicity_zsun_grid[:, -1])
        )
        if gas_metallicity_zsun_grid is not None
        else None,
        "birth_metallicity_zsun_starforming_median": float(
            np.nanmedian(birth_metallicity_zsun_grid[starforming_grid])
        )
        if birth_metallicity_zsun_grid is not None and np.any(starforming_grid)
        else None,
        "enable_time_delay": enable_time_delay,
        "time_grid_mode": histories.metadata.get("time_grid_mode", "uniform_in_t"),
        "dt_gyr": float(histories.metadata.get("dt_gyr_median", dt_gyr)),
        "burst_lookback_max_myr": float(burst_lookback_max_myr),
        "burst_scatter_enabled": float(burst_scatter_dex) > 0.0,
        "burst_scatter_dex": float(burst_scatter_dex),
        "burst_scatter_timescale_myr": float(burst_scatter_timescale_myr),
        "burst_scatter_random_seed": None
        if burst_scatter_seed_used is None
        else int(burst_scatter_seed_used),
        "burst_scatter_preserve_mean": bool(burst_scatter_preserve_mean),
        "burst_scatter_mass_conserving": bool(burst_scatter_preserve_mean),
        "burst_sfr_multiplier_median": float(np.median(burst_sfr_multiplier_grid[starforming_grid]))
        if np.any(starforming_grid)
        else 1.0,
        "burst_sfr_multiplier_p16": float(np.percentile(burst_sfr_multiplier_grid[starforming_grid], 16.0))
        if np.any(starforming_grid)
        else 1.0,
        "burst_sfr_multiplier_p84": float(np.percentile(burst_sfr_multiplier_grid[starforming_grid], 84.0))
        if np.any(starforming_grid)
        else 1.0,
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
        uv_luminosities_popiii=np.asarray(uv_luminosities_popiii, dtype=float),
        redshift_grid=redshift_grid,
        floor_mass=floor_mass,
        active_grid=active_grid,
        imf_topheavy_source_grid=topheavy_source_grid,
        popiii_source_grid=popiii_source_grid,
        metadata=metadata,
        gas_metallicity_zsun_grid=gas_metallicity_zsun_grid,
        birth_metallicity_zsun_grid=None if birth_metallicity_zsun_grid is None else birth_metallicity_zsun_grid,
        metal_mass_grid=metal_mass_grid,
        gas_mass_grid=gas_mass_grid,
    )
