from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hmf import MassFunction

from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from .imf import DEFAULT_IMF_TRANSITION_PARAMETERS, IMFTransitionParameters, validate_imf_mode
from .pipeline import (
    DEFAULT_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_METALLICITY,
    default_worker_count,
    run_halo_uv_pipeline,
)


LOGM_MIN = 9.0
LOGM_MAX = 13.0
AB_ZEROPOINT_LNU = 51.60
MASS_FUNCTION_MODEL_HMF_REED07 = "hmf_reed07"
MASS_FUNCTION_MODELS = (MASS_FUNCTION_MODEL_HMF_REED07,)
DEFAULT_MASS_FUNCTION_MODEL = MASS_FUNCTION_MODEL_HMF_REED07
DEFAULT_HMF_DLOG10M = 0.005
MASS_FUNCTION_NS = 0.965
MASS_FUNCTION_SIGMA8 = 0.811
MASS_FUNCTION_H = 0.674
MASS_FUNCTION_OMEGA_M = 0.315
MASS_FUNCTION_OMEGA_B_H2 = 0.0224
HMF_REED07_FITTING_FUNCTION = "Reed07"
DEPRECATED_MASS_FUNCTION_MODELS = {"massfunc_st", "hmf_watson13_fof"}


@dataclass(frozen=True)
class UVLFSamplingResult:
    samples: dict[str, np.ndarray]
    uvlf: dict[str, np.ndarray]
    metadata: dict[str, Any]


def uv_luminosity_to_muv(luminosity_nu: np.ndarray | float) -> np.ndarray | float:
    luminosity = np.asarray(luminosity_nu, dtype=float)
    muv = np.full_like(luminosity, np.nan, dtype=float)
    positive = luminosity > 0.0
    muv[positive] = -2.5 * np.log10(luminosity[positive]) + AB_ZEROPOINT_LNU
    if np.ndim(luminosity_nu) == 0:
        return float(muv)
    return muv


def validate_mass_function_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized in DEPRECATED_MASS_FUNCTION_MODELS:
        raise ValueError(
            f"{normalized} is no longer supported for AuroraLF production runs; "
            f"use {MASS_FUNCTION_MODEL_HMF_REED07}."
        )
    if normalized not in MASS_FUNCTION_MODELS:
        choices = ", ".join(MASS_FUNCTION_MODELS)
        raise ValueError(f"mass_function_model must be one of: {choices}")
    return normalized


def _hmf_reed07_dndm(
    halo_mass_msun: np.ndarray,
    z_obs: float,
    *,
    hmf_dlog10m: float,
) -> np.ndarray:
    if hmf_dlog10m <= 0.0:
        raise ValueError("hmf_dlog10m must be positive")

    h = MASS_FUNCTION_H
    halo_mass_hmf = halo_mass_msun * h
    log_mass_hmf = np.log10(halo_mass_hmf)
    grid_min = np.floor((float(np.min(log_mass_hmf)) - 2.0 * hmf_dlog10m) / hmf_dlog10m) * hmf_dlog10m
    grid_max = np.ceil((float(np.max(log_mass_hmf)) + 2.0 * hmf_dlog10m) / hmf_dlog10m) * hmf_dlog10m
    grid_max += hmf_dlog10m

    mass_function = MassFunction(
        Mmin=grid_min,
        Mmax=grid_max,
        dlog10m=float(hmf_dlog10m),
        z=float(z_obs),
        hmf_model=HMF_REED07_FITTING_FUNCTION,
        sigma_8=MASS_FUNCTION_SIGMA8,
        n=MASS_FUNCTION_NS,
        cosmo_params={
            "H0": 100.0 * h,
            "Om0": MASS_FUNCTION_OMEGA_M,
            "Ob0": MASS_FUNCTION_OMEGA_B_H2 / h**2,
        },
        transfer_params={"extrapolate_with_eh": True},
    )

    grid_mass_msun = np.asarray(mass_function.m, dtype=float) / h
    grid_dndm = np.asarray(mass_function.dndm, dtype=float) * h**4
    valid = np.isfinite(grid_mass_msun) & np.isfinite(grid_dndm) & (grid_mass_msun > 0.0) & (grid_dndm > 0.0)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned too few positive mass-function samples")

    log_grid_mass = np.log(grid_mass_msun[valid])
    log_grid_dndm = np.log(grid_dndm[valid])
    log_mass = np.log(halo_mass_msun)
    if np.min(log_mass) < log_grid_mass[0] or np.max(log_mass) > log_grid_mass[-1]:
        raise RuntimeError(
            f"{MASS_FUNCTION_MODEL_HMF_REED07} interpolation grid does not cover the requested halo masses"
        )
    return np.exp(np.interp(log_mass, log_grid_mass, log_grid_dndm))


def compute_reed07_halo_mass_function_dndm(
    halo_mass_msun: np.ndarray | float,
    z_obs: float,
    *,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> np.ndarray | float:
    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo masses must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo masses must be positive")

    mass_1d = np.atleast_1d(mass)
    dndm = _hmf_reed07_dndm(mass_1d, float(z_obs), hmf_dlog10m=float(hmf_dlog10m))
    if not np.all(np.isfinite(dndm)):
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned non-finite dn/dM values")
    if np.any(dndm < 0.0):
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned negative dn/dM values")
    dndm = np.asarray(dndm, dtype=float).reshape(mass_1d.shape)
    if np.ndim(halo_mass_msun) == 0:
        return float(dndm[0])
    return dndm.reshape(mass.shape)


def compute_halo_mass_function_dndm(
    halo_mass_msun: np.ndarray | float,
    z_obs: float,
    *,
    mass_function_model: str = DEFAULT_MASS_FUNCTION_MODEL,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> np.ndarray | float:
    model = validate_mass_function_model(mass_function_model)
    if model != MASS_FUNCTION_MODEL_HMF_REED07:
        raise RuntimeError(f"unsupported mass function model after validation: {model}")
    return compute_reed07_halo_mass_function_dndm(
        halo_mass_msun,
        z_obs,
        hmf_dlog10m=hmf_dlog10m,
    )


def _resolve_bin_edges(values: np.ndarray, quantity: str, bins: int | np.ndarray) -> np.ndarray:
    if isinstance(bins, np.ndarray):
        if bins.ndim != 1 or bins.size < 2:
            raise ValueError("bins array must be 1D with at least two edges")
        return np.asarray(bins, dtype=float)

    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer or a 1D numpy array")

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite samples available to build histogram edges")

    if quantity == "luminosity":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            raise RuntimeError("no positive luminosity samples available to build histogram edges")
        return np.logspace(np.log10(np.min(positive)), np.log10(np.max(positive)), bins + 1)

    return np.linspace(np.min(finite), np.max(finite), bins + 1)


def _format_progress(completed: int, total: int, elapsed_seconds: float) -> str:
    fraction = completed / total
    filled = int(round(30 * fraction))
    bar = "#" * filled + "-" * (30 - filled)
    rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0.0 else float("inf")
    eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "inf"
    return (
        f"[{bar}] {completed}/{total} "
        f"({fraction * 100.0:.2f}%) "
        f"elapsed={elapsed_seconds:.1f}s "
        f"eta={eta_text}\n"
    )


def _write_progress(progress_path: Path, completed: int, total: int, elapsed_seconds: float) -> str:
    text = _format_progress(completed=completed, total=total, elapsed_seconds=elapsed_seconds)
    progress_path.write_text(text, encoding="utf-8")
    return text


def _run_single_mass_sample(
    args: tuple[
        int,
        float,
        float,
        float,
        float,
        int,
        float,
        int,
        str,
        bool,
        str,
        str,
        float | None,
        str,
        IMFTransitionParameters,
        int | None,
        SFRModelParameters,
    ],
) -> tuple[int, float, np.ndarray, np.ndarray, float, int, int]:
    (
        mass_index,
        log_mass,
        mass,
        weight,
        z_obs,
        n_tracks,
        z_start_max,
        n_grid,
        sampler,
        enable_time_delay,
        ssp_file,
        topheavy_ssp_file,
        topheavy_ssp_metallicity,
        imf_mode,
        imf_transition_parameters,
        random_seed,
        sfr_model_parameters,
    ) = args

    t0 = time.perf_counter()
    pipeline_result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=float(mass),
        z_start_max=z_start_max,
        n_grid=n_grid,
        random_seed=random_seed,
        sampler=sampler,
        enable_time_delay=enable_time_delay,
        workers=1,
        ssp_file=ssp_file,
        topheavy_ssp_file=topheavy_ssp_file,
        topheavy_ssp_metallicity=topheavy_ssp_metallicity,
        imf_mode=imf_mode,
        imf_transition_parameters=imf_transition_parameters,
        sfr_model_parameters=sfr_model_parameters,
    )
    duration = time.perf_counter() - t0
    luminosity = np.asarray(pipeline_result.uv_luminosities, dtype=float)
    topheavy_luminosity = np.asarray(pipeline_result.uv_luminosities_topheavy, dtype=float)
    topheavy_light_fraction = np.zeros_like(luminosity, dtype=float)
    positive_light = luminosity > 0.0
    topheavy_light_fraction[positive_light] = topheavy_luminosity[positive_light] / luminosity[positive_light]
    return (
        mass_index,
        log_mass,
        luminosity,
        topheavy_light_fraction,
        duration,
        int(pipeline_result.metadata["topheavy_source_count"]),
        int(pipeline_result.metadata["starforming_source_count"]),
    )


def sample_uvlf_from_hmf(
    z_obs: float,
    N_mass: int = 3000,
    n_tracks: int = 1000,
    random_seed: int | None = 42,
    *,
    quantity: str = "Muv",
    bins: int | np.ndarray = 40,
    logM_min: float = LOGM_MIN,
    logM_max: float = LOGM_MAX,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    sampler: str = "mcbride",
    enable_time_delay: bool = False,
    pipeline_workers: int | None = None,
    ssp_file: str = DEFAULT_SSP_FILE,
    topheavy_ssp_file: str | None = None,
    topheavy_ssp_metallicity: float | None = DEFAULT_TOPHEAVY_SSP_METALLICITY,
    imf_mode: str = "canonical",
    imf_transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
    progress_path: str | Path | None = None,
    print_progress: bool = False,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
    mass_function_model: str = DEFAULT_MASS_FUNCTION_MODEL,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> UVLFSamplingResult:
    """Sample a UVLF by Monte Carlo integration over a halo mass function."""

    if quantity not in {"Muv", "luminosity"}:
        raise ValueError("quantity must be either 'Muv' or 'luminosity'")
    if N_mass < 1 or n_tracks < 1:
        raise ValueError("N_mass and n_tracks must both be positive")
    if logM_max <= logM_min:
        raise ValueError("logM_max must be larger than logM_min")
    imf_mode = validate_imf_mode(imf_mode)
    mass_function_model = validate_mass_function_model(mass_function_model)
    if topheavy_ssp_file is None:
        topheavy_ssp_file = DEFAULT_TOPHEAVY_SSP_FILE

    pipeline_workers = default_worker_count() if pipeline_workers is None else int(pipeline_workers)
    progress_file = None if progress_path is None else Path(progress_path).expanduser().resolve()
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_text = _write_progress(progress_file, completed=0, total=N_mass, elapsed_seconds=0.0)
        if print_progress:
            print(progress_text.strip(), flush=True)
    rng = np.random.default_rng(random_seed)

    t0 = time.perf_counter()
    logMh = rng.uniform(logM_min, logM_max, size=N_mass)
    Mh = np.power(10.0, logMh)
    dndm = np.asarray(
        compute_halo_mass_function_dndm(
            Mh,
            z_obs,
            mass_function_model=mass_function_model,
            hmf_dlog10m=hmf_dlog10m,
        ),
        dtype=float,
    )
    dndlogM = Mh * np.log(10.0) * dndm
    mass_weight = (logM_max - logM_min) * dndlogM / N_mass

    total_samples = N_mass * n_tracks
    sample_logMh = np.empty(total_samples, dtype=float)
    sample_Mh = np.empty(total_samples, dtype=float)
    sample_mass_weight = np.empty(total_samples, dtype=float)
    sample_track_index = np.empty(total_samples, dtype=int)
    sample_luminosity = np.empty(total_samples, dtype=float)
    sample_topheavy_light_fraction = np.empty(total_samples, dtype=float)
    sample_sample_weight = np.empty(total_samples, dtype=float)
    sample_Muv = np.empty(total_samples, dtype=float)
    per_mass_pipeline_seconds = np.empty(N_mass, dtype=float)
    topheavy_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    starforming_source_count_by_mass = np.empty(N_mass, dtype=np.int64)

    progress_stride = max(1, N_mass // 100)
    tasks = [
        (
            mass_index,
            float(log_mass),
            float(mass),
            float(weight),
            float(z_obs),
            int(n_tracks),
            float(z_start_max),
            int(n_grid),
            sampler,
            bool(enable_time_delay),
            ssp_file,
            str(topheavy_ssp_file),
            topheavy_ssp_metallicity,
            imf_mode,
            imf_transition_parameters,
            None if random_seed is None else int(random_seed + mass_index),
            sfr_model_parameters,
        )
        for mass_index, (log_mass, mass, weight) in enumerate(zip(logMh, Mh, mass_weight, strict=True))
    ]

    if max(1, pipeline_workers) == 1:
        results_iter = (_run_single_mass_sample(task) for task in tasks)
        completed = 0
        for (
            mass_index,
            log_mass,
            luminosity,
            topheavy_light_fraction,
            duration,
            topheavy_source_count,
            starforming_source_count,
        ) in results_iter:
            if luminosity.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of luminosity samples")
            if topheavy_light_fraction.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of top-heavy fractions")

            start = mass_index * n_tracks
            stop = start + n_tracks
            sample_logMh[start:stop] = log_mass
            sample_Mh[start:stop] = Mh[mass_index]
            sample_mass_weight[start:stop] = mass_weight[mass_index]
            sample_track_index[start:stop] = np.arange(n_tracks, dtype=int)
            sample_luminosity[start:stop] = luminosity
            sample_topheavy_light_fraction[start:stop] = topheavy_light_fraction
            sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
            sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
            per_mass_pipeline_seconds[mass_index] = duration
            topheavy_source_count_by_mass[mass_index] = topheavy_source_count
            starforming_source_count_by_mass[mass_index] = starforming_source_count

            completed += 1
            if progress_file is not None and (completed == N_mass or completed % progress_stride == 0):
                progress_text = _write_progress(
                    progress_file,
                    completed=completed,
                    total=N_mass,
                    elapsed_seconds=time.perf_counter() - t0,
                )
                if print_progress:
                    print(progress_text.strip(), flush=True)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=max(1, pipeline_workers)) as executor:
            future_to_index = {executor.submit(_run_single_mass_sample, task): task[0] for task in tasks}
            for future in as_completed(future_to_index):
                (
                    mass_index,
                    log_mass,
                    luminosity,
                    topheavy_light_fraction,
                    duration,
                    topheavy_source_count,
                    starforming_source_count,
                ) = future.result()
                if luminosity.size != n_tracks:
                    raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of luminosity samples")
                if topheavy_light_fraction.size != n_tracks:
                    raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of top-heavy fractions")

                start = mass_index * n_tracks
                stop = start + n_tracks
                sample_logMh[start:stop] = log_mass
                sample_Mh[start:stop] = Mh[mass_index]
                sample_mass_weight[start:stop] = mass_weight[mass_index]
                sample_track_index[start:stop] = np.arange(n_tracks, dtype=int)
                sample_luminosity[start:stop] = luminosity
                sample_topheavy_light_fraction[start:stop] = topheavy_light_fraction
                sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
                sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
                per_mass_pipeline_seconds[mass_index] = duration
                topheavy_source_count_by_mass[mass_index] = topheavy_source_count
                starforming_source_count_by_mass[mass_index] = starforming_source_count

                completed += 1
                if progress_file is not None and (completed == N_mass or completed % progress_stride == 0):
                    progress_text = _write_progress(
                        progress_file,
                        completed=completed,
                        total=N_mass,
                        elapsed_seconds=time.perf_counter() - t0,
                    )
                    if print_progress:
                        print(progress_text.strip(), flush=True)

    if quantity == "luminosity":
        histogram_values = sample_luminosity
    else:
        histogram_values = sample_Muv

    bin_edges = _resolve_bin_edges(histogram_values, quantity=quantity, bins=bins)
    valid_mask = np.isfinite(histogram_values) & np.isfinite(sample_sample_weight)
    if quantity == "luminosity":
        valid_mask &= histogram_values > 0.0

    weighted_counts, used_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
        weights=sample_sample_weight[valid_mask],
    )
    raw_counts, raw_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
    )
    if not np.allclose(used_edges, raw_edges, rtol=0.0, atol=0.0):
        raise RuntimeError("weighted and raw histogram bin edges differ")
    weight_squared_counts, squared_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
        weights=np.square(sample_sample_weight[valid_mask]),
    )
    if not np.allclose(used_edges, squared_edges, rtol=0.0, atol=0.0):
        raise RuntimeError("weighted and squared-weight histogram bin edges differ")
    bin_width = np.diff(used_edges)
    phi = weighted_counts / bin_width
    weighted_count_sigma = np.sqrt(weight_squared_counts)
    phi_sigma = weighted_count_sigma / bin_width
    effective_counts = np.divide(
        np.square(weighted_counts),
        weight_squared_counts,
        out=np.zeros_like(weighted_counts, dtype=float),
        where=weight_squared_counts > 0.0,
    )
    bin_centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    total_seconds = time.perf_counter() - t0

    samples = {
        "logMh": sample_logMh,
        "Mh": sample_Mh,
        "mass_weight": sample_mass_weight,
        "track_index": sample_track_index,
        "luminosity": sample_luminosity,
        "topheavy_light_fraction": sample_topheavy_light_fraction,
        "Muv": sample_Muv,
        "sample_weight": sample_sample_weight,
    }
    uvlf = {
        "quantity": np.array([quantity]),
        "bin_edges": used_edges,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "raw_counts": raw_counts.astype(np.int64),
        "weighted_counts": weighted_counts,
        "weight_squared_counts": weight_squared_counts,
        "weighted_count_sigma": weighted_count_sigma,
        "effective_counts": effective_counts,
        "phi": phi,
        "phi_sigma": phi_sigma,
    }
    metadata = {
        "z_obs": z_obs,
        "N_mass": N_mass,
        "n_tracks": n_tracks,
        "random_seed": random_seed,
        "logM_min": logM_min,
        "logM_max": logM_max,
        "mass_function_model": mass_function_model,
        "hmf_dlog10m": hmf_dlog10m,
        "mass_function_parameters": {
            "ns": MASS_FUNCTION_NS,
            "sigma8": MASS_FUNCTION_SIGMA8,
            "h": MASS_FUNCTION_H,
            "omegam": MASS_FUNCTION_OMEGA_M,
            "omegab_h2": MASS_FUNCTION_OMEGA_B_H2,
        },
        "pipeline_workers": max(1, pipeline_workers),
        "quantity": quantity,
        "ssp_file": ssp_file,
        "topheavy_ssp_file": topheavy_ssp_file,
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "imf_mode": imf_mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "growth_time_threshold_myr": float(imf_transition_parameters.growth_time_threshold_myr),
        },
        "enable_time_delay": enable_time_delay,
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "sampling_seconds": total_seconds,
        "per_mass_pipeline_seconds": per_mass_pipeline_seconds,
        "topheavy_source_count_by_mass": topheavy_source_count_by_mass,
        "starforming_source_count_by_mass": starforming_source_count_by_mass,
        "topheavy_source_fraction": float(np.sum(topheavy_source_count_by_mass) / np.sum(starforming_source_count_by_mass))
        if np.sum(starforming_source_count_by_mass) > 0
        else 0.0,
        "topheavy_light_fraction_median": float(
            np.median(sample_topheavy_light_fraction[np.isfinite(sample_topheavy_light_fraction)])
        )
        if np.any(np.isfinite(sample_topheavy_light_fraction))
        else 0.0,
        "progress_path": None if progress_file is None else str(progress_file),
    }
    return UVLFSamplingResult(samples=samples, uvlf=uvlf, metadata=metadata)
