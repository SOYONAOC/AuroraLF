from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hmf import MassFunction

from auroralf.chemistry import MZRBirthMetallicityParameters, RegulatorMetallicityParameters
from auroralf.mah import (
    MAH_BACKEND_MCBRIDE,
    MAH_BACKEND_THESAN,
    MAH_BACKEND_TNG,
    THESAN_TIME_GRID_SNAPSHOT,
    validate_thesan_time_grid_mode,
    TNG_TIME_GRID_SNAPSHOT,
    validate_mah_backend,
    validate_tng_time_grid_mode,
)
from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from .imf import DEFAULT_IMF_TRANSITION_PARAMETERS, IMF_MODE_CANONICAL, IMFTransitionParameters, validate_imf_mode
from .pipeline import (
    DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    DEFAULT_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_METALLICITY,
    default_worker_count,
    run_halo_uv_pipeline,
)


LOGM_MIN = 9.0
LOGM_MAX = 13.0
AB_ZEROPOINT_LNU = 51.60
ATOMIC_COOLING_TEMPERATURE_K = 1.0e4
ATOMIC_COOLING_MU = 0.61
POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN = 2.5e5
POPIII_H2_COOLING_REDSHIFT_FACTOR = 26.0
POPIII_LW_FEEDBACK_COEFFICIENT = 22.87
POPIII_LW_FEEDBACK_EXPONENT = 0.47
DEFAULT_LW_BACKGROUND_J21 = 0.0
STELLAR_CHANNEL_BELOW_POPIII_MIN = "below_popiii_min"
STELLAR_CHANNEL_POPIII = "popiii"
STELLAR_CHANNEL_POPII = "popii"
STELLAR_CHANNELS = (
    STELLAR_CHANNEL_BELOW_POPIII_MIN,
    STELLAR_CHANNEL_POPIII,
    STELLAR_CHANNEL_POPII,
)
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


def compute_atomic_cooling_mass_msun(
    z_obs: np.ndarray | float,
    *,
    virial_temperature_k: float = ATOMIC_COOLING_TEMPERATURE_K,
    mu: float = ATOMIC_COOLING_MU,
) -> np.ndarray | float:
    """Return the halo mass corresponding to a virial temperature threshold."""

    if float(virial_temperature_k) <= 0.0:
        raise ValueError("virial_temperature_k must be positive")
    if float(mu) <= 0.0:
        raise ValueError("mu must be positive")

    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    try:
        import massfunc as mf
    except ImportError as exc:
        raise ImportError("atomic cooling mass requires the optional dependency massfunc") from exc

    threshold = np.asarray(
        mf.SFRD().M_vir(float(mu), float(virial_temperature_k), redshift),
        dtype=float,
    )
    if threshold.shape != redshift.shape:
        threshold = np.broadcast_to(threshold, redshift.shape).copy()
    if not np.all(np.isfinite(threshold)):
        raise RuntimeError("massfunc.SFRD().M_vir returned non-finite atomic cooling masses")
    if np.any(threshold <= 0.0):
        raise RuntimeError("massfunc.SFRD().M_vir returned non-positive atomic cooling masses")
    if np.ndim(z_obs) == 0:
        return float(threshold)
    return threshold


def compute_popiii_lw_minimum_mass_msun(
    z_obs: np.ndarray | float,
    *,
    lw_background_j21: np.ndarray | float = DEFAULT_LW_BACKGROUND_J21,
) -> np.ndarray | float:
    """Return the Pop III minihalo cooling mass including LW feedback.

    ``lw_background_j21`` is the LW intensity in units of
    ``1e-21 erg s^-1 cm^-2 Hz^-1 sr^-1``.  The default zero background reduces
    to the H2 molecular-cooling floor used by the Pop III literature library.
    """

    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    lw_background = np.asarray(lw_background_j21, dtype=float)
    if not np.all(np.isfinite(lw_background)):
        raise ValueError("lw_background_j21 must be finite")
    if np.any(lw_background < 0.0):
        raise ValueError("lw_background_j21 must be non-negative")

    try:
        redshift, lw_background = np.broadcast_arrays(redshift, lw_background)
    except ValueError as exc:
        raise ValueError("lw_background_j21 must be scalar or broadcast with z_obs") from exc

    h2_floor = POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN * (
        POPIII_H2_COOLING_REDSHIFT_FACTOR / (1.0 + redshift)
    )
    threshold = h2_floor * (
        1.0 + POPIII_LW_FEEDBACK_COEFFICIENT * lw_background**POPIII_LW_FEEDBACK_EXPONENT
    )
    if not np.all(np.isfinite(threshold)):
        raise RuntimeError("Pop III LW minimum mass calculation returned non-finite masses")
    if np.any(threshold <= 0.0):
        raise RuntimeError("Pop III LW minimum mass calculation returned non-positive masses")
    if np.ndim(z_obs) == 0 and np.ndim(lw_background_j21) == 0:
        return float(threshold)
    return np.asarray(threshold, dtype=float)


def classify_halo_stellar_channels(
    halo_mass_msun: np.ndarray | float,
    *,
    z_obs: np.ndarray | float,
    lw_background_j21: np.ndarray | float = DEFAULT_LW_BACKGROUND_J21,
    virial_temperature_k: float = ATOMIC_COOLING_TEMPERATURE_K,
    mu: float = ATOMIC_COOLING_MU,
) -> np.ndarray | str:
    """Classify halos into below-PopIII, Pop III minihalo, or Pop II channels."""

    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo masses must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo masses must be positive")

    threshold = np.asarray(
        compute_atomic_cooling_mass_msun(
            z_obs,
            virial_temperature_k=virial_temperature_k,
            mu=mu,
        ),
        dtype=float,
    )
    popiii_minimum = np.asarray(
        compute_popiii_lw_minimum_mass_msun(
            z_obs,
            lw_background_j21=lw_background_j21,
        ),
        dtype=float,
    )
    try:
        mass, popiii_minimum, threshold = np.broadcast_arrays(mass, popiii_minimum, threshold)
    except ValueError as exc:
        raise ValueError("z_obs and lw_background_j21 must be scalar or broadcast with halo_mass_msun") from exc

    channels = np.full(
        mass.shape,
        STELLAR_CHANNEL_BELOW_POPIII_MIN,
        dtype=f"<U{max(len(c) for c in STELLAR_CHANNELS)}",
    )
    channels[mass >= popiii_minimum] = STELLAR_CHANNEL_POPIII
    channels[mass >= threshold] = STELLAR_CHANNEL_POPII
    if np.ndim(halo_mass_msun) == 0 and np.ndim(z_obs) == 0 and np.ndim(lw_background_j21) == 0:
        return str(channels.item())
    return np.asarray(channels, dtype=f"<U{max(len(channel) for channel in STELLAR_CHANNELS)}")


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


def _finite_median_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


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
        str,
        str | None,
        float,
        int,
        float,
        str,
        str | None,
        float,
        int,
        float,
        str,
        bool,
        str,
        str,
        float | None,
        str,
        IMFTransitionParameters,
        int | None,
        SFRModelParameters,
        MZRBirthMetallicityParameters | None,
        RegulatorMetallicityParameters | None,
        int | None,
        float,
        float,
        int | None,
        bool,
    ],
) -> tuple[int, float, np.ndarray, np.ndarray, float, int, int, float, float]:
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
        mah_backend,
        tng_mah_cache_path,
        tng_mass_bin_width_dex,
        tng_min_candidates,
        tng_smoothing_myr,
        tng_time_grid_mode,
        thesan_mah_cache_path,
        thesan_mass_bin_width_dex,
        thesan_min_candidates,
        thesan_smoothing_myr,
        thesan_time_grid_mode,
        enable_time_delay,
        ssp_file,
        topheavy_ssp_file,
        topheavy_ssp_metallicity,
        imf_mode,
        imf_transition_parameters,
        random_seed,
        sfr_model_parameters,
        mzr_metallicity_parameters,
        regulator_metallicity_parameters,
        metallicity_random_seed,
        burst_scatter_dex,
        burst_scatter_timescale_myr,
        burst_scatter_random_seed,
        burst_scatter_preserve_mean,
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
        mah_backend=mah_backend,
        tng_mah_cache_path=tng_mah_cache_path,
        tng_mass_bin_width_dex=tng_mass_bin_width_dex,
        tng_min_candidates=tng_min_candidates,
        tng_smoothing_myr=tng_smoothing_myr,
        tng_time_grid_mode=tng_time_grid_mode,
        thesan_mah_cache_path=thesan_mah_cache_path,
        thesan_mass_bin_width_dex=thesan_mass_bin_width_dex,
        thesan_min_candidates=thesan_min_candidates,
        thesan_smoothing_myr=thesan_smoothing_myr,
        thesan_time_grid_mode=thesan_time_grid_mode,
        enable_time_delay=enable_time_delay,
        workers=1,
        ssp_file=ssp_file,
        topheavy_ssp_file=topheavy_ssp_file,
        topheavy_ssp_metallicity=topheavy_ssp_metallicity,
        imf_mode=imf_mode,
        imf_transition_parameters=imf_transition_parameters,
        sfr_model_parameters=sfr_model_parameters,
        mzr_metallicity_parameters=mzr_metallicity_parameters,
        regulator_metallicity_parameters=regulator_metallicity_parameters,
        metallicity_random_seed=metallicity_random_seed,
        burst_scatter_dex=burst_scatter_dex,
        burst_scatter_timescale_myr=burst_scatter_timescale_myr,
        burst_scatter_random_seed=burst_scatter_random_seed,
        burst_scatter_preserve_mean=burst_scatter_preserve_mean,
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
        float(pipeline_result.metadata["final_gas_metallicity_zsun_median"])
        if pipeline_result.metadata["final_gas_metallicity_zsun_median"] is not None
        else np.nan,
        float(pipeline_result.metadata["birth_metallicity_zsun_starforming_median"])
        if pipeline_result.metadata["birth_metallicity_zsun_starforming_median"] is not None
        else np.nan,
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
    lw_background_j21: float = DEFAULT_LW_BACKGROUND_J21,
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None = None,
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None = None,
    metallicity_random_seed: int | None = None,
    burst_scatter_dex: float = 0.0,
    burst_scatter_timescale_myr: float = DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    burst_scatter_random_seed: int | None = None,
    burst_scatter_preserve_mean: bool = True,
) -> UVLFSamplingResult:
    """Sample a UVLF by Monte Carlo integration over a halo mass function."""

    if quantity not in {"Muv", "luminosity"}:
        raise ValueError("quantity must be either 'Muv' or 'luminosity'")
    if N_mass < 1 or n_tracks < 1:
        raise ValueError("N_mass and n_tracks must both be positive")
    if logM_max <= logM_min:
        raise ValueError("logM_max must be larger than logM_min")
    if float(burst_scatter_dex) < 0.0:
        raise ValueError("burst_scatter_dex must be non-negative")
    if float(burst_scatter_timescale_myr) <= 0.0:
        raise ValueError("burst_scatter_timescale_myr must be positive")
    if np.ndim(np.asarray(lw_background_j21, dtype=float)) != 0:
        raise ValueError("lw_background_j21 must be scalar")
    imf_mode = validate_imf_mode(imf_mode)
    mah_backend = validate_mah_backend(mah_backend)
    tng_time_grid_mode = validate_tng_time_grid_mode(tng_time_grid_mode)
    thesan_time_grid_mode = validate_thesan_time_grid_mode(thesan_time_grid_mode)
    if mah_backend == MAH_BACKEND_TNG and tng_mah_cache_path is None:
        raise ValueError("tng_mah_cache_path is required when mah_backend='tng'")
    if mah_backend == MAH_BACKEND_THESAN and thesan_mah_cache_path is None:
        raise ValueError("thesan_mah_cache_path is required when mah_backend='thesan'")
    if float(tng_mass_bin_width_dex) <= 0.0:
        raise ValueError("tng_mass_bin_width_dex must be positive")
    if int(tng_min_candidates) <= 0:
        raise ValueError("tng_min_candidates must be positive")
    if float(tng_smoothing_myr) < 0.0:
        raise ValueError("tng_smoothing_myr must be non-negative")
    if float(thesan_mass_bin_width_dex) <= 0.0:
        raise ValueError("thesan_mass_bin_width_dex must be positive")
    if int(thesan_min_candidates) <= 0:
        raise ValueError("thesan_min_candidates must be positive")
    if float(thesan_smoothing_myr) < 0.0:
        raise ValueError("thesan_smoothing_myr must be non-negative")
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
    if (
        imf_mode != IMF_MODE_CANONICAL
        and imf_transition_parameters.metallicity_topheavy_max_zsun is not None
        and not birth_metallicity_source_enabled
    ):
        raise ValueError(
            "a birth metallicity source must be provided when metallicity_topheavy_max_zsun is set"
        )
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
    atomic_cooling_mass_msun = float(compute_atomic_cooling_mass_msun(z_obs))
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=float(lw_background_j21))
    )
    stellar_channel_by_mass = np.asarray(
        classify_halo_stellar_channels(Mh, z_obs=z_obs, lw_background_j21=float(lw_background_j21)),
        dtype=f"<U{max(len(channel) for channel in STELLAR_CHANNELS)}",
    )
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
    sample_stellar_channel = np.empty(total_samples, dtype=stellar_channel_by_mass.dtype)
    sample_atomic_cooling_mass_msun = np.empty(total_samples, dtype=float)
    sample_popiii_minimum_mass_msun = np.empty(total_samples, dtype=float)
    sample_sample_weight = np.empty(total_samples, dtype=float)
    sample_Muv = np.empty(total_samples, dtype=float)
    per_mass_pipeline_seconds = np.empty(N_mass, dtype=float)
    topheavy_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    starforming_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    final_gas_metallicity_zsun_median_by_mass = np.full(N_mass, np.nan, dtype=float)
    birth_metallicity_zsun_starforming_median_by_mass = np.full(N_mass, np.nan, dtype=float)

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
            mah_backend,
            None if tng_mah_cache_path is None else str(tng_mah_cache_path),
            float(tng_mass_bin_width_dex),
            int(tng_min_candidates),
            float(tng_smoothing_myr),
            str(tng_time_grid_mode),
            None if thesan_mah_cache_path is None else str(thesan_mah_cache_path),
            float(thesan_mass_bin_width_dex),
            int(thesan_min_candidates),
            float(thesan_smoothing_myr),
            str(thesan_time_grid_mode),
            bool(enable_time_delay),
            ssp_file,
            str(topheavy_ssp_file),
            topheavy_ssp_metallicity,
            imf_mode,
            imf_transition_parameters,
            None if random_seed is None else int(random_seed + mass_index),
            sfr_model_parameters,
            mzr_metallicity_parameters,
            regulator_metallicity_parameters,
            None if metallicity_random_seed is None else int(metallicity_random_seed + mass_index),
            float(burst_scatter_dex),
            float(burst_scatter_timescale_myr),
            None if burst_scatter_random_seed is None else int(burst_scatter_random_seed + mass_index),
            bool(burst_scatter_preserve_mean),
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
            final_gas_metallicity_zsun_median,
            birth_metallicity_zsun_starforming_median,
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
            sample_stellar_channel[start:stop] = stellar_channel_by_mass[mass_index]
            sample_atomic_cooling_mass_msun[start:stop] = atomic_cooling_mass_msun
            sample_popiii_minimum_mass_msun[start:stop] = popiii_minimum_mass_msun
            sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
            sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
            per_mass_pipeline_seconds[mass_index] = duration
            topheavy_source_count_by_mass[mass_index] = topheavy_source_count
            starforming_source_count_by_mass[mass_index] = starforming_source_count
            final_gas_metallicity_zsun_median_by_mass[mass_index] = final_gas_metallicity_zsun_median
            birth_metallicity_zsun_starforming_median_by_mass[mass_index] = (
                birth_metallicity_zsun_starforming_median
            )

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
                    final_gas_metallicity_zsun_median,
                    birth_metallicity_zsun_starforming_median,
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
                sample_stellar_channel[start:stop] = stellar_channel_by_mass[mass_index]
                sample_atomic_cooling_mass_msun[start:stop] = atomic_cooling_mass_msun
                sample_popiii_minimum_mass_msun[start:stop] = popiii_minimum_mass_msun
                sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
                sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
                per_mass_pipeline_seconds[mass_index] = duration
                topheavy_source_count_by_mass[mass_index] = topheavy_source_count
                starforming_source_count_by_mass[mass_index] = starforming_source_count
                final_gas_metallicity_zsun_median_by_mass[mass_index] = final_gas_metallicity_zsun_median
                birth_metallicity_zsun_starforming_median_by_mass[mass_index] = (
                    birth_metallicity_zsun_starforming_median
                )

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
        "stellar_channel": sample_stellar_channel,
        "atomic_cooling_mass_msun": sample_atomic_cooling_mass_msun,
        "popiii_minimum_mass_msun": sample_popiii_minimum_mass_msun,
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
        "halo_mass_by_mass": Mh,
        "log_halo_mass_by_mass": logMh,
        "mass_weight_by_mass": mass_weight,
        "atomic_cooling_temperature_k": ATOMIC_COOLING_TEMPERATURE_K,
        "atomic_cooling_mu": ATOMIC_COOLING_MU,
        "atomic_cooling_mass_msun": atomic_cooling_mass_msun,
        "popiii_h2_cooling_mass_normalization_msun": POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN,
        "popiii_h2_cooling_redshift_factor": POPIII_H2_COOLING_REDSHIFT_FACTOR,
        "popiii_lw_feedback_coefficient": POPIII_LW_FEEDBACK_COEFFICIENT,
        "popiii_lw_feedback_exponent": POPIII_LW_FEEDBACK_EXPONENT,
        "lw_background_j21": float(lw_background_j21),
        "popiii_minimum_mass_msun": popiii_minimum_mass_msun,
        "stellar_channel_by_mass": stellar_channel_by_mass,
        "stellar_channel_counts": {
            channel: int(np.count_nonzero(stellar_channel_by_mass == channel))
            for channel in STELLAR_CHANNELS
        },
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
        "mah_backend": mah_backend,
        "sampler": sampler,
        "tng_mah_cache_path": None if tng_mah_cache_path is None else str(Path(tng_mah_cache_path).expanduser().resolve()),
        "tng_mass_bin_width_dex": None if mah_backend != MAH_BACKEND_TNG else float(tng_mass_bin_width_dex),
        "tng_min_candidates": None if mah_backend != MAH_BACKEND_TNG else int(tng_min_candidates),
        "tng_smoothing_myr": None if mah_backend != MAH_BACKEND_TNG else float(tng_smoothing_myr),
        "tng_time_grid_mode": None if mah_backend != MAH_BACKEND_TNG else str(tng_time_grid_mode),
        "thesan_mah_cache_path": None
        if thesan_mah_cache_path is None
        else str(Path(thesan_mah_cache_path).expanduser().resolve()),
        "thesan_mass_bin_width_dex": None
        if mah_backend != MAH_BACKEND_THESAN
        else float(thesan_mass_bin_width_dex),
        "thesan_min_candidates": None if mah_backend != MAH_BACKEND_THESAN else int(thesan_min_candidates),
        "thesan_smoothing_myr": None if mah_backend != MAH_BACKEND_THESAN else float(thesan_smoothing_myr),
        "thesan_time_grid_mode": None if mah_backend != MAH_BACKEND_THESAN else str(thesan_time_grid_mode),
        "quantity": quantity,
        "ssp_file": ssp_file,
        "topheavy_ssp_file": topheavy_ssp_file,
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "imf_mode": imf_mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "source_redshift_gate_enabled": bool(imf_transition_parameters.source_redshift_gate_enabled),
            "growth_time_threshold_myr": float(imf_transition_parameters.growth_time_threshold_myr),
            "metallicity_topheavy_max_zsun": None
            if imf_transition_parameters.metallicity_topheavy_max_zsun is None
            else float(imf_transition_parameters.metallicity_topheavy_max_zsun),
        },
        "enable_time_delay": enable_time_delay,
        "burst_scatter_enabled": float(burst_scatter_dex) > 0.0,
        "burst_scatter_dex": float(burst_scatter_dex),
        "burst_scatter_timescale_myr": float(burst_scatter_timescale_myr),
        "burst_scatter_random_seed": burst_scatter_random_seed,
        "burst_scatter_preserve_mean": bool(burst_scatter_preserve_mean),
        "burst_scatter_mass_conserving": bool(burst_scatter_preserve_mean),
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
        "mzr_metallicity_enabled": mzr_metallicity_parameters is not None,
        "regulator_metallicity_enabled": regulator_metallicity_parameters is not None,
        "metallicity_source": "mzr"
        if mzr_metallicity_parameters is not None
        else "regulator"
        if regulator_metallicity_parameters is not None
        else "none",
        "metallicity_random_seed": metallicity_random_seed,
        "mzr_metallicity_parameters": mzr_metallicity_parameters.as_metadata()
        if mzr_metallicity_parameters is not None
        else None,
        "regulator_metallicity_parameters": regulator_metallicity_parameters.as_metadata()
        if regulator_metallicity_parameters is not None
        else None,
        "final_gas_metallicity_zsun_median_by_mass": final_gas_metallicity_zsun_median_by_mass,
        "birth_metallicity_zsun_starforming_median_by_mass": birth_metallicity_zsun_starforming_median_by_mass,
        "final_gas_metallicity_zsun_median": _finite_median_or_none(final_gas_metallicity_zsun_median_by_mass)
        if regulator_metallicity_parameters is not None
        else None,
        "birth_metallicity_zsun_starforming_median": _finite_median_or_none(
            birth_metallicity_zsun_starforming_median_by_mass
        )
        if birth_metallicity_source_enabled
        else None,
        "progress_path": None if progress_file is None else str(progress_file),
    }
    return UVLFSamplingResult(samples=samples, uvlf=uvlf, metadata=metadata)
