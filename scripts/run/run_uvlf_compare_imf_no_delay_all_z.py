#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from auroralf.uvlf import (
    DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    DEFAULT_MASS_FUNCTION_MODEL,
    compute_dust_attenuated_uvlf,
    sample_uvlf_from_hmf,
)
from auroralf.uvlf.imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY,
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMFTransitionParameters,
    validate_imf_mode,
)


DEFAULT_Z_VALUES = (6.0, 8.0, 10.0, 12.5)
DEFAULT_LOGM_MIN = 9.0
DEFAULT_LOGM_MAX = 13.0
DEFAULT_MUV_MIN = -24.5
DEFAULT_MUV_MAX = -15.0
DEFAULT_VARIANT_MODES = (
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
)


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _default_output_prefix(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "data_save" / f"uvlf_imf_mode_compare_allz_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare delayed-SFR UVLFs for canonical Pop II and two mild top-heavy Pop II IMF variants. "
            "The variants are source-time z-gated and MAH-burst-gated, not global SSP replacements."
        )
    )
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--variant-modes", nargs="+", type=str, default=list(DEFAULT_VARIANT_MODES))
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--muv-min", type=float, default=DEFAULT_MUV_MIN)
    parser.add_argument("--muv-max", type=float, default=DEFAULT_MUV_MAX)
    parser.add_argument("--logM-min", type=float, default=DEFAULT_LOGM_MIN)
    parser.add_argument("--logM-max", type=float, default=DEFAULT_LOGM_MAX)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    time_delay_group = parser.add_mutually_exclusive_group()
    time_delay_group.add_argument("--enable-time-delay", dest="enable_time_delay", action="store_true", default=True)
    time_delay_group.add_argument("--disable-time-delay", dest="enable_time_delay", action="store_false")
    parser.add_argument("--epsilon-0", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.epsilon_0)
    parser.add_argument("--fstar-characteristic-mass", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.characteristic_mass)
    parser.add_argument("--fstar-beta", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.beta_star)
    parser.add_argument("--fstar-gamma", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.gamma_star)
    parser.add_argument("--canonical-ssp-file", type=str, default=DEFAULT_CANONICAL_SSP_FILE)
    parser.add_argument("--topheavy-ssp-file", type=str, default=DEFAULT_MILD_TOPHEAVY_SSP_FILE)
    parser.add_argument("--topheavy-ssp-metallicity", type=float, default=DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY)
    parser.add_argument("--z-topheavy-min", type=float, default=DEFAULT_IMF_TRANSITION_PARAMETERS.z_topheavy_min)
    parser.add_argument(
        "--growth-time-threshold-myr",
        type=float,
        default=DEFAULT_IMF_TRANSITION_PARAMETERS.growth_time_threshold_myr,
    )
    parser.add_argument("--burst-scatter-dex", type=float, default=0.0)
    parser.add_argument("--burst-scatter-timescale-myr", type=float, default=DEFAULT_BURST_SCATTER_TIMESCALE_MYR)
    parser.add_argument("--burst-scatter-random-seed", type=int, default=None)
    parser.add_argument("--disable-burst-scatter-mean-preservation", action="store_true")
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--apply-dust", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def _resolve_prefix(project_root: Path, output_prefix: str | None) -> Path:
    if output_prefix is None:
        return _default_output_prefix(project_root)
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = (project_root / prefix).resolve()
    else:
        prefix = prefix.resolve()
    return prefix.with_suffix("") if prefix.suffix else prefix


def _require_slurm_job() -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    raise RuntimeError(
        "This UVLF comparison is a compute job and must run inside a SLURM allocation. "
        "Submit it with scripts/submit/submit_uvlf_imf_compare.py instead of running it on the login node."
    )


def _run_single_imf_mode_uvlf(
    *,
    z_obs: float,
    n_mass: int,
    n_tracks: int,
    bins: np.ndarray,
    logm_min: float,
    logm_max: float,
    random_seed: int,
    z_start_max: float,
    n_grid: int,
    sampler: str,
    workers: int,
    enable_time_delay: bool,
    canonical_ssp_file: Path,
    topheavy_ssp_file: Path,
    topheavy_ssp_metallicity: float | None,
    imf_mode: str,
    imf_transition_parameters: IMFTransitionParameters,
    progress_path: Path,
    print_progress: bool,
    sfr_model_parameters: SFRModelParameters,
    burst_scatter_dex: float,
    burst_scatter_timescale_myr: float,
    burst_scatter_random_seed: int | None,
    burst_scatter_preserve_mean: bool,
    mass_function_model: str,
) -> dict[str, np.ndarray | dict[str, object]]:
    result = sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=n_mass,
        n_tracks=n_tracks,
        random_seed=random_seed,
        quantity="Muv",
        bins=bins,
        logM_min=logm_min,
        logM_max=logm_max,
        z_start_max=z_start_max,
        n_grid=n_grid,
        sampler=sampler,
        enable_time_delay=enable_time_delay,
        pipeline_workers=workers,
        ssp_file=str(canonical_ssp_file),
        topheavy_ssp_file=str(topheavy_ssp_file),
        topheavy_ssp_metallicity=topheavy_ssp_metallicity,
        imf_mode=imf_mode,
        imf_transition_parameters=imf_transition_parameters,
        progress_path=progress_path,
        print_progress=print_progress,
        sfr_model_parameters=sfr_model_parameters,
        burst_scatter_dex=burst_scatter_dex,
        burst_scatter_timescale_myr=burst_scatter_timescale_myr,
        burst_scatter_random_seed=burst_scatter_random_seed,
        burst_scatter_preserve_mean=burst_scatter_preserve_mean,
        mass_function_model=mass_function_model,
    )
    return {
        "bin_edges": np.asarray(result.uvlf["bin_edges"], dtype=float),
        "bin_centers": np.asarray(result.uvlf["bin_centers"], dtype=float),
        "bin_width": np.asarray(result.uvlf["bin_width"], dtype=float),
        "raw_counts": np.asarray(result.uvlf["raw_counts"], dtype=np.int64),
        "weighted_counts": np.asarray(result.uvlf["weighted_counts"], dtype=float),
        "weight_squared_counts": np.asarray(result.uvlf["weight_squared_counts"], dtype=float),
        "weighted_count_sigma": np.asarray(result.uvlf["weighted_count_sigma"], dtype=float),
        "effective_counts": np.asarray(result.uvlf["effective_counts"], dtype=float),
        "phi": np.asarray(result.uvlf["phi"], dtype=float),
        "phi_sigma": np.asarray(result.uvlf["phi_sigma"], dtype=float),
        "metadata": result.metadata,
    }


def _apply_optional_dust(phi: np.ndarray, centers: np.ndarray, z_obs: float, apply_dust: bool) -> np.ndarray:
    if not apply_dust:
        return phi
    dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=centers,
        intrinsic_phi=phi,
        z=z_obs,
        muv_obs=centers,
    )
    return np.asarray(dust["phi_obs"], dtype=float)


def _scale_mc_sigma_to_final_phi(
    *,
    intrinsic_phi: np.ndarray,
    intrinsic_phi_sigma: np.ndarray,
    final_phi: np.ndarray,
) -> np.ndarray:
    fractional_sigma = np.divide(
        intrinsic_phi_sigma,
        intrinsic_phi,
        out=np.full_like(intrinsic_phi_sigma, np.nan, dtype=float),
        where=intrinsic_phi > 0.0,
    )
    return final_phi * fractional_sigma


def main() -> None:
    args = _parse_args()
    _require_slurm_job()
    project_root = PROJECT_ROOT
    data_save_dir = project_root / "data_save"
    outputs_dir = project_root / "outputs"
    data_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    variant_modes = () if args.canonical_only else tuple(validate_imf_mode(mode) for mode in args.variant_modes)
    if IMF_MODE_CANONICAL in variant_modes:
        raise ValueError("variant-modes must not include canonical; it is run automatically as the baseline")
    imf_modes = (IMF_MODE_CANONICAL, *variant_modes)
    mass_function_model = DEFAULT_MASS_FUNCTION_MODEL

    output_prefix = _resolve_prefix(project_root, args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = output_prefix.name
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = outputs_dir / f"{stem}.txt"

    canonical_ssp_file = (project_root / args.canonical_ssp_file).resolve()
    topheavy_ssp_file = (project_root / args.topheavy_ssp_file).resolve()
    if not canonical_ssp_file.exists():
        raise FileNotFoundError(f"Canonical SSP file not found: {canonical_ssp_file}")
    if not topheavy_ssp_file.exists():
        raise FileNotFoundError(f"Mild top-heavy SSP file not found: {topheavy_ssp_file}")

    z_values = [float(z) for z in args.z_values]
    if len(z_values) == 0:
        raise ValueError("at least one redshift must be provided")
    if args.N_mass < 1 or args.n_tracks < 1 or args.bins < 1:
        raise ValueError("N-mass, n-tracks, and bins must all be positive")
    if args.muv_max <= args.muv_min:
        raise ValueError("muv-max must be larger than muv-min")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must be larger than logM-min")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 <= float(args.epsilon_0) <= 1.0:
        raise ValueError("epsilon-0 must lie in [0, 1]")
    if float(args.fstar_characteristic_mass) <= 0.0:
        raise ValueError("fstar-characteristic-mass must be positive")
    if float(args.fstar_beta) < 0.0:
        raise ValueError("fstar-beta must be non-negative")
    if float(args.fstar_gamma) < 0.0:
        raise ValueError("fstar-gamma must be non-negative")
    if float(args.burst_scatter_dex) < 0.0:
        raise ValueError("burst-scatter-dex must be non-negative")
    if float(args.burst_scatter_timescale_myr) <= 0.0:
        raise ValueError("burst-scatter-timescale-myr must be positive")

    imf_transition_parameters = IMFTransitionParameters(
        z_topheavy_min=float(args.z_topheavy_min),
        growth_time_threshold_myr=float(args.growth_time_threshold_myr),
    )
    sfr_model_parameters = SFRModelParameters(
        epsilon_0=float(args.epsilon_0),
        characteristic_mass=float(args.fstar_characteristic_mass),
        beta_star=float(args.fstar_beta),
        gamma_star=float(args.fstar_gamma),
    )

    bins = np.linspace(float(args.muv_min), float(args.muv_max), int(args.bins) + 1, dtype=float)
    payload: dict[str, np.ndarray] = {
        "z_values": np.asarray(z_values, dtype=float),
        "mode_names": np.asarray(imf_modes),
        "variant_mode_names": np.asarray(variant_modes),
        "shared_bin_edges": bins,
        "workers": np.asarray([int(args.workers)], dtype=int),
        "N_mass": np.asarray([int(args.N_mass)], dtype=int),
        "n_tracks": np.asarray([int(args.n_tracks)], dtype=int),
        "random_seed": np.asarray([int(args.random_seed)], dtype=int),
        "z_start_max": np.asarray([float(args.z_start_max)], dtype=float),
        "n_grid": np.asarray([int(args.n_grid)], dtype=int),
        "bins_count": np.asarray([int(args.bins)], dtype=int),
        "muv_min": np.asarray([float(args.muv_min)], dtype=float),
        "muv_max": np.asarray([float(args.muv_max)], dtype=float),
        "logM_min": np.asarray([float(args.logM_min)], dtype=float),
        "logM_max": np.asarray([float(args.logM_max)], dtype=float),
        "apply_dust": np.asarray([bool(args.apply_dust)]),
        "mass_function_model": np.asarray([mass_function_model]),
        "enable_time_delay": np.asarray([bool(args.enable_time_delay)]),
        "burst_scatter_dex": np.asarray([float(args.burst_scatter_dex)], dtype=float),
        "burst_scatter_timescale_myr": np.asarray([float(args.burst_scatter_timescale_myr)], dtype=float),
        "burst_scatter_random_seed": np.asarray(
            [-1 if args.burst_scatter_random_seed is None else int(args.burst_scatter_random_seed)],
            dtype=int,
        ),
        "burst_scatter_preserve_mean": np.asarray([not bool(args.disable_burst_scatter_mean_preservation)]),
        "epsilon_0": np.asarray([float(args.epsilon_0)], dtype=float),
        "fstar_characteristic_mass": np.asarray([float(args.fstar_characteristic_mass)], dtype=float),
        "fstar_beta": np.asarray([float(args.fstar_beta)], dtype=float),
        "fstar_gamma": np.asarray([float(args.fstar_gamma)], dtype=float),
        "canonical_ssp_file": np.asarray([str(canonical_ssp_file)]),
        "topheavy_ssp_file": np.asarray([str(topheavy_ssp_file)]),
        "topheavy_ssp_metallicity": np.asarray(
            [np.nan if args.topheavy_ssp_metallicity is None else float(args.topheavy_ssp_metallicity)],
            dtype=float,
        ),
        "z_topheavy_min": np.asarray([float(imf_transition_parameters.z_topheavy_min)], dtype=float),
        "growth_time_threshold_myr": np.asarray(
            [float(imf_transition_parameters.growth_time_threshold_myr)],
            dtype=float,
        ),
    }

    summary_lines = [
        f"python: {sys.executable}",
        f"npz_path: {npz_path}",
        f"summary_path: {summary_path}",
        f"canonical_ssp_file: {canonical_ssp_file}",
        f"topheavy_ssp_file: {topheavy_ssp_file}",
        f"topheavy_ssp_metallicity: {args.topheavy_ssp_metallicity}",
        f"imf_modes: {' '.join(imf_modes)}",
        f"z_topheavy_min: {imf_transition_parameters.z_topheavy_min:g}",
        f"growth_time_threshold_myr: {imf_transition_parameters.growth_time_threshold_myr:g}",
        f"workers: {args.workers}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"bins: {args.bins}",
        f"muv_range: [{args.muv_min}, {args.muv_max}]",
        f"logM_range: [{args.logM_min}, {args.logM_max}]",
        f"z_values: {' '.join(str(z) for z in z_values)}",
        f"apply_dust: {bool(args.apply_dust)}",
        f"mass_function_model: {mass_function_model}",
        f"epsilon_0: {float(args.epsilon_0)}",
        f"fstar_characteristic_mass: {float(args.fstar_characteristic_mass)}",
        f"fstar_beta: {float(args.fstar_beta)}",
        f"fstar_gamma: {float(args.fstar_gamma)}",
        f"enable_time_delay: {bool(args.enable_time_delay)}",
        f"burst_scatter_dex: {float(args.burst_scatter_dex)}",
        f"burst_scatter_timescale_myr: {float(args.burst_scatter_timescale_myr)}",
        f"burst_scatter_random_seed: {args.burst_scatter_random_seed}",
        f"burst_scatter_preserve_mean: {not bool(args.disable_burst_scatter_mean_preservation)}",
        "",
    ]

    t0 = time.perf_counter()
    for z_index, z_obs in enumerate(z_values):
        z_tag = _tag_from_z(z_obs)
        seed = int(args.random_seed + 1000 * z_index)
        burst_seed = None if args.burst_scatter_random_seed is None else int(args.burst_scatter_random_seed + 1000 * z_index)
        mode_results: dict[str, dict[str, np.ndarray | dict[str, object]]] = {}

        print(
            f"Computing z={z_obs:g} with shared seed={seed}, workers={args.workers}, "
            f"mass_function_model={mass_function_model}, modes={','.join(imf_modes)}, "
            f"enable_time_delay={bool(args.enable_time_delay)}, burst_scatter_dex={float(args.burst_scatter_dex):g}",
            flush=True,
        )

        for imf_mode in imf_modes:
            progress = outputs_dir / f"{stem}_{z_tag}_{imf_mode}_progress.txt"
            mode_results[imf_mode] = _run_single_imf_mode_uvlf(
                z_obs=z_obs,
                n_mass=int(args.N_mass),
                n_tracks=int(args.n_tracks),
                bins=bins,
                logm_min=float(args.logM_min),
                logm_max=float(args.logM_max),
                random_seed=seed,
                z_start_max=float(args.z_start_max),
                n_grid=int(args.n_grid),
                sampler=str(args.sampler),
                workers=int(args.workers),
                enable_time_delay=bool(args.enable_time_delay),
                canonical_ssp_file=canonical_ssp_file,
                topheavy_ssp_file=topheavy_ssp_file,
                topheavy_ssp_metallicity=args.topheavy_ssp_metallicity,
                imf_mode=imf_mode,
                imf_transition_parameters=imf_transition_parameters,
                progress_path=progress,
                print_progress=bool(args.print_progress),
                sfr_model_parameters=sfr_model_parameters,
                burst_scatter_dex=float(args.burst_scatter_dex),
                burst_scatter_timescale_myr=float(args.burst_scatter_timescale_myr),
                burst_scatter_random_seed=burst_seed,
                burst_scatter_preserve_mean=not bool(args.disable_burst_scatter_mean_preservation),
                mass_function_model=mass_function_model,
            )

        canonical = mode_results[IMF_MODE_CANONICAL]
        centers = np.asarray(canonical["bin_centers"], dtype=float)
        bin_width = np.asarray(canonical["bin_width"], dtype=float)
        canonical_phi_intrinsic = np.asarray(canonical["phi"], dtype=float)
        canonical_phi_intrinsic_sigma = np.asarray(canonical["phi_sigma"], dtype=float)
        canonical_phi_final = _apply_optional_dust(
            phi=canonical_phi_intrinsic,
            centers=centers,
            z_obs=z_obs,
            apply_dust=bool(args.apply_dust),
        )
        canonical_phi_final_sigma = _scale_mc_sigma_to_final_phi(
            intrinsic_phi=canonical_phi_intrinsic,
            intrinsic_phi_sigma=canonical_phi_intrinsic_sigma,
            final_phi=canonical_phi_final,
        )
        payload[f"{z_tag}_bin_edges"] = np.asarray(canonical["bin_edges"], dtype=float)
        payload[f"{z_tag}_bin_centers"] = centers
        payload[f"{z_tag}_bin_width"] = bin_width
        payload[f"{z_tag}_seed"] = np.asarray([seed], dtype=int)
        payload[f"{z_tag}_burst_scatter_seed"] = np.asarray([-1 if burst_seed is None else int(burst_seed)], dtype=int)
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_intrinsic_phi"] = canonical_phi_intrinsic
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_intrinsic_phi_sigma"] = canonical_phi_intrinsic_sigma
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_phi"] = canonical_phi_final
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_phi_sigma_mc"] = canonical_phi_final_sigma
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_raw_counts"] = np.asarray(canonical["raw_counts"], dtype=np.int64)
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_weighted_counts"] = canonical_phi_final * bin_width
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_intrinsic_weighted_counts"] = np.asarray(
            canonical["weighted_counts"], dtype=float
        )
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_weight_squared_counts"] = np.asarray(
            canonical["weight_squared_counts"], dtype=float
        )
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_effective_counts"] = np.asarray(
            canonical["effective_counts"], dtype=float
        )
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_sampling_seconds"] = np.asarray(
            [float(canonical["metadata"]["sampling_seconds"])],
            dtype=float,
        )
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_topheavy_source_fraction"] = np.asarray(
            [float(canonical["metadata"].get("topheavy_source_fraction", 0.0))],
            dtype=float,
        )
        payload[f"{z_tag}_{IMF_MODE_CANONICAL}_topheavy_light_fraction_median"] = np.asarray(
            [float(canonical["metadata"].get("topheavy_light_fraction_median", 0.0))],
            dtype=float,
        )

        summary_lines.append(f"z={z_obs:g}")
        summary_lines.append(f"  seed={seed}")
        summary_lines.append(f"  burst_scatter_seed={burst_seed}")
        summary_lines.append(
            f"  {IMF_MODE_CANONICAL}_sampling_seconds={float(canonical['metadata']['sampling_seconds']):.3f}"
        )
        summary_lines.append(
            "  "
            f"{IMF_MODE_CANONICAL}_topheavy_source_fraction="
            f"{float(canonical['metadata'].get('topheavy_source_fraction', 0.0)):.6f}"
        )
        summary_lines.append(
            "  "
            f"{IMF_MODE_CANONICAL}_topheavy_light_fraction_median="
            f"{float(canonical['metadata'].get('topheavy_light_fraction_median', 0.0)):.6f}"
        )
        summary_lines.append(
            f"  {IMF_MODE_CANONICAL}_phi_median={float(np.nanmedian(canonical_phi_final[np.isfinite(canonical_phi_final)])):.6e}"
        )

        for imf_mode in variant_modes:
            result = mode_results[imf_mode]
            mode_centers = np.asarray(result["bin_centers"], dtype=float)
            if not np.allclose(centers, mode_centers, rtol=0.0, atol=0.0):
                raise RuntimeError(f"bin centers differ between canonical and {imf_mode} at z={z_obs:g}")

            phi_intrinsic = np.asarray(result["phi"], dtype=float)
            phi_intrinsic_sigma = np.asarray(result["phi_sigma"], dtype=float)
            phi_final = _apply_optional_dust(
                phi=phi_intrinsic,
                centers=mode_centers,
                z_obs=z_obs,
                apply_dust=bool(args.apply_dust),
            )
            phi_final_sigma = _scale_mc_sigma_to_final_phi(
                intrinsic_phi=phi_intrinsic,
                intrinsic_phi_sigma=phi_intrinsic_sigma,
                final_phi=phi_final,
            )
            ratio = np.divide(
                phi_final,
                canonical_phi_final,
                out=np.full_like(phi_final, np.nan),
                where=canonical_phi_final > 0.0,
            )
            payload[f"{z_tag}_{imf_mode}_intrinsic_phi"] = phi_intrinsic
            payload[f"{z_tag}_{imf_mode}_intrinsic_phi_sigma"] = phi_intrinsic_sigma
            payload[f"{z_tag}_{imf_mode}_phi"] = phi_final
            payload[f"{z_tag}_{imf_mode}_phi_sigma_mc"] = phi_final_sigma
            payload[f"{z_tag}_{imf_mode}_raw_counts"] = np.asarray(result["raw_counts"], dtype=np.int64)
            payload[f"{z_tag}_{imf_mode}_weighted_counts"] = phi_final * bin_width
            payload[f"{z_tag}_{imf_mode}_intrinsic_weighted_counts"] = np.asarray(
                result["weighted_counts"], dtype=float
            )
            payload[f"{z_tag}_{imf_mode}_weight_squared_counts"] = np.asarray(
                result["weight_squared_counts"], dtype=float
            )
            payload[f"{z_tag}_{imf_mode}_effective_counts"] = np.asarray(result["effective_counts"], dtype=float)
            payload[f"{z_tag}_{imf_mode}_phi_ratio_over_{IMF_MODE_CANONICAL}"] = ratio
            payload[f"{z_tag}_{imf_mode}_sampling_seconds"] = np.asarray(
                [float(result["metadata"]["sampling_seconds"])],
                dtype=float,
            )
            payload[f"{z_tag}_{imf_mode}_topheavy_source_fraction"] = np.asarray(
                [float(result["metadata"].get("topheavy_source_fraction", 0.0))],
                dtype=float,
            )
            payload[f"{z_tag}_{imf_mode}_topheavy_light_fraction_median"] = np.asarray(
                [float(result["metadata"].get("topheavy_light_fraction_median", 0.0))],
                dtype=float,
            )

            overlap = np.isfinite(ratio) & np.isfinite(canonical_phi_final) & np.isfinite(phi_final)
            summary_lines.append(f"  {imf_mode}_sampling_seconds={float(result['metadata']['sampling_seconds']):.3f}")
            summary_lines.append(
                "  "
                f"{imf_mode}_topheavy_source_fraction="
                f"{float(result['metadata'].get('topheavy_source_fraction', 0.0)):.6f}"
            )
            summary_lines.append(
                "  "
                f"{imf_mode}_topheavy_light_fraction_median="
                f"{float(result['metadata'].get('topheavy_light_fraction_median', 0.0)):.6f}"
            )
            if np.any(overlap):
                summary_lines.append(f"  {imf_mode}_ratio_median={float(np.nanmedian(ratio[overlap])):.6f}")
                summary_lines.append(f"  {imf_mode}_ratio_min={float(np.nanmin(ratio[overlap])):.6f}")
                summary_lines.append(f"  {imf_mode}_ratio_max={float(np.nanmax(ratio[overlap])):.6f}")
            else:
                summary_lines.append(f"  {imf_mode}_ratio_median=nan")

        summary_lines.append("")
        np.savez(npz_path, **payload)
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Finished z={z_obs:g}; partial results saved to {npz_path}", flush=True)

    total_seconds = time.perf_counter() - t0
    payload["total_seconds"] = np.asarray([total_seconds], dtype=float)
    np.savez(npz_path, **payload)
    summary_lines.append(f"total_seconds={total_seconds:.3f}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_npz={npz_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)
    print(f"total_seconds={total_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
