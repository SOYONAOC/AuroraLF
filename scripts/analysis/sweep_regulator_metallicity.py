#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.chemistry import (  # noqa: E402
    RegulatorMetallicityParameters,
    compute_regulator_metallicity,
    equivalent_oxygen_abundance_from_zsun,
    fire2_highz_mzr_oh12,
    jades_lowmass_mzr_oh12,
)
from auroralf.mah import Cosmology, generate_halo_histories  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402


GHZ2_LOGMSTAR = 8.8
GHZ2_LOGMSTAR_ERR = 0.2
GHZ2_ZSUN = 0.05
GHZ2_ZSUN_LOW = 0.02
GHZ2_ZSUN_HIGH = 0.17


@dataclass(frozen=True)
class TrackBundle:
    logmh: float
    t_grid_gyr: np.ndarray
    z_grid: np.ndarray
    mh_grid: np.ndarray
    sfr_grid: np.ndarray
    active_grid: np.ndarray
    logmstar_median: float
    logmstar_p16: float
    logmstar_p84: float


def _parse_float_grid(text: str) -> list[float]:
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep algebraic gas-regulator metallicity parameters against high-z MZR constraints."
    )
    parser.add_argument("--z-final", type=float, default=12.5)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument(
        "--log-masses",
        nargs="+",
        type=float,
        default=[9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
        help="Final halo masses in log10(Mh/Msun).",
    )
    parser.add_argument("--n-tracks", type=int, default=256)
    parser.add_argument("--n-grid", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--enable-time-delay", action="store_true")
    parser.add_argument("--gas-fraction-norm-grid", type=_parse_float_grid, default=_parse_float_grid("0.02 0.05 0.10 0.20 0.35"))
    parser.add_argument("--gas-fraction-mass-slope-grid", type=_parse_float_grid, default=_parse_float_grid("0.00 0.10 0.20"))
    parser.add_argument("--metal-loading-norm-grid", type=_parse_float_grid, default=_parse_float_grid("0 1 3 5 8 12 20"))
    parser.add_argument("--metal-loading-mass-slope-grid", type=_parse_float_grid, default=_parse_float_grid("-0.50 -0.30 0.00"))
    parser.add_argument("--metal-yield-grid", type=_parse_float_grid, default=_parse_float_grid("0.010 0.015 0.020"))
    parser.add_argument("--returned-fraction", type=float, default=0.4)
    parser.add_argument("--inflow-metallicity-zsun", type=float, default=0.0)
    parser.add_argument("--gas-fraction-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-loading-redshift-slope", type=float, default=0.0)
    parser.add_argument("--score-jades-weight", type=float, default=0.35)
    parser.add_argument("--score-ghz2-weight", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _resolve_prefix(output_prefix: str | None, z_final: float) -> Path:
    if output_prefix is None:
        tag = f"z{str(float(z_final)).replace('.', 'p')}"
        return PROJECT_ROOT / "outputs" / f"regulator_metallicity_sweep_{tag}"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    return prefix.resolve().with_suffix("") if prefix.suffix else prefix.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_tracks < 1:
        raise ValueError("n-tracks must be positive")
    if args.n_grid < 2:
        raise ValueError("n-grid must be at least two")
    if args.z_start_max <= args.z_final:
        raise ValueError("z-start-max must be greater than z-final")
    if not 0.0 <= args.returned_fraction < 1.0:
        raise ValueError("returned-fraction must lie in [0, 1)")
    if args.inflow_metallicity_zsun < 0.0:
        raise ValueError("inflow-metallicity-zsun must be non-negative")
    if args.score_jades_weight < 0.0 or args.score_ghz2_weight < 0.0:
        raise ValueError("score weights must be non-negative")
    for name in (
        "gas_fraction_norm_grid",
        "metal_yield_grid",
    ):
        values = getattr(args, name)
        if any(float(item) <= 0.0 for item in values):
            raise ValueError(f"{name.replace('_', '-')} must contain positive values")
    if any(float(item) < 0.0 for item in args.metal_loading_norm_grid):
        raise ValueError("metal-loading-norm-grid must be non-negative")


def _dt_from_grid(cosmology: Cosmology, z_final: float, z_start_max: float, n_grid: int) -> float:
    from astropy.cosmology import FlatLambdaCDM

    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    t_start = float(astro.age(z_start_max).value)
    t_end = float(astro.age(z_final).value)
    return (t_end - t_start) / float(n_grid - 1)


def _prepare_tracks(args: argparse.Namespace, cosmology: Cosmology) -> list[TrackBundle]:
    bundles: list[TrackBundle] = []
    dt_gyr = _dt_from_grid(cosmology, float(args.z_final), float(args.z_start_max), int(args.n_grid))
    for mass_index, logmh in enumerate(args.log_masses):
        histories = generate_halo_histories(
            n_tracks=int(args.n_tracks),
            z_final=float(args.z_final),
            Mh_final=float(10.0 ** float(logmh)),
            z_start_max=float(args.z_start_max),
            cosmology=cosmology,
            random_seed=int(args.random_seed + 1000 * mass_index),
            time_grid_mode="uniform_in_t",
            dt=dt_gyr,
            store_inactive_history=True,
        )
        sfr_tracks = compute_sfr_from_tracks(
            histories.tracks,
            enable_time_delay=bool(args.enable_time_delay),
        )
        n_halos = int(args.n_tracks)
        n_steps = int(histories.metadata["grid_size"])
        t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, n_steps)
        z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(n_halos, n_steps)
        mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, n_steps)
        sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, n_steps)
        active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, n_steps)
        mstar = _surviving_stellar_mass_grid(
            t_grid_gyr=t_grid,
            sfr_grid=sfr_grid,
            active_grid=active_grid & (sfr_grid > 0.0),
            returned_fraction=float(args.returned_fraction),
        )[:, -1]
        positive = mstar[np.isfinite(mstar) & (mstar > 0.0)]
        if positive.size == 0:
            raise RuntimeError(f"no positive final stellar masses for logMh={float(logmh):g}")
        bundles.append(
            TrackBundle(
                logmh=float(logmh),
                t_grid_gyr=t_grid,
                z_grid=z_grid,
                mh_grid=mh_grid,
                sfr_grid=sfr_grid,
                active_grid=active_grid & (sfr_grid > 0.0),
                logmstar_median=float(np.log10(np.median(positive))),
                logmstar_p16=float(np.log10(np.percentile(positive, 16.0))),
                logmstar_p84=float(np.log10(np.percentile(positive, 84.0))),
            )
        )
    return bundles


def _surviving_stellar_mass_grid(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    returned_fraction: float,
) -> np.ndarray:
    t_grid = np.asarray(t_grid_gyr, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    dt_gyr = np.zeros_like(t_grid, dtype=float)
    dt_gyr[:, 1:] = np.diff(t_grid, axis=1)
    formed = np.where(active & (dt_gyr > 0.0) & (sfr > 0.0), sfr * dt_gyr * 1.0e9, 0.0)
    return (1.0 - float(returned_fraction)) * np.cumsum(formed, axis=1)


def _final_active_median(values: np.ndarray, active_grid: np.ndarray) -> float:
    final_values = np.asarray(values, dtype=float)[:, -1]
    final_active = np.asarray(active_grid, dtype=bool)[:, -1]
    selected = final_values[final_active & np.isfinite(final_values) & (final_values > 0.0)]
    if selected.size == 0:
        raise RuntimeError("no finite positive final active values")
    return float(np.median(selected))


def _evaluate_parameters(
    *,
    bundles: list[TrackBundle],
    parameters: RegulatorMetallicityParameters,
    baryon_fraction: float,
    metallicity_random_seed: int,
    score_jades_weight: float,
    score_ghz2_weight: float,
) -> dict[str, float | str]:
    logmstar = []
    oh12 = []
    zgas = []
    gas_fraction_final = []
    metal_loading_final = []
    for mass_index, bundle in enumerate(bundles):
        result = compute_regulator_metallicity(
            t_grid_gyr=bundle.t_grid_gyr,
            z_grid=bundle.z_grid,
            mh_grid=bundle.mh_grid,
            sfr_grid=bundle.sfr_grid,
            active_grid=bundle.active_grid,
            baryon_fraction=baryon_fraction,
            parameters=parameters,
            random_seed=int(metallicity_random_seed + 1000 * mass_index),
        )
        zgas_median = _final_active_median(result.gas_metallicity_zsun_grid, bundle.active_grid)
        zgas.append(zgas_median)
        oh12.append(float(equivalent_oxygen_abundance_from_zsun(zgas_median)))
        logmstar.append(bundle.logmstar_median)
        gas_fraction_final.append(_final_active_median(result.gas_fraction_grid, bundle.active_grid))
        metal_loading_final.append(_final_active_median(result.metal_loading_grid + 1.0e-300, bundle.active_grid))

    logmstar_array = np.asarray(logmstar, dtype=float)
    oh12_array = np.asarray(oh12, dtype=float)
    fire2 = np.asarray(fire2_highz_mzr_oh12(logmstar_array), dtype=float)
    jades = np.asarray(jades_lowmass_mzr_oh12(logmstar_array), dtype=float)
    order = np.argsort(logmstar_array)
    logmstar_sorted = logmstar_array[order]
    oh12_sorted = oh12_array[order]
    if GHZ2_LOGMSTAR < float(np.min(logmstar_sorted)) or GHZ2_LOGMSTAR > float(np.max(logmstar_sorted)):
        nearest = int(np.argmin(np.abs(logmstar_sorted - GHZ2_LOGMSTAR)))
        ghz2_model = float(oh12_sorted[nearest])
        ghz2_mass_penalty = float(abs(logmstar_sorted[nearest] - GHZ2_LOGMSTAR) / GHZ2_LOGMSTAR_ERR)
    else:
        ghz2_model = float(np.interp(GHZ2_LOGMSTAR, logmstar_sorted, oh12_sorted))
        ghz2_mass_penalty = 0.0
    ghz2_target = float(equivalent_oxygen_abundance_from_zsun(GHZ2_ZSUN))
    ghz2_offset = ghz2_model - ghz2_target
    rms_fire2 = float(np.sqrt(np.mean(np.square(oh12_array - fire2))))
    rms_jades = float(np.sqrt(np.mean(np.square(oh12_array - jades))))
    score = rms_fire2 + float(score_jades_weight) * rms_jades + float(score_ghz2_weight) * abs(ghz2_offset)
    score += 0.1 * ghz2_mass_penalty
    return {
        "score": float(score),
        "rms_fire2_dex": rms_fire2,
        "rms_jades_dex": rms_jades,
        "ghz2_model_oh12": ghz2_model,
        "ghz2_target_oh12": ghz2_target,
        "ghz2_offset_dex": ghz2_offset,
        "median_zgas_zsun": float(np.median(zgas)),
        "median_gas_fraction_final": float(np.median(gas_fraction_final)),
        "median_metal_loading_final": float(np.median(metal_loading_final)),
        "model_logmstar": " ".join(f"{item:.5g}" for item in logmstar_array),
        "model_oh12": " ".join(f"{item:.5g}" for item in oh12_array),
        "model_zgas_zsun": " ".join(f"{item:.5g}" for item in np.asarray(zgas, dtype=float)),
    }


def _iter_parameter_grid(args: argparse.Namespace) -> tuple[int, itertools.product]:
    grids = (
        args.gas_fraction_norm_grid,
        args.gas_fraction_mass_slope_grid,
        args.metal_loading_norm_grid,
        args.metal_loading_mass_slope_grid,
        args.metal_yield_grid,
    )
    total = 1
    for grid in grids:
        total *= len(grid)
    return total, itertools.product(*grids)


def _plot_best_mzr(
    *,
    path_prefix: Path,
    bundles: list[TrackBundle],
    best_row: dict[str, float | str],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    logmstar = np.asarray([bundle.logmstar_median for bundle in bundles], dtype=float)
    logmstar_p16 = np.asarray([bundle.logmstar_p16 for bundle in bundles], dtype=float)
    logmstar_p84 = np.asarray([bundle.logmstar_p84 for bundle in bundles], dtype=float)
    oh12 = np.fromstring(str(best_row["model_oh12"]), sep=" ")
    zgas = np.fromstring(str(best_row["model_zgas_zsun"]), sep=" ")
    x_min = min(6.0, float(np.min(logmstar)) - 0.25)
    x_max = max(9.3, float(np.max(logmstar)) + 0.25)
    x_relation = np.linspace(x_min, x_max, 256)
    ax.plot(x_relation, fire2_highz_mzr_oh12(x_relation), color="#029e73", lw=2.2, label="FIRE-2 high-z MZR")
    ax.plot(x_relation, jades_lowmass_mzr_oh12(x_relation), color="#de8f05", lw=2.0, ls="--", label="JADES low-mass MZR")
    ghz2_oh12 = float(equivalent_oxygen_abundance_from_zsun(GHZ2_ZSUN))
    ghz2_lower = ghz2_oh12 - float(equivalent_oxygen_abundance_from_zsun(GHZ2_ZSUN_LOW))
    ghz2_upper = float(equivalent_oxygen_abundance_from_zsun(GHZ2_ZSUN_HIGH)) - ghz2_oh12
    ax.errorbar(
        [GHZ2_LOGMSTAR],
        [ghz2_oh12],
        xerr=[[GHZ2_LOGMSTAR_ERR], [GHZ2_LOGMSTAR_ERR]],
        yerr=[[ghz2_lower], [ghz2_upper]],
        fmt="*",
        ms=14,
        color="#cc78bc",
        mec="black",
        mew=0.6,
        capsize=3,
        label="GHZ2/GLASS-z12",
        zorder=5,
    )
    ax.errorbar(
        logmstar,
        oh12,
        xerr=[logmstar - logmstar_p16, logmstar_p84 - logmstar],
        marker="o",
        ms=6,
        lw=2.0,
        color="#0173b2",
        mec="white",
        mew=0.6,
        capsize=2.5,
        label="AuroraLF regulator best scan",
        zorder=6,
    )
    for bundle, x_value, y_value, z_value in zip(bundles, logmstar, oh12, zgas, strict=True):
        ax.text(
            x_value + 0.04,
            y_value - 0.06,
            rf"$\log M_h={bundle.logmh:g}$; $Z={z_value:.2f}Z_\odot$",
            fontsize=7.0,
            color="0.25",
        )
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$12+\log({\rm O/H})$")
    y_min = min(float(np.min(oh12)), ghz2_oh12 - ghz2_lower, float(np.min(fire2_highz_mzr_oh12(x_relation)))) - 0.18
    y_max = max(float(np.max(oh12)), ghz2_oh12 + ghz2_upper, float(np.max(jades_lowmass_mzr_oh12(x_relation)))) + 0.18
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.text(
        0.98,
        0.03,
        r"$Z/Z_\odot$ converted with solar $12+\log({\rm O/H})=8.69$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.35",
    )
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_prefix.with_suffix(".png")
    pdf_path = path_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _plot_residuals(
    *,
    path_prefix: Path,
    bundles: list[TrackBundle],
    best_row: dict[str, float | str],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
    logmstar = np.asarray([bundle.logmstar_median for bundle in bundles], dtype=float)
    oh12 = np.fromstring(str(best_row["model_oh12"]), sep=" ")
    fire2 = np.asarray(fire2_highz_mzr_oh12(logmstar), dtype=float)
    jades = np.asarray(jades_lowmass_mzr_oh12(logmstar), dtype=float)
    ax.axhline(0.0, color="0.25", lw=1.2)
    ax.plot(logmstar, oh12 - fire2, marker="o", lw=2.0, color="#029e73", label="AuroraLF - FIRE-2")
    ax.plot(logmstar, oh12 - jades, marker="s", lw=2.0, color="#de8f05", ls="--", label="AuroraLF - JADES")
    ax.axvline(GHZ2_LOGMSTAR, color="#cc78bc", lw=1.2, ls=":", label="GHZ2 stellar mass")
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel("metallicity residual [dex]")
    ax.set_ylim(-0.75, 0.75)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.6, loc="best")
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_prefix.with_suffix(".png")
    pdf_path = path_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _plot_score_landscape(
    *,
    path_prefix: Path,
    rows: list[dict[str, float | str]],
    best_row: dict[str, float | str],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    fgas_values = sorted({float(row["gas_fraction_norm"]) for row in rows})
    loading_values = sorted({float(row["metal_loading_norm"]) for row in rows})
    score_grid = np.full((len(loading_values), len(fgas_values)), np.nan, dtype=float)
    for i, loading in enumerate(loading_values):
        for j, fgas in enumerate(fgas_values):
            selected = [
                float(row["score"])
                for row in rows
                if float(row["gas_fraction_norm"]) == fgas and float(row["metal_loading_norm"]) == loading
            ]
            if selected:
                score_grid[i, j] = float(np.min(selected))
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    image = ax.imshow(score_grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(fgas_values)))
    ax.set_xticklabels([f"{item:g}" for item in fgas_values])
    ax.set_yticks(np.arange(len(loading_values)))
    ax.set_yticklabels([f"{item:g}" for item in loading_values])
    ax.set_xlabel(r"$f_{\rm gas}$ normalization")
    ax.set_ylabel(r"$\lambda_Z$ normalization")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("best score in slope/yield subgrid")
    best_fgas = fgas_values.index(float(best_row["gas_fraction_norm"]))
    best_loading = loading_values.index(float(best_row["metal_loading_norm"]))
    ax.scatter([best_fgas], [best_loading], marker="*", s=150, color="#de8f05", edgecolor="black", lw=0.7)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_prefix.with_suffix(".png")
    pdf_path = path_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_prefix = _resolve_prefix(args.output_prefix, float(args.z_final))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_prefix = PROJECT_ROOT / "data_save" / output_prefix.name
    data_prefix.parent.mkdir(parents=True, exist_ok=True)

    cosmology = Cosmology()
    baryon_fraction = cosmology.omega_b / cosmology.omega_m
    t0 = time.perf_counter()
    bundles = _prepare_tracks(args, cosmology)
    total, parameter_iter = _iter_parameter_grid(args)
    rows: list[dict[str, float | str]] = []

    for index, (
        gas_fraction_norm,
        gas_fraction_mass_slope,
        metal_loading_norm,
        metal_loading_mass_slope,
        metal_yield,
    ) in enumerate(parameter_iter, start=1):
        params = RegulatorMetallicityParameters(
            gas_fraction_norm=float(gas_fraction_norm),
            gas_fraction_mass_slope=float(gas_fraction_mass_slope),
            gas_fraction_redshift_slope=float(args.gas_fraction_redshift_slope),
            returned_fraction=float(args.returned_fraction),
            metal_yield=float(metal_yield),
            inflow_metallicity_zsun=float(args.inflow_metallicity_zsun),
            metal_loading_norm=float(metal_loading_norm),
            metal_loading_mass_slope=float(metal_loading_mass_slope),
            metal_loading_redshift_slope=float(args.metal_loading_redshift_slope),
        )
        metrics = _evaluate_parameters(
            bundles=bundles,
            parameters=params,
            baryon_fraction=baryon_fraction,
            metallicity_random_seed=int(args.metallicity_random_seed),
            score_jades_weight=float(args.score_jades_weight),
            score_ghz2_weight=float(args.score_ghz2_weight),
        )
        rows.append(
            {
                "rank": 0,
                **asdict(params),
                **metrics,
            }
        )
        if index == 1 or index == total or index % max(1, total // 10) == 0:
            print(f"evaluated {index}/{total}", flush=True)

    rows.sort(key=lambda row: float(row["score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    best_row = rows[0]

    score_csv = data_prefix.with_name(f"{data_prefix.name}_scores.csv")
    _write_rows(score_csv, rows)
    best_csv = data_prefix.with_name(f"{data_prefix.name}_best_points.csv")
    best_point_rows: list[dict[str, float | str]] = []
    best_oh12 = np.fromstring(str(best_row["model_oh12"]), sep=" ")
    best_zgas = np.fromstring(str(best_row["model_zgas_zsun"]), sep=" ")
    for bundle, oh12_value, zgas_value in zip(bundles, best_oh12, best_zgas, strict=True):
        best_point_rows.append(
            {
                "logmh": bundle.logmh,
                "logmstar_median": bundle.logmstar_median,
                "logmstar_p16": bundle.logmstar_p16,
                "logmstar_p84": bundle.logmstar_p84,
                "zgas_zsun": float(zgas_value),
                "oh12": float(oh12_value),
                "fire2_oh12": float(fire2_highz_mzr_oh12(bundle.logmstar_median)),
                "jades_oh12": float(jades_lowmass_mzr_oh12(bundle.logmstar_median)),
            }
        )
    _write_rows(best_csv, best_point_rows)

    mzr_png, mzr_pdf = _plot_best_mzr(
        path_prefix=output_prefix.with_name(f"{output_prefix.name}_best_mzr"),
        bundles=bundles,
        best_row=best_row,
    )
    residual_png, residual_pdf = _plot_residuals(
        path_prefix=output_prefix.with_name(f"{output_prefix.name}_residuals"),
        bundles=bundles,
        best_row=best_row,
    )
    score_png, score_pdf = _plot_score_landscape(
        path_prefix=output_prefix.with_name(f"{output_prefix.name}_score_landscape"),
        rows=rows,
        best_row=best_row,
    )

    summary_path = output_prefix.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"z_final: {float(args.z_final):g}\n")
        handle.write(f"z_start_max: {float(args.z_start_max):g}\n")
        handle.write(f"log_masses: {' '.join(f'{float(item):g}' for item in args.log_masses)}\n")
        handle.write(f"n_tracks: {int(args.n_tracks)}\n")
        handle.write(f"n_grid: {int(args.n_grid)}\n")
        handle.write(f"enable_time_delay: {bool(args.enable_time_delay)}\n")
        handle.write(f"score_csv: {score_csv}\n")
        handle.write(f"best_points_csv: {best_csv}\n")
        handle.write(f"best_mzr_pdf: {mzr_pdf}\n")
        handle.write(f"residual_pdf: {residual_pdf}\n")
        handle.write(f"score_landscape_pdf: {score_pdf}\n")
        handle.write("\nBest regulator parameters:\n")
        for key in (
            "gas_fraction_norm",
            "gas_fraction_mass_slope",
            "metal_loading_norm",
            "metal_loading_mass_slope",
            "metal_yield",
            "returned_fraction",
            "inflow_metallicity_zsun",
            "score",
            "rms_fire2_dex",
            "rms_jades_dex",
            "ghz2_offset_dex",
            "median_zgas_zsun",
        ):
            handle.write(f"{key}: {best_row[key]}\n")
        handle.write(f"\nelapsed_seconds: {time.perf_counter() - t0:.2f}\n")

    print(f"best score: {float(best_row['score']):.4f}", flush=True)
    print(f"best parameters: {summary_path}", flush=True)
    print(f"saved: {mzr_png} {residual_png} {score_png}", flush=True)


if __name__ == "__main__":
    main()
