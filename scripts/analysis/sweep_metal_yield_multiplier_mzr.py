#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.chemistry import (  # noqa: E402
    CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER,
    MetalEnrichmentParameters,
    equivalent_oxygen_abundance_from_zsun,
    fire2_highz_mzr_oh12,
    jades_lowmass_mzr_oh12,
    max_positive_mzr_offset_dex,
)
from auroralf.uvlf import (  # noqa: E402
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN,
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMFTransitionParameters,
)
from auroralf.uvlf.pipeline import run_halo_uv_pipeline  # noqa: E402


MODE_NAMES = (
    IMF_MODE_CANONICAL,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
)
MODE_LABELS = {
    IMF_MODE_CANONICAL: "canonical",
    IMF_MODE_Z_GATED_MILD_TOPHEAVY: r"$z_{\rm src}\geq10$, low-$Z$ top-heavy",
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY: r"MAH-burst, low-$Z$ top-heavy",
}
MODE_COLORS = {
    IMF_MODE_CANONICAL: "black",
    IMF_MODE_Z_GATED_MILD_TOPHEAVY: "#c44e52",
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY: "#1f77b4",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the top-heavy metal yield multiplier against high-redshift MZR constraints."
    )
    parser.add_argument("--z-final", type=float, default=12.5)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--log-masses", nargs="+", type=float, default=[9.0, 10.0, 11.0, 12.0])
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[1.0, 1.2, CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER, 1.4, 1.6, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--metal-yield", type=float, default=0.02)
    parser.add_argument("--metal-gas-fraction-of-baryons", type=float, default=0.5)
    parser.add_argument("--metal-returned-fraction", type=float, default=0.4)
    parser.add_argument("--metal-mass-loading-norm", type=float, default=5.0)
    parser.add_argument("--metal-yield-scatter-dex", type=float, default=0.2)
    parser.add_argument("--metal-mass-loading-scatter-dex", type=float, default=0.3)
    parser.add_argument("--metal-birth-scatter-dex", type=float, default=0.15)
    parser.add_argument("--z-topheavy-min", type=float, default=DEFAULT_IMF_TRANSITION_PARAMETERS.z_topheavy_min)
    parser.add_argument("--enable-source-redshift-topheavy-gate", action="store_true")
    parser.add_argument("--metallicity-topheavy-max-zsun", type=float, default=DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN)
    parser.add_argument(
        "--growth-time-threshold-myr",
        type=float,
        default=DEFAULT_IMF_TRANSITION_PARAMETERS.growth_time_threshold_myr,
    )
    parser.add_argument("--fire2-positive-tolerance-dex", type=float, default=0.3)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _resolve_prefix(output_prefix: str | None, z_final: float) -> Path:
    if output_prefix is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / f"metal_yield_multiplier_sweep_z{str(z_final).replace('.', 'p')}_{timestamp}"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    return prefix.resolve().with_suffix("") if prefix.suffix else prefix.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_tracks < 1:
        raise ValueError("n-tracks must be positive")
    if args.n_grid < 2:
        raise ValueError("n-grid must be at least 2")
    if not args.log_masses:
        raise ValueError("at least one log mass is required")
    if not args.multipliers:
        raise ValueError("at least one multiplier is required")
    if any(float(item) <= 0.0 for item in args.multipliers):
        raise ValueError("multipliers must be positive")
    if float(args.fire2_positive_tolerance_dex) < 0.0:
        raise ValueError("fire2-positive-tolerance-dex must be non-negative")
    if float(args.metallicity_topheavy_max_zsun) <= 0.0:
        raise ValueError("metallicity-topheavy-max-zsun must be positive")


def _surviving_stellar_mass_summary(result, returned_fraction: float) -> dict[str, float]:
    n_halos = int(result.metadata["n_tracks"])
    n_steps = int(result.metadata["steps_per_halo"])
    t_grid = np.asarray(result.sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, n_steps)
    sfr_grid = np.asarray(result.sfr_tracks["SFR"], dtype=float).reshape(n_halos, n_steps)
    active_grid = np.asarray(result.sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, n_steps)

    surviving_mass = np.zeros(n_halos, dtype=float)
    for halo_index in range(n_halos):
        mask = active_grid[halo_index] & np.isfinite(sfr_grid[halo_index]) & (sfr_grid[halo_index] > 0.0)
        if np.count_nonzero(mask) < 2:
            continue
        formed_mass = float(np.trapezoid(sfr_grid[halo_index, mask], x=t_grid[halo_index, mask] * 1.0e9))
        surviving_mass[halo_index] = max(formed_mass, 0.0) * (1.0 - returned_fraction)

    positive = surviving_mass[np.isfinite(surviving_mass) & (surviving_mass > 0.0)]
    if positive.size == 0:
        raise RuntimeError("no positive surviving stellar masses")
    return {
        "logmstar_median": float(np.log10(np.median(positive))),
        "logmstar_p16": float(np.log10(np.percentile(positive, 16.0))),
        "logmstar_p84": float(np.log10(np.percentile(positive, 84.0))),
    }


def _final_grid_median(grid: np.ndarray, active_grid: np.ndarray) -> float:
    values = np.asarray(grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    if values.shape != active.shape:
        raise ValueError(f"grid and active_grid must have matching shape, got {values.shape} and {active.shape}")
    final_values = values[:, -1]
    final_active = active[:, -1]
    selected = final_values[final_active & np.isfinite(final_values)]
    if selected.size == 0:
        raise RuntimeError("no finite final active metallicity values")
    return float(np.median(selected))


def _plot_sweep(
    *,
    output_prefix: Path,
    score_rows: list[dict[str, float | str]],
    tolerance: float,
) -> tuple[Path, Path]:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 7.5,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    multipliers = sorted({float(row["multiplier"]) for row in score_rows})
    for mode in MODE_NAMES:
        rows = [row for row in score_rows if row["mode"] == mode]
        rows.sort(key=lambda row: float(row["multiplier"]))
        ax.plot(
            [float(row["multiplier"]) for row in rows],
            [float(row["max_positive_fire2_offset_dex"]) for row in rows],
            marker="o",
            lw=2.0,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
    ax.axhline(tolerance, color="0.35", ls="--", lw=1.3, label=rf"target: $\Delta_{{\rm FIRE2}}\leq{tolerance:.2f}$ dex")
    ax.set_xlabel("top-heavy metal yield multiplier")
    ax.set_ylabel(r"max positive FIRE-2 MZR offset [dex]")
    ax.set_xlim(min(multipliers) - 0.05, max(multipliers) + 0.05)
    ax.set_ylim(0.0, max(tolerance * 1.35, max(float(row["max_positive_fire2_offset_dex"]) for row in score_rows) * 1.10))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_prefix = _resolve_prefix(args.output_prefix, args.z_final)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data_save").mkdir(parents=True, exist_ok=True)

    log_masses = [float(item) for item in args.log_masses]
    multipliers = [float(item) for item in args.multipliers]
    transition_parameters = IMFTransitionParameters(
        z_topheavy_min=float(args.z_topheavy_min),
        source_redshift_gate_enabled=bool(args.enable_source_redshift_topheavy_gate),
        growth_time_threshold_myr=float(args.growth_time_threshold_myr),
        metallicity_topheavy_max_zsun=float(args.metallicity_topheavy_max_zsun),
    )

    rows: list[dict[str, float | str]] = []
    score_rows: list[dict[str, float | str]] = []
    stellar_by_mass: dict[float, dict[str, float]] = {}
    t0 = time.perf_counter()

    for multiplier in multipliers:
        print(f"multiplier={multiplier:g}", flush=True)
        metal_parameters = MetalEnrichmentParameters(
            gas_fraction_of_baryons=float(args.metal_gas_fraction_of_baryons),
            metal_yield=float(args.metal_yield),
            topheavy_yield_multiplier=multiplier,
            returned_fraction=float(args.metal_returned_fraction),
            mass_loading_norm=float(args.metal_mass_loading_norm),
            yield_scatter_dex=float(args.metal_yield_scatter_dex),
            mass_loading_scatter_dex=float(args.metal_mass_loading_scatter_dex),
            birth_metallicity_scatter_dex=float(args.metal_birth_scatter_dex),
        )
        for mass_index, log_mass in enumerate(log_masses):
            mass = 10.0**log_mass
            for mode_index, mode in enumerate(MODE_NAMES):
                seed = int(args.random_seed + 1000 * mass_index)
                metallicity_seed = int(args.metallicity_random_seed + 1000 * mass_index + 100000 * mode_index)
                result = run_halo_uv_pipeline(
                    n_tracks=int(args.n_tracks),
                    z_final=float(args.z_final),
                    Mh_final=float(mass),
                    z_start_max=float(args.z_start_max),
                    n_grid=int(args.n_grid),
                    random_seed=seed,
                    workers=1,
                    imf_mode=mode,
                    imf_transition_parameters=transition_parameters,
                    metal_enrichment_parameters=metal_parameters,
                    metallicity_random_seed=metallicity_seed,
                )
                if result.gas_metallicity_zsun_grid is None:
                    raise RuntimeError("metallicity grids were not produced")
                if log_mass not in stellar_by_mass:
                    stellar_by_mass[log_mass] = _surviving_stellar_mass_summary(
                        result,
                        returned_fraction=float(args.metal_returned_fraction),
                    )
                stellar = stellar_by_mass[log_mass]
                zgas = _final_grid_median(result.gas_metallicity_zsun_grid, result.active_grid)
                oh12 = float(equivalent_oxygen_abundance_from_zsun(zgas))
                fire2 = float(fire2_highz_mzr_oh12(stellar["logmstar_median"]))
                jades = float(jades_lowmass_mzr_oh12(stellar["logmstar_median"]))
                rows.append(
                    {
                        "multiplier": multiplier,
                        "logmh": log_mass,
                        "mode": mode,
                        "logmstar_median": stellar["logmstar_median"],
                        "logmstar_p16": stellar["logmstar_p16"],
                        "logmstar_p84": stellar["logmstar_p84"],
                        "zgas_zsun": zgas,
                        "oh12": oh12,
                        "fire2_oh12": fire2,
                        "jades_oh12": jades,
                        "delta_fire2_dex": oh12 - fire2,
                        "delta_jades_dex": oh12 - jades,
                        "source_redshift_gate_enabled": bool(args.enable_source_redshift_topheavy_gate),
                        "metallicity_topheavy_max_zsun": float(args.metallicity_topheavy_max_zsun),
                        "topheavy_source_fraction": float(result.metadata["topheavy_source_fraction"]),
                        "topheavy_light_fraction_median": float(result.metadata["topheavy_light_fraction_median"]),
                    }
                )

    for multiplier in multipliers:
        for mode in MODE_NAMES:
            selected = [row for row in rows if float(row["multiplier"]) == multiplier and row["mode"] == mode]
            selected.sort(key=lambda row: float(row["logmh"]))
            model = np.asarray([float(row["oh12"]) for row in selected], dtype=float)
            fire2 = np.asarray([float(row["fire2_oh12"]) for row in selected], dtype=float)
            jades = np.asarray([float(row["jades_oh12"]) for row in selected], dtype=float)
            score_rows.append(
                {
                    "multiplier": multiplier,
                    "mode": mode,
                    "max_positive_fire2_offset_dex": max_positive_mzr_offset_dex(model, fire2),
                    "max_positive_jades_offset_dex": max_positive_mzr_offset_dex(model, jades),
                    "median_fire2_offset_dex": float(np.median(model - fire2)),
                    "max_fire2_offset_dex": float(np.max(model - fire2)),
                    "min_fire2_offset_dex": float(np.min(model - fire2)),
                }
            )

    table_path = PROJECT_ROOT / "data_save" / f"{output_prefix.name}.csv"
    score_path = PROJECT_ROOT / "data_save" / f"{output_prefix.name}_scores.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "multiplier",
            "logmh",
            "mode",
            "logmstar_median",
            "logmstar_p16",
            "logmstar_p84",
            "zgas_zsun",
            "oh12",
            "fire2_oh12",
            "jades_oh12",
            "delta_fire2_dex",
            "delta_jades_dex",
            "source_redshift_gate_enabled",
            "metallicity_topheavy_max_zsun",
            "topheavy_source_fraction",
            "topheavy_light_fraction_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "multiplier",
            "mode",
            "max_positive_fire2_offset_dex",
            "max_positive_jades_offset_dex",
            "median_fire2_offset_dex",
            "max_fire2_offset_dex",
            "min_fire2_offset_dex",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    png_path, pdf_path = _plot_sweep(
        output_prefix=output_prefix,
        score_rows=score_rows,
        tolerance=float(args.fire2_positive_tolerance_dex),
    )
    candidate_rows = [
        row
        for row in score_rows
        if row["mode"] in (IMF_MODE_Z_GATED_MILD_TOPHEAVY, IMF_MODE_MAH_BURST_MILD_TOPHEAVY)
        and float(row["max_positive_fire2_offset_dex"]) <= float(args.fire2_positive_tolerance_dex)
    ]
    candidate_by_multiplier = {
        multiplier: [
            row
            for row in candidate_rows
            if float(row["multiplier"]) == multiplier
        ]
        for multiplier in multipliers
    }
    passing_multipliers = [
        multiplier
        for multiplier, candidates in candidate_by_multiplier.items()
        if len(candidates) == 2
    ]
    recommended = max(passing_multipliers) if passing_multipliers else np.nan

    summary_path = output_prefix.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"z_final: {float(args.z_final):g}\n")
        handle.write(f"log_masses: {' '.join(f'{item:g}' for item in log_masses)}\n")
        handle.write(f"multipliers: {' '.join(f'{item:g}' for item in multipliers)}\n")
        handle.write(f"n_tracks: {int(args.n_tracks)}\n")
        handle.write(f"n_grid: {int(args.n_grid)}\n")
        handle.write(f"source_redshift_gate_enabled: {bool(args.enable_source_redshift_topheavy_gate)}\n")
        handle.write(f"metallicity_topheavy_max_zsun: {float(args.metallicity_topheavy_max_zsun):g}\n")
        handle.write(f"fire2_positive_tolerance_dex: {float(args.fire2_positive_tolerance_dex):g}\n")
        handle.write(f"recommended_largest_passing_multiplier: {recommended:g}\n")
        handle.write(f"saved_table: {table_path}\n")
        handle.write(f"saved_scores: {score_path}\n")
        handle.write(f"saved_png: {png_path}\n")
        handle.write(f"saved_pdf: {pdf_path}\n")
        handle.write(f"total_seconds: {time.perf_counter() - t0:.3f}\n\n")
        for row in score_rows:
            handle.write(
                f"mTH={float(row['multiplier']):g} {row['mode']}: "
                f"max_pos_FIRE2={float(row['max_positive_fire2_offset_dex']):.3f}, "
                f"median_FIRE2={float(row['median_fire2_offset_dex']):+.3f}\n"
            )
    print(f"recommended_largest_passing_multiplier={recommended:g}")
    print(f"saved_table={table_path}")
    print(f"saved_scores={score_path}")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
