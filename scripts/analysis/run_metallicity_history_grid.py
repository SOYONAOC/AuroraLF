#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from auroralf.chemistry import (
    CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER,
    MetalEnrichmentParameters,
    summarize_metallicity_history,
)
from auroralf.uvlf import (
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN,
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMFTransitionParameters,
)
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


DEFAULT_MODES = (
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
        description="Compute and plot stochastic metallicity histories for representative halo masses."
    )
    parser.add_argument("--z-final", type=float, default=12.5)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--log-masses", nargs="+", type=float, default=[9.0, 10.0, 11.0, 12.0])
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--metal-yield", type=float, default=0.02)
    parser.add_argument("--metal-topheavy-yield-multiplier", type=float, default=CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER)
    parser.add_argument("--metal-gas-fraction-of-baryons", type=float, default=0.5)
    parser.add_argument("--metal-returned-fraction", type=float, default=0.4)
    parser.add_argument("--metal-mass-loading-norm", type=float, default=5.0)
    parser.add_argument("--metal-yield-scatter-dex", type=float, default=0.2)
    parser.add_argument("--metal-mass-loading-scatter-dex", type=float, default=0.3)
    parser.add_argument("--metal-birth-scatter-dex", type=float, default=0.15)
    parser.add_argument("--z-topheavy-min", type=float, default=DEFAULT_IMF_TRANSITION_PARAMETERS.z_topheavy_min)
    parser.add_argument("--metallicity-topheavy-max-zsun", type=float, default=DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN)
    parser.add_argument(
        "--growth-time-threshold-myr",
        type=float,
        default=DEFAULT_IMF_TRANSITION_PARAMETERS.growth_time_threshold_myr,
    )
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _resolve_prefix(output_prefix: str | None, z_final: float) -> Path:
    if output_prefix is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "data_save" / f"metallicity_history_z{str(z_final).replace('.', 'p')}_{timestamp}"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    return prefix.resolve().with_suffix("") if prefix.suffix else prefix.resolve()


def _mass_tag(log_mass: float) -> str:
    return f"logM{str(float(log_mass)).replace('.', 'p')}"


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_tracks < 1:
        raise ValueError("n-tracks must be positive")
    if args.n_grid < 2:
        raise ValueError("n-grid must be at least 2")
    if len(args.log_masses) == 0:
        raise ValueError("at least one log mass is required")
    if float(args.metal_gas_fraction_of_baryons) <= 0.0:
        raise ValueError("metal-gas-fraction-of-baryons must be positive")
    if float(args.metal_yield) < 0.0:
        raise ValueError("metal-yield must be non-negative")
    if float(args.metal_topheavy_yield_multiplier) <= 0.0:
        raise ValueError("metal-topheavy-yield-multiplier must be positive")
    if not 0.0 <= float(args.metal_returned_fraction) < 1.0:
        raise ValueError("metal-returned-fraction must lie in [0, 1)")
    if float(args.metal_mass_loading_norm) < 0.0:
        raise ValueError("metal-mass-loading-norm must be non-negative")
    if float(args.metal_yield_scatter_dex) < 0.0:
        raise ValueError("metal-yield-scatter-dex must be non-negative")
    if float(args.metal_mass_loading_scatter_dex) < 0.0:
        raise ValueError("metal-mass-loading-scatter-dex must be non-negative")
    if float(args.metal_birth_scatter_dex) < 0.0:
        raise ValueError("metal-birth-scatter-dex must be non-negative")
    if float(args.metallicity_topheavy_max_zsun) <= 0.0:
        raise ValueError("metallicity-topheavy-max-zsun must be positive")


def _plot_histories(
    *,
    output_prefix: Path,
    payload: dict[str, np.ndarray],
    log_masses: list[float],
    mode_names: tuple[str, ...],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    n_cols = len(log_masses)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.1 * n_cols, 6.0),
        sharex="col",
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, log_mass in enumerate(log_masses):
        mass_tag = _mass_tag(log_mass)
        column_values: list[np.ndarray] = []
        for mode in mode_names:
            column_values.append(np.asarray(payload[f"{mass_tag}_{mode}_gas_median"], dtype=float))
            column_values.append(np.asarray(payload[f"{mass_tag}_{mode}_birth_median"], dtype=float))
        finite = np.concatenate([item[np.isfinite(item)] for item in column_values])
        y_max = float(np.nanmax(finite)) if finite.size > 0 else 1.0
        for row, quantity in enumerate(("gas", "birth")):
            ax = axes[row, column]
            for mode in mode_names:
                z = np.asarray(payload[f"{mass_tag}_{mode}_z"], dtype=float)
                median = np.asarray(payload[f"{mass_tag}_{mode}_{quantity}_median"], dtype=float)
                p16 = np.asarray(payload[f"{mass_tag}_{mode}_{quantity}_p16"], dtype=float)
                p84 = np.asarray(payload[f"{mass_tag}_{mode}_{quantity}_p84"], dtype=float)
                valid = np.isfinite(z) & np.isfinite(median)
                if not np.any(valid):
                    continue
                color = MODE_COLORS[mode]
                label = MODE_LABELS[mode]
                ax.plot(z[valid], median[valid], lw=2.0, color=color, label=label)
                spread = valid & np.isfinite(p16) & np.isfinite(p84)
                if np.any(spread):
                    ax.fill_between(z[spread], p16[spread], p84[spread], color=color, alpha=0.14, lw=0.0)
            ax.set_xlim(float(np.nanmax(payload[f"{mass_tag}_{mode_names[0]}_z"])), float(np.nanmin(payload[f"{mass_tag}_{mode_names[0]}_z"])))
            ax.set_ylim(0.0, y_max * 1.08)
            ax.grid(alpha=0.22)
            if row == 0:
                ax.set_title(rf"$\log_{{10}} M_h(z=12.5)={log_mass:g}$")
            if column == 0:
                ylabel = r"$Z_{\rm gas}/Z_\odot$" if quantity == "gas" else r"$Z_{\rm birth}/Z_\odot$"
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("redshift")
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=7.4, loc="upper left")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_prefix = _resolve_prefix(args.output_prefix, args.z_final)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    log_masses = [float(item) for item in args.log_masses]
    mode_names = DEFAULT_MODES
    transition_parameters = IMFTransitionParameters(
        z_topheavy_min=float(args.z_topheavy_min),
        growth_time_threshold_myr=float(args.growth_time_threshold_myr),
        metallicity_topheavy_max_zsun=float(args.metallicity_topheavy_max_zsun),
    )
    metal_parameters = MetalEnrichmentParameters(
        gas_fraction_of_baryons=float(args.metal_gas_fraction_of_baryons),
        metal_yield=float(args.metal_yield),
        topheavy_yield_multiplier=float(args.metal_topheavy_yield_multiplier),
        returned_fraction=float(args.metal_returned_fraction),
        mass_loading_norm=float(args.metal_mass_loading_norm),
        yield_scatter_dex=float(args.metal_yield_scatter_dex),
        mass_loading_scatter_dex=float(args.metal_mass_loading_scatter_dex),
        birth_metallicity_scatter_dex=float(args.metal_birth_scatter_dex),
    )

    payload: dict[str, np.ndarray] = {
        "z_final": np.asarray([float(args.z_final)], dtype=float),
        "z_start_max": np.asarray([float(args.z_start_max)], dtype=float),
        "log_masses": np.asarray(log_masses, dtype=float),
        "mode_names": np.asarray(mode_names),
        "n_tracks": np.asarray([int(args.n_tracks)], dtype=int),
        "n_grid": np.asarray([int(args.n_grid)], dtype=int),
        "random_seed": np.asarray([int(args.random_seed)], dtype=int),
        "metallicity_random_seed": np.asarray([int(args.metallicity_random_seed)], dtype=int),
        "metal_yield": np.asarray([float(args.metal_yield)], dtype=float),
        "metal_topheavy_yield_multiplier": np.asarray([float(args.metal_topheavy_yield_multiplier)], dtype=float),
        "metal_gas_fraction_of_baryons": np.asarray([float(args.metal_gas_fraction_of_baryons)], dtype=float),
        "metal_returned_fraction": np.asarray([float(args.metal_returned_fraction)], dtype=float),
        "metal_mass_loading_norm": np.asarray([float(args.metal_mass_loading_norm)], dtype=float),
        "z_topheavy_min": np.asarray([float(args.z_topheavy_min)], dtype=float),
        "growth_time_threshold_myr": np.asarray([float(args.growth_time_threshold_myr)], dtype=float),
        "metallicity_topheavy_max_zsun": np.asarray([float(args.metallicity_topheavy_max_zsun)], dtype=float),
    }
    summary_lines = [
        f"output_prefix: {output_prefix}",
        f"z_final: {float(args.z_final):g}",
        f"z_start_max: {float(args.z_start_max):g}",
        f"log_masses: {' '.join(f'{item:g}' for item in log_masses)}",
        f"mode_names: {' '.join(mode_names)}",
        f"n_tracks: {int(args.n_tracks)}",
        f"n_grid: {int(args.n_grid)}",
        f"metal_yield: {float(args.metal_yield):g}",
        f"metal_topheavy_yield_multiplier: {float(args.metal_topheavy_yield_multiplier):g}",
        f"metallicity_topheavy_max_zsun: {float(args.metallicity_topheavy_max_zsun):g}",
        "",
    ]

    t0 = time.perf_counter()
    for mass_index, log_mass in enumerate(log_masses):
        mass = 10.0**log_mass
        mass_tag = _mass_tag(log_mass)
        summary_lines.append(f"logM={log_mass:g}")
        for mode_index, mode in enumerate(mode_names):
            seed = int(args.random_seed + 1000 * mass_index)
            metallicity_seed = int(args.metallicity_random_seed + 1000 * mass_index + 100000 * mode_index)
            print(
                f"Computing logM={log_mass:g}, mode={mode}, seed={seed}, metallicity_seed={metallicity_seed}",
                flush=True,
            )
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
            if result.gas_metallicity_zsun_grid is None or result.birth_metallicity_zsun_grid is None:
                raise RuntimeError("metallicity grids were not produced")

            n_halos = int(result.metadata["n_tracks"])
            n_steps = int(result.metadata["steps_per_halo"])
            z_grid = np.asarray(result.sfr_tracks["z"], dtype=float).reshape(n_halos, n_steps)
            sfr_grid = np.asarray(result.sfr_tracks["SFR"], dtype=float).reshape(n_halos, n_steps)
            starforming_grid = result.active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
            history = summarize_metallicity_history(
                z_grid=z_grid,
                gas_metallicity_zsun_grid=result.gas_metallicity_zsun_grid,
                birth_metallicity_zsun_grid=result.birth_metallicity_zsun_grid,
                active_grid=result.active_grid,
                starforming_grid=starforming_grid,
                topheavy_source_grid=result.imf_topheavy_source_grid,
            )

            prefix = f"{mass_tag}_{mode}"
            for key, value in history.items():
                payload[f"{prefix}_{key}"] = np.asarray(value)
            payload[f"{prefix}_topheavy_source_fraction_total"] = np.asarray(
                [float(result.metadata["topheavy_source_fraction"])],
                dtype=float,
            )
            payload[f"{prefix}_topheavy_light_fraction_median"] = np.asarray(
                [float(result.metadata["topheavy_light_fraction_median"])],
                dtype=float,
            )
            payload[f"{prefix}_uv_luminosity_median"] = np.asarray(
                [float(np.median(result.uv_luminosities[np.isfinite(result.uv_luminosities)]))],
                dtype=float,
            )
            final_index = int(np.flatnonzero(np.isfinite(history["gas_median"]))[-1])
            final_birth_indices = np.flatnonzero(np.isfinite(history["birth_median"]))
            final_birth = float(history["birth_median"][final_birth_indices[-1]]) if final_birth_indices.size else np.nan
            summary_lines.append(
                "  "
                f"{mode}: Zgas_final={float(history['gas_median'][final_index]):.6g}, "
                f"Zbirth_final={final_birth:.6g}, "
                f"topheavy_source_fraction={float(result.metadata['topheavy_source_fraction']):.6f}, "
                f"topheavy_light_fraction_median={float(result.metadata['topheavy_light_fraction_median']):.6f}"
            )
        summary_lines.append("")

    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(npz_path, **payload)
    png_path, pdf_path = _plot_histories(output_prefix=PROJECT_ROOT / "outputs" / output_prefix.name, payload=payload, log_masses=log_masses, mode_names=mode_names)
    summary_path = PROJECT_ROOT / "outputs" / f"{output_prefix.name}.txt"
    summary_lines.append(f"total_seconds: {time.perf_counter() - t0:.3f}")
    summary_lines.append(f"saved_npz: {npz_path}")
    summary_lines.append(f"saved_png: {png_path}")
    summary_lines.append(f"saved_pdf: {pdf_path}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"saved_npz={npz_path}")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
