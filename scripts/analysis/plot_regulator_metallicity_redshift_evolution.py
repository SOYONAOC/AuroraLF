#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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
)
from auroralf.mah import Cosmology, generate_halo_histories  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402


GHZ2_ZSUN = 0.05


@dataclass(frozen=True)
class RedshiftSummary:
    logmh: float
    z: np.ndarray
    median: np.ndarray
    p16: np.ndarray
    p84: np.ndarray
    count: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regulator metallicity evolution with redshift along fixed MAH/SFR histories."
    )
    parser.add_argument("--z-final", type=float, default=12.5)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--z-plot-max", type=float, default=30.0)
    parser.add_argument("--log-masses", nargs="+", type=float, default=[10.0, 11.0, 12.0])
    parser.add_argument("--n-tracks", type=int, default=256)
    parser.add_argument("--n-grid", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--enable-time-delay", action="store_true")
    parser.add_argument(
        "--gas-fraction-norm",
        type=float,
        default=0.02,
        help="f_res = Mgas / (fb Mh), not Mgas / (Mgas + Mstar).",
    )
    parser.add_argument("--gas-fraction-mass-slope", type=float, default=0.0, help="Mass slope for f_res.")
    parser.add_argument("--gas-fraction-redshift-slope", type=float, default=0.0, help="Redshift slope for f_res.")
    parser.add_argument("--metal-loading-norm", type=float, default=20.0)
    parser.add_argument("--metal-loading-mass-slope", type=float, default=-0.5)
    parser.add_argument("--metal-loading-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-yield", type=float, default=0.01)
    parser.add_argument("--returned-fraction", type=float, default=0.4)
    parser.add_argument("--inflow-metallicity-zsun", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _resolve_prefix(output_prefix: str | None, z_final: float) -> Path:
    if output_prefix is None:
        tag = f"z{str(float(z_final)).replace('.', 'p')}"
        return PROJECT_ROOT / "outputs" / f"regulator_metallicity_redshift_evolution_{tag}"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    return prefix.resolve().with_suffix("") if prefix.suffix else prefix.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_tracks < 1:
        raise ValueError("n-tracks must be positive")
    if args.n_grid < 2:
        raise ValueError("n-grid must be at least two")
    if not args.log_masses:
        raise ValueError("at least one log mass is required")
    if args.z_start_max <= args.z_final:
        raise ValueError("z-start-max must be greater than z-final")
    if args.z_plot_max <= args.z_final or args.z_plot_max > args.z_start_max:
        raise ValueError("z-plot-max must lie between z-final and z-start-max")
    if args.gas_fraction_norm <= 0.0:
        raise ValueError("gas-fraction-norm must be positive")
    if args.metal_loading_norm < 0.0:
        raise ValueError("metal-loading-norm must be non-negative")
    if args.metal_yield < 0.0:
        raise ValueError("metal-yield must be non-negative")
    if not 0.0 <= args.returned_fraction < 1.0:
        raise ValueError("returned-fraction must lie in [0, 1)")
    if args.inflow_metallicity_zsun < 0.0:
        raise ValueError("inflow-metallicity-zsun must be non-negative")


def _dt_from_grid(cosmology: Cosmology, z_final: float, z_start_max: float, n_grid: int) -> float:
    from astropy.cosmology import FlatLambdaCDM

    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    t_start = float(astro.age(z_start_max).value)
    t_end = float(astro.age(z_final).value)
    return (t_end - t_start) / float(n_grid - 1)


def _summarize_by_step(
    *,
    z_grid: np.ndarray,
    metallicity_grid: np.ndarray,
    mask_grid: np.ndarray,
    z_plot_max: float,
    z_final: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_axis = np.nanmedian(np.asarray(z_grid, dtype=float), axis=0)
    values = np.asarray(metallicity_grid, dtype=float)
    mask = np.asarray(mask_grid, dtype=bool) & np.isfinite(values) & (values > 0.0)
    n_steps = values.shape[1]
    median = np.full(n_steps, np.nan, dtype=float)
    p16 = np.full(n_steps, np.nan, dtype=float)
    p84 = np.full(n_steps, np.nan, dtype=float)
    count = np.zeros(n_steps, dtype=np.int64)
    for step_index in range(n_steps):
        selected = values[mask[:, step_index], step_index]
        count[step_index] = int(selected.size)
        if selected.size == 0:
            continue
        p16[step_index], median[step_index], p84[step_index] = np.percentile(selected, [16.0, 50.0, 84.0])
    keep = (
        np.isfinite(z_axis)
        & (z_axis >= float(z_final))
        & (z_axis <= float(z_plot_max))
        & np.isfinite(median)
        & (count > 0)
    )
    return z_axis[keep], median[keep], p16[keep], p84[keep], count[keep]


def _compute_summaries(args: argparse.Namespace, parameters: RegulatorMetallicityParameters) -> list[RedshiftSummary]:
    cosmology = Cosmology()
    dt_gyr = _dt_from_grid(cosmology, float(args.z_final), float(args.z_start_max), int(args.n_grid))
    summaries: list[RedshiftSummary] = []
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
            cosmology=cosmology,
            enable_time_delay=bool(args.enable_time_delay),
        )
        n_halos = int(args.n_tracks)
        n_steps = int(histories.metadata["grid_size"])
        t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, n_steps)
        z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(n_halos, n_steps)
        mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, n_steps)
        sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, n_steps)
        active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, n_steps)
        starforming_grid = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
        result = compute_regulator_metallicity(
            t_grid_gyr=t_grid,
            z_grid=z_grid,
            mh_grid=mh_grid,
            sfr_grid=sfr_grid,
            active_grid=starforming_grid,
            cosmology=cosmology,
            parameters=parameters,
            random_seed=int(args.metallicity_random_seed + 1000 * mass_index),
        )
        z_axis, median, p16, p84, count = _summarize_by_step(
            z_grid=z_grid,
            metallicity_grid=result.gas_metallicity_zsun_grid,
            mask_grid=starforming_grid,
            z_plot_max=float(args.z_plot_max),
            z_final=float(args.z_final),
        )
        if z_axis.size == 0:
            raise RuntimeError(f"no positive regulator metallicity samples for logMh={float(logmh):g}")
        summaries.append(
            RedshiftSummary(
                logmh=float(logmh),
                z=z_axis,
                median=median,
                p16=p16,
                p84=p84,
                count=count,
            )
        )
    return summaries


def _plot_summaries(
    *,
    output_prefix: Path,
    summaries: list[RedshiftSummary],
    z_plot_max: float,
    z_final: float,
) -> tuple[Path, Path]:
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(summaries)))
    all_positive: list[np.ndarray] = []
    for color, summary in zip(colors, summaries, strict=True):
        label = rf"$\log M_h(z={z_final:g})={summary.logmh:g}$"
        ax.plot(summary.z, summary.median, lw=2.2, color=color, label=label)
        ax.fill_between(summary.z, summary.p16, summary.p84, color=color, alpha=0.18, lw=0.0)
        all_positive.append(summary.p16[np.isfinite(summary.p16) & (summary.p16 > 0.0)])
        all_positive.append(summary.p84[np.isfinite(summary.p84) & (summary.p84 > 0.0)])
    ax.axhline(GHZ2_ZSUN, color="#cc78bc", lw=1.5, ls=":", label=r"$Z=0.05Z_\odot$")
    positive = np.concatenate([item for item in all_positive if item.size > 0])
    if positive.size == 0:
        raise RuntimeError("no positive metallicity values available for log-scale plot")
    y_min = max(float(np.nanmin(positive)) * 0.6, 1.0e-5)
    y_max = max(float(np.nanmax(positive)) * 1.4, GHZ2_ZSUN * 1.5)
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(float(z_plot_max), float(z_final))
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$Z_{\rm gas}/Z_\odot$")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    ax.text(
        0.03,
        0.96,
        "fixed MAH/SFR; regulator parameters from best z=12.5 scan",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="0.35",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _write_csv(path: Path, summaries: list[RedshiftSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "logmh_zfinal",
                "redshift",
                "zgas_median_zsun",
                "zgas_p16_zsun",
                "zgas_p84_zsun",
                "oh12_median",
                "sample_count",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            oh12 = equivalent_oxygen_abundance_from_zsun(summary.median)
            for z_value, median, p16, p84, oh12_value, count in zip(
                summary.z,
                summary.median,
                summary.p16,
                summary.p84,
                oh12,
                summary.count,
                strict=True,
            ):
                writer.writerow(
                    {
                        "logmh_zfinal": summary.logmh,
                        "redshift": float(z_value),
                        "zgas_median_zsun": float(median),
                        "zgas_p16_zsun": float(p16),
                        "zgas_p84_zsun": float(p84),
                        "oh12_median": float(oh12_value),
                        "sample_count": int(count),
                    }
                )


def _write_summary(path: Path, args: argparse.Namespace, parameters: RegulatorMetallicityParameters, summaries: list[RedshiftSummary]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"z_final: {float(args.z_final):g}\n")
        handle.write(f"z_plot_max: {float(args.z_plot_max):g}\n")
        handle.write(f"log_masses: {' '.join(f'{float(item):g}' for item in args.log_masses)}\n")
        handle.write(f"n_tracks: {int(args.n_tracks)}\n")
        handle.write(f"n_grid: {int(args.n_grid)}\n")
        handle.write(f"enable_time_delay: {bool(args.enable_time_delay)}\n")
        handle.write("\nRegulator parameters:\n")
        for key, value in parameters.as_metadata().items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nFinal redshift medians:\n")
        for summary in summaries:
            nearest = int(np.argmin(np.abs(summary.z - float(args.z_final))))
            handle.write(
                f"logMh={summary.logmh:g}: z={summary.z[nearest]:.4g}, "
                f"Zgas={summary.median[nearest]:.6g} Zsun, "
                f"12+log(O/H)={float(equivalent_oxygen_abundance_from_zsun(summary.median[nearest])):.4g}\n"
            )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_prefix = _resolve_prefix(args.output_prefix, float(args.z_final))
    parameters = RegulatorMetallicityParameters(
        gas_fraction_norm=float(args.gas_fraction_norm),
        gas_fraction_mass_slope=float(args.gas_fraction_mass_slope),
        gas_fraction_redshift_slope=float(args.gas_fraction_redshift_slope),
        returned_fraction=float(args.returned_fraction),
        metal_yield=float(args.metal_yield),
        inflow_metallicity_zsun=float(args.inflow_metallicity_zsun),
        metal_loading_norm=float(args.metal_loading_norm),
        metal_loading_mass_slope=float(args.metal_loading_mass_slope),
        metal_loading_redshift_slope=float(args.metal_loading_redshift_slope),
    )
    summaries = _compute_summaries(args, parameters)
    png_path, pdf_path = _plot_summaries(
        output_prefix=output_prefix,
        summaries=summaries,
        z_plot_max=float(args.z_plot_max),
        z_final=float(args.z_final),
    )
    csv_path = (PROJECT_ROOT / "data_save" / output_prefix.name).with_suffix(".csv")
    _write_csv(csv_path, summaries)
    summary_path = output_prefix.with_suffix(".txt")
    _write_summary(summary_path, args, parameters, summaries)
    print(f"saved: {png_path} {pdf_path}", flush=True)
    print(f"csv: {csv_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
