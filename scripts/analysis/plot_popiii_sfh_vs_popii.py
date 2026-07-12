#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.mah.models import Cosmology
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import (
    POPIII_UPPER_MASS_MODE_ATOMIC,
    PopIIISFRParameters,
    compute_popiii_sfr_visbal2015_from_grids,
)
from auroralf.uvlf import run_halo_uv_pipeline


DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "popiii_sfh_vs_popii_brightest_heii_track"
DEFAULT_SLIDE_OUTPUT = (
    PROJECT_ROOT / "slides" / "group_meeting_popiii_20260622" / "assets" / "popiii_sfh_vs_popii_slide.pdf"
)
DEFAULT_EXTREME_POPIII_UV_SSP_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "schaerer2010_pop3"
    / "pop3_ge0_sal_500_050_is4.25"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the model-forward Pop III SFH against the Pop II SFR for one halo track."
    )
    parser.add_argument("--z", type=float, default=10.583)
    parser.add_argument("--logMh", type=float, default=8.272727272727273)
    parser.add_argument("--n-tracks", type=int, default=8)
    parser.add_argument("--track-index", type=int, default=4)
    parser.add_argument("--n-grid", type=int, default=960)
    parser.add_argument("--random-seed", type=int, default=111)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--lookback-max-myr", type=float, default=100.0)
    parser.add_argument("--young-window-myr", type=float, default=3.0)
    parser.add_argument("--heii-window-myr", type=float, default=10.0)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--popiii-epsilon-star", type=float, default=1.0e-3)
    parser.add_argument("--popiii-mp", type=float, default=1.0e7)
    parser.add_argument("--popiii-alpha-star", type=float, default=0.0)
    parser.add_argument("--popiii-beta-star", type=float, default=0.0)
    parser.add_argument("--include-visbal2015-sfh", action="store_true")
    parser.add_argument("--visbal-fstar", type=float, default=0.1)
    parser.add_argument("--eta-duty-values", type=str, default="1.0,0.1,0.01")
    parser.add_argument("--popiii-uv-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_UV_SSP_FILE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--slide-output", type=Path, default=DEFAULT_SLIDE_OUTPUT)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.z <= 0.0:
        raise ValueError("--z must be positive")
    if args.n_tracks <= 0:
        raise ValueError("--n-tracks must be positive")
    if not 0 <= args.track_index < args.n_tracks:
        raise ValueError("--track-index must satisfy 0 <= track-index < n-tracks")
    if args.n_grid <= 2:
        raise ValueError("--n-grid must be greater than 2")
    if args.z_start_max <= args.z:
        raise ValueError("--z-start-max must be greater than --z")
    if args.lookback_max_myr <= 0.0:
        raise ValueError("--lookback-max-myr must be positive")
    if args.young_window_myr <= 0.0:
        raise ValueError("--young-window-myr must be positive")
    if args.heii_window_myr <= 0.0:
        raise ValueError("--heii-window-myr must be positive")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")
    if not 0.0 <= args.popiii_epsilon_star <= 1.0:
        raise ValueError("--popiii-epsilon-star must lie in [0, 1]")
    if args.popiii_mp <= 0.0:
        raise ValueError("--popiii-mp must be positive")
    if not 0.0 <= args.visbal_fstar <= 1.0:
        raise ValueError("--visbal-fstar must lie in [0, 1]")
    parse_eta_duty_values(args.eta_duty_values)


def parse_eta_duty_values(raw_values: str) -> np.ndarray:
    if not raw_values.strip():
        raise ValueError("--eta-duty-values must contain at least one value")
    values = np.array([float(value.strip()) for value in raw_values.split(",")], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("--eta-duty-values must contain at least one value")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("--eta-duty-values must contain finite positive values")
    return values


def eta_duty_column_suffix(eta_duty: float) -> str:
    return f"{eta_duty:g}".replace("-", "m").replace(".", "p")


def positive_for_log_plot(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array[~np.isfinite(array) | (array <= 0.0)] = np.nan
    return array


def integrate_mass_within_lookback(
    *,
    lookback_myr: np.ndarray,
    sfr_msun_yr: np.ndarray,
    max_lookback_myr: float,
) -> float:
    lookback = np.asarray(lookback_myr, dtype=float)
    sfr = np.asarray(sfr_msun_yr, dtype=float)
    if lookback.ndim != 1 or sfr.ndim != 1:
        raise ValueError("lookback_myr and sfr_msun_yr must be 1D arrays")
    if lookback.size != sfr.size:
        raise ValueError("lookback_myr and sfr_msun_yr must have the same length")
    if not np.all(np.isfinite(lookback)) or np.any(lookback < 0.0):
        raise ValueError("lookback_myr must contain finite non-negative values")
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise ValueError("sfr_msun_yr must contain finite non-negative values")
    if max_lookback_myr <= 0.0:
        raise ValueError("max_lookback_myr must be positive")

    order = np.argsort(lookback, kind="stable")
    lookback_sorted = lookback[order]
    sfr_sorted = sfr[order]
    if lookback_sorted[0] > 1.0e-8:
        raise ValueError("lookback_myr must include the observation time at lookback=0")
    upper = min(float(max_lookback_myr), float(lookback_sorted[-1]))
    if upper <= 0.0:
        return 0.0

    interior = lookback_sorted[(lookback_sorted > 0.0) & (lookback_sorted < upper)]
    lookback_used = np.unique(np.concatenate((np.array([0.0]), interior, np.array([upper]))))
    sfr_used = np.interp(lookback_used, lookback_sorted, sfr_sorted)
    return float(np.trapezoid(sfr_used, x=lookback_used * 1.0e6))


def cumulative_mass_since_observation(
    *,
    lookback_myr: np.ndarray,
    sfr_msun_yr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookback = np.asarray(lookback_myr, dtype=float)
    sfr = np.asarray(sfr_msun_yr, dtype=float)
    if lookback.shape != sfr.shape:
        raise ValueError("lookback_myr and sfr_msun_yr must have the same shape")
    order = np.argsort(lookback, kind="stable")
    x = lookback[order]
    y = sfr[order]
    cumulative = np.zeros_like(x, dtype=float)
    if x.size > 1:
        dx_yr = np.diff(x) * 1.0e6
        cumulative[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx_yr)
    return x, cumulative


def _write_rows(path: Path, rows: list[dict[str, float | int | bool]], extra_fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lookback_myr",
        "t_gyr",
        "z",
        "Mh_msun",
        "dMh_dt_raw_msun_gyr",
        "sfr_popii_msun_yr",
        "sfr_popiii_msun_yr",
        "popiii_active",
        "fstar_popiii",
        "popiii_duty_cycle",
        "popiii_lower_mass_msun",
        "popiii_upper_mass_msun",
        "cumulative_popii_mass_msun",
        "cumulative_popiii_mass_msun",
    ]
    if extra_fieldnames is not None:
        fieldnames.extend(extra_fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    output_prefix: Path,
    slide_output: Path,
    lookback_myr: np.ndarray,
    sfr_popii: np.ndarray,
    sfr_popiii: np.ndarray,
    cumulative_lookback_myr: np.ndarray,
    cumulative_popii: np.ndarray,
    cumulative_popiii: np.ndarray,
    visbal_sfr_by_column: dict[str, np.ndarray] | None,
    visbal_eta_by_column: dict[str, float],
    mh_over_mcool_visbal: np.ndarray | None,
    summary: dict[str, float],
) -> None:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), constrained_layout=True)
    ax_sfr, ax_right = axes

    show_recent_windows = not bool(visbal_sfr_by_column)
    for ax in axes:
        if show_recent_windows:
            ax.axvspan(0.0, summary["heii_window_myr"], color="#f59e0b", alpha=0.10, lw=0)
            ax.axvspan(0.0, summary["young_window_myr"], color="#10b981", alpha=0.12, lw=0)
        ax.set_xlim(summary["lookback_max_myr"], 0.0)
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.08)

    ax_sfr.plot(lookback_myr, positive_for_log_plot(sfr_popii), color="#2563eb", lw=2.0, label="Pop II SFR")
    ax_sfr.plot(lookback_myr, positive_for_log_plot(sfr_popiii), color="#059669", lw=2.0, label="Pop III SFR")
    visbal_positive: list[np.ndarray] = []
    if visbal_sfr_by_column:
        visbal_colors = ["#d97706", "#9333ea", "#dc2626", "#0f766e", "#475569"]
        for index, (column, sfr_visbal) in enumerate(visbal_sfr_by_column.items()):
            color = visbal_colors[index % len(visbal_colors)]
            eta_value = visbal_eta_by_column[column]
            ax_sfr.plot(
                lookback_myr,
                positive_for_log_plot(sfr_visbal),
                color=color,
                lw=1.7,
                ls="--",
                label=rf"V15 $\eta_{{\rm duty}}={eta_value:g}$",
            )
            visbal_positive.append(sfr_visbal[sfr_visbal > 0.0])
    ax_sfr.set_yscale("log")
    positive_sfr_inputs = [sfr_popii[sfr_popii > 0.0], sfr_popiii[sfr_popiii > 0.0], *visbal_positive]
    positive_sfr_chunks = [values for values in positive_sfr_inputs if values.size]
    positive_sfr = np.concatenate(positive_sfr_chunks) if positive_sfr_chunks else np.array([], dtype=float)
    if positive_sfr.size:
        ax_sfr.set_ylim(max(np.min(positive_sfr) * 0.5, 1.0e-8), np.max(positive_sfr) * 2.5)
    ax_sfr.set_xlabel(r"lookback before $z_{\rm obs}$ [Myr]")
    ax_sfr.set_ylabel(r"SFR [$M_\odot\,{\rm yr}^{-1}$]")
    ax_sfr.set_title("instantaneous SFH")
    ax_sfr.legend(loc="center left", frameon=False)

    if mh_over_mcool_visbal is None:
        ax_right.plot(
            cumulative_lookback_myr,
            positive_for_log_plot(cumulative_popii),
            color="#2563eb",
            lw=2.0,
            label=r"Pop II $M_\star(<t_{\rm lb})$",
        )
        ax_right.plot(
            cumulative_lookback_myr,
            positive_for_log_plot(cumulative_popiii),
            color="#059669",
            lw=2.0,
            label=r"Pop III $M_\star(<t_{\rm lb})$",
        )
        ax_right.set_yscale("log")
        positive_mass = np.concatenate((cumulative_popii[cumulative_popii > 0.0], cumulative_popiii[cumulative_popiii > 0.0]))
        if positive_mass.size:
            ax_right.set_ylim(max(np.min(positive_mass) * 0.5, 1.0e0), np.max(positive_mass) * 2.5)
        ax_right.set_xlabel(r"lookback before $z_{\rm obs}$ [Myr]")
        ax_right.set_ylabel(r"formed stellar mass [$M_\odot$]")
        ax_right.set_title("mass formed in recent window")
        ax_right.legend(loc="lower left", frameon=False)
    else:
        ax_right.axhspan(1.0, 2.0, color="#f59e0b", alpha=0.18, lw=0, label=r"V15 $1-2M_{\rm cool}$")
        ax_right.plot(lookback_myr, positive_for_log_plot(mh_over_mcool_visbal), color="#111827", lw=2.0)
        ax_right.set_yscale("log")
        positive_ratio = mh_over_mcool_visbal[np.isfinite(mh_over_mcool_visbal) & (mh_over_mcool_visbal > 0.0)]
        if positive_ratio.size:
            lower = min(np.min(positive_ratio) * 0.75, 0.9)
            upper = max(np.max(positive_ratio) * 1.35, 2.2)
            ax_right.set_ylim(lower, upper)
        ax_right.set_xlabel(r"lookback before $z_{\rm obs}$ [Myr]")
        ax_right.set_ylabel(r"$M_h/M_{\rm cool}^{\rm V15}$")
        ax_right.set_title("V15 atomic-cooling window")
        ax_right.legend(loc="lower left", frameon=False)
    ax_right.text(
        0.04,
        0.95,
        rf"$z_{{\rm obs}}={summary['z_obs']:.3f}$" + "\n"
        + rf"$\log M_h={summary['logmh']:.2f}$, track {int(summary['track_index'])}" + "\n"
        + rf"${{\rm SFR}}_{{III}}(0)={summary['sfr_popiii_final']:.2e}$" + "\n"
        + rf"$M_{{III}}(<3\,{{\rm Myr}})={summary['mass_popiii_young']:.2e}M_\odot$",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )

    caption = "Pop III SFH is forward-modeled from halo assembly and the Pop III SFR prescription; it is not inferred from HeII."
    if visbal_sfr_by_column:
        caption = (
            rf"V15 curves project Eq. 10 onto one halo history with $f_\star={summary['visbal_fstar']:.0e}$; "
            "the original mean signal requires an HMF integral over halos."
        )
    fig.text(
        0.5,
        -0.03,
        caption,
        ha="center",
        va="top",
        fontsize=7.4,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500, bbox_inches="tight")
    slide_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(slide_output, dpi=500, bbox_inches="tight")


def main() -> None:
    args = _parse_args()
    cosmology = Cosmology()
    _validate_args(args)
    eta_duty_values = parse_eta_duty_values(args.eta_duty_values)
    output_prefix = _resolve_project_path(args.output_prefix)
    slide_output = _resolve_project_path(args.slide_output)
    popiii_uv_ssp_file = _resolve_project_path(args.popiii_uv_ssp_file)
    if not popiii_uv_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III UV SSP file not found: {popiii_uv_ssp_file}")

    popiii_params = PopIIISFRParameters(
        epsilon_star=float(args.popiii_epsilon_star),
        pivot_mass_msun=float(args.popiii_mp),
        alpha_star=float(args.popiii_alpha_star),
        beta_star=float(args.popiii_beta_star),
        lw_background_j21=float(args.lw_background_j21),
        upper_mass_mode=POPIII_UPPER_MASS_MODE_ATOMIC,
        upper_mass_msun=None,
    )
    mh_final = 10.0 ** float(args.logMh)
    result = run_halo_uv_pipeline(
        n_tracks=int(args.n_tracks),
        z_final=float(args.z),
        Mh_final=mh_final,
        cosmology=cosmology,
        random_seeds=derive_pipeline_random_seeds(
            int(args.random_seed),
            redshift=float(args.z),
            mass_index=0,
        ),
        z_start_max=float(args.z_start_max),
        n_grid=int(args.n_grid),
        workers=1,
        enable_time_delay=True,
        enable_popiii=True,
        popiii_sfr_parameters=popiii_params,
        popiii_ssp_file=str(popiii_uv_ssp_file),
        imf_mode="canonical",
    )
    steps_per_halo = int(result.redshift_grid.size)
    track_index = int(args.track_index)

    def track(name: str) -> np.ndarray:
        return np.asarray(result.sfr_tracks[name], dtype=float).reshape(int(args.n_tracks), steps_per_halo)[track_index]

    t_gyr = track("t_gyr")
    z_grid = track("z")
    mh_grid = track("Mh")
    dmhdt_raw_grid = track("dMh_dt_raw")
    sfr_popii = track("SFR")
    sfr_popiii = track("SFR_popiii")
    fstar_popiii = track("fstar_popiii")
    duty_popiii = track("popiii_duty_cycle")
    lower_mass = track("popiii_lower_mass_msun")
    upper_mass = track("popiii_upper_mass_msun")
    active_track = np.asarray(result.sfr_tracks["active_flag"], dtype=bool).reshape(int(args.n_tracks), steps_per_halo)[
        track_index
    ]
    popiii_active = np.asarray(result.popiii_source_grid, dtype=bool).reshape(int(args.n_tracks), steps_per_halo)[track_index]
    if np.any(np.diff(t_gyr) <= 0.0):
        raise ValueError("selected track time grid must be strictly increasing")

    visbal_sfr_by_column: dict[str, np.ndarray] = {}
    visbal_eta_by_column: dict[str, float] = {}
    mcool_visbal = np.full_like(mh_grid, np.nan, dtype=float)
    mh_over_mcool_visbal = np.full_like(mh_grid, np.nan, dtype=float)
    visbal_atomic_window = np.zeros_like(active_track, dtype=bool)
    if args.include_visbal2015_sfh:
        for eta_duty in eta_duty_values:
            visbal_result = compute_popiii_sfr_visbal2015_from_grids(
                mh_grid=mh_grid[None, :],
                z_grid=z_grid[None, :],
                active_grid=active_track[None, :],
                fstar=float(args.visbal_fstar),
                eta_duty=float(eta_duty),
                cosmology=cosmology,
            )
            column = f"sfr_popiii_visbal_eta_{eta_duty_column_suffix(float(eta_duty))}"
            visbal_sfr_by_column[column] = np.asarray(visbal_result.sfr_grid[0], dtype=float)
            visbal_eta_by_column[column] = float(eta_duty)
            mcool_visbal = np.asarray(visbal_result.mcool_msun_grid[0], dtype=float)
            mh_over_mcool_visbal = np.asarray(visbal_result.mh_over_mcool_grid[0], dtype=float)
            visbal_atomic_window = np.asarray(visbal_result.atomic_window_grid[0], dtype=bool)

    lookback_myr = (float(t_gyr[-1]) - t_gyr) * 1.0e3
    plot_mask = lookback_myr <= float(args.lookback_max_myr)
    if not np.any(plot_mask):
        raise ValueError("selected lookback window contains no time samples")

    cum_lookback, cum_popii = cumulative_mass_since_observation(lookback_myr=lookback_myr, sfr_msun_yr=sfr_popii)
    _, cum_popiii = cumulative_mass_since_observation(lookback_myr=lookback_myr, sfr_msun_yr=sfr_popiii)
    mass_popii_young = integrate_mass_within_lookback(
        lookback_myr=lookback_myr,
        sfr_msun_yr=sfr_popii,
        max_lookback_myr=float(args.young_window_myr),
    )
    mass_popiii_young = integrate_mass_within_lookback(
        lookback_myr=lookback_myr,
        sfr_msun_yr=sfr_popiii,
        max_lookback_myr=float(args.young_window_myr),
    )
    mass_popii_heii = integrate_mass_within_lookback(
        lookback_myr=lookback_myr,
        sfr_msun_yr=sfr_popii,
        max_lookback_myr=float(args.heii_window_myr),
    )
    mass_popiii_heii = integrate_mass_within_lookback(
        lookback_myr=lookback_myr,
        sfr_msun_yr=sfr_popiii,
        max_lookback_myr=float(args.heii_window_myr),
    )

    cumulative_popii_by_row = np.interp(lookback_myr, cum_lookback, cum_popii)
    cumulative_popiii_by_row = np.interp(lookback_myr, cum_lookback, cum_popiii)
    rows: list[dict[str, float | int | bool]] = []
    for index in range(t_gyr.size):
        rows.append(
            {
                "lookback_myr": float(lookback_myr[index]),
                "t_gyr": float(t_gyr[index]),
                "z": float(z_grid[index]),
                "Mh_msun": float(mh_grid[index]),
                "dMh_dt_raw_msun_gyr": float(dmhdt_raw_grid[index]),
                "sfr_popii_msun_yr": float(sfr_popii[index]),
                "sfr_popiii_msun_yr": float(sfr_popiii[index]),
                "popiii_active": bool(popiii_active[index]),
                "fstar_popiii": float(fstar_popiii[index]),
                "popiii_duty_cycle": float(duty_popiii[index]),
                "popiii_lower_mass_msun": float(lower_mass[index]),
                "popiii_upper_mass_msun": float(upper_mass[index]),
                "cumulative_popii_mass_msun": float(cumulative_popii_by_row[index]),
                "cumulative_popiii_mass_msun": float(cumulative_popiii_by_row[index]),
            }
        )
        if args.include_visbal2015_sfh:
            for column, sfr_visbal in visbal_sfr_by_column.items():
                rows[-1][column] = float(sfr_visbal[index])
            rows[-1]["mcool_visbal_msun"] = float(mcool_visbal[index])
            rows[-1]["mh_over_mcool_visbal"] = float(mh_over_mcool_visbal[index])
            rows[-1]["visbal_atomic_window"] = bool(visbal_atomic_window[index])
    extra_fieldnames: list[str] | None = None
    if args.include_visbal2015_sfh:
        extra_fieldnames = [
            *visbal_sfr_by_column.keys(),
            "mcool_visbal_msun",
            "mh_over_mcool_visbal",
            "visbal_atomic_window",
        ]
    _write_rows(output_prefix.with_suffix(".csv"), rows, extra_fieldnames=extra_fieldnames)

    summary = {
        "z_obs": float(args.z),
        "logmh": float(args.logMh),
        "track_index": float(track_index),
        "lookback_max_myr": float(args.lookback_max_myr),
        "young_window_myr": float(args.young_window_myr),
        "heii_window_myr": float(args.heii_window_myr),
        "sfr_popii_final": float(sfr_popii[-1]),
        "sfr_popiii_final": float(sfr_popiii[-1]),
        "mass_popii_young": float(mass_popii_young),
        "mass_popiii_young": float(mass_popiii_young),
        "mass_popii_heii": float(mass_popii_heii),
        "mass_popiii_heii": float(mass_popiii_heii),
        "include_visbal2015_sfh": float(bool(args.include_visbal2015_sfh)),
        "visbal_fstar": float(args.visbal_fstar),
    }
    visbal_npz_fields = {
        f"{column}_grid": values for column, values in visbal_sfr_by_column.items()
    }
    if args.include_visbal2015_sfh:
        visbal_npz_fields.update(
            {
                "visbal_eta_duty_values": eta_duty_values,
                "mcool_visbal_msun_grid": mcool_visbal,
                "mh_over_mcool_visbal_grid": mh_over_mcool_visbal,
                "visbal_atomic_window_grid": visbal_atomic_window,
            }
        )
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        **{name: np.asarray([value]) for name, value in summary.items()},
        **visbal_npz_fields,
        popiii_sfr_parameters=np.asarray([str(popiii_params.as_metadata())]),
        popiii_uv_ssp_file=np.asarray([str(popiii_uv_ssp_file)]),
    )
    _plot(
        output_prefix=output_prefix,
        slide_output=slide_output,
        lookback_myr=lookback_myr[plot_mask],
        sfr_popii=sfr_popii[plot_mask],
        sfr_popiii=sfr_popiii[plot_mask],
        cumulative_lookback_myr=cum_lookback[cum_lookback <= float(args.lookback_max_myr)],
        cumulative_popii=cum_popii[cum_lookback <= float(args.lookback_max_myr)],
        cumulative_popiii=cum_popiii[cum_lookback <= float(args.lookback_max_myr)],
        visbal_sfr_by_column={
            column: values[plot_mask] for column, values in visbal_sfr_by_column.items()
        }
        if args.include_visbal2015_sfh
        else None,
        visbal_eta_by_column=visbal_eta_by_column,
        mh_over_mcool_visbal=mh_over_mcool_visbal[plot_mask] if args.include_visbal2015_sfh else None,
        summary=summary,
    )
    print(f"wrote {output_prefix.with_suffix('.csv')}")
    print(f"wrote {output_prefix.with_suffix('.npz')}")
    print(f"wrote {output_prefix.with_suffix('.pdf')}")
    print(f"wrote {output_prefix.with_suffix('.png')}")
    print(f"wrote {slide_output}")
    print(f"final PopII SFR={sfr_popii[-1]:.6e} Msun/yr")
    print(f"final PopIII SFR={sfr_popiii[-1]:.6e} Msun/yr")
    print(f"PopII mass formed in last {args.young_window_myr:g} Myr={mass_popii_young:.6e} Msun")
    print(f"PopIII mass formed in last {args.young_window_myr:g} Myr={mass_popiii_young:.6e} Msun")
    print(f"PopII mass formed in last {args.heii_window_myr:g} Myr={mass_popii_heii:.6e} Msun")
    print(f"PopIII mass formed in last {args.heii_window_myr:g} Myr={mass_popiii_heii:.6e} Msun")
    if args.include_visbal2015_sfh:
        for column, sfr_visbal in visbal_sfr_by_column.items():
            print(f"{column} final SFR={sfr_visbal[-1]:.6e} Msun/yr")


if __name__ == "__main__":
    main()
