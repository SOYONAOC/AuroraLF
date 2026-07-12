#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_popiii_lw_minimum_mass_msun
from auroralf.mah import Cosmology
from auroralf.sfr import POPIII_UPPER_MASS_MODE_ATOMIC, POPIII_UPPER_MASS_MODE_FIXED, PopIIISFRParameters
from auroralf.uvlf import sample_uvlf_from_hmf
from scripts.plot.plot_group_meeting_popiii_components_uvlf import DEFAULT_EXTREME_POPIII_SSP_FILE, PROJECT_ROOT
from scripts.plot.plot_popiii_mup_pisn_proxy import _pisn_events_per_stellar_mass


DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "popiii_mup_pisn_rate_from_sfr"
DEFAULT_Z_VALUES = (6.0, 8.0, 10.0, 12.5, 14.5)
SCENARIO_LABELS = {
    "current": r"current $M_{\rm up,halo}=M_{\rm vir}(10^4{\rm K})$",
    "fixed_mup1e10": r"fixed $M_{\rm up,halo}=10^{10}M_\odot$",
}
SCENARIO_COLORS = {
    "current": "#1F5C8B",
    "fixed_mup1e10": "#202020",
}


@dataclass(frozen=True)
class Scenario:
    key: str
    upper_mass_mode: str
    upper_mass_msun: float | None


@dataclass(frozen=True)
class RatePoint:
    z: float
    scenario: str
    rho_sfr_popiii: float
    pisn_rate_density: float
    popiii_source_fraction: float
    popiii_upper_mass_msun: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot n_dot_PISN(z)=rho_SFR,III(z) eta_PISN for Pop III halo Mup choices."
    )
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--N-mass", type=int, default=720)
    parser.add_argument("--n-tracks", type=int, default=24)
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=14501)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--fixed-upper-mass-msun", type=float, default=1.0e10)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--muv-min", type=float, default=-26.0)
    parser.add_argument("--muv-max", type=float, default=1.5)
    parser.add_argument("--muv-bin-width", type=float, default=0.5)
    parser.add_argument("--popiii-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_SSP_FILE)
    parser.add_argument("--imf-slope", type=float, default=2.35)
    parser.add_argument("--imf-min-msun", type=float, default=50.0)
    parser.add_argument("--imf-max-msun", type=float, default=500.0)
    parser.add_argument("--pisn-min-msun", type=float, default=140.0)
    parser.add_argument("--pisn-max-msun", type=float, default=260.0)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Read an existing rate CSV and only redraw the figure.",
    )
    parser.add_argument(
        "--no-ratio-panel",
        action="store_true",
        help="Omit the fixed/current ratio panel from the output figure.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.z_values) == 0:
        raise ValueError("--z-values must contain at least one redshift")
    if np.any(~np.isfinite(np.asarray(args.z_values, dtype=float))) or np.any(np.asarray(args.z_values) < 0.0):
        raise ValueError("--z-values must be finite and non-negative")
    if args.N_mass <= 0:
        raise ValueError("--N-mass must be positive")
    if args.n_tracks <= 0:
        raise ValueError("--n-tracks must be positive")
    if args.n_grid <= 1:
        raise ValueError("--n-grid must be greater than 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.logM_max <= 0.0:
        raise ValueError("--logM-max must be positive")
    if args.fixed_upper_mass_msun <= 0.0 or not np.isfinite(args.fixed_upper_mass_msun):
        raise ValueError("--fixed-upper-mass-msun must be finite and positive")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")
    if args.muv_max <= args.muv_min:
        raise ValueError("--muv-max must be larger than --muv-min")
    if args.muv_bin_width <= 0.0:
        raise ValueError("--muv-bin-width must be positive")


def _scenarios(args: argparse.Namespace) -> tuple[Scenario, Scenario]:
    return (
        Scenario(
            key="current",
            upper_mass_mode=POPIII_UPPER_MASS_MODE_ATOMIC,
            upper_mass_msun=None,
        ),
        Scenario(
            key="fixed_mup1e10",
            upper_mass_mode=POPIII_UPPER_MASS_MODE_FIXED,
            upper_mass_msun=float(args.fixed_upper_mass_msun),
        ),
    )


def _run_point(
    *,
    cosmology: Cosmology,
    args: argparse.Namespace,
    z: float,
    scenario: Scenario,
    eta_pisn_per_msun: float,
    popiii_ssp_file: Path,
) -> RatePoint:
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(float(z), lw_background_j21=float(args.lw_background_j21))
    )
    logm_min = float(np.log10(popiii_minimum_mass_msun))
    if args.logM_max <= logm_min:
        raise ValueError(f"--logM-max must exceed log10(M_popIII_min) at z={z:g}")
    bin_edges = np.arange(args.muv_min, args.muv_max + args.muv_bin_width, args.muv_bin_width)
    if bin_edges.size < 2:
        raise RuntimeError("MUV bin construction produced fewer than two bin edges")
    params = PopIIISFRParameters(
        lw_background_j21=float(args.lw_background_j21),
        upper_mass_mode=scenario.upper_mass_mode,
        upper_mass_msun=scenario.upper_mass_msun,
    )
    result = sample_uvlf_from_hmf(
        z_obs=float(z),
        cosmology=cosmology,
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        base_seed=int(args.random_seed),
        quantity="Muv",
        bins=bin_edges,
        logM_min=logm_min,
        logM_max=float(args.logM_max),
        z_start_max=50.0,
        n_grid=int(args.n_grid),
        sampler="mcbride",
        enable_time_delay=True,
        pipeline_workers=int(args.workers),
        enable_popiii=True,
        popiii_sfr_parameters=params,
        popiii_ssp_file=str(popiii_ssp_file),
        imf_mode="canonical",
    )
    popiii_sfr = np.asarray(result.samples["popiii_sfr"], dtype=float)
    sample_weight = np.asarray(result.samples["sample_weight"], dtype=float)
    if popiii_sfr.shape != sample_weight.shape:
        raise RuntimeError("sample popiii_sfr and sample_weight shapes differ")
    finite = np.isfinite(popiii_sfr) & np.isfinite(sample_weight) & (sample_weight > 0.0)
    if not np.any(finite):
        raise RuntimeError(f"no finite Pop III SFR samples at z={z:g}, scenario={scenario.key}")
    rho_sfr = float(np.sum(popiii_sfr[finite] * sample_weight[finite]))
    metadata_sfrd = float(result.metadata["popiii_sfrd_msun_yr_mpc3"])
    if not np.isclose(rho_sfr, metadata_sfrd, rtol=1.0e-10, atol=1.0e-30):
        raise RuntimeError(
            f"sample Pop III SFRD {rho_sfr:.8e} disagrees with metadata {metadata_sfrd:.8e}"
        )
    upper_mass = (
        float(np.asarray(result.metadata["atomic_cooling_mass_msun"]).reshape(-1)[0])
        if scenario.upper_mass_mode == POPIII_UPPER_MASS_MODE_ATOMIC
        else float(scenario.upper_mass_msun)
    )
    return RatePoint(
        z=float(z),
        scenario=scenario.key,
        rho_sfr_popiii=rho_sfr,
        pisn_rate_density=rho_sfr * float(eta_pisn_per_msun),
        popiii_source_fraction=float(result.metadata["popiii_source_fraction"]),
        popiii_upper_mass_msun=upper_mass,
    )


def _write_csv(path: Path, points: list[RatePoint], *, eta_pisn_per_msun: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"eta_pisn_per_msun={eta_pisn_per_msun:.10e}"])
        writer.writerow(
            [
                "scenario",
                "z",
                "rho_sfr_popiii_msun_yr^-1_Mpc^-3",
                "pisn_rate_density_yr^-1_Mpc^-3",
                "popiii_source_fraction",
                "popiii_upper_mass_msun",
            ]
        )
        for point in sorted(points, key=lambda item: (item.scenario, item.z)):
            writer.writerow(
                [
                    point.scenario,
                    f"{point.z:.8e}",
                    f"{point.rho_sfr_popiii:.8e}",
                    f"{point.pisn_rate_density:.8e}",
                    f"{point.popiii_source_fraction:.8e}",
                    f"{point.popiii_upper_mass_msun:.8e}",
                ]
            )


def _read_csv(path: Path) -> tuple[float, list[RatePoint]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            eta_row = next(reader)
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"rate CSV is missing required header rows: {path}") from exc
        if len(eta_row) != 1 or not eta_row[0].startswith("eta_pisn_per_msun="):
            raise ValueError(f"rate CSV first row must contain eta_pisn_per_msun: {path}")
        eta_pisn_per_msun = float(eta_row[0].split("=", maxsplit=1)[1])
        required_header = [
            "scenario",
            "z",
            "rho_sfr_popiii_msun_yr^-1_Mpc^-3",
            "pisn_rate_density_yr^-1_Mpc^-3",
            "popiii_source_fraction",
            "popiii_upper_mass_msun",
        ]
        if header != required_header:
            raise ValueError(f"unexpected rate CSV header in {path}: {header}")
        points = [
            RatePoint(
                z=float(row[1]),
                scenario=row[0],
                rho_sfr_popiii=float(row[2]),
                pisn_rate_density=float(row[3]),
                popiii_source_fraction=float(row[4]),
                popiii_upper_mass_msun=float(row[5]),
            )
            for row in reader
        ]
    if len(points) == 0:
        raise ValueError(f"rate CSV contains no data rows: {path}")
    return eta_pisn_per_msun, points


def _series(points: list[RatePoint], scenario: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted([point for point in points if point.scenario == scenario], key=lambda item: item.z)
    if len(selected) == 0:
        raise RuntimeError(f"no points available for scenario {scenario}")
    return np.asarray([point.z for point in selected], dtype=float), np.asarray(
        [getattr(point, field) for point in selected],
        dtype=float,
    )


def _positive_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    positive = np.concatenate(values)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise RuntimeError("no positive values available for log y-axis")
    return float(np.min(positive) * 0.72), float(np.max(positive) * 1.55)


def _draw_figure(
    path: Path,
    points: list[RatePoint],
    *,
    eta_pisn_per_msun: float,
    include_ratio_panel: bool = True,
) -> None:
    plt.style.use("apj")
    ncols = 3 if include_ratio_panel else 2
    fig_width = 13.6 if include_ratio_panel else 9.4
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.9), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    ax_sfrd, ax_rate = axes[:2]

    sfrd_values = []
    rate_values = []
    for scenario in SCENARIO_LABELS:
        z, sfrd = _series(points, scenario, "rho_sfr_popiii")
        _, rate = _series(points, scenario, "pisn_rate_density")
        sfrd_values.append(sfrd)
        rate_values.append(rate)
        ax_sfrd.plot(
            z,
            sfrd,
            color=SCENARIO_COLORS[scenario],
            marker="o",
            linewidth=2.45,
            label=SCENARIO_LABELS[scenario],
        )
        ax_rate.plot(
            z,
            rate,
            color=SCENARIO_COLORS[scenario],
            marker="o",
            linewidth=2.45,
            label=SCENARIO_LABELS[scenario],
        )

    if include_ratio_panel:
        ax_ratio = axes[2]
        z_current, rate_current = _series(points, "current", "pisn_rate_density")
        z_fixed, rate_fixed = _series(points, "fixed_mup1e10", "pisn_rate_density")
        if not np.allclose(z_current, z_fixed, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("scenario redshift grids differ")
        ratio = np.divide(rate_fixed, rate_current, out=np.full_like(rate_fixed, np.nan), where=rate_current > 0.0)
        ax_ratio.plot(z_current, ratio, color="#8C5FBF", marker="D", linewidth=2.45)
        ax_ratio.axhline(1.0, color="0.25", linewidth=1.1, linestyle=":")

    ax_sfrd.set_yscale("log")
    ax_rate.set_yscale("log")
    ax_sfrd.set_ylim(*_positive_ylim(sfrd_values))
    ax_rate.set_ylim(*_positive_ylim(rate_values))
    if include_ratio_panel:
        ax_ratio.set_yscale("log")
        ratio_positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if ratio_positive.size == 0:
            raise RuntimeError("no positive fixed/current ratio values")
        ax_ratio.set_ylim(float(np.min(ratio_positive) * 0.75), float(np.max(ratio_positive) * 1.35))

    for ax in axes:
        ax.set_xlim(5.7, 14.8)
        ax.set_xlabel(r"$z$")
        ax.grid(True, which="major", color="#C8D2DF", linewidth=0.72, alpha=0.85)
        ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.42, alpha=0.70)
    ax_sfrd.set_ylabel(r"$\rho_{\rm SFR,III}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
    ax_rate.set_ylabel(r"$\dot n_{\rm PISN}=\rho_{\rm SFR,III}\eta_{\rm PISN}$ [yr$^{-1}$ Mpc$^{-3}$]")
    if include_ratio_panel:
        ax_ratio.set_ylabel(r"fixed/current")
    ax_sfrd.legend(loc="best", frameon=True, fontsize=10.2)
    ax_rate.text(
        0.04,
        0.94,
        rf"$\eta_{{\rm PISN}}={eta_pisn_per_msun:.2e}\,M_\odot^{{-1}}$",
        transform=ax_rate.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        color="#1F3A5F",
    )
    fig.suptitle(
        r"Direct PISN rate-density diagnostic from Pop III SFRD",
        y=0.995,
        fontsize=15.0,
        color="#1F3A5F",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=500)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cosmology = Cosmology()
    _validate_args(args)
    if args.input_csv is None:
        popiii_ssp_file = _resolve_path(args.popiii_ssp_file)
        if not popiii_ssp_file.is_file():
            raise FileNotFoundError(f"Pop III SSP file not found: {popiii_ssp_file}")
        eta_pisn = _pisn_events_per_stellar_mass(
            imf_slope=float(args.imf_slope),
            imf_min_msun=float(args.imf_min_msun),
            imf_max_msun=float(args.imf_max_msun),
            pisn_min_msun=float(args.pisn_min_msun),
            pisn_max_msun=float(args.pisn_max_msun),
        )
        points: list[RatePoint] = []
        for z in sorted(float(value) for value in args.z_values):
            for scenario in _scenarios(args):
                point = _run_point(
                    cosmology=cosmology,
                    args=args,
                    z=z,
                    scenario=scenario,
                    eta_pisn_per_msun=eta_pisn,
                    popiii_ssp_file=popiii_ssp_file,
                )
                points.append(point)
                print(
                    f"z={z:g} {scenario.key}: rho_SFRIII={point.rho_sfr_popiii:.6e}, "
                    f"n_dot_PISN={point.pisn_rate_density:.6e}",
                    flush=True,
                )
    else:
        input_csv = _resolve_path(args.input_csv)
        eta_pisn, points = _read_csv(input_csv)
        print(f"Read {input_csv}")

    output_prefix = _resolve_path(args.output_prefix)
    figure_path = output_prefix.with_suffix(".pdf")
    csv_path = output_prefix.with_suffix(".csv")
    _draw_figure(
        figure_path,
        points,
        eta_pisn_per_msun=eta_pisn,
        include_ratio_panel=not args.no_ratio_panel,
    )
    if args.input_csv is None:
        _write_csv(csv_path, points, eta_pisn_per_msun=eta_pisn)
        print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
