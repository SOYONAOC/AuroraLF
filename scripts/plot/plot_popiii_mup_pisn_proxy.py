#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from auroralf.constants import PLANCK18_H0_KM_S_MPC, PLANCK18_OMEGA_B, PLANCK18_OMEGA_M
from auroralf.ssp import load_popiii_uv_luminosity_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_Z_TAGS = ("z6", "z8", "z10", "z12p5", "z14p5")
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "popiii_mup_pisn_proxy"
SCENARIO_KEYS = ("current", "fixed_mup1e10")
SCENARIO_LABELS = {
    "current": r"current $M_{\rm up,halo}=M_{\rm vir}(10^4{\rm K})$",
    "fixed_mup1e10": r"fixed $M_{\rm up,halo}=10^{10}M_\odot$",
}
SCENARIO_COLORS = {
    "current": "#1F5C8B",
    "fixed_mup1e10": "#202020",
}
YEAR_PER_MYR = 1.0e6
DEG2_PER_SR = (180.0 / np.pi) ** 2


@dataclass(frozen=True)
class ScenarioSeries:
    z: np.ndarray
    luminosity_density: np.ndarray
    sfrd_by_visibility: dict[float, np.ndarray]
    pisn_rate_by_visibility: dict[float, np.ndarray]
    surface_rate_by_visibility: dict[float, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a PISN-rate proxy from Pop III UV luminosity density in the Mup comparison outputs."
    )
    parser.add_argument("--z-tags", nargs="+", default=list(DEFAULT_Z_TAGS))
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="outputs/uvlf_{ztag}_popiii_mup_dust_observations.npz",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--visibility-myr", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--central-visibility-myr", type=float, default=3.0)
    parser.add_argument("--imf-slope", type=float, default=2.35)
    parser.add_argument("--imf-min-msun", type=float, default=50.0)
    parser.add_argument("--imf-max-msun", type=float, default=500.0)
    parser.add_argument("--pisn-min-msun", type=float, default=140.0)
    parser.add_argument("--pisn-max-msun", type=float, default=260.0)
    return parser.parse_args()


def _resolve_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _power_integral(lower: float, upper: float, exponent: float) -> float:
    if lower <= 0.0 or upper <= lower:
        raise ValueError("power-law integration bounds must be positive and increasing")
    if np.isclose(exponent, -1.0, rtol=0.0, atol=1.0e-14):
        return float(np.log(upper / lower))
    return float((upper ** (exponent + 1.0) - lower ** (exponent + 1.0)) / (exponent + 1.0))


def _pisn_events_per_stellar_mass(
    *,
    imf_slope: float,
    imf_min_msun: float,
    imf_max_msun: float,
    pisn_min_msun: float,
    pisn_max_msun: float,
) -> float:
    """Return the PISN-event count per unit formed stellar mass.

    The numerator integrates a power-law IMF over the classical non-rotating,
    zero-metallicity PISN progenitor interval of Heger & Woosley (2002), DOI:
    10.1086/338487, arXiv:astro-ph/0107037.  A slope of 2.35 is the Salpeter
    (1955) convention, DOI: 10.1086/145971.  The adopted IMF limits and the
    conversion of UV-inferred Pop III star formation into a rate are explicit
    AuroraLF proxy assumptions, not a full delay-time/explosion model.
    """
    if not np.isfinite(imf_slope):
        raise ValueError("imf_slope must be finite")
    if imf_min_msun <= 0.0 or imf_max_msun <= imf_min_msun:
        raise ValueError("IMF mass bounds must be positive and increasing")
    if pisn_min_msun < imf_min_msun or pisn_max_msun > imf_max_msun or pisn_max_msun <= pisn_min_msun:
        raise ValueError("PISN mass window must lie inside the IMF bounds")
    number_integral = _power_integral(pisn_min_msun, pisn_max_msun, -float(imf_slope))
    mass_integral = _power_integral(imf_min_msun, imf_max_msun, 1.0 - float(imf_slope))
    if mass_integral <= 0.0:
        raise RuntimeError("IMF mass integral is non-positive")
    return number_integral / mass_integral


def _integrated_lnu_per_sfr(
    *,
    ages_myr: np.ndarray,
    lnu_per_msun: np.ndarray,
    visibility_myr: float,
) -> float:
    ages = np.asarray(ages_myr, dtype=float)
    luminosity = np.asarray(lnu_per_msun, dtype=float)
    if ages.ndim != 1 or luminosity.ndim != 1 or ages.size != luminosity.size:
        raise ValueError("SSP age and luminosity arrays must be 1D arrays with the same length")
    if np.any(~np.isfinite(ages)) or np.any(~np.isfinite(luminosity)):
        raise ValueError("SSP age and luminosity arrays must be finite")
    if np.any(ages <= 0.0) or np.any(np.diff(ages) <= 0.0):
        raise ValueError("SSP ages must be positive and strictly increasing")
    if np.any(luminosity < 0.0):
        raise ValueError("SSP luminosity per mass must be non-negative")
    if visibility_myr <= ages[0] or visibility_myr > ages[-1]:
        raise ValueError(
            f"visibility_myr must lie inside the SSP age grid ({ages[0]:.3g}, {ages[-1]:.3g}] Myr"
        )

    in_window = ages < float(visibility_myr)
    age_segment = np.concatenate([ages[in_window], np.asarray([float(visibility_myr)])])
    luminosity_at_visibility = np.interp(
        np.log10(float(visibility_myr)),
        np.log10(ages),
        luminosity,
    )
    luminosity_segment = np.concatenate([luminosity[in_window], np.asarray([luminosity_at_visibility])])
    integral_myr = float(np.trapezoid(luminosity_segment, age_segment))
    if integral_myr <= 0.0:
        raise RuntimeError("integrated Pop III SSP luminosity per SFR is non-positive")
    return integral_myr * YEAR_PER_MYR


def _load_input_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for ztag in args.z_tags:
        paths.append(_resolve_path(str(args.input_pattern).format(ztag=ztag)))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input NPZ files:\n{formatted}")
    return paths


def _weighted_luminosity_density(payload: np.lib.npyio.NpzFile, scenario: str) -> float:
    luminosity = np.asarray(payload[f"{scenario}_popiii_luminosity"], dtype=float)
    weight = np.asarray(payload[f"{scenario}_sample_weight"], dtype=float)
    if luminosity.shape != weight.shape:
        raise ValueError(f"{scenario} Pop III luminosity and sample weight arrays have different shapes")
    valid = np.isfinite(luminosity) & np.isfinite(weight) & (luminosity > 0.0) & (weight > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(luminosity[valid] * weight[valid]))


def _build_series(
    *,
    paths: list[Path],
    visibility_myr: tuple[float, ...],
    eta_pisn_per_msun: float,
) -> tuple[dict[str, ScenarioSeries], Path]:
    cosmology = FlatLambdaCDM(H0=PLANCK18_H0_KM_S_MPC, Om0=PLANCK18_OMEGA_M, Ob0=PLANCK18_OMEGA_B)
    z_values = []
    luminosity_density: dict[str, list[float]] = {scenario: [] for scenario in SCENARIO_KEYS}
    ssp_paths = []

    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            z_values.append(float(np.asarray(payload["z"]).reshape(-1)[0]))
            ssp_paths.append(Path(str(np.asarray(payload["popiii_ssp_file"]).reshape(-1)[0])).expanduser().resolve())
            for scenario in SCENARIO_KEYS:
                luminosity_density[scenario].append(_weighted_luminosity_density(payload, scenario))

    unique_ssp_paths = sorted(set(ssp_paths))
    if len(unique_ssp_paths) != 1:
        raise ValueError(f"Input NPZ files do not agree on Pop III SSP path: {unique_ssp_paths}")
    popiii_ssp_path = unique_ssp_paths[0]
    ages_myr, lnu_per_msun = load_popiii_uv_luminosity_table(popiii_ssp_path)
    lnu_per_sfr = {
        float(window): _integrated_lnu_per_sfr(
            ages_myr=ages_myr,
            lnu_per_msun=lnu_per_msun,
            visibility_myr=float(window),
        )
        for window in visibility_myr
    }

    z = np.asarray(z_values, dtype=float)
    order = np.argsort(z)
    z = z[order]
    differential_volume = cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) / DEG2_PER_SR

    series: dict[str, ScenarioSeries] = {}
    for scenario in SCENARIO_KEYS:
        rho_lnu = np.asarray(luminosity_density[scenario], dtype=float)[order]
        sfrd_by_visibility = {
            window: rho_lnu / lnu_per_sfr[window]
            for window in visibility_myr
        }
        pisn_rate_by_visibility = {
            window: sfrd_by_visibility[window] * float(eta_pisn_per_msun)
            for window in visibility_myr
        }
        surface_rate_by_visibility = {
            window: pisn_rate_by_visibility[window] * differential_volume / (1.0 + z)
            for window in visibility_myr
        }
        series[scenario] = ScenarioSeries(
            z=z,
            luminosity_density=rho_lnu,
            sfrd_by_visibility=sfrd_by_visibility,
            pisn_rate_by_visibility=pisn_rate_by_visibility,
            surface_rate_by_visibility=surface_rate_by_visibility,
        )
    return series, popiii_ssp_path


def _positive_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    positive = np.concatenate([np.asarray(value, dtype=float) for value in values])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise RuntimeError("No positive values available for log-axis limits")
    return float(np.min(positive) * 0.55), float(np.max(positive) * 1.8)


def _write_csv(
    path: Path,
    *,
    series: dict[str, ScenarioSeries],
    visibility_myr: tuple[float, ...],
    eta_pisn_per_msun: float,
    popiii_ssp_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"eta_pisn_per_msun={eta_pisn_per_msun:.10e}", f"popiii_ssp_file={popiii_ssp_path}"])
        writer.writerow(
            [
                "scenario",
                "z",
                "rho_lnu_popiii_erg_s^-1_Hz^-1_Mpc^-3",
                *[f"sfrd_popiii_tvis{window:g}myr_msun_yr^-1_Mpc^-3" for window in visibility_myr],
                *[f"pisn_rate_tvis{window:g}myr_yr^-1_Mpc^-3" for window in visibility_myr],
                *[f"pisn_surface_tvis{window:g}myr_yr^-1_deg^-2_dz^-1" for window in visibility_myr],
            ]
        )
        for scenario, scenario_series in series.items():
            for index, z_value in enumerate(scenario_series.z):
                writer.writerow(
                    [
                        scenario,
                        f"{float(z_value):.8e}",
                        f"{float(scenario_series.luminosity_density[index]):.8e}",
                        *[
                            f"{float(scenario_series.sfrd_by_visibility[window][index]):.8e}"
                            for window in visibility_myr
                        ],
                        *[
                            f"{float(scenario_series.pisn_rate_by_visibility[window][index]):.8e}"
                            for window in visibility_myr
                        ],
                        *[
                            f"{float(scenario_series.surface_rate_by_visibility[window][index]):.8e}"
                            for window in visibility_myr
                        ],
                    ]
                )


def _draw_figure(
    path: Path,
    *,
    series: dict[str, ScenarioSeries],
    visibility_myr: tuple[float, ...],
    central_visibility_myr: float,
    eta_pisn_per_msun: float,
) -> None:
    if central_visibility_myr not in visibility_myr:
        raise ValueError("--central-visibility-myr must be one of --visibility-myr")
    low_window = max(visibility_myr)
    high_window = min(visibility_myr)

    plt.style.use("apj")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9))
    ax_rate, ax_surface, ax_ratio = axes
    for scenario, scenario_series in series.items():
        color = SCENARIO_COLORS[scenario]
        z = scenario_series.z
        rate = scenario_series.pisn_rate_by_visibility[central_visibility_myr]
        rate_low = scenario_series.pisn_rate_by_visibility[low_window]
        rate_high = scenario_series.pisn_rate_by_visibility[high_window]
        surface = scenario_series.surface_rate_by_visibility[central_visibility_myr]
        surface_low = scenario_series.surface_rate_by_visibility[low_window]
        surface_high = scenario_series.surface_rate_by_visibility[high_window]
        ax_rate.fill_between(z, rate_low, rate_high, color=color, alpha=0.16, linewidth=0.0)
        ax_surface.fill_between(z, surface_low, surface_high, color=color, alpha=0.16, linewidth=0.0)
        ax_rate.plot(z, rate, color=color, marker="o", linewidth=2.35, label=SCENARIO_LABELS[scenario])
        ax_surface.plot(z, surface, color=color, marker="o", linewidth=2.35, label=SCENARIO_LABELS[scenario])

    current = series["current"].pisn_rate_by_visibility[central_visibility_myr]
    fixed = series["fixed_mup1e10"].pisn_rate_by_visibility[central_visibility_myr]
    ratio = np.divide(fixed, current, out=np.full_like(fixed, np.nan), where=current > 0.0)
    ax_ratio.plot(series["current"].z, ratio, color="#8C5FBF", marker="D", linewidth=2.45)
    ax_ratio.axhline(1.0, color="0.25", linewidth=1.1, linestyle=":")

    ax_rate.set_yscale("log")
    ax_surface.set_yscale("log")
    ax_ratio.set_yscale("log")
    ax_rate.set_ylim(
        *_positive_ylim(
            [
                item.pisn_rate_by_visibility[window]
                for item in series.values()
                for window in visibility_myr
            ]
        )
    )
    ax_surface.set_ylim(
        *_positive_ylim(
            [
                item.surface_rate_by_visibility[window]
                for item in series.values()
                for window in visibility_myr
            ]
        )
    )
    ratio_positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if ratio_positive.size == 0:
        raise RuntimeError("No positive fixed/current PISN proxy ratios available")
    ax_ratio.set_ylim(float(np.min(ratio_positive) * 0.72), float(np.max(ratio_positive) * 1.35))

    for ax in axes:
        ax.set_xlim(5.7, 14.8)
        ax.set_xlabel(r"$z$")
        ax.grid(True, which="major", color="#C8D2DF", linewidth=0.72, alpha=0.85)
        ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.42, alpha=0.70)
    ax_rate.set_ylabel(r"$\dot n_{\rm PISN}$ proxy [yr$^{-1}$ Mpc$^{-3}$]")
    ax_surface.set_ylabel(r"$dN_{\rm PISN}/dz/d\Omega/dt_{\rm obs}$ [yr$^{-1}$ deg$^{-2}$]")
    ax_ratio.set_ylabel(r"fixed/current")

    ax_rate.legend(loc="lower right", frameon=True, fontsize=10.0)
    ax_surface.text(
        0.04,
        0.95,
        "no survey depth/cadence selection",
        transform=ax_surface.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1F3A5F",
    )
    ax_ratio.text(
        0.04,
        0.95,
        rf"$\eta_{{\rm PISN}}={eta_pisn_per_msun:.2e}\,M_\odot^{{-1}}$",
        transform=ax_ratio.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1F3A5F",
    )
    fig.suptitle(
        rf"PISN proxy from Pop III UV density; solid: $t_{{\rm vis}}={central_visibility_myr:g}$ Myr, "
        rf"band: {min(visibility_myr):g}--{max(visibility_myr):g} Myr",
        y=0.995,
        fontsize=14.5,
        color="#1F3A5F",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=500)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    visibility_myr = tuple(sorted({float(value) for value in args.visibility_myr}))
    if len(visibility_myr) < 2:
        raise ValueError("--visibility-myr must contain at least two distinct values")
    if float(args.central_visibility_myr) not in visibility_myr:
        raise ValueError("--central-visibility-myr must be one of --visibility-myr")
    eta_pisn = _pisn_events_per_stellar_mass(
        imf_slope=float(args.imf_slope),
        imf_min_msun=float(args.imf_min_msun),
        imf_max_msun=float(args.imf_max_msun),
        pisn_min_msun=float(args.pisn_min_msun),
        pisn_max_msun=float(args.pisn_max_msun),
    )
    input_paths = _load_input_paths(args)
    series, popiii_ssp_path = _build_series(
        paths=input_paths,
        visibility_myr=visibility_myr,
        eta_pisn_per_msun=eta_pisn,
    )
    output_prefix = _resolve_path(args.output_prefix)
    figure_path = output_prefix.with_suffix(".pdf")
    csv_path = output_prefix.with_suffix(".csv")
    _draw_figure(
        figure_path,
        series=series,
        visibility_myr=visibility_myr,
        central_visibility_myr=float(args.central_visibility_myr),
        eta_pisn_per_msun=eta_pisn,
    )
    _write_csv(
        csv_path,
        series=series,
        visibility_myr=visibility_myr,
        eta_pisn_per_msun=eta_pisn,
        popiii_ssp_path=popiii_ssp_path,
    )
    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")
    print(f"eta_pisn_per_msun={eta_pisn:.8e}")
    for scenario, scenario_series in series.items():
        rate = scenario_series.pisn_rate_by_visibility[float(args.central_visibility_myr)]
        surface = scenario_series.surface_rate_by_visibility[float(args.central_visibility_myr)]
        print(
            f"{scenario}: R_PISN(z=14.5)={rate[-1]:.6e} yr^-1 Mpc^-3, "
            f"surface(z=14.5)={surface[-1]:.6e} yr^-1 deg^-2 dz^-1"
        )


if __name__ == "__main__":
    main()
