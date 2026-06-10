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
)
from auroralf.mah import Cosmology, generate_halo_histories  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402
from auroralf.uvlf.hmf_sampling import compute_halo_mass_function_dndm  # noqa: E402


SOLAR_OH12 = 8.69
DEFAULT_OBSERVATION_CSV = (
    PROJECT_ROOT / "external_data" / "observations" / "metallicity" / "highz_metallicity_redshift_points.csv"
)
REQUIRED_OBSERVATION_COLUMNS = {
    "label",
    "source",
    "comparison_type",
    "redshift",
    "redshift_min",
    "redshift_max",
    "oh12",
    "oh12_low",
    "oh12_high",
    "marker",
    "color",
    "source_url",
    "notes",
}


@dataclass(frozen=True)
class MassMetallicitySummary:
    redshift: float
    logmh: float
    hmf_weight: float
    zgas_median_zsun: float
    zgas_p16_zsun: float
    zgas_p84_zsun: float
    sfr_median_msun_yr: float
    stellar_mass_median_msun: float
    starforming_count: int


@dataclass(frozen=True)
class RedshiftMetallicitySummary:
    redshift: float
    hmf_weighted_mean_zsun: float
    hmf_weighted_median_zsun: float
    hmf_weighted_p16_zsun: float
    hmf_weighted_p84_zsun: float
    hmf_sfr_weighted_mean_zsun: float
    hmf_weight_total: float
    sfr_weight_total: float


@dataclass(frozen=True)
class ObservationPoint:
    label: str
    source: str
    comparison_type: str
    redshift: float
    redshift_min: float
    redshift_max: float
    oh12: float
    oh12_low: float
    oh12_high: float
    marker: str
    color: str
    source_url: str
    notes: str


def _parse_float_grid(text: str) -> list[float]:
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HMF-weighted mean regulator gas metallicity as a function of redshift."
    )
    parser.add_argument("--redshifts", type=_parse_float_grid, default=_parse_float_grid("3.2 4 5.5 7 8 10 12.5"))
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--logM-min", type=float, default=9.5)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--N-mass", type=int, default=6)
    parser.add_argument("--n-tracks", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--enable-time-delay", action="store_true")
    parser.add_argument(
        "--gas-fraction-norm",
        type=float,
        default=0.02,
        help="Regulator f_res = Mgas / (fb Mh), not Mgas / (Mgas + Mstar).",
    )
    parser.add_argument("--gas-fraction-mass-slope", type=float, default=0.0)
    parser.add_argument("--gas-fraction-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-loading-norm", type=float, default=20.0)
    parser.add_argument("--metal-loading-mass-slope", type=float, default=-0.5)
    parser.add_argument("--metal-loading-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-yield", type=float, default=0.01)
    parser.add_argument("--returned-fraction", type=float, default=0.4)
    parser.add_argument("--inflow-metallicity-zsun", type=float, default=0.0)
    parser.add_argument(
        "--observation-csv",
        type=str,
        default=str(DEFAULT_OBSERVATION_CSV),
        help="CSV of observational gas-metallicity points to overlay.",
    )
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def _resolve_prefix(output_prefix: str | None) -> Path:
    if output_prefix is None:
        return PROJECT_ROOT / "outputs" / "regulator_metallicity_hmf_weighted_redshift"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    return prefix.resolve().with_suffix("") if prefix.suffix else prefix.resolve()


def _resolve_existing_path(path_text: str, description: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.redshifts) == 0:
        raise ValueError("redshifts must contain at least one value")
    if any(float(z) <= 0.0 for z in args.redshifts):
        raise ValueError("all redshifts must be positive")
    if args.z_start_max <= max(args.redshifts):
        raise ValueError("z-start-max must be greater than every requested redshift")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must be larger than logM-min")
    if args.N_mass < 2:
        raise ValueError("N-mass must be at least two")
    if args.n_tracks < 1:
        raise ValueError("n-tracks must be positive")
    if args.n_grid < 2:
        raise ValueError("n-grid must be at least two")
    if args.gas_fraction_norm <= 0.0 or args.gas_fraction_norm > 1.0:
        raise ValueError("gas-fraction-norm must lie in (0, 1]")
    if args.metal_loading_norm < 0.0:
        raise ValueError("metal-loading-norm must be non-negative")
    if args.metal_yield < 0.0:
        raise ValueError("metal-yield must be non-negative")
    if not 0.0 <= args.returned_fraction < 1.0:
        raise ValueError("returned-fraction must lie in [0, 1)")
    if args.inflow_metallicity_zsun < 0.0:
        raise ValueError("inflow-metallicity-zsun must be non-negative")
    _resolve_existing_path(str(args.observation_csv), "observation CSV")


def _dt_from_grid(cosmology: Cosmology, z_final: float, z_start_max: float, n_grid: int) -> float:
    from astropy.cosmology import FlatLambdaCDM

    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    t_start = float(astro.age(z_start_max).value)
    t_end = float(astro.age(z_final).value)
    return (t_end - t_start) / float(n_grid - 1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if value_array.shape != weight_array.shape:
        raise ValueError("values and weights must have matching shapes")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("no positive finite weights for weighted quantile")
    selected_values = value_array[valid]
    selected_weights = weight_array[valid]
    order = np.argsort(selected_values)
    sorted_values = selected_values[order]
    sorted_weights = selected_weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = float(quantile) * float(cumulative[-1])
    return float(sorted_values[min(np.searchsorted(cumulative, threshold, side="left"), sorted_values.size - 1)])


def _final_positive_median(values: np.ndarray, mask: np.ndarray, name: str) -> tuple[float, float, float, int]:
    final_values = np.asarray(values, dtype=float)[:, -1]
    final_mask = np.asarray(mask, dtype=bool)[:, -1]
    selected = final_values[final_mask & np.isfinite(final_values) & (final_values > 0.0)]
    if selected.size == 0:
        raise RuntimeError(f"no positive final {name} values")
    p16, median, p84 = np.percentile(selected, [16.0, 50.0, 84.0])
    return float(median), float(p16), float(p84), int(selected.size)


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


def _mass_weight_grid(log_masses: np.ndarray, redshift: float) -> np.ndarray:
    masses = np.power(10.0, np.asarray(log_masses, dtype=float))
    dndm = np.asarray(compute_halo_mass_function_dndm(masses, float(redshift)), dtype=float)
    dndlogm = masses * np.log(10.0) * dndm
    if not np.all(np.isfinite(dndlogm)) or np.any(dndlogm <= 0.0):
        raise RuntimeError("HMF weights must be finite and positive")
    dlogm = float(np.median(np.diff(np.sort(log_masses))))
    return dndlogm * dlogm


def _evaluate_one_mass(
    *,
    redshift: float,
    logmh: float,
    mass_index: int,
    redshift_index: int,
    args: argparse.Namespace,
    cosmology: Cosmology,
    parameters: RegulatorMetallicityParameters,
) -> tuple[float, float, float, float, float, int]:
    dt_gyr = _dt_from_grid(cosmology, float(redshift), float(args.z_start_max), int(args.n_grid))
    histories = generate_halo_histories(
        n_tracks=int(args.n_tracks),
        z_final=float(redshift),
        Mh_final=float(10.0 ** float(logmh)),
        z_start_max=float(args.z_start_max),
        cosmology=cosmology,
        random_seed=int(args.random_seed + 1000 * redshift_index + 100 * mass_index),
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
    starforming_grid = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
    metallicity = compute_regulator_metallicity(
        t_grid_gyr=t_grid,
        z_grid=z_grid,
        mh_grid=mh_grid,
        sfr_grid=sfr_grid,
        active_grid=starforming_grid,
        baryon_fraction=cosmology.omega_b / cosmology.omega_m,
        parameters=parameters,
        random_seed=int(args.metallicity_random_seed + 1000 * redshift_index + 100 * mass_index),
    )
    z_median, z_p16, z_p84, count = _final_positive_median(
        metallicity.gas_metallicity_zsun_grid,
        starforming_grid,
        "metallicity",
    )
    sfr_median, _, _, _ = _final_positive_median(sfr_grid, starforming_grid, "SFR")
    stellar_mass_grid = _surviving_stellar_mass_grid(
        t_grid_gyr=t_grid,
        sfr_grid=sfr_grid,
        active_grid=starforming_grid,
        returned_fraction=float(args.returned_fraction),
    )
    mstar_median, _, _, _ = _final_positive_median(stellar_mass_grid, starforming_grid, "stellar mass")
    return z_median, z_p16, z_p84, sfr_median, mstar_median, count


def _compute_summaries(
    args: argparse.Namespace,
    parameters: RegulatorMetallicityParameters,
) -> tuple[list[MassMetallicitySummary], list[RedshiftMetallicitySummary]]:
    cosmology = Cosmology()
    redshifts = sorted([float(item) for item in args.redshifts])
    log_masses = np.linspace(float(args.logM_min), float(args.logM_max), int(args.N_mass))
    mass_rows: list[MassMetallicitySummary] = []
    redshift_rows: list[RedshiftMetallicitySummary] = []
    for redshift_index, redshift in enumerate(redshifts):
        hmf_weights = _mass_weight_grid(log_masses, redshift)
        z_by_mass = np.empty(log_masses.size, dtype=float)
        sfr_by_mass = np.empty(log_masses.size, dtype=float)
        for mass_index, logmh in enumerate(log_masses):
            z_median, z_p16, z_p84, sfr_median, mstar_median, count = _evaluate_one_mass(
                redshift=redshift,
                logmh=float(logmh),
                mass_index=mass_index,
                redshift_index=redshift_index,
                args=args,
                cosmology=cosmology,
                parameters=parameters,
            )
            z_by_mass[mass_index] = z_median
            sfr_by_mass[mass_index] = sfr_median
            mass_rows.append(
                MassMetallicitySummary(
                    redshift=float(redshift),
                    logmh=float(logmh),
                    hmf_weight=float(hmf_weights[mass_index]),
                    zgas_median_zsun=float(z_median),
                    zgas_p16_zsun=float(z_p16),
                    zgas_p84_zsun=float(z_p84),
                    sfr_median_msun_yr=float(sfr_median),
                    stellar_mass_median_msun=float(mstar_median),
                    starforming_count=int(count),
                )
            )
        hmf_mean = float(np.average(z_by_mass, weights=hmf_weights))
        hmf_median = _weighted_quantile(z_by_mass, hmf_weights, 0.5)
        hmf_p16 = _weighted_quantile(z_by_mass, hmf_weights, 0.16)
        hmf_p84 = _weighted_quantile(z_by_mass, hmf_weights, 0.84)
        sfr_weights = hmf_weights * sfr_by_mass
        if not np.any(np.isfinite(sfr_weights) & (sfr_weights > 0.0)):
            raise RuntimeError(f"no positive HMF x SFR weights at z={redshift:g}")
        hmf_sfr_mean = float(np.average(z_by_mass, weights=sfr_weights))
        redshift_rows.append(
            RedshiftMetallicitySummary(
                redshift=float(redshift),
                hmf_weighted_mean_zsun=hmf_mean,
                hmf_weighted_median_zsun=hmf_median,
                hmf_weighted_p16_zsun=hmf_p16,
                hmf_weighted_p84_zsun=hmf_p84,
                hmf_sfr_weighted_mean_zsun=hmf_sfr_mean,
                hmf_weight_total=float(np.sum(hmf_weights)),
                sfr_weight_total=float(np.sum(sfr_weights)),
            )
        )
        print(
            f"z={redshift:g}: HMF mean Z={hmf_mean:.5g}, "
            f"HMFxSFR mean Z={hmf_sfr_mean:.5g}",
            flush=True,
        )
    return mass_rows, redshift_rows


def _zsun_to_oh12(zsun: np.ndarray | float) -> np.ndarray:
    z = np.asarray(zsun, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("Z/Zsun values must be positive and finite before O/H conversion")
    return SOLAR_OH12 + np.log10(z)


def _oh12_to_zsun(oh12: np.ndarray | float) -> np.ndarray:
    abundance = np.asarray(oh12, dtype=float)
    if np.any(~np.isfinite(abundance)):
        raise ValueError("12+log(O/H) values must be finite before Z/Zsun conversion")
    return np.power(10.0, abundance - SOLAR_OH12)


def _zsun_to_oh12_axis(zsun: np.ndarray | float) -> np.ndarray:
    z = np.asarray(zsun, dtype=float)
    result = np.full_like(z, np.nan, dtype=float)
    valid = np.isfinite(z) & (z > 0.0)
    result[valid] = SOLAR_OH12 + np.log10(z[valid])
    return result


def _required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = row.get(field, "").strip()
    if value == "":
        raise ValueError(f"observation row {row_number} has empty required field: {field}")
    return value


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    text = _required_text(row, field, row_number)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"observation row {row_number} field {field} is not a float: {text!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"observation row {row_number} field {field} is not finite: {text!r}")
    return value


def _read_observation_points(path: Path) -> list[ObservationPoint]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"observation CSV has no header: {path}")
        missing = sorted(REQUIRED_OBSERVATION_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"observation CSV missing required columns {missing}: {path}")
        rows: list[ObservationPoint] = []
        for row_number, row in enumerate(reader, start=2):
            redshift = _required_float(row, "redshift", row_number)
            redshift_min = _required_float(row, "redshift_min", row_number)
            redshift_max = _required_float(row, "redshift_max", row_number)
            oh12 = _required_float(row, "oh12", row_number)
            oh12_low = _required_float(row, "oh12_low", row_number)
            oh12_high = _required_float(row, "oh12_high", row_number)
            if redshift <= 0.0 or redshift_min <= 0.0 or redshift_max <= 0.0:
                raise ValueError(f"observation row {row_number} redshifts must be positive")
            if redshift_min > redshift or redshift_max < redshift:
                raise ValueError(f"observation row {row_number} redshift must lie within redshift_min/max")
            if oh12_low > oh12 or oh12_high < oh12:
                raise ValueError(f"observation row {row_number} oh12 must lie within oh12_low/high")
            if _required_text(row, "marker", row_number) == "":
                raise ValueError(f"observation row {row_number} marker is empty")
            if _required_text(row, "color", row_number) == "":
                raise ValueError(f"observation row {row_number} color is empty")
            rows.append(
                ObservationPoint(
                    label=_required_text(row, "label", row_number),
                    source=_required_text(row, "source", row_number),
                    comparison_type=_required_text(row, "comparison_type", row_number),
                    redshift=redshift,
                    redshift_min=redshift_min,
                    redshift_max=redshift_max,
                    oh12=oh12,
                    oh12_low=oh12_low,
                    oh12_high=oh12_high,
                    marker=_required_text(row, "marker", row_number),
                    color=_required_text(row, "color", row_number),
                    source_url=_required_text(row, "source_url", row_number),
                    notes=_required_text(row, "notes", row_number),
                )
            )
    if not rows:
        raise ValueError(f"observation CSV contains no data rows: {path}")
    return rows


def _plot_redshift_evolution(
    *,
    output_prefix: Path,
    rows: list[RedshiftMetallicitySummary],
    obs_rows: list[ObservationPoint],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(7.4, 5.25), constrained_layout=True)
    redshift = np.asarray([row.redshift for row in rows], dtype=float)
    hmf_mean = np.asarray([row.hmf_weighted_mean_zsun for row in rows], dtype=float)
    hmf_median = np.asarray([row.hmf_weighted_median_zsun for row in rows], dtype=float)
    hmf_p16 = np.asarray([row.hmf_weighted_p16_zsun for row in rows], dtype=float)
    hmf_p84 = np.asarray([row.hmf_weighted_p84_zsun for row in rows], dtype=float)
    hmf_sfr_mean = np.asarray([row.hmf_sfr_weighted_mean_zsun for row in rows], dtype=float)

    order = np.argsort(redshift)
    redshift = redshift[order]
    hmf_mean = hmf_mean[order]
    hmf_median = hmf_median[order]
    hmf_p16 = hmf_p16[order]
    hmf_p84 = hmf_p84[order]
    hmf_sfr_mean = hmf_sfr_mean[order]

    ax.plot(redshift, _zsun_to_oh12(hmf_mean), color="#0173b2", lw=2.5, label="AuroraLF HMF-weighted mean")
    ax.plot(
        redshift,
        _zsun_to_oh12(hmf_sfr_mean),
        color="#0173b2",
        lw=2.1,
        ls=":",
        label="AuroraLF HMF x SFR weighted mean",
    )
    ax.plot(
        redshift,
        _zsun_to_oh12(hmf_median),
        color="#029e73",
        lw=2.0,
        ls="--",
        label="AuroraLF HMF-weighted median",
    )
    ax.fill_between(
        redshift,
        _zsun_to_oh12(hmf_p16),
        _zsun_to_oh12(hmf_p84),
        color="#029e73",
        alpha=0.16,
        lw=0.0,
        label="HMF weighted 16-84%",
    )

    seen_labels: set[str] = set()
    for point in obs_rows:
        label = str(point.label)
        plot_label = label if label not in seen_labels else None
        seen_labels.add(label)
        y = float(point.oh12)
        yerr_low = y - float(point.oh12_low)
        yerr_high = float(point.oh12_high) - y
        ax.errorbar(
            [float(point.redshift)],
            [y],
            yerr=[[yerr_low], [yerr_high]],
            fmt=str(point.marker),
            ms=11 if str(point.marker) == "*" else 6.5,
            color=str(point.color),
            mec="black",
            mew=0.6,
            capsize=3,
            label=plot_label,
            zorder=8,
        )

    x_min = min(float(np.min(redshift)), min(float(point.redshift) for point in obs_rows)) - 0.35
    x_max = max(float(np.max(redshift)), max(float(point.redshift) for point in obs_rows)) + 0.35
    ax.set_xlim(x_min, x_max)
    all_oh = np.concatenate(
        [
            _zsun_to_oh12(hmf_p16),
            _zsun_to_oh12(hmf_p84),
            np.asarray([float(point.oh12_low) for point in obs_rows], dtype=float),
            np.asarray([float(point.oh12_high) for point in obs_rows], dtype=float),
        ]
    )
    ax.set_ylim(float(np.nanmin(all_oh)) - 0.18, float(np.nanmax(all_oh)) + 0.18)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"equiv. $12+\log({\rm O/H})$")
    fig.suptitle(
        rf"Reed07 HMF over ${args.logM_min:g}<\log M_h<{args.logM_max:g}$; "
        "points are galaxy/MZR constraints",
        fontsize=8.4,
    )
    ax.grid(alpha=0.22)
    ax.legend(
        frameon=False,
        fontsize=6.7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.2,
    )

    secondary = ax.secondary_yaxis("right", functions=(_oh12_to_zsun, _zsun_to_oh12_axis))
    secondary.set_ylabel(r"equiv. $Z/Z_\odot$")
    secondary.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    secondary.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _write_summary_csv(path: Path, rows: list[RedshiftMetallicitySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "redshift",
                "hmf_weighted_mean_zsun",
                "hmf_weighted_mean_oh12",
                "hmf_weighted_median_zsun",
                "hmf_weighted_median_oh12",
                "hmf_weighted_p16_zsun",
                "hmf_weighted_p84_zsun",
                "hmf_sfr_weighted_mean_zsun",
                "hmf_sfr_weighted_mean_oh12",
                "hmf_weight_total",
                "sfr_weight_total",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "redshift": row.redshift,
                    "hmf_weighted_mean_zsun": row.hmf_weighted_mean_zsun,
                    "hmf_weighted_mean_oh12": float(_zsun_to_oh12(row.hmf_weighted_mean_zsun)),
                    "hmf_weighted_median_zsun": row.hmf_weighted_median_zsun,
                    "hmf_weighted_median_oh12": float(_zsun_to_oh12(row.hmf_weighted_median_zsun)),
                    "hmf_weighted_p16_zsun": row.hmf_weighted_p16_zsun,
                    "hmf_weighted_p84_zsun": row.hmf_weighted_p84_zsun,
                    "hmf_sfr_weighted_mean_zsun": row.hmf_sfr_weighted_mean_zsun,
                    "hmf_sfr_weighted_mean_oh12": float(_zsun_to_oh12(row.hmf_sfr_weighted_mean_zsun)),
                    "hmf_weight_total": row.hmf_weight_total,
                    "sfr_weight_total": row.sfr_weight_total,
                }
            )


def _write_mass_csv(path: Path, rows: list[MassMetallicitySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "redshift",
                "logmh",
                "hmf_weight",
                "zgas_median_zsun",
                "zgas_median_oh12",
                "zgas_p16_zsun",
                "zgas_p84_zsun",
                "sfr_median_msun_yr",
                "stellar_mass_median_msun",
                "starforming_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "redshift": row.redshift,
                    "logmh": row.logmh,
                    "hmf_weight": row.hmf_weight,
                    "zgas_median_zsun": row.zgas_median_zsun,
                    "zgas_median_oh12": float(_zsun_to_oh12(row.zgas_median_zsun)),
                    "zgas_p16_zsun": row.zgas_p16_zsun,
                    "zgas_p84_zsun": row.zgas_p84_zsun,
                    "sfr_median_msun_yr": row.sfr_median_msun_yr,
                    "stellar_mass_median_msun": row.stellar_mass_median_msun,
                    "starforming_count": row.starforming_count,
                }
            )


def _write_text_summary(
    path: Path,
    args: argparse.Namespace,
    parameters: RegulatorMetallicityParameters,
    rows: list[RedshiftMetallicitySummary],
    obs_rows: list[ObservationPoint],
    observation_csv: Path,
    figure_pdf: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"redshifts: {' '.join(f'{float(item):g}' for item in args.redshifts)}\n")
        handle.write(f"logM range: {float(args.logM_min):g} {float(args.logM_max):g}\n")
        handle.write(f"N_mass: {int(args.N_mass)}\n")
        handle.write(f"n_tracks: {int(args.n_tracks)}\n")
        handle.write(f"n_grid: {int(args.n_grid)}\n")
        handle.write(f"enable_time_delay: {bool(args.enable_time_delay)}\n")
        handle.write(f"observation_csv: {observation_csv}\n")
        handle.write(f"figure_pdf: {figure_pdf}\n")
        handle.write("\nRegulator parameters:\n")
        for key, value in parameters.as_metadata().items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nRedshift summary:\n")
        for row in rows:
            handle.write(
                f"z={row.redshift:g}: HMF mean={row.hmf_weighted_mean_zsun:.6g} Zsun "
                f"({float(_zsun_to_oh12(row.hmf_weighted_mean_zsun)):.4g}), "
                f"HMFxSFR mean={row.hmf_sfr_weighted_mean_zsun:.6g} Zsun "
                f"({float(_zsun_to_oh12(row.hmf_sfr_weighted_mean_zsun)):.4g})\n"
            )
        handle.write("\nObservation overlays:\n")
        for point in obs_rows:
            handle.write(
                f"{point.source}: z={point.redshift:g}, "
                f"12+log(O/H)={point.oh12:g} [{point.oh12_low:g}, {point.oh12_high:g}], "
                f"type={point.comparison_type}, label={point.label}, source={point.source_url}, "
                f"notes={point.notes}\n"
            )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_prefix = _resolve_prefix(args.output_prefix)
    observation_csv = _resolve_existing_path(str(args.observation_csv), "observation CSV")
    obs_rows = _read_observation_points(observation_csv)
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
    mass_rows, redshift_rows = _compute_summaries(args, parameters)
    png_path, pdf_path = _plot_redshift_evolution(
        output_prefix=output_prefix,
        rows=redshift_rows,
        obs_rows=obs_rows,
        args=args,
    )
    data_prefix = PROJECT_ROOT / "data_save" / output_prefix.name
    _write_summary_csv(data_prefix.with_name(f"{data_prefix.name}_summary.csv"), redshift_rows)
    _write_mass_csv(data_prefix.with_name(f"{data_prefix.name}_by_mass.csv"), mass_rows)
    summary_path = output_prefix.with_suffix(".txt")
    _write_text_summary(summary_path, args, parameters, redshift_rows, obs_rows, observation_csv, pdf_path)
    print(f"saved: {png_path} {pdf_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
