#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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


DEFAULT_RELATION_CSV = (
    PROJECT_ROOT / "external_data" / "observations" / "metallicity" / "highz_mzr_relations.csv"
)
DEFAULT_ANCHOR_CSV = (
    PROJECT_ROOT / "external_data" / "observations" / "metallicity" / "highz_mzr_anchor_points.csv"
)
RELATION_COLUMNS = {
    "label",
    "source",
    "redshift",
    "redshift_min",
    "redshift_max",
    "pivot_logmstar",
    "slope",
    "intercept_oh12",
    "intercept_low",
    "intercept_high",
    "valid_logmstar_min",
    "valid_logmstar_max",
    "color",
    "linestyle",
    "source_url",
    "notes",
}
ANCHOR_COLUMNS = {
    "label",
    "source",
    "comparison_type",
    "redshift",
    "logmstar",
    "logmstar_low",
    "logmstar_high",
    "oh12",
    "oh12_low",
    "oh12_high",
    "marker",
    "color",
    "source_url",
    "notes",
}


@dataclass(frozen=True)
class MZRRelation:
    label: str
    source: str
    redshift: float
    redshift_min: float
    redshift_max: float
    pivot_logmstar: float
    slope: float
    intercept_oh12: float
    intercept_low: float
    intercept_high: float
    valid_logmstar_min: float
    valid_logmstar_max: float
    color: str
    linestyle: str
    source_url: str
    notes: str


@dataclass(frozen=True)
class MZRAnchor:
    label: str
    source: str
    comparison_type: str
    redshift: float
    logmstar: float
    logmstar_low: float
    logmstar_high: float
    oh12: float
    oh12_low: float
    oh12_high: float
    marker: str
    color: str
    source_url: str
    notes: str


@dataclass(frozen=True)
class ModelPoint:
    redshift: float
    logmh: float
    logmstar_median: float
    logmstar_p16: float
    logmstar_p84: float
    zgas_median_zsun: float
    zgas_p16_zsun: float
    zgas_p84_zsun: float
    oh12_median: float
    oh12_p16: float
    oh12_p84: float
    starforming_count: int


def _parse_float_grid(text: str) -> list[float]:
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate regulator gas metallicities against high-z observed mass-metallicity "
            "relations using observation-matched stellar-mass and redshift coordinates."
        )
    )
    parser.add_argument("--redshifts", type=_parse_float_grid, default=_parse_float_grid("3.2 5.5 8.0 12.34"))
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--logM-min", type=float, default=9.5)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--N-mass", type=int, default=6)
    parser.add_argument("--n-tracks", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metallicity-random-seed", type=int, default=123)
    parser.add_argument("--enable-time-delay", action="store_true")
    parser.add_argument("--gas-fraction-norm", type=float, default=0.02)
    parser.add_argument("--gas-fraction-mass-slope", type=float, default=0.0)
    parser.add_argument("--gas-fraction-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-loading-norm", type=float, default=20.0)
    parser.add_argument("--metal-loading-mass-slope", type=float, default=-0.5)
    parser.add_argument("--metal-loading-redshift-slope", type=float, default=0.0)
    parser.add_argument("--metal-yield", type=float, default=0.01)
    parser.add_argument("--returned-fraction", type=float, default=0.4)
    parser.add_argument("--inflow-metallicity-zsun", type=float, default=0.0)
    parser.add_argument("--relation-csv", type=str, default=str(DEFAULT_RELATION_CSV))
    parser.add_argument("--anchor-csv", type=str, default=str(DEFAULT_ANCHOR_CSV))
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


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


def _resolve_output_prefix(output_prefix: str | None) -> Path:
    if output_prefix is None:
        return PROJECT_ROOT / "outputs" / "regulator_metallicity_mzr_validation"
    path = Path(output_prefix).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().with_suffix("") if path.suffix else path.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if any(float(item) <= 0.0 for item in args.redshifts):
        raise ValueError("all redshifts must be positive")
    if args.z_start_max <= max(args.redshifts):
        raise ValueError("z-start-max must be greater than every requested redshift")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must exceed logM-min")
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
    _resolve_existing_path(str(args.relation_csv), "MZR relation CSV")
    _resolve_existing_path(str(args.anchor_csv), "MZR anchor CSV")


def _required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = row.get(field, "").strip()
    if value == "":
        raise ValueError(f"row {row_number} has empty required field: {field}")
    return value


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    text = _required_text(row, field, row_number)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"row {row_number} field {field} is not a float: {text!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"row {row_number} field {field} is not finite")
    return value


def _read_relations(path: Path) -> list[MZRRelation]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"MZR relation CSV has no header: {path}")
        missing = sorted(RELATION_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"MZR relation CSV missing columns {missing}: {path}")
        rows: list[MZRRelation] = []
        for row_number, row in enumerate(reader, start=2):
            redshift = _required_float(row, "redshift", row_number)
            redshift_min = _required_float(row, "redshift_min", row_number)
            redshift_max = _required_float(row, "redshift_max", row_number)
            intercept = _required_float(row, "intercept_oh12", row_number)
            low = _required_float(row, "intercept_low", row_number)
            high = _required_float(row, "intercept_high", row_number)
            valid_min = _required_float(row, "valid_logmstar_min", row_number)
            valid_max = _required_float(row, "valid_logmstar_max", row_number)
            if redshift_min > redshift or redshift_max < redshift:
                raise ValueError(f"row {row_number} redshift must lie inside redshift_min/max")
            if low > intercept or high < intercept:
                raise ValueError(f"row {row_number} intercept_oh12 must lie inside intercept_low/high")
            if valid_max <= valid_min:
                raise ValueError(f"row {row_number} valid_logmstar_max must exceed valid_logmstar_min")
            rows.append(
                MZRRelation(
                    label=_required_text(row, "label", row_number),
                    source=_required_text(row, "source", row_number),
                    redshift=redshift,
                    redshift_min=redshift_min,
                    redshift_max=redshift_max,
                    pivot_logmstar=_required_float(row, "pivot_logmstar", row_number),
                    slope=_required_float(row, "slope", row_number),
                    intercept_oh12=intercept,
                    intercept_low=low,
                    intercept_high=high,
                    valid_logmstar_min=valid_min,
                    valid_logmstar_max=valid_max,
                    color=_required_text(row, "color", row_number),
                    linestyle=_required_text(row, "linestyle", row_number),
                    source_url=_required_text(row, "source_url", row_number),
                    notes=_required_text(row, "notes", row_number),
                )
            )
    if not rows:
        raise ValueError(f"MZR relation CSV has no data rows: {path}")
    return rows


def _read_anchors(path: Path) -> list[MZRAnchor]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"MZR anchor CSV has no header: {path}")
        missing = sorted(ANCHOR_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"MZR anchor CSV missing columns {missing}: {path}")
        rows: list[MZRAnchor] = []
        for row_number, row in enumerate(reader, start=2):
            redshift = _required_float(row, "redshift", row_number)
            logmstar = _required_float(row, "logmstar", row_number)
            logmstar_low = _required_float(row, "logmstar_low", row_number)
            logmstar_high = _required_float(row, "logmstar_high", row_number)
            oh12 = _required_float(row, "oh12", row_number)
            oh12_low = _required_float(row, "oh12_low", row_number)
            oh12_high = _required_float(row, "oh12_high", row_number)
            if logmstar_low > logmstar or logmstar_high < logmstar:
                raise ValueError(f"row {row_number} logmstar must lie inside logmstar_low/high")
            if oh12_low > oh12 or oh12_high < oh12:
                raise ValueError(f"row {row_number} oh12 must lie inside oh12_low/high")
            rows.append(
                MZRAnchor(
                    label=_required_text(row, "label", row_number),
                    source=_required_text(row, "source", row_number),
                    comparison_type=_required_text(row, "comparison_type", row_number),
                    redshift=redshift,
                    logmstar=logmstar,
                    logmstar_low=logmstar_low,
                    logmstar_high=logmstar_high,
                    oh12=oh12,
                    oh12_low=oh12_low,
                    oh12_high=oh12_high,
                    marker=_required_text(row, "marker", row_number),
                    color=_required_text(row, "color", row_number),
                    source_url=_required_text(row, "source_url", row_number),
                    notes=_required_text(row, "notes", row_number),
                )
            )
    return rows


def _dt_from_grid(cosmology: Cosmology, z_final: float, z_start_max: float, n_grid: int) -> float:
    from astropy.cosmology import FlatLambdaCDM

    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    t_start = float(astro.age(z_start_max).value)
    t_end = float(astro.age(z_final).value)
    return (t_end - t_start) / float(n_grid - 1)


def _percentiles(values: np.ndarray, name: str) -> tuple[float, float, float]:
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise RuntimeError(f"no finite values for {name}")
    p16, median, p84 = np.percentile(selected, [16.0, 50.0, 84.0])
    return float(median), float(p16), float(p84)


def _evaluate_point(
    *,
    redshift: float,
    logmh: float,
    redshift_index: int,
    mass_index: int,
    args: argparse.Namespace,
    cosmology: Cosmology,
    parameters: RegulatorMetallicityParameters,
) -> ModelPoint:
    dt_gyr = _dt_from_grid(cosmology, float(redshift), float(args.z_start_max), int(args.n_grid))
    histories = generate_halo_histories(
        n_tracks=int(args.n_tracks),
        z_final=float(redshift),
        Mh_final=float(10.0 ** float(logmh)),
        z_start_max=float(args.z_start_max),
        cosmology=cosmology,
        random_seed=int(args.random_seed + 10000 * redshift_index + 100 * mass_index),
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
    metallicity = compute_regulator_metallicity(
        t_grid_gyr=t_grid,
        z_grid=z_grid,
        mh_grid=mh_grid,
        sfr_grid=sfr_grid,
        active_grid=starforming_grid,
        cosmology=cosmology,
        parameters=parameters,
        random_seed=int(args.metallicity_random_seed + 10000 * redshift_index + 100 * mass_index),
    )
    final_active = starforming_grid[:, -1]
    mstar = np.asarray(metallicity.stellar_mass_msun_grid, dtype=float)[:, -1]
    zgas = np.asarray(metallicity.gas_metallicity_zsun_grid, dtype=float)[:, -1]
    selected = final_active & np.isfinite(mstar) & np.isfinite(zgas) & (mstar > 0.0) & (zgas > 0.0)
    if np.count_nonzero(selected) == 0:
        raise RuntimeError(f"no valid final star-forming tracks for z={redshift:g}, logMh={logmh:g}")
    logmstar = np.log10(mstar[selected])
    zgas_selected = zgas[selected]
    oh12 = np.asarray(equivalent_oxygen_abundance_from_zsun(zgas_selected), dtype=float)
    logmstar_median, logmstar_p16, logmstar_p84 = _percentiles(logmstar, "logMstar")
    zgas_median, zgas_p16, zgas_p84 = _percentiles(zgas_selected, "Zgas")
    oh12_median, oh12_p16, oh12_p84 = _percentiles(oh12, "12+log(O/H)")
    return ModelPoint(
        redshift=float(redshift),
        logmh=float(logmh),
        logmstar_median=logmstar_median,
        logmstar_p16=logmstar_p16,
        logmstar_p84=logmstar_p84,
        zgas_median_zsun=zgas_median,
        zgas_p16_zsun=zgas_p16,
        zgas_p84_zsun=zgas_p84,
        oh12_median=oh12_median,
        oh12_p16=oh12_p16,
        oh12_p84=oh12_p84,
        starforming_count=int(np.count_nonzero(selected)),
    )


def _relation_oh12(relation: MZRRelation, logmstar: np.ndarray | float, *, intercept: float | None = None) -> np.ndarray:
    x = np.asarray(logmstar, dtype=float)
    base = float(relation.intercept_oh12 if intercept is None else intercept)
    return base + float(relation.slope) * (x - float(relation.pivot_logmstar))


def _relations_for_redshift(relations: list[MZRRelation], redshift: float) -> list[MZRRelation]:
    return [
        relation
        for relation in relations
        if float(relation.redshift_min) <= float(redshift) <= float(relation.redshift_max)
    ]


def _anchors_for_redshift(anchors: list[MZRAnchor], redshift: float) -> list[MZRAnchor]:
    return [anchor for anchor in anchors if abs(float(anchor.redshift) - float(redshift)) <= 0.35]


def _compute_model_points(
    args: argparse.Namespace,
    parameters: RegulatorMetallicityParameters,
) -> list[ModelPoint]:
    cosmology = Cosmology()
    redshifts = sorted([float(item) for item in args.redshifts])
    log_masses = np.linspace(float(args.logM_min), float(args.logM_max), int(args.N_mass))
    rows: list[ModelPoint] = []
    for redshift_index, redshift in enumerate(redshifts):
        for mass_index, logmh in enumerate(log_masses):
            rows.append(
                _evaluate_point(
                    redshift=redshift,
                    logmh=float(logmh),
                    redshift_index=redshift_index,
                    mass_index=mass_index,
                    args=args,
                    cosmology=cosmology,
                    parameters=parameters,
                )
            )
        print(f"computed model MZR points at z={redshift:g}", flush=True)
    return rows


def _write_model_csv(path: Path, rows: list[ModelPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ModelPoint.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _write_residual_csv(
    path: Path,
    rows: list[ModelPoint],
    relations: list[MZRRelation],
) -> list[dict[str, float | str]]:
    residual_rows: list[dict[str, float | str]] = []
    for point in rows:
        for relation in _relations_for_redshift(relations, point.redshift):
            reference = float(_relation_oh12(relation, point.logmstar_median))
            residual_rows.append(
                {
                    "redshift": point.redshift,
                    "logmh": point.logmh,
                    "logmstar_median": point.logmstar_median,
                    "oh12_median": point.oh12_median,
                    "relation_label": relation.label,
                    "relation_oh12": reference,
                    "model_minus_relation_dex": point.oh12_median - reference,
                    "inside_relation_mass_range": (
                        float(relation.valid_logmstar_min)
                        <= float(point.logmstar_median)
                        <= float(relation.valid_logmstar_max)
                    ),
                }
            )
    if not residual_rows:
        raise RuntimeError("no model points overlap any MZR relation redshift ranges")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(residual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(residual_rows)
    return residual_rows


def _plot_validation(
    *,
    output_prefix: Path,
    rows: list[ModelPoint],
    relations: list[MZRRelation],
    anchors: list[MZRAnchor],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    plt.style.use("apj")
    redshifts = sorted({float(row.redshift) for row in rows})
    ncols = 2
    nrows = int(math.ceil(len(redshifts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6, 3.45 * nrows), sharex=False, sharey=True, constrained_layout=True)
    axis_array = np.atleast_1d(axes).ravel()
    all_logmstar = np.asarray([row.logmstar_median for row in rows], dtype=float)
    all_oh12 = np.asarray([row.oh12_median for row in rows], dtype=float)
    model_color = "#0173b2"

    for axis_index, ax in enumerate(axis_array):
        if axis_index >= len(redshifts):
            ax.axis("off")
            continue
        redshift = redshifts[axis_index]
        selected = [row for row in rows if float(row.redshift) == redshift]
        selected.sort(key=lambda item: item.logmstar_median)
        x = np.asarray([row.logmstar_median for row in selected], dtype=float)
        y = np.asarray([row.oh12_median for row in selected], dtype=float)
        xerr = [
            x - np.asarray([row.logmstar_p16 for row in selected], dtype=float),
            np.asarray([row.logmstar_p84 for row in selected], dtype=float) - x,
        ]
        yerr = [
            y - np.asarray([row.oh12_p16 for row in selected], dtype=float),
            np.asarray([row.oh12_p84 for row in selected], dtype=float) - y,
        ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker="o",
            ms=5.3,
            lw=1.5,
            capsize=2.2,
            color=model_color,
            mec="white",
            mew=0.6,
            label=r"AuroraLF at matched $M_\star,z$",
            zorder=5,
        )
        for point in selected:
            ax.text(
                point.logmstar_median + 0.025,
                point.oh12_median - 0.045,
                rf"${point.logmh:g}$",
                fontsize=6.4,
                color="0.28",
            )
        panel_relations = _relations_for_redshift(relations, redshift)
        for relation in panel_relations:
            xmin = min(float(np.min(x)) - 0.15, float(relation.valid_logmstar_min))
            xmax = max(float(np.max(x)) + 0.15, float(relation.valid_logmstar_max))
            x_relation = np.linspace(xmin, xmax, 200)
            y_relation = _relation_oh12(relation, x_relation)
            y_low = _relation_oh12(relation, x_relation, intercept=float(relation.intercept_low))
            y_high = _relation_oh12(relation, x_relation, intercept=float(relation.intercept_high))
            ax.plot(
                x_relation,
                y_relation,
                color=relation.color,
                lw=1.8,
                ls=relation.linestyle,
                label=relation.label,
            )
            ax.fill_between(x_relation, y_low, y_high, color=relation.color, alpha=0.12, lw=0.0)
        for anchor in _anchors_for_redshift(anchors, redshift):
            ax.errorbar(
                [anchor.logmstar],
                [anchor.oh12],
                xerr=[[anchor.logmstar - anchor.logmstar_low], [anchor.logmstar_high - anchor.logmstar]],
                yerr=[[anchor.oh12 - anchor.oh12_low], [anchor.oh12_high - anchor.oh12]],
                fmt=anchor.marker,
                ms=11 if anchor.marker == "*" else 6.5,
                color=anchor.color,
                mec="black",
                mew=0.6,
                capsize=2.7,
                label=anchor.label,
                zorder=8,
            )
        if not panel_relations and not _anchors_for_redshift(anchors, redshift):
            ax.text(
                0.04,
                0.92,
                "no adopted MZR relation at this redshift",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="0.35",
            )
        ax.set_title(rf"$z={redshift:g}$", fontsize=9.5)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=6.1, loc="upper left")
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        if axis_index % ncols == 0:
            ax.set_ylabel(r"$12+\log({\rm O/H})$")

    x_min = float(np.nanmin(all_logmstar)) - 0.35
    x_max = float(np.nanmax(all_logmstar)) + 0.35
    relation_min = []
    relation_max = []
    for relation in relations:
        x_relation = np.linspace(relation.valid_logmstar_min, relation.valid_logmstar_max, 50)
        relation_min.append(float(np.min(_relation_oh12(relation, x_relation, intercept=relation.intercept_low))))
        relation_max.append(float(np.max(_relation_oh12(relation, x_relation, intercept=relation.intercept_high))))
    anchor_oh = [anchor.oh12_low for anchor in anchors] + [anchor.oh12_high for anchor in anchors]
    y_min = min(float(np.nanmin(all_oh12)), min(relation_min), min(anchor_oh)) - 0.18
    y_max = max(float(np.nanmax(all_oh12)), max(relation_max), max(anchor_oh)) + 0.18
    for ax in axis_array[: len(redshifts)]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    fig.suptitle(
        rf"Observation-matched regulator $Z_{{gas}}(M_\star,z)$ vs MZR; "
        rf"${args.logM_min:g}<\log M_h<{args.logM_max:g}$",
        fontsize=10.0,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    parameters: RegulatorMetallicityParameters,
    relation_csv: Path,
    anchor_csv: Path,
    figure_pdf: Path,
    model_csv: Path,
    residual_csv: Path,
    residual_rows: list[dict[str, float | str]],
) -> None:
    by_relation: dict[str, list[float]] = {}
    for row in residual_rows:
        label = str(row["relation_label"])
        by_relation.setdefault(label, []).append(float(row["model_minus_relation_dex"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"redshifts: {' '.join(f'{float(item):g}' for item in args.redshifts)}\n")
        handle.write(f"logM range: {float(args.logM_min):g} {float(args.logM_max):g}\n")
        handle.write(f"N_mass: {int(args.N_mass)}\n")
        handle.write(f"n_tracks: {int(args.n_tracks)}\n")
        handle.write(f"n_grid: {int(args.n_grid)}\n")
        handle.write(f"enable_time_delay: {bool(args.enable_time_delay)}\n")
        handle.write(f"relation_csv: {relation_csv}\n")
        handle.write(f"anchor_csv: {anchor_csv}\n")
        handle.write(f"figure_pdf: {figure_pdf}\n")
        handle.write(f"model_csv: {model_csv}\n")
        handle.write(f"residual_csv: {residual_csv}\n")
        handle.write("\nRegulator parameters:\n")
        for key, value in parameters.as_metadata().items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nResidual summary by relation:\n")
        for label, values in sorted(by_relation.items()):
            array = np.asarray(values, dtype=float)
            handle.write(
                f"{label}: median={float(np.median(array)):.4g} dex, "
                f"rms={float(np.sqrt(np.mean(np.square(array)))):.4g} dex, "
                f"min={float(np.min(array)):.4g}, max={float(np.max(array)):.4g}\n"
            )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    relation_csv = _resolve_existing_path(str(args.relation_csv), "MZR relation CSV")
    anchor_csv = _resolve_existing_path(str(args.anchor_csv), "MZR anchor CSV")
    relations = _read_relations(relation_csv)
    anchors = _read_anchors(anchor_csv)
    output_prefix = _resolve_output_prefix(args.output_prefix)
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
    rows = _compute_model_points(args, parameters)
    data_prefix = PROJECT_ROOT / "data_save" / output_prefix.name
    model_csv = data_prefix.with_name(f"{data_prefix.name}_model_points.csv")
    residual_csv = data_prefix.with_name(f"{data_prefix.name}_residuals.csv")
    _write_model_csv(model_csv, rows)
    residual_rows = _write_residual_csv(residual_csv, rows, relations)
    png_path, pdf_path = _plot_validation(
        output_prefix=output_prefix,
        rows=rows,
        relations=relations,
        anchors=anchors,
        args=args,
    )
    summary_path = output_prefix.with_suffix(".txt")
    _write_summary(
        path=summary_path,
        args=args,
        parameters=parameters,
        relation_csv=relation_csv,
        anchor_csv=anchor_csv,
        figure_pdf=pdf_path,
        model_csv=model_csv,
        residual_csv=residual_csv,
        residual_rows=residual_rows,
    )
    print(f"saved: {png_path} {pdf_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
