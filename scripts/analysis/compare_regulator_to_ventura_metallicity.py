#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.chemistry import RegulatorMetallicityParameters, compute_regulator_metallicity
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters, compute_sfr_from_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data_save" / "ventura2024_qmhz.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data_save" / "regulator_vs_ventura2024_metallicity.csv"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "regulator_vs_ventura2024_metallicity"
YEARS_PER_GYR = 1.0e9


@dataclass(frozen=True)
class VenturaRow:
    redshift: float
    log_halo_mass: float
    q_z: float
    n_halos: int
    z_polluted_zsun: float


@dataclass(frozen=True)
class ComparisonRow:
    redshift: float
    log_halo_mass: float
    q_z: float
    n_halos: int
    z_polluted_zsun: float
    model_zgas_median_zsun: float
    model_zgas_p16_zsun: float
    model_zgas_p84_zsun: float
    model_birth_z_median_zsun: float
    model_stellar_mass_median_msun: float
    model_gas_mass_median_msun: float
    model_metal_mass_median_msun: float
    final_starforming_fraction: float
    final_active_fraction: float
    log10_model_to_ventura: float
    status: str


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _read_ventura_table(path: Path) -> list[VenturaRow]:
    if not path.exists():
        raise FileNotFoundError(f"V2024 Q(Mh,z) table not found: {path}")
    required = {"redshift", "log_halo_mass", "Q_Z", "n_halos", "Z_polluted_zsun"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"V2024 table has no header: {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"V2024 table missing column(s): {', '.join(sorted(missing))}")
        rows = []
        for raw in reader:
            row = VenturaRow(
                redshift=float(raw["redshift"]),
                log_halo_mass=float(raw["log_halo_mass"]),
                q_z=float(raw["Q_Z"]),
                n_halos=int(raw["n_halos"]),
                z_polluted_zsun=float(raw["Z_polluted_zsun"]),
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"V2024 table has no data rows: {path}")

    redshift = np.asarray([row.redshift for row in rows], dtype=float)
    log_halo_mass = np.asarray([row.log_halo_mass for row in rows], dtype=float)
    q_z = np.asarray([row.q_z for row in rows], dtype=float)
    n_halos = np.asarray([row.n_halos for row in rows], dtype=int)
    z_polluted = np.asarray([row.z_polluted_zsun for row in rows], dtype=float)
    if (
        np.any(~np.isfinite(redshift))
        or np.any(~np.isfinite(log_halo_mass))
        or np.any(~np.isfinite(q_z))
        or np.any(~np.isfinite(z_polluted))
    ):
        raise ValueError("V2024 table contains non-finite numeric values")
    if np.any((q_z < 0.0) | (q_z > 1.0)):
        raise ValueError("V2024 Q_Z values must lie in [0, 1]")
    if np.any(n_halos <= 0):
        raise ValueError("V2024 n_halos values must be positive")
    if np.any(z_polluted <= 0.0):
        raise ValueError("V2024 Z_polluted_zsun values must be positive for log-space comparison")
    return rows


def _dt_from_grid(cosmology: Cosmology, z_final: float, z_start_max: float, n_grid: int) -> float:
    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    t_start = float(astro.age(float(z_start_max)).value)
    t_end = float(astro.age(float(z_final)).value)
    if t_end <= t_start:
        raise ValueError("z_start_max must be greater than every V2024 comparison redshift")
    return (t_end - t_start) / float(int(n_grid) - 1)


def _reshape_tracks(
    tracks: dict[str, np.ndarray],
    *,
    n_tracks: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_grid = np.asarray(tracks["t_gyr"], dtype=float).reshape(n_tracks, n_steps)
    z_grid = np.asarray(tracks["z"], dtype=float).reshape(n_tracks, n_steps)
    mh_grid = np.asarray(tracks["Mh"], dtype=float).reshape(n_tracks, n_steps)
    sfr_grid = np.asarray(tracks["SFR"], dtype=float).reshape(n_tracks, n_steps)
    active_grid = np.asarray(tracks["active_flag"], dtype=bool).reshape(n_tracks, n_steps)
    return t_grid, z_grid, mh_grid, sfr_grid, active_grid


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _median_positive(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _compare_row(
    row: VenturaRow,
    *,
    row_index: int,
    n_tracks: int,
    n_grid: int,
    z_start_max: float,
    random_seed: int,
    enable_time_delay: bool,
    cosmology: Cosmology,
    sfr_parameters: SFRModelParameters,
    regulator_parameters: RegulatorMetallicityParameters,
) -> ComparisonRow:
    dt_gyr = _dt_from_grid(cosmology, row.redshift, z_start_max, n_grid)
    histories = generate_halo_histories(
        n_tracks=int(n_tracks),
        z_final=float(row.redshift),
        Mh_final=float(10.0 ** row.log_halo_mass),
        z_start_max=float(z_start_max),
        M_min=None,
        cosmology=cosmology,
        random_seed=int(random_seed + 1000 * row_index),
        time_grid_mode="uniform_in_t",
        dt=dt_gyr,
        store_inactive_history=True,
        sampler="mcbride",
    )
    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        cosmology=cosmology,
        enable_time_delay=bool(enable_time_delay),
        model_parameters=sfr_parameters,
    )
    n_steps = int(histories.metadata["grid_size"])
    t_grid, z_grid, mh_grid, sfr_grid, active_grid = _reshape_tracks(
        sfr_tracks,
        n_tracks=int(n_tracks),
        n_steps=n_steps,
    )
    starforming_grid = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)
    final_active = np.asarray(active_grid[:, -1], dtype=bool)
    final_starforming = np.asarray(starforming_grid[:, -1], dtype=bool)

    if not np.any(starforming_grid):
        return ComparisonRow(
            redshift=row.redshift,
            log_halo_mass=row.log_halo_mass,
            q_z=row.q_z,
            n_halos=row.n_halos,
            z_polluted_zsun=row.z_polluted_zsun,
            model_zgas_median_zsun=float("nan"),
            model_zgas_p16_zsun=float("nan"),
            model_zgas_p84_zsun=float("nan"),
            model_birth_z_median_zsun=float("nan"),
            model_stellar_mass_median_msun=float("nan"),
            model_gas_mass_median_msun=float("nan"),
            model_metal_mass_median_msun=float("nan"),
            final_starforming_fraction=float(np.mean(final_starforming)),
            final_active_fraction=float(np.mean(final_active)),
            log10_model_to_ventura=float("nan"),
            status="no_starforming_steps",
        )

    regulator_result = compute_regulator_metallicity(
        t_grid_gyr=t_grid,
        z_grid=z_grid,
        mh_grid=mh_grid,
        sfr_grid=sfr_grid,
        active_grid=starforming_grid,
        cosmology=cosmology,
        parameters=regulator_parameters,
        random_seed=int(random_seed + 1000 * row_index),
    )
    final_mask = final_starforming
    final_zgas = regulator_result.gas_metallicity_zsun_grid[:, -1][final_mask]
    final_birth = regulator_result.birth_metallicity_zsun_grid[:, -1][final_mask]
    final_stellar = regulator_result.stellar_mass_msun_grid[:, -1][final_mask]
    final_gas = regulator_result.gas_mass_grid[:, -1][final_mask]
    final_metal = regulator_result.metal_mass_grid[:, -1][final_mask]
    median_zgas = _median_positive(final_zgas)
    log_offset = (
        float(np.log10(median_zgas / float(row.z_polluted_zsun)))
        if np.isfinite(median_zgas) and median_zgas > 0.0
        else float("nan")
    )
    status = "ok" if np.isfinite(median_zgas) else "inactive_at_final_redshift"
    return ComparisonRow(
        redshift=row.redshift,
        log_halo_mass=row.log_halo_mass,
        q_z=row.q_z,
        n_halos=row.n_halos,
        z_polluted_zsun=row.z_polluted_zsun,
        model_zgas_median_zsun=median_zgas,
        model_zgas_p16_zsun=_finite_percentile(final_zgas, 16.0),
        model_zgas_p84_zsun=_finite_percentile(final_zgas, 84.0),
        model_birth_z_median_zsun=_median_positive(final_birth),
        model_stellar_mass_median_msun=_median_positive(final_stellar),
        model_gas_mass_median_msun=_median_positive(final_gas),
        model_metal_mass_median_msun=_median_positive(final_metal),
        final_starforming_fraction=float(np.mean(final_starforming)),
        final_active_fraction=float(np.mean(final_active)),
        log10_model_to_ventura=log_offset,
        status=status,
    )


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.12g}"


def _write_comparison_csv(path: Path, rows: list[ComparisonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "redshift",
        "log_halo_mass",
        "Q_Z",
        "n_halos",
        "Z_polluted_zsun_v2024",
        "Zgas_zsun_auroralf_median",
        "Zgas_zsun_auroralf_p16",
        "Zgas_zsun_auroralf_p84",
        "Zbirth_zsun_auroralf_median",
        "Mstar_msun_auroralf_median",
        "Mgas_msun_auroralf_median",
        "Mmetal_msun_auroralf_median",
        "final_starforming_fraction",
        "final_active_fraction",
        "log10_Zgas_over_Zpolluted",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "redshift": _format_float(row.redshift),
                    "log_halo_mass": _format_float(row.log_halo_mass),
                    "Q_Z": _format_float(row.q_z),
                    "n_halos": str(int(row.n_halos)),
                    "Z_polluted_zsun_v2024": _format_float(row.z_polluted_zsun),
                    "Zgas_zsun_auroralf_median": _format_float(row.model_zgas_median_zsun),
                    "Zgas_zsun_auroralf_p16": _format_float(row.model_zgas_p16_zsun),
                    "Zgas_zsun_auroralf_p84": _format_float(row.model_zgas_p84_zsun),
                    "Zbirth_zsun_auroralf_median": _format_float(row.model_birth_z_median_zsun),
                    "Mstar_msun_auroralf_median": _format_float(row.model_stellar_mass_median_msun),
                    "Mgas_msun_auroralf_median": _format_float(row.model_gas_mass_median_msun),
                    "Mmetal_msun_auroralf_median": _format_float(row.model_metal_mass_median_msun),
                    "final_starforming_fraction": _format_float(row.final_starforming_fraction),
                    "final_active_fraction": _format_float(row.final_active_fraction),
                    "log10_Zgas_over_Zpolluted": _format_float(row.log10_model_to_ventura),
                    "status": row.status,
                }
            )


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("centers must be a non-empty 1D array")
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("centers must be strictly increasing")
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _grid_from_rows(rows: list[ComparisonRow], attr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    redshifts = np.array(sorted({row.redshift for row in rows}), dtype=float)
    log_masses = np.array(sorted({row.log_halo_mass for row in rows}), dtype=float)
    grid = np.full((redshifts.size, log_masses.size), np.nan, dtype=float)
    z_index = {float(value): index for index, value in enumerate(redshifts)}
    mass_index = {float(value): index for index, value in enumerate(log_masses)}
    for row in rows:
        grid[z_index[float(row.redshift)], mass_index[float(row.log_halo_mass)]] = float(getattr(row, attr))
    return redshifts, log_masses, grid


def _positive_lognorm(values: np.ndarray):
    from matplotlib.colors import LogNorm

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        raise RuntimeError("cannot plot log-scaled heatmap with no positive finite values")
    return LogNorm(vmin=float(np.min(finite)), vmax=float(np.max(finite)))


def _make_plot(rows: list[ComparisonRow], output_prefix: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.style.use("apj")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    redshifts, log_masses, ventura_z = _grid_from_rows(rows, "z_polluted_zsun")
    _, _, model_z = _grid_from_rows(rows, "model_zgas_median_zsun")
    _, _, log_offset = _grid_from_rows(rows, "log10_model_to_ventura")
    logm_edges = _centers_to_edges(log_masses)
    z_edges = _centers_to_edges(redshifts)

    finite_offset = log_offset[np.isfinite(log_offset)]
    if finite_offset.size == 0:
        raise RuntimeError("no finite AuroraLF/V2024 metallicity offsets to plot")
    offset_limit = max(0.1, float(np.nanmax(np.abs(finite_offset))))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), constrained_layout=True)
    panels = [
        (axes[0, 0], ventura_z, "V2024 polluted IGM", r"$Z_{\rm polluted}/Z_\odot$", _positive_lognorm(ventura_z)),
        (axes[0, 1], model_z, "AuroraLF 1D regulator", r"$Z_{\rm gas}/Z_\odot$", _positive_lognorm(model_z)),
        (
            axes[1, 0],
            log_offset,
            "AuroraLF / V2024",
            r"$\log_{10}(Z_{\rm gas}/Z_{\rm polluted})$",
            TwoSlopeNorm(vmin=-offset_limit, vcenter=0.0, vmax=offset_limit),
        ),
    ]
    for ax, values, title, label, norm in panels:
        mesh = ax.pcolormesh(logm_edges, z_edges, values, shading="auto", norm=norm)
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
        ax.set_ylabel(r"$z$")
        ax.set_xticks(log_masses)
        ax.set_yticks(redshifts)

    ax = axes[1, 1]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.85, log_masses.size))
    for color, log_mass in zip(colors, log_masses, strict=True):
        selected = [row for row in rows if row.log_halo_mass == float(log_mass)]
        selected.sort(key=lambda item: item.redshift)
        z_values = np.asarray([row.redshift for row in selected], dtype=float)
        v_values = np.asarray([row.z_polluted_zsun for row in selected], dtype=float)
        m_values = np.asarray([row.model_zgas_median_zsun for row in selected], dtype=float)
        ax.plot(z_values, v_values, color=color, linestyle="--", marker="o", label=rf"V2024 $\log M_h={log_mass:g}$")
        finite_model = np.isfinite(m_values) & (m_values > 0.0)
        if np.any(finite_model):
            ax.plot(
                z_values[finite_model],
                m_values[finite_model],
                color=color,
                linestyle="-",
                marker="s",
                label=rf"AuroraLF $\log M_h={log_mass:g}$",
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$Z/Z_\odot$")
    ax.set_title("Mass-bin tracks")
    ax.legend(fontsize=8, ncols=1)

    fig.suptitle("AuroraLF 1D MAH regulator vs V2024/Meraxes polluted IGM metallicity")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AuroraLF 1D McBride MAH + regulator gas metallicities with "
            "the V2024/Meraxes Q(Mh,z) polluted IGM metallicity table."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--n-tracks", type=int, default=96)
    parser.add_argument("--n-grid", type=int, default=120)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--random-seed", type=int, default=24680)
    parser.add_argument("--disable-time-delay", action="store_true")
    parser.add_argument("--epsilon-0", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.epsilon_0)
    parser.add_argument("--fstar-characteristic-mass", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.characteristic_mass)
    parser.add_argument("--fstar-beta", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.beta_star)
    parser.add_argument("--fstar-gamma", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.gamma_star)
    parser.add_argument("--regulator-gas-fraction-norm", type=float, default=0.02)
    parser.add_argument("--regulator-gas-fraction-mass-slope", type=float, default=0.0)
    parser.add_argument("--regulator-gas-fraction-redshift-slope", type=float, default=0.0)
    parser.add_argument("--regulator-yield", type=float, default=0.01)
    parser.add_argument("--regulator-returned-fraction", type=float, default=0.4)
    parser.add_argument("--regulator-inflow-metallicity-zsun", type=float, default=0.0)
    parser.add_argument("--regulator-metal-loading-norm", type=float, default=20.0)
    parser.add_argument("--regulator-metal-loading-mass-slope", type=float, default=-0.5)
    parser.add_argument("--regulator-metal-loading-redshift-slope", type=float, default=0.0)
    parser.add_argument("--regulator-metallicity-scatter-dex", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if int(args.n_tracks) <= 0:
        raise ValueError("--n-tracks must be positive")
    if int(args.n_grid) < 2:
        raise ValueError("--n-grid must be at least 2")
    if float(args.z_start_max) <= 0.0:
        raise ValueError("--z-start-max must be positive")

    input_csv = _resolve_path(args.input_csv)
    output_csv = _resolve_path(args.output_csv)
    output_prefix = _resolve_path(args.output_prefix)
    ventura_rows = _read_ventura_table(input_csv)

    sfr_parameters = SFRModelParameters(
        epsilon_0=float(args.epsilon_0),
        characteristic_mass=float(args.fstar_characteristic_mass),
        beta_star=float(args.fstar_beta),
        gamma_star=float(args.fstar_gamma),
    )
    regulator_parameters = RegulatorMetallicityParameters(
        gas_fraction_norm=float(args.regulator_gas_fraction_norm),
        gas_fraction_mass_slope=float(args.regulator_gas_fraction_mass_slope),
        gas_fraction_redshift_slope=float(args.regulator_gas_fraction_redshift_slope),
        metal_yield=float(args.regulator_yield),
        returned_fraction=float(args.regulator_returned_fraction),
        inflow_metallicity_zsun=float(args.regulator_inflow_metallicity_zsun),
        metal_loading_norm=float(args.regulator_metal_loading_norm),
        metal_loading_mass_slope=float(args.regulator_metal_loading_mass_slope),
        metal_loading_redshift_slope=float(args.regulator_metal_loading_redshift_slope),
        metallicity_scatter_dex=float(args.regulator_metallicity_scatter_dex),
    )
    cosmology = Cosmology()
    comparison_rows = [
        _compare_row(
            row,
            row_index=index,
            n_tracks=int(args.n_tracks),
            n_grid=int(args.n_grid),
            z_start_max=float(args.z_start_max),
            random_seed=int(args.random_seed),
            enable_time_delay=not bool(args.disable_time_delay),
            cosmology=cosmology,
            sfr_parameters=sfr_parameters,
            regulator_parameters=regulator_parameters,
        )
        for index, row in enumerate(ventura_rows)
    ]
    _write_comparison_csv(output_csv, comparison_rows)
    _make_plot(comparison_rows, output_prefix)

    finite_offsets = np.asarray(
        [row.log10_model_to_ventura for row in comparison_rows if np.isfinite(row.log10_model_to_ventura)],
        dtype=float,
    )
    if finite_offsets.size == 0:
        raise RuntimeError("comparison produced no finite AuroraLF/V2024 metallicity offsets")
    print(f"rows: {len(comparison_rows)}")
    print(f"finite_offsets: {finite_offsets.size}")
    print(f"median_log10_Zgas_over_Zpolluted: {float(np.median(finite_offsets)):.6g}")
    print(f"min_log10_Zgas_over_Zpolluted: {float(np.min(finite_offsets)):.6g}")
    print(f"max_log10_Zgas_over_Zpolluted: {float(np.max(finite_offsets)):.6g}")
    print(f"output_csv: {output_csv}")
    print(f"output_pdf: {output_prefix.with_suffix('.pdf')}")
    print(f"output_png: {output_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
