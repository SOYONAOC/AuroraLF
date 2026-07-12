#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.mah import Cosmology
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import POPIII_UPPER_MASS_MODE_ATOMIC, POPIII_UPPER_MASS_MODE_FIXED, PopIIISFRParameters
from auroralf.ssp import (
    DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON,
    compute_final_ssp_heplus_rate_from_sfr_grid,
    heii1640_luminosity_from_heplus_rate,
    load_popiii_heplus_ionizing_photon_table,
)
from auroralf.uvlf import run_halo_uv_pipeline


HEBE_FLUX_ERG_S_CM2 = 1.11e-19
HEBE_FLUX_ERR_ERG_S_CM2 = 0.17e-19
HEBE_COMPONENT_C1_FLUX_ERG_S_CM2 = 6.7e-20
HEBE_COMPONENT_C2_FLUX_ERG_S_CM2 = 5.1e-20
HEBE_VELOCITY_SEPARATION_KMS = 126.0
HEII1640_REST_A = 1640.42
SPEED_OF_LIGHT_KMS = 299792.458

DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "popiii_heii_actual_sfh_vs_hebe"
DEFAULT_HEBE_OBSERVATION_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "observations"
    / "hebe"
    / "maiolino_rusta_2026_heii_constraints.csv"
)
DEFAULT_EXTREME_POPIII_UV_SSP_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "schaerer2010_pop3"
    / "pop3_ge0_sal_500_050_is4.25"
)
DEFAULT_EXTREME_POPIII_HEII1640_SSP_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "schaerer2010_pop3"
    / "pop3_ge0_sal_500_050_is4.22"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AuroraLF Pop III SFH-convolved HeII 1640 fluxes to the Hebe JWST fit."
    )
    parser.add_argument("--z", type=float, default=10.583)
    parser.add_argument("--logM-min", type=float, default=6.0)
    parser.add_argument("--logM-max", type=float, default=11.0)
    parser.add_argument("--N-mass", type=int, default=12)
    parser.add_argument("--n-tracks", type=int, default=8)
    parser.add_argument("--n-grid", type=int, default=480)
    parser.add_argument("--random-seed", type=int, default=106)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--heii-lookback-max-myr", type=float, default=10.0)
    parser.add_argument("--young-mass-window-myr", type=float, default=3.0)
    parser.add_argument("--lens-magnification", type=float, default=1.42)
    parser.add_argument("--heii-caseb-erg-per-photon", type=float, default=DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON)
    parser.add_argument("--heplus-covering-factor", type=float, default=1.0)
    parser.add_argument("--heplus-escape-fraction", type=float, default=0.0)
    parser.add_argument("--heii-photoionization-efficiency", type=float, default=1.0)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--popiii-epsilon-star", type=float, default=1.0e-3)
    parser.add_argument("--popiii-mp", type=float, default=1.0e7)
    parser.add_argument("--popiii-alpha-star", type=float, default=0.0)
    parser.add_argument("--popiii-beta-star", type=float, default=0.0)
    parser.add_argument("--popiii-upper-mass-mode", choices=(POPIII_UPPER_MASS_MODE_ATOMIC, POPIII_UPPER_MASS_MODE_FIXED), default=POPIII_UPPER_MASS_MODE_ATOMIC)
    parser.add_argument("--popiii-upper-mass-msun", type=float, default=None)
    parser.add_argument("--popiii-uv-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_UV_SSP_FILE)
    parser.add_argument("--heii-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_HEII1640_SSP_FILE)
    parser.add_argument("--hebe-observation-file", type=Path, default=DEFAULT_HEBE_OBSERVATION_FILE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.z <= 0.0:
        raise ValueError("--z must be positive")
    if args.logM_max <= args.logM_min:
        raise ValueError("--logM-max must be larger than --logM-min")
    if args.N_mass <= 0:
        raise ValueError("--N-mass must be positive")
    if args.n_tracks <= 0:
        raise ValueError("--n-tracks must be positive")
    if args.n_grid <= 2:
        raise ValueError("--n-grid must be greater than 2")
    if args.z_start_max <= args.z:
        raise ValueError("--z-start-max must be greater than --z")
    if args.heii_lookback_max_myr <= 0.0:
        raise ValueError("--heii-lookback-max-myr must be positive")
    if args.young_mass_window_myr <= 0.0:
        raise ValueError("--young-mass-window-myr must be positive")
    if args.lens_magnification <= 0.0:
        raise ValueError("--lens-magnification must be positive")
    if args.heii_caseb_erg_per_photon <= 0.0:
        raise ValueError("--heii-caseb-erg-per-photon must be positive")
    if not 0.0 <= args.heplus_covering_factor <= 1.0:
        raise ValueError("--heplus-covering-factor must lie in [0, 1]")
    if not 0.0 <= args.heplus_escape_fraction <= 1.0:
        raise ValueError("--heplus-escape-fraction must lie in [0, 1]")
    if args.heii_photoionization_efficiency < 0.0:
        raise ValueError("--heii-photoionization-efficiency must be non-negative")
    if not 0.0 <= args.popiii_epsilon_star <= 1.0:
        raise ValueError("--popiii-epsilon-star must lie in [0, 1]")
    if args.popiii_mp <= 0.0:
        raise ValueError("--popiii-mp must be positive")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")
    if args.popiii_upper_mass_mode == POPIII_UPPER_MASS_MODE_FIXED:
        if args.popiii_upper_mass_msun is None or args.popiii_upper_mass_msun <= 0.0:
            raise ValueError("--popiii-upper-mass-msun must be positive when using fixed upper-mass mode")


def _build_cosmology(cosmology: Cosmology) -> FlatLambdaCDM:
    return FlatLambdaCDM(
        H0=cosmology.h0_km_s_mpc,
        Om0=cosmology.omega_m,
        Ob0=cosmology.omega_b,
    )


def _load_hebe_observation_constraints(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    required_columns = {"quantity", "component", "value", "err", "unit", "source", "notes"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Hebe observation file is empty: {path}")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Hebe observation file is missing required columns: {sorted(missing)}")
        constraints: dict[tuple[str, str], dict[str, float | str]] = {}
        for row_number, row in enumerate(reader, start=2):
            quantity = str(row["quantity"]).strip()
            component = str(row["component"]).strip()
            if not quantity or not component:
                raise ValueError(f"Hebe observation file has blank quantity/component at row {row_number}")
            key = (quantity, component)
            if key in constraints:
                raise ValueError(f"Hebe observation file has duplicate constraint {key} at row {row_number}")
            err_text = str(row["err"]).strip()
            constraints[key] = {
                "value": float(row["value"]),
                "err": float(err_text) if err_text else float("nan"),
                "unit": str(row["unit"]).strip(),
                "source": str(row["source"]).strip(),
                "notes": str(row["notes"]).strip(),
            }
    return constraints


def _require_constraint(
    constraints: dict[tuple[str, str], dict[str, float | str]],
    quantity: str,
    component: str,
) -> dict[str, float | str]:
    key = (quantity, component)
    if key not in constraints:
        raise KeyError(f"Hebe observation constraint is missing required row: {key}")
    return constraints[key]


def _integrate_recent_mass_msun(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    window_myr: float,
) -> np.ndarray:
    t_grid = np.asarray(t_grid_gyr, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    if t_grid.shape != sfr.shape or t_grid.shape != active.shape:
        raise ValueError("t_grid_gyr, sfr_grid, and active_grid must have identical shapes")
    result = np.zeros(t_grid.shape[0], dtype=float)
    window_gyr = float(window_myr) / 1.0e3
    for row_index in range(t_grid.shape[0]):
        t_row = t_grid[row_index]
        if np.any(np.diff(t_row) <= 0.0):
            raise ValueError("each t_grid_gyr row must be strictly increasing")
        t_obs = float(t_row[-1])
        lower = max(float(t_row[0]), t_obs - window_gyr)
        source_sfr = np.where(active[row_index], sfr[row_index], 0.0)
        interior = t_row[(t_row > lower) & (t_row < t_obs)]
        t_used = np.unique(np.concatenate((np.array([lower], dtype=float), interior, np.array([t_obs]))))
        sfr_used = np.interp(t_used, t_row, source_sfr)
        result[row_index] = float(np.trapezoid(sfr_used, x=t_used * 1.0e9))
    return result


def _hebe_line_template(
    lambda_um: np.ndarray,
    *,
    total_flux: float,
    lambda_center_um: float,
    component_c1_flux: float,
    component_c2_flux: float,
    velocity_separation_kms: float,
) -> np.ndarray:
    c1_fraction = component_c1_flux / (component_c1_flux + component_c2_flux)
    c2_fraction = 1.0 - c1_fraction
    sigma_inst_kms = SPEED_OF_LIGHT_KMS / (2.355 * 2700.0)
    sigma_intr_kms = 25.0
    sigma_kms = float(np.sqrt(sigma_inst_kms**2 + sigma_intr_kms**2))
    velocity_kms = SPEED_OF_LIGHT_KMS * (lambda_um / lambda_center_um - 1.0)
    lambda_center_a = lambda_center_um * 1.0e4

    def profile_component(v0_kms: float, flux: float) -> np.ndarray:
        profile_per_kms = flux / (np.sqrt(2.0 * np.pi) * sigma_kms) * np.exp(
            -0.5 * ((velocity_kms - v0_kms) / sigma_kms) ** 2
        )
        return profile_per_kms * SPEED_OF_LIGHT_KMS / lambda_center_a

    separation = float(velocity_separation_kms)
    return profile_component(-0.5 * separation, total_flux * c1_fraction) + profile_component(
        0.5 * separation,
        total_flux * c2_fraction,
    )


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "logMh_final",
        "Mh_final_msun",
        "track_index",
        "heplus_photon_rate_s",
        "line_luminosity_erg_s",
        "flux_erg_s_cm2",
        "flux_ratio_to_hebe",
        "luminosity_ratio_to_hebe_intrinsic",
        "recent_popiii_mass_msun",
        "final_popiii_sfr_msun_yr",
        "popiii_uv_luminosity_erg_s_hz",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    *,
    rows: list[dict[str, float]],
    output_prefix: Path,
    z: float,
    lens_magnification: float,
    hebe_flux_erg_s_cm2: float,
    hebe_flux_err_erg_s_cm2: float,
    hebe_component_c1_flux_erg_s_cm2: float,
    hebe_component_c2_flux_erg_s_cm2: float,
    hebe_velocity_separation_kms: float,
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

    logmh = np.asarray([row["logMh_final"] for row in rows], dtype=float)
    flux_ratio = np.asarray([row["flux_ratio_to_hebe"] for row in rows], dtype=float)
    recent_mass = np.asarray([row["recent_popiii_mass_msun"] for row in rows], dtype=float)
    unique_logmh = np.unique(logmh)
    median = np.array([np.median(flux_ratio[logmh == value]) for value in unique_logmh])
    p16 = np.array([np.percentile(flux_ratio[logmh == value], 16.0) for value in unique_logmh])
    p84 = np.array([np.percentile(flux_ratio[logmh == value], 84.0) for value in unique_logmh])
    max_index = int(np.nanargmax(flux_ratio))
    max_flux = rows[max_index]["flux_erg_s_cm2"]

    lambda_center_um = HEII1640_REST_A * (1.0 + z) / 1.0e4
    velocity = np.linspace(-400.0, 400.0, 1600)
    lambda_um = lambda_center_um * (1.0 + velocity / SPEED_OF_LIGHT_KMS)
    observed_profile = _hebe_line_template(
        lambda_um,
        total_flux=hebe_flux_erg_s_cm2,
        lambda_center_um=lambda_center_um,
        component_c1_flux=hebe_component_c1_flux_erg_s_cm2,
        component_c2_flux=hebe_component_c2_flux_erg_s_cm2,
        velocity_separation_kms=hebe_velocity_separation_kms,
    )
    observed_lo = _hebe_line_template(
        lambda_um,
        total_flux=hebe_flux_erg_s_cm2 - hebe_flux_err_erg_s_cm2,
        lambda_center_um=lambda_center_um,
        component_c1_flux=hebe_component_c1_flux_erg_s_cm2,
        component_c2_flux=hebe_component_c2_flux_erg_s_cm2,
        velocity_separation_kms=hebe_velocity_separation_kms,
    )
    observed_hi = _hebe_line_template(
        lambda_um,
        total_flux=hebe_flux_erg_s_cm2 + hebe_flux_err_erg_s_cm2,
        lambda_center_um=lambda_center_um,
        component_c1_flux=hebe_component_c1_flux_erg_s_cm2,
        component_c2_flux=hebe_component_c2_flux_erg_s_cm2,
        velocity_separation_kms=hebe_velocity_separation_kms,
    )
    model_profile = _hebe_line_template(
        lambda_um,
        total_flux=max_flux,
        lambda_center_um=lambda_center_um,
        component_c1_flux=hebe_component_c1_flux_erg_s_cm2,
        component_c2_flux=hebe_component_c2_flux_erg_s_cm2,
        velocity_separation_kms=hebe_velocity_separation_kms,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    ax_ratio, ax_line = axes
    ax_ratio.scatter(logmh, flux_ratio, c=np.log10(np.clip(recent_mass, 1.0, None)), s=20, alpha=0.62, cmap="viridis")
    ax_ratio.plot(unique_logmh, median, color="black", lw=1.8, label="median")
    ax_ratio.fill_between(unique_logmh, p16, p84, color="black", alpha=0.14, linewidth=0, label="16-84%")
    ax_ratio.axhline(1.0, color="#b91c1c", lw=1.3, ls="--", label="Hebe flux")
    ax_ratio.set_yscale("log")
    y_positive = flux_ratio[flux_ratio > 0.0]
    if y_positive.size:
        ax_ratio.set_ylim(max(np.min(y_positive) * 0.45, 1.0e-8), max(np.max(y_positive) * 2.5, 2.0))
    ax_ratio.set_xlabel(r"$\log_{10}(M_{\rm h,final}/M_\odot)$")
    ax_ratio.set_ylabel(r"$F_{\rm HeII}^{\rm model}/F_{\rm HeII}^{\rm Hebe}$")
    ax_ratio.set_title(r"SFH-convolved Pop III HeII")
    ax_ratio.grid(True, which="major", alpha=0.22)
    ax_ratio.grid(True, which="minor", alpha=0.08)
    ax_ratio.legend(loc="upper left", frameon=False)

    scale = 1.0e-20
    ax_line.fill_between(lambda_um, observed_lo / scale, observed_hi / scale, color="0.2", alpha=0.14, linewidth=0)
    ax_line.plot(lambda_um, observed_profile / scale, color="#111827", lw=2.3, label="Hebe observed fit")
    ax_line.plot(lambda_um, model_profile / scale, color="#15803d", lw=2.0, ls="--", label="brightest AuroraLF track")
    ax_line.set_xlabel(r"observed wavelength [$\mu$m]")
    ax_line.set_ylabel(r"$F_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax_line.set_title(r"Brightest model line profile")
    ax_line.grid(True, which="major", alpha=0.22)
    ax_line.grid(True, which="minor", alpha=0.08)
    ax_line.legend(loc="upper right", frameon=False)
    ax_line.text(
        0.03,
        0.93,
        rf"$F_{{\rm max}}/F_{{\rm Hebe}}={flux_ratio[max_index]:.2g}$" + "\n"
        + rf"$M_{{\rm h}}=10^{{{rows[max_index]['logMh_final']:.2f}}}M_\odot$" + "\n"
        + rf"$M_{{\rm PopIII,recent}}={rows[max_index]['recent_popiii_mass_msun']:.2e}M_\odot$",
        transform=ax_line.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )

    fig.text(
        0.5,
        -0.03,
        rf"z={z:.3f}, lens magnification $\mu={lens_magnification:.2f}$. Line profile uses Hebe velocity template; AuroraLF predicts total HeII luminosity from Pop III SFH convolution.",
        ha="center",
        va="top",
        fontsize=7.4,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500, bbox_inches="tight")


def main() -> None:
    args = _parse_args()
    aurora_cosmology = Cosmology()
    _validate_args(args)
    output_prefix = _resolve_project_path(args.output_prefix)
    popiii_uv_ssp_file = _resolve_project_path(args.popiii_uv_ssp_file)
    heii_ssp_file = _resolve_project_path(args.heii_ssp_file)
    hebe_observation_file = _resolve_project_path(args.hebe_observation_file)
    if not popiii_uv_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III UV SSP file not found: {popiii_uv_ssp_file}")
    if not heii_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III HeII SSP file not found: {heii_ssp_file}")
    if not hebe_observation_file.is_file():
        raise FileNotFoundError(f"Hebe observation file not found: {hebe_observation_file}")

    hebe_constraints = _load_hebe_observation_constraints(hebe_observation_file)
    hebe_flux = float(_require_constraint(hebe_constraints, "heii1640_flux_observed", "total")["value"])
    hebe_flux_err = float(_require_constraint(hebe_constraints, "heii1640_flux_observed", "total")["err"])
    hebe_luminosity_intrinsic = float(
        _require_constraint(hebe_constraints, "heii1640_luminosity_intrinsic_clean", "total")["value"]
    )
    hebe_component_c1_flux = float(_require_constraint(hebe_constraints, "heii1640_flux_observed", "C1")["value"])
    hebe_component_c2_flux = float(_require_constraint(hebe_constraints, "heii1640_flux_observed", "C2")["value"])
    hebe_velocity_separation = float(_require_constraint(hebe_constraints, "velocity_separation", "C1_C2")["value"])

    heii_age_myr, heplus_q_per_msun = load_popiii_heplus_ionizing_photon_table(heii_ssp_file)
    cosmology = _build_cosmology(aurora_cosmology)
    luminosity_distance_cm = cosmology.luminosity_distance(float(args.z)).to_value(u.cm)
    conversion_factor = (
        float(args.heii_caseb_erg_per_photon)
        * float(args.heplus_covering_factor)
        * (1.0 - float(args.heplus_escape_fraction))
        * float(args.heii_photoionization_efficiency)
    )
    required_heplus_photon_rate = hebe_luminosity_intrinsic / conversion_factor
    popiii_params = PopIIISFRParameters(
        epsilon_star=float(args.popiii_epsilon_star),
        pivot_mass_msun=float(args.popiii_mp),
        alpha_star=float(args.popiii_alpha_star),
        beta_star=float(args.popiii_beta_star),
        lw_background_j21=float(args.lw_background_j21),
        upper_mass_mode=str(args.popiii_upper_mass_mode),
        upper_mass_msun=args.popiii_upper_mass_msun,
    )

    rows: list[dict[str, float]] = []
    logm_values = np.linspace(float(args.logM_min), float(args.logM_max), int(args.N_mass))
    start = time.perf_counter()
    for mass_index, logm in enumerate(logm_values):
        mh_final = float(10.0**logm)
        result = run_halo_uv_pipeline(
            n_tracks=int(args.n_tracks),
            z_final=float(args.z),
            Mh_final=mh_final,
            cosmology=aurora_cosmology,
            random_seeds=derive_pipeline_random_seeds(
                int(args.random_seed),
                redshift=float(args.z),
                mass_index=mass_index,
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
        t_grid = np.asarray(result.sfr_tracks["t_gyr"], dtype=float).reshape(int(args.n_tracks), steps_per_halo)
        sfr_popiii = np.asarray(result.sfr_tracks["SFR_popiii"], dtype=float).reshape(int(args.n_tracks), steps_per_halo)
        heplus_photon_rate = compute_final_ssp_heplus_rate_from_sfr_grid(
            t_grid_gyr=t_grid,
            sfr_grid=sfr_popiii,
            active_grid=np.asarray(result.popiii_source_grid, dtype=bool),
            ssp_age_myr=heii_age_myr,
            ssp_q_heplus_per_msun=heplus_q_per_msun,
            lookback_max_myr=float(args.heii_lookback_max_myr),
        )
        line_luminosity = heii1640_luminosity_from_heplus_rate(
            heplus_photon_rate,
            caseb_erg_per_photon=float(args.heii_caseb_erg_per_photon),
            covering_factor=float(args.heplus_covering_factor),
            escape_fraction=float(args.heplus_escape_fraction),
            photoionization_efficiency=float(args.heii_photoionization_efficiency),
        )
        recent_mass = _integrate_recent_mass_msun(
            t_grid_gyr=t_grid,
            sfr_grid=sfr_popiii,
            active_grid=np.asarray(result.popiii_source_grid, dtype=bool),
            window_myr=float(args.young_mass_window_myr),
        )
        flux = float(args.lens_magnification) * line_luminosity / (4.0 * np.pi * luminosity_distance_cm**2)
        uv_popiii = np.asarray(result.uv_luminosities_popiii, dtype=float)
        final_sfr = sfr_popiii[:, -1]
        for track_index in range(int(args.n_tracks)):
            rows.append(
                {
                    "logMh_final": float(logm),
                    "Mh_final_msun": mh_final,
                    "track_index": float(track_index),
                    "heplus_photon_rate_s": float(heplus_photon_rate[track_index]),
                    "line_luminosity_erg_s": float(line_luminosity[track_index]),
                    "flux_erg_s_cm2": float(flux[track_index]),
                    "flux_ratio_to_hebe": float(flux[track_index] / hebe_flux),
                    "luminosity_ratio_to_hebe_intrinsic": float(
                        line_luminosity[track_index] / hebe_luminosity_intrinsic
                    ),
                    "recent_popiii_mass_msun": float(recent_mass[track_index]),
                    "final_popiii_sfr_msun_yr": float(final_sfr[track_index]),
                    "popiii_uv_luminosity_erg_s_hz": float(uv_popiii[track_index]),
                }
            )
        elapsed = time.perf_counter() - start
        print(
            f"[{mass_index + 1}/{len(logm_values)}] logMh={logm:.2f} "
            f"median F/F_Hebe={np.median(flux / hebe_flux):.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )

    _write_rows(output_prefix.with_suffix(".csv"), rows)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        rows=np.asarray([[row[name] for name in rows[0].keys()] for row in rows], dtype=float),
        columns=np.asarray(list(rows[0].keys())),
        hebe_flux_erg_s_cm2=np.asarray([hebe_flux]),
        hebe_flux_err_erg_s_cm2=np.asarray([hebe_flux_err]),
        hebe_luminosity_intrinsic_clean_erg_s=np.asarray([hebe_luminosity_intrinsic]),
        hebe_required_heplus_photon_rate_s=np.asarray([required_heplus_photon_rate]),
        z=np.asarray([float(args.z)]),
        lens_magnification=np.asarray([float(args.lens_magnification)]),
        heii_caseb_erg_per_photon=np.asarray([float(args.heii_caseb_erg_per_photon)]),
        heplus_covering_factor=np.asarray([float(args.heplus_covering_factor)]),
        heplus_escape_fraction=np.asarray([float(args.heplus_escape_fraction)]),
        heii_photoionization_efficiency=np.asarray([float(args.heii_photoionization_efficiency)]),
        hebe_observation_file=np.asarray([str(hebe_observation_file)]),
        heii_ssp_file=np.asarray([str(heii_ssp_file)]),
        popiii_uv_ssp_file=np.asarray([str(popiii_uv_ssp_file)]),
        popiii_sfr_parameters=np.asarray([str(popiii_params.as_metadata())]),
    )
    _plot_results(
        rows=rows,
        output_prefix=output_prefix,
        z=float(args.z),
        lens_magnification=float(args.lens_magnification),
        hebe_flux_erg_s_cm2=hebe_flux,
        hebe_flux_err_erg_s_cm2=hebe_flux_err,
        hebe_component_c1_flux_erg_s_cm2=hebe_component_c1_flux,
        hebe_component_c2_flux_erg_s_cm2=hebe_component_c2_flux,
        hebe_velocity_separation_kms=hebe_velocity_separation,
    )
    flux_ratios = np.asarray([row["flux_ratio_to_hebe"] for row in rows], dtype=float)
    heplus_rates = np.asarray([row["heplus_photon_rate_s"] for row in rows], dtype=float)
    print(f"wrote {output_prefix.with_suffix('.csv')}")
    print(f"wrote {output_prefix.with_suffix('.npz')}")
    print(f"wrote {output_prefix.with_suffix('.pdf')}")
    print(f"wrote {output_prefix.with_suffix('.png')}")
    print(f"max F/F_Hebe={np.max(flux_ratios):.3e}; median F/F_Hebe={np.median(flux_ratios):.3e}")
    print(
        f"max Q(He+)={np.max(heplus_rates):.3e} s^-1; "
        f"required Q(He+)={required_heplus_photon_rate:.3e} s^-1"
    )


if __name__ == "__main__":
    main()
