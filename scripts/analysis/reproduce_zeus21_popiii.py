#!/usr/bin/env python3
"""Reproduce the Zeus21 fiducial Pop III global and halo-mass histories.

This script follows the public Pop II+III fiducial tutorial accompanying
Cruz et al. (2025, arXiv:2407.18294).  It deliberately imports Zeus21 from a
pinned source checkout rather than treating it as an undeclared AuroraLF
dependency.  The resulting CSV is an explicit bridge product: its redshift-
dependent LW background can be supplied to AuroraLF's molecular-cooling
threshold without changing either model's equations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from auroralf.cooling import compute_popiii_lw_minimum_mass_msun


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZEUS21_ROOT = PROJECT_ROOT / "third_party" / "zeus21"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data_save" / "zeus21_popiii_fiducial.csv"
DEFAULT_OUTPUT_FIGURE = PROJECT_ROOT / "outputs" / "zeus21_popiii_fiducial.png"
DEFAULT_OUTPUT_MASS_NPZ = PROJECT_ROOT / "data_save" / "zeus21_popiii_mass_distribution.npz"
DEFAULT_OUTPUT_MASS_FIGURE = PROJECT_ROOT / "outputs" / "zeus21_popiii_mass_distribution.png"
DEFAULT_OUTPUT_COMPOSITION_CSV = (
    PROJECT_ROOT / "data_save" / "zeus21_popii_popiii_mass_bin_fractions.csv"
)
DEFAULT_OUTPUT_COMPOSITION_FIGURE = (
    PROJECT_ROOT / "outputs" / "zeus21_popii_popiii_mass_fraction.png"
)
EXPECTED_ZEUS21_COMMIT = "9f2d2105e99e74096092e2061082a79c3f85eaca"
PAPER_ARXIV_ID = "2407.18294"
MASS_BIN_EDGES_MSUN = np.asarray(
    (1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e14),
    dtype=float,
)
POP_III_FRACTION_LEVELS = np.asarray((0.9, 0.5, 0.1), dtype=float)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeus21-root", type=Path, default=DEFAULT_ZEUS21_ROOT)
    parser.add_argument("--expected-commit", default=EXPECTED_ZEUS21_COMMIT)
    parser.add_argument("--z-min", type=float, default=10.0)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_OUTPUT_FIGURE)
    parser.add_argument("--output-mass-npz", type=Path, default=DEFAULT_OUTPUT_MASS_NPZ)
    parser.add_argument("--output-mass-figure", type=Path, default=DEFAULT_OUTPUT_MASS_FIGURE)
    parser.add_argument(
        "--output-composition-csv",
        type=Path,
        default=DEFAULT_OUTPUT_COMPOSITION_CSV,
    )
    parser.add_argument(
        "--output-composition-figure",
        type=Path,
        default=DEFAULT_OUTPUT_COMPOSITION_FIGURE,
    )
    return parser.parse_args()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_checkout(root: Path, expected_commit: str) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"Zeus21 source checkout is missing: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            f"Zeus21 checkout is at {commit}, expected pinned commit {expected_commit}"
        )
    dirty_paths = _git_output(root, "status", "--short", "--untracked-files=no")
    if dirty_paths:
        raise RuntimeError(f"Zeus21 tracked source has local changes:\n{dirty_paths}")
    return commit


def _build_zeus21_fiducial(
    zeus21: object,
    z_min: float,
) -> tuple[object, object, object, object]:
    if not np.isfinite(z_min) or z_min < 5.0 or z_min >= 50.0:
        raise ValueError("z_min must be finite and lie in [5, 50)")

    user = zeus21.User_Parameters(precisionboost=1.2)
    cosmo_input = zeus21.Cosmo_Parameters_Input(
        omegac=0.11933,
        omegab=0.02242,
        h_fid=0.6766,
        As=np.exp(3.047) * 1.0e-10,
        ns=0.9665,
        tau_fid=0.0544,
        USE_RELATIVE_VELOCITIES=True,
        Flag_emulate_21cmfast=False,
    )
    classy_cosmo = zeus21.runclass(cosmo_input)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_input, classy_cosmo)
    # Building the correlation structure also registers the zero-lag
    # correlation table used internally by get_T21_coefficients.
    zeus21.Correlations(user, cosmo, classy_cosmo)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy_cosmo)
    astro = zeus21.Astro_Parameters(
        user,
        cosmo,
        astromodel=0,
        accretion_model=0,
        alphastar=0.5,
        betastar=-0.5,
        epsstar=0.1,
        Mc=3.0e11,
        dlog10epsstardz=0.0,
        fesc10=0.1,
        alphaesc=0.0,
        L40_xray=10.0**0.5,
        E0_xray=500.0,
        alpha_xray=-1.0,
        Emax_xray_norm=2000.0,
        Nalpha_lyA_II=9690,
        Nalpha_lyA_III=17900,
        Mturn_fixed=None,
        FLAG_MTURN_SHARP=False,
        C0dust=4.43,
        C1dust=1.99,
        sigmaUV=0.5,
        USE_POPIII=True,
        USE_LW_FEEDBACK=True,
        alphastar_III=0.0,
        betastar_III=0.0,
        fstar_III=1.0e-3,
        Mc_III=1.0e7,
        dlog10epsstardz_III=0.0,
        fesc7_III=10.0**-1.35,
        alphaesc_III=-0.3,
        L40_xray_III=10.0**0.5,
        alpha_xray_III=-1.0,
        A_LW=2.0,
        beta_LW=0.6,
        A_vcb=1.0,
        beta_vcb=1.8,
    )
    coefficients = zeus21.get_T21_coefficients(
        user,
        cosmo,
        classy_cosmo,
        astro,
        hmf,
        zmin=z_min,
    )
    return coefficients, cosmo, astro, hmf


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    lengths = {np.asarray(values).size for values in columns.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"output columns have inconsistent lengths: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for values in zip(*(np.asarray(value, dtype=float) for value in columns.values()), strict=True):
            writer.writerow({name: f"{value:.12e}" for name, value in zip(columns, values, strict=True)})


def _integrate_mass_bins(
    *,
    log10_halo_mass: np.ndarray,
    dsfrd_dlog10m: np.ndarray,
    mass_bin_edges_msun: np.ndarray,
) -> np.ndarray:
    log_mass = np.asarray(log10_halo_mass, dtype=float)
    kernel = np.asarray(dsfrd_dlog10m, dtype=float)
    bin_edges = np.asarray(mass_bin_edges_msun, dtype=float)
    if log_mass.ndim != 1 or kernel.ndim != 2:
        raise ValueError("log10_halo_mass must be 1D and dsfrd_dlog10m must be 2D")
    if kernel.shape[1] != log_mass.size:
        raise ValueError("the mass axis of dsfrd_dlog10m must match log10_halo_mass")
    if not np.all(np.isfinite(log_mass)) or np.any(np.diff(log_mass) <= 0.0):
        raise ValueError("log10_halo_mass must be finite and strictly increasing")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("dsfrd_dlog10m must be finite and non-negative")
    if (
        bin_edges.ndim != 1
        or bin_edges.size < 2
        or not np.all(np.isfinite(bin_edges))
        or np.any(bin_edges <= 0.0)
        or np.any(np.diff(bin_edges) <= 0.0)
    ):
        raise ValueError("mass_bin_edges_msun must be finite, positive, and increasing")

    log_edges = np.log10(bin_edges)
    if log_edges[0] < log_mass[0] or log_edges[-1] > log_mass[-1]:
        raise ValueError("mass-bin edges must lie inside the halo-mass grid")

    integrated = np.empty((kernel.shape[0], bin_edges.size - 1), dtype=float)
    for bin_index, (lower, upper) in enumerate(
        zip(log_edges[:-1], log_edges[1:], strict=True)
    ):
        interior = (log_mass > lower) & (log_mass < upper)
        sample_mass = np.concatenate(([lower], log_mass[interior], [upper]))
        for redshift_index, row in enumerate(kernel):
            sample_kernel = np.concatenate(
                (
                    [np.interp(lower, log_mass, row)],
                    row[interior],
                    [np.interp(upper, log_mass, row)],
                )
            )
            integrated[redshift_index, bin_index] = np.trapezoid(
                sample_kernel,
                sample_mass,
            )
    if not np.all(np.isfinite(integrated)) or np.any(integrated < 0.0):
        raise RuntimeError("mass-bin integration produced invalid SFRD")
    return integrated


def _instantaneous_popiii_fraction(
    *,
    sfr_popii_per_halo: np.ndarray,
    sfr_popiii_per_halo: np.ndarray,
) -> np.ndarray:
    popii = np.asarray(sfr_popii_per_halo, dtype=float)
    popiii = np.asarray(sfr_popiii_per_halo, dtype=float)
    if popii.shape != popiii.shape or popii.ndim != 2:
        raise ValueError("Pop II and Pop III per-halo SFR arrays must be matching 2D arrays")
    if (
        not np.all(np.isfinite(popii))
        or not np.all(np.isfinite(popiii))
        or np.any(popii < 0.0)
        or np.any(popiii < 0.0)
    ):
        raise ValueError("per-halo SFR arrays must be finite and non-negative")
    total = popii + popiii
    if np.any(total <= 0.0):
        raise RuntimeError("Pop II plus Pop III per-halo SFR must be positive")
    return popiii / total


def _fraction_transition_masses(
    *,
    log10_halo_mass: np.ndarray,
    popiii_fraction: np.ndarray,
    fraction_levels: np.ndarray,
) -> np.ndarray:
    log_mass = np.asarray(log10_halo_mass, dtype=float)
    fraction = np.asarray(popiii_fraction, dtype=float)
    levels = np.asarray(fraction_levels, dtype=float)
    if log_mass.ndim != 1 or fraction.ndim != 2 or fraction.shape[1] != log_mass.size:
        raise ValueError("fraction mass axis must match the 1D halo-mass grid")
    if (
        levels.ndim != 1
        or not np.all(np.isfinite(levels))
        or np.any((levels <= 0.0) | (levels >= 1.0))
    ):
        raise ValueError("fraction levels must lie strictly between zero and one")
    if not np.all(np.isfinite(fraction)) or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("Pop III fraction must be finite and lie in [0, 1]")
    if np.any(np.diff(fraction, axis=1) > 1.0e-10):
        raise RuntimeError("Pop III SFR fraction is not monotonic with halo mass")

    transition_log_mass = np.empty((fraction.shape[0], levels.size), dtype=float)
    for redshift_index, row in enumerate(fraction):
        for level_index, level in enumerate(levels):
            crossed = np.flatnonzero(row <= level)
            if crossed.size == 0 or crossed[0] == 0:
                raise RuntimeError(
                    f"fraction level {level} is not bracketed at redshift row {redshift_index}"
                )
            upper_index = int(crossed[0])
            lower_index = upper_index - 1
            lower_fraction = row[lower_index]
            upper_fraction = row[upper_index]
            if lower_fraction == upper_fraction:
                raise RuntimeError("cannot interpolate a flat fraction transition")
            weight = (level - lower_fraction) / (upper_fraction - lower_fraction)
            transition_log_mass[redshift_index, level_index] = (
                log_mass[lower_index]
                + weight * (log_mass[upper_index] - log_mass[lower_index])
            )
    return 10.0**transition_log_mass


def _write_mass_bin_composition_csv(
    *,
    path: Path,
    redshift: np.ndarray,
    mass_bin_edges_msun: np.ndarray,
    sfrd_popii_by_bin: np.ndarray,
    sfrd_popiii_by_bin: np.ndarray,
) -> None:
    z = np.asarray(redshift, dtype=float)
    edges = np.asarray(mass_bin_edges_msun, dtype=float)
    popii = np.asarray(sfrd_popii_by_bin, dtype=float)
    popiii = np.asarray(sfrd_popiii_by_bin, dtype=float)
    expected_shape = (z.size, edges.size - 1)
    if popii.shape != expected_shape or popiii.shape != expected_shape:
        raise ValueError("mass-bin SFRD arrays do not match redshift and bin edges")
    total_by_bin = popii + popiii
    if np.any(total_by_bin <= 0.0):
        raise RuntimeError("every saved halo-mass bin must have positive total SFRD")
    total_by_redshift = np.sum(total_by_bin, axis=1)
    if np.any(total_by_redshift <= 0.0):
        raise RuntimeError("total mass-binned SFRD must be positive")

    fieldnames = (
        "redshift",
        "halo_mass_lower_msun",
        "halo_mass_upper_msun",
        "sfrd_popii_msun_yr_mpc3",
        "sfrd_popiii_msun_yr_mpc3",
        "popii_fraction_of_bin_sfr",
        "popiii_fraction_of_bin_sfr",
        "bin_fraction_of_total_sfrd",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for redshift_index, redshift_value in enumerate(z):
            for bin_index, (lower, upper) in enumerate(
                zip(edges[:-1], edges[1:], strict=True)
            ):
                bin_total = total_by_bin[redshift_index, bin_index]
                values = (
                    redshift_value,
                    lower,
                    upper,
                    popii[redshift_index, bin_index],
                    popiii[redshift_index, bin_index],
                    popii[redshift_index, bin_index] / bin_total,
                    popiii[redshift_index, bin_index] / bin_total,
                    bin_total / total_by_redshift[redshift_index],
                )
                writer.writerow(
                    {
                        name: f"{value:.12e}"
                        for name, value in zip(fieldnames, values, strict=True)
                    }
                )


def _plot_history(path: Path, columns: dict[str, np.ndarray]) -> None:
    z = columns["redshift"]
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True)

    axes[0].semilogy(z, columns["sfrd_popii_msun_yr_mpc3"], label="Pop II", color="#4C78A8")
    axes[0].semilogy(z, columns["sfrd_popiii_msun_yr_mpc3"], label="Pop III", color="#E45756")
    axes[0].set_ylabel(r"SFRD [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].semilogy(z, columns["j21_lw_popii"], label="Pop II", color="#4C78A8")
    axes[1].semilogy(z, columns["j21_lw_popiii"], label="Pop III", color="#E45756")
    axes[1].semilogy(z, columns["j21_lw_total"], label="total", color="#222222", linewidth=1.8)
    axes[1].set_ylabel(r"$J_{21}^{\rm LW}$")
    axes[1].legend(frameon=False, ncol=3)

    axes[2].semilogy(
        z,
        columns["mmol_no_feedback_msun"],
        label="no feedback",
        color="#72B7B2",
    )
    axes[2].semilogy(
        z,
        columns["mmol_lw_only_msun"],
        label="self-consistent LW",
        color="#F58518",
    )
    axes[2].semilogy(
        z,
        columns["mmol_lw_vcb_mean_msun"],
        label=r"LW + mean $v_{cb}$",
        color="#B279A2",
    )
    axes[2].set_xlabel("redshift z")
    axes[2].set_ylabel(r"$M_{\rm mol}$ [$M_\odot$]")
    axes[2].legend(frameon=False, ncol=3, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(direction="in", which="both", top=True, right=True)
    axes[-1].set_xlim(float(np.nanmax(z)), float(np.nanmin(z)))
    figure.suptitle("Zeus21 fiducial Pop III reproduction (Cruz et al. 2025)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(figure)


def _mass_contribution_quantiles(
    *,
    log_halo_mass: np.ndarray,
    dsfrd_dlnm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_mass = np.asarray(log_halo_mass, dtype=float)
    contribution = np.asarray(dsfrd_dlnm, dtype=float)
    if log_mass.ndim != 1 or contribution.ndim != 2:
        raise ValueError("log_halo_mass must be 1D and dsfrd_dlnm must be 2D")
    if contribution.shape[1] != log_mass.size:
        raise ValueError("the mass axis of dsfrd_dlnm must match log_halo_mass")
    if not np.all(np.isfinite(log_mass)) or np.any(np.diff(log_mass) <= 0.0):
        raise ValueError("log_halo_mass must be finite and strictly increasing")
    if not np.all(np.isfinite(contribution)) or np.any(contribution < 0.0):
        raise ValueError("dsfrd_dlnm must be finite and non-negative")

    interval = 0.5 * (contribution[:, 1:] + contribution[:, :-1]) * np.diff(log_mass)
    cumulative = np.concatenate(
        (np.zeros((contribution.shape[0], 1), dtype=float), np.cumsum(interval, axis=1)),
        axis=1,
    )
    total = cumulative[:, -1]
    if np.any(total <= 0.0):
        raise RuntimeError("mass-resolved Pop III SFRD has a non-positive integral")
    cumulative /= total[:, None]

    quantile_log_mass = np.empty((contribution.shape[0], 3), dtype=float)
    for row_index, row in enumerate(cumulative):
        quantile_log_mass[row_index] = np.interp((0.16, 0.50, 0.84), row, log_mass)
    peak_mass = np.exp(log_mass[np.argmax(contribution, axis=1)])
    return (
        peak_mass,
        np.exp(quantile_log_mass[:, 0]),
        np.exp(quantile_log_mass[:, 1]),
        np.exp(quantile_log_mass[:, 2]),
    )


def _plot_mass_distribution(
    *,
    path: Path,
    redshift: np.ndarray,
    halo_mass_msun: np.ndarray,
    dsfrd_dlog10m_popiii: np.ndarray,
    peak_mass_msun: np.ndarray,
    p16_mass_msun: np.ndarray,
    median_mass_msun: np.ndarray,
    p84_mass_msun: np.ndarray,
    molecular_threshold_msun: np.ndarray,
    atomic_threshold_msun: np.ndarray,
) -> None:
    contribution = np.asarray(dsfrd_dlog10m_popiii, dtype=float)
    positive = contribution > 0.0
    if not np.any(positive):
        raise RuntimeError("cannot plot an identically zero Pop III mass distribution")
    maximum = float(np.max(contribution[positive]))
    meaningful = np.any(contribution >= maximum * 1.0e-8, axis=0)
    if not np.any(meaningful):
        raise RuntimeError("failed to identify the meaningful halo-mass plotting range")
    mass_indices = np.flatnonzero(meaningful)
    mass_lower = halo_mass_msun[max(int(mass_indices[0]) - 1, 0)]
    mass_upper = halo_mass_msun[min(int(mass_indices[-1]) + 1, halo_mass_msun.size - 1)]

    figure, (map_axis, slice_axis) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    image = map_axis.pcolormesh(
        redshift,
        halo_mass_msun,
        np.ma.masked_less_equal(contribution.T, 0.0),
        norm=LogNorm(vmin=maximum * 1.0e-8, vmax=maximum),
        shading="auto",
        cmap="magma",
    )
    map_axis.fill_between(
        redshift,
        p16_mass_msun,
        p84_mass_msun,
        color="white",
        alpha=0.18,
        linewidth=0.0,
        label="16--84% SFRD contribution",
    )
    map_axis.plot(redshift, median_mass_msun, color="white", linewidth=1.8, label="median")
    map_axis.plot(redshift, peak_mass_msun, color="#5DD9C1", linewidth=1.4, label="mode")
    map_axis.plot(
        redshift,
        molecular_threshold_msun,
        color="#4CC9F0",
        linestyle="--",
        linewidth=1.2,
        label=r"$M_{\rm mol}$ (LW+$v_{cb}$)",
    )
    map_axis.plot(
        redshift,
        atomic_threshold_msun,
        color="#F9C74F",
        linestyle=":",
        linewidth=1.5,
        label=r"$M_{\rm atom}$",
    )
    map_axis.set(
        xlabel="redshift z",
        ylabel=r"halo mass $M_h$ [$M_\odot$]",
        xlim=(float(np.max(redshift)), float(np.min(redshift))),
        ylim=(mass_lower, mass_upper),
        yscale="log",
    )
    map_axis.legend(loc="upper right", fontsize=8, framealpha=0.82)
    colorbar = figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        pad=0.16,
        fraction=0.08,
        aspect=28,
    )
    colorbar.set_label(
        r"$\mathrm{d}\dot\rho_{\star,\rm III}/\mathrm{d}\log_{10}M_h$ "
        r"[$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]"
    )

    colors = plt.cm.viridis(np.linspace(0.05, 0.90, 5))
    for target_redshift, color in zip((10.0, 15.0, 20.0, 25.0, 30.0), colors, strict=True):
        index = int(np.argmin(np.abs(redshift - target_redshift)))
        slice_axis.loglog(
            halo_mass_msun,
            contribution[index],
            color=color,
            label=rf"$z={redshift[index]:.1f}$",
        )
    slice_axis.set(
        xlabel=r"halo mass $M_h$ [$M_\odot$]",
        ylabel=(
            r"$\mathrm{d}\dot\rho_{\star,\rm III}/\mathrm{d}\log_{10}M_h$ "
            r"[$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]"
        ),
        xlim=(mass_lower, mass_upper),
    )
    positive_slices = contribution[
        [int(np.argmin(np.abs(redshift - value))) for value in (10.0, 15.0, 20.0, 25.0, 30.0)]
    ]
    slice_floor = float(np.min(positive_slices[positive_slices > maximum * 1.0e-8]))
    slice_axis.set_ylim(slice_floor * 0.7, float(np.max(positive_slices)) * 1.6)
    slice_axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Zeus21 fiducial: Pop III SFRD contribution by halo mass")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(figure)


def _plot_population_composition(
    *,
    path: Path,
    redshift: np.ndarray,
    halo_mass_msun: np.ndarray,
    popiii_sfr_fraction: np.ndarray,
    total_dsfrd_dlog10m: np.ndarray,
    molecular_threshold_msun: np.ndarray,
    atomic_threshold_msun: np.ndarray,
) -> None:
    z = np.asarray(redshift, dtype=float)
    halo_mass = np.asarray(halo_mass_msun, dtype=float)
    fraction = np.asarray(popiii_sfr_fraction, dtype=float)
    total_kernel = np.asarray(total_dsfrd_dlog10m, dtype=float)
    expected_shape = (z.size, halo_mass.size)
    if fraction.shape != expected_shape or total_kernel.shape != expected_shape:
        raise ValueError("composition arrays must match the redshift and halo-mass grids")
    if (
        not np.all(np.isfinite(fraction))
        or np.any((fraction < 0.0) | (fraction > 1.0))
        or not np.all(np.isfinite(total_kernel))
        or np.any(total_kernel < 0.0)
    ):
        raise ValueError("composition plot received invalid values")
    row_maximum = np.max(total_kernel, axis=1)
    if np.any(row_maximum <= 0.0):
        raise RuntimeError("composition plot requires positive SFRD at every redshift")
    active = total_kernel >= row_maximum[:, None] * 1.0e-4

    figure, (map_axis, slice_axis) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    map_axis.set_facecolor("#D9D9D9")
    image = map_axis.pcolormesh(
        z,
        halo_mass,
        np.ma.masked_where(~active, fraction).T,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        cmap="RdBu_r",
    )
    contour_levels = np.sort(POP_III_FRACTION_LEVELS)
    contours = map_axis.contour(
        z,
        halo_mass,
        fraction.T,
        levels=contour_levels,
        colors="black",
        linewidths=(0.8, 1.5, 0.8),
    )
    map_axis.clabel(
        contours,
        fmt={0.1: r"$10\%$", 0.5: r"$50\%$", 0.9: r"$90\%$"},
        fontsize=8,
        inline=True,
    )
    map_axis.plot(
        z,
        molecular_threshold_msun,
        color="#17BECF",
        linestyle="--",
        linewidth=1.4,
        label=r"$M_{\rm mol}$ (LW+$v_{cb}$)",
    )
    map_axis.plot(
        z,
        atomic_threshold_msun,
        color="#FFBF00",
        linestyle=":",
        linewidth=1.7,
        label=r"$M_{\rm atom}$",
    )
    map_axis.set(
        xlabel="redshift z",
        ylabel=r"halo mass $M_h$ [$M_\odot$]",
        xlim=(float(np.max(z)), float(np.min(z))),
        ylim=(float(halo_mass[0]), 1.0e9),
        yscale="log",
    )
    map_axis.text(
        0.02,
        0.025,
        r"gray: total SFRD $<10^{-4}$ of the peak at fixed $z$",
        transform=map_axis.transAxes,
        fontsize=7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )
    map_axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    colorbar = figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        pad=0.16,
        fraction=0.08,
        aspect=28,
    )
    colorbar.set_label(r"instantaneous Pop III SFR fraction $f_{\rm III}$")

    colors = plt.cm.viridis(np.linspace(0.03, 0.95, 6))
    for target_redshift, color in zip(
        (10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        colors,
        strict=True,
    ):
        index = int(np.argmin(np.abs(z - target_redshift)))
        slice_axis.semilogx(
            halo_mass,
            fraction[index],
            color=color,
            label=rf"$z={z[index]:.1f}$",
        )
    slice_axis.axhline(0.5, color="#444444", linestyle="--", linewidth=0.9)
    slice_axis.set(
        xlabel=r"halo mass $M_h$ [$M_\odot$]",
        ylabel=r"instantaneous Pop III SFR fraction $f_{\rm III}$",
        xlim=(float(halo_mass[0]), 1.0e9),
        ylim=(-0.02, 1.02),
    )
    slice_axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.suptitle("Zeus21 fiducial: Pop II / Pop III instantaneous SFR composition")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(figure)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = _parse_args()
    zeus21_root = args.zeus21_root.resolve()
    commit = _validate_source_checkout(zeus21_root, args.expected_commit)
    sys.path.insert(0, str(zeus21_root))

    import zeus21  # type: ignore[import-not-found]  # noqa: PLC0415
    from zeus21.sfrd import (  # type: ignore[import-not-found]  # noqa: PLC0415
        Matom,
        Mmol,
        Mmol_LW,
        SFRD_II_integrand,
        SFRD_III_integrand,
        SFR_II,
        SFR_III,
    )

    started = time.perf_counter()
    coefficients, cosmo, astro, hmf = _build_zeus21_fiducial(zeus21, args.z_min)
    runtime_seconds = time.perf_counter() - started

    redshift = np.asarray(coefficients.zintegral, dtype=float)
    j21_popii = np.asarray(coefficients.J_21_LW_II, dtype=float)
    j21_popiii = np.asarray(coefficients.J_21_LW_III, dtype=float)
    j21_total = j21_popii + j21_popiii
    j21_interpolator = lambda z: np.interp(z, redshift, j21_total)  # noqa: E731

    mmol_no_feedback = np.asarray(
        compute_popiii_lw_minimum_mass_msun(redshift, lw_background_j21=0.0),
        dtype=float,
    )
    mmol_lw_auroralf = np.asarray(
        compute_popiii_lw_minimum_mass_msun(redshift, lw_background_j21=j21_total),
        dtype=float,
    )
    mmol_lw_zeus21 = np.asarray(Mmol_LW(astro, j21_interpolator, redshift), dtype=float)
    np.testing.assert_allclose(mmol_lw_auroralf, mmol_lw_zeus21, rtol=2.0e-12, atol=0.0)
    mmol_full = np.asarray(
        Mmol(
            astro,
            cosmo,
            coefficients.J21LW_interp_conv_avg,
            redshift,
            cosmo.vcb_avg,
        ),
        dtype=float,
    )
    matom = np.asarray(Matom(redshift), dtype=float)

    redshift_grid, halo_mass_grid = np.meshgrid(
        redshift,
        np.asarray(hmf.Mhtab, dtype=float),
        indexing="ij",
    )
    dsfrd_dlnm_popii = np.asarray(
        SFRD_II_integrand(
            astro,
            cosmo,
            hmf,
            halo_mass_grid,
            redshift_grid,
            redshift_grid,
        ),
        dtype=float,
    )
    dsfrd_dlnm_popiii = np.asarray(
        SFRD_III_integrand(
            astro,
            cosmo,
            hmf,
            halo_mass_grid,
            coefficients.J21LW_interp_conv_avg,
            redshift_grid,
            redshift_grid,
            cosmo.vcb_avg,
        ),
        dtype=float,
    )
    hmf_dndm = np.exp(
        hmf.logHMFint((np.log(halo_mass_grid), redshift_grid))
    )
    sfr_popii_per_halo = np.asarray(
        SFR_II(
            astro,
            cosmo,
            hmf,
            halo_mass_grid,
            redshift_grid,
            redshift_grid,
        ),
        dtype=float,
    )
    sfr_popiii_per_halo = np.asarray(
        SFR_III(
            astro,
            cosmo,
            hmf,
            halo_mass_grid,
            coefficients.J21LW_interp_conv_avg,
            redshift_grid,
            redshift_grid,
            cosmo.vcb_avg,
        ),
        dtype=float,
    )
    for name, values in (
        ("dsfrd_dlnm_popii", dsfrd_dlnm_popii),
        ("dsfrd_dlnm_popiii", dsfrd_dlnm_popiii),
        ("hmf_dndm", hmf_dndm),
        ("sfr_popii_per_halo", sfr_popii_per_halo),
        ("sfr_popiii_per_halo", sfr_popiii_per_halo),
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(f"Zeus21 mass-resolved {name} contains invalid values")

    composition_halo_mass = np.geomspace(1.0e5, 1.0e10, 501)
    composition_redshift_grid, composition_halo_mass_grid = np.meshgrid(
        redshift,
        composition_halo_mass,
        indexing="ij",
    )
    composition_sfr_popii = np.asarray(
        SFR_II(
            astro,
            cosmo,
            hmf,
            composition_halo_mass_grid,
            composition_redshift_grid,
            composition_redshift_grid,
        ),
        dtype=float,
    )
    composition_sfr_popiii = np.asarray(
        SFR_III(
            astro,
            cosmo,
            hmf,
            composition_halo_mass_grid,
            coefficients.J21LW_interp_conv_avg,
            composition_redshift_grid,
            composition_redshift_grid,
            cosmo.vcb_avg,
        ),
        dtype=float,
    )
    popiii_instantaneous_fraction = _instantaneous_popiii_fraction(
        sfr_popii_per_halo=composition_sfr_popii,
        sfr_popiii_per_halo=composition_sfr_popiii,
    )
    fraction_transition_mass = _fraction_transition_masses(
        log10_halo_mass=np.log10(composition_halo_mass),
        popiii_fraction=popiii_instantaneous_fraction,
        fraction_levels=POP_III_FRACTION_LEVELS,
    )
    composition_hmf_dndm = np.exp(
        hmf.logHMFint(
            (np.log(composition_halo_mass_grid), composition_redshift_grid)
        )
    )
    composition_total_dsfrd_dlog10m = (
        composition_hmf_dndm
        * (composition_sfr_popii + composition_sfr_popiii)
        * composition_halo_mass_grid
        * np.log(10.0)
    )
    if (
        not np.all(np.isfinite(composition_total_dsfrd_dlog10m))
        or np.any(composition_total_dsfrd_dlog10m < 0.0)
    ):
        raise RuntimeError("dense Pop II plus Pop III SFRD kernel contains invalid values")

    integrated_popii = np.trapezoid(dsfrd_dlnm_popii, hmf.logtabMh, axis=1)
    integrated_popiii = np.trapezoid(dsfrd_dlnm_popiii, hmf.logtabMh, axis=1)
    popii_closure_error = np.max(
        np.abs(integrated_popii - coefficients.SFRD_II_avg) / coefficients.SFRD_II_avg
    )
    popiii_closure_error = np.max(
        np.abs(integrated_popiii - coefficients.SFRD_III_avg) / coefficients.SFRD_III_avg
    )
    if popii_closure_error > 5.0e-3 or popiii_closure_error > 5.0e-3:
        raise RuntimeError(
            "mass-resolved Zeus21 SFRD does not close to the global history: "
            f"Pop II error={popii_closure_error:.3e}, "
            f"Pop III error={popiii_closure_error:.3e}"
        )

    dsfrd_dlog10m_popii = dsfrd_dlnm_popii * np.log(10.0)
    dsfrd_dlog10m_popiii = dsfrd_dlnm_popiii * np.log(10.0)
    sfrd_popii_by_mass_bin = _integrate_mass_bins(
        log10_halo_mass=np.log10(hmf.Mhtab),
        dsfrd_dlog10m=dsfrd_dlog10m_popii,
        mass_bin_edges_msun=MASS_BIN_EDGES_MSUN,
    )
    sfrd_popiii_by_mass_bin = _integrate_mass_bins(
        log10_halo_mass=np.log10(hmf.Mhtab),
        dsfrd_dlog10m=dsfrd_dlog10m_popiii,
        mass_bin_edges_msun=MASS_BIN_EDGES_MSUN,
    )
    np.testing.assert_allclose(
        np.sum(sfrd_popii_by_mass_bin, axis=1),
        integrated_popii,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sum(sfrd_popiii_by_mass_bin, axis=1),
        integrated_popiii,
        rtol=2.0e-12,
        atol=0.0,
    )
    sfrd_total_by_mass_bin = sfrd_popii_by_mass_bin + sfrd_popiii_by_mass_bin
    if np.any(sfrd_total_by_mass_bin <= 0.0):
        raise RuntimeError("every halo-mass bin must have positive total SFRD")
    popiii_fraction_by_mass_bin = (
        sfrd_popiii_by_mass_bin / sfrd_total_by_mass_bin
    )
    total_sfrd_fraction_by_mass_bin = (
        sfrd_total_by_mass_bin
        / np.sum(sfrd_total_by_mass_bin, axis=1)[:, None]
    )

    peak_mass, p16_mass, median_mass, p84_mass = _mass_contribution_quantiles(
        log_halo_mass=np.asarray(hmf.logtabMh, dtype=float),
        dsfrd_dlnm=dsfrd_dlnm_popiii,
    )
    global_sfrd_total = np.asarray(
        coefficients.SFRD_II_avg + coefficients.SFRD_III_avg,
        dtype=float,
    )
    if np.any(global_sfrd_total <= 0.0):
        raise RuntimeError("global Pop II plus Pop III SFRD must be positive")

    columns = {
        "redshift": redshift,
        "sfrd_popii_msun_yr_mpc3": np.asarray(coefficients.SFRD_II_avg, dtype=float),
        "sfrd_popiii_msun_yr_mpc3": np.asarray(coefficients.SFRD_III_avg, dtype=float),
        "popii_fraction_of_total_sfrd": (
            np.asarray(coefficients.SFRD_II_avg, dtype=float) / global_sfrd_total
        ),
        "popiii_fraction_of_total_sfrd": (
            np.asarray(coefficients.SFRD_III_avg, dtype=float) / global_sfrd_total
        ),
        "j21_lw_popii": j21_popii,
        "j21_lw_popiii": j21_popiii,
        "j21_lw_total": j21_total,
        "mmol_no_feedback_msun": mmol_no_feedback,
        "mmol_lw_only_msun": mmol_lw_auroralf,
        "mmol_lw_vcb_mean_msun": mmol_full,
        "matom_msun": matom,
        "popiii_sfrd_peak_halo_mass_msun": peak_mass,
        "popiii_sfrd_p16_halo_mass_msun": p16_mass,
        "popiii_sfrd_median_halo_mass_msun": median_mass,
        "popiii_sfrd_p84_halo_mass_msun": p84_mass,
        "popiii_sfr_fraction_90pct_halo_mass_msun": fraction_transition_mass[:, 0],
        "popiii_sfr_fraction_50pct_halo_mass_msun": fraction_transition_mass[:, 1],
        "popiii_sfr_fraction_10pct_halo_mass_msun": fraction_transition_mass[:, 2],
    }
    if not all(np.all(np.isfinite(values)) for values in columns.values()):
        raise RuntimeError("Zeus21 reproduction produced non-finite output")
    if np.any(columns["sfrd_popiii_msun_yr_mpc3"] < 0.0):
        raise RuntimeError("Zeus21 reproduction produced negative Pop III SFRD")

    output_mass_npz = args.output_mass_npz.resolve()
    output_mass_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_mass_npz,
        schema_version=np.asarray("auroralf.zeus21_popiii_mass_distribution.v2"),
        redshift=redshift,
        halo_mass_msun=np.asarray(hmf.Mhtab, dtype=float),
        hmf_dndm_mpc3_msun=hmf_dndm,
        popii_sfr_per_halo_msun_yr=sfr_popii_per_halo,
        popiii_sfr_per_halo_msun_yr=sfr_popiii_per_halo,
        dsfrd_dlog10m_popii_msun_yr_mpc3=dsfrd_dlog10m_popii,
        dsfrd_dlog10m_popiii_msun_yr_mpc3=dsfrd_dlog10m_popiii,
        composition_halo_mass_msun=composition_halo_mass,
        composition_sfr_popii_per_halo_msun_yr=composition_sfr_popii,
        composition_sfr_popiii_per_halo_msun_yr=composition_sfr_popiii,
        popiii_instantaneous_sfr_fraction=popiii_instantaneous_fraction,
        popiii_fraction_levels=POP_III_FRACTION_LEVELS,
        popiii_fraction_transition_halo_mass_msun=fraction_transition_mass,
        composition_total_dsfrd_dlog10m_msun_yr_mpc3=(
            composition_total_dsfrd_dlog10m
        ),
        mass_bin_edges_msun=MASS_BIN_EDGES_MSUN,
        sfrd_popii_by_mass_bin_msun_yr_mpc3=sfrd_popii_by_mass_bin,
        sfrd_popiii_by_mass_bin_msun_yr_mpc3=sfrd_popiii_by_mass_bin,
        popiii_sfr_fraction_by_mass_bin=popiii_fraction_by_mass_bin,
        total_sfrd_fraction_by_mass_bin=total_sfrd_fraction_by_mass_bin,
    )

    plt.style.use("apj")
    _write_csv(args.output_csv.resolve(), columns)
    _write_mass_bin_composition_csv(
        path=args.output_composition_csv.resolve(),
        redshift=redshift,
        mass_bin_edges_msun=MASS_BIN_EDGES_MSUN,
        sfrd_popii_by_bin=sfrd_popii_by_mass_bin,
        sfrd_popiii_by_bin=sfrd_popiii_by_mass_bin,
    )
    _plot_history(args.output_figure.resolve(), columns)
    _plot_mass_distribution(
        path=args.output_mass_figure.resolve(),
        redshift=redshift,
        halo_mass_msun=np.asarray(hmf.Mhtab, dtype=float),
        dsfrd_dlog10m_popiii=dsfrd_dlnm_popiii * np.log(10.0),
        peak_mass_msun=peak_mass,
        p16_mass_msun=p16_mass,
        median_mass_msun=median_mass,
        p84_mass_msun=p84_mass,
        molecular_threshold_msun=mmol_full,
        atomic_threshold_msun=matom,
    )
    _plot_population_composition(
        path=args.output_composition_figure.resolve(),
        redshift=redshift,
        halo_mass_msun=composition_halo_mass,
        popiii_sfr_fraction=popiii_instantaneous_fraction,
        total_dsfrd_dlog10m=composition_total_dsfrd_dlog10m,
        molecular_threshold_msun=mmol_full,
        atomic_threshold_msun=matom,
    )
    metadata_path = args.output_csv.resolve().with_suffix(".metadata.json")
    metadata = {
        "schema_version": "auroralf.zeus21_popiii_fiducial.v1",
        "paper_arxiv_id": PAPER_ARXIV_ID,
        "source_url": "https://github.com/ZeusCosmo/Zeus21",
        "source_commit": commit,
        "runtime_seconds": runtime_seconds,
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("zeus21", "classy", "numpy", "scipy", "mcfit", "powerbox", "pyfftw")
        },
        "parameters": {
            "z_min": float(args.z_min),
            "precisionboost": 1.2,
            "fstar_popiii": 1.0e-3,
            "A_LW": 2.0,
            "beta_LW": 0.6,
            "A_vcb": 1.0,
            "beta_vcb": 1.8,
            "relative_velocities": True,
        },
        "outputs": {
            "csv": _portable_path(args.output_csv),
            "figure": _portable_path(args.output_figure),
            "mass_distribution_npz": _portable_path(args.output_mass_npz),
            "mass_distribution_figure": _portable_path(args.output_mass_figure),
            "mass_bin_composition_csv": _portable_path(args.output_composition_csv),
            "population_composition_figure": _portable_path(
                args.output_composition_figure
            ),
        },
        "mass_distribution": {
            "quantity": "dSFRD/dlog10(Mh)",
            "units": "Msun yr^-1 Mpc^-3",
            "halo_mass_units": "Msun",
            "mass_grid_size": int(np.asarray(hmf.Mhtab).size),
            "mass_grid_min_msun": float(np.min(hmf.Mhtab)),
            "mass_grid_max_msun": float(np.max(hmf.Mhtab)),
            "popii_global_closure_max_relative_error": float(popii_closure_error),
            "popiii_global_closure_max_relative_error": float(popiii_closure_error),
        },
        "population_composition": {
            "point_quantity": "SFR_III / (SFR_II + SFR_III) at fixed Mh and z",
            "bin_quantity": "integral SFRD_III / integral (SFRD_II + SFRD_III)",
            "fraction_units": "dimensionless",
            "composition_mass_grid_size": int(composition_halo_mass.size),
            "composition_mass_grid_min_msun": float(composition_halo_mass[0]),
            "composition_mass_grid_max_msun": float(composition_halo_mass[-1]),
            "fraction_transition_levels": POP_III_FRACTION_LEVELS.tolist(),
            "mass_bin_edges_msun": MASS_BIN_EDGES_MSUN.tolist(),
            "plot_active_sfrd_threshold_relative_to_redshift_peak": 1.0e-4,
            "interpretation": "conditional ensemble-mean instantaneous SFR fraction",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
