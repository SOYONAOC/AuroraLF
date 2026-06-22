#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.ssp import load_popiii_uv_luminosity_table, load_uv1600_table
from auroralf.uvlf.hmf_sampling import uv_luminosity_to_muv
from auroralf.uvlf.imf import DEFAULT_CANONICAL_SSP_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_PATH = (
    PROJECT_ROOT
    / "slides"
    / "group_meeting_popiii_20260622"
    / "assets"
    / "popiii_popii_muv_age_1_10myr_slide.pdf"
)
DEFAULT_TABLE_PATH = PROJECT_ROOT / "outputs" / "popiii_popii_muv_age_1_10myr_slide.csv"
DEFAULT_EXTREME_POPIII_SSP_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "schaerer2010_pop3"
    / "pop3_ge0_sal_500_050_is4.25"
)
DEFAULT_POPIII_LABEL = r"Pop III extreme, $50$--$500\,M_\odot$"
DEFAULT_POPII_LABEL = r"Pop II canonical"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pop II and Pop III SSP MUV evolution for group-meeting slides.")
    parser.add_argument("--stellar-mass-msun", type=float, default=1.0e9)
    parser.add_argument("--age-min-myr", type=float, default=1.0)
    parser.add_argument("--age-max-myr", type=float, default=10.0)
    parser.add_argument("--n-age", type=int, default=240)
    parser.add_argument("--popii-ssp-file", type=Path, default=Path(DEFAULT_CANONICAL_SSP_FILE))
    parser.add_argument("--popiii-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_SSP_FILE)
    parser.add_argument("--popii-label", type=str, default=DEFAULT_POPII_LABEL)
    parser.add_argument("--popiii-label", type=str, default=DEFAULT_POPIII_LABEL)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def _interpolate_luminosity(age_grid_myr: np.ndarray, ages_myr: np.ndarray, luminosity_per_msun: np.ndarray) -> np.ndarray:
    if ages_myr.ndim != 1 or luminosity_per_msun.ndim != 1:
        raise ValueError("SSP age and luminosity arrays must be 1D")
    if ages_myr.size != luminosity_per_msun.size:
        raise ValueError("SSP age and luminosity arrays must have the same length")
    if np.any(ages_myr <= 0.0) or np.any(np.diff(ages_myr) <= 0.0):
        raise ValueError("SSP ages must be positive and strictly increasing")
    if np.any(luminosity_per_msun <= 0.0):
        raise ValueError("SSP luminosities must be positive for MUV interpolation")

    luminosity = np.full_like(age_grid_myr, np.nan, dtype=float)
    in_range = (age_grid_myr >= ages_myr[0]) & (age_grid_myr <= ages_myr[-1])
    if np.any(in_range):
        log_luminosity = np.interp(
            np.log10(age_grid_myr[in_range]),
            np.log10(ages_myr),
            np.log10(luminosity_per_msun),
        )
        luminosity[in_range] = np.power(10.0, log_luminosity)
    return luminosity


def main() -> None:
    args = _parse_args()
    if args.stellar_mass_msun <= 0.0:
        raise ValueError("--stellar-mass-msun must be positive")
    if args.age_min_myr <= 0.0:
        raise ValueError("--age-min-myr must be positive")
    if args.age_max_myr <= args.age_min_myr:
        raise ValueError("--age-max-myr must be larger than --age-min-myr")
    if args.n_age < 8:
        raise ValueError("--n-age must be at least 8")

    popii_ssp_file = _resolve_path(args.popii_ssp_file)
    popiii_ssp_file = _resolve_path(args.popiii_ssp_file)
    if not popii_ssp_file.is_file():
        raise FileNotFoundError(f"Pop II SSP file not found: {popii_ssp_file}")
    if not popiii_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III SSP file not found: {popiii_ssp_file}")

    age_grid = np.linspace(float(args.age_min_myr), float(args.age_max_myr), int(args.n_age))
    popii_age, popii_lnu_per_msun = load_uv1600_table(popii_ssp_file)
    popiii_age, popiii_lnu_per_msun = load_popiii_uv_luminosity_table(popiii_ssp_file)

    popii_lnu = _interpolate_luminosity(age_grid, popii_age, popii_lnu_per_msun) * float(args.stellar_mass_msun)
    popiii_lnu = _interpolate_luminosity(age_grid, popiii_age, popiii_lnu_per_msun) * float(args.stellar_mass_msun)
    popii_muv = np.asarray(uv_luminosity_to_muv(popii_lnu), dtype=float)
    popiii_muv = np.asarray(uv_luminosity_to_muv(popiii_lnu), dtype=float)

    table_path = _resolve_path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        table_path,
        np.column_stack([age_grid, popii_muv, popiii_muv, popii_lnu, popiii_lnu]),
        delimiter=",",
        header=(
            "age_myr,Muv_popii,Muv_popiii,Lnu_popii_erg_s^-1_Hz^-1,Lnu_popiii_erg_s^-1_Hz^-1\n"
            f"stellar_mass_msun={float(args.stellar_mass_msun):.8e},"
            f"popii_ssp_file={popii_ssp_file},popiii_ssp_file={popiii_ssp_file}"
        ),
        comments="# ",
    )

    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(age_grid, popii_muv, color="#1F5C8B", linewidth=2.7, label=str(args.popii_label))
    ax.plot(age_grid, popiii_muv, color="#2A9D8F", linewidth=2.9, label=str(args.popiii_label))
    ax.set_xlim(float(args.age_min_myr), float(args.age_max_myr))
    y_all = np.concatenate([popii_muv, popiii_muv])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        raise RuntimeError("No finite MUV values available for plotting")
    y_margin = 0.25 * (float(np.max(y_all)) - float(np.min(y_all)))
    ax.set_ylim(float(np.max(y_all)) + y_margin, float(np.min(y_all)) - y_margin)
    ax.set_xlabel("SSP age [Myr]")
    ax.set_ylabel(r"$M_{\rm UV}$ for $10^9\,M_\odot$")
    ax.grid(True, which="major", color="#C8D2DF", linewidth=0.75, alpha=0.85)
    ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.45, alpha=0.70)
    ax.legend(loc="lower right", frameon=True, fontsize=13.5)
    ax.text(
        0.035,
        0.93,
        r"$Z_{\rm PopIII}=0$ SSP, instantaneous burst",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="#1F3A5F",
    )
    if float(popiii_age[-1]) < float(args.age_max_myr):
        ax.text(
            0.965,
            0.93,
            rf"Pop III table ends at {float(popiii_age[-1]):.1f} Myr",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12.5,
            color="#4A5568",
        )
    fig.tight_layout()

    figure_path = _resolve_path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=500)
    plt.close(fig)

    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
