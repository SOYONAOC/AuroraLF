#!/usr/bin/env python3
"""Compare the production Reed07 HMF with Sheth--Tormen at z=6."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hmf import MassFunction

from auroralf.mah import Cosmology
from auroralf.uvlf import compute_reed07_halo_mass_function_dndm
from auroralf.uvlf.hmf_sampling import MASS_FUNCTION_NS, MASS_FUNCTION_SIGMA8


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "hmf_reed07_st_z6"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data_save" / "hmf_reed07_st_z6.csv"
BENCHMARK_MASSES_MSUN = np.power(10.0, np.arange(8.0, 13.1, 1.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=6.0)
    parser.add_argument("--logM-min", type=float, default=8.0)
    parser.add_argument("--logM-max", type=float, default=13.0)
    parser.add_argument("--n-mass", type=int, default=501)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _compute_st_dndm(
    halo_mass_msun: np.ndarray,
    z_obs: float,
    *,
    cosmology: Cosmology,
) -> np.ndarray:
    """Evaluate hmf's ST/SMT fit in physical Mpc^-3 Msun^-1 units."""
    mass = np.asarray(halo_mass_msun, dtype=float)
    if mass.ndim != 1 or mass.size < 2:
        raise ValueError("halo_mass_msun must be a one-dimensional array with at least two entries")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("halo masses must be finite and positive")

    h = cosmology.h0_km_s_mpc / 100.0
    dlog10m = 0.01
    log_h = np.log10(h)
    grid_min = np.floor((np.log10(mass.min()) + log_h - 2.0 * dlog10m) / dlog10m) * dlog10m
    grid_max = np.ceil((np.log10(mass.max()) + log_h + 2.0 * dlog10m) / dlog10m) * dlog10m
    grid_max += dlog10m
    mass_function = MassFunction(
        Mmin=grid_min,
        Mmax=grid_max,
        dlog10m=dlog10m,
        z=float(z_obs),
        hmf_model="ST",
        sigma_8=MASS_FUNCTION_SIGMA8,
        n=MASS_FUNCTION_NS,
        cosmo_params={
            "H0": cosmology.h0_km_s_mpc,
            "Om0": cosmology.omega_m,
            "Ob0": cosmology.omega_b,
        },
        transfer_params={"extrapolate_with_eh": True},
    )
    grid_mass_msun = np.asarray(mass_function.m, dtype=float) / h
    grid_dndm = np.asarray(mass_function.dndm, dtype=float) * h**4
    valid = (
        np.isfinite(grid_mass_msun)
        & np.isfinite(grid_dndm)
        & (grid_mass_msun > 0.0)
        & (grid_dndm > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("hmf ST returned too few finite positive samples")
    result = np.exp(
        np.interp(
            np.log(mass),
            np.log(grid_mass_msun[valid]),
            np.log(grid_dndm[valid]),
        )
    )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("interpolated ST HMF contains invalid values")
    return result


def main() -> None:
    args = _parse_args()
    if args.z < 0.0:
        raise ValueError("--z must be non-negative")
    if args.logM_max <= args.logM_min:
        raise ValueError("--logM-max must exceed --logM-min")
    if args.n_mass < 2:
        raise ValueError("--n-mass must be at least two")

    cosmology = Cosmology()
    log_mass = np.linspace(float(args.logM_min), float(args.logM_max), int(args.n_mass))
    halo_mass_msun = np.power(10.0, log_mass)
    reed07_dndm = np.asarray(
        compute_reed07_halo_mass_function_dndm(
            halo_mass_msun,
            float(args.z),
            cosmology=cosmology,
        ),
        dtype=float,
    )
    st_dndm = _compute_st_dndm(halo_mass_msun, float(args.z), cosmology=cosmology)
    st_over_reed07 = st_dndm / reed07_dndm
    dndlog10m_reed07 = np.log(10.0) * halo_mass_msun * reed07_dndm
    dndlog10m_st = np.log(10.0) * halo_mass_msun * st_dndm

    csv_path = _resolve_project_path(args.csv_path)
    output_prefix = _resolve_project_path(args.output_prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "z",
                "Mh_Msun",
                "Reed07_dndlog10M_Mpc-3_dex-1",
                "ST_dndlog10M_Mpc-3_dex-1",
                "ST_over_Reed07",
            ],
        )
        writer.writeheader()
        for mass, reed07, st, ratio in zip(
            halo_mass_msun,
            dndlog10m_reed07,
            dndlog10m_st,
            st_over_reed07,
            strict=True,
        ):
            writer.writerow(
                {
                    "z": float(args.z),
                    "Mh_Msun": float(mass),
                    "Reed07_dndlog10M_Mpc-3_dex-1": float(reed07),
                    "ST_dndlog10M_Mpc-3_dex-1": float(st),
                    "ST_over_Reed07": float(ratio),
                }
            )

    plt.style.use("apj")
    fig, (ax_hmf, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(11.0, 5.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    ax_hmf.plot(
        halo_mass_msun,
        dndlog10m_reed07,
        color="#1F3A5F",
        lw=2.6,
        label="Reed07 (production)",
    )
    ax_hmf.plot(
        halo_mass_msun,
        dndlog10m_st,
        color="#DE8F05",
        lw=2.4,
        ls="--",
        label="Sheth--Tormen (ST/SMT)",
    )
    ax_hmf.set_xscale("log")
    ax_hmf.set_yscale("log")
    ax_hmf.set_ylabel(r"$dn/d\log_{10}M_h$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax_hmf.set_title(rf"Halo mass function at $z={float(args.z):g}$")
    ax_hmf.legend(frameon=False, loc="upper right")
    ax_hmf.grid(alpha=0.22)

    ax_ratio.plot(halo_mass_msun, st_over_reed07, color="#DE8F05", lw=2.4)
    ax_ratio.axhline(1.0, color="0.35", lw=1.2, ls=":")
    ax_ratio.axvspan(1.0e9, 1.0e13, color="#1F3A5F", alpha=0.055, lw=0.0)
    ax_ratio.text(
        1.25e9,
        2.15,
        r"$z=6$ slide sampling range",
        color="#1F3A5F",
        fontsize=10,
        va="top",
    )
    ax_ratio.set_xscale("log")
    ax_ratio.set_ylim(0.75, 2.7)
    ax_ratio.set_xlabel(r"Halo mass $M_h$ [$M_\odot$]")
    ax_ratio.set_ylabel("ST / Reed07")
    ax_ratio.grid(alpha=0.22)

    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)

    benchmark_st = _compute_st_dndm(BENCHMARK_MASSES_MSUN, float(args.z), cosmology=cosmology)
    benchmark_reed07 = np.asarray(
        compute_reed07_halo_mass_function_dndm(
            BENCHMARK_MASSES_MSUN,
            float(args.z),
            cosmology=cosmology,
        ),
        dtype=float,
    )
    print("Mh_Msun ST_over_Reed07 delta_dex")
    for mass, ratio in zip(BENCHMARK_MASSES_MSUN, benchmark_st / benchmark_reed07, strict=True):
        print(f"{mass:.6e} {ratio:.8f} {np.log10(ratio):+.8f}")
    print(f"saved_pdf={output_prefix.with_suffix('.pdf')}")
    print(f"saved_png={output_prefix.with_suffix('.png')}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
