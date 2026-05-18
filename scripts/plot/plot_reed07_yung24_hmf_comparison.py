#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function as colossus_mass_function

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf import compute_reed07_halo_mass_function_dndm
from auroralf.uvlf.hmf_sampling import (
    MASS_FUNCTION_H,
    MASS_FUNCTION_NS,
    MASS_FUNCTION_OMEGA_B_H2,
    MASS_FUNCTION_OMEGA_M,
    MASS_FUNCTION_SIGMA8,
)


DEFAULT_Z_VALUES = (6.0, 12.5, 14.5)
DEFAULT_OUTPUT_PREFIX = "outputs/hmf_reed07_yung24_z6_z12p5_z14p5"
DEFAULT_CSV_PATH = "data_save/hmf_reed07_yung24_z6_z12p5_z14p5.csv"
BENCHMARK_MASSES = np.array([1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13], dtype=float)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hmf Reed07 FoF and Colossus Yung24 vir halo mass functions.")
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--logM-min", type=float, default=9.0)
    parser.add_argument("--logM-max", type=float, default=13.0)
    parser.add_argument("--n-mass", type=int, default=241)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    return parser.parse_args()


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _set_colossus_cosmology() -> None:
    h = MASS_FUNCTION_H
    params = {
        "flat": True,
        "H0": 100.0 * h,
        "Om0": MASS_FUNCTION_OMEGA_M,
        "Ob0": MASS_FUNCTION_OMEGA_B_H2 / h**2,
        "sigma8": MASS_FUNCTION_SIGMA8,
        "ns": MASS_FUNCTION_NS,
    }
    cosmology.addCosmology("AuroraLF_hmf_compare", params)
    cosmology.setCosmology("AuroraLF_hmf_compare")


def _colossus_yung24_dndm_physical(halo_mass_msun: np.ndarray, z_obs: float) -> np.ndarray:
    h = MASS_FUNCTION_H
    halo_mass_msun_over_h = np.asarray(halo_mass_msun, dtype=float) * h
    dndlnm_h3_mpc3 = np.asarray(
        colossus_mass_function.massFunction(
            halo_mass_msun_over_h,
            float(z_obs),
            q_in="M",
            q_out="dndlnM",
            mdef="vir",
            model="yung24",
        ),
        dtype=float,
    )
    return dndlnm_h3_mpc3 * h**3 / np.asarray(halo_mass_msun, dtype=float)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _set_log_ylim(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(np.asarray(item, dtype=float)) for item in values])
    ax.set_ylim(float(np.min(positive)) * 0.35, float(np.max(positive)) * 2.4)


def _set_ratio_ylim(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = _finite_positive(np.concatenate([np.asarray(item, dtype=float) for item in values]))
    ax.set_ylim(max(1.0e-4, float(np.min(positive)) * 0.65), float(np.max(positive)) * 1.35)


def main() -> None:
    args = _parse_args()
    if args.n_mass < 2:
        raise ValueError("n-mass must be at least 2")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must be larger than logM-min")

    _set_colossus_cosmology()

    output_prefix = _resolve_project_path(args.output_prefix)
    csv_path = _resolve_project_path(args.csv_path)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.txt")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    z_values = [float(z) for z in args.z_values]
    log_mass = np.linspace(float(args.logM_min), float(args.logM_max), int(args.n_mass))
    halo_mass = np.power(10.0, log_mass)

    hmf_by_z_model: dict[tuple[float, str], np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    summary_lines = [
        "models: hmf Reed07 FoF, Colossus Yung24 vir",
        "units: dn/dM in physical Mpc^-3 Msun^-1; input halo mass in physical Msun",
        f"cosmology: h={MASS_FUNCTION_H}, Om0={MASS_FUNCTION_OMEGA_M}, Obh2={MASS_FUNCTION_OMEGA_B_H2}, "
        f"sigma8={MASS_FUNCTION_SIGMA8}, ns={MASS_FUNCTION_NS}",
        "",
    ]

    for z_obs in z_values:
        reed07 = np.asarray(
            compute_reed07_halo_mass_function_dndm(
                halo_mass,
                z_obs,
            ),
            dtype=float,
        )
        yung24 = _colossus_yung24_dndm_physical(halo_mass, z_obs)
        if not np.all(np.isfinite(yung24)) or np.any(yung24 <= 0.0):
            raise RuntimeError(f"Colossus Yung24 returned non-positive or non-finite values at z={z_obs:g}")

        hmf_by_z_model[(z_obs, "reed07_fof")] = reed07
        hmf_by_z_model[(z_obs, "yung24_vir")] = yung24
        for model, dndm in [("hmf_reed07_fof", reed07), ("colossus_yung24_vir", yung24)]:
            for mass, value in zip(halo_mass, dndm, strict=True):
                rows.append(
                    {
                        "z": float(z_obs),
                        "model": model,
                        "Mh_Msun": float(mass),
                        "dndM_Mpc3_Msun": float(value),
                    }
                )

        benchmark_reed07 = np.asarray(
            compute_reed07_halo_mass_function_dndm(
                BENCHMARK_MASSES,
                z_obs,
            ),
            dtype=float,
        )
        benchmark_yung24 = _colossus_yung24_dndm_physical(BENCHMARK_MASSES, z_obs)
        benchmark_ratio = benchmark_yung24 / benchmark_reed07
        summary_lines.append(f"z={z_obs:g}")
        summary_lines.append(
            "  benchmark_masses_Msun: " + " ".join(f"{mass:.3g}" for mass in BENCHMARK_MASSES)
        )
        summary_lines.append(
            "  Yung24_vir_over_Reed07_FoF: " + " ".join(f"{ratio:.6g}" for ratio in benchmark_ratio)
        )
        summary_lines.append("")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["z", "model", "Mh_Msun", "dndM_Mpc3_Msun"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plt.style.use("apj")
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.6})
    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(4.1 * len(z_values), 5.4),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.05, 1.0]},
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, z_obs in enumerate(z_values):
        ax_top = axes[0, column]
        ax_ratio = axes[1, column]
        reed07 = hmf_by_z_model[(z_obs, "reed07_fof")]
        yung24 = hmf_by_z_model[(z_obs, "yung24_vir")]
        y_values = [reed07, yung24]

        ax_top.plot(halo_mass, reed07, color="black", lw=2.2, label="Reed07 FoF (hmf)")
        ax_top.plot(halo_mass, yung24, color="#1f77b4", lw=2.2, ls="-.", label="Yung24 vir (Colossus)")
        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_title(rf"$z={z_obs:g}$")
        ax_top.grid(alpha=0.22)
        _set_log_ylim(ax_top, y_values)
        if column == 0:
            ax_top.set_ylabel(r"$dn/dM$ [Mpc$^{-3}$ $M_\odot^{-1}$]")
            ax_top.legend(frameon=False, loc="lower left")

        ratio = yung24 / reed07
        valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
        ax_ratio.plot(halo_mass[valid_ratio], ratio[valid_ratio], color="#1f77b4", lw=2.0)
        ax_ratio.axhline(1.0, color="0.35", ls=":", lw=1.0)
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.grid(alpha=0.22)
        _set_ratio_ylim(ax_ratio, [ratio[valid_ratio]])
        ax_ratio.set_xlabel(r"$M_{\rm h}$ [$M_\odot$]")
        if column == 0:
            ax_ratio.set_ylabel("Yung24 vir / Reed07 FoF")

    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    print(f"saved_png={output_prefix.with_suffix('.png')}", flush=True)
    print(f"saved_pdf={output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
