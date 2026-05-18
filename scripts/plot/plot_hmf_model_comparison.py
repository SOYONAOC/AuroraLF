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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf import (
    MASS_FUNCTION_MODEL_HMF_REED07,
    compute_halo_mass_function_dndm,
    validate_mass_function_model,
)


DEFAULT_Z_VALUES = (6.0, 12.5, 14.5)
DEFAULT_MODELS = (MASS_FUNCTION_MODEL_HMF_REED07,)
MODEL_LABELS = {
    MASS_FUNCTION_MODEL_HMF_REED07: "hmf Reed07",
}
MODEL_COLORS = {
    MASS_FUNCTION_MODEL_HMF_REED07: "black",
}
MODEL_LINESTYLES = {
    MASS_FUNCTION_MODEL_HMF_REED07: "-",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the current AuroraLF halo mass function backend.")
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--models", nargs="+", type=str, default=list(DEFAULT_MODELS))
    parser.add_argument("--reference-model", type=str, default=MASS_FUNCTION_MODEL_HMF_REED07)
    parser.add_argument("--logM-min", type=float, default=9.0)
    parser.add_argument("--logM-max", type=float, default=13.0)
    parser.add_argument("--n-mass", type=int, default=241)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/hmf_reed07_z6_z12p5_z14p5",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data_save/hmf_reed07_z6_z12p5_z14p5.csv",
    )
    return parser.parse_args()


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _set_log_ylim(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(item) for item in values])
    ax.set_ylim(float(np.min(positive)) * 0.35, float(np.max(positive)) * 2.4)


def _set_ratio_ylim(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(item) for item in values])
    low = max(1.0e-4, float(np.min(positive)) * 0.65)
    high = min(3.0, float(np.max(positive)) * 1.35)
    if low >= high:
        high = low * 10.0
    ax.set_ylim(low, high)


def main() -> None:
    args = _parse_args()
    models = tuple(validate_mass_function_model(model) for model in args.models)
    reference_model = validate_mass_function_model(args.reference_model)
    if reference_model not in models:
        raise ValueError("reference-model must be included in --models")
    if args.n_mass < 2:
        raise ValueError("n-mass must be at least 2")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must be larger than logM-min")

    output_prefix = Path(args.output_prefix).expanduser()
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    log_mass = np.linspace(float(args.logM_min), float(args.logM_max), int(args.n_mass))
    halo_mass = np.power(10.0, log_mass)
    z_values = [float(z) for z in args.z_values]

    hmf_by_z_model: dict[tuple[float, str], np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    for z_obs in z_values:
        for model in models:
            dndm = np.asarray(
                compute_halo_mass_function_dndm(
                    halo_mass,
                    z_obs,
                    mass_function_model=model,
                ),
                dtype=float,
            )
            hmf_by_z_model[(z_obs, model)] = dndm
            for mass, value in zip(halo_mass, dndm, strict=True):
                rows.append(
                    {
                        "z": float(z_obs),
                        "model": model,
                        "Mh_Msun": float(mass),
                        "dndM_Mpc3_Msun": float(value),
                    }
                )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["z", "model", "Mh_Msun", "dndM_Mpc3_Msun"])
        writer.writeheader()
        writer.writerows(rows)

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
        y_values: list[np.ndarray] = []
        ratio_values: list[np.ndarray] = []
        reference = hmf_by_z_model[(z_obs, reference_model)]

        for model in models:
            dndm = hmf_by_z_model[(z_obs, model)]
            valid = np.isfinite(dndm) & (dndm > 0.0)
            y_values.append(dndm[valid])
            ax_top.plot(
                halo_mass[valid],
                dndm[valid],
                color=MODEL_COLORS.get(model, "0.25"),
                ls=MODEL_LINESTYLES.get(model, "-"),
                lw=2.0,
                label=MODEL_LABELS.get(model, model),
            )

            ratio = np.divide(dndm, reference, out=np.full_like(dndm, np.nan), where=reference > 0.0)
            ratio_valid = np.isfinite(ratio) & (ratio > 0.0)
            ratio_values.append(ratio[ratio_valid])
            ax_ratio.plot(
                halo_mass[ratio_valid],
                ratio[ratio_valid],
                color=MODEL_COLORS.get(model, "0.25"),
                ls=MODEL_LINESTYLES.get(model, "-"),
                lw=1.8,
            )

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_title(rf"$z={z_obs:g}$")
        ax_top.grid(alpha=0.22)
        _set_log_ylim(ax_top, y_values)
        if column == 0:
            ax_top.set_ylabel(r"$dn/dM$ [Mpc$^{-3}$ $M_\odot^{-1}$]")
            ax_top.legend(frameon=False, loc="lower left")

        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.axhline(1.0, color="0.35", ls=":", lw=1.0)
        ax_ratio.grid(alpha=0.22)
        _set_ratio_ylim(ax_ratio, ratio_values)
        ax_ratio.set_xlabel(r"$M_{\rm h}$ [$M_\odot$]")
        if column == 0:
            ax_ratio.set_ylabel(f"ratio / {MODEL_LABELS[reference_model]}")

    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    print(f"saved_png={output_prefix.with_suffix('.png')}", flush=True)
    print(f"saved_pdf={output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
