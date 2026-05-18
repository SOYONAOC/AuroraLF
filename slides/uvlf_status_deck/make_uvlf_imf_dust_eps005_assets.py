#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("apj")


DEFAULT_NPZ = "../../data_save/uvlf_imf_dust_eps005_compare_allz_20260324_142304.npz"
DEFAULT_SECONDARY_NPZ = "../../data_save/uvlf_imf_dust_eps005_topheavy100100_compare_allz.npz"
DEFAULT_OUTPUT_DIR = "assets"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slide assets for the dust+eps0=0.05 IMF UVLF comparison."
    )
    parser.add_argument("--npz-path", type=str, default=DEFAULT_NPZ)
    parser.add_argument("--secondary-npz-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _z_tag(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _obs_dir_from_z(project_root: Path, z_value: float) -> Path:
    return project_root / "obsdata" / f"redshift_{str(float(z_value)).replace('.', 'p').rstrip('0').rstrip('p')}"


def _load_observational_uvlf(project_root: Path, z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = _obs_dir_from_z(project_root, z_value)
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_path in sorted(obs_dir.glob("*.npz")):
        payload = np.load(file_path)
        label_array = np.asarray(payload["label"])
        label = str(label_array[0]) if label_array.size > 0 else file_path.stem
        datasets.append(
            {
                "label": label,
                "Muv": np.asarray(payload["muverr"], dtype=float),
                "phi": np.asarray(payload["phierr"], dtype=float),
                "mag_err": np.asarray(payload["mag_err"], dtype=float),
                "phi_err_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                "phi_err_up": np.asarray(payload["phi_err_up"], dtype=float),
            }
        )
    return datasets


def main() -> None:
    args = _parse_args()
    slides_dir = Path(__file__).resolve().parent
    project_root = slides_dir.parents[1]
    npz_path = (slides_dir / args.npz_path).resolve() if not Path(args.npz_path).is_absolute() else Path(args.npz_path)
    secondary_npz_path = None
    if args.secondary_npz_path is not None:
        secondary_npz_path = (
            (slides_dir / args.secondary_npz_path).resolve()
            if not Path(args.secondary_npz_path).is_absolute()
            else Path(args.secondary_npz_path)
        )
    output_dir = (slides_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    secondary_data = np.load(secondary_npz_path) if secondary_npz_path is not None and secondary_npz_path.exists() else None
    z_values = [float(z) for z in np.asarray(data["z_values"], dtype=float)]
    obs_markers = ["o", "s", "^", "D", "P", "X"]

    for z_obs in z_values:
        tag = _z_tag(z_obs)
        centers = np.asarray(data[f"{tag}_bin_centers"], dtype=float)
        phi_old = np.asarray(data[f"{tag}_old_phi"], dtype=float)
        phi_top = np.asarray(data[f"{tag}_topheavy_phi"], dtype=float)
        ratio = np.asarray(data[f"{tag}_phi_ratio_topheavy_over_old"], dtype=float)
        phi_top_100100 = None
        ratio_top_100100 = None
        if secondary_data is not None:
            secondary_centers = np.asarray(secondary_data[f"{tag}_bin_centers"], dtype=float)
            if not np.allclose(centers, secondary_centers, rtol=0.0, atol=0.0):
                raise ValueError(f"secondary bin centers do not match primary centers at z={z_obs:g}")
            phi_top_100100 = np.asarray(secondary_data[f"{tag}_topheavy_phi"], dtype=float)
            ratio_top_100100 = np.asarray(secondary_data[f"{tag}_phi_ratio_topheavy_over_old"], dtype=float)
        obs_sets = _load_observational_uvlf(project_root, z_obs)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.9, 6.2),
            constrained_layout=False,
            sharex=True,
            gridspec_kw={"height_ratios": [2.45, 1.45]},
        )
        fig.subplots_adjust(left=0.112, right=0.992, top=0.988, bottom=0.09, hspace=0.18)
        ax_top, ax_bottom = axes

        valid_old = np.isfinite(phi_old) & (phi_old > 0.0)
        valid_top = np.isfinite(phi_top) & (phi_top > 0.0)
        ax_top.plot(centers[valid_old], phi_old[valid_old], color="black", lw=2.2, label="legacy IMF")
        ax_top.plot(centers[valid_top], phi_top[valid_top], color="#c44e52", lw=2.2, label="top-heavy IMF (100-300)")
        if phi_top_100100 is not None:
            valid_top_100100 = np.isfinite(phi_top_100100) & (phi_top_100100 > 0.0)
            ax_top.plot(
                centers[valid_top_100100],
                phi_top_100100[valid_top_100100],
                color="#dd8452",
                lw=2.1,
                ls="--",
                label="top-heavy IMF (100-100)",
            )

        for obs_index, obs in enumerate(obs_sets):
            marker = obs_markers[obs_index % len(obs_markers)]
            muv = np.asarray(obs["Muv"], dtype=float)
            phi = np.asarray(obs["phi"], dtype=float)
            mag_err = np.asarray(obs["mag_err"], dtype=float)
            phi_err_lo = np.asarray(obs["phi_err_lo"], dtype=float)
            phi_err_up = np.asarray(obs["phi_err_up"], dtype=float)
            valid = np.isfinite(muv) & np.isfinite(phi) & (phi > 0.0)
            if not np.any(valid):
                continue
            ax_top.errorbar(
                muv[valid],
                phi[valid],
                xerr=mag_err[valid],
                yerr=np.vstack([phi_err_lo[valid], phi_err_up[valid]]),
                fmt=marker,
                ms=5.2,
                color="#1f4e79",
                mec="white",
                mew=0.6,
                elinewidth=0.9,
                capsize=1.8,
                alpha=0.92,
                label=str(obs["label"]),
            )

        ax_top.set_yscale("log")
        ax_top.set_xlim(-24.5, -15.0)
        ax_top.set_ylim(1.0e-7, 1.0e-1)
        ax_top.margins(x=0.01)
        ax_top.tick_params(axis="both", which="both", labelsize=12)
        ax_top.grid(alpha=0.22)
        ax_top.set_ylabel(r"$\phi(M_{\rm UV})$ [dex$^{-1}$ Mpc$^{-3}$]", labelpad=3)
        ax_top.tick_params(axis="x", which="both", labelbottom=True)
        ax_top.text(
            0.07,
            0.92,
            rf"$z={z_obs:g}$",
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            fontsize=18,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
        )
        ax_top.legend(frameon=False, fontsize=7.8, loc="lower left")

        valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
        ax_bottom.plot(centers[valid_ratio], ratio[valid_ratio], color="#c44e52", lw=2.1, label="100-300 / legacy")
        if ratio_top_100100 is not None:
            valid_ratio_100100 = np.isfinite(ratio_top_100100) & (ratio_top_100100 > 0.0)
            ax_bottom.plot(
                centers[valid_ratio_100100],
                ratio_top_100100[valid_ratio_100100],
                color="#dd8452",
                lw=2.0,
                ls="--",
                label="100-100 / legacy",
            )
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_bottom.set_xlim(-24.5, -15.0)
        upper = max(3.2, float(np.nanmax(ratio[valid_ratio])) * 1.03 if np.any(valid_ratio) else 3.2)
        if ratio_top_100100 is not None:
            valid_ratio_100100 = np.isfinite(ratio_top_100100) & (ratio_top_100100 > 0.0)
            if np.any(valid_ratio_100100):
                upper = max(upper, float(np.nanmax(ratio_top_100100[valid_ratio_100100])) * 1.03)
        ax_bottom.set_ylim(0.95, upper)
        ax_bottom.margins(x=0.01)
        ax_bottom.tick_params(axis="both", which="both", labelsize=12)
        ax_bottom.grid(alpha=0.22)
        ax_bottom.tick_params(axis="x", which="both", labelbottom=True)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        ax_bottom.set_ylabel("top-heavy\nstrength ratio", labelpad=2)
        if ratio_top_100100 is not None:
            ax_bottom.legend(frameon=False, fontsize=7.2, loc="upper right")

        output_path = output_dir / f"uvlf_imf_dust_eps005_{tag}.pdf"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"saved_pdf={output_path}")


if __name__ == "__main__":
    main()
