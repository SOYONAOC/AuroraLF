#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf.imf import IMF_MODE_CANONICAL
from auroralf.io.analysis import (
    BURST_CONFIG_DIFFERENCES,
    load_uvlf_result,
    require_compatible_results,
    select_mode_result,
)
from auroralf.results import UVLFRunResult


OBS_UVLF_DIR = PROJECT_ROOT / "external_data" / "observations" / "uvlf"
OBS_FILES = {
    12.5: (
        "redshift_12p5/bouwens.npz",
        "redshift_12p5/donnan24.npz",
        "redshift_12p5/harikane23_uvlf_z12.npz",
    ),
}
MODE_LABELS = {
    "canonical": "canonical",
    "z10_mild_topheavy": "low-Z mild TH",
    "mah_burst_mild_topheavy": "MAH-burst TH",
}
MODE_COLORS = {
    "canonical": "black",
    "z10_mild_topheavy": "#c44e52",
    "mah_burst_mild_topheavy": "#1f77b4",
}
OBS_MUV_MIN = -22.5
OBS_MUV_MAX = -16.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UVLFs with and without SFR burst scatter.")
    parser.add_argument("--no-burst-hdf5", required=True)
    parser.add_argument("--burst-hdf5", required=True)
    parser.add_argument("--z", type=float, default=12.5)
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional IMF mode to plot by itself, for example z10_mild_topheavy.",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args(argv)


def _resolve_path(path_text: str, *, must_exist: bool) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    if z_value not in OBS_FILES:
        raise ValueError(f"no observational UVLF files configured for z={z_value:g}")
    datasets: list[dict[str, np.ndarray | str]] = []
    for relative_path in OBS_FILES[z_value]:
        file_path = OBS_UVLF_DIR / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"required observation file not found: {file_path}")
        with np.load(file_path, allow_pickle=False) as payload:
            label_array = np.asarray(payload["label"]).astype(str)
            label = str(label_array[0]) if label_array.size > 0 else file_path.stem
            is_upper_limit = (
                np.asarray(payload["is_upper_limit"], dtype=bool)
                if "is_upper_limit" in payload.files
                else np.zeros_like(np.asarray(payload["phierr"], dtype=float), dtype=bool)
            )
            datasets.append(
                {
                    "label": label,
                    "Muv": np.asarray(payload["muverr"], dtype=float),
                    "phi": np.asarray(payload["phierr"], dtype=float),
                    "mag_err": np.asarray(payload["mag_err"], dtype=float),
                    "phi_err_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                    "phi_err_up": np.asarray(payload["phi_err_up"], dtype=float),
                    "is_upper_limit": is_upper_limit,
                }
            )
    return datasets


def _load_model_result(path: Path) -> UVLFRunResult:
    return load_uvlf_result(path)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def main() -> None:
    args = _parse_args()
    no_burst_path = _resolve_path(args.no_burst_hdf5, must_exist=True)
    burst_path = _resolve_path(args.burst_hdf5, must_exist=True)
    output_prefix = _resolve_path(args.output_prefix, must_exist=False).with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    no_burst = _load_model_result(no_burst_path)
    burst = _load_model_result(burst_path)
    try:
        require_compatible_results(
            no_burst,
            burst,
            allowed_config_differences=BURST_CONFIG_DIFFERENCES,
            context="burst comparison",
        )
        mode_names = list(no_burst.config.stellar_population.imf_modes)
        if args.mode is not None:
            if args.mode not in mode_names:
                raise ValueError(f"requested mode is not present in HDF5 artifacts: {args.mode}")
            mode_names = [str(args.mode)]
        elif IMF_MODE_CANONICAL not in mode_names:
            raise ValueError("canonical mode is required for this comparison")
        single_mode = len(mode_names) == 1

        plt.style.use("apj")
        plt.rcParams.update(
            {
                "font.size": 8.5,
                "axes.titlesize": 10.0,
                "axes.labelsize": 9.5,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 6.3,
            }
        )
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(5.35, 5.55),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.25, 1.0]},
        )
        ax_top, ax_ratio = axes
        ylim_values: list[np.ndarray] = []
        ratio_values: list[np.ndarray] = []
        summary_lines = [
            f"no_burst_hdf5: {no_burst_path}",
            f"burst_hdf5: {burst_path}",
            f"z: {float(args.z):g}",
            f"mode: {' '.join(mode_names)}",
            f"burst_scatter_dex: {burst.config.star_formation.burst_scatter_dex:g}",
            "burst_scatter_timescale_myr: "
            f"{burst.config.star_formation.burst_scatter_correlation_timescale_myr:g}",
            "",
        ]

        for mode in mode_names:
            no_burst_series = select_mode_result(
                no_burst,
                redshift=float(args.z),
                mode=mode,
            )
            burst_series = select_mode_result(
                burst,
                redshift=float(args.z),
                mode=mode,
            )
            centers = no_burst_series.bin_centers_muv
            color = MODE_COLORS.get(mode, "0.4")
            label = "low-Z TH" if single_mode and mode == "z10_mild_topheavy" else MODE_LABELS.get(mode, mode.replace("_", " "))
            phi_no = no_burst_series.phi_observed_per_mpc3_per_mag
            phi_burst = burst_series.phi_observed_per_mpc3_per_mag
            valid_no = np.isfinite(phi_no) & (phi_no > 0.0)
            valid_burst = np.isfinite(phi_burst) & (phi_burst > 0.0)
            ylim_values.append(phi_no[valid_no])
            ylim_values.append(phi_burst[valid_burst])
            if single_mode:
                no_burst_label = f"{label}: no burst"
                burst_label = f"{label}: + SFR burst"
            else:
                no_burst_label = None
                burst_label = label
            ax_top.plot(
                centers[valid_no],
                phi_no[valid_no],
                color=color,
                ls="--",
                lw=1.9,
                alpha=0.75,
                label=no_burst_label,
            )
            ax_top.plot(
                centers[valid_burst],
                phi_burst[valid_burst],
                color=color,
                ls="-",
                lw=2.4,
                label=burst_label,
            )

            ratio = np.divide(phi_burst, phi_no, out=np.full_like(phi_burst, np.nan), where=phi_no > 0.0)
            obs_mask = (centers >= OBS_MUV_MIN) & (centers <= OBS_MUV_MAX)
            valid_ratio = np.isfinite(ratio) & (ratio > 0.0) & obs_mask
            ratio_values.append(ratio[valid_ratio])
            ax_ratio.plot(centers[valid_ratio], ratio[valid_ratio], color=color, lw=2.0, label=label)
            summary_lines.append(f"{mode}_burst_over_no_burst_obs_median={float(np.nanmedian(ratio[valid_ratio])):.6f}")
            summary_lines.append(f"{mode}_burst_over_no_burst_obs_min={float(np.nanmin(ratio[valid_ratio])):.6f}")
            summary_lines.append(f"{mode}_burst_over_no_burst_obs_max={float(np.nanmax(ratio[valid_ratio])):.6f}")
        summary_lines.append("")

        obs_markers = ["o", "s", "^", "D"]
        for obs_index, obs in enumerate(_load_observational_uvlf(float(args.z))):
            marker = obs_markers[obs_index % len(obs_markers)]
            muv = np.asarray(obs["Muv"], dtype=float)
            phi = np.asarray(obs["phi"], dtype=float)
            mag_err = np.asarray(obs["mag_err"], dtype=float)
            phi_err_lo = np.asarray(obs["phi_err_lo"], dtype=float)
            phi_err_up = np.asarray(obs["phi_err_up"], dtype=float)
            is_upper_limit = np.asarray(obs["is_upper_limit"], dtype=bool)
            valid = np.isfinite(muv) & np.isfinite(phi) & (phi > 0.0)
            if not np.any(valid):
                raise RuntimeError(f"observation dataset has no valid positive points: {obs['label']}")
            ylim_values.append(phi[valid])
            ax_top.errorbar(
                muv[valid],
                phi[valid],
                xerr=mag_err[valid],
                yerr=np.vstack([phi_err_lo[valid], phi_err_up[valid]]),
                uplims=is_upper_limit[valid],
                fmt=marker,
                ms=5.2,
                color="#1f4e79",
                mec="white",
                mew=0.6,
                elinewidth=1.0,
                capsize=2.0,
                alpha=0.9,
                label=str(obs["label"]),
                zorder=3,
            )

        positive_ylim = np.concatenate([_finite_positive(item) for item in ylim_values])
        ax_top.set_yscale("log")
        ax_top.set_xlim(-24.5, -15.0)
        ax_top.set_ylim(max(1.0e-10, float(np.min(positive_ylim)) * 0.35), float(np.max(positive_ylim)) * 2.5)
        ax_top.set_ylabel(r"$\phi(M_{\rm UV})$ [mag$^{-1}$ Mpc$^{-3}$]")
        title_prefix = "gate mode" if single_mode else "all modes"
        ax_top.set_title(
            rf"$z={float(args.z):g}$ UVLF: {title_prefix}, no burst (dashed) vs burst (solid)",
            pad=3,
        )
        ax_top.grid(alpha=0.22)
        ax_top.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.7)

        positive_ratio = np.concatenate([_finite_positive(item) for item in ratio_values])
        ax_ratio.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_ratio.axvspan(-24.5, OBS_MUV_MIN, color="0.94", zorder=-10)
        ax_ratio.axvspan(OBS_MUV_MAX, -15.0, color="0.94", zorder=-10)
        ax_ratio.set_ylim(max(0.5, float(np.min(positive_ratio)) * 0.82), float(np.max(positive_ratio)) * 1.18)
        ax_ratio.set_xlabel(r"$M_{\rm UV}$")
        ax_ratio.set_ylabel("burst / no burst")
        ax_ratio.grid(alpha=0.22)

        png_path = output_prefix.with_suffix(".png")
        pdf_path = output_prefix.with_suffix(".pdf")
        txt_path = output_prefix.parent / f"{output_prefix.name}_summary.txt"
        fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    finally:
        del no_burst, burst

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")


if __name__ == "__main__":
    main()
