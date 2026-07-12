#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.io.analysis import load_uvlf_result, select_mode_result
from auroralf.results import UVLFRunResult

DEFAULT_HDF5_PATH = PROJECT_ROOT / "data_save" / "uvlf_current_full_noburst_z6_z10_z12p5_20260522.h5"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "popii_uvlf_fit_check_z6_z12p5"
OBSERVATION_FILES = {
    6.0: (
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_6" / "Finkelstein_uvlf_z6.npz",
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_6" / "bouwens21_uvlf_z6.npz",
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_6" / "bowler_uvlf_z6.npz",
    ),
    12.5: (
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_12p5" / "bouwens.npz",
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_12p5" / "donnan24.npz",
        PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_12p5" / "harikane23_uvlf_z12.npz",
    ),
}
OBSERVATION_COLORS = {
    "Finkelstein+15": "#005AB5",
    "Bouwens+21": "#DC3220",
    "Bowler+15": "#009E73",
    "Bouwens+23, $z\\sim12-13$": "#DC3220",
    "Donnan+24, $z\\sim12.5$": "#7A4EAB",
    "Harikane+23, $z\\sim12$": "#E69F00",
}
OBSERVATION_MARKERS = {
    "Finkelstein+15": "o",
    "Bouwens+21": "s",
    "Bowler+15": "D",
    "Bouwens+23, $z\\sim12-13$": "s",
    "Donnan+24, $z\\sim12.5$": "o",
    "Harikane+23, $z\\sim12$": "^",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot canonical Pop II UVLF fit checks at z=6 and z=12.5 from a UVLF HDF5 artifact."
    )
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--min-raw-counts", type=int, default=10)
    parser.add_argument("--z-values", nargs="+", type=float, default=[6.0, 12.5])
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_observation(path: Path) -> dict[str, np.ndarray | str]:
    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    required = ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up", "label")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise KeyError(f"Observation file {path} is missing required fields: {missing}")

    label = str(np.asarray(payload["label"])[0])
    muv = np.asarray(payload["muverr"], dtype=float)
    phi = np.asarray(payload["phierr"], dtype=float)
    mag_err = np.asarray(payload["mag_err"], dtype=float)
    phi_err_lo = np.asarray(payload["phi_err_lo"], dtype=float)
    phi_err_up = np.asarray(payload["phi_err_up"], dtype=float)
    upper_limit = (
        np.asarray(payload["is_upper_limit"], dtype=bool)
        if "is_upper_limit" in payload.files
        else np.zeros_like(phi, dtype=bool)
    )
    shapes = {array.shape for array in (muv, phi, mag_err, phi_err_lo, phi_err_up, upper_limit)}
    if len(shapes) != 1:
        raise ValueError(f"Observation file {path} has inconsistent array shapes: {shapes}")
    finite = np.isfinite(muv) & np.isfinite(phi) & np.isfinite(mag_err)
    finite &= np.isfinite(phi_err_lo) & np.isfinite(phi_err_up)
    if not np.all(finite):
        raise ValueError(f"Observation file {path} contains non-finite values")
    if np.any(phi <= 0.0):
        raise ValueError(f"Observation file {path} contains non-positive phi values")
    if np.any(phi_err_lo < 0.0) or np.any(phi_err_up < 0.0):
        raise ValueError(f"Observation file {path} contains negative phi errors")

    return {
        "label": label,
        "muv": muv,
        "phi": phi,
        "mag_err": mag_err,
        "phi_err_lo": phi_err_lo,
        "phi_err_up": phi_err_up,
        "upper_limit": upper_limit,
    }


def _load_model_result(path: Path) -> UVLFRunResult:
    return load_uvlf_result(path)


def _load_model(
    result: UVLFRunResult,
    z_value: float,
    min_raw_counts: int,
) -> dict[str, np.ndarray]:
    series = select_mode_result(result, redshift=z_value, mode="canonical")
    centers = series.bin_centers_muv
    phi = series.phi_observed_per_mpc3_per_mag
    intrinsic_phi = series.phi_intrinsic_per_mpc3_per_mag
    raw_counts = series.raw_counts
    valid = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0) & (raw_counts >= int(min_raw_counts))
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(f"Fewer than three model bins pass min_raw_counts={min_raw_counts} at z={z_value:g}")
    order = np.argsort(centers[valid])
    return {
        "centers": centers[valid][order],
        "phi": phi[valid][order],
        "intrinsic_phi": intrinsic_phi[valid][order],
        "raw_counts": raw_counts[valid][order],
    }


def _interp_log_phi(model: dict[str, np.ndarray], muv: np.ndarray) -> np.ndarray:
    centers = np.asarray(model["centers"], dtype=float)
    phi = np.asarray(model["phi"], dtype=float)
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("Model MUV centers must be strictly increasing for interpolation")
    log_phi = np.log10(phi)
    return np.power(10.0, np.interp(muv, centers, log_phi, left=np.nan, right=np.nan))


def _write_ratio_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise RuntimeError("No fit-check rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "z",
        "dataset",
        "Muv",
        "Muv_err",
        "phi_obs",
        "phi_err_lo",
        "phi_err_up",
        "is_upper_limit",
        "phi_model",
        "model_over_obs",
        "delta_log10_model_over_obs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_offsets(rows: list[dict[str, float | int | str]], z_value: float) -> str:
    selected = [
        row
        for row in rows
        if float(row["z"]) == float(z_value)
        and int(row["is_upper_limit"]) == 0
        and np.isfinite(float(row["delta_log10_model_over_obs"]))
    ]
    if not selected:
        raise RuntimeError(f"No finite detection offsets available for z={z_value:g}")
    delta = np.array([float(row["delta_log10_model_over_obs"]) for row in selected], dtype=float)
    return (
        f"z={z_value:g}: Nobs={delta.size}, "
        f"median_delta_log10(model/obs)={np.median(delta):+.3f} dex, "
        f"mean_abs_delta={np.mean(np.abs(delta)):.3f} dex, "
        f"max_abs_delta={np.max(np.abs(delta)):.3f} dex"
    )


def main() -> None:
    args = _parse_args()
    if args.min_raw_counts < 1:
        raise ValueError("--min-raw-counts must be at least 1")

    hdf5_path = _resolve_path(args.hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"UVLF HDF5 artifact not found: {hdf5_path}")
    output_prefix = _resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    result = _load_model_result(hdf5_path)
    try:
        plt.style.use("apj")
        fig, axes = plt.subplots(
            2,
            len(args.z_values),
            figsize=(5.25 * len(args.z_values), 6.2),
            sharex="col",
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        if len(args.z_values) == 1:
            axes = np.asarray(axes).reshape(2, 1)

        for column, z_value in enumerate(args.z_values):
            z_value = float(z_value)
            if z_value not in OBSERVATION_FILES:
                raise KeyError(f"No observation file list is configured for z={z_value:g}")
            model = _load_model(result, z_value, args.min_raw_counts)
            observations = [_load_observation(path) for path in OBSERVATION_FILES[z_value]]

            ax = axes[0, column]
            ax.plot(
                model["centers"],
                model["phi"],
                color="#1F5C8B",
                lw=2.4,
                label="canonical Pop II",
                zorder=5,
            )

            ratio_ax = axes[1, column]
            ratio_ax.axhline(1.0, color="0.25", lw=1.0, ls=":")
            obs_muv_for_limits: list[np.ndarray] = []
            y_values: list[np.ndarray] = []
            for obs in observations:
                label = str(obs["label"])
                color = OBSERVATION_COLORS.get(label)
                marker = OBSERVATION_MARKERS.get(label)
                if color is None or marker is None:
                    raise KeyError(f"No plotting style configured for observation label: {label}")
                muv = np.asarray(obs["muv"], dtype=float)
                phi_obs = np.asarray(obs["phi"], dtype=float)
                upper_limit = np.asarray(obs["upper_limit"], dtype=bool)
                obs_muv_for_limits.append(muv)
                y_values.append(phi_obs)
                ax.errorbar(
                    muv,
                    phi_obs,
                    xerr=np.asarray(obs["mag_err"], dtype=float),
                    yerr=[
                        np.asarray(obs["phi_err_lo"], dtype=float),
                        np.asarray(obs["phi_err_up"], dtype=float),
                    ],
                    uplims=upper_limit,
                    fmt=marker,
                    ms=5.6,
                    color=color,
                    markeredgecolor="0.15",
                    markeredgewidth=0.5,
                    capsize=2.4,
                    elinewidth=0.9,
                    linestyle="none",
                    label=label,
                    zorder=10,
                )

                phi_model_at_obs = _interp_log_phi(model, muv)
                ratio = phi_model_at_obs / phi_obs
                finite_ratio = np.isfinite(ratio) & (ratio > 0.0)
                if np.any(finite_ratio):
                    phi_err_lo = np.asarray(obs["phi_err_lo"], dtype=float)
                    phi_err_up = np.asarray(obs["phi_err_up"], dtype=float)
                    ratio_err_lo = ratio - phi_model_at_obs / (phi_obs + phi_err_up)
                    finite_upper_error = phi_obs > phi_err_lo
                    ratio_err_up = np.divide(
                        phi_model_at_obs,
                        phi_obs - phi_err_lo,
                        out=np.full_like(ratio, np.nan),
                        where=finite_upper_error,
                    ) - ratio
                    finite_error = finite_ratio & np.isfinite(ratio_err_lo) & np.isfinite(ratio_err_up)
                    finite_error &= (ratio_err_lo >= 0.0) & (ratio_err_up >= 0.0)
                    finite_marker_only = finite_ratio & ~finite_error
                    if np.any(finite_marker_only):
                        ratio_ax.errorbar(
                            muv[finite_marker_only],
                            ratio[finite_marker_only],
                            xerr=np.asarray(obs["mag_err"], dtype=float)[finite_marker_only],
                            fmt=marker,
                            ms=5.0,
                            color=color,
                            markeredgecolor="0.15",
                            markeredgewidth=0.45,
                            linestyle="none",
                        )
                    ratio_ax.errorbar(
                        muv[finite_error],
                        ratio[finite_error],
                        xerr=np.asarray(obs["mag_err"], dtype=float)[finite_error],
                        yerr=[ratio_err_lo[finite_error], ratio_err_up[finite_error]],
                        uplims=upper_limit[finite_error],
                        fmt=marker,
                        ms=5.0,
                        color=color,
                        markeredgecolor="0.15",
                        markeredgewidth=0.45,
                        capsize=2.2,
                        elinewidth=0.8,
                        linestyle="none",
                    )

                for index in range(muv.size):
                    model_value = float(phi_model_at_obs[index])
                    ratio_value = float(ratio[index]) if np.isfinite(ratio[index]) else np.nan
                    rows.append(
                        {
                            "z": z_value,
                            "dataset": label,
                            "Muv": float(muv[index]),
                            "Muv_err": float(np.asarray(obs["mag_err"], dtype=float)[index]),
                            "phi_obs": float(phi_obs[index]),
                            "phi_err_lo": float(np.asarray(obs["phi_err_lo"], dtype=float)[index]),
                            "phi_err_up": float(np.asarray(obs["phi_err_up"], dtype=float)[index]),
                            "is_upper_limit": int(bool(upper_limit[index])),
                            "phi_model": model_value,
                            "model_over_obs": ratio_value,
                            "delta_log10_model_over_obs": float(np.log10(ratio_value)) if ratio_value > 0.0 else np.nan,
                        }
                    )

            ax.set_yscale("log")
            ax.set_xlim(-24.5, -15.0)
            obs_muv_all = np.concatenate(obs_muv_for_limits)
            model_limit_mask = np.asarray(model["centers"], dtype=float) >= float(np.min(obs_muv_all) - 0.75)
            model_limit_mask &= np.asarray(model["centers"], dtype=float) <= float(np.max(obs_muv_all) + 0.75)
            if not np.any(model_limit_mask):
                raise RuntimeError(f"No model bins overlap the observation magnitude range at z={z_value:g}")
            y_values.append(np.asarray(model["phi"], dtype=float)[model_limit_mask])
            positive_y = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in y_values])
            ax.set_ylim(max(1.0e-13, float(np.min(positive_y)) * 0.35), float(np.max(positive_y)) * 3.0)
            ax.set_title(rf"$z={z_value:g}$")
            ax.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ mag$^{-1}$]")
            ax.grid(True, which="major", color="#CBD3DD", lw=0.7, alpha=0.85)
            ax.grid(True, which="minor", color="#E5E9EF", lw=0.45, alpha=0.75)
            ax.legend(frameon=True, fontsize=8, loc="lower right")

            ratio_ax.set_yscale("log")
            ratio_ax.set_xlim(-24.5, -15.0)
            ratio_values = np.array(
                [
                    float(row["model_over_obs"])
                    for row in rows
                    if float(row["z"]) == z_value and np.isfinite(float(row["model_over_obs"]))
                ],
                dtype=float,
            )
            ratio_values = ratio_values[ratio_values > 0.0]
            if ratio_values.size == 0:
                raise RuntimeError(f"No finite model/obs ratios available for z={z_value:g}")
            ratio_ax.set_ylim(max(0.03, float(np.min(ratio_values)) * 0.45), min(100.0, float(np.max(ratio_values)) * 2.2))
            ratio_ax.set_xlabel(r"$M_{\rm UV}$")
            ratio_ax.set_ylabel("model/obs")
            ratio_ax.grid(True, which="major", color="#CBD3DD", lw=0.7, alpha=0.85)
            ratio_ax.grid(True, which="minor", color="#E5E9EF", lw=0.45, alpha=0.75)
    finally:
        del result

    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_suffix(".txt")
    _write_ratio_csv(csv_path, rows)
    summary_lines = [
        f"hdf5_path: {hdf5_path}",
        "model: canonical Pop II, dust-attenuated UVLF",
        f"min_raw_counts: {args.min_raw_counts}",
    ]
    for z_value in args.z_values:
        summary_lines.append(_summarize_offsets(rows, float(z_value)))
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    fig.savefig(pdf_path, dpi=500)
    fig.savefig(png_path, dpi=500)
    plt.close(fig)

    print(f"wrote_pdf={pdf_path}")
    print(f"wrote_png={png_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_summary={summary_path}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
