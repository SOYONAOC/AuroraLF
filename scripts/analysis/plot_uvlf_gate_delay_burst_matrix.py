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


OBS_UVLF_DIR = PROJECT_ROOT / "external_data" / "observations" / "uvlf"
OBS_FILES = {
    12.5: (
        "redshift_12p5/bouwens.npz",
        "redshift_12p5/donnan24.npz",
        "redshift_12p5/harikane23_uvlf_z12.npz",
    ),
}
MODE = "z10_mild_topheavy"
OBS_MUV_MIN = -22.5
OBS_MUV_MAX = -16.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gate-mode UVLFs across delay and burst settings.")
    parser.add_argument("--no-delay-no-burst-npz", required=True)
    parser.add_argument("--no-delay-burst-npz", required=True)
    parser.add_argument("--delay-no-burst-npz", required=True)
    parser.add_argument("--delay-burst-npz", required=True)
    parser.add_argument("--z", type=float, default=12.5)
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _resolve_path(path_text: str, *, must_exist: bool) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def _z_tag(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


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


def _load_case(path: Path, *, tag: str, mode: str, reference_centers: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        center_key = f"{tag}_bin_centers"
        phi_key = f"{tag}_{mode}_phi"
        mode_names = [str(item) for item in np.asarray(payload["mode_names"])]
        if mode not in mode_names:
            raise ValueError(f"mode {mode} is not present in {path}")
        if center_key not in payload.files or phi_key not in payload.files:
            raise KeyError(f"missing {center_key} or {phi_key} in {path}")
        centers = np.asarray(payload[center_key], dtype=float)
        phi = np.asarray(payload[phi_key], dtype=float)
    if reference_centers is not None and not np.allclose(centers, reference_centers, rtol=0.0, atol=0.0):
        raise ValueError(f"UVLF bin centers differ for {path}")
    return centers, phi


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)


def main() -> None:
    args = _parse_args()
    paths = {
        "no_delay_no_burst": _resolve_path(args.no_delay_no_burst_npz, must_exist=True),
        "no_delay_burst": _resolve_path(args.no_delay_burst_npz, must_exist=True),
        "delay_no_burst": _resolve_path(args.delay_no_burst_npz, must_exist=True),
        "delay_burst": _resolve_path(args.delay_burst_npz, must_exist=True),
    }
    output_prefix = _resolve_path(args.output_prefix, must_exist=False).with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    tag = _z_tag(float(args.z))
    centers: np.ndarray | None = None
    phi_by_case: dict[str, np.ndarray] = {}
    for case_name, path in paths.items():
        case_centers, phi = _load_case(path, tag=tag, mode=str(args.mode), reference_centers=centers)
        if centers is None:
            centers = case_centers
        phi_by_case[case_name] = phi
    if centers is None:
        raise RuntimeError("no UVLF cases were loaded")

    cases = (
        ("no_delay_no_burst", "no delay, no burst", "#a23b3b", "--", 1.75),
        ("no_delay_burst", "no delay, +burst", "#c94c4c", "-", 2.35),
        ("delay_no_burst", "delay, no burst", "#2f6f9f", "--", 1.75),
        ("delay_burst", "delay, +burst", "#1f77b4", "-", 2.35),
    )
    baseline = phi_by_case["no_delay_no_burst"]
    obs_mask = (centers >= OBS_MUV_MIN) & (centers <= OBS_MUV_MAX)

    summary_lines = [
        f"z: {float(args.z):g}",
        f"mode: {args.mode}",
        *(f"{name}_npz: {path}" for name, path in paths.items()),
        "",
    ]

    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 6.2,
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

    for case_name, label, color, linestyle, linewidth in cases:
        phi = phi_by_case[case_name]
        valid = np.isfinite(phi) & (phi > 0.0)
        ylim_values.append(phi[valid])
        ax_top.plot(centers[valid], phi[valid], color=color, ls=linestyle, lw=linewidth, label=label)

        case_ratio = _ratio(phi, baseline)
        valid_ratio = np.isfinite(case_ratio) & (case_ratio > 0.0) & obs_mask
        ratio_values.append(case_ratio[valid_ratio])
        if case_name != "no_delay_no_burst":
            ax_ratio.plot(centers[valid_ratio], case_ratio[valid_ratio], color=color, ls=linestyle, lw=2.0, label=label)

        summary_lines.append(f"{case_name}_phi_median={float(np.nanmedian(phi[valid])):.6e}")
        summary_lines.append(f"{case_name}_over_no_delay_no_burst_obs_median={float(np.nanmedian(case_ratio[valid_ratio])):.6f}")
        summary_lines.append(f"{case_name}_over_no_delay_no_burst_obs_min={float(np.nanmin(case_ratio[valid_ratio])):.6f}")
        summary_lines.append(f"{case_name}_over_no_delay_no_burst_obs_max={float(np.nanmax(case_ratio[valid_ratio])):.6f}")
    summary_lines.append("")

    pair_ratios = {
        "no_delay_burst_over_no_delay_no_burst": _ratio(
            phi_by_case["no_delay_burst"], phi_by_case["no_delay_no_burst"]
        ),
        "delay_no_burst_over_no_delay_no_burst": _ratio(
            phi_by_case["delay_no_burst"], phi_by_case["no_delay_no_burst"]
        ),
        "delay_burst_over_delay_no_burst": _ratio(
            phi_by_case["delay_burst"], phi_by_case["delay_no_burst"]
        ),
        "delay_burst_over_no_delay_burst": _ratio(
            phi_by_case["delay_burst"], phi_by_case["no_delay_burst"]
        ),
    }
    for name, values in pair_ratios.items():
        valid = np.isfinite(values) & (values > 0.0) & obs_mask
        summary_lines.append(f"{name}_obs_median={float(np.nanmedian(values[valid])):.6f}")
        summary_lines.append(f"{name}_obs_min={float(np.nanmin(values[valid])):.6f}")
        summary_lines.append(f"{name}_obs_max={float(np.nanmax(values[valid])):.6f}")

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
            ms=5.1,
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
    ax_top.set_title(rf"$z={float(args.z):g}$ gate UVLF: delay and SFR burst", pad=3)
    ax_top.grid(alpha=0.22)
    ax_top.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.8)

    positive_ratio = np.concatenate([_finite_positive(item) for item in ratio_values])
    ax_ratio.axhline(1.0, color="0.35", ls="--", lw=1.0)
    ax_ratio.axvspan(-24.5, OBS_MUV_MIN, color="0.94", zorder=-10)
    ax_ratio.axvspan(OBS_MUV_MAX, -15.0, color="0.94", zorder=-10)
    ax_ratio.set_ylim(max(0.4, float(np.min(positive_ratio)) * 0.82), float(np.max(positive_ratio)) * 1.18)
    ax_ratio.set_xlabel(r"$M_{\rm UV}$")
    ax_ratio.set_ylabel("relative to\nno delay/no burst")
    ax_ratio.grid(alpha=0.22)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.parent / f"{output_prefix.name}_summary.txt"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")


if __name__ == "__main__":
    main()
