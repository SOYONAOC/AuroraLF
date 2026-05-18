#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("apj")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf import intrinsic_muv_from_observed


REDSHIFTS = [6.0, 8.0, 10.0, 12.5]
DEFAULT_CACHE_DIR = "../../temp_data"
DEFAULT_OUTPUT_DIR = "assets"
DEFAULT_PREVIEW_DIR = "../../outputs"
DEFAULT_MUV_MIN = -25.0
DEFAULT_MUV_MAX = -15.0
DEFAULT_MUV_BIN_WIDTH = 0.5
DEFAULT_LOGMH_MIN = 9.0
DEFAULT_LOGMH_MAX = 13.0
DEFAULT_LOGMH_BIN_WIDTH = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate separate UVLF halo-mass composition slide assets from cached current-parameter samples."
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dir", type=str, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--redshifts", type=float, nargs="+", default=REDSHIFTS)
    parser.add_argument("--muv-min", type=float, default=DEFAULT_MUV_MIN)
    parser.add_argument("--muv-max", type=float, default=DEFAULT_MUV_MAX)
    parser.add_argument("--muv-bin-width", type=float, default=DEFAULT_MUV_BIN_WIDTH)
    parser.add_argument("--logmh-min", type=float, default=DEFAULT_LOGMH_MIN)
    parser.add_argument("--logmh-max", type=float, default=DEFAULT_LOGMH_MAX)
    parser.add_argument("--logmh-bin-width", type=float, default=DEFAULT_LOGMH_BIN_WIDTH)
    return parser.parse_args()


def _z_tag(z_value: float) -> str:
    return str(float(z_value)).replace(".", "p").rstrip("0").rstrip("p")


def _load_cached_samples(cache_dir: Path, z_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(cache_dir / f"uvlf_z{_z_tag(z_value)}.npz")
    return (
        np.asarray(payload["sample_muv"], dtype=float),
        np.asarray(payload["sample_logMh"], dtype=float),
        np.asarray(payload["sample_weight"], dtype=float),
    )


def _observed_muv_from_intrinsic(
    intrinsic_muv: np.ndarray,
    z_value: float,
) -> np.ndarray:
    intrinsic = np.asarray(intrinsic_muv, dtype=float)
    valid = np.isfinite(intrinsic)
    result = np.full_like(intrinsic, np.nan, dtype=float)
    if not np.any(valid):
        return result

    intrinsic_valid = intrinsic[valid]
    obs_grid = np.linspace(
        float(np.nanmin(intrinsic_valid)) - 2.0,
        float(np.nanmax(intrinsic_valid)) + 8.0,
        6000,
        dtype=float,
    )
    intrinsic_from_obs = np.asarray(intrinsic_muv_from_observed(obs_grid, z_value), dtype=float)
    order = np.argsort(intrinsic_from_obs)
    intrinsic_sorted = intrinsic_from_obs[order]
    obs_sorted = obs_grid[order]
    unique_intrinsic, unique_index = np.unique(intrinsic_sorted, return_index=True)
    unique_obs = obs_sorted[unique_index]
    result[valid] = np.interp(
        intrinsic_valid,
        unique_intrinsic,
        unique_obs,
        left=unique_obs[0],
        right=unique_obs[-1],
    )
    return result


def _load_observational_uvlf(project_root: Path, z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = project_root / "data" / f"redshift_{_z_tag(z_value)}"
    datasets: list[dict[str, np.ndarray | str]] = []
    if not obs_dir.is_dir():
        return datasets

    for file_path in sorted(obs_dir.glob("*.npz")):
        payload = np.load(file_path, allow_pickle=True)
        datasets.append(
            {
                "label": str(payload["label"][0]),
                "muv": np.asarray(payload["muverr"], dtype=float),
                "phi": np.asarray(payload["phierr"], dtype=float),
                "phi_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                "phi_up": np.asarray(payload["phi_err_up"], dtype=float),
                "mag_err": np.asarray(payload["mag_err"], dtype=float),
            }
        )
    return datasets


def _weighted_phi_per_mag(
    sample_muv: np.ndarray,
    sample_weight: np.ndarray,
    muv_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(sample_muv, bins=muv_edges, weights=sample_weight)
    widths = np.diff(muv_edges)
    phi = np.zeros_like(counts, dtype=float)
    positive = widths > 0.0
    phi[positive] = counts[positive] / widths[positive]
    centers = 0.5 * (muv_edges[:-1] + muv_edges[1:])
    return centers, phi


def _mass_composition_fraction(
    sample_muv: np.ndarray,
    sample_logmh: np.ndarray,
    sample_weight: np.ndarray,
    muv_edges: np.ndarray,
    logmh_edges: np.ndarray,
) -> np.ndarray:
    n_mass_bins = logmh_edges.size - 1
    n_muv_bins = muv_edges.size - 1
    weighted = np.zeros((n_mass_bins, n_muv_bins), dtype=float)

    for index in range(n_mass_bins):
        left = logmh_edges[index]
        right = logmh_edges[index + 1]
        if index == n_mass_bins - 1:
            mass_mask = (sample_logmh >= left) & (sample_logmh <= right)
        else:
            mass_mask = (sample_logmh >= left) & (sample_logmh < right)
        if not np.any(mass_mask):
            continue
        weighted[index], _ = np.histogram(
            sample_muv[mass_mask],
            bins=muv_edges,
            weights=sample_weight[mass_mask],
        )

    total = np.sum(weighted, axis=0, keepdims=True)
    fractions = np.zeros_like(weighted)
    valid = total[0] > 0.0
    fractions[:, valid] = weighted[:, valid] / total[:, valid]
    return fractions


def _mass_bin_labels(logmh_edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for left, right in zip(logmh_edges[:-1], logmh_edges[1:], strict=True):
        labels.append(rf"$10^{{{left:g}}}$-$10^{{{right:g}}}\,M_\odot$")
    return labels


def _plot_single_redshift(
    *,
    z_value: float,
    project_root: Path,
    cache_dir: Path,
    output_dir: Path,
    preview_dir: Path,
    muv_edges: np.ndarray,
    logmh_edges: np.ndarray,
) -> None:
    sample_muv_intrinsic, sample_logmh, sample_weight = _load_cached_samples(cache_dir=cache_dir, z_value=z_value)
    sample_muv_obs = _observed_muv_from_intrinsic(sample_muv_intrinsic, z_value)
    centers, phi = _weighted_phi_per_mag(sample_muv=sample_muv_obs, sample_weight=sample_weight, muv_edges=muv_edges)
    fractions = _mass_composition_fraction(
        sample_muv=sample_muv_obs,
        sample_logmh=sample_logmh,
        sample_weight=sample_weight,
        muv_edges=muv_edges,
        logmh_edges=logmh_edges,
    )
    obs_sets = _load_observational_uvlf(project_root=project_root, z_value=z_value)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11.9, 8.7),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.11, hspace=0.06)

    valid_phi = phi > 0.0
    ax_top.step(centers[valid_phi], phi[valid_phi], where="mid", color="black", lw=2.2, label="Current model")

    obs_markers = ["o", "s", "^", "D", "P", "X"]
    obs_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
    for obs_index, obs in enumerate(obs_sets):
        marker = obs_markers[obs_index % len(obs_markers)]
        color = obs_colors[obs_index % len(obs_colors)]
        valid = np.isfinite(obs["muv"]) & np.isfinite(obs["phi"]) & (np.asarray(obs["phi"]) > 0.0)
        if not np.any(valid):
            continue
        ax_top.errorbar(
            np.asarray(obs["muv"])[valid],
            np.asarray(obs["phi"])[valid],
            xerr=np.asarray(obs["mag_err"])[valid],
            yerr=np.vstack([np.asarray(obs["phi_lo"])[valid], np.asarray(obs["phi_up"])[valid]]),
            fmt=marker,
            ms=5.5,
            color=color,
            mec="white",
            mew=0.6,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.95,
            label=str(obs["label"]),
        )

    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$\phi(M_{\rm UV}^{\rm obs})$")
    ax_top.set_ylim(1.0e-6, 1.0e-1)
    ax_top.grid(alpha=0.2)
    ax_top.legend(frameon=False, fontsize=9.2, ncol=2, loc="lower right")
    ax_top.text(
        0.035,
        0.93,
        rf"$z={z_value:g}$",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=26,
    )

    bar_left = muv_edges[:-1]
    widths = np.diff(muv_edges)
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, logmh_edges.size - 1))
    labels = _mass_bin_labels(logmh_edges)
    cumulative = np.zeros_like(bar_left, dtype=float)
    for fraction_row, color, label in zip(fractions, colors, labels, strict=True):
        ax_bottom.bar(
            bar_left,
            fraction_row,
            width=widths,
            bottom=cumulative,
            align="edge",
            color=color,
            edgecolor="white",
            linewidth=1.5,
            label=label,
        )
        cumulative += fraction_row

    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.set_xlim(muv_edges[0], muv_edges[-1])
    ax_bottom.set_xlabel(r"$M_{\rm UV}^{\rm obs}$")
    ax_bottom.set_ylabel(r"Fraction in each $M_{\rm UV}^{\rm obs}$ bin")
    ax_bottom.grid(alpha=0.2, axis="y")

    output_path = output_dir / f"uvlf_full_mass_composition_z{_z_tag(z_value)}_m25_m15.pdf"
    preview_path = preview_dir / f"uvlf_full_mass_composition_z{_z_tag(z_value)}_m25_m15.png"
    fig.savefig(output_path, dpi=500)
    fig.savefig(preview_path, dpi=500)
    plt.close(fig)
    print(f"saved_pdf={output_path}")
    print(f"saved_png={preview_path}")


def main() -> None:
    args = _parse_args()
    slides_dir = Path(__file__).resolve().parent
    project_root = slides_dir.parents[1]
    cache_dir = (slides_dir / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    output_dir = (slides_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    preview_dir = (slides_dir / args.preview_dir).resolve() if not Path(args.preview_dir).is_absolute() else Path(args.preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    redshifts = list(args.redshifts)

    missing = [f"uvlf_z{_z_tag(z)}.npz" for z in redshifts if not (cache_dir / f"uvlf_z{_z_tag(z)}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            "missing cached UVLF sample files in cache_dir: " + ", ".join(missing)
        )

    muv_edges = np.arange(args.muv_min, args.muv_max + args.muv_bin_width, args.muv_bin_width, dtype=float)
    logmh_edges = np.arange(args.logmh_min, args.logmh_max + args.logmh_bin_width, args.logmh_bin_width, dtype=float)

    for z_value in redshifts:
        _plot_single_redshift(
            z_value=z_value,
            project_root=project_root,
            cache_dir=cache_dir,
            output_dir=output_dir,
            preview_dir=preview_dir,
            muv_edges=muv_edges,
            logmh_edges=logmh_edges,
        )


if __name__ == "__main__":
    main()
