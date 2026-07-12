#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.mah import Cosmology
from auroralf.uvlf import sample_uvlf_from_hmf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_PATH = (
    PROJECT_ROOT
    / "slides"
    / "group_meeting_popiii_20260622"
    / "assets"
    / "uvlf_z14p5_popii_only_slide.pdf"
)
DEFAULT_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popii_only_slide.csv"
DEFAULT_OBSERVATION_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_observations_slide.csv"
DEFAULT_OBSERVATION_PATHS = (
    PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_14" / "whitler25_jades_z14p3.npz",
    PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_15" / "donnan24_primer_z14p5.npz",
)
OBSERVATION_LABELS = {
    "JADES": r"JADES ($z_{\rm med}=14.3$)",
    "PRIMER": r"PRIMER ($13.5<z<15.5$)",
}
OBSERVATION_STYLES = {
    "JADES": {"marker": "D", "color": "#7A4EAB"},
    "PRIMER": {"marker": "o", "color": "#2A9D8F"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Pop II-only z=14.5 UVLF slide figure for the 2026-06-22 group meeting."
    )
    parser.add_argument("--z", type=float, default=14.5)
    parser.add_argument("--N-mass", type=int, default=720)
    parser.add_argument("--n-tracks", type=int, default=24)
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--logM-max", type=float, default=13.0)
    parser.add_argument("--muv-min", type=float, default=-25.5)
    parser.add_argument("--muv-max", type=float, default=1.5)
    parser.add_argument("--muv-bin-width", type=float, default=0.5)
    parser.add_argument("--smooth-sigma-mag", type=float, default=0.60)
    parser.add_argument("--plot-min-raw-counts", type=int, default=10)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--observation-table-path", type=Path, default=DEFAULT_OBSERVATION_TABLE_PATH)
    parser.add_argument("--observation-paths", nargs="+", type=Path, default=list(DEFAULT_OBSERVATION_PATHS))
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def _load_observation_table(path: Path) -> dict[str, np.ndarray | str]:
    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    required = ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up", "is_upper_limit", "label", "source")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise KeyError(f"Observation file {path} is missing required fields: {missing}")

    label = str(np.asarray(payload["label"])[0])
    source = str(np.asarray(payload["source"])[0])
    z_note = str(np.asarray(payload["z_note"])[0]) if "z_note" in payload.files else ""
    table = {
        "label": label,
        "source": source,
        "z_note": z_note,
        "muv": np.asarray(payload["muverr"], dtype=float),
        "muv_err": np.asarray(payload["mag_err"], dtype=float),
        "phi": np.asarray(payload["phierr"], dtype=float),
        "phi_lo": np.asarray(payload["phi_err_lo"], dtype=float),
        "phi_up": np.asarray(payload["phi_err_up"], dtype=float),
        "upper_limit": np.asarray(payload["is_upper_limit"], dtype=bool),
    }
    sizes = {np.asarray(value).shape for key, value in table.items() if key not in {"label", "source", "z_note"}}
    if len(sizes) != 1:
        raise ValueError(f"Observation file {path} has inconsistent array shapes: {sizes}")
    finite = np.isfinite(table["muv"]) & np.isfinite(table["muv_err"]) & np.isfinite(table["phi"])
    finite &= np.isfinite(table["phi_lo"]) & np.isfinite(table["phi_up"])
    if not np.all(finite):
        raise ValueError(f"Observation file {path} contains non-finite values")
    if np.any(table["phi"] <= 0.0):
        raise ValueError(f"Observation file {path} contains non-positive phi values")
    return table


def _compute_plot_curve(
    centers: np.ndarray,
    phi: np.ndarray,
    raw_counts: np.ndarray,
    *,
    min_raw_counts: int,
    smooth_sigma_mag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plot_mask = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0)
    plot_mask &= np.asarray(raw_counts, dtype=int) >= int(min_raw_counts)
    if np.count_nonzero(plot_mask) < 3:
        raise RuntimeError(
            "Fewer than three positive UVLF bins pass --plot-min-raw-counts; "
            "lower the threshold or increase the sampling."
        )

    x_plot = np.asarray(centers[plot_mask], dtype=float)
    y_raw = np.asarray(phi[plot_mask], dtype=float)
    order = np.argsort(x_plot)
    x_plot = x_plot[order]
    y_raw = y_raw[order]

    if smooth_sigma_mag == 0.0:
        return x_plot, y_raw, plot_mask

    distances = x_plot[:, np.newaxis] - x_plot[np.newaxis, :]
    kernel = np.exp(-0.5 * (distances / smooth_sigma_mag) ** 2)
    kernel_sum = np.sum(kernel, axis=1)
    if np.any(kernel_sum <= 0.0) or not np.all(np.isfinite(kernel_sum)):
        raise RuntimeError("Invalid smoothing kernel normalization")
    log_phi_smooth = kernel @ np.log10(y_raw) / kernel_sum
    y_smooth = np.power(10.0, log_phi_smooth)
    if not np.all(np.isfinite(y_smooth)) or np.any(y_smooth <= 0.0):
        raise RuntimeError("Smoothed Pop II-only UVLF contains invalid values")
    return x_plot, y_smooth, plot_mask


def _write_observation_csv(path: Path, observations: list[dict[str, np.ndarray | str]]) -> None:
    rows: list[tuple[str, str, str, float, float, float, float, float, bool]] = []
    for obs in observations:
        label = str(obs["label"])
        source = str(obs["source"])
        z_note = str(obs["z_note"])
        for index in range(np.asarray(obs["muv"]).size):
            rows.append(
                (
                    label,
                    source,
                    z_note,
                    float(np.asarray(obs["muv"])[index]),
                    float(np.asarray(obs["muv_err"])[index]),
                    float(np.asarray(obs["phi"])[index]),
                    float(np.asarray(obs["phi_lo"])[index]),
                    float(np.asarray(obs["phi_up"])[index]),
                    bool(np.asarray(obs["upper_limit"])[index]),
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "source",
                "z_note",
                "Muv",
                "Muv_err",
                "phi_Mpc^-3_mag^-1",
                "phi_err_lo",
                "phi_err_up",
                "is_upper_limit",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    f"{row[3]:.8e}",
                    f"{row[4]:.8e}",
                    f"{row[5]:.8e}",
                    f"{row[6]:.8e}",
                    f"{row[7]:.8e}",
                    int(row[8]),
                ]
            )


def main() -> None:
    args = _parse_args()
    cosmology = Cosmology()
    if args.N_mass <= 0:
        raise ValueError("--N-mass must be positive")
    if args.n_tracks <= 0:
        raise ValueError("--n-tracks must be positive")
    if args.n_grid <= 1:
        raise ValueError("--n-grid must be greater than 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.logM_max <= 0.0:
        raise ValueError("--logM-max must be positive")
    if args.muv_max <= args.muv_min:
        raise ValueError("--muv-max must be larger than --muv-min")
    if args.muv_bin_width <= 0.0:
        raise ValueError("--muv-bin-width must be positive")
    if args.smooth_sigma_mag < 0.0:
        raise ValueError("--smooth-sigma-mag must be non-negative")
    if args.plot_min_raw_counts < 1:
        raise ValueError("--plot-min-raw-counts must be at least 1")
    if len(args.observation_paths) == 0:
        raise ValueError("--observation-paths must contain at least one file")

    z_obs = float(args.z)
    atomic_mass_msun = float(
        compute_atomic_cooling_mass_msun(z_obs, cosmology=cosmology)
    )
    logm_min = float(np.log10(atomic_mass_msun))
    if args.logM_max <= logm_min:
        raise ValueError("--logM-max must be larger than log10(M_atom)")

    bin_edges = np.arange(args.muv_min, args.muv_max + args.muv_bin_width, args.muv_bin_width)
    if bin_edges.size < 2:
        raise RuntimeError("MUV bin construction produced fewer than two bin edges")

    result = sample_uvlf_from_hmf(
        z_obs=z_obs,
        cosmology=cosmology,
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        base_seed=int(args.random_seed),
        quantity="Muv",
        bins=bin_edges,
        logM_min=logm_min,
        logM_max=float(args.logM_max),
        z_start_max=50.0,
        n_grid=int(args.n_grid),
        sampler="mcbride",
        enable_time_delay=True,
        pipeline_workers=int(args.workers),
        enable_popiii=False,
        imf_mode="canonical",
    )

    centers = np.asarray(result.uvlf["bin_centers"], dtype=float)
    phi = np.asarray(result.uvlf["phi"], dtype=float)
    phi_sigma = np.asarray(result.uvlf["phi_sigma"], dtype=float)
    raw_counts = np.asarray(result.uvlf["raw_counts"], dtype=int)
    valid = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0)
    if not np.any(valid):
        raise RuntimeError("Pop II-only UVLF has no positive bins")
    plot_centers, plot_phi, plot_mask = _compute_plot_curve(
        centers,
        phi,
        raw_counts,
        min_raw_counts=int(args.plot_min_raw_counts),
        smooth_sigma_mag=float(args.smooth_sigma_mag),
    )

    table_path = _resolve_path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    phi_plot = np.full_like(phi, np.nan, dtype=float)
    phi_plot[np.where(plot_mask)[0][np.argsort(centers[plot_mask])]] = plot_phi
    table = np.column_stack([centers, phi, phi_sigma, raw_counts, phi_plot])
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header=(
            "Muv_center,phi_Mpc^-3_mag^-1,phi_sigma_Mpc^-3_mag^-1,raw_counts,phi_plot_smoothed_Mpc^-3_mag^-1\n"
            f"z={z_obs},N_mass={args.N_mass},n_tracks={args.n_tracks},n_grid={args.n_grid},"
            f"logM_min=log10(M_atom)={logm_min:.8f},logM_max={args.logM_max},"
            "enable_popiii=False,imf_mode=canonical,enable_time_delay=True,"
            f"smooth_sigma_mag={args.smooth_sigma_mag},plot_min_raw_counts={args.plot_min_raw_counts}"
        ),
        comments="# ",
    )

    observation_paths = [_resolve_path(path) for path in args.observation_paths]
    observations = [_load_observation_table(path) for path in observation_paths]
    observation_table_path = _resolve_path(args.observation_table_path)
    _write_observation_csv(observation_table_path, observations)

    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        plot_centers,
        plot_phi,
        color="#1F5C8B",
        linewidth=2.8,
        label="Pop II only",
        zorder=4,
    )
    for obs in observations:
        label = str(obs["label"])
        style = OBSERVATION_STYLES.get(label)
        if style is None:
            raise KeyError(f"No plotting style configured for observation label: {label}")
        legend_label = OBSERVATION_LABELS.get(label, label)
        upper_limit = np.asarray(obs["upper_limit"], dtype=bool)
        detection = ~upper_limit
        if np.any(detection):
            ax.errorbar(
                np.asarray(obs["muv"])[detection],
                np.asarray(obs["phi"])[detection],
                xerr=np.asarray(obs["muv_err"])[detection],
                yerr=[
                    np.asarray(obs["phi_lo"])[detection],
                    np.asarray(obs["phi_up"])[detection],
                ],
                fmt=str(style["marker"]),
                ms=8.5,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.7,
                capsize=3.5,
                elinewidth=1.25,
                linestyle="none",
                label=legend_label,
                zorder=10,
            )
        if np.any(upper_limit):
            ax.errorbar(
                np.asarray(obs["muv"])[upper_limit],
                np.asarray(obs["phi"])[upper_limit],
                xerr=np.asarray(obs["muv_err"])[upper_limit],
                yerr=0.35 * np.asarray(obs["phi"])[upper_limit],
                uplims=True,
                fmt=str(style["marker"]),
                ms=8.5,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.7,
                capsize=3.5,
                elinewidth=1.25,
                linestyle="none",
                label=legend_label if not np.any(detection) else None,
                zorder=10,
            )

    ax.set_yscale("log")
    ax.set_xlim(args.muv_min, -4.5)
    y_positive = phi[valid]
    ax.set_ylim(max(np.min(y_positive) * 0.35, 1.0e-15), np.max(y_positive) * 3.0)
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\Phi\ [{\rm Mpc}^{-3}\ {\rm mag}^{-1}]$")
    ax.grid(True, which="major", color="#C8D2DF", linewidth=0.75, alpha=0.85)
    ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.45, alpha=0.70)
    ax.legend(loc="lower right", frameon=True, fontsize=16)
    ax.text(
        0.035,
        0.93,
        rf"$z={z_obs:.1f}$, canonical Pop II",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        color="#1F3A5F",
    )
    fig.tight_layout()

    figure_path = _resolve_path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=500)
    plt.close(fig)

    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {observation_table_path}")


if __name__ == "__main__":
    main()
