#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from auroralf.constants import (
    PLANCK15_H0_GYR,
    PLANCK15_OMEGA_B,
    PLANCK15_OMEGA_LAMBDA,
    PLANCK15_OMEGA_M,
)
from auroralf.mah.generator import generate_halo_histories
from auroralf.mah import Cosmology
from auroralf.mah.tng import generate_tng_halo_histories


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data_save" / "tng_mah_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_mah_compare"
M_FINAL = 1.0e10
N_TRACKS = 2000
MASS_BIN_WIDTH_DEX = 0.15
MIN_CANDIDATES = 5


@dataclass(frozen=True)
class CacheSpec:
    label: str
    z_final: float
    path: Path


CACHE_SPECS = (
    CacheSpec(
        label="z6",
        z_final=6.01075739884,
        path=CACHE_DIR / "TNG100-1-Dark_sublink_mpb_z6p011_n2092.hdf5",
    ),
    CacheSpec(
        label="z8",
        z_final=8.01217294887,
        path=CACHE_DIR / "TNG100-1-Dark_sublink_mpb_z8p012_n1671.hdf5",
    ),
    CacheSpec(
        label="z10",
        z_final=9.99659046619,
        path=CACHE_DIR / "TNG100-1-Dark_sublink_mpb_z9p997_n1286.hdf5",
    ),
    CacheSpec(
        label="z12",
        z_final=11.9802133153,
        path=CACHE_DIR / "TNG100-1-Dark_sublink_mpb_z11p980_n908.hdf5",
    ),
)


def _set_plot_style() -> None:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def _require_inputs() -> None:
    missing = [path for path in (spec.path for spec in CACHE_SPECS) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing TNG MAH cache file(s): {names}")


def _tracks_to_grid(result, n_tracks: int) -> dict[str, np.ndarray]:
    n_steps = int(result.metadata["grid_size"])
    grids: dict[str, np.ndarray] = {}
    for name in ("z", "t_gyr", "Mh", "dMh_dt_raw", "active_flag"):
        values = np.asarray(result.tracks[name])
        if values.size != n_tracks * n_steps:
            raise ValueError(f"track column {name!r} cannot be reshaped to ({n_tracks}, {n_steps})")
        grids[name] = values.reshape(n_tracks, n_steps)
    grids["active_flag"] = grids["active_flag"].astype(bool)
    return grids


def _percentiles_by_step(
    values: np.ndarray,
    valid_mask: np.ndarray,
    percentiles: tuple[float, ...] = (16.0, 50.0, 84.0),
    min_count: int = MIN_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != valid_mask.shape:
        raise ValueError("values and valid_mask must have matching shapes")
    output = np.full((len(percentiles), values.shape[1]), np.nan, dtype=float)
    counts = np.count_nonzero(valid_mask & np.isfinite(values), axis=0)
    for step in range(values.shape[1]):
        if counts[step] < int(min_count):
            continue
        column = values[:, step]
        valid = valid_mask[:, step] & np.isfinite(column)
        output[:, step] = np.nanpercentile(column[valid], percentiles)
    return output, counts


def _positive_peak_smar(dmhdt: np.ndarray, active: np.ndarray) -> np.ndarray:
    smar = np.where(active & np.isfinite(dmhdt) & (dmhdt > 0.0), dmhdt / M_FINAL, np.nan)
    has_positive = np.any(np.isfinite(smar), axis=1)
    if not np.any(has_positive):
        raise ValueError("no positive accretion rates are available for peak sMAR")
    peak = np.full(smar.shape[0], np.nan, dtype=float)
    peak[has_positive] = np.nanmax(smar[has_positive], axis=1)
    return peak[np.isfinite(peak)]


def _build_samples(
    spec: CacheSpec,
    seed_offset: int,
    *,
    cosmology: Cosmology,
) -> dict[str, object]:
    tng = generate_tng_halo_histories(
        n_tracks=N_TRACKS,
        z_final=spec.z_final,
        Mh_final=M_FINAL,
        cosmology=cosmology,
        cache_path=spec.path,
        mass_bin_width_dex=MASS_BIN_WIDTH_DEX,
        min_candidates=MIN_CANDIDATES,
        random_seed=1000 + seed_offset,
    )
    tng_grid = _tracks_to_grid(tng, N_TRACKS)
    z_grid = np.asarray(tng_grid["z"][0], dtype=float)
    mc_z_final = float(z_grid[-1])

    mcbride = generate_halo_histories(
        n_tracks=N_TRACKS,
        z_final=mc_z_final,
        Mh_final=M_FINAL,
        cosmology=cosmology,
        z_start_max=float(z_grid[0]),
        M_min=0.0,
        random_seed=2000 + seed_offset,
        time_grid_mode="custom",
        custom_grid=z_grid,
        store_inactive_history=True,
        sampler="mcbride",
    )
    mcbride_grid = _tracks_to_grid(mcbride, N_TRACKS)
    np.testing.assert_allclose(mcbride_grid["z"][0], z_grid)

    return {
        "spec": spec,
        "z_grid": z_grid,
        "tng": tng,
        "tng_grid": tng_grid,
        "mcbride": mcbride,
        "mcbride_grid": mcbride_grid,
    }


def _plot_tracks(samples: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    _set_plot_style()
    colors = {"tng": "#D55E00", "mcbride": "#0072B2"}
    fig, axes = plt.subplots(
        nrows=len(samples),
        ncols=2,
        figsize=(9.2, 10.2),
        sharex=False,
        constrained_layout=True,
    )

    for row, sample in enumerate(samples):
        spec = sample["spec"]
        z_grid = np.asarray(sample["z_grid"], dtype=float)
        tng_result = sample["tng"]
        tng_grid = sample["tng_grid"]
        mc_grid = sample["mcbride_grid"]

        tng_active = np.asarray(tng_grid["active_flag"], dtype=bool)
        mc_active = np.asarray(mc_grid["active_flag"], dtype=bool)
        tng_mass_ratio = np.asarray(tng_grid["Mh"], dtype=float) / M_FINAL
        mc_mass_ratio = np.asarray(mc_grid["Mh"], dtype=float) / M_FINAL
        tng_smar = np.asarray(tng_grid["dMh_dt_raw"], dtype=float) / M_FINAL
        mc_smar = np.asarray(mc_grid["dMh_dt_raw"], dtype=float) / M_FINAL

        mass_tng, count_tng = _percentiles_by_step(tng_mass_ratio, tng_active)
        mass_mc, _ = _percentiles_by_step(mc_mass_ratio, mc_active)
        smar_tng, _ = _percentiles_by_step(tng_smar, tng_active & (tng_smar > 0.0))
        smar_mc, _ = _percentiles_by_step(mc_smar, mc_active & (mc_smar > 0.0))

        ax_mass = axes[row, 0]
        ax_smar = axes[row, 1]

        ax_mass.fill_between(z_grid, mass_tng[0], mass_tng[2], color=colors["tng"], alpha=0.20, lw=0)
        ax_mass.plot(z_grid, mass_tng[1], color=colors["tng"], label="TNG MPB")
        ax_mass.fill_between(z_grid, mass_mc[0], mass_mc[2], color=colors["mcbride"], alpha=0.18, lw=0)
        ax_mass.plot(z_grid, mass_mc[1], color=colors["mcbride"], ls="--", label="McBride09")
        ax_mass.set_yscale("log")
        ax_mass.set_ylim(8.0e-5, 1.4)
        ax_mass.set_xlim(float(z_grid[0]), float(z_grid[-1]))
        ax_mass.set_ylabel(r"$M_h/M_{h,\rm final}$")
        ax_mass.text(
            0.03,
            0.08,
            (
                rf"$z_f={spec.z_final:.2f}$; "
                rf"TNG candidates={int(tng_result.metadata['candidate_count'])}; "
                rf"unresolved={100.0 * float(tng_result.metadata['unresolved_step_fraction']):.0f}\%"
            ),
            transform=ax_mass.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )

        ax_smar.fill_between(z_grid, smar_tng[0], smar_tng[2], color=colors["tng"], alpha=0.20, lw=0)
        ax_smar.plot(z_grid, smar_tng[1], color=colors["tng"], label="TNG MPB")
        ax_smar.fill_between(z_grid, smar_mc[0], smar_mc[2], color=colors["mcbride"], alpha=0.18, lw=0)
        ax_smar.plot(z_grid, smar_mc[1], color=colors["mcbride"], ls="--", label="McBride09")
        ax_smar.set_yscale("log")
        ax_smar.set_ylim(3.0e-2, 3.0e2)
        ax_smar.set_xlim(float(z_grid[0]), float(z_grid[-1]))
        ax_smar.set_ylabel(r"$\dot M_h/M_{h,\rm final}\ [{\rm Gyr}^{-1}]$")

        if row == 0:
            ax_mass.set_title("Mass assembly")
            ax_smar.set_title("Positive accretion")
            ax_mass.legend(loc="upper left", fontsize=8, frameon=False)
            ax_smar.legend(loc="upper left", fontsize=8, frameon=False)
        if row == len(samples) - 1:
            ax_mass.set_xlabel(r"redshift $z$")
            ax_smar.set_xlabel(r"redshift $z$")

    png_path = output_dir / "tng_vs_mcbride_mah_logM10.png"
    pdf_path = output_dir / "tng_vs_mcbride_mah_logM10.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path, pdf_path


def _plot_peak_smar(samples: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    _set_plot_style()
    colors = {"tng": "#D55E00", "mcbride": "#0072B2"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    positions = np.arange(len(samples), dtype=float)
    width = 0.32

    for offset, backend, display, color in (
        (-width / 2, "tng", "TNG", colors["tng"]),
        (width / 2, "mcbride", "McBride09", colors["mcbride"]),
    ):
        values = []
        for sample in samples:
            grid = sample[f"{backend}_grid"] if backend == "tng" else sample["mcbride_grid"]
            values.append(
                _positive_peak_smar(
                    np.asarray(grid["dMh_dt_raw"], dtype=float),
                    np.asarray(grid["active_flag"], dtype=bool),
                )
            )
        parts = ax.violinplot(
            values,
            positions=positions + offset,
            widths=width * 0.85,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(1.6)

        p84 = [np.nanpercentile(value, 84.0) for value in values]
        p95 = [np.nanpercentile(value, 95.0) for value in values]
        ax.scatter(positions + offset, p84, color=color, s=18, marker="o", label=f"{display} p84")
        ax.scatter(positions + offset, p95, color=color, s=24, marker="^", label=f"{display} p95")

    labels = [rf"${sample['spec'].z_final:.2f}$" for sample in samples]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"final redshift $z_f$")
    ax.set_ylabel(r"track peak $\dot M_h/M_{h,\rm final}\ [{\rm Gyr}^{-1}]$")
    ax.set_yscale("log")
    ax.set_ylim(3.0e-1, 1.0e3)
    ax.legend(ncols=2, fontsize=8, frameon=False, loc="upper right")

    png_path = output_dir / "tng_vs_mcbride_peak_smar_logM10.png"
    pdf_path = output_dir / "tng_vs_mcbride_peak_smar_logM10.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path, pdf_path


def _write_summary(samples: list[dict[str, object]], output_dir: Path) -> Path:
    csv_path = output_dir / "tng_vs_mcbride_logM10_summary.csv"
    fieldnames = [
        "backend",
        "z_final",
        "n_tracks",
        "n_steps",
        "candidate_count",
        "unresolved_step_fraction",
        "negative_dmhdt_clip_fraction",
        "peak_smar_valid_tracks",
        "peak_smar_p50_gyr_inv",
        "peak_smar_p84_gyr_inv",
        "peak_smar_p95_gyr_inv",
        "peak_smar_p99_gyr_inv",
    ]
    rows: list[dict[str, object]] = []
    for sample in samples:
        for backend, grid_name, result_name in (
            ("tng", "tng_grid", "tng"),
            ("mcbride", "mcbride_grid", "mcbride"),
        ):
            grid = sample[grid_name]
            result = sample[result_name]
            peak = _positive_peak_smar(
                np.asarray(grid["dMh_dt_raw"], dtype=float),
                np.asarray(grid["active_flag"], dtype=bool),
            )
            metadata = result.metadata
            rows.append(
                {
                    "backend": backend,
                    "z_final": f"{float(sample['spec'].z_final):.12g}",
                    "n_tracks": int(metadata["n_tracks"]),
                    "n_steps": int(metadata["grid_size"]),
                    "candidate_count": metadata.get("candidate_count", ""),
                    "unresolved_step_fraction": metadata.get("unresolved_step_fraction", ""),
                    "negative_dmhdt_clip_fraction": metadata.get("negative_dmhdt_clip_fraction", ""),
                    "peak_smar_valid_tracks": int(peak.size),
                    "peak_smar_p50_gyr_inv": f"{np.nanpercentile(peak, 50.0):.8g}",
                    "peak_smar_p84_gyr_inv": f"{np.nanpercentile(peak, 84.0):.8g}",
                    "peak_smar_p95_gyr_inv": f"{np.nanpercentile(peak, 95.0):.8g}",
                    "peak_smar_p99_gyr_inv": f"{np.nanpercentile(peak, 99.0):.8g}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    cosmology = Cosmology(
        h0=PLANCK15_H0_GYR,
        omega_m=PLANCK15_OMEGA_M,
        omega_b=PLANCK15_OMEGA_B,
        omega_lambda=PLANCK15_OMEGA_LAMBDA,
    )
    _require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = [
        _build_samples(spec, index, cosmology=cosmology)
        for index, spec in enumerate(CACHE_SPECS)
    ]
    track_png, track_pdf = _plot_tracks(samples, OUTPUT_DIR)
    peak_png, peak_pdf = _plot_peak_smar(samples, OUTPUT_DIR)
    csv_path = _write_summary(samples, OUTPUT_DIR)
    print(f"saved={track_png}")
    print(f"saved={track_pdf}")
    print(f"saved={peak_png}")
    print(f"saved={peak_pdf}")
    print(f"saved={csv_path}")


if __name__ == "__main__":
    main()
