#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/auroralf_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.mah.models import KM_PER_MPC, SECONDS_PER_GYR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESAN_CACHE = (
    PROJECT_ROOT
    / "data_save/thesan_mah_cache/thesan-dark-1_LHaloTree_allchunks_z11p882_n371_smoke.hdf5"
)
TNG_CACHE = PROJECT_ROOT / "data_save/tng_mah_cache/TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/thesan_z12_mah_compare"
MASS_BINS = ((9.0, 9.25), (9.25, 9.5), (9.5, 10.0), (10.0, 10.5))
MCBRIDE_TRACKS_PER_BIN = 250


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing MAH cache: {path}")
    with h5py.File(path, "r") as handle:
        required = ("z_grid", "t_gyr_grid", "mass_ratio", "resolved_mask", "logM_final")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")
        data = {
            "path": path,
            "attrs": dict(handle.attrs),
            "z_grid": np.asarray(handle["z_grid"], dtype=float),
            "t_gyr_grid": np.asarray(handle["t_gyr_grid"], dtype=float),
            "mass_ratio": np.asarray(handle["mass_ratio"], dtype=float),
            "resolved_mask": np.asarray(handle["resolved_mask"], dtype=bool),
            "logM_final": np.asarray(handle["logM_final"], dtype=float),
        }
    if data["mass_ratio"].shape != data["resolved_mask"].shape:
        raise ValueError(f"{path} mass_ratio and resolved_mask shapes differ")
    if data["mass_ratio"].shape[1] != data["z_grid"].size:
        raise ValueError(f"{path} mass_ratio time dimension does not match z_grid")
    if data["logM_final"].size != data["mass_ratio"].shape[0]:
        raise ValueError(f"{path} logM_final does not match mass_ratio rows")
    if np.any(np.diff(data["z_grid"]) >= 0.0):
        raise ValueError(f"{path} z_grid must be strictly decreasing")
    if np.any(np.diff(data["t_gyr_grid"]) <= 0.0):
        raise ValueError(f"{path} t_gyr_grid must be strictly increasing")
    return data


def _tracks_to_grid(result: Any, n_tracks: int) -> dict[str, np.ndarray]:
    n_steps = int(result.metadata["grid_size"])
    grids: dict[str, np.ndarray] = {}
    for key in ("z", "t_gyr", "Mh", "active_flag"):
        values = np.asarray(result.tracks[key])
        if values.size != int(n_tracks) * n_steps:
            raise ValueError(f"track column {key!r} cannot be reshaped to ({n_tracks}, {n_steps})")
        grids[key] = values.reshape(int(n_tracks), n_steps)
    grids["active_flag"] = grids["active_flag"].astype(bool)
    return grids


def _lookback_myr(time_gyr: np.ndarray) -> np.ndarray:
    return (float(time_gyr[-1]) - np.asarray(time_gyr, dtype=float)) * 1.0e3


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _select_bin(cache: dict[str, Any], lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    logm = np.asarray(cache["logM_final"], dtype=float)
    resolved_count = np.count_nonzero(np.asarray(cache["resolved_mask"], dtype=bool), axis=1)
    indices = np.flatnonzero((logm >= float(lo)) & (logm < float(hi)) & (resolved_count >= 2))
    ratio = np.where(
        np.asarray(cache["resolved_mask"], dtype=bool)[indices],
        np.asarray(cache["mass_ratio"], dtype=float)[indices],
        np.nan,
    )
    return indices, ratio


def _interpolate_track_at_lookback(lookback: np.ndarray, ratio: np.ndarray, target_myr: float) -> np.ndarray:
    values = np.full(ratio.shape[0], np.nan, dtype=float)
    source_x = np.asarray(lookback, dtype=float)
    interp_x = source_x[::-1]
    for row in range(ratio.shape[0]):
        valid = np.isfinite(ratio[row]) & (ratio[row] > 0.0)
        if np.count_nonzero(valid) < 2:
            continue
        x = source_x[valid][::-1]
        y = np.log(ratio[row, valid][::-1])
        if float(target_myr) < float(np.min(x)) or float(target_myr) > float(np.max(x)):
            continue
        if np.any(np.diff(x) <= 0.0):
            order = np.argsort(x)
            x = x[order]
            y = y[order]
        values[row] = float(np.exp(np.interp(float(target_myr), x, y)))
    return values


def _build_mcbride_ratio(
    *,
    cosmology: Cosmology,
    z_grid: np.ndarray,
    logm_center: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_tracks = MCBRIDE_TRACKS_PER_BIN
    mh_final = 10.0 ** float(logm_center)
    result = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=float(z_grid[-1]),
        Mh_final=mh_final,
        cosmology=cosmology,
        z_start_max=float(z_grid[0]),
        M_min=0.0,
        random_seed=int(seed),
        time_grid_mode="custom",
        custom_grid=np.asarray(z_grid, dtype=float),
        store_inactive_history=True,
        sampler="mcbride",
    )
    grid = _tracks_to_grid(result, n_tracks)
    ratio = np.asarray(grid["Mh"], dtype=float) / mh_final
    ratio = np.where(np.asarray(grid["active_flag"], dtype=bool), ratio, np.nan)
    return np.asarray(grid["t_gyr"][0], dtype=float), ratio


def _plot(
    thesan: dict[str, Any],
    tng: dict[str, Any],
    output_dir: Path,
    *,
    cosmology: Cosmology,
) -> tuple[Path, Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    colors = {"thesan": "#009E73", "tng": "#D55E00", "mcbride": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, constrained_layout=True)
    axes_flat = axes.reshape(-1)

    rows: list[dict[str, Any]] = []
    max_lookback = 0.0
    for panel_index, ((lo, hi), ax) in enumerate(zip(MASS_BINS, axes_flat, strict=True)):
        thesan_indices, thesan_ratio = _select_bin(thesan, lo, hi)
        tng_indices, tng_ratio = _select_bin(tng, lo, hi)
        if thesan_indices.size == 0:
            raise ValueError(f"THESAN cache has no resolved tracks in logM bin [{lo}, {hi})")
        if tng_indices.size == 0:
            raise ValueError(f"TNG cache has no resolved tracks in logM bin [{lo}, {hi})")

        mc_t, mc_ratio = _build_mcbride_ratio(
            cosmology=cosmology,
            z_grid=np.asarray(thesan["z_grid"], dtype=float),
            logm_center=0.5 * (lo + hi),
            seed=9000 + panel_index,
        )
        thesan_lookback = _lookback_myr(np.asarray(thesan["t_gyr_grid"], dtype=float))
        tng_lookback = _lookback_myr(np.asarray(tng["t_gyr_grid"], dtype=float))
        mc_lookback = _lookback_myr(mc_t)
        max_lookback = max(max_lookback, float(np.nanmax(thesan_lookback)), float(np.nanmax(tng_lookback)))

        ax.plot(mc_lookback, mc_ratio.T, color=colors["mcbride"], alpha=0.055, lw=0.55, solid_capstyle="round")
        ax.plot(tng_lookback, tng_ratio.T, color=colors["tng"], alpha=0.07, lw=0.70, solid_capstyle="round")
        ax.plot(thesan_lookback, thesan_ratio.T, color=colors["thesan"], alpha=0.16, lw=0.65, solid_capstyle="round")

        tng_x = np.broadcast_to(tng_lookback[None, :], tng_ratio.shape)
        thesan_x = np.broadcast_to(thesan_lookback[None, :], thesan_ratio.shape)
        ax.scatter(tng_x[np.isfinite(tng_ratio)], tng_ratio[np.isfinite(tng_ratio)], s=6.0, color=colors["tng"], alpha=0.18, linewidths=0.0)
        ax.scatter(
            thesan_x[np.isfinite(thesan_ratio)],
            thesan_ratio[np.isfinite(thesan_ratio)],
            s=3.5,
            color=colors["thesan"],
            alpha=0.12,
            linewidths=0.0,
        )

        positive = np.concatenate([_finite_positive(thesan_ratio), _finite_positive(tng_ratio), _finite_positive(mc_ratio)])
        if positive.size == 0:
            raise RuntimeError("no finite positive MAH values available for plotting")
        ymin = max(1.0e-5, 10.0 ** np.floor(np.log10(float(np.nanpercentile(positive, 0.5)))))
        ax.set_yscale("log")
        ax.set_ylim(ymin, 1.35)
        ax.set_title(
            rf"$\log M_f=[{lo:.2f},{hi:.2f})$"
            "\n"
            rf"$N_{{\rm THESAN}}={thesan_indices.size},\ N_{{\rm TNG}}={tng_indices.size}$"
        )
        ax.set_ylabel(r"$M_h/M_{h,\rm final}$")
        ax.set_xlabel(r"lookback from final snapshot [Myr]")

        for backend, ratio, lookback, n_tracks in (
            ("thesan", thesan_ratio, thesan_lookback, int(thesan_indices.size)),
            ("tng", tng_ratio, tng_lookback, int(tng_indices.size)),
            ("mcbride", mc_ratio, mc_lookback, int(mc_ratio.shape[0])),
        ):
            row: dict[str, Any] = {
                "backend": backend,
                "logM_lo": float(lo),
                "logM_hi": float(hi),
                "n_tracks": n_tracks,
            }
            for target_myr in (100.0, 50.0, 30.0, 10.0):
                values = _interpolate_track_at_lookback(lookback, ratio, target_myr)
                finite = values[np.isfinite(values)]
                row[f"median_Mratio_at_{int(target_myr)}Myr"] = float(np.nanmedian(finite)) if finite.size else np.nan
                row[f"p16_Mratio_at_{int(target_myr)}Myr"] = float(np.nanpercentile(finite, 16.0)) if finite.size else np.nan
                row[f"p84_Mratio_at_{int(target_myr)}Myr"] = float(np.nanpercentile(finite, 84.0)) if finite.size else np.nan
                row[f"n_interp_at_{int(target_myr)}Myr"] = int(finite.size)
            rows.append(row)

    for ax in axes_flat:
        ax.set_xlim(max_lookback * 1.03, -4.0)

    handles = [
        mlines.Line2D([], [], color=colors["thesan"], lw=1.8, label="THESAN-dark-1 LHaloTree all selected chunks"),
        mlines.Line2D([], [], color=colors["tng"], lw=1.8, label="TNG100-1-Dark SubLink"),
        mlines.Line2D([], [], color=colors["mcbride"], lw=1.8, label="McBride09 Monte Carlo"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.045))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "thesan_tng_mcbride_z12_mah_spaghetti.png"
    pdf_path = output_dir / "thesan_tng_mcbride_z12_mah_spaghetti.pdf"
    summary_path = output_dir / "thesan_tng_mcbride_z12_mah_summary.csv"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return png_path, pdf_path, summary_path


def main() -> None:
    cosmology = Cosmology(
        h0=67.74 * SECONDS_PER_GYR / KM_PER_MPC,
        omega_m=0.3089,
        omega_b=0.0486,
        omega_lambda=0.6911,
    )
    thesan = _load_cache(THESAN_CACHE)
    tng = _load_cache(TNG_CACHE)
    png_path, pdf_path, summary_path = _plot(
        thesan,
        tng,
        OUTPUT_DIR,
        cosmology=cosmology,
    )
    print(f"wrote_png={png_path}", flush=True)
    print(f"wrote_pdf={pdf_path}", flush=True)
    print(f"wrote_summary={summary_path}", flush=True)
    print(f"thesan_z_final={float(thesan['z_grid'][-1]):.9f}", flush=True)
    print(f"tng_z_final={float(tng['z_grid'][-1]):.9f}", flush=True)
    print(f"thesan_grid_size={int(thesan['z_grid'].size)}", flush=True)
    print(f"tng_grid_size={int(tng['z_grid'].size)}", flush=True)


if __name__ == "__main__":
    main()
