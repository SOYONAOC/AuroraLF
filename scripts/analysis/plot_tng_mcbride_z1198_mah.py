from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from auroralf.mah import generate_halo_histories
from auroralf.mah.tng import (
    _load_tng_cache,
    _regrid_mass_ratio_uniform_in_t,
    _slice_grid,
    _smooth_mass_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Z_FINAL = 11.9802133153
TNG_CACHE = PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_z1198_mah_compare"


def _tracks_to_grid(result: Any, n_tracks: int) -> dict[str, np.ndarray]:
    n_steps = int(result.metadata["grid_size"])
    grids: dict[str, np.ndarray] = {}
    for key in ("z", "t_gyr", "Mh", "dMh_dt", "active_flag"):
        array = np.asarray(result.tracks[key])
        if array.size != int(n_tracks) * n_steps:
            raise ValueError(f"track column {key!r} cannot be reshaped to ({n_tracks}, {n_steps})")
        grids[key] = array.reshape(int(n_tracks), n_steps)
    grids["active_flag"] = grids["active_flag"].astype(bool)
    return grids


def _lookback_myr(time_gyr: np.ndarray) -> np.ndarray:
    return (float(time_gyr[-1]) - np.asarray(time_gyr, dtype=float)) * 1.0e3


def _load_all_tng_candidates(
    *,
    logm_final: float,
    min_candidates: int,
    mass_bin_width_dex: float,
    target_n_grid: int,
    z_start_max: float,
) -> dict[str, Any]:
    _, cache = _load_tng_cache(TNG_CACHE, z_final=Z_FINAL)
    z_grid, t_grid, mass_ratio, resolved_mask = _slice_grid(
        np.asarray(cache["z_grid"], dtype=float),
        np.asarray(cache["t_gyr_grid"], dtype=float),
        np.asarray(cache["mass_ratio"], dtype=float),
        np.asarray(cache["resolved_mask"], dtype=bool),
        z_final=Z_FINAL,
        z_start_max=z_start_max,
    )
    mass_ratio = _smooth_mass_ratio(mass_ratio, t_grid, smoothing_myr=0.0)
    mass_ratio /= mass_ratio[:, -1][:, None]
    mass_ratio[:, -1] = 1.0

    logm_cache = np.asarray(cache["logM_final"], dtype=float)
    candidate_mask = np.abs(logm_cache - float(logm_final)) <= float(mass_bin_width_dex)
    candidate_mask &= np.count_nonzero(resolved_mask, axis=1) >= 2
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size < int(min_candidates):
        raise ValueError(
            "TNG MAH candidate count "
            f"{candidate_indices.size} is below min_candidates={int(min_candidates)} "
            f"for log10(Mh_final)={float(logm_final):.3f} within {float(mass_bin_width_dex):.3f} dex"
        )

    raw_ratio = mass_ratio[candidate_indices]
    raw_resolved = resolved_mask[candidate_indices]
    uniform_z, uniform_t, uniform_ratio, uniform_resolved = _regrid_mass_ratio_uniform_in_t(
        z_grid=z_grid,
        t_gyr_grid=t_grid,
        mass_ratio=raw_ratio,
        resolved_mask=raw_resolved,
        target_n_grid=int(target_n_grid),
    )

    return {
        "candidate_indices": candidate_indices.astype(np.int64),
        "source_subhalo_id": np.asarray(cache["source_subhalo_id"], dtype=np.int64)[candidate_indices],
        "candidate_count": int(candidate_indices.size),
        "source_simulation": str(cache["source_simulation"]),
        "raw_z": z_grid,
        "raw_t_gyr": t_grid,
        "raw_mass_ratio": raw_ratio,
        "raw_resolved_mask": raw_resolved,
        "uniform_z": uniform_z,
        "uniform_t_gyr": uniform_t,
        "uniform_mass_ratio": uniform_ratio,
        "uniform_resolved_mask": uniform_resolved,
    }


def _build_pair(
    *,
    logm_final: float,
    min_candidates: int,
    mass_bin_width_dex: float,
    target_n_grid: int,
    z_start_max: float,
    seed: int,
) -> dict[str, Any]:
    mh_final = float(10.0**logm_final)
    tng = _load_all_tng_candidates(
        logm_final=logm_final,
        min_candidates=min_candidates,
        mass_bin_width_dex=mass_bin_width_dex,
        target_n_grid=target_n_grid,
        z_start_max=z_start_max,
    )
    n_tracks = int(tng["candidate_count"])
    mcbride = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=float(tng["uniform_z"][-1]),
        Mh_final=mh_final,
        z_start_max=float(tng["uniform_z"][0]),
        M_min=0.0,
        random_seed=seed + 10_000,
        time_grid_mode="custom",
        custom_grid=np.asarray(tng["uniform_z"], dtype=float),
        store_inactive_history=True,
        sampler="mcbride",
    )
    return {
        "logm_final": float(logm_final),
        "mh_final": mh_final,
        "tng": tng,
        "mcbride": mcbride,
        "mc_grid": _tracks_to_grid(mcbride, n_tracks),
    }


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _plot(samples: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    plt.style.use("apj")
    colors = {"tng": "#D55E00", "mcbride": "#0072B2"}
    fig, axes = plt.subplots(1, len(samples), figsize=(10.0, 3.8), sharey=False, constrained_layout=True)
    if len(samples) == 1:
        axes = np.asarray([axes])

    for ax, sample in zip(axes, samples, strict=True):
        tng = sample["tng"]
        mc_grid = sample["mc_grid"]
        mh_final = float(sample["mh_final"])

        tng_lookback = _lookback_myr(np.asarray(tng["uniform_t_gyr"], dtype=float))
        tng_mass = np.where(
            np.asarray(tng["uniform_resolved_mask"], dtype=bool),
            np.asarray(tng["uniform_mass_ratio"], dtype=float),
            np.nan,
        )
        mc_lookback = _lookback_myr(np.asarray(mc_grid["t_gyr"][0], dtype=float))
        mc_mass = np.asarray(mc_grid["Mh"], dtype=float) / mh_final

        ax.plot(mc_lookback, mc_mass.T, color=colors["mcbride"], alpha=0.08, lw=0.55, solid_capstyle="round")
        ax.plot(tng_lookback, tng_mass.T, color=colors["tng"], alpha=0.17, lw=0.75, solid_capstyle="round")

        raw_lookback = _lookback_myr(np.asarray(tng["raw_t_gyr"], dtype=float))
        raw_mass = np.asarray(tng["raw_mass_ratio"], dtype=float)
        raw_resolved = np.asarray(tng["raw_resolved_mask"], dtype=bool)
        raw_x = np.broadcast_to(raw_lookback[None, :], raw_mass.shape)
        ax.scatter(
            raw_x[raw_resolved],
            raw_mass[raw_resolved],
            s=7.0,
            marker="o",
            color=colors["tng"],
            alpha=0.30,
            linewidths=0.0,
            zorder=3,
        )

        mass_values = np.concatenate(
            [
                _finite_positive(tng_mass),
                _finite_positive(mc_mass),
            ]
        )
        if mass_values.size == 0:
            raise RuntimeError("no finite positive MAH values available for plotting")
        ymin = 10.0 ** np.floor(np.log10(float(np.nanpercentile(mass_values, 0.5))))
        ymin = max(min(ymin, 1.0e-2), 1.0e-7)

        ax.set_title(
            rf"$z_f=11.98,\ \log M_f={sample['logm_final']:.2f}$"
            "\n"
            rf"$N_{{\rm TNG}}={int(tng['candidate_count'])}$ tracks"
        )
        ax.set_yscale("log")
        ax.set_ylim(ymin, 1.35)
        ax.set_xlim(205.0, -5.0)
        ax.set_xlabel(r"lookback from $z_f$ [Myr]")
        ax.set_ylabel(r"$M_h/M_{h,\rm final}$")

    handles = [
        mlines.Line2D([], [], color=colors["tng"], lw=1.8, label="TNG candidates, one line per halo"),
        mlines.Line2D([], [], color=colors["mcbride"], lw=1.8, label="McBride09, same number of tracks"),
        mlines.Line2D(
            [],
            [],
            color=colors["tng"],
            marker="o",
            ls="none",
            markersize=4,
            alpha=0.65,
            label="TNG raw snapshot points",
        ),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.07))

    png_path = output_dir / "tng_vs_mcbride_z11p98_mah_tracks.png"
    pdf_path = output_dir / "tng_vs_mcbride_z11p98_mah_tracks.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _write_summary(samples: list[dict[str, Any]], output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        tng = sample["tng"]
        mh_final = float(sample["mh_final"])
        mc_mass = np.asarray(sample["mc_grid"]["Mh"], dtype=float) / mh_final
        tng_mass = np.where(
            np.asarray(tng["uniform_resolved_mask"], dtype=bool),
            np.asarray(tng["uniform_mass_ratio"], dtype=float),
            np.nan,
        )
        for backend, mass, lookback in (
            ("tng", tng_mass, _lookback_myr(np.asarray(tng["uniform_t_gyr"], dtype=float))),
            ("mcbride", mc_mass, _lookback_myr(np.asarray(sample["mc_grid"]["t_gyr"][0], dtype=float))),
        ):
            row: dict[str, Any] = {
                "backend": backend,
                "z_final": Z_FINAL,
                "logM_final": sample["logm_final"],
                "n_tracks": int(mass.shape[0]),
                "tng_candidate_count": int(tng["candidate_count"]) if backend == "tng" else "",
            }
            for lb in (100.0, 50.0, 30.0, 10.0):
                values = np.full(mass.shape[0], np.nan, dtype=float)
                for halo_index in range(mass.shape[0]):
                    valid = np.isfinite(mass[halo_index])
                    if (
                        np.count_nonzero(valid) >= 2
                        and lb >= np.nanmin(lookback[valid])
                        and lb <= np.nanmax(lookback[valid])
                    ):
                        order = np.argsort(lookback[valid])
                        values[halo_index] = np.interp(lb, lookback[valid][order], mass[halo_index, valid][order])
                row[f"mass_ratio_lookback_{int(lb)}myr_p16"] = float(np.nanpercentile(values, 16.0))
                row[f"mass_ratio_lookback_{int(lb)}myr_p50"] = float(np.nanpercentile(values, 50.0))
                row[f"mass_ratio_lookback_{int(lb)}myr_p84"] = float(np.nanpercentile(values, 84.0))
            rows.append(row)

    path = output_dir / "tng_vs_mcbride_z11p98_mah_tracks_summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot every available z=11.98 TNG MAH candidate against McBride09.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.15)
    parser.add_argument("--target-n-grid", type=int, default=240)
    parser.add_argument("--z-start-max", type=float, default=20.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not TNG_CACHE.exists():
        raise FileNotFoundError(f"TNG MAH cache not found: {TNG_CACHE}")
    samples = [
        _build_pair(
            logm_final=logm,
            min_candidates=int(args.min_candidates),
            mass_bin_width_dex=float(args.mass_bin_width_dex),
            target_n_grid=int(args.target_n_grid),
            z_start_max=float(args.z_start_max),
            seed=101_000 + int(round(logm * 100.0)),
        )
        for logm in (10.0, 10.25)
    ]
    png_path, pdf_path = _plot(samples, output_dir)
    summary_path = _write_summary(samples, output_dir)
    print(f"saved_plot_png={png_path}")
    print(f"saved_plot_pdf={pdf_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
