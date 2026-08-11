from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from auroralf.constants import (
    PLANCK15_H0_GYR,
    PLANCK15_OMEGA_B,
    PLANCK15_OMEGA_LAMBDA,
    PLANCK15_OMEGA_M,
)
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.mah.tng import generate_tng_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf, uv_luminosity_to_muv
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_science_readiness"
DEFAULT_EVENT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tng_merger_events_n1000_allz"
    / "tng_merger_event_cache_summary.csv"
)


@dataclass(frozen=True)
class CacheSpec:
    label: str
    z_final: float
    snapshot: int
    cache_path: Path
    hmf_logm_min: float = 9.0
    hmf_logm_max: float = 10.0


DEFAULT_CACHE_SPECS = (
    CacheSpec(
        label="z6",
        z_final=6.0107574,
        snapshot=13,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z6p011_n8956.hdf5",
    ),
    CacheSpec(
        label="z8",
        z_final=8.01217295,
        snapshot=8,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z8p012_n6988.hdf5",
    ),
    CacheSpec(
        label="z10",
        z_final=9.99659047,
        snapshot=4,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z9p997_n5157.hdf5",
    ),
    CacheSpec(
        label="z12",
        z_final=11.98021332,
        snapshot=2,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5",
    ),
)


def _percentiles(values: np.ndarray, percentiles: tuple[float, ...] = (5.0, 16.0, 50.0, 84.0, 95.0)) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite values available for percentile summary")
    output = np.percentile(finite, percentiles)
    return {f"p{int(percentile):02d}": float(value) for percentile, value in zip(percentiles, output, strict=True)}


def _tracks_to_grid(tracks: dict[str, np.ndarray], n_tracks: int) -> dict[str, np.ndarray]:
    halo_id = np.asarray(tracks["halo_id"], dtype=int)
    if halo_id.size == 0:
        raise RuntimeError("halo history tracks are empty")
    if halo_id.size % int(n_tracks) != 0:
        raise RuntimeError("flattened tracks cannot be reshaped by n_tracks")
    steps_per_halo = halo_id.size // int(n_tracks)
    grid: dict[str, np.ndarray] = {}
    for key, value in tracks.items():
        array = np.asarray(value)
        if array.shape[0] == halo_id.size:
            grid[key] = array.reshape(int(n_tracks), steps_per_halo)
    return grid


def _candidate_count(cache_path: Path, logm_final: float, width_dex: float) -> int:
    with h5py.File(cache_path, "r") as handle:
        logm = np.asarray(handle["logM_final"], dtype=float)
    return int(np.count_nonzero(np.abs(logm - float(logm_final)) <= float(width_dex)))


def _summarize_history_grid(grid: dict[str, np.ndarray], mh_final: float) -> dict[str, float]:
    mh = np.asarray(grid["Mh"], dtype=float)
    dmhdt_raw = np.asarray(grid["dMh_dt_raw"], dtype=float)
    active = np.asarray(grid["active_flag"], dtype=bool)
    # The raw MAH derivative is retained in Msun/Gyr for assembly diagnostics.
    smar_gyr = dmhdt_raw / float(mh_final)
    positive = active & np.isfinite(smar_gyr) & (smar_gyr > 0.0)
    peak_smar = np.full(mh.shape[0], np.nan, dtype=float)
    rows_with_positive = np.any(positive, axis=1)
    peak_smar[rows_with_positive] = np.nanmax(np.where(positive, smar_gyr, np.nan)[rows_with_positive], axis=1)

    transition = active[:, 1:] & active[:, :-1] & np.isfinite(mh[:, 1:]) & np.isfinite(mh[:, :-1]) & (mh[:, :-1] > 0.0)
    jump = mh[:, 1:] / mh[:, :-1] - 1.0
    n_transition = int(np.count_nonzero(transition))
    if n_transition == 0:
        raise RuntimeError("no active mass transitions available for jump summary")
    positive_jump = transition & (jump > 0.0)
    large_jump_0p5 = transition & (jump > 0.5)
    large_jump_1p0 = transition & (jump > 1.0)

    summary = {
        "active_transition_count": float(n_transition),
        "positive_jump_fraction": float(np.count_nonzero(positive_jump) / n_transition),
        "jump_gt_0p5_fraction": float(np.count_nonzero(large_jump_0p5) / n_transition),
        "jump_gt_1p0_fraction": float(np.count_nonzero(large_jump_1p0) / n_transition),
        "peak_smar_gyr_count": float(np.count_nonzero(np.isfinite(peak_smar))),
    }
    summary.update({f"peak_smar_gyr_{key}": value for key, value in _percentiles(peak_smar).items()})
    return summary


def _summarize_uv_pipeline(result: Any, mh_final: float) -> dict[str, float]:
    muv = np.asarray(uv_luminosity_to_muv(np.asarray(result.uv_luminosities, dtype=float)), dtype=float)
    sfr_grid = _tracks_to_grid(result.sfr_tracks, int(result.metadata["n_tracks"]))
    sfr = np.asarray(sfr_grid["SFR"], dtype=float)
    active = np.asarray(sfr_grid["active_flag"], dtype=bool)
    positive_sfr = active & np.isfinite(sfr) & (sfr > 0.0)
    peak_sfr = np.full(sfr.shape[0], np.nan, dtype=float)
    rows = np.any(positive_sfr, axis=1)
    peak_sfr[rows] = np.nanmax(np.where(positive_sfr, sfr, np.nan)[rows], axis=1)
    history_summary = _summarize_history_grid(_tracks_to_grid(result.histories.tracks, int(result.metadata["n_tracks"])), mh_final)

    summary: dict[str, float] = {
        "muv_finite_count": float(np.count_nonzero(np.isfinite(muv))),
        "uv_lnu_p50": float(np.nanmedian(np.asarray(result.uv_luminosities, dtype=float))),
        "peak_sfr_count": float(np.count_nonzero(np.isfinite(peak_sfr))),
        "time_step_count": float(int(result.metadata["steps_per_halo"])),
        "dt_gyr_metadata": float(result.metadata["dt_gyr"]),
    }
    summary.update({f"muv_{key}": value for key, value in _percentiles(muv).items()})
    summary.update({f"peak_sfr_{key}": value for key, value in _percentiles(peak_sfr).items()})
    summary.update(history_summary)
    return summary


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows available to write {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_mah_shape_check(
    specs: tuple[CacheSpec, ...],
    *,
    cosmology: Cosmology,
    logm_values: tuple[float, ...],
    n_tracks: int,
    tng_time_grid_mode: str,
    tng_target_n_grid: int,
    mass_bin_width_dex: float,
    min_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        for mass_index, logm in enumerate(logm_values):
            mh_final = float(10.0**logm)
            candidates = _candidate_count(spec.cache_path, logm, mass_bin_width_dex)
            if candidates < min_candidates:
                raise ValueError(
                    f"{spec.label} logM={logm:.2f} has {candidates} TNG candidates, "
                    f"below min_candidates={min_candidates}"
                )

            tng_result = generate_tng_halo_histories(
                n_tracks=n_tracks,
                z_final=spec.z_final,
                Mh_final=mh_final,
                cosmology=cosmology,
                cache_path=spec.cache_path,
                z_start_max=20.1,
                mass_bin_width_dex=mass_bin_width_dex,
                min_candidates=min_candidates,
                random_seed=10_000 + 100 * spec_index + int(round(logm * 10.0)),
                time_grid_mode=tng_time_grid_mode,
                target_n_grid=tng_target_n_grid if tng_time_grid_mode == "uniform_in_t" else None,
            )
            tng_grid = _tracks_to_grid(tng_result.tracks, n_tracks)
            z_grid = np.asarray(tng_grid["z"][0], dtype=float)
            mc_result = generate_halo_histories(
                n_tracks=n_tracks,
                z_final=float(z_grid[-1]),
                Mh_final=mh_final,
                cosmology=cosmology,
                z_start_max=float(z_grid[0]),
                M_min=0.0,
                random_seed=20_000 + 100 * spec_index + int(round(logm * 10.0)),
                time_grid_mode="custom",
                custom_grid=z_grid,
                store_inactive_history=True,
                sampler="mcbride",
            )
            mc_grid = _tracks_to_grid(mc_result.tracks, n_tracks)

            for backend, result, grid in (
                ("tng", tng_result, tng_grid),
                ("mcbride", mc_result, mc_grid),
            ):
                row: dict[str, Any] = {
                    "label": spec.label,
                    "z_final": spec.z_final,
                    "snapshot": spec.snapshot,
                    "logM_final": logm,
                    "Mh_final": mh_final,
                    "backend": backend,
                    "n_tracks": n_tracks,
                    "grid_size": int(z_grid.size),
                    "tng_time_grid_mode": tng_time_grid_mode if backend == "tng" else "",
                    "candidate_count": candidates if backend == "tng" else "",
                    "negative_dmhdt_clip_fraction": result.metadata["negative_dmhdt_clip_fraction"],
                    "unresolved_step_fraction": result.metadata.get("unresolved_step_fraction", ""),
                }
                row.update(_summarize_history_grid(grid, mh_final))
                rows.append(row)
            print(f"mah_shape_done label={spec.label} logM={logm:.2f}", flush=True)
    return rows


def run_fixed_mass_uv_check(
    specs: tuple[CacheSpec, ...],
    *,
    cosmology: Cosmology,
    logm_values: tuple[float, ...],
    n_tracks: int,
    tng_time_grid_mode: str,
    mass_bin_width_dex: float,
    min_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        for logm in logm_values:
            mh_final = float(10.0**logm)
            candidates = _candidate_count(spec.cache_path, logm, mass_bin_width_dex)
            if candidates < min_candidates:
                raise ValueError(
                    f"{spec.label} logM={logm:.2f} has {candidates} TNG candidates, "
                    f"below min_candidates={min_candidates}"
                )
            for backend in ("mcbride", "tng"):
                result = run_halo_uv_pipeline(
                    n_tracks=n_tracks,
                    z_final=spec.z_final,
                    Mh_final=mh_final,
                    cosmology=cosmology,
                    random_seeds=derive_pipeline_random_seeds(
                        30_000 + 1000 * spec_index,
                        redshift=spec.z_final,
                        mass_index=mass_index,
                    ),
                    z_start_max=20.1,
                    n_grid=240,
                    mah_backend=backend,
                    tng_mah_cache_path=spec.cache_path if backend == "tng" else None,
                    tng_mass_bin_width_dex=mass_bin_width_dex,
                    tng_min_candidates=min_candidates,
                    tng_time_grid_mode=tng_time_grid_mode,
                    enable_time_delay=True,
                    workers=1,
                )
                row: dict[str, Any] = {
                    "label": spec.label,
                    "z_final": spec.z_final,
                    "snapshot": spec.snapshot,
                    "logM_final": logm,
                    "Mh_final": mh_final,
                    "backend": backend,
                    "n_tracks": n_tracks,
                    "candidate_count": candidates if backend == "tng" else "",
                    "time_grid_mode": result.metadata["time_grid_mode"],
                    "negative_dmhdt_clip_fraction": result.metadata["negative_dmhdt_clip_fraction"],
                    "tng_candidate_count": result.metadata.get("tng_candidate_count", ""),
                }
                row.update(_summarize_uv_pipeline(result, mh_final))
                rows.append(row)
                print(f"fixed_uv_done label={spec.label} logM={logm:.2f} backend={backend}", flush=True)
    return rows


def run_hmf_uvlf_smoke(
    specs: tuple[CacheSpec, ...],
    *,
    cosmology: Cosmology,
    n_mass: int,
    n_tracks: int,
    tng_time_grid_mode: str,
    mass_bin_width_dex: float,
    min_candidates: int,
) -> list[dict[str, Any]]:
    bins = np.linspace(-24.0, -8.0, 33)
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        for backend in ("mcbride", "tng"):
            result = sample_uvlf_from_hmf(
                z_obs=spec.z_final,
                cosmology=cosmology,
                N_mass=n_mass,
                n_tracks=n_tracks,
                base_seed=40_000 + 1000 * spec_index,
                bins=bins,
                logM_min=spec.hmf_logm_min,
                logM_max=spec.hmf_logm_max,
                z_start_max=20.1,
                n_grid=240,
                mah_backend=backend,
                tng_mah_cache_path=spec.cache_path if backend == "tng" else None,
                tng_mass_bin_width_dex=mass_bin_width_dex,
                tng_min_candidates=min_candidates,
                tng_time_grid_mode=tng_time_grid_mode,
                enable_time_delay=True,
                pipeline_workers=1,
            )
            muv = np.asarray(result.samples["Muv"], dtype=float)
            weight = np.asarray(result.samples["sample_weight"], dtype=float)
            finite = np.isfinite(muv) & np.isfinite(weight)
            if not np.any(finite):
                raise RuntimeError(f"no finite HMF UV samples for {spec.label} {backend}")
            phi = np.asarray(result.uvlf["phi"], dtype=float)
            row: dict[str, Any] = {
                "label": spec.label,
                "z_final": spec.z_final,
                "snapshot": spec.snapshot,
                "backend": backend,
                "N_mass": n_mass,
                "n_tracks": n_tracks,
                "logM_min": spec.hmf_logm_min,
                "logM_max": spec.hmf_logm_max,
                "finite_sample_count": int(np.count_nonzero(finite)),
                "nonzero_phi_bins": int(np.count_nonzero(phi > 0.0)),
                "weighted_density_muv_lt_18": float(np.sum(weight[finite & (muv < -18.0)])),
                "weighted_density_muv_lt_16": float(np.sum(weight[finite & (muv < -16.0)])),
                "weighted_density_muv_lt_14": float(np.sum(weight[finite & (muv < -14.0)])),
                "phi_max": float(np.nanmax(phi)),
                "sampling_seconds": float(result.metadata["sampling_seconds"]),
            }
            row.update({f"muv_{key}": value for key, value in _percentiles(muv[finite]).items()})
            rows.append(row)
            print(f"hmf_uvlf_done label={spec.label} backend={backend}", flush=True)
    return rows


def _read_event_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_summary_plot(
    *,
    output_dir: Path,
    event_summary_path: Path,
    mah_rows: list[dict[str, Any]],
    fixed_uv_rows: list[dict[str, Any]],
    hmf_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    event_rows = _read_event_summary(event_summary_path)
    snap_to_z = {spec.snapshot: spec.z_final for spec in DEFAULT_CACHE_SPECS}

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), constrained_layout=True)

    ax = axes[0, 0]
    z_event = np.array([snap_to_z[int(row["final_snapshot"])] for row in event_rows], dtype=float)
    major_1to4 = np.array([float(row["halo_fraction_mu_peak_ge_0p25"]) for row in event_rows], dtype=float)
    major_1to10 = np.array([float(row["halo_fraction_mu_peak_ge_0p10"]) for row in event_rows], dtype=float)
    ax.plot(z_event, major_1to4, marker="o", label=r"$\mu_{\rm peak}\geq0.25$")
    ax.plot(z_event, major_1to10, marker="s", label=r"$\mu_{\rm peak}\geq0.10$")
    ax.set_xlabel(r"final redshift $z_f$")
    ax.set_ylabel("halo fraction with event")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for backend, color, marker in (("tng", "#D55E00", "o"), ("mcbride", "#0072B2", "s")):
        rows = [row for row in mah_rows if row["backend"] == backend and float(row["logM_final"]) == 10.0]
        z = np.array([float(row["z_final"]) for row in rows], dtype=float)
        p50 = np.array([float(row["peak_smar_gyr_p50"]) for row in rows], dtype=float)
        p95 = np.array([float(row["peak_smar_gyr_p95"]) for row in rows], dtype=float)
        ax.plot(z, p50, marker=marker, color=color, label=f"{backend} p50")
        ax.plot(z, p95, marker=marker, color=color, ls="--", alpha=0.75, label=f"{backend} p95")
    ax.set_yscale("log")
    ax.set_xlabel(r"final redshift $z_f$")
    ax.set_ylabel(r"peak $\dot M_h/M_h$ [Gyr$^{-1}$]")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    for backend, color, marker in (("tng", "#D55E00", "o"), ("mcbride", "#0072B2", "s")):
        rows = [row for row in fixed_uv_rows if row["backend"] == backend and float(row["logM_final"]) == 10.0]
        z = np.array([float(row["z_final"]) for row in rows], dtype=float)
        p16 = np.array([float(row["muv_p16"]) for row in rows], dtype=float)
        p50 = np.array([float(row["muv_p50"]) for row in rows], dtype=float)
        p84 = np.array([float(row["muv_p84"]) for row in rows], dtype=float)
        ax.plot(z, p50, marker=marker, color=color, label=backend)
        ax.fill_between(z, p16, p84, color=color, alpha=0.18, lw=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"final redshift $z_f$")
    ax.set_ylabel(r"$M_{\rm UV}$ at $M_h=10^{10}M_\odot$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for backend, color, marker in (("tng", "#D55E00", "o"), ("mcbride", "#0072B2", "s")):
        rows = [row for row in hmf_rows if row["backend"] == backend]
        z = np.array([float(row["z_final"]) for row in rows], dtype=float)
        p16 = np.array([float(row["muv_p16"]) for row in rows], dtype=float)
        p50 = np.array([float(row["muv_p50"]) for row in rows], dtype=float)
        p84 = np.array([float(row["muv_p84"]) for row in rows], dtype=float)
        ax.plot(z, p50, marker=marker, color=color, label=backend)
        ax.fill_between(z, p16, p84, color=color, alpha=0.18, lw=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"final redshift $z_f$")
    ax.set_ylabel(r"HMF smoke median $M_{\rm UV}$")
    ax.legend(frameon=False, fontsize=8)

    png_path = output_dir / "tng_science_readiness_summary.png"
    pdf_path = output_dir / "tng_science_readiness_summary.pdf"
    fig.savefig(png_path, dpi=500)
    fig.savefig(pdf_path, dpi=500)
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small real-data science-readiness check for the TNG MAH backend.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-summary", type=Path, default=DEFAULT_EVENT_SUMMARY)
    parser.add_argument("--n-tracks-mah", type=int, default=1000)
    parser.add_argument("--n-tracks-fixed-uv", type=int, default=250)
    parser.add_argument("--hmf-n-mass", type=int, default=4)
    parser.add_argument("--hmf-n-tracks", type=int, default=40)
    parser.add_argument("--tng-time-grid-mode", choices=("snapshot", "uniform_in_t"), default="snapshot")
    parser.add_argument("--tng-target-n-grid", type=int, default=240)
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.15)
    parser.add_argument("--min-candidates", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cosmology = Cosmology(
        h0=PLANCK15_H0_GYR,
        omega_m=PLANCK15_OMEGA_M,
        omega_b=PLANCK15_OMEGA_B,
        omega_lambda=PLANCK15_OMEGA_LAMBDA,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_summary_path = args.event_summary.expanduser().resolve()
    if not event_summary_path.exists():
        raise FileNotFoundError(f"event summary not found: {event_summary_path}")
    for spec in DEFAULT_CACHE_SPECS:
        if not spec.cache_path.exists():
            raise FileNotFoundError(f"TNG MAH cache not found: {spec.cache_path}")

    logm_values = (9.5, 10.0)
    mah_rows = run_mah_shape_check(
        DEFAULT_CACHE_SPECS,
        cosmology=cosmology,
        logm_values=logm_values,
        n_tracks=int(args.n_tracks_mah),
        tng_time_grid_mode=str(args.tng_time_grid_mode),
        tng_target_n_grid=int(args.tng_target_n_grid),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        min_candidates=int(args.min_candidates),
    )
    fixed_uv_rows = run_fixed_mass_uv_check(
        DEFAULT_CACHE_SPECS,
        cosmology=cosmology,
        logm_values=logm_values,
        n_tracks=int(args.n_tracks_fixed_uv),
        tng_time_grid_mode=str(args.tng_time_grid_mode),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        min_candidates=int(args.min_candidates),
    )
    hmf_rows = run_hmf_uvlf_smoke(
        DEFAULT_CACHE_SPECS,
        cosmology=cosmology,
        n_mass=int(args.hmf_n_mass),
        n_tracks=int(args.hmf_n_tracks),
        tng_time_grid_mode=str(args.tng_time_grid_mode),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        min_candidates=int(args.min_candidates),
    )

    mah_path = output_dir / "tng_vs_mcbride_mah_shape_summary.csv"
    fixed_uv_path = output_dir / "tng_vs_mcbride_fixed_mass_uv_summary.csv"
    hmf_path = output_dir / "tng_vs_mcbride_hmf_uvlf_smoke_summary.csv"
    _write_rows(mah_path, mah_rows)
    _write_rows(fixed_uv_path, fixed_uv_rows)
    _write_rows(hmf_path, hmf_rows)
    png_path, pdf_path = make_summary_plot(
        output_dir=output_dir,
        event_summary_path=event_summary_path,
        mah_rows=mah_rows,
        fixed_uv_rows=fixed_uv_rows,
        hmf_rows=hmf_rows,
    )
    print(f"saved_mah_summary={mah_path}")
    print(f"saved_fixed_uv_summary={fixed_uv_path}")
    print(f"saved_hmf_summary={hmf_path}")
    print(f"saved_plot_png={png_path}")
    print(f"saved_plot_pdf={pdf_path}")


if __name__ == "__main__":
    main()
