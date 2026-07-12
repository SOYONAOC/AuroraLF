from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from auroralf.mah import Cosmology
from auroralf.mah.models import KM_PER_MPC, SECONDS_PER_GYR
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf, uv_luminosity_to_muv
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_bright_end_muv20"


@dataclass(frozen=True)
class CacheSpec:
    label: str
    z_final: float
    cache_path: Path
    hmf_logm_min: float
    hmf_logm_max: float


SPECS = (
    CacheSpec(
        label="z6",
        z_final=6.0107574,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z6p011_n8956.hdf5",
        hmf_logm_min=10.40,
        hmf_logm_max=11.75,
    ),
    CacheSpec(
        label="z8",
        z_final=8.01217295,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z8p012_n6988.hdf5",
        hmf_logm_min=10.40,
        hmf_logm_max=11.20,
    ),
    CacheSpec(
        label="z10",
        z_final=9.99659047,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z9p997_n5157.hdf5",
        hmf_logm_min=10.00,
        hmf_logm_max=10.75,
    ),
    CacheSpec(
        label="z12",
        z_final=11.98021332,
        cache_path=PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5",
        hmf_logm_min=9.80,
        hmf_logm_max=10.25,
    ),
)


def _candidate_count(cache_path: Path, logm_final: float, width_dex: float) -> int:
    with h5py.File(cache_path, "r") as handle:
        logm = np.asarray(handle["logM_final"], dtype=float)
    return int(np.count_nonzero(np.abs(logm - float(logm_final)) <= float(width_dex)))


def _percentiles(values: np.ndarray, percentiles: tuple[float, ...] = (5.0, 16.0, 50.0, 84.0, 95.0)) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite values available for percentile summary")
    output = np.percentile(finite, percentiles)
    return {f"p{int(percentile):02d}": float(value) for percentile, value in zip(percentiles, output, strict=True)}


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


def run_fixed_mass_scan(
    *,
    cosmology: Cosmology,
    n_tracks: int,
    min_candidates: int,
    mass_bin_width_dex: float,
) -> list[dict[str, Any]]:
    logm_grid = np.arange(10.0, 11.76, 0.25)
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPECS):
        for mass_index, logm in enumerate(logm_grid):
            candidates = _candidate_count(spec.cache_path, float(logm), mass_bin_width_dex)
            if candidates < min_candidates:
                continue
            mh_final = float(10.0**logm)
            for backend in ("mcbride", "tng"):
                result = run_halo_uv_pipeline(
                    n_tracks=n_tracks,
                    z_final=spec.z_final,
                    Mh_final=mh_final,
                    cosmology=cosmology,
                    random_seeds=derive_pipeline_random_seeds(
                        71_000 + 1000 * spec_index,
                        redshift=spec.z_final,
                        mass_index=mass_index,
                    ),
                    z_start_max=20.1,
                    n_grid=240,
                    mah_backend=backend,
                    tng_mah_cache_path=spec.cache_path if backend == "tng" else None,
                    tng_mass_bin_width_dex=mass_bin_width_dex,
                    tng_min_candidates=min_candidates,
                    tng_time_grid_mode="uniform_in_t",
                    enable_time_delay=True,
                    workers=1,
                )
                muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
                row: dict[str, Any] = {
                    "label": spec.label,
                    "z_final": spec.z_final,
                    "backend": backend,
                    "logM_final": float(logm),
                    "Mh_final": mh_final,
                    "n_tracks": n_tracks,
                    "candidate_count": candidates if backend == "tng" else "",
                    "time_grid_mode": result.metadata["time_grid_mode"],
                }
                row.update({f"muv_{key}": value for key, value in _percentiles(muv).items()})
                rows.append(row)
            print(f"fixed_done label={spec.label} logM={logm:.2f} candidates={candidates}", flush=True)
    return rows


def run_bright_hmf_uvlf(
    *,
    cosmology: Cosmology,
    n_mass: int,
    n_tracks: int,
    min_candidates: int,
    mass_bin_width_dex: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bins = np.arange(-22.25, -17.75 + 0.5, 0.5)
    for spec_index, spec in enumerate(SPECS):
        for backend in ("mcbride", "tng"):
            result = sample_uvlf_from_hmf(
                z_obs=spec.z_final,
                cosmology=cosmology,
                N_mass=n_mass,
                n_tracks=n_tracks,
                base_seed=81_000 + 1000 * spec_index,
                bins=bins,
                logM_min=spec.hmf_logm_min,
                logM_max=spec.hmf_logm_max,
                z_start_max=20.1,
                n_grid=240,
                mah_backend=backend,
                tng_mah_cache_path=spec.cache_path if backend == "tng" else None,
                tng_mass_bin_width_dex=mass_bin_width_dex,
                tng_min_candidates=min_candidates,
                tng_time_grid_mode="uniform_in_t",
                enable_time_delay=True,
                pipeline_workers=1,
            )
            centers = np.asarray(result.uvlf["bin_centers"], dtype=float)
            phi = np.asarray(result.uvlf["phi"], dtype=float)
            phi_sigma = np.asarray(result.uvlf["phi_sigma"], dtype=float)
            raw_counts = np.asarray(result.uvlf["raw_counts"], dtype=np.int64)
            effective_counts = np.asarray(result.uvlf["effective_counts"], dtype=float)
            target_index = int(np.argmin(np.abs(centers + 20.0)))
            if not np.isclose(centers[target_index], -20.0):
                raise RuntimeError("UVLF bins do not contain a center at MUV=-20")
            muv = np.asarray(result.samples["Muv"], dtype=float)
            row: dict[str, Any] = {
                "label": spec.label,
                "z_final": spec.z_final,
                "backend": backend,
                "N_mass": n_mass,
                "n_tracks": n_tracks,
                "logM_min": spec.hmf_logm_min,
                "logM_max": spec.hmf_logm_max,
                "target_muv": float(centers[target_index]),
                "phi_muv_minus20": float(phi[target_index]),
                "phi_sigma_muv_minus20": float(phi_sigma[target_index]),
                "raw_counts_muv_minus20": int(raw_counts[target_index]),
                "effective_counts_muv_minus20": float(effective_counts[target_index]),
                "nonzero_phi_bins": int(np.count_nonzero(phi > 0.0)),
                "sampling_seconds": float(result.metadata["sampling_seconds"]),
            }
            row.update({f"sample_muv_{key}": value for key, value in _percentiles(muv).items()})
            rows.append(row)
            print(f"hmf_done label={spec.label} backend={backend}", flush=True)
    return rows


def summarize_phi_ratios(hmf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        pair = {row["backend"]: row for row in hmf_rows if row["label"] == spec.label}
        if "tng" not in pair or "mcbride" not in pair:
            raise RuntimeError(f"missing HMF rows for {spec.label}")
        tng_phi = float(pair["tng"]["phi_muv_minus20"])
        mc_phi = float(pair["mcbride"]["phi_muv_minus20"])
        ratio = np.nan
        if mc_phi > 0.0:
            ratio = tng_phi / mc_phi
        rows.append(
            {
                "label": spec.label,
                "z_final": spec.z_final,
                "phi_tng_muv_minus20": tng_phi,
                "phi_mcbride_muv_minus20": mc_phi,
                "phi_ratio_tng_over_mcbride": float(ratio),
                "raw_counts_tng": int(pair["tng"]["raw_counts_muv_minus20"]),
                "raw_counts_mcbride": int(pair["mcbride"]["raw_counts_muv_minus20"]),
                "effective_counts_tng": float(pair["tng"]["effective_counts_muv_minus20"]),
                "effective_counts_mcbride": float(pair["mcbride"]["effective_counts_muv_minus20"]),
            }
        )
    return rows


def make_fixed_mass_plot(output_dir: Path, fixed_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    plt.style.use("apj")
    colors = {"tng": "#D55E00", "mcbride": "#0072B2"}
    markers = {"tng": "o", "mcbride": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)

    for ax, spec in zip(axes.flat, SPECS, strict=True):
        for backend in ("mcbride", "tng"):
            rows = [row for row in fixed_rows if row["label"] == spec.label and row["backend"] == backend]
            if not rows:
                continue
            logm = np.array([float(row["logM_final"]) for row in rows], dtype=float)
            p16 = np.array([float(row["muv_p16"]) for row in rows], dtype=float)
            p50 = np.array([float(row["muv_p50"]) for row in rows], dtype=float)
            p84 = np.array([float(row["muv_p84"]) for row in rows], dtype=float)
            ax.plot(logm, p50, marker=markers[backend], color=colors[backend], label=backend)
            ax.fill_between(logm, p16, p84, color=colors[backend], alpha=0.18, lw=0)
        ax.axhline(-20.0, color="0.25", lw=1.0, ls=":")
        ax.invert_yaxis()
        ax.set_title(rf"${spec.label.replace('z', 'z=')}$")
        ax.set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
        ax.set_ylabel(r"$M_{\rm UV}$")
        ax.legend(frameon=False, fontsize=8)

    png_path = output_dir / "tng_bright_end_fixed_mass_muv20.png"
    pdf_path = output_dir / "tng_bright_end_fixed_mass_muv20.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def make_phi_ratio_plot(output_dir: Path, ratio_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    plt.style.use("apj")
    z = np.array([float(row["z_final"]) for row in ratio_rows], dtype=float)
    ratio = np.array([float(row["phi_ratio_tng_over_mcbride"]) for row in ratio_rows], dtype=float)
    tng_phi = np.array([float(row["phi_tng_muv_minus20"]) for row in ratio_rows], dtype=float)
    mc_phi = np.array([float(row["phi_mcbride_muv_minus20"]) for row in ratio_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(z, mc_phi, marker="s", color="#0072B2", label="mcbride")
    ax.plot(z, tng_phi, marker="o", color="#D55E00", label="tng")
    positive = np.concatenate([mc_phi[mc_phi > 0.0], tng_phi[tng_phi > 0.0]])
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(float(np.min(positive)) * 0.5, float(np.max(positive)) * 2.0)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$\phi(M_{\rm UV}=-20)$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    finite = np.isfinite(ratio)
    ax.axhline(1.0, color="0.25", lw=1.0, ls=":")
    ax.plot(z[finite], ratio[finite], marker="o", color="black")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$\phi_{\rm TNG}/\phi_{\rm McBride}$ at $M_{\rm UV}=-20$")
    if np.any(finite):
        ax.set_ylim(0.0, max(2.0, float(np.nanmax(ratio[finite])) * 1.2))

    png_path = output_dir / "tng_bright_end_phi_muv20_ratio.png"
    pdf_path = output_dir / "tng_bright_end_phi_muv20_ratio.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TNG-vs-McBride bright-end UVLF around MUV=-20.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-n-tracks", type=int, default=300)
    parser.add_argument("--hmf-n-mass", type=int, default=60)
    parser.add_argument("--hmf-n-tracks", type=int, default=80)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--hmf-min-candidates", type=int, default=20)
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cosmology = Cosmology(
        h0=67.74 * SECONDS_PER_GYR / KM_PER_MPC,
        omega_m=0.3089,
        omega_b=0.0486,
        omega_lambda=0.6911,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        if not spec.cache_path.exists():
            raise FileNotFoundError(f"TNG MAH cache not found: {spec.cache_path}")

    fixed_rows = run_fixed_mass_scan(
        cosmology=cosmology,
        n_tracks=int(args.fixed_n_tracks),
        min_candidates=int(args.min_candidates),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
    )
    hmf_rows = run_bright_hmf_uvlf(
        cosmology=cosmology,
        n_mass=int(args.hmf_n_mass),
        n_tracks=int(args.hmf_n_tracks),
        min_candidates=int(args.hmf_min_candidates),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
    )
    ratio_rows = summarize_phi_ratios(hmf_rows)

    fixed_path = output_dir / "fixed_mass_muv20_scan.csv"
    hmf_path = output_dir / "hmf_bright_end_muv20_summary.csv"
    ratio_path = output_dir / "phi_muv20_ratio_summary.csv"
    _write_rows(fixed_path, fixed_rows)
    _write_rows(hmf_path, hmf_rows)
    _write_rows(ratio_path, ratio_rows)
    fixed_png, fixed_pdf = make_fixed_mass_plot(output_dir, fixed_rows)
    ratio_png, ratio_pdf = make_phi_ratio_plot(output_dir, ratio_rows)
    print(f"saved_fixed_scan={fixed_path}")
    print(f"saved_hmf_summary={hmf_path}")
    print(f"saved_ratio_summary={ratio_path}")
    print(f"saved_fixed_plot_png={fixed_png}")
    print(f"saved_fixed_plot_pdf={fixed_pdf}")
    print(f"saved_ratio_plot_png={ratio_png}")
    print(f"saved_ratio_plot_pdf={ratio_pdf}")


if __name__ == "__main__":
    main()
