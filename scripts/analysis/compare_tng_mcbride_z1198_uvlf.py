from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from auroralf.mah import Cosmology
from auroralf.mah.models import KM_PER_MPC, SECONDS_PER_GYR
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf, uv_luminosity_to_muv
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Z_FINAL = 11.9802133153
TNG_CACHE = PROJECT_ROOT / "data_save" / "tng_mah_cache" / "TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_z1198_uvlf_compare"


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
    rows: list[dict[str, Any]] = []
    for mass_index, logm in enumerate((9.0, 9.25, 9.5, 9.75, 10.0, 10.25)):
        mh_final = float(10.0**logm)
        for backend in ("mcbride", "tng"):
            result = run_halo_uv_pipeline(
                n_tracks=n_tracks,
                z_final=Z_FINAL,
                Mh_final=mh_final,
                cosmology=cosmology,
                random_seeds=derive_pipeline_random_seeds(
                    91_000,
                    redshift=Z_FINAL,
                    mass_index=mass_index,
                ),
                z_start_max=20.1,
                n_grid=240,
                mah_backend=backend,
                tng_mah_cache_path=TNG_CACHE if backend == "tng" else None,
                tng_mass_bin_width_dex=mass_bin_width_dex,
                tng_min_candidates=min_candidates,
                tng_time_grid_mode="uniform_in_t",
                enable_time_delay=True,
                workers=1,
            )
            muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
            row: dict[str, Any] = {
                "z_final": Z_FINAL,
                "backend": backend,
                "logM_final": float(logm),
                "Mh_final": mh_final,
                "n_tracks": n_tracks,
                "time_grid_mode": result.metadata["time_grid_mode"],
                "tng_candidate_count": result.metadata.get("tng_candidate_count", ""),
            }
            row.update({f"muv_{key}": value for key, value in _percentiles(muv).items()})
            rows.append(row)
        print(f"fixed_done logM={logm:.2f}", flush=True)
    return rows


def run_uvlf(
    *,
    cosmology: Cosmology,
    n_mass: int,
    n_tracks: int,
    min_candidates: int,
    mass_bin_width_dex: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bins = np.arange(-20.0, -12.0 + 0.5, 0.5)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for backend in ("mcbride", "tng"):
        result = sample_uvlf_from_hmf(
            z_obs=Z_FINAL,
            cosmology=cosmology,
            N_mass=n_mass,
            n_tracks=n_tracks,
            base_seed=92_000,
            bins=bins,
            logM_min=9.0,
            logM_max=10.25,
            z_start_max=20.1,
            n_grid=240,
            mah_backend=backend,
            tng_mah_cache_path=TNG_CACHE if backend == "tng" else None,
            tng_mass_bin_width_dex=mass_bin_width_dex,
            tng_min_candidates=min_candidates,
            tng_time_grid_mode="uniform_in_t",
            enable_time_delay=True,
            pipeline_workers=1,
        )
        results[backend] = result
        samples_muv = np.asarray(result.samples["Muv"], dtype=float)
        sample_weight = np.asarray(result.samples["sample_weight"], dtype=float)
        finite = np.isfinite(samples_muv) & np.isfinite(sample_weight)
        row: dict[str, Any] = {
            "z_final": Z_FINAL,
            "backend": backend,
            "N_mass": n_mass,
            "n_tracks": n_tracks,
            "logM_min": 9.0,
            "logM_max": 10.25,
            "weighted_density_muv_lt_18": float(np.sum(sample_weight[finite & (samples_muv < -18.0)])),
            "weighted_density_muv_lt_17": float(np.sum(sample_weight[finite & (samples_muv < -17.0)])),
            "weighted_density_muv_lt_16": float(np.sum(sample_weight[finite & (samples_muv < -16.0)])),
            "sampling_seconds": float(result.metadata["sampling_seconds"]),
            "nonzero_phi_bins": int(np.count_nonzero(np.asarray(result.uvlf["phi"], dtype=float) > 0.0)),
        }
        row.update({f"sample_muv_{key}": value for key, value in _percentiles(samples_muv).items()})
        rows.append(row)
        print(f"uvlf_done backend={backend}", flush=True)
    return rows, results


def build_bin_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    mc = results["mcbride"]
    tng = results["tng"]
    centers = np.asarray(mc.uvlf["bin_centers"], dtype=float)
    mc_phi = np.asarray(mc.uvlf["phi"], dtype=float)
    tng_phi = np.asarray(tng.uvlf["phi"], dtype=float)
    mc_sigma = np.asarray(mc.uvlf["phi_sigma"], dtype=float)
    tng_sigma = np.asarray(tng.uvlf["phi_sigma"], dtype=float)
    rows: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        ratio = np.nan
        if mc_phi[index] > 0.0:
            ratio = tng_phi[index] / mc_phi[index]
        rows.append(
            {
                "z_final": Z_FINAL,
                "Muv_center": float(center),
                "phi_mcbride": float(mc_phi[index]),
                "phi_tng": float(tng_phi[index]),
                "phi_ratio_tng_over_mcbride": float(ratio),
                "phi_sigma_mcbride": float(mc_sigma[index]),
                "phi_sigma_tng": float(tng_sigma[index]),
                "raw_counts_mcbride": int(mc.uvlf["raw_counts"][index]),
                "raw_counts_tng": int(tng.uvlf["raw_counts"][index]),
                "effective_counts_mcbride": float(mc.uvlf["effective_counts"][index]),
                "effective_counts_tng": float(tng.uvlf["effective_counts"][index]),
            }
        )
    return rows


def make_plot(output_dir: Path, fixed_rows: list[dict[str, Any]], bin_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    plt.style.use("apj")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    colors = {"mcbride": "#0072B2", "tng": "#D55E00"}
    markers = {"mcbride": "s", "tng": "o"}

    ax = axes[0]
    for backend in ("mcbride", "tng"):
        rows = [row for row in fixed_rows if row["backend"] == backend]
        logm = np.array([float(row["logM_final"]) for row in rows], dtype=float)
        p16 = np.array([float(row["muv_p16"]) for row in rows], dtype=float)
        p50 = np.array([float(row["muv_p50"]) for row in rows], dtype=float)
        p84 = np.array([float(row["muv_p84"]) for row in rows], dtype=float)
        ax.plot(logm, p50, marker=markers[backend], color=colors[backend], label=backend)
        ax.fill_between(logm, p16, p84, color=colors[backend], alpha=0.18, lw=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
    ax.set_ylabel(r"$M_{\rm UV}$")
    ax.set_title(r"$z=11.98$ fixed mass")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    rows = bin_rows
    centers = np.array([float(row["Muv_center"]) for row in rows], dtype=float)
    for backend, key in (("mcbride", "phi_mcbride"), ("tng", "phi_tng")):
        phi = np.array([float(row[key]) for row in rows], dtype=float)
        positive = phi > 0.0
        ax.plot(centers[positive], phi[positive], marker=markers[backend], color=colors[backend], label=backend)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax.set_title(r"HMF weighted UVLF")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ratio = np.array([float(row["phi_ratio_tng_over_mcbride"]) for row in rows], dtype=float)
    neff_mc = np.array([float(row["effective_counts_mcbride"]) for row in rows], dtype=float)
    neff_tng = np.array([float(row["effective_counts_tng"]) for row in rows], dtype=float)
    finite = np.isfinite(ratio) & (neff_mc >= 20.0) & (neff_tng >= 20.0)
    ax.axhline(1.0, color="0.25", lw=1.0, ls=":")
    ax.plot(centers[finite], ratio[finite], marker="o", color="black")
    ax.invert_xaxis()
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\phi_{\rm TNG}/\phi_{\rm McBride}$")
    if np.any(finite):
        ax.set_ylim(0.0, max(2.0, float(np.nanmax(ratio[finite])) * 1.2))

    png_path = output_dir / "tng_vs_mcbride_z11p98_uvlf.png"
    pdf_path = output_dir / "tng_vs_mcbride_z11p98_uvlf.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TNG and McBride09 UVLF at z=11.980 within available TNG mass range.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-n-tracks", type=int, default=500)
    parser.add_argument("--hmf-n-mass", type=int, default=120)
    parser.add_argument("--hmf-n-tracks", type=int, default=120)
    parser.add_argument("--min-candidates", type=int, default=20)
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
    if not TNG_CACHE.exists():
        raise FileNotFoundError(f"TNG cache not found: {TNG_CACHE}")
    fixed_rows = run_fixed_mass_scan(
        cosmology=cosmology,
        n_tracks=int(args.fixed_n_tracks),
        min_candidates=int(args.min_candidates),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
    )
    summary_rows, results = run_uvlf(
        cosmology=cosmology,
        n_mass=int(args.hmf_n_mass),
        n_tracks=int(args.hmf_n_tracks),
        min_candidates=int(args.min_candidates),
        mass_bin_width_dex=float(args.mass_bin_width_dex),
    )
    bin_rows = build_bin_rows(results)
    fixed_path = output_dir / "fixed_mass_summary.csv"
    summary_path = output_dir / "uvlf_integrated_summary.csv"
    bin_path = output_dir / "uvlf_bin_summary.csv"
    _write_rows(fixed_path, fixed_rows)
    _write_rows(summary_path, summary_rows)
    _write_rows(bin_path, bin_rows)
    png_path, pdf_path = make_plot(output_dir, fixed_rows, bin_rows)
    print(f"saved_fixed_summary={fixed_path}")
    print(f"saved_integrated_summary={summary_path}")
    print(f"saved_bin_summary={bin_path}")
    print(f"saved_plot_png={png_path}")
    print(f"saved_plot_pdf={pdf_path}")


if __name__ == "__main__":
    main()
