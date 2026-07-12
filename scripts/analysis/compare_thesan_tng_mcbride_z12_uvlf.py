#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/auroralf_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from auroralf.mah import Cosmology
from auroralf.mah.models import KM_PER_MPC, SECONDS_PER_GYR
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf, uv_luminosity_to_muv
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thesan_tng_z12_uvlf_compare"
MASS_BIN_WIDTH_DEX = 0.20
MIN_CANDIDATES = 5
FIXED_N_TRACKS = 240
HMF_N_MASS = 80
HMF_N_TRACKS = 80
LOGM_GRID = (9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5)
HMF_LOGM_MIN = 9.0
HMF_LOGM_MAX = 10.5
Z_START_MAX = 20.0
N_GRID = 80


@dataclass(frozen=True)
class BackendSpec:
    label: str
    backend: str
    z_final: float
    cache_path: Path | None
    color: str
    marker: str


THESAN = BackendSpec(
    label="THESAN-dark-1",
    backend="thesan",
    z_final=11.881653785705566,
    cache_path=PROJECT_ROOT
    / "data_save/thesan_mah_cache/thesan-dark-1_LHaloTree_allchunks_z11p882_n371_smoke.hdf5",
    color="#009E73",
    marker="o",
)
TNG = BackendSpec(
    label="TNG100-1-Dark",
    backend="tng",
    z_final=11.9802133153003,
    cache_path=PROJECT_ROOT / "data_save/tng_mah_cache/TNG100-1-Dark_sublink_mpb_z11p980_n3448.hdf5",
    color="#D55E00",
    marker="s",
)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.15,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def _percentiles(values: np.ndarray, percentiles: tuple[float, ...] = (16.0, 50.0, 84.0)) -> dict[str, float]:
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


def _pipeline_kwargs(spec: BackendSpec) -> dict[str, Any]:
    if spec.backend == "mcbride":
        return {"mah_backend": "mcbride"}
    if spec.backend == "thesan":
        if spec.cache_path is None or not spec.cache_path.exists():
            raise FileNotFoundError(f"THESAN cache not found: {spec.cache_path}")
        return {
            "mah_backend": "thesan",
            "thesan_mah_cache_path": spec.cache_path,
            "thesan_mass_bin_width_dex": MASS_BIN_WIDTH_DEX,
            "thesan_min_candidates": MIN_CANDIDATES,
            "thesan_time_grid_mode": "uniform_in_t",
        }
    if spec.backend == "tng":
        if spec.cache_path is None or not spec.cache_path.exists():
            raise FileNotFoundError(f"TNG cache not found: {spec.cache_path}")
        return {
            "mah_backend": "tng",
            "tng_mah_cache_path": spec.cache_path,
            "tng_mass_bin_width_dex": MASS_BIN_WIDTH_DEX,
            "tng_min_candidates": MIN_CANDIDATES,
            "tng_time_grid_mode": "uniform_in_t",
        }
    raise RuntimeError(f"unsupported backend spec: {spec.backend}")


def run_fixed_mass_scan(*, cosmology: Cosmology) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sim_index, sim_spec in enumerate((THESAN, TNG)):
        mc_spec = BackendSpec(
            label=f"McBride09 matched to {sim_spec.label}",
            backend="mcbride",
            z_final=sim_spec.z_final,
            cache_path=None,
            color="#0072B2",
            marker="^",
        )
        for mass_index, logm in enumerate(LOGM_GRID):
            mh_final = float(10.0**float(logm))
            for spec in (mc_spec, sim_spec):
                result = run_halo_uv_pipeline(
                    n_tracks=FIXED_N_TRACKS,
                    z_final=spec.z_final,
                    Mh_final=mh_final,
                    cosmology=cosmology,
                    random_seeds=derive_pipeline_random_seeds(
                        101_000 + 1000 * sim_index,
                        redshift=spec.z_final,
                        mass_index=mass_index,
                    ),
                    z_start_max=Z_START_MAX,
                    n_grid=N_GRID,
                    enable_time_delay=True,
                    workers=1,
                    **_pipeline_kwargs(spec),
                )
                muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
                row: dict[str, Any] = {
                    "comparison": sim_spec.backend,
                    "backend": spec.backend,
                    "label": spec.label,
                    "z_final": float(spec.z_final),
                    "logM_final": float(logm),
                    "n_tracks": FIXED_N_TRACKS,
                    "candidate_count": result.metadata.get(f"{spec.backend}_candidate_count", ""),
                    "time_grid_mode": result.metadata["time_grid_mode"],
                }
                row.update({f"muv_{key}": value for key, value in _percentiles(muv).items()})
                rows.append(row)
            print(f"fixed_done comparison={sim_spec.backend} logM={logm:.2f}", flush=True)
    return rows


def _run_hmf_for_spec(spec: BackendSpec, *, seed: int, cosmology: Cosmology) -> Any:
    bins = np.arange(-22.0, -13.0 + 0.5, 0.5)
    return sample_uvlf_from_hmf(
        z_obs=spec.z_final,
        cosmology=cosmology,
        N_mass=HMF_N_MASS,
        n_tracks=HMF_N_TRACKS,
        base_seed=seed,
        bins=bins,
        logM_min=HMF_LOGM_MIN,
        logM_max=HMF_LOGM_MAX,
        z_start_max=Z_START_MAX,
        n_grid=N_GRID,
        enable_time_delay=True,
        pipeline_workers=1,
        **_pipeline_kwargs(spec),
    )


def run_hmf_uvlf(
    *,
    cosmology: Cosmology,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    integrated_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for sim_index, sim_spec in enumerate((THESAN, TNG)):
        mc_spec = BackendSpec(
            label=f"McBride09 matched to {sim_spec.label}",
            backend="mcbride",
            z_final=sim_spec.z_final,
            cache_path=None,
            color="#0072B2",
            marker="^",
        )
        for backend_index, spec in enumerate((mc_spec, sim_spec)):
            key = f"{sim_spec.backend}_{spec.backend}"
            result = _run_hmf_for_spec(
                spec,
                seed=111_000 + 1000 * sim_index + 500 * backend_index,
                cosmology=cosmology,
            )
            results[key] = result
            samples_muv = np.asarray(result.samples["Muv"], dtype=float)
            sample_weight = np.asarray(result.samples["sample_weight"], dtype=float)
            finite = np.isfinite(samples_muv) & np.isfinite(sample_weight)
            row: dict[str, Any] = {
                "comparison": sim_spec.backend,
                "backend": spec.backend,
                "label": spec.label,
                "z_final": float(spec.z_final),
                "N_mass": HMF_N_MASS,
                "n_tracks": HMF_N_TRACKS,
                "logM_min": HMF_LOGM_MIN,
                "logM_max": HMF_LOGM_MAX,
                "weighted_density_muv_lt_20": float(np.sum(sample_weight[finite & (samples_muv < -20.0)])),
                "weighted_density_muv_lt_19": float(np.sum(sample_weight[finite & (samples_muv < -19.0)])),
                "weighted_density_muv_lt_18": float(np.sum(sample_weight[finite & (samples_muv < -18.0)])),
                "weighted_density_muv_lt_17": float(np.sum(sample_weight[finite & (samples_muv < -17.0)])),
                "sampling_seconds": float(result.metadata["sampling_seconds"]),
                "nonzero_phi_bins": int(np.count_nonzero(np.asarray(result.uvlf["phi"], dtype=float) > 0.0)),
            }
            row.update({f"sample_muv_{name}": value for name, value in _percentiles(samples_muv).items()})
            integrated_rows.append(row)
            centers = np.asarray(result.uvlf["bin_centers"], dtype=float)
            phi = np.asarray(result.uvlf["phi"], dtype=float)
            phi_sigma = np.asarray(result.uvlf["phi_sigma"], dtype=float)
            raw_counts = np.asarray(result.uvlf["raw_counts"], dtype=np.int64)
            effective_counts = np.asarray(result.uvlf["effective_counts"], dtype=float)
            for index, center in enumerate(centers):
                bin_rows.append(
                    {
                        "comparison": sim_spec.backend,
                        "backend": spec.backend,
                        "label": spec.label,
                        "z_final": float(spec.z_final),
                        "Muv_center": float(center),
                        "phi": float(phi[index]),
                        "phi_sigma": float(phi_sigma[index]),
                        "raw_counts": int(raw_counts[index]),
                        "effective_counts": float(effective_counts[index]),
                    }
                )
            print(f"hmf_done comparison={sim_spec.backend} backend={spec.backend}", flush=True)
    return integrated_rows, bin_rows, results


def build_ratio_rows(bin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in ("thesan", "tng"):
        sim_rows = [row for row in bin_rows if row["comparison"] == comparison and row["backend"] == comparison]
        mc_rows = [row for row in bin_rows if row["comparison"] == comparison and row["backend"] == "mcbride"]
        if len(sim_rows) != len(mc_rows):
            raise RuntimeError(f"mismatched UVLF bins for {comparison}")
        for sim_row, mc_row in zip(sim_rows, mc_rows, strict=True):
            if not np.isclose(float(sim_row["Muv_center"]), float(mc_row["Muv_center"])):
                raise RuntimeError(f"mismatched Muv bin centers for {comparison}")
            mc_phi = float(mc_row["phi"])
            sim_phi = float(sim_row["phi"])
            ratio = np.nan if mc_phi <= 0.0 else sim_phi / mc_phi
            rows.append(
                {
                    "comparison": comparison,
                    "z_final": float(sim_row["z_final"]),
                    "Muv_center": float(sim_row["Muv_center"]),
                    "phi_sim": sim_phi,
                    "phi_mcbride": mc_phi,
                    "phi_ratio_sim_over_mcbride": float(ratio),
                    "effective_counts_sim": float(sim_row["effective_counts"]),
                    "effective_counts_mcbride": float(mc_row["effective_counts"]),
                    "raw_counts_sim": int(sim_row["raw_counts"]),
                    "raw_counts_mcbride": int(mc_row["raw_counts"]),
                }
            )
    return rows


def make_plot(fixed_rows: list[dict[str, Any]], bin_rows: list[dict[str, Any]], ratio_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    _set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), constrained_layout=True)
    specs = {"thesan": THESAN, "tng": TNG}

    ax = axes[0]
    for comparison, spec in specs.items():
        rows = [row for row in fixed_rows if row["comparison"] == comparison and row["backend"] == comparison]
        mc_rows = [row for row in fixed_rows if row["comparison"] == comparison and row["backend"] == "mcbride"]
        logm = np.array([float(row["logM_final"]) for row in rows], dtype=float)
        sim_p50 = np.array([float(row["muv_p50"]) for row in rows], dtype=float)
        mc_p50 = np.array([float(row["muv_p50"]) for row in mc_rows], dtype=float)
        ax.plot(logm, sim_p50, marker=spec.marker, color=spec.color, label=spec.label)
        ax.plot(logm, mc_p50, ls="--", color=spec.color, alpha=0.70, label=f"McBride at {comparison} z")
    ax.axhline(-20.0, color="0.2", ls=":", lw=1.0)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
    ax.set_ylabel(r"median $M_{\rm UV}$")
    ax.set_title("Fixed final mass")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for comparison, spec in specs.items():
        rows = [row for row in bin_rows if row["comparison"] == comparison and row["backend"] == comparison]
        centers = np.array([float(row["Muv_center"]) for row in rows], dtype=float)
        phi = np.array([float(row["phi"]) for row in rows], dtype=float)
        positive = phi > 0.0
        ax.plot(centers[positive], phi[positive], marker=spec.marker, color=spec.color, label=spec.label)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax.set_title("HMF-weighted UVLF")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for comparison, spec in specs.items():
        rows = [row for row in ratio_rows if row["comparison"] == comparison]
        centers = np.array([float(row["Muv_center"]) for row in rows], dtype=float)
        ratio = np.array([float(row["phi_ratio_sim_over_mcbride"]) for row in rows], dtype=float)
        neff_sim = np.array([float(row["effective_counts_sim"]) for row in rows], dtype=float)
        neff_mc = np.array([float(row["effective_counts_mcbride"]) for row in rows], dtype=float)
        valid = np.isfinite(ratio) & (neff_sim >= 5.0) & (neff_mc >= 5.0)
        ax.plot(centers[valid], ratio[valid], marker=spec.marker, color=spec.color, label=spec.label)
    ax.axhline(1.0, color="0.2", ls=":", lw=1.0)
    ax.invert_xaxis()
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\phi_{\rm sim}/\phi_{\rm McBride}$")
    ax.set_title("Matched-redshift ratio")
    ax.legend(frameon=False, fontsize=8)

    png_path = OUTPUT_DIR / "thesan_tng_mcbride_z12_uvlf_compare.png"
    pdf_path = OUTPUT_DIR / "thesan_tng_mcbride_z12_uvlf_compare.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    cosmology = Cosmology(
        h0=67.74 * SECONDS_PER_GYR / KM_PER_MPC,
        omega_m=0.3089,
        omega_b=0.0486,
        omega_lambda=0.6911,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_rows = run_fixed_mass_scan(cosmology=cosmology)
    integrated_rows, bin_rows, _ = run_hmf_uvlf(cosmology=cosmology)
    ratio_rows = build_ratio_rows(bin_rows)
    fixed_path = OUTPUT_DIR / "fixed_mass_muv_summary.csv"
    integrated_path = OUTPUT_DIR / "uvlf_integrated_summary.csv"
    bin_path = OUTPUT_DIR / "uvlf_bin_summary.csv"
    ratio_path = OUTPUT_DIR / "uvlf_ratio_summary.csv"
    _write_rows(fixed_path, fixed_rows)
    _write_rows(integrated_path, integrated_rows)
    _write_rows(bin_path, bin_rows)
    _write_rows(ratio_path, ratio_rows)
    png_path, pdf_path = make_plot(fixed_rows, bin_rows, ratio_rows)
    print(f"saved_fixed_summary={fixed_path}")
    print(f"saved_integrated_summary={integrated_path}")
    print(f"saved_bin_summary={bin_path}")
    print(f"saved_ratio_summary={ratio_path}")
    print(f"saved_plot_png={png_path}")
    print(f"saved_plot_pdf={pdf_path}")


if __name__ == "__main__":
    main()
