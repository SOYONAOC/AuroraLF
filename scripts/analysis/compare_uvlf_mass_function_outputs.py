#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf.hmf_sampling import AB_ZEROPOINT_LNU
from auroralf.io.analysis import (
    HMF_CONFIG_DIFFERENCES,
    load_uvlf_result,
    require_compatible_results,
    select_mode_result,
)
from auroralf.results import UVLFRunResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two UVLF HDF5 artifacts.")
    parser.add_argument("--reference-hdf5", required=True, help="Baseline UVLF HDF5 artifact.")
    parser.add_argument("--candidate-hdf5", required=True, help="Candidate UVLF HDF5 artifact.")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="hmf_reed07")
    parser.add_argument("--z-values", nargs="*", type=float, default=None)
    parser.add_argument("--modes", nargs="*", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args(argv)


def _resolve_prefix(project_root: Path, output_prefix: str | None) -> Path:
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root / "data_save" / f"uvlf_mass_function_compare_{timestamp}"
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = (project_root / prefix).resolve()
    else:
        prefix = prefix.resolve()
    return prefix.with_suffix("") if prefix.suffix else prefix


def _load_model_result(path: Path) -> UVLFRunResult:
    return load_uvlf_result(path)


def _load_z_values(reference: UVLFRunResult, candidate: UVLFRunResult) -> list[float]:
    require_compatible_results(
        reference,
        candidate,
        allowed_config_differences=HMF_CONFIG_DIFFERENCES,
        context="mass-function comparison",
    )
    return list(reference.config.redshifts)


def _load_modes(reference: UVLFRunResult, candidate: UVLFRunResult) -> list[str]:
    require_compatible_results(
        reference,
        candidate,
        allowed_config_differences=HMF_CONFIG_DIFFERENCES,
        context="mass-function comparison",
    )
    return list(reference.config.stellar_population.imf_modes)


def _uv_luminosity_density(phi: np.ndarray, centers: np.ndarray, bin_width: np.ndarray) -> float:
    luminosity = np.power(10.0, (AB_ZEROPOINT_LNU - centers) / 2.5)
    valid = np.isfinite(phi) & np.isfinite(luminosity) & np.isfinite(bin_width)
    return float(np.sum(phi[valid] * luminosity[valid] * bin_width[valid]))


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "outputs"
    data_save_dir = project_root / "data_save"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_save_dir.mkdir(parents=True, exist_ok=True)

    prefix = _resolve_prefix(project_root, args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    summary_path = outputs_dir / f"{prefix.name}.txt"

    reference_path = Path(args.reference_hdf5).expanduser().resolve()
    candidate_path = Path(args.candidate_hdf5).expanduser().resolve()
    reference = _load_model_result(reference_path)
    candidate = _load_model_result(candidate_path)
    require_compatible_results(
        reference,
        candidate,
        allowed_config_differences=HMF_CONFIG_DIFFERENCES,
        context="mass-function comparison",
    )
    z_values = _load_z_values(reference, candidate) if args.z_values is None else [float(z) for z in args.z_values]
    modes = _load_modes(reference, candidate) if args.modes is None else [str(mode) for mode in args.modes]

    rows: list[dict[str, float | str]] = []
    summary_lines = [
        f"reference_hdf5: {reference_path}",
        f"candidate_hdf5: {candidate_path}",
        f"reference_label: {args.reference_label}",
        f"candidate_label: {args.candidate_label}",
        f"csv_path: {csv_path}",
        "",
    ]

    for z_obs in z_values:
        summary_lines.append(f"z={z_obs:g}")
        for mode in modes:
            reference_series = select_mode_result(reference, redshift=z_obs, mode=mode)
            candidate_series = select_mode_result(candidate, redshift=z_obs, mode=mode)
            centers = reference_series.bin_centers_muv
            bin_width = reference_series.bin_width_mag
            ref_phi = reference_series.phi_observed_per_mpc3_per_mag
            cand_phi = candidate_series.phi_observed_per_mpc3_per_mag
            ratio = np.divide(
                cand_phi,
                ref_phi,
                out=np.full_like(cand_phi, np.nan, dtype=float),
                where=ref_phi > 0.0,
            )
            delta_dex = np.log10(
                ratio,
                out=np.full_like(ratio, np.nan),
                where=ratio > 0.0,
            )
            overlap = np.isfinite(ratio) & (ratio > 0.0)
            if not np.any(overlap):
                raise RuntimeError(
                    f"no overlapping positive UVLF bins for z={z_obs:g}, mode={mode}"
                )

            ref_rho_uv = _uv_luminosity_density(ref_phi, centers, bin_width)
            cand_rho_uv = _uv_luminosity_density(cand_phi, centers, bin_width)
            rho_uv_ratio = cand_rho_uv / ref_rho_uv if ref_rho_uv > 0.0 else np.nan
            summary_lines.append(
                "  "
                f"{mode}: median_ratio={float(np.nanmedian(ratio[overlap])):.6g}, "
                f"min_ratio={float(np.nanmin(ratio[overlap])):.6g}, "
                f"max_ratio={float(np.nanmax(ratio[overlap])):.6g}, "
                f"rho_uv_ratio={rho_uv_ratio:.6g}"
            )

            for center, width, ref_value, cand_value, ratio_value, dex_value in zip(
                centers,
                bin_width,
                ref_phi,
                cand_phi,
                ratio,
                delta_dex,
                strict=True,
            ):
                rows.append(
                    {
                        "z": float(z_obs),
                        "mode": mode,
                        "Muv_center": float(center),
                        "bin_width": float(width),
                        f"phi_{args.reference_label}": float(ref_value),
                        f"phi_{args.candidate_label}": float(cand_value),
                        f"ratio_{args.candidate_label}_over_{args.reference_label}": float(ratio_value),
                        "delta_dex": float(dex_value),
                    }
                )
        summary_lines.append("")

    fieldnames = [
        "z",
        "mode",
        "Muv_center",
        "bin_width",
        f"phi_{args.reference_label}",
        f"phi_{args.candidate_label}",
        f"ratio_{args.candidate_label}_over_{args.reference_label}",
        "delta_dex",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_csv={csv_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
