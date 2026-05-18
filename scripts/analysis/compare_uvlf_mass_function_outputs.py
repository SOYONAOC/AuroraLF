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


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two UVLF NPZ files.")
    parser.add_argument("--reference-npz", required=True, help="Baseline UVLF NPZ.")
    parser.add_argument("--candidate-npz", required=True, help="Candidate UVLF NPZ.")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="hmf_reed07")
    parser.add_argument("--z-values", nargs="*", type=float, default=None)
    parser.add_argument("--modes", nargs="*", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


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


def _require_array(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(f"NPZ is missing required key: {key}")
    return np.asarray(npz[key])


def _load_z_values(reference: np.lib.npyio.NpzFile, candidate: np.lib.npyio.NpzFile) -> list[float]:
    reference_z = np.asarray(_require_array(reference, "z_values"), dtype=float)
    candidate_z = np.asarray(_require_array(candidate, "z_values"), dtype=float)
    if reference_z.shape != candidate_z.shape or not np.allclose(reference_z, candidate_z, rtol=0.0, atol=0.0):
        raise ValueError("reference and candidate runs do not have identical z_values arrays")
    return [float(z) for z in reference_z]


def _load_modes(reference: np.lib.npyio.NpzFile, candidate: np.lib.npyio.NpzFile) -> list[str]:
    reference_modes = np.asarray(_require_array(reference, "mode_names")).astype(str)
    candidate_modes = np.asarray(_require_array(candidate, "mode_names")).astype(str)
    if reference_modes.shape != candidate_modes.shape or not np.array_equal(reference_modes, candidate_modes):
        raise ValueError("reference and candidate runs do not have identical mode_names arrays")
    return [str(mode) for mode in reference_modes]


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

    with np.load(Path(args.reference_npz).expanduser().resolve(), allow_pickle=False) as reference:
        with np.load(Path(args.candidate_npz).expanduser().resolve(), allow_pickle=False) as candidate:
            z_values = _load_z_values(reference, candidate) if args.z_values is None else [float(z) for z in args.z_values]
            modes = _load_modes(reference, candidate) if args.modes is None else [str(mode) for mode in args.modes]

            rows: list[dict[str, float | str]] = []
            summary_lines = [
                f"reference_npz: {Path(args.reference_npz).expanduser().resolve()}",
                f"candidate_npz: {Path(args.candidate_npz).expanduser().resolve()}",
                f"reference_label: {args.reference_label}",
                f"candidate_label: {args.candidate_label}",
                f"csv_path: {csv_path}",
                "",
            ]

            for z_obs in z_values:
                z_tag = _tag_from_z(z_obs)
                centers = np.asarray(_require_array(reference, f"{z_tag}_bin_centers"), dtype=float)
                candidate_centers = np.asarray(_require_array(candidate, f"{z_tag}_bin_centers"), dtype=float)
                if not np.allclose(centers, candidate_centers, rtol=0.0, atol=0.0):
                    raise ValueError(f"bin centers differ at z={z_obs:g}")
                bin_width = np.asarray(_require_array(reference, f"{z_tag}_bin_width"), dtype=float)
                candidate_bin_width = np.asarray(_require_array(candidate, f"{z_tag}_bin_width"), dtype=float)
                if not np.allclose(bin_width, candidate_bin_width, rtol=0.0, atol=0.0):
                    raise ValueError(f"bin widths differ at z={z_obs:g}")

                summary_lines.append(f"z={z_obs:g}")
                for mode in modes:
                    ref_phi = np.asarray(_require_array(reference, f"{z_tag}_{mode}_phi"), dtype=float)
                    cand_phi = np.asarray(_require_array(candidate, f"{z_tag}_{mode}_phi"), dtype=float)
                    ratio = np.divide(
                        cand_phi,
                        ref_phi,
                        out=np.full_like(cand_phi, np.nan, dtype=float),
                        where=ref_phi > 0.0,
                    )
                    delta_dex = np.log10(ratio, out=np.full_like(ratio, np.nan), where=ratio > 0.0)
                    overlap = np.isfinite(ratio) & (ratio > 0.0)
                    if not np.any(overlap):
                        raise RuntimeError(f"no overlapping positive UVLF bins for z={z_obs:g}, mode={mode}")

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
