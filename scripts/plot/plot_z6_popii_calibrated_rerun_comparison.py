#!/usr/bin/env python3
"""Compare the frozen z=6 Pop II UVLF with an exact-parameter current rerun."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auroralf.io.analysis import load_uvlf_result, select_mode_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_NPZ = PROJECT_ROOT / "data_save" / "uvlf_current_full_noburst_z6_z10_z12p5_20260522.npz"
DEFAULT_RERUN_HDF5 = PROJECT_ROOT / "data_save" / "uvlf_z6_popii_calibrated_rerun.h5"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "z6_popii_calibrated_rerun_comparison"
DEFAULT_CSV = PROJECT_ROOT / "data_save" / "z6_popii_calibrated_rerun_comparison.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--rerun-hdf5", type=Path, default=DEFAULT_RERUN_HDF5)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def _resolve_existing(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve(strict=True)


def _resolve_output(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def main() -> None:
    args = _parse_args()
    reference_path = _resolve_existing(args.reference_npz)
    rerun_path = _resolve_existing(args.rerun_hdf5)
    output_prefix = _resolve_output(args.output_prefix)
    csv_path = _resolve_output(args.csv)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(reference_path, allow_pickle=False) as payload:
        reference_centers = np.asarray(payload["z6p0_bin_centers"], dtype=float)
        reference_observed = np.asarray(payload["z6p0_canonical_phi"], dtype=float)
        reference_intrinsic = np.asarray(payload["z6p0_canonical_intrinsic_phi"], dtype=float)
        reference_counts = np.asarray(payload["z6p0_canonical_raw_counts"], dtype=int)
        expected_reference = {
            "epsilon_0": 0.12,
            "fstar_characteristic_mass": 10.0**11.7,
            "fstar_beta": 0.66,
            "fstar_gamma": 0.65,
            "N_mass": 3000,
            "n_tracks": 1000,
            "n_grid": 240,
            "logM_min": 9.0,
            "logM_max": 13.0,
            "enable_time_delay": True,
            "apply_dust": True,
            "mass_function_model": "hmf_reed07",
            "random_seed": 42,
        }
        for key, expected in expected_reference.items():
            value = np.asarray(payload[key])
            if value.shape != (1,):
                raise ValueError(f"reference metadata {key!r} must have shape (1,)")
            actual = value[0]
            if actual != expected:
                raise ValueError(f"reference metadata mismatch for {key}: {actual!r} != {expected!r}")

    rerun = load_uvlf_result(rerun_path)
    series = select_mode_result(rerun, redshift=6.0, mode="canonical")
    rerun_centers = np.asarray(series.bin_centers_muv, dtype=float)
    rerun_observed = np.asarray(series.phi_observed_per_mpc3_per_mag, dtype=float)
    rerun_intrinsic = np.asarray(series.phi_intrinsic_per_mpc3_per_mag, dtype=float)
    rerun_counts = np.asarray(series.raw_counts, dtype=int)

    np.testing.assert_allclose(rerun_centers, reference_centers, rtol=0.0, atol=2.0e-15)
    if not (
        reference_observed.shape
        == reference_intrinsic.shape
        == reference_counts.shape
        == rerun_observed.shape
        == rerun_intrinsic.shape
        == rerun_counts.shape
    ):
        raise ValueError("reference and rerun UVLF arrays do not have matching shapes")

    valid = (
        np.isfinite(reference_observed)
        & np.isfinite(reference_intrinsic)
        & np.isfinite(rerun_observed)
        & np.isfinite(rerun_intrinsic)
        & (reference_observed > 0.0)
        & (reference_intrinsic > 0.0)
        & (rerun_observed > 0.0)
        & (rerun_intrinsic > 0.0)
        & (reference_counts >= 10)
        & (rerun_counts >= 10)
    )
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("fewer than three common positive bins pass raw_counts >= 10")

    delta_observed_dex = np.log10(rerun_observed / reference_observed)
    delta_intrinsic_dex = np.log10(rerun_intrinsic / reference_intrinsic)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Muv",
                "reference_observed_phi",
                "rerun_observed_phi",
                "delta_observed_dex",
                "reference_intrinsic_phi",
                "rerun_intrinsic_phi",
                "delta_intrinsic_dex",
                "reference_raw_counts",
                "rerun_raw_counts",
                "used",
            ],
        )
        writer.writeheader()
        for index, magnitude in enumerate(reference_centers):
            writer.writerow(
                {
                    "Muv": float(magnitude),
                    "reference_observed_phi": float(reference_observed[index]),
                    "rerun_observed_phi": float(rerun_observed[index]),
                    "delta_observed_dex": float(delta_observed_dex[index]),
                    "reference_intrinsic_phi": float(reference_intrinsic[index]),
                    "rerun_intrinsic_phi": float(rerun_intrinsic[index]),
                    "delta_intrinsic_dex": float(delta_intrinsic_dex[index]),
                    "reference_raw_counts": int(reference_counts[index]),
                    "rerun_raw_counts": int(rerun_counts[index]),
                    "used": int(valid[index]),
                }
            )

    observed_used = delta_observed_dex[valid]
    intrinsic_used = delta_intrinsic_dex[valid]
    summary_lines = [
        f"reference={reference_path}",
        f"rerun={rerun_path}",
        f"common_bins={int(np.count_nonzero(valid))}",
        f"observed_median_delta_dex={float(np.median(observed_used)):.9f}",
        f"observed_mean_abs_delta_dex={float(np.mean(np.abs(observed_used))):.9f}",
        f"observed_max_abs_delta_dex={float(np.max(np.abs(observed_used))):.9f}",
        f"intrinsic_median_delta_dex={float(np.median(intrinsic_used)):.9f}",
        f"intrinsic_mean_abs_delta_dex={float(np.mean(np.abs(intrinsic_used))):.9f}",
        f"intrinsic_max_abs_delta_dex={float(np.max(np.abs(intrinsic_used))):.9f}",
    ]
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plt.style.use("apj")
    fig, (ax_uvlf, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    ax_uvlf.plot(
        reference_centers[valid],
        reference_observed[valid],
        color="#1F3A5F",
        lw=2.7,
        label="frozen slide baseline (2026-05-22)",
    )
    ax_uvlf.plot(
        rerun_centers[valid],
        rerun_observed[valid],
        color="#DE8F05",
        lw=2.4,
        ls="--",
        label="current-code rerun (same physics parameters)",
    )
    ax_uvlf.set_yscale("log")
    ax_uvlf.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax_uvlf.set_title(r"Canonical Pop II UVLF at $z=6$")
    ax_uvlf.legend(frameon=False, loc="lower right")
    ax_uvlf.grid(alpha=0.22)

    ax_delta.axhspan(-0.05, 0.05, color="#029E73", alpha=0.10, lw=0.0)
    ax_delta.axhline(0.0, color="0.35", lw=1.1, ls=":")
    ax_delta.plot(
        reference_centers[valid],
        delta_observed_dex[valid],
        color="#DE8F05",
        lw=2.3,
        marker="o",
        ms=4.2,
        label="dust-attenuated",
    )
    ax_delta.plot(
        reference_centers[valid],
        delta_intrinsic_dex[valid],
        color="#1F3A5F",
        lw=2.0,
        ls="--",
        marker="s",
        ms=3.8,
        label="intrinsic",
    )
    ax_delta.set_xlim(-24.5, -15.0)
    ax_delta.set_xlabel(r"$M_{\rm UV}$")
    ax_delta.set_ylabel(r"$\log_{10}(\Phi_{\rm rerun}/\Phi_{\rm frozen})$")
    ax_delta.legend(frameon=False, loc="lower right", ncol=2)
    ax_delta.grid(alpha=0.22)

    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
    print("\n".join(summary_lines))
    print(f"saved_pdf={output_prefix.with_suffix('.pdf')}")
    print(f"saved_png={output_prefix.with_suffix('.png')}")
    print(f"saved_csv={csv_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
