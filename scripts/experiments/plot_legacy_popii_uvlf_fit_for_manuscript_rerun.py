#!/usr/bin/env python3
"""Replot the May 2026 full canonical Pop II UVLF legacy NPZ for slide review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plot.plot_popii_uvlf_fit_check_z6_z12p5 import (
    OBSERVATION_COLORS,
    OBSERVATION_FILES,
    OBSERVATION_MARKERS,
    _load_observation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data_save"
    / "uvlf_current_full_noburst_z6_z10_z12p5_20260522.npz"
)


def _scalar(payload: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in payload.files:
        raise KeyError(f"legacy NPZ is missing required key {key!r}")
    values = np.asarray(payload[key])
    if values.shape != (1,):
        raise ValueError(f"legacy NPZ key {key!r} must have shape (1,)")
    return values[0]


def _model(payload: np.lib.npyio.NpzFile, redshift: float) -> dict[str, np.ndarray]:
    tag = f"z{str(float(redshift)).replace('.', 'p')}"
    keys = {
        "centers": f"{tag}_bin_centers",
        "phi": f"{tag}_canonical_phi",
        "raw_counts": f"{tag}_canonical_raw_counts",
    }
    missing = [key for key in keys.values() if key not in payload.files]
    if missing:
        raise KeyError(f"legacy NPZ is missing required keys: {missing}")
    centers = np.asarray(payload[keys["centers"]], dtype=float)
    phi = np.asarray(payload[keys["phi"]], dtype=float)
    raw_counts = np.asarray(payload[keys["raw_counts"]])
    if centers.shape != (20,) or phi.shape != centers.shape or raw_counts.shape != centers.shape:
        raise ValueError(f"unexpected canonical UVLF array shapes at z={redshift:g}")
    valid = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0) & (raw_counts >= 10)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(f"fewer than three canonical bins pass raw_counts >= 10 at z={redshift:g}")
    order = np.argsort(centers[valid])
    return {
        "centers": centers[valid][order],
        "phi": phi[valid][order],
    }


def _interp_log_phi(model: dict[str, np.ndarray], magnitude: np.ndarray) -> np.ndarray:
    centers = model["centers"]
    phi = model["phi"]
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("model magnitude centers must be strictly increasing")
    return np.power(
        10.0,
        np.interp(magnitude, centers, np.log10(phi), left=np.nan, right=np.nan),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    source = args.npz.expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as payload:
        if not np.array_equal(np.asarray(payload["z_values"], dtype=float), [6.0, 10.0, 12.5]):
            raise ValueError("legacy NPZ redshift grid must be exactly [6, 10, 12.5]")
        expected_scalars = {
            "N_mass": 3000,
            "n_tracks": 1000,
            "n_grid": 240,
            "bins_count": 20,
            "random_seed": 42,
        }
        for key, expected in expected_scalars.items():
            actual = int(_scalar(payload, key))
            if actual != expected:
                raise ValueError(f"legacy NPZ {key}={actual}, expected {expected}")
        if not bool(_scalar(payload, "apply_dust")):
            raise ValueError("legacy NPZ must contain the dust-attenuated production run")
        models = {redshift: _model(payload, redshift) for redshift in (6.0, 12.5)}

    plt.style.use("apj")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 6.2),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    rows: list[dict[str, object]] = []
    for column, redshift in enumerate((6.0, 12.5)):
        model = models[redshift]
        observations = [_load_observation(path) for path in OBSERVATION_FILES[redshift]]
        upper = axes[0, column]
        lower = axes[1, column]
        upper.plot(model["centers"], model["phi"], color="#1F5C8B", lw=2.4, label="canonical Pop II")
        lower.axhline(1.0, color="0.25", lw=1.0, ls=":")
        positive_phi = [model["phi"]]
        positive_ratio: list[np.ndarray] = []
        for observation in observations:
            label = str(observation["label"])
            color = OBSERVATION_COLORS[label]
            marker = OBSERVATION_MARKERS[label]
            magnitude = np.asarray(observation["muv"], dtype=float)
            phi_obs = np.asarray(observation["phi"], dtype=float)
            upper_limit = np.asarray(observation["upper_limit"], dtype=bool)
            upper.errorbar(
                magnitude,
                phi_obs,
                xerr=np.asarray(observation["mag_err"], dtype=float),
                yerr=[observation["phi_err_lo"], observation["phi_err_up"]],
                uplims=upper_limit,
                fmt=marker,
                ms=5.6,
                color=color,
                markeredgecolor="0.15",
                markeredgewidth=0.5,
                capsize=2.4,
                elinewidth=0.9,
                linestyle="none",
                label=label,
            )
            model_at_obs = _interp_log_phi(model, magnitude)
            ratio = model_at_obs / phi_obs
            finite = np.isfinite(ratio) & (ratio > 0.0)
            lower.plot(magnitude[finite], ratio[finite], marker, color=color, ms=5.0, linestyle="none")
            positive_phi.append(phi_obs)
            positive_ratio.append(ratio[finite])
            for index in range(magnitude.size):
                rows.append(
                    {
                        "z": redshift,
                        "dataset": label,
                        "Muv": float(magnitude[index]),
                        "phi_obs": float(phi_obs[index]),
                        "phi_model": float(model_at_obs[index]),
                        "model_over_obs": float(ratio[index]),
                        "is_upper_limit": int(upper_limit[index]),
                    }
                )
        y_values = np.concatenate(positive_phi)
        y_values = y_values[np.isfinite(y_values) & (y_values > 0.0)]
        ratios = np.concatenate(positive_ratio)
        upper.set_yscale("log")
        upper.set_xlim(-24.5, -15.0)
        upper.set_ylim(max(1.0e-13, float(np.min(y_values)) * 0.35), float(np.max(y_values)) * 3.0)
        upper.set_title(rf"$z={redshift:g}$")
        upper.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ mag$^{-1}$]")
        upper.legend(frameon=True, fontsize=8, loc="lower right")
        lower.set_yscale("log")
        lower.set_xlim(-24.5, -15.0)
        lower.set_ylim(max(0.03, float(np.min(ratios)) * 0.45), min(100.0, float(np.max(ratios)) * 2.2))
        lower.set_xlabel(r"$M_{\rm UV}$")
        lower.set_ylabel("model/obs")
        for axis in (upper, lower):
            axis.grid(True, which="major", color="#CBD3DD", lw=0.7, alpha=0.85)
            axis.grid(True, which="minor", color="#E5E9EF", lw=0.45, alpha=0.75)

    output = args.output.expanduser().resolve()
    metrics = args.metrics.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500)
    plt.close(fig)
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"source_npz={source}")
    print(f"wrote_pdf={output}")
    print(f"wrote_metrics={metrics}")
    for redshift in (6.0, 12.5):
        values = np.asarray(
            [
                np.log10(float(row["model_over_obs"]))
                for row in rows
                if row["z"] == redshift
                and row["is_upper_limit"] == 0
                and np.isfinite(float(row["model_over_obs"]))
                and float(row["model_over_obs"]) > 0.0
            ]
        )
        print(f"z={redshift:g} median_delta_log10_model_over_obs={np.median(values):+.6f}")


if __name__ == "__main__":
    main()
