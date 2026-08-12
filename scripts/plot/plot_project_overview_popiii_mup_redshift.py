#!/usr/bin/env python3
"""Plot fixed-Mup Pop III UVLF diagnostics at z=6, 10, and 12.5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET = (
    PROJECT_ROOT
    / "slides"
    / "auroralf_project_overview"
    / "assets"
    / "popiii_mup1e10_redshift.pdf"
)
DEFAULT_PREVIEW = PROJECT_ROOT / "outputs" / "popiii_mup1e10_redshift_preview.png"
DEFAULT_METRICS = PROJECT_ROOT / "data_save" / "popiii_mup1e10_redshift_offsets.csv"
SOURCE_PATHS = {
    6.0: PROJECT_ROOT / "data_save" / "uvlf_z6_popiii_mup1e10_highstat.npz",
    10.0: PROJECT_ROOT / "data_save" / "uvlf_z10_popiii_mup1e10_highstat.npz",
    12.5: PROJECT_ROOT / "data_save" / "uvlf_z12p5_popiii_mup1e10_highstat.npz",
}
EXPECTED_N_MASS = {6.0: 5062, 10.0: 5283, 12.5: 5383}
EXPECTED_BASE_SEED = {6.0: 42, 10.0: 1042, 12.5: 2042}
OBSERVATION_PATHS = {
    6.0: (
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_6"
        / "Finkelstein_uvlf_z6.npz",
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_6"
        / "bouwens21_uvlf_z6.npz",
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_6"
        / "bowler_uvlf_z6.npz",
    ),
    10.0: (
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_10"
        / "donnan24.npz",
    ),
    12.5: (
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_12p5"
        / "bouwens.npz",
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_12p5"
        / "donnan24.npz",
        PROJECT_ROOT
        / "external_data"
        / "observations"
        / "uvlf"
        / "redshift_12p5"
        / "harikane23_uvlf_z12.npz",
    ),
}
COMPONENTS = (
    ("popii", "Pop II", "#1F3A5F", "--", 2.4),
    ("total", "Pop II + mean Pop III", "#029E73", "-.", 2.4),
    ("total_burst", "Pop II + burst Pop III", "#DE8F05", "-", 3.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
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


def _scalar(payload: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in payload.files:
        raise KeyError(f"source NPZ is missing required key {key!r}")
    values = np.asarray(payload[key])
    if values.shape != (1,):
        raise ValueError(f"source NPZ key {key!r} must have shape (1,), got {values.shape}")
    return values[0]


def _load_observations(z_obs: float) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for path in OBSERVATION_PATHS[z_obs]:
        source = _resolve_existing(path)
        with np.load(source, allow_pickle=False) as payload:
            required = ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up", "label")
            missing = [key for key in required if key not in payload.files]
            if missing:
                raise KeyError(f"observation file {source} is missing fields {missing}")
            muv = np.asarray(payload["muverr"], dtype=float)
            phi = np.asarray(payload["phierr"], dtype=float)
            mag_err = np.asarray(payload["mag_err"], dtype=float)
            phi_err_lo = np.asarray(payload["phi_err_lo"], dtype=float)
            phi_err_up = np.asarray(payload["phi_err_up"], dtype=float)
            upper_limit = (
                np.asarray(payload["is_upper_limit"], dtype=bool)
                if "is_upper_limit" in payload.files
                else np.zeros_like(phi, dtype=bool)
            )
            shapes = {
                array.shape
                for array in (muv, phi, mag_err, phi_err_lo, phi_err_up, upper_limit)
            }
            if len(shapes) != 1:
                raise ValueError(f"observation file {source} has inconsistent shapes: {shapes}")
            finite = np.isfinite(muv) & np.isfinite(phi) & np.isfinite(mag_err)
            finite &= np.isfinite(phi_err_lo) & np.isfinite(phi_err_up)
            if not np.all(finite) or np.any(phi <= 0.0):
                raise ValueError(f"observation file {source} contains invalid values")
            observations.append(
                {
                    "label": str(np.asarray(payload["label"]).reshape(-1)[0]),
                    "muv": muv,
                    "phi": phi,
                    "mag_err": mag_err,
                    "phi_err_lo": phi_err_lo,
                    "phi_err_up": phi_err_up,
                    "upper_limit": upper_limit,
                }
            )
    return observations


def _load_model(z_obs: float) -> dict[str, object]:
    source = _resolve_existing(SOURCE_PATHS[z_obs])
    with np.load(source, allow_pickle=False) as payload:
        expected = {
            "z": z_obs,
            "N_mass": EXPECTED_N_MASS[z_obs],
            "n_tracks": 1000,
            "n_grid": 240,
            "base_seed": EXPECTED_BASE_SEED[z_obs],
            "smooth_sigma_mag": 0.6,
            "popiii_burst_sigma_mag": 2.0,
            "popiii_burst_quadrature_order": 31,
            "plot_min_raw_counts": 10,
            "lw_background_j21": 0.0,
            "hmf_dlog10m": 0.02,
            "epsilon_0": 0.12,
            "fstar_characteristic_mass": 10.0**11.7,
            "fstar_beta": 0.66,
            "fstar_gamma": 0.65,
        }
        for key, expected_value in expected.items():
            actual = float(_scalar(payload, key))
            if actual != float(expected_value):
                raise ValueError(f"{source} has {key}={actual}, expected {expected_value}")
        if not bool(_scalar(payload, "apply_dust")):
            raise ValueError(f"{source} must contain dust-attenuated model curves")
        if not bool(_scalar(payload, "enable_time_delay")):
            raise ValueError(f"{source} must use the calibrated time-delay SFR model")
        if str(_scalar(payload, "mass_function_model")) != "hmf_reed07":
            raise ValueError(f"{source} must use the Reed07 HMF")
        if bool(_scalar(payload, "samples_included")):
            raise ValueError(f"{source} must omit per-halo samples from the reusable summary")
        if str(_scalar(payload, "fixed_mup1e10_upper_mass_mode")) != "fixed":
            raise ValueError(f"{source} must use fixed Pop III upper-mass mode")
        upper_mass = float(_scalar(payload, "fixed_mup1e10_upper_mass_msun"))
        if not np.isclose(upper_mass, 1.0e10, rtol=0.0, atol=1.0):
            raise ValueError(f"{source} has M_up={upper_mass}, expected 1e10 Msun")

        centers = np.asarray(payload["bin_centers"], dtype=float)
        if centers.ndim != 1 or centers.size < 3 or np.any(np.diff(centers) <= 0.0):
            raise ValueError(f"{source} bin_centers must be a strictly increasing 1D array")
        curves: dict[str, np.ndarray] = {}
        for component_key, *_ in COMPONENTS:
            key = f"fixed_mup1e10_phi_plot_{component_key}"
            if key not in payload.files:
                raise KeyError(f"{source} is missing required curve {key!r}")
            curve = np.asarray(payload[key], dtype=float)
            if curve.shape != centers.shape:
                raise ValueError(f"{source} curve {key!r} has shape {curve.shape}")
            curves[component_key] = curve
    return {"source": source, "centers": centers, "curves": curves}


def _interpolate_log_phi(x: np.ndarray, y: np.ndarray, muv: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("fewer than three positive model bins are available")
    return np.power(
        10.0,
        np.interp(muv, x[valid], np.log10(y[valid]), left=np.nan, right=np.nan),
    )


def main() -> None:
    args = _parse_args()
    redshifts = (6.0, 10.0, 12.5)
    models = {z_obs: _load_model(z_obs) for z_obs in redshifts}
    observations = {z_obs: _load_observations(z_obs) for z_obs in redshifts}

    metric_rows: list[dict[str, object]] = []
    median_boosts: dict[float, float] = {}
    for z_obs in redshifts:
        centers = np.asarray(models[z_obs]["centers"], dtype=float)
        curves = models[z_obs]["curves"]
        popii = np.asarray(curves["popii"], dtype=float)
        burst = np.asarray(curves["total_burst"], dtype=float)
        boost_values: list[float] = []
        for observation in observations[z_obs]:
            muv = np.asarray(observation["muv"], dtype=float)
            phi_obs = np.asarray(observation["phi"], dtype=float)
            upper_limit = np.asarray(observation["upper_limit"], dtype=bool)
            phi_popii = _interpolate_log_phi(centers, popii, muv)
            phi_burst = _interpolate_log_phi(centers, burst, muv)
            boost = np.log10(phi_burst / phi_popii)
            residual = np.log10(phi_burst / phi_obs)
            for index in range(muv.size):
                metric_rows.append(
                    {
                        "z": z_obs,
                        "dataset": str(observation["label"]),
                        "Muv": float(muv[index]),
                        "phi_obs": float(phi_obs[index]),
                        "phi_popii": float(phi_popii[index]),
                        "phi_popii_plus_burst_popiii": float(phi_burst[index]),
                        "boost_log10_total_burst_over_popii": float(boost[index]),
                        "residual_log10_model_over_obs": float(residual[index]),
                        "is_upper_limit": int(upper_limit[index]),
                    }
                )
                if not upper_limit[index] and np.isfinite(boost[index]):
                    boost_values.append(float(boost[index]))
        if not boost_values:
            raise RuntimeError(f"no finite Pop III boost values at z={z_obs:g}")
        median_boosts[z_obs] = float(np.median(boost_values))

    plt.style.use("apj")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.55))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.25, wspace=0.19)
    observation_markers = ("o", "s", "D")
    for axis, z_obs in zip(axes, redshifts, strict=True):
        centers = np.asarray(models[z_obs]["centers"], dtype=float)
        curves = models[z_obs]["curves"]
        all_obs_muv = np.concatenate(
            [np.asarray(obs["muv"], dtype=float) for obs in observations[z_obs]]
        )
        all_obs_mag_err = np.concatenate(
            [np.asarray(obs["mag_err"], dtype=float) for obs in observations[z_obs]]
        )
        x_min = float(np.min(all_obs_muv - all_obs_mag_err) - 0.35)
        x_max = float(np.max(all_obs_muv + all_obs_mag_err) + 0.35)
        y_values: list[np.ndarray] = []
        for component_key, label, color, linestyle, linewidth in COMPONENTS:
            curve = np.asarray(curves[component_key], dtype=float)
            in_view = (
                np.isfinite(curve)
                & (curve > 0.0)
                & (centers >= x_min)
                & (centers <= x_max)
            )
            if np.count_nonzero(in_view) < 3:
                raise RuntimeError(
                    f"fewer than three {component_key} bins lie in the z={z_obs:g} view"
                )
            axis.plot(
                centers[in_view],
                curve[in_view],
                color=color,
                ls=linestyle,
                lw=linewidth,
                label=label,
                zorder=4,
            )
            y_values.append(curve[in_view])

        for obs_index, observation in enumerate(observations[z_obs]):
            phi_obs = np.asarray(observation["phi"], dtype=float)
            phi_low = phi_obs - np.asarray(observation["phi_err_lo"], dtype=float)
            phi_high = phi_obs + np.asarray(observation["phi_err_up"], dtype=float)
            y_values.extend([phi_obs, phi_low[phi_low > 0.0], phi_high])
            axis.errorbar(
                np.asarray(observation["muv"], dtype=float),
                phi_obs,
                xerr=np.asarray(observation["mag_err"], dtype=float),
                yerr=[
                    np.asarray(observation["phi_err_lo"], dtype=float),
                    np.asarray(observation["phi_err_up"], dtype=float),
                ],
                uplims=np.asarray(observation["upper_limit"], dtype=bool),
                fmt=observation_markers[obs_index % len(observation_markers)],
                ms=5.8,
                color="0.25",
                markerfacecolor="white",
                markeredgecolor="0.2",
                markeredgewidth=0.7,
                capsize=2.6,
                elinewidth=0.95,
                linestyle="none",
                label="observations" if obs_index == 0 else "_nolegend_",
                zorder=8,
            )

        positive = np.concatenate(y_values)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size == 0:
            raise RuntimeError(f"no positive y values at z={z_obs:g}")
        axis.set_yscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(float(np.min(positive) * 0.55), float(np.max(positive) * 1.8))
        axis.set_title(rf"$z={z_obs:g}$", color="#1F3A5F", fontsize=14, pad=7.0)
        axis.set_xlabel(r"$M_{\rm UV}$")
        axis.tick_params(axis="both", which="major", labelsize=9.5)
        axis.grid(True, which="major", color="#CBD3DD", lw=0.7, alpha=0.70)
        axis.grid(True, which="minor", color="#E5E9EF", lw=0.4, alpha=0.45)
    axes[0].set_ylabel(r"$\Phi_{\rm obs}$ [Mpc$^{-3}$ mag$^{-1}$]", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [
        labels.index("Pop II"),
        labels.index("Pop II + mean Pop III"),
        labels.index("Pop II + burst Pop III"),
        labels.index("observations"),
    ]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=10.0,
        columnspacing=1.4,
        handlelength=2.5,
    )

    asset = _resolve_output(args.asset)
    preview = _resolve_output(args.preview)
    metrics = _resolve_output(args.metrics)
    asset.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset, dpi=500, bbox_inches="tight")
    fig.savefig(preview, dpi=500, bbox_inches="tight")
    plt.close(fig)

    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    for z_obs in redshifts:
        print(f"z={z_obs:g}:median_popiii_boost_dex={median_boosts[z_obs]:+.6f}")
        print(f"z={z_obs:g}:source={models[z_obs]['source']}")
    print(f"saved_asset={asset}")
    print(f"saved_preview={preview}")
    print(f"saved_metrics={metrics}")


if __name__ == "__main__":
    main()
