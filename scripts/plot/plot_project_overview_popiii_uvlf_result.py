#!/usr/bin/env python3
"""Plot the current z=14.5 Pop III UVLF diagnostic for the project overview."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.uvlf import compute_dust_attenuated_uvlf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data_save" / "uvlf_z14p5_popiii_mup_highstat.npz"
DEFAULT_ASSET = (
    PROJECT_ROOT
    / "slides"
    / "auroralf_project_overview"
    / "assets"
    / "popiii_uvlf_result.pdf"
)
DEFAULT_PREVIEW = PROJECT_ROOT / "outputs" / "popiii_uvlf_result_preview.png"
DEFAULT_METRICS = PROJECT_ROOT / "data_save" / "popiii_uvlf_result_offsets.csv"
OBSERVATION_PATHS = (
    PROJECT_ROOT
    / "external_data"
    / "observations"
    / "uvlf"
    / "redshift_14"
    / "whitler25_jades_z14p3.npz",
    PROJECT_ROOT
    / "external_data"
    / "observations"
    / "uvlf"
    / "redshift_15"
    / "donnan24_primer_z14p5.npz",
)
SCENARIOS = (
    ("current", r"Atomic cap: $M_{\rm up}=4.8\times10^7\,M_\odot$"),
    ("fixed_mup1e10", r"Extended occupation: $M_{\rm up}=10^{10}\,M_\odot$"),
)
COMPONENTS = (
    ("popii", "Pop II", "#1F3A5F", "--", 2.5),
    ("total", "Pop II + mean Pop III", "#029E73", "-.", 2.5),
    ("total_burst", "Pop II + burst Pop III", "#DE8F05", "-", 3.1),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def _resolve_existing(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve(strict=True)


def _resolve_output(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _scalar(payload: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in payload.files:
        raise KeyError(f"source NPZ is missing required key {key!r}")
    values = np.asarray(payload[key])
    if values.shape != (1,):
        raise ValueError(f"source NPZ key {key!r} must have shape (1,), got {values.shape}")
    return values[0]


def _load_observations() -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for path in OBSERVATION_PATHS:
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
            if not np.all(np.isfinite(muv)) or not np.all(np.isfinite(phi)) or np.any(phi <= 0.0):
                raise ValueError(f"observation file {source} contains invalid MUV or phi values")
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


def _smooth_curve(
    centers: np.ndarray,
    intrinsic_phi: np.ndarray,
    raw_counts: np.ndarray,
    *,
    z_obs: float,
    sigma_mag: float,
    min_raw_counts: int,
) -> tuple[np.ndarray, np.ndarray]:
    dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=centers,
        intrinsic_phi=intrinsic_phi,
        z=z_obs,
        muv_obs=centers,
    )
    phi = np.asarray(dust["phi_obs"], dtype=float)
    valid = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0)
    valid &= np.asarray(raw_counts, dtype=int) >= int(min_raw_counts)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("fewer than three model bins pass the plotting cuts")
    x = np.asarray(centers[valid], dtype=float)
    y = np.asarray(phi[valid], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    distances = x[:, np.newaxis] - x[np.newaxis, :]
    kernel = np.exp(-0.5 * np.square(distances / float(sigma_mag)))
    normalization = kernel.sum(axis=1)
    if np.any(normalization <= 0.0) or not np.all(np.isfinite(normalization)):
        raise RuntimeError("invalid Gaussian smoothing normalization")
    smoothed = np.power(10.0, kernel @ np.log10(y) / normalization)
    if not np.all(np.isfinite(smoothed)) or np.any(smoothed <= 0.0):
        raise RuntimeError("smoothed UVLF contains invalid values")
    return x, smoothed


def _interpolate_log_phi(x: np.ndarray, y: np.ndarray, muv: np.ndarray) -> np.ndarray:
    return np.power(
        10.0,
        np.interp(muv, x, np.log10(y), left=np.nan, right=np.nan),
    )


def main() -> None:
    args = _parse_args()
    source = _resolve_existing(args.source)
    observations = _load_observations()
    with np.load(source, allow_pickle=False) as payload:
        required_scalars = {
            "z": 14.5,
            "N_mass": 5451,
            "n_tracks": 1000,
            "n_grid": 240,
            "base_seed": 14501,
            "logM_max": 13.0,
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
        for key, expected in required_scalars.items():
            actual = float(_scalar(payload, key))
            if actual != float(expected):
                raise ValueError(f"source NPZ {key}={actual}, expected {expected}")
        if not bool(_scalar(payload, "enable_time_delay")):
            raise ValueError("source NPZ must use the calibrated time-delay SFR model")
        if str(_scalar(payload, "mass_function_model")) != "hmf_reed07":
            raise ValueError("source NPZ must use the Reed07 HMF")
        if bool(_scalar(payload, "samples_included")):
            raise ValueError("source NPZ must omit per-halo arrays from the reusable summary")
        if not bool(_scalar(payload, "apply_dust")):
            raise ValueError("source NPZ must record the dust-attenuated comparison mode")
        if str(_scalar(payload, "current_upper_mass_mode")) != "atomic":
            raise ValueError("current scenario must use the atomic Pop III upper-mass mode")
        if str(_scalar(payload, "fixed_mup1e10_upper_mass_mode")) != "fixed":
            raise ValueError("extended scenario must use the fixed Pop III upper-mass mode")
        if not np.isclose(
            float(_scalar(payload, "fixed_mup1e10_upper_mass_msun")),
            1.0e10,
            rtol=0.0,
            atol=1.0,
        ):
            raise ValueError("extended scenario must use M_up=1e10 Msun")

        centers = np.asarray(payload["bin_centers"], dtype=float)
        if centers.ndim != 1 or centers.size < 3 or np.any(np.diff(centers) <= 0.0):
            raise ValueError("source NPZ bin_centers must be a strictly increasing 1D array")
        sigma_mag = float(_scalar(payload, "smooth_sigma_mag"))
        min_raw_counts = int(_scalar(payload, "plot_min_raw_counts"))
        curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for scenario_key, _ in SCENARIOS:
            for component_key, *_ in COMPONENTS:
                phi_key = f"{scenario_key}_phi_{component_key}"
                count_key = f"{scenario_key}_count_{component_key}"
                if phi_key not in payload.files or count_key not in payload.files:
                    raise KeyError(f"source NPZ is missing {phi_key!r} or {count_key!r}")
                curves[(scenario_key, component_key)] = _smooth_curve(
                    centers,
                    np.asarray(payload[phi_key], dtype=float),
                    np.asarray(payload[count_key], dtype=int),
                    z_obs=14.5,
                    sigma_mag=sigma_mag,
                    min_raw_counts=min_raw_counts,
                )

    all_muv = np.concatenate([np.asarray(obs["muv"], dtype=float) for obs in observations])
    all_mag_err = np.concatenate(
        [np.asarray(obs["mag_err"], dtype=float) for obs in observations]
    )
    x_min = float(np.min(all_muv - all_mag_err) - 0.20)
    x_max = float(np.max(all_muv + all_mag_err) + 0.20)
    observation_phi = np.concatenate(
        [np.asarray(obs["phi"], dtype=float) for obs in observations]
    )
    observation_high = np.concatenate(
        [
            np.asarray(obs["phi"], dtype=float) + np.asarray(obs["phi_err_up"], dtype=float)
            for obs in observations
        ]
    )
    comparison_values: list[np.ndarray] = []
    for x, y in curves.values():
        in_view = (x >= x_min) & (x <= x_max)
        if not np.any(in_view):
            raise RuntimeError("a model curve has no bins in the observation window")
        comparison_values.append(y[in_view])
    model_positive = np.concatenate(comparison_values)
    y_min = max(float(np.min(model_positive) * 0.65), float(np.min(observation_phi) * 1.0e-3))
    y_max = float(np.max(observation_high) * 1.7)
    if not 0.0 < y_min < y_max:
        raise RuntimeError(f"invalid plot limits: y_min={y_min}, y_max={y_max}")

    metric_rows: list[dict[str, object]] = []
    median_offsets: dict[tuple[str, str], float] = {}
    for scenario_key, _ in SCENARIOS:
        for component_key, *_ in COMPONENTS:
            x, y = curves[(scenario_key, component_key)]
            offsets: list[float] = []
            for obs in observations:
                muv = np.asarray(obs["muv"], dtype=float)
                phi_obs = np.asarray(obs["phi"], dtype=float)
                upper_limit = np.asarray(obs["upper_limit"], dtype=bool)
                phi_model = _interpolate_log_phi(x, y, muv)
                delta = np.log10(phi_model / phi_obs)
                for index in range(muv.size):
                    metric_rows.append(
                        {
                            "scenario": scenario_key,
                            "component": component_key,
                            "dataset": str(obs["label"]),
                            "Muv": float(muv[index]),
                            "phi_obs": float(phi_obs[index]),
                            "phi_model": float(phi_model[index]),
                            "delta_log10_model_over_obs": float(delta[index]),
                            "is_upper_limit": int(upper_limit[index]),
                        }
                    )
                    if not upper_limit[index] and np.isfinite(delta[index]):
                        offsets.append(float(delta[index]))
            if not offsets:
                raise RuntimeError(f"no finite observed offsets for {scenario_key}:{component_key}")
            median_offsets[(scenario_key, component_key)] = float(np.median(offsets))

    plt.style.use("apj")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.45), sharex=True, sharey=True)
    for axis, (scenario_key, title) in zip(axes, SCENARIOS, strict=True):
        for component_key, label, color, linestyle, linewidth in COMPONENTS:
            x, y = curves[(scenario_key, component_key)]
            in_view = (x >= x_min) & (x <= x_max)
            axis.plot(
                x[in_view],
                y[in_view],
                color=color,
                ls=linestyle,
                lw=linewidth,
                label=label,
                zorder=4,
            )
        for obs in observations:
            label = str(obs["label"])
            marker = "D" if label == "JADES" else "o"
            color = "#7A4EAB" if label == "JADES" else "#2A9D8F"
            axis.errorbar(
                np.asarray(obs["muv"], dtype=float),
                np.asarray(obs["phi"], dtype=float),
                xerr=np.asarray(obs["mag_err"], dtype=float),
                yerr=[
                    np.asarray(obs["phi_err_lo"], dtype=float),
                    np.asarray(obs["phi_err_up"], dtype=float),
                ],
                uplims=np.asarray(obs["upper_limit"], dtype=bool),
                fmt=marker,
                ms=7.4,
                color=color,
                markeredgecolor="0.15",
                markeredgewidth=0.6,
                capsize=3.2,
                elinewidth=1.15,
                linestyle="none",
                label=label,
                zorder=8,
            )
        axis.set_title(title, fontsize=14, color="#1F3A5F", pad=7.0)
        axis.set_yscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel(r"$M_{\rm UV}$")
        axis.tick_params(axis="both", which="major", labelsize=10.5)
        axis.grid(True, which="major", color="#CBD3DD", lw=0.7, alpha=0.70)
        axis.grid(True, which="minor", color="#E5E9EF", lw=0.4, alpha=0.45)
    axes[0].set_ylabel(r"$\Phi_{\rm obs}$ [Mpc$^{-3}$ mag$^{-1}$]", fontsize=12.5)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [
        labels.index("Pop II"),
        labels.index("Pop II + mean Pop III"),
        labels.index("Pop II + burst Pop III"),
        labels.index("JADES"),
        labels.index("PRIMER"),
    ]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=10.2,
        columnspacing=1.25,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), w_pad=2.2)

    asset = _resolve_output(args.asset)
    preview = _resolve_output(args.preview)
    metrics = _resolve_output(args.metrics)
    asset.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset, dpi=500)
    fig.savefig(preview, dpi=500)
    plt.close(fig)

    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"source={source}")
    print(f"saved_asset={asset}")
    print(f"saved_preview={preview}")
    print(f"saved_metrics={metrics}")
    for scenario_key, _ in SCENARIOS:
        for component_key, *_ in COMPONENTS:
            print(
                f"{scenario_key}:{component_key}:"
                f"median_delta_dex={median_offsets[(scenario_key, component_key)]:+.6f}"
            )


if __name__ == "__main__":
    main()
