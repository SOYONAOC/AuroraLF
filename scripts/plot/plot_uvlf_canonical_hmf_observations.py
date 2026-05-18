#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf import MASS_FUNCTION_MODEL_HMF_REED07
from auroralf.uvlf.imf import IMF_MODE_CANONICAL


DEFAULT_Z_VALUES = (6.0, 12.5, 14.5)
MODEL_LABELS = {
    MASS_FUNCTION_MODEL_HMF_REED07: "hmf Reed07 canonical",
}
MODEL_COLORS = {
    MASS_FUNCTION_MODEL_HMF_REED07: "black",
}
MODEL_LINESTYLES = {
    MASS_FUNCTION_MODEL_HMF_REED07: "-",
}
OBS_UVLF_DIR = Path("external_data/observations/uvlf")
OBS_FILES = {
    6.0: (
        "redshift_6/Finkelstein_uvlf_z6.npz",
        "redshift_6/bouwens21_uvlf_z6.npz",
        "redshift_6/bowler_uvlf_z6.npz",
    ),
    12.5: (
        "redshift_12p5/bouwens.npz",
        "redshift_12p5/donnan24.npz",
        "redshift_12p5/harikane23_uvlf_z12.npz",
    ),
    14.5: (
        "redshift_15/donnan24_primer_z14p5.npz",
        "redshift_14/whitler25_jades_z14p3.npz",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the canonical Reed07 UVLF against observations.")
    parser.add_argument("--reed07-npz", required=True)
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_reed07_canonical_vs_observations_z6_z12p5_z14p5",
    )
    return parser.parse_args()


def _z_tag(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    if z_value not in OBS_FILES:
        raise ValueError(f"no observational UVLF files configured for z={z_value:g}")
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_name in OBS_FILES[z_value]:
        file_path = _resolve_path(str(OBS_UVLF_DIR / file_name))
        with np.load(file_path, allow_pickle=False) as payload:
            label_array = np.asarray(payload["label"]).astype(str)
            label = str(label_array[0]) if label_array.size > 0 else file_path.stem
            is_upper_limit = (
                np.asarray(payload["is_upper_limit"], dtype=bool)
                if "is_upper_limit" in payload.files
                else np.zeros_like(np.asarray(payload["phierr"], dtype=float), dtype=bool)
            )
            datasets.append(
                {
                    "label": label,
                    "Muv": np.asarray(payload["muverr"], dtype=float),
                    "phi": np.asarray(payload["phierr"], dtype=float),
                    "mag_err": np.asarray(payload["mag_err"], dtype=float),
                    "phi_err_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                    "phi_err_up": np.asarray(payload["phi_err_up"], dtype=float),
                    "is_upper_limit": is_upper_limit,
                }
            )
    return datasets


def _require_array(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(f"NPZ is missing required key: {key}")
    return np.asarray(npz[key])


def _load_model_series(
    npz: np.lib.npyio.NpzFile,
    *,
    z_obs: float,
) -> tuple[np.ndarray, np.ndarray]:
    tag = _z_tag(z_obs)
    centers = np.asarray(_require_array(npz, f"{tag}_bin_centers"), dtype=float)
    phi = np.asarray(_require_array(npz, f"{tag}_{IMF_MODE_CANONICAL}_phi"), dtype=float)
    if centers.shape != phi.shape:
        raise ValueError(f"bin centers and canonical phi shapes differ for z={z_obs:g}")
    return centers, phi


def _interp_model_at_obs(centers: np.ndarray, phi: np.ndarray, obs_muv: np.ndarray) -> np.ndarray:
    valid = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("not enough positive model UVLF bins for interpolation")
    order = np.argsort(centers[valid])
    x = centers[valid][order]
    log_y = np.log(phi[valid][order])
    result = np.full_like(obs_muv, np.nan, dtype=float)
    inside = np.isfinite(obs_muv) & (obs_muv >= x[0]) & (obs_muv <= x[-1])
    result[inside] = np.exp(np.interp(obs_muv[inside], x, log_y))
    return result


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _set_log_ylim(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(np.asarray(item, dtype=float)) for item in values])
    ax.set_ylim(max(1.0e-9, float(np.min(positive)) * 0.4), float(np.max(positive)) * 3.0)


def main() -> None:
    args = _parse_args()
    model_paths = {
        MASS_FUNCTION_MODEL_HMF_REED07: _resolve_path(args.reed07_npz),
    }
    z_values = [float(z) for z in args.z_values]
    output_prefix = Path(args.output_prefix).expanduser()
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.txt")

    model_data = {
        model: np.load(path, allow_pickle=False)
        for model, path in model_paths.items()
    }
    try:
        apply_dust_values = {
            bool(np.asarray(_require_array(data, "apply_dust"))[0])
            for data in model_data.values()
        }
        if len(apply_dust_values) != 1:
            raise ValueError("model NPZ files do not agree on apply_dust")
        apply_dust = apply_dust_values.pop()

        plt.style.use("apj")
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 7.2,
            }
        )
        fig, axes = plt.subplots(
            1,
            len(z_values),
            figsize=(4.15 * len(z_values), 4.1),
            sharey=True,
        )
        fig.subplots_adjust(left=0.075, right=0.992, bottom=0.17, top=0.82, wspace=0.04)
        axes = np.atleast_1d(axes)
        obs_markers = ["o", "s", "^", "D", "P", "X"]
        summary_lines = [
            f"reed07_npz: {model_paths[MASS_FUNCTION_MODEL_HMF_REED07]}",
            f"apply_dust: {apply_dust}",
            "",
        ]

        for column, z_obs in enumerate(z_values):
            ax = axes[column]
            ylim_values: list[np.ndarray] = []
            model_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for model, data in model_data.items():
                centers, phi = _load_model_series(data, z_obs=z_obs)
                model_series[model] = (centers, phi)
                valid = np.isfinite(phi) & (phi > 0.0)
                ylim_values.append(phi[valid])
                ax.plot(
                    centers[valid],
                    phi[valid],
                    color=MODEL_COLORS[model],
                    ls=MODEL_LINESTYLES[model],
                    lw=2.1,
                    label=MODEL_LABELS[model],
                    zorder=2,
                )

            obs_sets = _load_observational_uvlf(z_obs)
            summary_lines.append(f"z={z_obs:g}")
            summary_lines.append(
                "  observations="
                + "; ".join(str(obs["label"]) for obs in obs_sets)
            )
            all_obs_muv: list[np.ndarray] = []
            all_obs_phi: list[np.ndarray] = []
            all_obs_upper: list[np.ndarray] = []
            for obs_index, obs in enumerate(obs_sets):
                marker = obs_markers[obs_index % len(obs_markers)]
                muv = np.asarray(obs["Muv"], dtype=float)
                phi = np.asarray(obs["phi"], dtype=float)
                mag_err = np.asarray(obs["mag_err"], dtype=float)
                phi_err_lo = np.asarray(obs["phi_err_lo"], dtype=float)
                phi_err_up = np.asarray(obs["phi_err_up"], dtype=float)
                is_upper_limit = np.asarray(obs["is_upper_limit"], dtype=bool)
                valid = np.isfinite(muv) & np.isfinite(phi) & (phi > 0.0)
                if not np.any(valid):
                    continue
                ylim_values.append(phi[valid])
                all_obs_muv.append(muv[valid])
                all_obs_phi.append(phi[valid])
                all_obs_upper.append(is_upper_limit[valid])
                ax.errorbar(
                    muv[valid],
                    phi[valid],
                    xerr=mag_err[valid],
                    yerr=np.vstack([phi_err_lo[valid], phi_err_up[valid]]),
                    uplims=is_upper_limit[valid],
                    fmt=marker,
                    ms=5.5,
                    color="0.25",
                    mec="white",
                    mew=0.6,
                    elinewidth=1.0,
                    capsize=2.0,
                    alpha=0.9,
                    label="_nolegend_",
                    zorder=3,
                )

            obs_muv = np.concatenate(all_obs_muv)
            obs_phi = np.concatenate(all_obs_phi)
            obs_upper = np.concatenate(all_obs_upper)
            detected = ~obs_upper
            for model, (centers, phi_model) in model_series.items():
                model_at_obs = _interp_model_at_obs(centers, phi_model, obs_muv)
                ratio = np.divide(
                    model_at_obs,
                    obs_phi,
                    out=np.full_like(model_at_obs, np.nan),
                    where=obs_phi > 0.0,
                )
                overlap = detected & np.isfinite(ratio) & (ratio > 0.0)
                if not np.any(overlap):
                    raise RuntimeError(f"no overlapping detections for z={z_obs:g}, model={model}")
                summary_lines.append(
                    "  "
                    f"{model}_canonical_obs_ratio_median={float(np.nanmedian(ratio[overlap])):.6g}"
                )
                summary_lines.append(
                    "  "
                    f"{model}_canonical_obs_ratio_min={float(np.nanmin(ratio[overlap])):.6g}"
                )
                summary_lines.append(
                    "  "
                    f"{model}_canonical_obs_ratio_max={float(np.nanmax(ratio[overlap])):.6g}"
                )
            summary_lines.append("")

            ax.set_yscale("log")
            ax.set_xlim(-24.5, -15.0)
            _set_log_ylim(ax, ylim_values)
            ax.grid(alpha=0.22)
            ax.set_title(rf"$z={z_obs:g}$")
            ax.set_xlabel(r"$M_{\rm UV}$")
            if column == 0:
                ax.set_ylabel(r"$\phi(M_{\rm UV})$ [mag$^{-1}$ Mpc$^{-3}$]")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                ls=MODEL_LINESTYLES[model],
                lw=2.2,
                label=MODEL_LABELS[model],
            )
            for model in model_data
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.25",
                lw=0.0,
                markerfacecolor="0.25",
                markeredgecolor="white",
                markersize=5.5,
                label="observations",
            )
        )
        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=min(4, len(legend_handles)),
            fontsize=7.4,
        )

        fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    finally:
        for data in model_data.values():
            data.close()

    print(f"saved_png={output_prefix.with_suffix('.png')}", flush=True)
    print(f"saved_pdf={output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
