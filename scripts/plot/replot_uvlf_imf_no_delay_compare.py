#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.uvlf import compute_dust_attenuated_uvlf
from auroralf.uvlf.imf import IMF_MODE_CANONICAL


MODE_LABELS = {
    "canonical": "canonical Pop II",
    "z10_mild_topheavy": r"$z_{\rm src}\geq10$ mild top-heavy",
    "mah_burst_mild_topheavy": "MAH-burst mild top-heavy",
}
MODE_COLORS = {
    "canonical": "black",
    "z10_mild_topheavy": "#c44e52",
    "mah_burst_mild_topheavy": "#1f77b4",
}
OBS_MUV_MIN = -22.5
OBS_MUV_MAX = -16.0
OBS_UVLF_DIR = PROJECT_ROOT / "external_data" / "observations" / "uvlf"
OBS_FILES = {
    6.0: (
        "redshift_6/Finkelstein_uvlf_z6.npz",
        "redshift_6/bouwens21_uvlf_z6.npz",
        "redshift_6/bowler_uvlf_z6.npz",
    ),
    8.0: (
        "redshift_8/bowler_uvlf_z8.npz",
        "redshift_8/donnan_uvlf_z8.npz",
        "redshift_8/mclure_uvlf_z8.npz",
    ),
    9.0: (
        "redshift_9/bowler20.npz",
        "redshift_9/donnan23.npz",
        "redshift_9/donnan24.npz",
    ),
    10.0: (
        "redshift_10/donnan24.npz",
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
    parser = argparse.ArgumentParser(
        description="Plot UVLF comparisons for canonical and mild top-heavy Pop II IMF modes."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data_save/uvlf_imf_mode_compare_allz_latest.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_imf_mode_compare_allz_latest",
    )
    return parser.parse_args()


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _z_tag(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _obs_paths_from_z(z_value: float) -> tuple[Path, ...]:
    if z_value in OBS_FILES:
        return tuple(OBS_UVLF_DIR / item for item in OBS_FILES[z_value])
    obs_dir = OBS_UVLF_DIR / f"redshift_{str(float(z_value)).replace('.', 'p').rstrip('0').rstrip('p')}"
    return tuple(sorted(obs_dir.glob("*.npz")))


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_path in _obs_paths_from_z(z_value):
        payload = np.load(file_path)
        label_array = np.asarray(payload["label"])
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


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode.replace("_", " "))


def _mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "0.35")


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _set_log_ylim_from_values(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(np.asarray(item, dtype=float)) for item in values])
    y_min = float(np.nanmin(positive))
    y_max = float(np.nanmax(positive))
    ax.set_ylim(max(1.0e-10, y_min * 0.35), y_max * 2.5)


def _set_ratio_ylim_from_values(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positive = np.concatenate([_finite_positive(np.asarray(item, dtype=float)) for item in values])
    y_min = float(np.nanmin(positive))
    y_max = float(np.nanmax(positive))
    if 0.95 <= y_min and y_max <= 1.05:
        ax.set_ylim(0.95, 1.05)
    else:
        ax.set_ylim(max(0.7, y_min * 0.82), max(1.25, y_max * 1.16))


def _mode_phi_for_plot(
    data: np.lib.npyio.NpzFile,
    *,
    tag: str,
    mode: str,
    z_obs: float,
    apply_dust: bool,
) -> tuple[np.ndarray, np.ndarray]:
    archived_phi = np.asarray(data[f"{tag}_{mode}_phi"], dtype=float)
    if not apply_dust:
        return archived_phi, archived_phi

    intrinsic_phi = np.asarray(data[f"{tag}_{mode}_intrinsic_phi"], dtype=float)
    centers = np.asarray(data[f"{tag}_bin_centers"], dtype=float)
    dust = compute_dust_attenuated_uvlf(centers, intrinsic_phi, float(z_obs))
    computed_phi = np.asarray(dust["phi_obs"], dtype=float)
    if not np.allclose(archived_phi, computed_phi, rtol=1.0e-12, atol=1.0e-30, equal_nan=True):
        max_abs = float(np.nanmax(np.abs(archived_phi - computed_phi)))
        raise RuntimeError(
            f"Archived dust phi differs from POPIII-style dust recomputation for {tag}/{mode}; "
            f"max_abs_diff={max_abs:.6e}. Regenerate the NPZ before plotting."
        )
    return intrinsic_phi, computed_phi


def main() -> None:
    args = _parse_args()
    npz_path = _resolve_project_path(args.npz_path)
    output_prefix = _resolve_project_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
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
    z_values = [float(z) for z in np.asarray(data["z_values"], dtype=float)]
    mode_names = [str(mode) for mode in np.asarray(data["mode_names"])]
    variant_modes = [mode for mode in mode_names if mode != IMF_MODE_CANONICAL]
    canonical_ssp_file = str(np.asarray(data["canonical_ssp_file"])[0])
    topheavy_ssp_file = str(np.asarray(data["topheavy_ssp_file"])[0])
    topheavy_ssp_metallicity = float(np.asarray(data["topheavy_ssp_metallicity"])[0])
    apply_dust = bool(np.asarray(data["apply_dust"])[0])
    phi_label = (
        r"$\phi_{\rm dust}(M_{\rm UV})$ [mag$^{-1}$ Mpc$^{-3}$]"
        if apply_dust
        else r"$\phi(M_{\rm UV})$ [mag$^{-1}$ Mpc$^{-3}$]"
    )
    uvlf_label = "dust-attenuated UVLF" if apply_dust else "intrinsic UVLF"

    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(4.15 * len(z_values), 5.6),
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    summary_lines = [
        f"npz_path: {npz_path}",
        f"mass_function_model: {str(np.asarray(data['mass_function_model'])[0]) if 'mass_function_model' in data.files else 'unknown'}",
        f"epsilon_0: {float(np.asarray(data['epsilon_0'])[0]) if 'epsilon_0' in data.files else float('nan'):.12g}",
        f"canonical_ssp_file: {canonical_ssp_file}",
        f"topheavy_ssp_file: {topheavy_ssp_file}",
        f"topheavy_ssp_metallicity: {topheavy_ssp_metallicity:g}",
        f"apply_dust: {apply_dust}",
        f"mode_names: {' '.join(mode_names)}",
        f"ratio_display_range: {OBS_MUV_MIN:g} <= Muv <= {OBS_MUV_MAX:g}",
        "",
    ]
    obs_markers = ["o", "s", "^", "D", "P", "X"]

    for column, z_obs in enumerate(z_values):
        tag = _z_tag(z_obs)
        centers = np.asarray(data[f"{tag}_bin_centers"], dtype=float)
        _, phi_canonical = _mode_phi_for_plot(
            data,
            tag=tag,
            mode=IMF_MODE_CANONICAL,
            z_obs=z_obs,
            apply_dust=apply_dust,
        )

        ax_top = axes[0, column]
        top_ylim_values = []
        valid_canonical = np.isfinite(phi_canonical) & (phi_canonical > 0.0)
        top_ylim_values.append(phi_canonical[valid_canonical])
        for mode in variant_modes:
            _, phi_mode = _mode_phi_for_plot(
                data,
                tag=tag,
                mode=mode,
                z_obs=z_obs,
                apply_dust=apply_dust,
            )
            valid = np.isfinite(phi_mode) & (phi_mode > 0.0)
            top_ylim_values.append(phi_mode[valid])
            ax_top.plot(
                centers[valid],
                phi_mode[valid],
                color=_mode_color(mode),
                lw=1.8,
                alpha=0.88,
                zorder=2,
                label=_mode_label(mode),
            )
        ax_top.plot(
            centers[valid_canonical],
            phi_canonical[valid_canonical],
            color=_mode_color(IMF_MODE_CANONICAL),
            lw=2.6,
            zorder=3,
            label=_mode_label(IMF_MODE_CANONICAL) + (" dust" if apply_dust else ""),
        )

        obs_sets = _load_observational_uvlf(z_obs)
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
            top_ylim_values.append(phi[valid])
            ax_top.errorbar(
                muv[valid],
                phi[valid],
                xerr=mag_err[valid],
                yerr=np.vstack([phi_err_lo[valid], phi_err_up[valid]]),
                uplims=is_upper_limit[valid],
                fmt=marker,
                ms=5.5,
                color="#1f4e79",
                mec="white",
                mew=0.6,
                elinewidth=1.0,
                capsize=2.0,
                alpha=0.92,
                label=str(obs["label"]),
            )
        ax_top.set_yscale("log")
        ax_top.set_xlim(-24.5, -15.0)
        _set_log_ylim_from_values(ax_top, top_ylim_values)
        ax_top.grid(alpha=0.22)
        ax_top.set_title(rf"$z={z_obs:g}$ {uvlf_label}")
        if column == 0:
            ax_top.set_ylabel(phi_label)
        ax_top.legend(frameon=False, fontsize=7.4, loc="lower left")

        ax_bottom = axes[1, column]
        ratio_ylim_values = []
        obs_range_mask = (centers >= OBS_MUV_MIN) & (centers <= OBS_MUV_MAX)
        ax_bottom.axvspan(-24.5, OBS_MUV_MIN, color="0.94", zorder=-10)
        ax_bottom.axvspan(OBS_MUV_MAX, -15.0, color="0.94", zorder=-10)
        for mode in variant_modes:
            _, phi_mode = _mode_phi_for_plot(
                data,
                tag=tag,
                mode=mode,
                z_obs=z_obs,
                apply_dust=apply_dust,
            )
            ratio = np.divide(
                phi_mode,
                phi_canonical,
                out=np.full_like(phi_mode, np.nan),
                where=phi_canonical > 0.0,
            )
            valid_ratio = np.isfinite(ratio) & (ratio > 0.0) & obs_range_mask
            ratio_ylim_values.append(ratio[valid_ratio])
            ax_bottom.plot(
                centers[valid_ratio],
                ratio[valid_ratio],
                color=_mode_color(mode),
                lw=2.0,
                label=_mode_label(mode),
            )
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_bottom.set_xlim(-24.5, -15.0)
        ax_bottom.grid(alpha=0.22)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        if column == 0:
            ax_bottom.set_ylabel("variant / canonical")
        _set_ratio_ylim_from_values(ax_bottom, ratio_ylim_values)

        summary_lines.append(f"z={z_obs:g}")
        for mode in variant_modes:
            _, phi_mode = _mode_phi_for_plot(
                data,
                tag=tag,
                mode=mode,
                z_obs=z_obs,
                apply_dust=apply_dust,
            )
            ratio = np.divide(
                phi_mode,
                phi_canonical,
                out=np.full_like(phi_mode, np.nan),
                where=phi_canonical > 0.0,
            )
            overlap = np.isfinite(ratio) & np.isfinite(phi_canonical)
            if np.any(overlap):
                summary_lines.append(f"  {mode}_ratio_median={float(np.nanmedian(ratio[overlap])):.6f}")
                summary_lines.append(f"  {mode}_ratio_min={float(np.nanmin(ratio[overlap])):.6f}")
                summary_lines.append(f"  {mode}_ratio_max={float(np.nanmax(ratio[overlap])):.6f}")
            obs_overlap = overlap & obs_range_mask
            if np.any(obs_overlap):
                summary_lines.append(
                    f"  {mode}_obs_range_ratio_median={float(np.nanmedian(ratio[obs_overlap])):.6f}"
                )
                summary_lines.append(
                    f"  {mode}_obs_range_ratio_min={float(np.nanmin(ratio[obs_overlap])):.6f}"
                )
                summary_lines.append(
                    f"  {mode}_obs_range_ratio_max={float(np.nanmax(ratio[obs_overlap])):.6f}"
                )
        summary_lines.append(f"  n_obs_sets={len(obs_sets)}")
        summary_lines.append("")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.parent / f"{output_prefix.name}_plot_summary.txt"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")


if __name__ == "__main__":
    main()
