#!/usr/bin/env python3
"""Recreate the old UVLF comparison style with the current model cache."""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from massfunc import Mass_func
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.mah.models import PLANCK18_OMEGA_B, PLANCK18_OMEGA_M
from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from auroralf.sfr.calculator import YEARS_PER_GYR
from auroralf.ssp import SSP_UV_LOOKBACK_MAX_MYR, interpolate_ssp_luminosity, load_uv1600_table
from auroralf.uvlf import compute_dust_attenuated_uvlf
from auroralf.uvlf.hmf_sampling import AB_ZEROPOINT_LNU


plt.style.use("apj")

OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_REDSHIFTS = [6, 8, 10, 12.5]
SSP_FILE = PROJECT_ROOT / "data" / "spectra-bin-imf135_300.BASEL.z002.a+00.dat"
LOGM_MIN = 9.0
LOGM_MAX = 13.0
INSTANT_UV_SCATTER_DEX = 0.0
CALIBRATION_REDSHIFT = 6.0
CALIBRATION_MUV_RANGE = (-22.5, -16.0)

SFR_PARAMS = SFRModelParameters(
    epsilon_0=0.08,
    characteristic_mass=DEFAULT_SFR_MODEL_PARAMETERS.characteristic_mass,
    beta_star=DEFAULT_SFR_MODEL_PARAMETERS.beta_star,
    gamma_star=DEFAULT_SFR_MODEL_PARAMETERS.gamma_star,
)

OBS_STYLE = {
    "Finkelstein+15": ("o", "#1f77b4"),
    "Bouwens+21": ("s", "#ff7f0e"),
    "Bowler+15": ("o", "#2ca02c"),
    "Bowler+20": ("o", "#1f77b4"),
    "Donnan+23": ("s", "#ff7f0e"),
    "McLure+13": ("o", "#2ca02c"),
    "Donnan+24": ("o", "#1f77b4"),
    "Bouwens+23, z ~ 12 - 13": ("s", "#1f77b4"),
    "Donnan+24, z ~ 12.5": ("o", "#ff7f0e"),
    "Harikane+23, z ~ 12": ("D", "#2ca02c"),
    r"Bouwens+23, $z\sim12-13$": ("s", "#1f77b4"),
    r"Donnan+24, $z\sim12.5$": ("o", "#ff7f0e"),
    r"Harikane+23, $z\sim12$": ("D", "#2ca02c"),
    "JADES": ("D", "#6a3d9a"),
    "PRIMER": ("o", "#2ca02c"),
    "JADES UL": ("v", "#6a3d9a"),
}

OBS_REDSHIFT_LABEL = {
    "JADES": r"JADES ($z\geq14$, $z_{\rm med}=14.3$)",
    "PRIMER": r"PRIMER ($13.5<z<15.5$)",
    "JADES UL": r"JADES UL ($16<z<22.5$)",
}

OBS_MARKERS = ["o", "s", "D", "^", "v", "P"]
OBS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#6a3d9a", "#d62728", "#8c564b"]


def z_tag(z_value: float | int) -> str:
    z_float = float(z_value)
    if z_float.is_integer():
        return str(int(z_float))
    return str(z_value).replace(".", "p")


def z_label(z_value: float | int) -> str:
    z_float = float(z_value)
    if z_float.is_integer():
        return str(int(z_float))
    return f"{z_float:g}"


def integrate_kuv_from_current_ssp() -> float:
    ages_myr, luv_per_msun = load_uv1600_table(SSP_FILE)
    ages_gyr = ages_myr / 1.0e3
    max_age_gyr = float(SSP_UV_LOOKBACK_MAX_MYR) / 1.0e3
    age_grid_gyr = np.linspace(0.0, max_age_gyr, 20000)
    luv_grid = interpolate_ssp_luminosity(
        age_grid_gyr,
        ssp_age_grid=ages_gyr,
        ssp_luv_grid=luv_per_msun,
    )
    return float(np.trapezoid(luv_grid, x=age_grid_gyr * YEARS_PER_GYR))


def stellar_formation_efficiency(mh: np.ndarray, epsilon_0: float) -> np.ndarray:
    ratio = np.asarray(mh, dtype=float) / SFR_PARAMS.characteristic_mass
    denominator = ratio ** (-SFR_PARAMS.beta_star) + ratio**SFR_PARAMS.gamma_star
    return 2.0 * float(epsilon_0) / denominator


def mean_accretion_rate_msun_per_yr(mh: np.ndarray, z_value: float) -> np.ndarray:
    return 24.1 * (mh / 1.0e12) ** 1.094 * (1.0 + 1.75 * z_value) * np.sqrt(
        0.315 * (1.0 + z_value) ** 3 + 0.685
    )


def interpolate_positive_log_phi(
    x_source: np.ndarray,
    phi_source: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    positive = np.isfinite(x_source) & np.isfinite(phi_source) & (phi_source > 0.0)
    result = np.full_like(x_target, np.nan, dtype=float)
    if np.count_nonzero(positive) < 2:
        return result
    result[:] = np.power(
        10.0,
        np.interp(
            x_target,
            np.asarray(x_source)[positive],
            np.log10(np.asarray(phi_source)[positive]),
            left=np.nan,
            right=np.nan,
        ),
    )
    return result


def dust_phi_interpolated_to_grid(
    x_target: np.ndarray,
    bc: np.ndarray,
    phi: np.ndarray,
    z_value: float,
) -> np.ndarray:
    dust = compute_dust_attenuated_uvlf(bc, phi, float(z_value))
    return interpolate_positive_log_phi(bc, dust["phi_obs"], x_target)


def prepare_instant_mass_grid(
    z_value: float,
    instant_epsilon_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hmf = Mass_func()
    hmf.sigma2_interpolation_set()
    hmf.dsig2dm_interpolation_set()

    log_mh = np.linspace(LOGM_MIN, LOGM_MAX, 2400)
    mh = np.power(10.0, log_mh)
    dndm = np.asarray(hmf.dndmst(mh, z_value), dtype=float)
    dndlogm = mh * np.log(10.0) * dndm

    baryon_fraction = PLANCK18_OMEGA_B / PLANCK18_OMEGA_M
    sfr = (
        baryon_fraction
        * stellar_formation_efficiency(mh, epsilon_0=instant_epsilon_0)
        * mean_accretion_rate_msun_per_yr(mh, z_value)
    )
    return log_mh, dndlogm, sfr


def instant_uvlf_from_mass_grid(
    muv_grid: np.ndarray,
    log_mh: np.ndarray,
    dndlogm: np.ndarray,
    sfr: np.ndarray,
    kuv: float,
    *,
    luminosity_scale: float = 1.0,
) -> np.ndarray:
    """Old instant curve from the HMF without rebuilding the mass grid."""

    luv_hat = float(luminosity_scale) * kuv * sfr

    log_luv_grid = (AB_ZEROPOINT_LNU - np.asarray(muv_grid, dtype=float)) / 2.5
    log_luv_hat = np.log10(np.clip(luv_hat, np.finfo(float).tiny, None))
    sigma = INSTANT_UV_SCATTER_DEX
    if sigma <= 0.0:
        muv_hat = -2.5 * log_luv_hat + AB_ZEROPOINT_LNU
        d_muv_d_logm = np.gradient(muv_hat, log_mh)
        phi_hat = dndlogm / np.abs(d_muv_d_logm)
        valid = (
            np.isfinite(muv_hat)
            & np.isfinite(phi_hat)
            & (phi_hat > 0.0)
            & (np.abs(d_muv_d_logm) > 0.0)
        )
        order = np.argsort(muv_hat[valid])
        result = np.zeros_like(muv_grid, dtype=float)
        model_muv = muv_hat[valid][order]
        model_phi = phi_hat[valid][order]
        model_muv, unique_index = np.unique(model_muv, return_index=True)
        model_phi = model_phi[unique_index]
        inside = (muv_grid >= model_muv[0]) & (muv_grid <= model_muv[-1])
        result[inside] = np.power(
            10.0,
            np.interp(muv_grid[inside], model_muv, np.log10(model_phi)),
        )
        return result

    kernel = np.exp(-0.5 * ((log_luv_grid[:, None] - log_luv_hat[None, :]) / sigma) ** 2)
    kernel /= np.sqrt(2.0 * np.pi) * sigma
    return 0.4 * np.trapezoid(kernel * dndlogm[None, :], x=log_mh, axis=1)


def instant_uvlf_old_method(
    muv_grid: np.ndarray,
    z_value: float,
    kuv: float,
    instant_epsilon_0: float,
    *,
    luminosity_scale: float = 1.0,
) -> np.ndarray:
    """Old instant-SFR mapping, using current SFR parameters and SSP-derived K_UV."""

    log_mh, dndlogm, sfr = prepare_instant_mass_grid(z_value, instant_epsilon_0)
    return instant_uvlf_from_mass_grid(
        muv_grid,
        log_mh,
        dndlogm,
        sfr,
        kuv,
        luminosity_scale=luminosity_scale,
    )


def load_current_model(z_value: float | int) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(PROJECT_ROOT / "temp_data" / f"uvlf_z{z_tag(z_value)}.npz")
    return np.asarray(payload["bin_centers"], dtype=float), np.asarray(payload["phi"], dtype=float)


def calibrate_instant_to_z6(kuv: float, instant_epsilon_0: float) -> dict[str, float]:
    z_value = CALIBRATION_REDSHIFT
    bc, phi = load_current_model(z_value)
    fit_grid = np.linspace(CALIBRATION_MUV_RANGE[0], CALIBRATION_MUV_RANGE[1], 180)
    target_phi = dust_phi_interpolated_to_grid(fit_grid, bc, phi, z_value)
    fit_mask = np.isfinite(target_phi) & (target_phi > 0.0)
    if np.count_nonzero(fit_mask) < 5:
        raise RuntimeError("not enough valid z=6 dust UVLF points for instant-SFR calibration")
    log_mh, dndlogm, sfr = prepare_instant_mass_grid(z_value, instant_epsilon_0)

    def objective(params: np.ndarray) -> float:
        (log10_luminosity_scale,) = params
        instant_phi = instant_uvlf_from_mass_grid(
            fit_grid,
            log_mh,
            dndlogm,
            sfr,
            kuv,
            luminosity_scale=10.0**float(log10_luminosity_scale),
        )
        instant_dust = compute_dust_attenuated_uvlf(fit_grid, instant_phi, z_value)["phi_obs"]
        valid = fit_mask & np.isfinite(instant_dust) & (instant_dust > 0.0)
        if np.count_nonzero(valid) < 5:
            return np.inf
        residual = np.log10(instant_dust[valid]) - np.log10(target_phi[valid])
        return float(np.mean(residual**2))

    coarse_best = (np.inf, 0.0)
    for log10_scale in np.linspace(-1.5, 1.5, 61):
        score = objective(np.array([log10_scale], dtype=float))
        if score < coarse_best[0]:
            coarse_best = (score, float(log10_scale))

    result = minimize(
        objective,
        x0=np.array([coarse_best[1]], dtype=float),
        bounds=[(-2.0, 2.0)],
        method="L-BFGS-B",
    )
    if not result.success:
        print(f"Warning: z=6 instant calibration did not fully converge: {result.message}")

    log10_scale = float(result.x[0])
    rms_dex = float(np.sqrt(objective(np.array([log10_scale], dtype=float))))
    return {
        "luminosity_scale": 10.0**log10_scale,
        "log10_luminosity_scale": log10_scale,
        "rms_dex": rms_dex,
    }


def load_observations(z_value: float | int) -> list[dict[str, np.ndarray | str]]:
    obs_dir = PROJECT_ROOT / "data" / f"redshift_{z_tag(z_value)}"
    if not obs_dir.is_dir():
        return []

    datasets = []
    for file_path in sorted(obs_dir.glob("*.npz")):
        payload = np.load(file_path, allow_pickle=True)
        label = str(np.asarray(payload["label"])[0])
        phi = np.asarray(payload["phierr"], dtype=float)
        upper_limit = (
            np.asarray(payload["is_upper_limit"], dtype=bool)
            if "is_upper_limit" in payload.files
            else np.zeros_like(phi, dtype=bool)
        )
        datasets.append(
            {
                "label": label,
                "muv": np.asarray(payload["muverr"], dtype=float),
                "muv_err": np.asarray(payload["mag_err"], dtype=float),
                "phi": phi,
                "phi_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                "phi_up": np.asarray(payload["phi_err_up"], dtype=float),
                "upper_limit": upper_limit,
                "z_note": str(np.asarray(payload["z_note"])[0]) if "z_note" in payload.files else "",
            }
        )
    return datasets


def load_observations_for_redshifts(redshifts: list[float | int]) -> list[dict[str, np.ndarray | str]]:
    datasets: list[dict[str, np.ndarray | str]] = []
    for z_value in redshifts:
        datasets.extend(load_observations(z_value))
    return datasets


def default_observation_redshifts(z_value: float | int) -> list[float | int]:
    if np.isclose(float(z_value), 14.5):
        return [14, 15]
    return [z_value]


def plot_single(
    z_value: float | int,
    kuv: float,
    instant_epsilon_0: float,
    *,
    instant_luminosity_scale: float = 1.0,
    observation_redshifts: list[float | int] | None = None,
    output_suffix: str = "",
) -> None:
    z_float = float(z_value)
    z_text = z_label(z_value)
    model_bc, model_phi = load_current_model(z_value)
    model_dust = compute_dust_attenuated_uvlf(model_bc, model_phi, z_float)

    instant_bc = np.linspace(-26.5, -11.5, 320)
    instant_phi = instant_uvlf_old_method(
        instant_bc,
        z_float,
        kuv,
        instant_epsilon_0=instant_epsilon_0,
        luminosity_scale=instant_luminosity_scale,
    )
    instant_dust = compute_dust_attenuated_uvlf(instant_bc, instant_phi, z_float)

    fig, ax = plt.subplots(figsize=(6.9, 5.0))

    model_mask = model_dust["phi_obs"] > 0.0
    ax.semilogy(
        model_bc[model_mask],
        model_dust["phi_obs"][model_mask],
        color="#c0392b",
        lw=2.0,
        label="Our model",
        zorder=5,
    )

    instant_mask = instant_dust["phi_obs"] > 0.0
    ax.semilogy(
        instant_bc[instant_mask],
        instant_dust["phi_obs"][instant_mask],
        color="#1f77ff",
        lw=2.0,
        ls="--",
        label="Instant (z=6 calib.)",
        zorder=4,
    )

    obs_redshifts = default_observation_redshifts(z_value) if observation_redshifts is None else observation_redshifts
    observations = load_observations_for_redshifts(obs_redshifts)
    for obs_index, obs in enumerate(observations):
        label = str(obs["label"])
        default_marker, default_color = OBS_STYLE.get(label, ("o", "C4"))
        marker = OBS_MARKERS[obs_index % len(OBS_MARKERS)] if len(observations) > 1 else default_marker
        color = OBS_COLORS[obs_index % len(OBS_COLORS)] if len(observations) > 1 else default_color
        legend_label = OBS_REDSHIFT_LABEL.get(label, label)
        upper_limit = np.asarray(obs["upper_limit"], dtype=bool)
        detection = ~upper_limit

        if np.any(detection):
            ax.errorbar(
                obs["muv"][detection],
                obs["phi"][detection],
                xerr=obs["muv_err"][detection],
                yerr=[obs["phi_lo"][detection], obs["phi_up"][detection]],
                fmt=marker,
                ms=7.0,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=legend_label,
                capsize=3,
                elinewidth=1.2,
                zorder=10,
            )

        if np.any(upper_limit):
            ax.errorbar(
                obs["muv"][upper_limit],
                obs["phi"][upper_limit],
                xerr=obs["muv_err"][upper_limit],
                yerr=0.35 * obs["phi"][upper_limit],
                uplims=True,
                fmt=marker,
                ms=7.0,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=legend_label if not np.any(detection) else None,
                capsize=3,
                elinewidth=1.2,
                zorder=10,
            )

    z_notes = (
        []
        if (observation_redshifts is not None or len(obs_redshifts) > 1)
        else [str(obs["z_note"]) for obs in observations if str(obs["z_note"])]
    )
    if z_notes:
        ax.text(
            0.97,
            0.95,
            "\n".join(z_notes),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.70",
                "alpha": 0.9,
            },
        )

    ax.set_title(rf"UVLF at $z={z_text}$", fontsize=13)
    if z_float >= 14.0:
        ax.set_xlim(-22.8, -16.0)
        ax.set_ylim(1.0e-7, 1.0e-2)
    else:
        ax.set_xlim(-26.0, -16.0)
        ax.set_ylim(1.0e-8, 1.0)
    ax.set_xlabel(r"$M_{\rm UV}^{\rm obs}$")
    ax.set_ylabel(r"$\phi(M_{\rm UV})$")
    ax.grid(False)
    ax.legend(fontsize=9, loc="lower left", frameon=False)

    param_text = (
        rf"model $\epsilon_0={SFR_PARAMS.epsilon_0:.3f}$" "\n"
        rf"$M_c=10^{{11.70}}\,M_\odot$" "\n"
        rf"$\beta_*={SFR_PARAMS.beta_star:.2f}$" "\n"
        rf"$\gamma_*={SFR_PARAMS.gamma_star:.2f}$"
    )
    ax.text(
        0.04,
        0.96,
        param_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.70", "alpha": 0.9},
    )

    fig.tight_layout()
    output_stem = OUTPUT_DIR / f"uvlf_old_method_current_params_z{z_tag(z_value)}{output_suffix}"
    fig.savefig(f"{output_stem}.pdf", dpi=500)
    fig.savefig(f"{output_stem}.png", dpi=500)
    plt.close(fig)
    print(f"saved={output_stem}.pdf")
    print(f"saved={output_stem}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redshifts",
        type=float,
        nargs="+",
        default=DEFAULT_REDSHIFTS,
        help="Redshifts to plot. Defaults to the original four-panel set.",
    )
    parser.add_argument(
        "--observation-redshifts",
        type=float,
        nargs="+",
        default=None,
        help="Observation redshift folders to overlay on every requested model redshift.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to each output filename stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kuv = integrate_kuv_from_current_ssp()
    print(f"K_UV(0-{SSP_UV_LOOKBACK_MAX_MYR:g} Myr) = {kuv:.6e}")
    print(f"instant_uv_scatter_dex={INSTANT_UV_SCATTER_DEX:.2f}")
    print(f"instant_epsilon_0={SFR_PARAMS.epsilon_0:.6f}")
    calibration = calibrate_instant_to_z6(kuv, instant_epsilon_0=SFR_PARAMS.epsilon_0)
    print(
        "z=6 instant calibration: "
        f"K_UV scale={calibration['luminosity_scale']:.4g} "
        f"(log10={calibration['log10_luminosity_scale']:.3f}), "
        f"RMS={calibration['rms_dex']:.3f} dex"
    )
    for z_value in args.redshifts:
        plot_single(
            z_value,
            kuv,
            instant_epsilon_0=SFR_PARAMS.epsilon_0,
            instant_luminosity_scale=calibration["luminosity_scale"],
            observation_redshifts=args.observation_redshifts,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
