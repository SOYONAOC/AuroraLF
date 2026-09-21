"""Plot the analytic input distribution of the Pop III critical halo mass ratio.

Use --stellar-halo for the complete 20260916 deck: halo mass on the left,
stellar mass on the right, both at z=14.5 and using the source model efficiency.
The stellar panel maps the input threshold, not a sampled burst population.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from auroralf.cooling import (
    ATOMIC_COOLING_MU,
    ATOMIC_COOLING_TEMPERATURE_K,
    compute_atomic_cooling_mass_msun,
)
from auroralf.mah import Cosmology

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "slides/popiii_heii_pisn_20260913/assets"
OUT = ROOT / "outputs/popiii_heii_pisn_20260913"
SAVE = ROOT / "data_save/popiii_heii_pisn_uvlf_20260913"
SOURCE = ROOT / "data_save/uvlf_current_z6_z8_z10_stratified/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stellar-halo", action="store_true")
    parser.add_argument(
        "--log10-mean",
        type=float,
        help="Explicit alternative input distribution; keeps the source sigma and efficiency",
    )
    args = parser.parse_args()
    assets, out, save = ASSETS, OUT, SAVE
    if args.stellar_halo:
        assets = ROOT / "slides/popiii_heii_pisn_complete_20260916/assets"
        out = ROOT / "outputs/popiii_heii_pisn_complete_20260916"
        save = ROOT / "data_save/popiii_heii_pisn_complete_20260916"
    manifest = json.loads(SOURCE.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete source model manifest")
    mu = manifest["configs"][0]["q_log10_mean"]
    sigma = manifest["configs"][0]["q_log10_sigma"]
    if sigma <= 0 or not np.isfinite([mu, sigma]).all():
        raise ValueError("Invalid Normal distribution parameters")
    for cfg in manifest["configs"]:
        assert cfg["q_log10_mean"] == mu and cfg["q_log10_sigma"] == sigma
    source_mu = mu
    if args.log10_mean is not None:
        if not np.isfinite(args.log10_mean):
            raise ValueError("Nonfinite alternative log10 mean")
        mu = args.log10_mean
    # At fixed redshift, log10(Mcrit) = log10(Mcool) + log10(Mcrit/Mcool).
    # Probability per dex is unchanged by this shift, unlike density per Msun.
    log_ratio = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 2001)
    ratio = 10**log_ratio
    density = np.exp(-0.5 * ((log_ratio - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    central_probability = math.erf(1 / math.sqrt(2))
    two_sigma_probability = math.erf(2 / math.sqrt(2))
    plotted_probability = math.erf(3 / math.sqrt(2))
    np.testing.assert_allclose(np.trapezoid(density, log_ratio), plotted_probability, rtol=1e-7)
    median = 10**mu
    low, high = 10 ** (mu - sigma), 10 ** (mu + sigma)
    low_two, high_two = 10 ** (mu - 2 * sigma), 10 ** (mu + 2 * sigma)
    plt.style.use("apj")
    fonts = Path("/home/zhuhourui/.local/share/fonts/microsoft-academic")
    for name in ["Arial.TTF", "Arialbd.TTF", "msyh.ttf", "msyhbd.ttf"]:
        font_manager.fontManager.addfont(str(fonts / name))
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Microsoft YaHei"],
            "font.size": 15,
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    cosmo = Cosmology()
    source_cosmo = manifest["source_model"]
    np.testing.assert_allclose(
        [cosmo.h0_km_s_mpc, cosmo.omega_m, cosmo.omega_b],
        [100 * source_cosmo["h"], source_cosmo["omega_m"], source_cosmo["omega_b"]],
        rtol=1e-12,
    )
    epsilon = source_cosmo["epsilon_b"]
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Invalid source model burst efficiency")
    stellar_factor = epsilon * cosmo.omega_b / cosmo.omega_m
    redshifts = [14.5, 14.5] if args.stellar_halo else [12.5, 14.5]
    cooling_masses = [compute_atomic_cooling_mass_msun(z, cosmology=cosmo) for z in redshifts]
    cjk = font_manager.FontProperties(fname=str(fonts / "msyh.ttf"), size=14)

    def scientific(value):
        exponent = int(np.floor(np.log10(value)))
        return rf"{value / 10**exponent:.2f}\times10^{{{exponent}}}"

    mass_summary = []
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), sharex=False, sharey=True)
    for index, (ax, z, cooling) in enumerate(zip(axs, redshifts, cooling_masses, strict=True)):
        stellar_panel = args.stellar_halo and index == 1
        scale = stellar_factor if stellar_panel else 1.0
        reference_mass = cooling * scale
        mass = ratio * reference_mass
        np.testing.assert_allclose(
            np.trapezoid(density, np.log10(mass)), plotted_probability, rtol=1e-7
        )
        # Draw the inclusive 2-sigma region first, then the darker 1-sigma core.
        # Sigma is defined in log10 mass; these are asymmetric in linear mass.
        bands = {}
        for n_sigma, color in [(2, "#deebf3"), (1, "#a9c8df")]:
            band_log_ratio = np.linspace(mu - n_sigma * sigma, mu + n_sigma * sigma, 2001)
            band_density = np.exp(-0.5 * ((band_log_ratio - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
            probability = math.erf(n_sigma / math.sqrt(2))
            np.testing.assert_allclose(
                np.trapezoid(band_density, band_log_ratio), probability, rtol=1e-6
            )
            bands[n_sigma] = ax.fill_between(
                10**band_log_ratio * reference_mass,
                0,
                band_density,
                color=color,
                label=rf"${n_sigma}\sigma$: {probability:.1%}",
            )
        ax.plot(mass, density, color="#286491", lw=2.5)
        ax.legend(
            handles=[bands[1], bands[2]],
            loc="center right",
            bbox_to_anchor=(1, 0.60),
            frameon=False,
            fontsize=12,
        )
        peak = 1 / (sigma * np.sqrt(2 * np.pi))
        if mu != 0:
            ax.vlines(median * reference_mass, 0, peak, color="#b86632", ls="--", lw=2)
        ax.axvline(reference_mass, color="#c44e52", lw=2.2)
        reference_label = (
            "冷却阈值对应质量\n" + rf"${scientific(reference_mass)}\,M_\odot$"
            if stellar_panel
            else "原子冷却阈值\n" + rf"$M_{{\rm cool}}={scientific(cooling)}\,M_\odot$"
        )
        if mu == 0:
            reference_label = "中位数 = " + reference_label
            if stellar_panel:
                reference_label = reference_label.replace("冷却阈值对应质量", "冷却阈值映射")
        ax.annotate(
            reference_label,
            xy=(reference_mass, 0.215),
            xycoords="data",
            xytext=(0.03, 0.95),
            textcoords="axes fraction",
            ha="left",
            va="top",
            color="#c44e52",
            fontproperties=cjk,
            arrowprops={"arrowstyle": "->", "color": "#c44e52", "lw": 1.4},
        )
        if mu != 0:
            ax.text(
                0.98,
                0.95,
                "中位数\n" + rf"${scientific(median * reference_mass)}\,M_\odot$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#b86632",
                fontproperties=cjk,
            )
        ax.set(
            xscale="log",
            xlim=(min(cooling_masses) * scale * ratio[0], max(cooling_masses) * scale * ratio[-1]),
            ylim=(0, 0.335),
            xlabel=(
                r"Pop III 初始恒星质量 $M_{\star,\rm III}\ [M_\odot]$"
                if stellar_panel
                else r"成星临界晕质量 $M_{h,\rm crit}\ [M_\odot]$"
            ),
            title=rf"$z={z}$",
        )
        ax.set_xticks(10.0 ** (np.arange(2, 11, 2) if stellar_panel else np.arange(4, 13, 2)))
        ax.xaxis.label.set_fontproperties(cjk)
        ax.grid(axis="y", alpha=0.15)
        mass_summary.append(
            {
                "redshift": z,
                "quantity": "initial_stellar_mass_at_threshold"
                if stellar_panel
                else "critical_halo_mass",
                "halo_to_plotted_mass_factor": scale,
                "plotted_mass_median_msun": median * reference_mass,
                "plotted_mass_one_sigma_interval_msun": [
                    low * reference_mass,
                    high * reference_mass,
                ],
                "plotted_mass_two_sigma_interval_msun": [
                    low_two * reference_mass,
                    high_two * reference_mass,
                ],
                "atomic_cooling_mass_msun": cooling,
                "critical_mass_median_msun": median * cooling,
                "central_one_sigma_interval_msun": [low * cooling, high * cooling],
                "central_two_sigma_interval_msun": [low_two * cooling, high_two * cooling],
            }
        )
    axs[0].set_ylabel("概率密度（每 dex）")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.90, wspace=0.10)
    for path in [assets, out, save]:
        path.mkdir(parents=True, exist_ok=True)
    figure = assets / "threshold_distribution.pdf"
    fig.savefig(figure)
    fig.savefig(out / "threshold_distribution.png", dpi=160)
    plt.close(fig)
    summary = {
        "source_log10_mass_ratio_mean": source_mu,
        "alternative_input_distribution": args.log10_mean is not None,
        "log10_mass_ratio_mean": mu,
        "log10_mass_ratio_sigma_dex": sigma,
        "log10_mass_ratio_variance": sigma**2,
        "mass_ratio_median": median,
        "one_sigma_multiplicative_factor": 10**sigma,
        "central_one_sigma_interval": [low, high],
        "central_one_sigma_probability": central_probability,
        "two_sigma_multiplicative_factor": 10 ** (2 * sigma),
        "central_two_sigma_interval": [low_two, high_two],
        "central_two_sigma_probability": two_sigma_probability,
        "probability_below_atomic_cooling_threshold": 0.5 * math.erfc(mu / sigma / math.sqrt(2)),
        "plot_density": "dP/dlog10(M/Msun), per dex of the labeled panel quantity",
        "stellar_mapping": {
            "enabled": args.stellar_halo,
            "epsilon_b": epsilon,
            "baryon_fraction": cosmo.omega_b / cosmo.omega_m,
            "mass_factor": stellar_factor,
            "definition": "Mstar,III = epsilon_b * (Omega_b/Omega_m) * Mh,crit; threshold mapping, not actual first-crossing host statistics",
        },
        "mass_distributions": mass_summary,
        "cooling_prescription": {
            "virial_temperature_k": ATOMIC_COOLING_TEMPERATURE_K,
            "mu": ATOMIC_COOLING_MU,
        },
        "plotted_probability": plotted_probability,
        "scope": "Analytic untruncated model input, one ratio draw per halo history; not a fit or predicted burst-host distribution",
        "sources": {
            str(path): digest(path)
            for path in [
                SOURCE,
                Path(__file__),
                ROOT / "auroralf/experiments/random_q.py",
                ROOT / "auroralf/mah/physics.py",
                ROOT / "auroralf/mah/models.py",
            ]
        },
        "figure": {str(figure): digest(figure)},
    }
    (save / "threshold_distribution.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ["sources", "figure"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
