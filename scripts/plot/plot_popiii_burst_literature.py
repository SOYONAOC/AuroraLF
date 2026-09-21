"""Reconstruct a published burst-mass fit; this is not a simulation or new fit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/popiii_burst_mass"
ASSETS = ROOT / "slides/popiii_burst_mass/assets"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Hazlett et al., arXiv:2403.05624v1, section 3.3.2 and Table 1.
    # Interpret their lognormal "center" as the median for this reconstruction.
    median, sigma = 150.0, 0.371
    s = np.log(10) * sigma
    x = np.linspace(np.log10(median) - 5 * sigma, np.log10(median) + 5 * sigma, 10001)
    mass = 10**x
    per_dex = np.exp(-0.5 * ((x - np.log10(median)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    per_mass = per_dex / (mass * np.log(10))
    np.testing.assert_allclose(np.trapezoid(per_dex, x), 1, atol=1e-6)
    np.testing.assert_allclose(np.trapezoid(per_mass, mass), 1, atol=1e-6)
    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update(
        {"font.family": "Arial", "text.usetex": False, "font.size": 13, "pdf.fonttype": 42}
    )
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.35), layout="constrained")
    axs[0].plot(x, per_dex, color="#1f527a")
    axs[0].set(
        xlabel=r"$\log_{10}(M_{\rm burst}/M_\odot)$",
        ylabel=r"$dP/d\log_{10}M$ [dex$^{-1}$]",
        xlim=(0.7, 3.65),
        ylim=(0, 1.15),
        title="Symmetric in log mass",
    )
    axs[1].plot(mass, per_mass, color="#008060")
    axs[1].set(
        xlabel=r"$M_{\rm burst}$ [$M_\odot$]",
        ylabel=r"$dP/dM$ [$M_\odot^{-1}$]",
        xlim=(0, 1000),
        ylim=(0, 0.005),
        title="Right-skewed in linear mass",
    )
    axs[1].axvline(median, color="#626b76", ls="--", label="Median: 150")
    axs[1].legend(frameon=False)
    fig.savefig(ASSETS / "hazlett_lognormal.pdf")
    fig.savefig(OUT / "hazlett_lognormal.png", dpi=160)
    plt.close(fig)
    stats = {
        "source": "https://arxiv.org/html/2403.05624v1#S3.SS3.SSS2",
        "kind": "analytic reconstruction, not raw simulation data",
        "center_interpretation": "median",
        "median_msun": median,
        "sigma_dex": sigma,
        "central_68_percent_msun": [median * 10 ** (-sigma), median * 10**sigma],
        "mean_msun": median * np.exp(s * s / 2),
        "mode_per_linear_mass_msun": median * np.exp(-s * s),
        "skewness_linear_mass": (np.exp(s * s) + 2) * np.sqrt(np.exp(s * s) - 1),
        "plot_right_panel_excluded_probability": "full distribution extends above 1000 Msun; no truncation in calculation",
    }
    (OUT / "fit_reconstruction.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
