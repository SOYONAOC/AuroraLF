"""Venditti Fig. 1 mean-curve comparison with the current AuroraLF SFRD."""

import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/ionizing_sources/sfrd_v1"
OUT = ROOT / "outputs/21cm_map/sfrd"
ASSETS = ROOT / "slides/assets/sfrd"
PAPER = (
    ROOT
    / "external_data/literature_sources/popiii_uvlf_library/papers/Venditti2023ANeedleInA/source/figures/SFRD_pop_av+obs+U12.pdf"
)
COLORS = {"popii": "#32a832", "popiii": "#e32222", "total": "#244bf0"}
STYLES = {"popii": "--", "popiii": "-", "total": ":"}
LABELS = {"popii": "Pop II", "popiii": "Pop III", "total": "Pop II + III"}


def reference_curves():
    """Recover vector vertices, calibrated by visible axis ticks (not raw data)."""
    svg = OUT / "venditti_fig1.svg"
    subprocess.run(["pdftocairo", "-svg", str(PAPER), str(svg)], check=True)
    found = {}
    colors = {
        "popiii": "100%,0%,0%",
        "popii": "19.607544%,80.39093%,19.607544%",
        "total": "0%,0%,100%",
    }
    for e in ET.parse(svg).getroot().iter():
        style, path = e.get("style", ""), e.get("d", "")
        if "fill:none;" not in style or "stroke-width:1.5;" not in style or len(path) < 200:
            continue
        for key, color in colors.items():
            if f"stroke:rgb({color});" not in style:
                continue
            assert key not in found
            assert e.get("transform") == "matrix(1,0,0,-1,0,360)"
            assert set(re.findall("[A-Za-z]", path)) <= {"M", "L"}
            xy = np.array(
                [list(map(float, v)) for v in re.findall(r"[ML]\s+([-\d.]+)\s+([-\d.]+)", path)]
            )
            # Left-panel x ticks: 7 and 17. Native PDF y ticks: -7 and -2 dex.
            z = 7 + (xy[:, 0] - 67.105469) * 10 / (329.183594 - 67.105469)
            log_sfr = -7 + (xy[:, 1] - 39.601562) * 5 / (324.328125 - 39.601562)
            keep = (z >= 6.5) & (z <= 19) & (log_sfr >= -7) & (log_sfr <= -1.5)
            order = np.argsort(z[keep])
            found[key] = (z[keep][order], 10 ** log_sfr[keep][order])
    assert set(found) == set(colors)
    with (RUN / "venditti2023_fig1_curves.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["population", "z", "sfrd_msun_yr_cmpc3"])
        for key, (zs, values) in found.items():
            writer.writerows(zip([key] * len(zs), zs, values, strict=True))
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    for filename in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / filename)
        )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "text.usetex": False,
        }
    )
    reference = reference_curves()
    data = np.genfromtxt(RUN / "sfrd.csv", delimiter=",", names=True)
    current = data[data["window_myr"] == 10]
    z = current["z"]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.7), sharex=True, sharey=True, layout="constrained"
    )
    for key in ["total", "popii", "popiii"]:
        zz, yy = reference[key]
        axes[0].plot(zz, np.log10(yy), color=COLORS[key], ls=STYLES[key], lw=2, label=LABELS[key])
        axes[1].plot(
            z, np.log10(current[key]), color=COLORS[key], ls=STYLES[key], lw=2, label=LABELS[key]
        )
    axes[0].set_title("Venditti et al. (2023), Fig. 1")
    axes[1].set_title("AuroraLF: current first-burst model")
    for ax in axes:
        ax.set(xlim=(6.5, 19), ylim=(-7, -1.5), xlabel="Redshift z")
        ax.set_xticks([7, 9, 11, 13, 15, 17, 19])
        ax.legend(loc="lower left", frameon=False)
        ax.text(
            0.97, 0.94, r"$\Delta t=10\ \mathrm{Myr}$", ha="right", va="top", transform=ax.transAxes
        )
    axes[0].set_ylabel(
        r"$\log_{10}[\rho_{\rm SFR}/(M_\odot\,\mathrm{yr}^{-1}\,\mathrm{cMpc}^{-3})]$"
    )
    fig.savefig(ASSETS / "sfrd_comparison.pdf")
    fig.savefig(OUT / "sfrd_comparison.png", dpi=160)
    fig.set_size_inches(11.2, 3.3)
    fig.savefig(ASSETS / "sfrd_comparison_slide.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.4),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for key in ["total", "popii", "popiii"]:
        axes[0].plot(
            z, np.log10(current[key]), color=COLORS[key], ls=STYLES[key], lw=2, label=LABELS[key]
        )
    axes[0].set(
        ylim=(-9.2, -1.3),
        ylabel=r"$\log_{10}[\rho_{\rm SFR}/(M_\odot\,\mathrm{yr}^{-1}\,\mathrm{cMpc}^{-3})]$",
    )
    axes[0].legend(loc="lower left", frameon=False)
    axes[0].set_title("AuroraLF: newly formed mass, averaged over 10 Myr")
    axes[1].plot(z, current["popiii_fraction"] * 100, color=COLORS["popiii"], lw=2)
    axes[1].set(
        xlim=(6, 30.2),
        ylim=(0, 105),
        yticks=[0, 50, 100],
        xlabel="Redshift z",
        ylabel="Pop III SFR\nfraction [%]",
    )
    axes[1].set_xticks([6, 10, 15, 20, 25, 30])
    fig.savefig(ASSETS / "sfrd_full.pdf")
    fig.savefig(OUT / "sfrd_full.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, key in zip(axes, ["popii", "popiii"], strict=True):
        for window in [1, 5, 20]:
            row = data[data["window_myr"] == window]
            ax.plot(z, 100 * (row[key] / current[key] - 1), label=f"{window} Myr")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(title=LABELS[key], xlabel="Redshift z", ylabel="Difference from 10 Myr [%]")
        ax.legend()
    fig.savefig(OUT / "window_check.png", dpi=140)
    plt.close(fig)
    comparisons = []
    for z0 in [7, 8, 10, 12.5, 15]:
        item = {"z": z0}
        for key in ["popii", "popiii"]:
            zz, yy = reference[key]
            ref = 10 ** np.interp(z0, zz, np.log10(yy))
            ours = 10 ** np.interp(z0, z, np.log10(current[key]))
            item[key] = {"literature": ref, "auroralf": ours, "ratio": ours / ref}
        comparisons.append(item)
    result = dict(
        reference="Venditti et al. (2023), MNRAS 522, 3809, Fig. 1 left",
        doi="10.1093/mnras/stad1201",
        reference_data="Vector vertices extracted from published mean curves; no observational points or volume scatter digitized",
        calibration=dict(
            x_ticks=[[7, 67.105469], [17, 329.183594]], y_ticks=[[-7, 39.601562], [-2, 324.328125]]
        ),
        comparisons=comparisons,
        paper_figure_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(),
        model_notes="Different physics and resolution; AuroraLF has no pristine-gas or metal-enrichment gate. SFRD is not ionizing emissivity.",
        current_csv_sha256=hashlib.sha256((RUN / "sfrd.csv").read_bytes()).hexdigest(),
    )
    (RUN / "literature_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
