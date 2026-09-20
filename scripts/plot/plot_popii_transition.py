"""Intrinsic UVLF, source suppression and spatial reionization comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_save/popii_transition_20260920"
OUT = ROOT / "outputs/popii_transition_20260920"
ASSETS = ROOT / "slides/assets/popii_transition_20260920"
COLORS = ["#687782", "#168177", "#C66A2B"]
LABELS = ["Original onset", "After Pop III: 0 Myr", "After Pop III: 30 Myr"]
STYLES = ["-", "--", "-"]


def setup():
    plt.style.use("apj")
    for name in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path.home() / ".local/share/fonts/microsoft-academic" / name)
        )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def uvlf():
    ratios = np.load(DATA / "uvlf_ratios.npz")
    for zs, name in [([6, 8, 10], "uvlf_low"), ([12.5, 14.5], "uvlf_high")]:
        fig, axes = plt.subplots(
            2,
            len(zs),
            figsize=(13 if len(zs) == 3 else 11.5, 5.8),
            sharex="col",
            gridspec_kw={"height_ratios": [2, 1]},
            layout="constrained",
            squeeze=False,
        )
        for j, z in enumerate(zs):
            d = np.load(DATA / f"uvlf/z{z:g}.npz")
            x = (d["edges"][1:] + d["edges"][:-1]) / 2
            for v in range(3):
                phi, se = d["phi"][v, 2], d["se"][v, 2]
                valid = phi > 0
                axes[0, j].plot(
                    x,
                    np.where(valid, phi, np.nan),
                    color=COLORS[v],
                    ls=STYLES[v],
                    lw=2,
                    label=LABELS[v],
                )
                axes[0, j].fill_between(
                    x,
                    np.where(phi > se, phi - se, np.nan),
                    np.where(valid, phi + se, np.nan),
                    color=COLORS[v],
                    alpha=0.10,
                    lw=0,
                )
                if v:
                    r, e = ratios[f"z{z:g}_ratio"][v, 2], ratios[f"z{z:g}_ratio_se"][v, 2]
                    axes[1, j].plot(x, r, color=COLORS[v], ls=STYLES[v], lw=2)
                    axes[1, j].fill_between(x, r - e, r + e, color=COLORS[v], alpha=0.18, lw=0)
            axes[0, j].set(
                yscale="log", ylim=(1e-8 if len(zs) == 3 else 1e-11, 0.1), title=f"z = {z:g}"
            )
            axes[1, j].axhline(1, color=COLORS[0], lw=1)
            axes[1, j].set(
                xlabel=r"Intrinsic $M_{1500}$ [AB mag]", xlim=(-24, -14), ylim=(0.80, 1.03)
            )
        axes[0, 0].set_ylabel(r"Total $\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
        axes[1, 0].set_ylabel("Ratio to original")
        axes[0, -1].legend(loc="upper left", frameon=False)
        save(fig, name)
    summary = json.loads((DATA / "uvlf_summary.json").read_text())
    z = np.array([r["z"] for r in summary["rows"]])
    source = np.load(DATA / "source_diagnostics.npz")
    zs = source["redshifts"]
    rates = source["rate_density"]
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
    for v in [1, 2]:
        for comp, ls in [("total", "-"), ("popii", ":")]:
            y = np.array([r[comp + "_relative"][v] for r in summary["rows"]])
            err = np.array([r[comp + "_relative_se"][v] for r in summary["rows"]])
            axs[0].errorbar(
                z,
                y,
                yerr=err,
                marker="o",
                ms=4,
                color=COLORS[v],
                ls=ls,
                label=f"{v == 2 and 30 or 0} Myr, "
                + ("II + III" if comp == "total" else "II only"),
            )
        axs[1].plot(zs, rates[:, v, :2].sum(1) / rates[:, 0, :2].sum(1), color=COLORS[v], ls="-")
        axs[1].plot(
            zs, rates[:, v, 0] / np.maximum(rates[:, 0, 0], 1e-100), color=COLORS[v], ls=":"
        )
    for ax in axs:
        ax.axhline(1, color=COLORS[0], lw=1)
        ax.set(xlabel="Redshift z", ylim=(0.65, 1.02), xlim=(5, 15))
    axs[0].set(ylabel="Luminosity density / original", title="Intrinsic UV at 1500 A")
    axs[1].set(ylabel="Escaped photon rate / original", title="Hydrogen-ionizing emissivity")
    axs[0].legend(loc="lower left", frameon=False, ncol=2)
    save(fig, "source_ratios")


def spatial():
    summary = json.loads((DATA / "spatial_summary.json").read_text())
    arr = np.load(DATA / "spatial_history.npz")
    rows = summary["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    for i, row in enumerate(rows):
        z = arr[row["name"] + "_z"]
        x = 1 - arr[row["name"] + "_qv"]
        for ax in axs:
            ax.plot(z, x, color=COLORS[i], ls=STYLES[i], lw=2.2, label=LABELS[i])
    with (ROOT / "external_data/observations/reionization/neutral_fraction.csv").open() as f:
        obs = list(csv.DictReader(f))
    usedlabels = set()
    for r in obs:
        zz, x, lo, hi, zlo, zhi = [
            float(r[k]) for k in ("z", "xhi", "lower", "upper", "z_min", "z_max")
        ]
        if r["kind"] == "interval" and zlo == zhi:
            label = "Mason+18 / Davies+18" if "narrow" not in usedlabels else None
            usedlabels.add("narrow")
            axs[0].errorbar(
                zz,
                x,
                yerr=[[x - lo], [hi - x]],
                fmt="o",
                color="black",
                mfc="white",
                ms=4,
                capsize=2,
                label=label,
            )
        if r["kind"] == "upper_limit":
            label = "McGreer+15 limits" if "limits" not in usedlabels else None
            usedlabels.add("limits")
            axs[1].errorbar(
                zz, x, yerr=0.05, uplims=True, fmt="none", color="black", capsize=3, label=label
            )
    for ax in axs:
        ax.set(xlabel="Redshift z", ylabel=r"Volume-averaged $x_{\rm HI}$", ylim=(-0.015, 1.02))
    axs[0].set_xlim(5, 16)
    axs[1].set_xlim(5, 8)
    axs[0].legend(loc="lower right", frameon=False)
    axs[1].legend(loc="upper left", frameon=False)
    save(fig, "reionization")
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    for i, row in enumerate(rows):
        name = row["name"]
        z = arr[name + "_tau_z"]
        tau = arr[name + "_tau"]
        axs[0].plot(z, tau, color=COLORS[i], ls=STYLES[i], lw=2, label=LABELS[i])
        reference = np.interp(z, arr["baseline_tau_z"], arr["baseline_tau"])
        axs[1].plot(z, 1e3 * (tau - reference), color=COLORS[i], ls=STYLES[i], lw=2)
    axs[0].set(
        xlabel="Upper integration redshift z",
        ylabel=r"Cumulative $\tau(0,z)$",
        xlim=(0, 20),
        ylim=(0, 0.065),
    )
    axs[0].legend(loc="lower right", frameon=False)
    axs[1].set(
        xlabel="Upper integration redshift z",
        ylabel=r"$10^3[\tau-\tau_{\rm original}]$",
        xlim=(0, 20),
    )
    save(fig, "tau")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["uvlf", "spatial"], required=True)
    a = p.parse_args()
    setup()
    {"uvlf": uvlf, "spatial": spatial}[a.stage]()
