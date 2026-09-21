"""Published V24 sensitivity paths and model envelopes, with explicit provenance."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "external_data/literature_sources/heii_observability/venditti24_fig2_vectors.json"
OUTPUT = ROOT / "outputs/heii_v24_overlay_20260917"
BAND_COLORS = {"No ML": "#bd6666", "Strong ML": "#3dabab"}


def load_v24():
    data = json.loads(DATA.read_text())
    for relative, expected in data["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"V24 source changed: {relative}")
    return data


def shade(ax, data):
    for band in data["bands"]:
        low, high = band["luminosity_range_erg_s"]
        ax.fill_between(
            data["band_display_z"],
            low,
            high,
            color=BAND_COLORS[band["model"]],
            alpha=0.18,
            zorder=0,
        )


def add_v24_reference(ax):
    """A clearly specified subset for the five-redshift slide; all curves exported below."""
    data = load_v24()
    shade(ax, data)
    for curve in data["curves"]:
        if (curve["resolving_power"], curve["exposure_h"], curve["integrated_snr"]) != (
            1000,
            50,
            5,
        ):
            continue
        ax.plot(
            curve["z"],
            curve["luminosity_erg_s"],
            color="#79572c" if curve["mode"] == "IFU" else "#62508e",
            ls="--" if curve["linewidth_kms"] == 500 else ":",
            lw=1.7,
            zorder=1,
        )
    handles = [Patch(facecolor=c, alpha=0.25, label=f"V24: {m}") for m, c in BAND_COLORS.items()]
    handles += [
        Line2D([], [], color=c, lw=1.7, label=f"V24: {m}")
        for m, c in [("IFU", "#79572c"), ("MOS", "#62508e")]
    ]
    handles += [
        Line2D([], [], color="#555555", ls=ls, label=f"{v} km/s")
        for v, ls in [(500, "--"), (50, ":")]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=3,
        fontsize=10.5,
        title=r"V24: $R\simeq1000$, 50 h, S/N$\simeq5$; $\eta_{\rm III}=0.01$–0.3",
        title_fontsize=10.5,
        frameon=True,
        framealpha=0.94,
        borderpad=0.5,
    )
    return data


def main():
    data = load_v24()
    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Arial",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, ["IFU", "MOS"]):
        shade(ax, data)
        for c in data["curves"]:
            if c["mode"] != mode:
                continue
            color = (
                "#c43131"
                if c["integrated_snr"] == 5
                else ("#738291" if c["exposure_h"] == 10 else "#b48a15")
            )
            ax.plot(
                c["z"],
                c["luminosity_erg_s"],
                color=color,
                ls="--" if c["linewidth_kms"] == 500 else ":",
                marker={100: "^", 1000: "s", 2700: "o"}[c["resolving_power"]],
                lw={100: 1, 1000: 1.7, 2700: 2.4}[c["resolving_power"]],
                ms=4,
            )
        ax.set(
            title=f"NIRSpec/{mode}",
            xlabel="Redshift z",
            yscale="log",
            xlim=(6.5, 10.7),
            ylim=(1.4e40, 2e42),
        )
    axes[0].set_ylabel(r"$L_{\mathrm{He\,II}\,1640}$ [erg s$^{-1}$]")
    handles = [Patch(facecolor=c, alpha=0.25, label=f"V24 {m}") for m, c in BAND_COLORS.items()]
    handles += [
        Line2D([], [], color=c, label=label)
        for c, label in [
            ("#738291", "10 h, S/N~3"),
            ("#b48a15", "50 h, S/N~3"),
            ("#c43131", "50 h, S/N~5"),
        ]
    ]
    handles += [
        Line2D([], [], color="k", marker=m, label=f"R~{r}")
        for m, r in [("^", 100), ("s", 1000), ("o", 2700)]
    ]
    handles += [
        Line2D([], [], color="k", ls=ls, label=f"{v} km/s") for v, ls in [(500, "--"), (50, ":")]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=11, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.79, wspace=0.12)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUTPUT / "v24_all_thresholds.pdf")
    fig.savefig(OUTPUT / "v24_all_thresholds.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
