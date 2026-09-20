"""Wide slide figure from the verified independent escape-fraction scan."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]


def main():
    summary = json.loads((ROOT / "data_save/popiii_escape_scan_20260918/summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Completed scan required")
    plt.style.use("apj")
    for filename in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path.home() / ".local/share/fonts/microsoft-academic" / filename)
        )
    plt.rcParams.update(
        {"font.family": "Arial", "text.usetex": False, "font.size": 12, "mathtext.fontset": "stix"}
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#777777", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#B65D27"]
    for result, color in zip(summary["results"], colors, strict=True):
        fesc = result["fesc_popiii"]
        histories = json.loads((Path(result["run"]) / "histories.json").read_text())
        h = histories["popii" if fesc == 0 else "popii_popiii"]
        ax.plot(
            h["redshifts"],
            1 - np.array(h["mean_xhii"]),
            color=color,
            lw=1.8,
            ls="--" if fesc in (0, 0.2) else "-",
            label=rf"$f_{{\rm esc,III}}={fesc:g}$",
        )
    with (ROOT / "external_data/observations/reionization/neutral_fraction.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    for row in observations:
        if row["kind"] != "interval":
            continue
        z, loz, hiz, x, lo, hi = (
            float(row[k]) for k in ["z", "z_min", "z_max", "xhi", "lower", "upper"]
        )
        broad = loz != hiz
        label = {
            "mason18_7": "Mason+18",
            "davies18_709": "Davies+18",
            "davies18_754": None,
            "mason26_65": "Mason+26: broad z bins",
            "mason26_93": None,
        }[row["id"]]
        ax.errorbar(
            z,
            x,
            xerr=[[z - loz], [hiz - z]],
            yerr=[[x - lo], [hi - x]],
            fmt="s" if broad else "o",
            color="#79549D" if broad else "black",
            mfc="white",
            ms=5,
            capsize=3,
            lw=1,
            zorder=5,
            label=label,
        )
    ax.set(
        xlim=(5.4, 10.8),
        ylim=(-0.025, 1.025),
        xlabel="Redshift z",
        ylabel=r"Neutral fraction $\langle x_{\rm HI}\rangle_V$",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    out = ROOT / "outputs/popiii_escape_scan_20260918"
    fig.savefig(out / "escape_scan_slide.png", dpi=160)
    fig.savefig(ROOT / "slides/popiii_heii_pisn_complete_20260916/assets/escape_scan.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
