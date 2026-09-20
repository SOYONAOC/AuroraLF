"""Plot the SSP yield tail and paired population-rate age-window errors."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from auroralf.ssp.ionizing import load_popiii_ionizing_kernel  # noqa: E402


def main():
    run = ROOT / "data_save/ionizing_sources/popiii_age_window_v1"
    preview = ROOT / "outputs/21cm_map/popiii_age_window"
    preview.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    model = json.loads((run / "manifest.json").read_text())["model"]
    k = load_popiii_ionizing_kernel(model["popiii_ssp"])
    with np.load(run / "comparison.npz") as data:
        zs, windows, loss, se = [
            data[key] for key in ["redshifts", "windows_myr", "relative_loss", "paired_se"]
        ]
    plt.style.use("apj")
    for path in (ROOT / "external_data/fonts/microsoft/arial").glob("*.TTF"):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "text.usetex": False,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.5), layout="constrained")
    age = np.linspace(2, 20, 600)
    axes[0].plot(
        age,
        100 * (1 - k.yield_photons(age) / k.yield_photons(100)),
        color="#24588a",
        lw=2,
    )
    axes[0].axvline(summary["ssp_yield_99pct_age_myr"], ls=":", color="#24588a")
    axes[0].set(
        title="Single-burst integrated yield",
        xlabel="Age cutoff [Myr]",
        ylabel="Missing yield [% of 100 Myr yield]",
    )
    selected = windows <= 20
    for target, color in zip(
        [6, 10, 20, 40], ["#24588a", "#d27b2c", "#359275", "#8862a4"], strict=True
    ):
        i = np.argmin(abs(zs - target))
        axes[1].plot(
            windows[selected],
            100 * loss[i, selected],
            marker="o",
            ms=3,
            color=color,
            label=f"z = {zs[i]:g}",
        )
    axes[1].plot(
        windows[selected],
        100 * np.max(loss + 2 * se, axis=0)[selected],
        color="#222222",
        ls="--",
        label="Worst tested z + 2 SE",
    )
    axes[1].set(
        title="Population instantaneous rate (Reed07)",
        xlabel="Age cutoff [Myr]",
        ylabel="Missing rate [% of 100 Myr rate]",
    )
    axes[1].legend(fontsize=8.5)
    for ax in axes:
        ax.axhline(1, color="#a32b32", ls="--", lw=1)
        ax.set(
            yscale="log",
            xlim=(2, 20),
            ylim=(0.008, 80),
            xticks=[2, 4, 6, 8, 10, 15, 20],
        )
    fig.savefig(preview / "convergence.pdf")
    fig.savefig(preview / "convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
