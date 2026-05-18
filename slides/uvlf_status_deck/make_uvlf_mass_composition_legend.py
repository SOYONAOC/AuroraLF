#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


plt.style.use("apj")


OUTPUT_STEM = "uvlf_mass_composition_legend_vertical"
LOGMH_MIN = 9.0
LOGMH_MAX = 13.0
LOGMH_BIN_WIDTH = 0.5


def _mass_bin_labels(logmh_edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for left, right in zip(logmh_edges[:-1], logmh_edges[1:], strict=True):
        labels.append(rf"$10^{{{left:g}}}$-$10^{{{right:g}}}$")
    return labels


def main() -> None:
    slides_dir = Path(__file__).resolve().parent
    output_dir = slides_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    logmh_edges = np.arange(LOGMH_MIN, LOGMH_MAX + LOGMH_BIN_WIDTH, LOGMH_BIN_WIDTH, dtype=float)
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, logmh_edges.size - 1))
    labels = _mass_bin_labels(logmh_edges)

    fig, ax = plt.subplots(figsize=(1.35, 4.85), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, len(labels) + 0.85)
    ax.axis("off")

    ax.text(
        0.5,
        len(labels) + 0.55,
        r"$M_{\rm h}\ [M_\odot]$",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    swatch_x = 0.05
    swatch_w = 0.25
    text_x = 0.37
    swatch_h = 0.62
    for index, (color, label) in enumerate(zip(colors[::-1], labels[::-1], strict=True)):
        y = index + 0.18
        ax.add_patch(
            Rectangle(
                (swatch_x, y),
                swatch_w,
                swatch_h,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(text_x, y + 0.5 * swatch_h, label, ha="left", va="center", fontsize=7.5)

    for suffix in ("pdf", "png"):
        output_path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        fig.savefig(output_path, dpi=500, transparent=True, bbox_inches="tight", pad_inches=0.02)
        print(f"saved_{suffix}={output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
