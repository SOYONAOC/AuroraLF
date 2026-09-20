"""Reionization histories and stage-matched maps/powers from actual run products."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from plot_ionizing_maps import save

CASES = [
    ("popii", "Pop II", "#286491"),
    ("popii_popiii", "+ resolved Pop III", "#b86632"),
    ("popii_popiii_upper", "+ pre-start upper", "#657349"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("slides/21cm_map/assets/results"))
    p.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/previews"))
    a = p.parse_args()
    a.assets.mkdir(parents=True, exist_ok=True)
    a.previews.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a.data / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("incomplete matched-stage analysis")
    for name, expected in summary["product_sha256"].items():
        if hashlib.sha256((a.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed matched-stage product: {name}")
    plt.style.use("apj")
    font_manager.fontManager.addfont("/usr/share/fonts/google-noto/NotoSans-Regular.ttf")
    font_manager.findfont("Noto Sans", fallback_to_default=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Noto Sans",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    for t in summary["targets"]:
        ax.axhline(t, color="0.8", ls=":", lw=1)
    for key, label, color in CASES:
        h = summary["history"][key]
        ax.plot(h["redshifts"], h["mean_xhii"], color=color, label=label, lw=2)
        stages = summary["cases"][key]["stages"]
        ax.scatter(
            [s["nearest_redshift"] for s in stages],
            [s["nearest_mean_xhii"] for s in stages],
            color=color,
            s=32,
            zorder=3,
        )
        z50 = summary["cases"][key]["milestones"]["0.5"]["estimated_redshift"]
        ax.plot([z50, z50], [0, 0.5], ls="--", color=color, alpha=0.6, lw=1)
        ax.text(
            z50 + 0.15,
            0.02,
            f"$z_{{50}}={z50:.2f}$",
            rotation=90,
            color=color,
            ha="right",
            va="bottom",
            fontsize=10,
        )
    ax.set(
        xlim=(25, 6),
        ylim=(0, 1.02),
        xlabel="Redshift (time increases to the right)",
        ylabel=r"Volume-mean ionized fraction $\langle x_{\mathrm{HII}}\rangle_V$",
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    ax.legend(loc="upper left", fontsize=11)
    save(fig, a.assets, a.previews, "reionization_history")

    for j, target in enumerate(summary["targets"]):
        rows = [summary["cases"][key]["stages"][j] for key, _, _ in CASES]
        for quantity, cmap, cb_label in [
            ("xhi_slice", "cividis", r"Neutral fraction $x_{\mathrm{HI}}$"),
            ("brightness_slice_mk", "viridis", r"$\delta T_b$ [mK]"),
        ]:
            fields = []
            for row in rows:
                with np.load(a.data / row["nearest_snapshot"]["file"]) as data:
                    fields.append(data[quantity])
            vmax = 1.0 if quantity == "xhi_slice" else max(float(f.max()) for f in fields)
            fig = plt.figure(figsize=(12, 5.2))
            axes = [fig.add_axes([0.06 + i * 0.32, 0.28, 0.25, 0.575]) for i in range(3)]
            cax = fig.add_axes([0.30, 0.11, 0.43, 0.04])
            for i, (field, row, (_, label, _)) in enumerate(zip(fields, rows, CASES, strict=True)):
                im = axes[i].imshow(
                    field.T,
                    origin="lower",
                    extent=(0, 300, 0, 300),
                    vmin=0,
                    vmax=vmax,
                    cmap=cmap,
                    interpolation="nearest",
                )
                axes[i].set_title(
                    f"{label}\nz={row['nearest_redshift']:.2f}, "
                    + r"$\langle x_{\mathrm{HII}}\rangle_V$="
                    + f"{row['nearest_snapshot']['mean_xhii']:.3f}",
                    fontsize=12,
                )
                axes[i].set(xlabel="y [comoving Mpc]", ylabel="z [comoving Mpc]")
            fig.colorbar(im, cax=cax, orientation="horizontal", label=cb_label)
            fig.suptitle(
                f"Target ionization {target:.0%}: nearest real snapshots; x = 150.5 comoving Mpc",
                fontsize=12,
                y=1.0,
            )
            name = "neutral" if quantity == "xhi_slice" else "brightness"
            save(fig, a.assets, a.previews, f"matched_{name}_{target:.2f}")

    for quantity, ylabel, name in [
        ("delta2_xhi", r"$\Delta^2_{x_{\mathrm{HI}}}$ [dimensionless]", "neutral"),
        ("delta2_21_mk2", r"$\Delta^2_{21}$ [mK$^2$]", "brightness"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True, layout="constrained")
        for j, (ax, target) in enumerate(zip(axes, summary["targets"], strict=True)):
            for key, label, color in CASES:
                stage = summary["cases"][key]["stages"][j]
                with np.load(a.data / stage["power_file"]) as data:
                    k, y = data["k_mpc_inverse"], data[quantity]
                    if not np.isfinite(y).all() or np.any(y <= 0):
                        raise ValueError("invalid power at a partially ionized stage")
                    ax.loglog(k, y, color=color, label=label, lw=1.8)
                    ax.fill_between(
                        k,
                        data[quantity + "_bracket_min"],
                        data[quantity + "_bracket_max"],
                        color=color,
                        alpha=0.15,
                    )
            ax.set(
                xlabel=r"$k$ [comoving Mpc$^{-1}$]",
                title=r"$\langle x_{\mathrm{HII}}\rangle_V$=" + f"{target:.2f}",
            )
            ax.grid(alpha=0.15)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=10, loc="lower right")
        save(fig, a.assets, a.previews, f"matched_{name}_power")


if __name__ == "__main__":
    main()
