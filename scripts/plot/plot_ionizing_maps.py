"""Review figures for real AuroraLF source budgets and paired SmallScale maps."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm


def save(fig, assets, previews, name):
    fig.savefig(assets / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(previews / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("slides/21cm_map/assets/results"))
    p.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/previews"))
    p.add_argument("--budget-only", action="store_true")
    a = p.parse_args()
    a.assets.mkdir(parents=True, exist_ok=True)
    a.previews.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    font_manager.fontManager.addfont("/usr/share/fonts/google-noto/NotoSans-Regular.ttf")
    font_manager.findfont("Noto Sans", fallback_to_default=False)
    # Match the supplied clean slide style; all axes and colour bars carry units.
    plt.rcParams.update(
        {
            "font.size": 12,
            "text.usetex": False,
            "font.family": "Noto Sans",
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 14,
        }
    )
    budget = json.loads((a.run / "photon_budget.json").read_text())
    z = np.array([r["z"] for r in budget])
    counts = np.array([r["photons_per_h"] for r in budget])
    error = np.array([r["se_per_h"] for r in budget])
    good = z <= 20
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(z[good], counts[good, 0], "o-", label="Pop II", color="#286491")
    ax.errorbar(
        z[good],
        counts[good, 1],
        yerr=error[good, 1],
        fmt="o-",
        color="#b86632",
        capsize=3,
        label="Resolved Pop III (MC error)",
    )
    ax.fill_between(
        z[good],
        counts[good, 1],
        counts[good, 1] + counts[good, 2],
        color="#b86632",
        alpha=0.2,
        label="Including pre-start Pop III: upper bound",
    )
    ax.axhline(1, ls=":", color="0.45", lw=1)
    ax.set(
        xlabel="Redshift",
        ylabel=r"Cumulative $N_{\gamma,\mathrm{esc}}/N_{\mathrm{H}}$",
        yscale="log",
        xlim=(6, 20),
    )
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, a.assets, a.previews, "photon_budget")
    if a.budget_only:
        return
    manifest = json.loads((a.run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("map run is not complete")
    history = json.loads((a.run / "histories.json").read_text())
    labels = [
        ("popii", "Pop II", "#286491"),
        ("popii_popiii", "Pop II + resolved Pop III", "#b86632"),
        ("popii_popiii_upper", "Including pre-start Pop III upper bound", "#657349"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for key, label, color in labels:
        h = history[key]
        ax.plot(h["redshifts"], 1 - np.array(h["mean_xhii"]), color=color, label=label)
    ax.set(
        xlabel="Redshift",
        ylabel="Volume-mean neutral fraction",
        xlim=(6, 20),
        ylim=(0, 1),
    )
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, a.assets, a.previews, "neutral_history")
    for z in manifest["selected_map_redshifts"]:
        fields = [
            np.load(a.run / "maps" / f"brightness_{key}_z{z:.2f}.npy", mmap_mode="r")[150].copy()
            for key, _, _ in labels[:2]
        ]
        difference = fields[1] - fields[0]
        vmax = max(float(x.max()) for x in fields)
        diffmax = max(float(np.max(abs(difference))), np.finfo(float).eps)
        fig = plt.figure(figsize=(12, 4.8))
        axes = [fig.add_axes([0.06 + i * 0.32, 0.28, 0.25, 0.625]) for i in range(3)]
        color_axes = [fig.add_axes([0.06 + i * 0.32, 0.10, 0.25, 0.045]) for i in range(3)]
        for i, x in enumerate(fields):
            im = axes[i].imshow(
                x.T,
                origin="lower",
                extent=(0, 300, 0, 300),
                vmin=0,
                vmax=vmax,
                cmap="viridis",
                interpolation="nearest",
            )
            axes[i].set_title(labels[i][1], fontsize=12)
            fig.colorbar(
                im,
                cax=color_axes[i],
                orientation="horizontal",
                label=r"$\delta T_b$ [mK]",
            )
        im = axes[2].imshow(
            difference.T,
            origin="lower",
            extent=(0, 300, 0, 300),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-diffmax, vcenter=0, vmax=diffmax),
            interpolation="nearest",
        )
        axes[2].set_title("Pop III change", fontsize=12)
        fig.colorbar(
            im,
            cax=color_axes[2],
            orientation="horizontal",
            label=r"$\Delta\delta T_b$ [mK]",
        )
        for ax in axes:
            ax.set(xlabel="y [comoving Mpc]", ylabel="z [comoving Mpc]")
        fig.suptitle(
            f"Redshift {z:.2f}; x = 150.5 comoving Mpc; saturated spin temperature",
            fontsize=12,
        )
        save(fig, a.assets, a.previews, f"map_z{z:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    selected = manifest["selected_map_redshifts"]
    for ax, z in zip(axes, [selected[0], selected[-2]], strict=True):
        for key, label, color in labels:
            short_label = {
                "popii": "Pop II",
                "popii_popiii": "+ Pop III",
                "popii_popiii_upper": "+ pre-start upper",
            }[key]
            with np.load(a.run / "maps" / f"power_{key}_z{z:.2f}.npz") as data:
                good = data["delta2_mk2"] > 0
                if np.any(good):
                    ax.loglog(
                        data["k_mpc_inverse"][good],
                        data["delta2_mk2"][good],
                        color=color,
                        label=short_label,
                    )
                else:
                    ax.plot([], [], color=color, label=short_label + " (zero power)")
        ax.set(
            xlabel=r"$k$ [comoving Mpc$^{-1}$]",
            ylabel=r"$\Delta^2_{21}$ [mK$^2$]",
            title=f"Redshift {z:.2f}",
        )
    for ax in axes:
        ax.legend(fontsize=10)
    save(fig, a.assets, a.previews, "brightness_power")


if __name__ == "__main__":
    main()
