"""Figures for completed instantaneous-source maps and their historical control."""

import argparse
import json
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

from auroralf.ssp.ionizing import load_bpass_ionizing_kernel, load_popiii_ionizing_kernel

CASES = [
    ("popii", "Pop II", "#286491"),
    ("popii_popiii", "Pop II + III", "#b86632"),
    ("popii_popiii_upper", "Including pre-start upper", "#657349"),
]


def style():
    plt.style.use("apj")
    font_manager.fontManager.addfont("/usr/share/fonts/google-noto/NotoSans-Regular.ttf")
    font_manager.findfont("Noto Sans", fallback_to_default=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Noto Sans",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.titlesize": 12,
        }
    )


def save(fig, a, name):
    fig.savefig(a.assets / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(a.previews / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def crossing(z, x, target):
    """First upward crossing in time, without enforcing a monotone history."""
    z, x = np.asarray(z), np.asarray(x)
    indices = np.flatnonzero((x[:-1] < target) & (x[1:] >= target))
    if not len(indices):
        return None
    i = int(indices[0])
    f = float((target - x[i]) / (x[i + 1] - x[i]))
    return dict(
        z=float((1 - f) * z[i] + f * z[i + 1]),
        bracket=[i, i + 1],
        weight=f,
        upward_crossing_count=len(indices),
    )


def kernels(a):
    cfg = tomllib.loads(Path("configs/experiments/ionizing_rates.toml").read_text())["model"]
    k2 = load_bpass_ionizing_kernel(cfg["popii_ssp"])
    k3 = load_popiii_ionizing_kernel(cfg["popiii_ssp"])
    age = np.geomspace(0.01, 100, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    for kernel, label, color in [(k2, "Pop II SSP", CASES[0][2]), (k3, "Pop III SSP", CASES[1][2])]:
        axes[0].plot(age, kernel.rate(age), label=label, color=color)
        axes[1].plot(age, kernel.rate(age) / kernel.rate(0), label=label, color=color)
    for ax in axes:
        ax.set(xscale="log", yscale="log", xlabel="Stellar age [Myr]", xlim=(0.01, 100))
        ax.axvline(3, ls=":", lw=1, color=".5")
    axes[0].set(ylabel=r"$q_{\rm H}$ [photons s$^{-1}$ $M_\odot^{-1}$]")
    axes[1].set(ylabel=r"$q_{\rm H}(a)/q_{\rm H}(0)$", ylim=(1e-9, 2))
    axes[0].legend()
    save(fig, a, "instantaneous_ssp_ages")
    values = {
        str(t): dict(popii=float(k2.rate(t) / k2.rate(0)), popiii=float(k3.rate(t) / k3.rate(0)))
        for t in [3, 5, 10, 100]
    }
    (a.previews / "instantaneous_ssp_ages.json").write_text(json.dumps(values, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--legacy", type=Path)
    p.add_argument("--kernels-only", action="store_true")
    p.add_argument("--rates-only", action="store_true")
    p.add_argument(
        "--primary-only",
        action="store_true",
        help="Render the two completed primary cases while an additional case is still running",
    )
    p.add_argument("--assets", type=Path, default=Path("slides/21cm_map/assets/instantaneous"))
    p.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/instantaneous"))
    a = p.parse_args()
    cases = CASES[:2] if a.primary_only else CASES
    a.assets.mkdir(parents=True, exist_ok=True)
    a.previews.mkdir(parents=True, exist_ok=True)
    style()
    kernels(a)
    if a.kernels_only:
        return
    if a.run is None:
        raise ValueError("a run with completed source-budget output is required")
    manifest = json.loads((a.run / "manifest.json").read_text())
    if manifest["source_manifest"]["status"] != "complete":
        raise ValueError("incomplete source table")
    if not a.rates_only:
        if a.legacy is None:
            raise ValueError("a completed historical map run is required")
        for directory in [a.run, a.legacy]:
            if a.primary_only and directory == a.run:
                h = json.loads((a.run / "histories.json").read_text())
                for key, _, _ in cases:
                    np.testing.assert_array_equal(h[key]["redshifts"], manifest["redshifts"])
                continue
            if json.loads((directory / "manifest.json").read_text())["status"] != "complete":
                raise ValueError(f"incomplete run {directory}")
    budget = json.loads((a.run / "rate_budget.json").read_text())
    z = np.array([v["z"] for v in budget])
    rate = np.array([v["rate_s_mpc3"] for v in budget])
    cov = np.array([v["covariance_s2_mpc6"] for v in budget])
    good = (z >= 6) & (z <= 20)
    total = rate[:, :2].sum(axis=1)
    share = np.divide(rate[:, 1], total, out=np.zeros_like(total), where=total > 0)
    grad = np.zeros((len(z), 2))
    grad[total > 0] = (
        np.stack([-rate[total > 0, 1], rate[total > 0, 0]], axis=-1) / total[total > 0, None] ** 2
    )
    share_se = np.sqrt(np.einsum("bi,bij,bj->b", grad, cov[:, :2, :2], grad))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    for j, label, color in [(0, "Pop II", CASES[0][2]), (1, "Resolved Pop III", CASES[1][2])]:
        axes[0].plot(z[good], rate[good, j], color=color, label=label)
        error = np.sqrt(cov[good, j, j])
        axes[0].fill_between(
            z[good], rate[good, j] - error, rate[good, j] + error, color=color, alpha=0.2
        )
    axes[0].set(yscale="log", ylabel=r"$\dot n_{\gamma,\rm esc}$ [s$^{-1}$ cMpc$^{-3}$]")
    axes[0].legend()
    axes[1].plot(z[good], 100 * share[good], color=CASES[1][2], label="Current Pop III share")
    axes[1].fill_between(
        z[good],
        100 * (share - share_se)[good],
        100 * (share + share_se)[good],
        color=CASES[1][2],
        alpha=0.2,
    )
    axes[1].axhline(50, color=".6", ls=":", lw=1)
    axes[1].set(ylabel="Pop III / (Pop II + Pop III) [%]", ylim=(0, 100))
    for ax in axes:
        ax.set(xlabel="Redshift", xlim=(20, 6))
    save(fig, a, "instantaneous_rate_budget")
    if a.rates_only:
        return
    history = json.loads((a.run / "histories.json").read_text())
    old = json.loads((a.legacy / "histories.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    metrics = {}
    for i, (key, label, color) in enumerate(cases):
        h, prev = history[key], old[key]
        if i < 2:
            ax = axes[i]
            ax.plot(
                prev["redshifts"],
                prev["mean_xhii"],
                ls="--",
                color=".5",
                label="Historical cumulative",
            )
            ax.plot(h["redshifts"], h["mean_xhii"], color=color, label="Instantaneous source")
            ax.set(title=label, xlim=(20, 6), ylim=(0, 1.02), xlabel="Redshift")
            ax.axhline(0.5, color=".8", ls=":", lw=1)
            ax.legend(fontsize=10, loc="upper left")
        metrics[key] = {
            str(t): dict(
                current=crossing(h["redshifts"], h["mean_xhii"], t),
                historical=crossing(prev["redshifts"], prev["mean_xhii"], t),
            )
            for t in [0.1, 0.5, 0.9]
        }
        metrics[key]["decreasing_steps"] = int(np.sum(np.diff(h["mean_xhii"]) < 0))
        metrics[key]["max_local_balance_residual"] = max(
            abs(
                r["step"]["cell_step_emitted"]
                - r["step"]["cell_step_recombined"]
                - r["step"]["cell_step_surplus"]
                - r["step"]["cell_step_delta_q"]
            )
            for r in h["rows"][1:]
        )
    axes[0].set_ylabel(r"Volume-mean $x_{\rm HII}$")
    save(fig, a, "instantaneous_history_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    for i, (key, label, color) in enumerate(cases):
        h = history[key]
        zv = np.array(h["redshifts"])
        q = np.array([r["one_zone_xhii"] for r in h["rows"]])
        massq = np.array([r["mass_weighted_xhii"] for r in h["rows"]])
        if i < 2:
            axes[0].plot(zv, q, color=color, label=label)
            axes[1].plot(zv, massq - q, color=color, label=label)
        metrics[key]["max_map_onezone_difference"] = float(np.max(abs(massq - q)))
    axes[0].set(ylabel="Mean-density one-zone filling fraction", ylim=(0, 1.02))
    axes[1].set(ylabel=r"$\langle x_{\rm HII}\rangle_M-Q_{\rm one-zone}$")
    axes[1].axhline(0, color=".6", ls=":", lw=1)
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.set(xlabel="Redshift", xlim=(20, 6))
    save(fig, a, "instantaneous_spatial_reference")
    manifest = json.loads((a.run / "manifest.json").read_text())
    selected = (
        sorted(
            float(path.stem.rsplit("_z", 1)[1])
            for path in (a.run / "maps").glob("brightness_popii_z*.npy")
        )
        if a.primary_only
        else manifest["selected_map_redshifts"]
    )
    for zz in selected:
        fields = [
            np.load(a.run / "maps" / f"brightness_{key}_z{zz:.2f}.npy", mmap_mode="r")[150].copy()
            for key, _, _ in CASES[:2]
        ]
        difference = fields[1] - fields[0]
        peak = float(np.max(abs(difference)))
        if peak == 0:
            peak = 1.0  # Display interval for an exactly zero signal.
        vmax = max(float(v.max()) for v in fields)
        if vmax <= 0:
            # This redshift contains no remaining brightness signal; use the
            # physical 1 mK display interval and retain exactly zero pixels.
            vmax = 1.0
        fig = plt.figure(figsize=(12, 4.8))
        axes = [fig.add_axes([0.06 + i * 0.32, 0.27, 0.25, 0.625]) for i in range(3)]
        caxes = [fig.add_axes([0.06 + i * 0.32, 0.10, 0.25, 0.04]) for i in range(3)]
        for i, field in enumerate(fields + [difference]):
            kwargs = (
                dict(cmap="viridis", vmin=0, vmax=vmax)
                if i < 2
                else dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-peak, vcenter=0, vmax=peak))
            )
            im = axes[i].imshow(
                field.T, origin="lower", extent=(0, 300, 0, 300), interpolation="nearest", **kwargs
            )
            axes[i].set(
                title=["Pop II", "Pop II + Pop III", "Change from Pop III"][i],
                xlabel="y [cMpc]",
                ylabel="z coordinate [cMpc]",
            )
            fig.colorbar(
                im,
                cax=caxes[i],
                orientation="horizontal",
                label=r"$\delta T_b$ [mK]" if i < 2 else r"$\Delta\delta T_b$ [mK]",
            )
        fig.suptitle(f"Instantaneous-source maps at redshift {zz:.2f}", y=0.99)
        save(fig, a, f"instantaneous_map_z{zz:.2f}")
    rows = []
    for target in [6, 8, 10, 12.5, 15, 20]:
        i = np.flatnonzero(z == target)[0]
        rows.append(
            dict(
                z=target,
                rates=rate[i].tolist(),
                popiii_share=float(share[i]),
                popiii_share_mc_se=float(share_se[i]),
                quadrature_relative=float(budget[i]["quadrature_error_s_mpc3"] / rate[i, 1]),
            )
        )
    (a.previews / "results.json").write_text(
        json.dumps(dict(milestones=metrics, rate_rows=rows), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(milestones=metrics, rate_rows=rows), indent=2))


if __name__ == "__main__":
    main()
