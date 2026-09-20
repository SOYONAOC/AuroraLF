"""Compare current escaped emission with cumulative Pop III source fractions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
from plot_ionizing_maps import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("slides/21cm_map/assets/results"))
    p.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/previews"))
    a = p.parse_args()
    s = json.loads((a.data / "summary.json").read_text())
    if s["status"] != "complete":
        raise ValueError("source audit incomplete")
    for name, expected in s["products_sha256"].items():
        if hashlib.sha256((a.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"audit product changed: {name}")
    for d in (a.assets, a.previews):
        d.mkdir(parents=True, exist_ok=True)
    rows = s["rows"]
    refined = json.loads((a.refined / "summary.json").read_text())
    if refined["status"] != "complete" or refined["source_sha256"] != s["source_sha256"]:
        raise ValueError("incomplete or unmatched refined audit")
    for name, expected in refined["products_sha256"].items():
        if hashlib.sha256((a.refined / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"refined audit product changed: {name}")
    rate_rows = {r["z"]: r for r in rows}
    rate_rows.update({r["z"]: r for r in refined["rows"]})
    comparison = []
    for row in rows:
        current_row = rate_rows[row["z"]]
        comparison.append(
            {
                "redshift": row["z"],
                "current_rate_tracks": refined["tracks"]
                if row["z"] <= 8
                else s["model"]["n_tracks"],
                "current_popii_fraction": 1 - current_row["current_popiii_share"],
                "current_popiii_fraction": current_row["current_popiii_share"],
                "current_fraction_mc_se": current_row["current_popiii_share_mc_se"],
                "cumulative_popii_fraction_original": 1 - row["cumulative_popiii_share"],
                "cumulative_popiii_fraction_original": row["cumulative_popiii_share"],
                "popii_current_photons_s_mpc3": current_row["current_escaped_rate_s_mpc3"][0],
                "popiii_current_photons_s_mpc3": current_row["current_escaped_rate_s_mpc3"][1],
            }
        )
    with (a.data / "rate_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    z = np.array([r["z"] for r in rows])
    rates = np.array([rate_rows[r["z"]]["current_escaped_rate_s_mpc3"] for r in rows])
    current = np.array([rate_rows[r["z"]]["current_popiii_share"] for r in rows])
    cumulative = np.array([r["cumulative_popiii_share"] for r in rows])
    se = np.array([rate_rows[r["z"]]["current_popiii_share_mc_se"] for r in rows])
    rate_errors = {}
    with np.load(a.data / "cells.npz") as cells, np.load(a.data / "weights.npz") as w:
        variance = np.diagonal(cells["rate_covariance_of_mean"], axis1=-2, axis2=-1)
        errors = np.sqrt(np.sum(w["weight_per_h"][:, :, None] ** 2 * variance, axis=1)) * float(
            w["n_h_mpc3"]
        )
        rate_errors.update(zip(cells["redshifts"], errors, strict=True))
    with np.load(a.refined / "cells.npz") as cells:
        variance = np.diagonal(cells["rate_covariance_of_mean"], axis1=-2, axis2=-1)
        errors = np.sqrt(np.sum(cells["weight_per_h"][:, :, None] ** 2 * variance, axis=1)) * float(
            cells["n_h_mpc3"]
        )
        rate_errors.update(zip(cells["redshifts"], errors, strict=True))
    rate_se = np.array([rate_errors[redshift] for redshift in z])
    if np.any(rates <= 0) or not np.isfinite(rates).all():
        raise ValueError("invalid current rates")
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
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    for i, (label, color) in enumerate([("Pop II", "#286491"), ("Resolved Pop III", "#b86632")]):
        axes[0].errorbar(
            z, rates[:, i], yerr=rate_se[:, i], fmt="o-", capsize=3, label=label, color=color
        )
    axes[0].set_yscale("log")
    axes[0].set(
        ylabel=r"Escaped ionizing emissivity [s$^{-1}$ cMpc$^{-3}$]",
        title="Current photon production",
    )
    axes[1].plot(z, cumulative, "o-", label="Cumulative photon share", color="#b86632")
    axes[1].errorbar(
        z,
        current,
        yerr=se,
        fmt="s--",
        label="Current rate share (MC SE)",
        capsize=3,
        color="#286491",
    )
    axes[1].axhline(0.5, color="0.5", lw=1, ls=":")
    axes[1].set(ylabel="Resolved Pop III share", ylim=(0, 1.03), title="Different time definitions")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set(xlim=(20.5, 5.5), xlabel="Redshift (time increases to the right)")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.15)
    save(fig, a.assets, a.previews, "ionizing_current_vs_cumulative")
    table = []
    for r in rows:
        for epsilon in [0.03, 0.01, 0.003]:
            factor = epsilon / s["model"]["epsilon_b"]
            n2, n3 = r["cumulative_photons_per_h"]
            q2, q3 = rate_rows[r["z"]]["current_escaped_rate_s_mpc3"]
            table.append(
                dict(
                    z=r["z"],
                    epsilon_b=epsilon,
                    cumulative_popiii_share=factor * n3 / (n2 + factor * n3),
                    current_popiii_share=factor * q3 / (q2 + factor * q3),
                )
            )
    with (a.data / "amplitude_sensitivity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    tex = [
        r"\begin{tabular}{L{0.13\textwidth}L{0.36\textwidth}L{0.37\textwidth}}",
        r"\toprule",
        r"红移 & 累计 Pop III 光子中的占比 & 当前 Pop III 光子率中的占比\\",
        r"\midrule",
    ]
    for r in reversed(rows):
        n = r["popiii_cumulative_with_prior_popii"] * 100
        q = rate_rows[r["z"]]["popiii_current_with_prior_popii"] * 100
        tex.append(f"{r['z']:g} & {n:.2f}\\% & {q:.2f}\\%" + r"\\[0.4em]")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (a.assets / "ionizing_order_table.tex").write_text("\n".join(tex) + "\n")
    print("Current-rate plot and fixed-history source amplitude sensitivity saved.")


if __name__ == "__main__":
    main()
