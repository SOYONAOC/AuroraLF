"""Summarize real z~6 crossing records, preserving interval/left censoring."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_save/atomic_crossing_z6_20260917"
ASSETS = ROOT / "slides/atomic_crossing_z6/assets"
PREVIEW = ROOT / "outputs/atomic_crossing_z6_20260917"


def quantiles(values, weights):
    order = np.argsort(values)
    x, w = np.array(values)[order], np.array(weights)[order]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), [0.16, 0.5, 0.84])]


def main():
    th = list(csv.DictReader((OUT / "thesan_halos.csv").open()))
    meta = json.loads((OUT / "thesan_manifest.json").read_text())
    tng_dir = ROOT / "data_save/tng_pristine_20260914"
    tg = [r for r in csv.DictReader((tng_dir / "halo_diagnostics.csv").open()) if float(r["z"]) < 7]
    tm = json.loads((tng_dir / "manifest.json").read_text())
    assert (
        hashlib.sha256((tng_dir / "halo_diagnostics.csv").read_bytes()).hexdigest()
        == tm["diagnostics_sha256"]
    )
    zgrid, tgrid = np.array(tm["snapshot_z"]), np.array(tm["snapshot_time_myr"])
    summaries = []
    floor = np.log10(meta["cooling_mass_msun"])
    bins = [(floor, 13), (9, 13), (floor, 9), (9, 10), (10, 11), (11, 12), (12, 13)]
    for lo, hi in bins:
        a = [r for r in th if lo <= np.log10(float(r["mvir_msun"])) < hi]
        if not a:
            continue
        assert all(r["status"] != "never" for r in a)
        low = np.array([float(r["z_recorded"]) for r in a])
        up = np.array([float(r["z_upper"]) if r["status"] == "bracketed" else np.inf for r in a])
        qlo, qhi = quantiles(low, np.ones(len(a))), quantiles(up, np.ones(len(a)))
        bracket = [r for r in a if r["status"] == "bracketed"]
        summaries.append(
            dict(
                simulation="THESAN-HR-large",
                logmass_low=lo,
                logmass_high=hi,
                count=len(a),
                bracketed_count=len(bracket),
                censored_count=sum(r["status"] == "censored" for r in a),
                incomplete_count=sum(r["status"] == "incomplete_early_history" for r in a),
                quantile_lower_z=qlo.tolist(),
                quantile_upper_z=[float(v) if np.isfinite(v) else None for v in qhi],
                fraction_crossed_by_z10_min=float(np.mean(low >= 10)),
                fraction_crossed_by_z10_max=float(np.mean(up >= 10)),
                bracketed_only_interp_quantiles=quantiles(
                    [float(r["z_interp"]) for r in bracket], np.ones(len(bracket))
                ).tolist(),
            )
        )
    for lo, hi in [(9, 13), (9, 10), (10, 11), (11, 12), (12, 13)]:
        a = [r for r in tg if lo <= float(r["logmass_low"]) < hi]
        for branch in ["main", "all"]:
            for n in [20, 100]:
                w = np.array([float(r["weight"]) for r in a])
                lb = np.array([float(r[f"lookback_{branch}_n{n}_myr"]) for r in a])
                found = lb >= 0
                zz = []
                for r, v in zip(a, lb, strict=True):
                    if v < 0:
                        continue
                    time = tgrid[int(r["snapshot"])] - v
                    k = int(np.argmin(abs(tgrid - time)))
                    assert abs(tgrid[k] - time) < 1e-6
                    zz.append(zgrid[k])
                summaries.append(
                    dict(
                        simulation="TNG100-1-Dark",
                        logmass_low=lo,
                        logmass_high=hi,
                        count=len(a),
                        catalog_count=float(w.sum()),
                        branch=branch,
                        nmin=n,
                        unknown_fraction=float(w[~found].sum() / w.sum()),
                        recorded_z_quantiles_conditional=quantiles(zz, w[found]).tolist(),
                    )
                )
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    ASSETS.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    colors = ["#2477aa", "#dc893b", "#8c5ba5", "#488864"]
    for (lo, hi), color in zip([(floor, 9), (9, 10), (10, 11), (11, 12)], colors, strict=True):
        a = [r for r in th if lo <= np.log10(float(r["mvir_msun"])) < hi]
        if not a:
            continue
        low = np.array([float(r["z_recorded"]) for r in a])
        up = np.array([float(r["z_upper"]) if r["status"] == "bracketed" else np.inf for r in a])
        grid = np.linspace(5.99, 20.5, 400)
        cdf_low = np.array([np.mean(up <= x) for x in grid])
        cdf_high = np.array([np.mean(low <= x) for x in grid])
        label = r"$M_{\rm cool}\! -\!10^9$" if lo == floor else rf"$10^{{{lo}}}\! -\!10^{{{hi}}}$"
        axes[0].fill_between(grid, cdf_low, cdf_high, color=color, alpha=0.18)
        axes[0].plot(grid, cdf_high, color=color, label=label)
    axes[0].set(
        xlim=(6, 20.5),
        ylim=(0, 1),
        xlabel=r"First-crossing redshift $z_{\rm cross}$",
        ylabel=r"Fraction with $z_{\rm cross}\leq z$",
        title=r"THESAN-HR-large: $z=5.994$",
    )
    axes[0].legend(title=r"$M_{\rm vir}(z\simeq6)\ [M_\odot]$", fontsize=9, loc="lower right")
    axes[0].axhline(0.5, color=".5", lw=0.8, ls=":")
    for n, color in [(20, colors[1]), (100, colors[2])]:
        a = [
            r
            for r in summaries
            if r["simulation"] == "TNG100-1-Dark"
            and r["branch"] == "main"
            and r["nmin"] == n
            and r["logmass_high"] - r["logmass_low"] == 1
        ]
        x = np.array([r["logmass_low"] + 0.5 for r in a])
        q = np.array([r["recorded_z_quantiles_conditional"] for r in a])
        axes[1].errorbar(
            x + (0.05 if n == 100 else -0.05),
            q[:, 1],
            yerr=np.array([q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]]),
            fmt="o",
            color=color,
            capsize=4,
            label=rf"$N_{{\rm DM}}\geq{n}$",
        )
    axes[1].set(
        xlim=(9, 13),
        ylim=(6, 22),
        xlabel=r"$\log_{10}[M_{200c}(z=6.01)/M_\odot]$",
        ylabel="Earliest recorded eligible redshift",
        title="TNG100-Dark: resolution-limited records",
    )
    axes[1].legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "crossings.pdf")
    fig.savefig(PREVIEW / "crossings.png", dpi=150)
    plt.close(fig)
    for r in summaries:
        if r["simulation"] == "THESAN-HR-large":
            print(json.dumps(r))


if __name__ == "__main__":
    main()
