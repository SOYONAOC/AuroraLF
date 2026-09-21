"""Verify completed PISN extrapolation artifacts and plot independent diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.experiments.artifacts import digest


def load_verified(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run: {path}")
    for name, expected in manifest["products"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Product hash mismatch: {path / name}")
    summary = json.loads((path / "summary.json").read_text())
    if summary["config"] != manifest["config"]:
        raise ValueError("summary / manifest config mismatch")
    cfg = summary["config"]
    expected = {(z, nm, ng) for z in cfg["redshifts"] for nm, ng in cfg["resolutions"]}
    actual = {(c["z"], c["n_mass"], c["n_grid"]) for c in summary["cases"]}
    if expected != actual or len(actual) != len(summary["cases"]):
        raise ValueError("Missing / duplicate cases")
    for c in summary["cases"]:
        with np.load(path / f"z{c['z']:g}_m{c['n_mass']}_t{c['n_grid']}.npz") as data:
            for key in data.files:
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"nonfinite {key}")
            if not np.isclose(
                data["weight"] @ data["mean_rate"], c["source_rate"], rtol=1e-12, atol=0
            ):
                raise ValueError("NPZ / JSON rate mismatch")
    best = {
        (c["z"]): c
        for c in summary["cases"]
        if [c["n_mass"], c["n_grid"]] == cfg["resolutions"][-1]
    }
    return summary, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data_save/pisn_lowz_20260918"))
    parser.add_argument("--tail", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/pisn_lowz_20260918"))
    args = parser.parse_args()
    summary, best = load_verified(args.source)
    tail_summary, tail = load_verified(args.tail) if args.tail else (None, {})
    if tail_summary:
        keys = ["q_log10_mean", "q_log10_sigma", "epsilon_b", "z_start", "lifetimes"]
        if any(summary["config"][k] != tail_summary["config"][k] for k in keys):
            raise ValueError("Tail model differs from baseline")
        if summary["config"]["logmass_max"] != tail_summary["config"]["logmass_min"]:
            raise ValueError("Mass intervals must be adjacent")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for z, c in best.items():
        extra = tail.get(z)
        rows.append(
            dict(
                z=z,
                rate_gpc3_yr=(c["source_rate"] + (extra["source_rate"] if extra else 0)) * 1e9,
                mc_se_gpc3_yr=np.hypot(c["mah_mc_se"], extra["mah_mc_se"] if extra else 0) * 1e9,
                observer_rate=c["observer_rate"] + (extra["observer_rate"] if extra else 0),
                base_rate_gpc3_yr=c["source_rate"] * 1e9,
                tail_rate_gpc3_yr=extra["source_rate"] * 1e9 if extra else None,
                logmass_min=summary["config"]["logmass_min"],
                logmass_max=tail_summary["config"]["logmass_max"]
                if extra
                else summary["config"]["logmass_max"],
            )
        )
    suffix = "with_tail" if tail else "original_mass_range"
    with (args.output / f"rates_{suffix}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plt.style.use("apj")
    fig, (ax, mass_ax) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    zs = np.array([r["z"] for r in rows])
    rates = np.array([r["rate_gpc3_yr"] for r in rows])
    errors = np.array([r["mc_se_gpc3_yr"] for r in rows])
    ax.errorbar(zs, rates, yerr=errors, fmt="o", capsize=4, label="All PISNe: model extrapolation")
    ax.hlines(100, 1, 3, color="0.45", ls="--")
    ax.annotate(
        "HSC: luminous templates only\norder-of-magnitude bound",
        xy=(2, 100),
        xytext=(4.8, 65),
        fontsize=10,
        color="0.35",
        arrowprops=dict(arrowstyle="->", color="0.45"),
    )
    ax.annotate("", xy=(1.4, 70), xytext=(1.4, 100), arrowprops=dict(arrowstyle="->", color="0.45"))
    for z, rate in zip(zs, rates, strict=True):
        ax.annotate(
            f"{rate:.1f}",
            xy=(z, rate),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xlabel="Redshift",
        ylabel=r"PISN / source yr / cGpc$^3$",
        yscale="log",
        xlim=(0.4, 15.2),
        ylim=(min(10, rates.min() / 1.6), rates.max() * 2.6),
        xticks=zs,
    )
    ax.legend(fontsize=10, loc="upper left")
    for z in [1.0, 2.0, 3.0]:
        c = best[z]
        pieces = [(args.source, c)]
        if z in tail:
            pieces.append((args.tail, tail[z]))
        masses, contributions = [], []
        for path, case in pieces:
            with np.load(path / f"z{z:g}_m{case['n_mass']}_t{case['n_grid']}.npz") as d:
                lm = np.log10(d["mass_msun"])
                masses.extend(lm)
                contributions.extend(d["weight"] * d["mean_rate"] * 1e9 / (lm[1] - lm[0]))
        mass_ax.plot(masses, contributions, label=f"z={z:g}")
    mass_ax.axvline(12, color="0.4", ls="--", lw=1)
    mass_ax.set(
        xlabel=r"$\log_{10}(M_{h,\mathrm{obs}}/M_\odot)$",
        ylabel=r"PISN / source yr / cGpc$^3$ / dex",
        yscale="log",
        ylim=(0.02, 1000),
    )
    mass_ax.legend(fontsize=10)
    mass_ax.text(
        0.03,
        0.04,
        "Dashed: original upper mass limit",
        transform=mass_ax.transAxes,
        fontsize=10,
        color="0.35",
    )
    domain = (
        r"Low-z halo masses: $10^5$--$10^{15}$ $M_\odot$"
        if tail
        else r"Halo masses: $10^5$--$10^{12}$ $M_\odot$ (upper range incomplete at low z)"
    )
    fig.suptitle("No enrichment / pristine-gas model; no survey selection\n" + domain, fontsize=12)
    fig.savefig(args.output / f"rates_{suffix}.pdf")
    fig.savefig(args.output / f"rates_{suffix}.png", dpi=150)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
