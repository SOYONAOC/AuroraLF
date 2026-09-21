"""Matched-endpoint comparison of McBride and interval-censored THESAN crossings."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_save/atomic_crossing_mcbride_20260917"
TH = ROOT / "data_save/atomic_crossing_z6_20260917"
ASSETS = ROOT / "slides/atomic_crossing_z6/assets"
PREVIEW = ROOT / "outputs/atomic_crossing_mcbride_20260917"


def q(x):
    return np.quantile(x, [0.16, 0.5, 0.84], method="inverted_cdf")


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert (
        hashlib.sha256((OUT / "histories.npz").read_bytes()).hexdigest()
        == manifest["product_sha256"]
    )
    rows = list(csv.DictReader((TH / "thesan_halos.csv").open()))
    tm = json.loads((TH / "thesan_manifest.json").read_text())
    with np.load(OUT / "histories.npz") as d:
        m, z, status, zn, sn = [
            d[k]
            for k in ["mass_msun", "crossing_z", "status", "native_crossing_z", "native_status"]
        ]
        np.testing.assert_array_equal(d["subhalo_id"], [int(r["subhalo_id"]) for r in rows])
    np.testing.assert_array_equal(m, [float(r["mvir_msun"]) for r in rows])
    simlo = np.array([float(r["z_recorded"]) for r in rows])
    simhi = np.array([float(r["z_upper"]) if r["status"] == "bracketed" else np.inf for r in rows])
    logm = np.log10(m)
    floor = np.log10(tm["cooling_mass_msun"])
    records = []
    astro = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)
    for lo, hi in [(floor, 13), (9, 13), (floor, 9), (9, 10), (10, 11)]:
        select = (logm >= lo) & (logm < hi)
        zz = z[select]
        ss = status[select]
        zzn = zn[select]
        ssn = sn[select]
        assert np.all(ss > 0)
        qmodel = q(zz)
        qsimlo = q(simlo[select])
        qsimhi = q(simhi[select])
        qupper = q(np.where(ss == 2, np.inf, zz))
        # Conditional MC error of the CDF, holding the real endpoint sample fixed.
        probability = np.mean(zz <= qmodel[1], axis=1)
        se = np.sqrt(np.sum(probability * (1 - probability) / (zz.shape[1] - 1))) / len(probability)
        median_mc_interval = np.quantile(
            zz, [0.5 - 1.96 * se, 0.5 + 1.96 * se], method="inverted_cdf"
        )
        shift = astro.age(qsimlo[1]).to_value("Myr") - astro.age(qmodel[1]).to_value("Myr")
        shift_min = astro.age(qsimhi[1]).to_value("Myr") - astro.age(qmodel[1]).to_value("Myr")
        r = dict(
            logmass_low=lo,
            logmass_high=hi,
            sim_halos=int(select.sum()),
            model_tracks=int(zz.size),
            model_quantile_lower_z=qmodel.tolist(),
            model_quantile_upper_z=[float(v) if np.isfinite(v) else None for v in qupper],
            sim_quantile_lower_z=qsimlo.tolist(),
            sim_quantile_upper_z=[float(v) if np.isfinite(v) else None for v in qsimhi],
            model_median_mc_95_interval=median_mc_interval.tolist(),
            model_before_z50_fraction=float(np.mean(ss == 2)),
            model_crossed_by_z10_fraction=float(np.mean(zz >= 10)),
            sim_crossed_by_z10_fraction_bounds=[
                float(np.mean(simlo[select] >= 10)),
                float(np.mean(simhi[select] >= 10)),
            ],
            earlier_model_median_time_myr=[float(shift_min), float(shift)],
            native_median_z_conditional=float(
                np.quantile(zzn[ssn > 0], 0.5, method="inverted_cdf")
            ),
            native_no_crossing_fraction=float(np.mean(ssn == 0)),
            max_cosmology_root_shift=float(
                np.max(abs(zz[(ss == 1) & (ssn == 1)] - zzn[(ss == 1) & (ssn == 1)]))
            ),
        )
        records.append(r)
        print(json.dumps(r), flush=True)
    result = dict(
        records=records,
        interpretation="Same exact THESAN final masses and endpoint; parameter sampling unchanged. Matched Planck15 primary, native Planck18 sensitivity. Intervals are snapshot/censoring bounds; MC interval excludes sample/cosmic variance.",
        manifest_sha256=hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
    )
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    PREVIEW.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    grid = np.linspace(5.99, 49.99, 350)
    for ax, (lo, hi) in zip(axes, [(floor, 9), (9, 10), (10, 11)], strict=True):
        select = (logm >= lo) & (logm < hi)
        lower = np.array([np.mean(simhi[select] <= x) for x in grid])
        upper = np.array([np.mean(simlo[select] <= x) for x in grid])
        model = np.sort(z[select].ravel())
        mc = np.searchsorted(model, grid, side="right") / len(model)
        ax.fill_between(
            grid, lower, upper, color="#297bb1", alpha=0.24, label="THESAN interval bounds"
        )
        ax.plot(grid, upper, color="#297bb1", lw=1.4)
        ax.plot(grid, lower, color="#297bb1", lw=1.4)
        ax.plot(grid, mc, color="#c35c35", lw=2, label="AuroraLF / McBride")
        ax.axhline(0.5, color=".4", ls=":", lw=0.8)
        ax.set(xlim=(6, 50), ylim=(0, 1), xlabel=r"First-crossing redshift $z_{\rm cross}$")
        label = (
            r"$M_{\rm cool}\leq M_{\rm vir}<10^9$"
            if lo == floor
            else rf"$10^{{{lo}}}\leq M_{{\rm vir}}<10^{{{hi}}}$"
        )
        ax.set_title(label + r" $M_\odot$" + "\n" + f"{select.sum()} matched endpoints")
        ax.text(
            0.96,
            0.06,
            rf"$z_{{\rm cross}}>50$: {100 * np.mean(status[select] == 2):.1f}\%",
            transform=ax.transAxes,
            ha="right",
            fontsize=10,
        )
    axes[0].set_ylabel(r"Fraction with $z_{\rm cross}\leq z$")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "mcbride_comparison.pdf")
    fig.savefig(PREVIEW / "mcbride_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
