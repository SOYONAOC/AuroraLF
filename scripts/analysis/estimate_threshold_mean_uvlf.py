"""Importance-sampling estimate of log10(q) mean 0.5 -> 0 at fixed 1.5 dex.

Uses the exact archived high-z samples behind the main slides. The Pop II
baseline is a control variate because its distribution is independent of q.
This changes the ensemble measure, including burst age and first-crossing mass;
it does not rescale individual stellar masses or luminosities.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from auroralf.experiments.artifacts import digest, read_completed_manifest
from auroralf.uvlf import uv_luminosity_to_muv

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "AuroraLF-visbal-duty"
SAVE = ROOT / "data_save/threshold_mean_sensitivity_20260918"
OUT = ROOT / "outputs/threshold_mean_sensitivity_20260918"
ASSETS = ROOT / "slides/popiii_heii_pisn_complete_20260916/assets"
INPUTS = {12.5: "random_q_z12p5_20260909", 14.5: "random_q_combined_20260909_rev02"}
EFFICIENCY = 0.03


def histogram_rows(magnitude, edges, factor):
    return np.array(
        [np.histogram(m, bins=edges, weights=r)[0] for m, r in zip(magnitude, factor, strict=True)]
    )


def combine(batches):
    """Equal independent batches; correlated old/new mass-cluster errors."""
    old = np.mean([b["old"].sum(axis=0) for b in batches], axis=0)
    new = np.mean([b["new"].sum(axis=0) for b in batches], axis=0)
    ratio = new / old
    variance = (
        sum(len(b["old"]) * np.var(b["new"] - ratio * b["old"], axis=0, ddof=1) for b in batches)
        / len(batches) ** 2
    )
    return {
        "old": old.tolist(),
        "new": new.tolist(),
        "ratio": ratio.tolist(),
        "ratio_mc_se": (np.sqrt(variance) / old).tolist(),
        "baseline": np.mean([b["baseline"].sum(axis=0) for b in batches], axis=0).tolist(),
        "raw_importance_new": np.mean([b["raw"].sum(axis=0) for b in batches], axis=0).tolist(),
        "counts": np.sum([b["counts"] for b in batches], axis=0).tolist(),
        "per_batch_ratio": [
            (b["new"].sum(axis=0) / b["old"].sum(axis=0)).tolist() for b in batches
        ],
    }


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    edges = np.arange(-22.5, -17.4, 1.0)
    cuts = np.array([-22.0, -21.0, -20.0, -19.0, -18.0])
    report = dict(
        source_mu=0.5,
        target_mu=0.0,
        sigma_dex=1.5,
        epsilon_b=EFFICIENCY,
        bin_edges=edges.tolist(),
        cumulative_cuts=cuts.tolist(),
        method="r=exp(((x-0.5)^2-x^2)/(2*1.5^2)); phi_new=phi_old+sum w*(r-1)*(I_total-I_PopII)",
        limitations=[
            "Reweighting estimate, not an independent rerun or observational fit.",
            "Same finite halo-mass range, 100 Myr UV window, SSP and dust-free assumptions as source slides.",
            "MC errors are paired mass-cluster errors; no physical-model uncertainty included.",
        ],
        sources={},
        redshifts={},
    )
    for z, name in INPUTS.items():
        directory = SOURCE / "data_save" / name
        summary = json.loads((directory / "summary.json").read_text())
        report["sources"][str(directory / "summary.json")] = digest(directory / "summary.json")
        with np.load(directory / "uvlf.npz") as cached:
            cached_edges = cached["bin_edges"].copy()
            cached_old = cached["eps0.03"].copy()
        bins, cumulative, recovered, diagnostics = [], [], [], []
        for run in summary["runs"]:
            path = SOURCE / run
            manifest_path = path / "manifest.json"
            sha = digest(manifest_path)
            if sha != summary["manifests_sha256"][str(Path(run) / "manifest.json")]:
                raise ValueError(f"Source manifest changed: {path}")
            manifest = read_completed_manifest(path, required_products=("samples.npz",))
            cfg = manifest["config"]
            if (cfg["z"], cfg["q_log10_mean"], cfg["q_log10_sigma"]) != (z, 0.5, 1.5):
                raise ValueError(f"Incompatible source: {path}")
            report["sources"][str(manifest_path)] = sha
            report["sources"][str(path / "samples.npz")] = manifest["products"]["samples.npz"]
            with np.load(path / "samples.npz") as data:
                p2 = data["popii"]
                p3 = data["popiii_per_efficiency"]
                logq = data["logq"]
                w = data["weight_per_track"][:, None]
            if not all(np.isfinite(a).all() for a in (p2, p3, logq, w)):
                raise ValueError("Nonfinite samples")
            if p2.shape != (cfg["n_mass"], cfg["n_tracks"]) or logq.shape != p2.shape:
                raise ValueError("Inconsistent sample shape")
            r = np.exp(((logq - 0.5) ** 2 - logq**2) / (2 * 1.5**2))
            ones = np.ones_like(r)
            m2 = uv_luminosity_to_muv(p2)
            mt = uv_luminosity_to_muv(p2 + EFFICIENCY * p3)
            recovered.append(
                (histogram_rows(mt, cached_edges, ones) * w).sum(axis=0) / np.diff(cached_edges)
            )
            base = histogram_rows(m2, edges, ones) * w
            old = histogram_rows(mt, edges, ones) * w
            delta = (histogram_rows(mt, edges, r - 1) - histogram_rows(m2, edges, r - 1)) * w
            raw = histogram_rows(mt, edges, r) * w
            # Both densities have identical normalization. A zero shift has r=1
            # and identically vanishing paired delta, independent of sampling.
            bins.append(
                dict(
                    old=old,
                    new=old + delta,
                    baseline=base,
                    raw=raw,
                    counts=histogram_rows(mt, edges, ones).sum(axis=0),
                )
            )
            ib = np.stack([(m2 <= cut).sum(axis=1) for cut in cuts], axis=1)
            it = np.stack([(mt <= cut).sum(axis=1) for cut in cuts], axis=1)
            dt = np.stack(
                [((r - 1) * ((mt <= cut).astype(float) - (m2 <= cut))).sum(axis=1) for cut in cuts],
                axis=1,
            )
            raw_c = np.stack([(r * (mt <= cut)).sum(axis=1) for cut in cuts], axis=1)
            cumulative.append(
                dict(
                    old=it * w,
                    new=(it + dt) * w,
                    baseline=ib * w,
                    raw=raw_c * w,
                    counts=it.sum(axis=0),
                )
            )
            diagnostics.append(
                dict(
                    run=run,
                    likelihood_mean=float(r.mean()),
                    likelihood_ess_fraction=float(r.sum() ** 2 / (r.size * (r**2).sum())),
                )
            )
            print(f"Validated and reweighted z={z}, {cfg['run_id']}", flush=True)
        np.testing.assert_allclose(np.mean(recovered, axis=0), cached_old, rtol=2e-12, atol=0)
        result = dict(
            differential=combine(bins), cumulative=combine(cumulative), diagnostics=diagnostics
        )
        if min(result["differential"]["new"]) <= 0:
            raise ValueError("Nonpositive control-variate estimate")
        report["redshifts"][str(z)] = result
    report["sources"][str(Path(__file__).resolve())] = digest(__file__)
    (SAVE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report)
    for z, result in report["redshifts"].items():
        print(
            z,
            "Muv",
            ((edges[1:] + edges[:-1]) / 2).tolist(),
            "ratio",
            result["differential"]["ratio"],
            flush=True,
        )


def plot(report):
    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Arial",
            "font.size": 13,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 4.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    edges = np.array(report["bin_edges"])
    x = (edges[1:] + edges[:-1]) / 2
    for col, (z, result) in enumerate(report["redshifts"].items()):
        d = result["differential"]
        ax, ratio_ax = axes[:, col]
        ax.plot(x, d["old"], color="#2b6d99", label=r"$\mu=0.5$ (current)")
        ax.plot(x, d["new"], color="#dc7734", label=r"$\mu=0$ (reweighted)")
        ax.plot(x, d["baseline"], "--", color="#808080", label="Pop II only")
        ax.set(yscale="log", title=f"$z={z}$", ylim=(1e-10, 8e-4))
        ratio_ax.errorbar(
            x, d["ratio"], yerr=d["ratio_mc_se"], color="#dc7734", fmt="o-", capsize=3
        )
        ratio_ax.axhline(1, color="#808080", linestyle=":")
        ratio_ax.set(xlim=(-22.2, -17.8), ylim=(0.5, 1.05), xlabel=r"$M_{\rm UV}$", xticks=x)
    axes[0, 0].set_ylabel(r"$\phi\ [\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}]$")
    axes[1, 0].set_ylabel(r"$\phi_{\mu=0}/\phi_{\mu=0.5}$")
    axes[0, 1].legend(fontsize=11)
    fig.savefig(ASSETS / "threshold_mean_uvlf.pdf")
    fig.savefig(OUT / "threshold_mean_uvlf.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
