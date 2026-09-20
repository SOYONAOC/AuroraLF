"""Tabulate all computed redshifts and plot cumulative escaped photon shares."""

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
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data_save/ionizing_sources/photon_shares"))
    p.add_argument("--assets", type=Path, default=Path("slides/21cm_map/assets/results"))
    p.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/previews"))
    a = p.parse_args()
    for directory in [a.output, a.assets, a.previews]:
        directory.mkdir(parents=True, exist_ok=True)
    source = a.run / "photon_budget.json"
    manifest = json.loads((a.run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("incomplete source run")
    budget = sorted(json.loads(source.read_text()), key=lambda r: -r["z"])
    z = np.array([r["z"] for r in budget])
    counts = np.array([r["photons_per_h"] for r in budget])
    if (
        z.ndim != 1
        or len(z) < 2
        or counts.shape != (len(z), 3)
        or not np.isfinite(z).all()
        or np.any(z < 0)
        or np.any(np.diff(z) >= 0)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise ValueError("invalid redshifts or photon counts")
    total_resolved, total_upper = counts[:, :2].sum(axis=1), counts.sum(axis=1)
    if np.any(total_resolved <= 0) or np.any(total_upper <= 0):
        raise ValueError("photon shares undefined for a zero total")
    shares_resolved = counts[:, :2] / total_resolved[:, None]
    shares_upper = counts / total_upper[:, None]
    np.testing.assert_allclose(shares_resolved.sum(axis=1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(shares_upper.sum(axis=1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(shares_upper * total_upper[:, None], counts, rtol=1e-14)
    rows = []
    for i, redshift in enumerate(z):
        rows.append(
            dict(
                redshift=float(redshift),
                popii_photons_per_h=counts[i, 0],
                popiii_resolved_photons_per_h=counts[i, 1],
                popiii_prestart_upper_extra_photons_per_h=counts[i, 2],
                resolved_total_photons_per_h=total_resolved[i],
                resolved_popii_fraction=shares_resolved[i, 0],
                resolved_popiii_fraction=shares_resolved[i, 1],
                upper_total_photons_per_h=total_upper[i],
                upper_popii_fraction=shares_upper[i, 0],
                upper_popiii_resolved_fraction=shares_upper[i, 1],
                upper_popiii_prestart_extra_fraction=shares_upper[i, 2],
                upper_popiii_total_fraction=shares_upper[i, 1:].sum(),
            )
        )
    with (a.output / "photon_shares.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                source=str(source.resolve()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                csv_sha256=hashlib.sha256(
                    (a.output / "photon_shares.csv").read_bytes()
                ).hexdigest(),
                redshifts=z.tolist(),
                fractions="0 to 1; counts are cumulative escaped photons per cosmic H atom",
                resolved_denominator="N_II + N_III_resolved",
                upper_denominator="N_II + N_III_resolved + N_III_prestart_upper_extra",
                limits="No redshift extrapolation, no instantaneous emission rate inferred; upper extra is a scenario bound, not a measured component or confidence interval",
            ),
            indent=2,
        )
        + "\n"
    )

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
    selected = z <= 20
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    labels = ["Pop II", "Resolved Pop III", "Pre-start Pop III upper extra"]
    colors = ["#286491", "#b86632", "#657349"]
    for ax, shares, title in zip(
        axes,
        [shares_resolved, shares_upper],
        ["Resolved budget", "Including pre-start upper contribution"],
        strict=True,
    ):
        x = np.arange(np.count_nonzero(selected))
        bottom = np.zeros_like(x, dtype=float)
        for j in range(shares.shape[1]):
            values = shares[selected, j] * 100
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.72,
                label=labels[j],
                color=colors[j],
                hatch="///" if j == 2 else None,
                edgecolor="white",
                linewidth=0.5,
            )
            for pos, low, val in zip(x, bottom, values, strict=True):
                if val >= 9:
                    ax.text(
                        pos,
                        low + val / 2,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=11,
                        bbox={"facecolor": colors[j], "edgecolor": "none", "pad": 0.5}
                        if j == 2
                        else None,
                    )
            bottom += values
        ax.set(
            xticks=x,
            xticklabels=[f"{v:g}" for v in z[selected]],
            ylim=(0, 100),
            xlabel="Redshift (time increases to the right)",
            title=title,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_ylabel("Share of cumulative escaped photons")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncols=3, fontsize=11)
    save(fig, a.assets, a.previews, "photon_shares")
    print(f"Verified shares at all {len(z)} source redshifts; CSV and plot written.")


if __name__ == "__main__":
    main()
