"""Render existing high-z He II results using the L_1640--z axes of V24 Fig. 2.

Only the plotting convention is adopted. The model selection and population
quantiles remain AuroraLF's; no instrumental sensitivity or new histories are
inferred. Run from the repository root with PYTHONPATH=. .venv/bin/python.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data_save/heii_observations_20260910/summary.json"
DECK = ROOT / "slides/popiii_heii_pisn_complete_20260916"
OUTPUT = ROOT / "outputs/heii_literature_figures_20260916"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--asset-name", default="heii_highz_luminosity")
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    summary = source["efficiencies"][str(args.epsilon)] if "efficiencies" in source else source
    if summary["epsilon"] != args.epsilon:
        raise ValueError("Requested efficiency is not present in the saved summary")
    for path, expected in source["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Changed source: {path}")

    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    # Common observation-scale limits across efficiencies. Extremely faint
    # evolved bursts are retained in statistics and explicitly marked off-axis.
    y_floor, y_ceiling = 1e40, 1e43
    records = []
    for i, result in enumerate(summary["results"]):
        target = result["target"]
        # The saved conversion includes target D_L and lensing. Inverting it
        # recovers intrinsic luminosity without reassigning the population z.
        conversion = result["flux_per_luminosity"]
        window = result["target_windows"][0]
        stats = window["methods"]["linear_log_age"]
        q = np.array(stats["flux_q16_q50_q84"]) / conversion
        observed = target["flux"] / conversion
        if not np.all(np.isfinite(q)) or np.any(q <= 0) or np.any(np.diff(q) < 0):
            raise ValueError("Invalid model luminosity quantiles")
        if not y_floor < q[1] <= q[2] < y_ceiling:
            raise ValueError("Model median/upper quantile needs wider common axis limits")
        lower_off_axis = bool(q[0] < y_floor)
        ax.errorbar(
            target["population_z"],
            q[1],
            yerr=[[q[1] - (1.35 * y_floor if lower_off_axis else q[0])], [q[2] - q[1]]],
            fmt="o",
            color="#286491",
            mfc="white",
            mew=2,
            ms=9,
            elinewidth=2.2,
            capsize=7,
            label="Model: median and 16–84% population range" if i == 0 else None,
        )
        if lower_off_axis:
            ax.annotate(
                "",
                xy=(target["population_z"], 1.03 * y_floor),
                xytext=(target["population_z"], 2.0 * y_floor),
                arrowprops=dict(arrowstyle="->", color="#286491", lw=2),
            )
            exponent = int(np.floor(np.log10(q[0])))
            ax.text(
                target["population_z"] + 0.1,
                1.4 * y_floor,
                rf"16th percentile: ${q[0] / 10**exponent:.2f}\times10^{{{exponent}}}$"
                + "\n(below axis)",
                color="#286491",
                fontsize=12,
                va="bottom",
            )
        upper = target["measurement"] == "upper_limit"
        error = observed * 0.4 if upper else target["flux_error"] / conversion
        ax.errorbar(
            target["z"],
            observed,
            yerr=error,
            uplims=upper,
            fmt="s",
            color="#c34435",
            ms=7,
            capsize=4,
            elinewidth=1.8,
            label="Observation: value ±1σ / 3σ upper limit" if i == 0 else None,
        )
        label = "GHZ2 · Castellano+24" if i == 0 else "GS-z14-1 · Wu+25"
        ax.text(
            target["z"],
            0.94,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
        )
        ax.text(
            target["population_z"] + 0.10,
            q[1],
            rf"$M_{{\rm UV}}={target['muv']:.2f}\pm0.25$" + "\n(UV selection window)",
            va="center",
            fontsize=12,
            color="#286491",
        )
        records.append(
            {
                "target": target["name"],
                "observed_z": target["z"],
                "population_z": target["population_z"],
                "model_luminosity_q16_q50_q84": q.tolist(),
                "observed_luminosity_or_upper_limit": observed,
                "measurement": target["measurement"],
                "muv_window": [window["muv_low"], window["muv_high"]],
                "effective_mass_clusters": stats["effective_mass_clusters"],
                "lower_quantile_below_axis": lower_off_axis,
                "unknown_weight_fraction": stats["unknown_weight_fraction"],
                "zero_fraction_known": stats["zero_fraction_known"],
            }
        )
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"Intrinsic $L_{\mathrm{He\,II}\,1640}$ [erg s$^{-1}$]",
        yscale="log",
        xlim=(11.9, 14.9),
        ylim=(y_floor, y_ceiling),
    )
    ax.set_xticks([12, 12.5, 13, 13.5, 14, 14.5])
    ax.grid(axis="y", which="major", alpha=0.15)
    fig.legend(loc="upper center", bbox_to_anchor=(0.52, 1), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.87)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK / "assets" / f"{args.asset_name}.pdf")
    fig.savefig(OUTPUT / f"{args.asset_name}.png", dpi=150)
    plt.close(fig)
    record = {
        "summary": str(args.summary.resolve().relative_to(ROOT)),
        "summary_sha256": digest(args.summary),
        "script_sha256": digest(Path(__file__)),
        "epsilon": summary["epsilon"],
        "reference": "Venditti et al. 2024, arXiv:2405.10940v2, Fig. 2",
        "adopted": "Intrinsic He II luminosity versus actual redshift; measurements and upper limits",
        "not_adopted": "V24 mean/efficiency envelope, IMF, selection, mass-loss cases and ETC thresholds",
        "model_selection": "Target-specific UV window; no age cut; unknown histories excluded; known zeros retained",
        "caveats": source["assumptions"],
        "targets": records,
    }
    (OUTPUT / f"{args.asset_name}-provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
