"""Plot continuum-selected and line-selected He II populations separately."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import font_manager
from matplotlib.lines import Line2D

from auroralf.experiments.artifacts import digest
from scripts.analysis.heii_v24_comparison import observations

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "slides/popiii_heii_pisn_complete_20260916"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data_save/heii_survey_depth_20260918/summary.json",
    )
    args = parser.parse_args()
    args.summary = args.summary.resolve()
    manifest = json.loads(args.summary.with_name("manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or digest(args.summary) != manifest["products"]["summary.json"]
    ):
        raise ValueError("Unverified survey-depth summary")
    summary = json.loads(args.summary.read_text())
    fixed_uv = summary.get("selection_mode") == "absolute_uv"
    sources = observations()["sources"]
    high_path = ROOT / "external_data/observations/heii/jwst_targets.json"
    targets = json.loads(high_path.read_text())["targets"]
    astro = FlatLambdaCDM(**summary["cosmology"])
    for target in targets:
        # Actual observational lens correction, separate from the unlensed
        # reference-survey populations. No observation is recut by that survey.
        factor = 4 * np.pi * astro.luminosity_distance(target["z"]).to_value("cm") ** 2
        factor /= target["magnification"]
        upper = target["measurement"] == "upper_limit"
        sources.append(
            {
                "id": target["name"],
                "z": target["z"],
                "primary": True,
                "measurement": "upper_limit" if upper else "detection",
                "luminosity": target["flux"] * factor,
                "luminosity_error": None if upper else target["flux_error"] * factor,
                "limit_sigma": target["upper_limit_sigma"] if upper else None,
            }
        )
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
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ymin, ymax = 3e38, 2e43
    floor = ymin * 1.3
    records = []

    def population(row, key, offset, color, label):
        stats = row[key]
        q = stats["q16_q50_q84"]
        x = row["z"] + offset
        if q is None:
            ax.text(x, ymin * 1.8, "0 selected", rotation=90, color=color, fontsize=9)
            return
        q = np.array(q)
        if not np.isfinite(q).all() or np.any(q < 0) or np.any(np.diff(q) < 0):
            raise ValueError("Invalid luminosity quantiles")
        if q[-1] >= ymax:
            raise ValueError("Increase upper plot limit")
        visible = np.maximum(q, floor)
        ax.plot([x, x], [visible[0], visible[2]], color=color, lw=1.6)
        for value in (q[0], q[2]):
            if value >= floor:
                ax.plot(x, value, marker="_", ms=10, mew=1.6, color=color)
        # Off-scale medians are triangles; do not invent a circle at the floor.
        ax.plot(
            x,
            visible[1],
            "v" if q[1] < floor else "o",
            color=color,
            mfc="white",
            mew=1.7,
            ms=7,
            label=label,
        )
        if q[0] < floor:
            ax.annotate(
                "",
                xy=(x, ymin * 1.02),
                xytext=(x, floor * 2),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4},
            )
        records.append(
            {
                "z": row["z"],
                "population": key,
                "quantiles": q.tolist(),
                "median_below_axis": bool(q[1] < floor),
            }
        )

    for i, row in enumerate(summary["results"]):
        if fixed_uv:
            age = summary["max_age_myr"]
            label = rf"Model: $M_{{\rm UV}}\leq {summary['absolute_uv_limit']:g}$"
            if age is not None:
                label += rf", age $\leq {age:g}$ Myr"
            population(row, "uv_selected_known", 0, "#286491", label if i == 0 else None)
        else:
            population(
                row, "uv_selected_known", -0.10, "#939aa2", "UV selected" if i == 0 else None
            )
            population(
                row,
                "uv_and_line_selected",
                0.10,
                "#286491",
                "UV + He II selected" if i == 0 else None,
            )
    redshifts = np.array([r["z"] for r in summary["results"]])
    # Calculate the threshold continuously; these are not interpolated model points.
    grid = np.linspace(redshifts.min(), redshifts.max(), 150)
    limit = (
        4 * np.pi * astro.luminosity_distance(grid).to_value("cm") ** 2 * summary["heii_flux_limit"]
    )
    if not fixed_uv:
        ax.plot(grid, limit, "--", lw=1.5, color="#b3812b", label=r"Reference $5\sigma$ line limit")
    labelled = set()
    for row in sources:
        alternate = not row["primary"]
        color = "#876f98" if alternate else "#c34435"
        label = "GN-z11: other apertures" if alternate else "Observation / upper limit"
        upper = row["measurement"] == "upper_limit"
        ax.errorbar(
            row["z"],
            row["luminosity"],
            yerr=row["luminosity"] * 0.4 if upper else row["luminosity_error"],
            uplims=upper,
            fmt="*" if alternate else "s",
            ms=9 if alternate else 6,
            color=color,
            mfc="white" if alternate else color,
            capsize=3,
            label=label if label not in labelled else None,
            zorder=5,
        )
        labelled.add(label)
        if upper:
            ax.annotate(
                rf"${row['limit_sigma']}\sigma$",
                (row["z"], row["luminosity"]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=12,
            )
    names = [
        "LAP1\nVanzella+23",
        "RXJ2129-A\nWang+24",
        "GN-z11\nMaiolino+24",
        "GHZ2\nCastellano+24",
        "GS-z14-1\nWu+25",
    ]
    for z, label in zip(redshifts, names, strict=True):
        ax.text(
            z,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"Intrinsic $L_{\mathrm{He\,II}\,1640}$ [erg s$^{-1}$]",
        yscale="log",
        xlim=(6.0, 14.6),
        ylim=(ymin, ymax),
    )
    ax.set_xticks(redshifts, [f"{z:g}" for z in redshifts])
    ax.grid(axis="y", which="major", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if fixed_uv:
        for i, label in enumerate(labels):
            if label.startswith("Model:"):
                handles[i] = Line2D([], [], marker="o", color="#286491", mfc="white", ls="none")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        ncol=2 if fixed_uv else 3,
        fontsize=11,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.74)
    name = "heii_uv_bright" if fixed_uv else "heii_survey_depth"
    output = ROOT / "outputs" / f"{name}_20260918"
    output.mkdir(exist_ok=True, parents=True)
    asset = DECK / "assets" / f"{name}.pdf"
    fig.savefig(asset)
    fig.savefig(output / f"{name}.png", dpi=150)
    plt.close(fig)
    provenance = {
        "summary": str(args.summary.relative_to(ROOT)),
        "summary_sha256": digest(args.summary),
        "plot_script_sha256": digest(Path(__file__)),
        "model": records,
        "observations": sources,
        "pdf_sha256": digest(asset),
        "observation_inputs": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [high_path, ROOT / "external_data/observations/heii/venditti2024_targets.json"]
        },
        "display": (
            "Blue distribution uses the explicit absolute UV and optional age selection; "
            "no line-flux cut, no redshift offsets. "
            if fixed_uv
            else "Blue distribution conditional on BOTH UV and line cuts; grey on UV only. "
            "Model x offsets +/-0.10 for visibility only; true redshifts unchanged. "
        )
        + "Down arrows mark off-scale intervals; downward triangles mark off-scale medians. "
        "Observed systems have different actual observing/lensing conditions and are references.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
