"""Compare He II observations with exploratory target-centered UV windows.

Reads completed products only. Run with PYTHONPATH=. .venv/bin/python.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "slides/popiii_heii_pisn_complete_20260916"
OUTPUT = ROOT / "outputs/heii_combined_20260917"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_completed(directory):
    directory = ROOT / "data_save" / directory
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "summary.json"
    if manifest["status"] != "complete" or digest(path) != manifest["products"]["summary.json"]:
        raise ValueError(f"Unverified result: {directory}")
    return json.loads(path.read_text()), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v24", action="store_true", help="Overlay published V24 reference conditions"
    )
    parser.add_argument("--low-sample-dir", default="heii_v24_targets_20260914")
    parser.add_argument("--high-sample-dir", default="heii_exact_targets_20260916")
    args = parser.parse_args()
    low, low_path = load_completed(args.low_sample_dir)
    high, high_path = load_completed(args.high_sample_dir)
    catalog = ROOT / "external_data/observations/heii/venditti2024_targets.json"
    if digest(catalog) != low["observations"]["catalog_sha256"]:
        raise ValueError("Low-redshift observation catalog changed")
    if not high["exact_redshifts"]:
        raise ValueError("Exact observed-redshift populations required")
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
    fig, ax = plt.subplots(figsize=(12, 4.3))
    if args.v24:
        from scripts.plot.plot_heii_v24_overlay import add_v24_reference

        add_v24_reference(ax)
    records = []

    def model(z, quantiles, marker, color, label, selection):
        q = np.asarray(quantiles)
        if not np.isfinite(q).all() or np.any(q <= 0) or np.any(np.diff(q) < 0):
            raise ValueError("Invalid luminosity quantiles")
        if not 3e38 < q[1] <= q[2] < 2e43:
            raise ValueError("Widen plotting limits to include model quantiles")
        lower_outside = q[0] < 3e38
        displayed_lower = 4e38 if lower_outside else q[0]
        ax.errorbar(
            z,
            q[1],
            yerr=[[q[1] - displayed_lower], [q[2] - q[1]]],
            fmt=marker,
            color=color,
            mfc="white",
            ms=8,
            mew=1.8,
            capsize=6,
            elinewidth=2,
            label=label,
            zorder=3,
        )
        if lower_outside:
            ax.annotate(
                "16th pct below axis",
                (z, 3.1e38),
                xytext=(z, 1.6e39),
                arrowprops={"arrowstyle": "->", "color": color},
                color=color,
                fontsize=10,
                ha="center",
            )
        records.append(
            dict(
                z=z,
                luminosity_q16_q50_q84=q.tolist(),
                selection=selection,
                q16_below_plot=bool(lower_outside),
            )
        )

    rx_rows = [
        result
        for result in low["model"]
        if "rx_uv_matched" in result["efficiencies"]["0.03"]["linear_log_age"]
    ]
    if len(rx_rows) != 1:
        raise ValueError("Exactly one RXJ2129 target-window result is required")
    for result in rx_rows:
        stats = result["efficiencies"]["0.03"]["linear_log_age"]["rx_uv_matched"]
        center = result["rx_intrinsic_muv_center"]
        width = stats["muv_window_half_width"]
        model(
            result["z"],
            stats["luminosity_q16_q50_q84"],
            "o",
            "#286491",
            "Model: exploratory UV window",
            f"MUV={center} +/-{width} mag; no burst-age cut; analyst-chosen window",
        )

    sources = [dict(row) for row in low["observations"]["sources"]]
    for i, result in enumerate(high["efficiencies"]["0.03"]["results"]):
        target = result["target"]
        if target["z"] != target["population_z"]:
            raise ValueError("High-redshift model differs from observed redshift")
        conversion = result["flux_per_luminosity"]
        stats = result["target_windows"][0]["methods"]["linear_log_age"]
        window = result["target_windows"][0]
        q = np.asarray(stats["flux_q16_q50_q84"]) / conversion
        model(
            target["z"],
            q,
            "o",
            "#286491",
            None,
            f"{window['muv_low']} <= MUV <= {window['muv_high']}; "
            "no burst-age cut; analyst-chosen window",
        )
        sources.append(
            dict(
                id=target["name"],
                z=target["z"],
                primary=True,
                measurement=target["measurement"],
                luminosity=target["flux"] / conversion,
                luminosity_error=None
                if target["measurement"] == "upper_limit"
                else target["flux_error"] / conversion,
                limit_sigma=3,
            )
        )

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
            ms=10 if alternate else 6,
            color=color,
            mfc="white" if alternate else color,
            capsize=3,
            elinewidth=1.5,
            label=label if label not in labelled else None,
            zorder=5,
        )
        labelled.add(label)
        if upper:
            ax.annotate(
                rf"${row['limit_sigma']}\sigma$",
                (row["z"], row["luminosity"]),
                xytext=(9, 0),
                textcoords="offset points",
                color=color,
                fontsize=12,
            )

    names = [
        (6.639, "LAP1\nVanzella+23"),
        (8.1623, "RXJ2129-A\nWang+24"),
        (10.6, "GN-z11\nMaiolino+24"),
        (12.342, "GHZ2\nCastellano+24"),
        (13.86, "GS-z14-1\nWu+25"),
    ]
    for z, label in names:
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
        ylim=(3e38, 2e43),
    )
    ax.set_xticks([z for z, _ in names], ["6.639", "8.1623", "10.600", "12.342", "13.860"])
    ax.grid(axis="y", which="major", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        ncol=2,
        fontsize=12,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.74)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = "heii_combined_v24" if args.v24 else "heii_combined_eps003"
    fig.savefig(DECK / "assets" / f"{name}.pdf")
    fig.savefig(OUTPUT / f"{name}.png", dpi=150)
    plt.close(fig)
    provenance = dict(
        epsilon=0.03,
        sources={str(p.relative_to(ROOT)): digest(p) for p in [low_path, high_path, catalog]},
        model=records,
        observations=sources,
        script_sha256=digest(Path(__file__)),
        interval="16–84% population distribution, not MC error",
        caveat="UV windows are analyst-chosen exploratory bins, not measurement errors "
        "or literature selection functions; no global MUV=-20 or age=3Myr cut. "
        "LAP1 and GN-z11 have observations only, without a matched model population. "
        "The RXJ2129 16th percentile is below the displayed axis and is marked by an arrow. "
        "GN-z11 aperture records are correlated, not independent galaxies. "
        "Low-z effective mass sampling remains limited; no new science calculation.",
    )
    if args.v24:
        from scripts.plot.plot_heii_v24_overlay import DATA

        provenance["v24_reference"] = dict(
            path=str(DATA.relative_to(ROOT)),
            sha256=digest(DATA),
            displayed_thresholds="R=1000,50h,S/N=5; IFU/MOS,500/50km/s; unlensed published reference",
            bands="V24 model ranges, eta_III=0.01--0.3; not AuroraLF errors or efficiency mapping",
        )
    (OUTPUT / ("provenance-v24.json" if args.v24 else "provenance.json")).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
