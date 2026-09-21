"""Separate He II population band from verified, completed model summaries.

The band spans the 16--84% population distribution at fixed epsilon=0.03.
By default high-z target windows remain separate. The explicit uniform-sample
extension uses the same M1500/age cuts at every redshift.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.plot.plot_heii_combined_redshifts import DECK, ROOT, digest, load_completed
from scripts.plot.plot_heii_v24_overlay import DATA, load_v24

OUTPUT = ROOT / "outputs/heii_model_band_20260917"
EPSILON = "0.03"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uniform-sample-dir",
        help="Completed data_save directory of the uniform M1500/young-burst redshift extension",
    )
    parser.add_argument("--low-sample-dir", default="heii_v24_targets_20260914")
    parser.add_argument("--high-sample-dir", default="heii_exact_targets_20260916")
    args = parser.parse_args()
    low, low_path = load_completed(args.low_sample_dir)
    high, high_path = load_completed(args.high_sample_dir)
    catalog = ROOT / "external_data/observations/heii/venditti2024_targets.json"
    if digest(catalog) != low["observations"]["catalog_sha256"]:
        raise ValueError("Observation catalog changed")
    if not high["exact_redshifts"]:
        raise ValueError("Exact target redshifts required")
    reference = load_v24()
    records = []
    for result in low["model"]:
        records.append(
            dict(
                z=result["z"],
                selection="MUV<=-20, burst age<=3 Myr",
                quantiles=result["efficiencies"][EPSILON]["linear_log_age"]["young_uv_bright"][
                    "luminosity_q16_q50_q84"
                ],
            )
        )
    sources = [dict(row) for row in low["observations"]["sources"]]
    for result in high["efficiencies"]["0.03"]["results"]:
        t = result["target"]
        if t["z"] != t["population_z"]:
            raise ValueError("Mismatched target redshift")
        quantiles = (
            np.asarray(result["target_windows"][0]["methods"]["linear_log_age"]["flux_q16_q50_q84"])
            / result["flux_per_luminosity"]
        ).tolist()
        records.append(
            dict(z=t["z"], selection=f"MUV={t['muv']} +/-0.25; no age cut", quantiles=quantiles)
        )
        sources.append(
            dict(
                z=t["z"],
                primary=True,
                measurement=t["measurement"],
                luminosity=t["flux"] / result["flux_per_luminosity"],
                luminosity_error=None
                if t["measurement"] == "upper_limit"
                else t["flux_error"] / result["flux_per_luminosity"],
                limit_sigma=3,
            )
        )
    for r in records:
        values = np.asarray(r["quantiles"])
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) < 0):
            raise ValueError("Invalid population quantiles")

    input_paths = [low_path, high_path, catalog, DATA]
    if args.uniform_sample_dir:
        extension, extension_path = load_completed(args.uniform_sample_dir)
        manifest_path = extension_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        low_manifest = json.loads((low_path.parent / "manifest.json").read_text())
        reference_config = low_manifest["plan"]["configs"][0]
        keys = (
            "popii_wavelength_a",
            "popii_ssp",
            "popiii_ssp",
            "lookback_myr",
            "q_log10_mean",
            "q_log10_sigma",
            "z_start",
            "logmass_min",
            "logmass_max",
            "n_grid",
            "sfr_convolution",
        )
        for cfg in manifest["plan"]["configs"]:
            if any(cfg[k] != reference_config[k] for k in keys):
                raise ValueError(
                    "Extension changes a physical parameter or UV selection wavelength"
                )
        if reference_config["popii_wavelength_a"] != 1500:
            raise ValueError("Uniform band requires M1500")
        records = []
        for result in low["model"] + extension["model"]:
            stats = result["efficiencies"][EPSILON]["linear_log_age"]["young_uv_bright"]
            q = np.asarray(stats["luminosity_q16_q50_q84"])
            if not np.isfinite(q).all() or np.any(q <= 0) or np.any(np.diff(q) < 0):
                raise ValueError("Invalid uniform population quantiles")
            records.append(
                dict(
                    z=result["z"],
                    selection="M1500<=-20, burst age<=3 Myr",
                    quantiles=q.tolist(),
                    effective_mass_clusters=stats["effective_mass_clusters"],
                    n_selected=stats["n_selected"],
                )
            )
        records.sort(key=lambda r: r["z"])
        if len(set(r["z"] for r in records)) != len(records):
            raise ValueError("Duplicate redshifts in extension")
        input_paths += [extension_path, manifest_path]
    connected = records if args.uniform_sample_dir else records[:3]
    isolated = [] if args.uniform_sample_dir else records[3:]

    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 4.3))
    # Log-linear guides connect only redshifts sharing the low-z selection.
    rz = np.array([r["z"] for r in connected])
    bounds = np.array([r["quantiles"] for r in connected])
    grid = np.linspace(rz.min(), rz.max(), 300)
    lo, median, hi = [10 ** np.interp(grid, rz, np.log10(bounds[:, k])) for k in (0, 1, 2)]
    ax.fill_between(grid, lo, hi, color="#378877", alpha=0.25)
    ax.plot(grid, lo, color="#378877", lw=1)
    ax.plot(grid, hi, color="#378877", lw=1)
    ax.plot(grid, median, color="#378877", lw=2)
    for r in isolated:
        # Width is for visibility only, not redshift uncertainty or a z bin.
        ax.fill_between(
            [r["z"] - 0.16, r["z"] + 0.16],
            r["quantiles"][0],
            r["quantiles"][2],
            color="#286491",
            alpha=0.25,
            edgecolor="#286491",
        )
        ax.plot([r["z"] - 0.16, r["z"] + 0.16], [r["quantiles"][1]] * 2, color="#286491", lw=2)
    for r in records:
        ax.plot(
            r["z"],
            r["quantiles"][1],
            marker="o",
            ls="none",
            color="#378877" if r in connected else "#286491",
            mfc="white",
            ms=7,
            mew=1.4,
            zorder=4,
        )

    for c in reference["curves"]:
        if (c["resolving_power"], c["exposure_h"], c["integrated_snr"]) == (1000, 50, 5):
            ax.plot(
                c["z"],
                c["luminosity_erg_s"],
                color="#79572c" if c["mode"] == "IFU" else "#62508e",
                ls="--" if c["linewidth_kms"] == 500 else ":",
                lw=1.6,
            )
    for r in sources:
        upper = r["measurement"] == "upper_limit"
        alternate = not r["primary"]
        color = "#876f98" if alternate else "#c34435"
        ax.errorbar(
            r["z"],
            r["luminosity"],
            yerr=0.4 * r["luminosity"] if upper else r["luminosity_error"],
            uplims=upper,
            fmt="*" if alternate else "s",
            color=color,
            ms=10 if alternate else 6,
            capsize=3,
            zorder=5,
        )
        if upper:
            ax.annotate(
                rf"${r['limit_sigma']}\sigma$",
                (r["z"], r["luminosity"]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
            )
    names = ["LAP1", "RXJ2129-A", "GN-z11", "GHZ2", "GS-z14-1"]
    observation_z = [6.639, 8.1623, 10.6, 12.342, 13.86]
    for z, name in zip(observation_z, names):
        ax.text(z, 1.03, name, transform=ax.get_xaxis_transform(), ha="center")
    handles = [
        Patch(facecolor=c, alpha=0.25, label=label)
        for c, label in [
            ("#378877", "16–84%: young, UV-bright"),
            ("#286491", "16–84%: target UV windows"),
        ]
    ]
    if args.uniform_sample_dir:
        handles = handles[:1]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color="#555555",
            mfc="white",
            ls="-",
            label=rf"Model median: $\epsilon_b={EPSILON}$",
        )
    ]
    handles += [Line2D([], [], marker="s", color="#c34435", ls="none", label="Observation / limit")]
    handles += [
        Line2D([], [], marker="*", color="#876f98", ls="none", label="GN-z11: other apertures")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=11)
    ref_handles = [
        Line2D([], [], color=c, label=f"V24 {m}")
        for m, c in [("IFU", "#79572c"), ("MOS", "#62508e")]
    ]
    ref_handles += [
        Line2D([], [], color="#555555", ls=s, label=f"{v} km/s")
        for s, v in [("--", 500), (":", 50)]
    ]
    ax.legend(
        handles=ref_handles,
        loc="lower right",
        ncol=2,
        fontsize=10,
        title=r"Thresholds: $R\simeq1000$, 50 h, S/N$\simeq5$",
        title_fontsize=10,
    )
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"Intrinsic $L_{\mathrm{He\,II}\,1640}$ [erg s$^{-1}$]",
        yscale="log",
        xlim=(6, 14.9 if args.uniform_sample_dir else 14.6),
        ylim=(3e38, max(1e43, 1.2 * max(r["quantiles"][2] for r in records))),
    )
    if args.uniform_sample_dir:
        ax.set_xticks(range(6, 16))
    else:
        ax.set_xticks([r["z"] for r in records], [f"{r['z']:.3f}" for r in records])
    ax.grid(axis="y", which="major", alpha=0.15)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.75)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    name = "heii_model_uniform_band" if args.uniform_sample_dir else "heii_model_population_band"
    fig.savefig(DECK / "assets" / f"{name}.pdf")
    fig.savefig(OUTPUT / f"{name}.png", dpi=150)
    plt.close(fig)
    provenance = dict(
        epsilon=EPSILON,
        model=records,
        observations=sources,
        sources={str(p.relative_to(ROOT)): digest(p) for p in input_paths},
        script_sha256=digest(Path(__file__)),
        band="HMF-weighted 16--84% population quantiles at epsilon=0.03; not MC or parameter uncertainty",
        interpolation="Log-linear guides between computed nodes; not additional calculated redshifts",
        high_z="Same M1500/age selection as low-z"
        if args.uniform_sample_dir
        else "Different target UV windows; separate display strips, half width0.16 is not redshift error",
        thresholds="V24 published IFU/MOS R1000 50h SNR5, linewidth500/50km/s; no V24 model bands",
        caveats="Small effective old low-z sample (5--11 mass clusters); aperture/whole-halo mismatch. "
        "No pristine gate. Thresholds unlensed and not target-specific. "
        + (
            "Uniform extension changes high-z selection; see source manifest for new runs."
            if args.uniform_sample_dir
            else "Identical quantiles to retained epsilon=0.03 errorbar figure; no new science runs."
        ),
    )
    (
        OUTPUT / ("provenance-uniform.json" if args.uniform_sample_dir else "provenance.json")
    ).write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
