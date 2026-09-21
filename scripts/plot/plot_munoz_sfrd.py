"""Compare the current SFRD with published vector curves in Munoz et al. Fig. 6."""

import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "external_data/literature_sources/munoz2022_eos/source/figures/SFRD_a0.pdf"
RUN = ROOT / "data_save/ionizing_sources/sfrd_z5_v1"
OUT = ROOT / "outputs/21cm_map/sfrd"
ASSET = ROOT / "slides/assets/sfrd"
COLORS = {"popii": "#852c64", "popiii": "black", "total": "#aaaaaa"}


def reference_curves():
    svg = OUT / "munoz_fig6.svg"
    subprocess.run(["pdftocairo", "-svg", str(PAPER), str(svg)], check=True)
    chunks = {k: [] for k in COLORS}
    for e in ET.parse(svg).getroot().iter():
        style, path = e.get("style", ""), e.get("d", "")
        if "fill:none" not in style or len(path) < 1000:
            continue
        if "stroke-width:7.472;" in style:
            key = "total"
        elif "stroke-width:3.736;" in style:
            key = "popiii" if "stroke:rgb(0%,0%,0%);" in style else "popii"
        else:
            continue
        assert e.get("transform") == "matrix(2.033333,0,0,2.033333,0,-0.4167)"
        assert set(re.findall("[A-Za-z]", path)) <= {"M", "L"}
        xy = np.array(re.findall(r"[ML]\s+([-\d.]+)\s+([-\d.]+)", path), dtype=float)
        chunks[key].append(xy)
    result = {}
    for key, parts in chunks.items():
        assert parts, key
        xy = np.concatenate(parts)
        # Visible ticks in native PDF coordinates: z=5,30 and log10(SFRD)=-5,-2.
        z = 5 + (xy[:, 0] - 116.999251) * 25 / (583.999199 - 116.999251)
        logy = -5 + (297.811284 - xy[:, 1]) * 3 / (297.811284 - 63.726333)
        # Do not extract or extrapolate invisible paths below the published axis.
        keep = (z >= 5 - 1e-6) & (z <= 30) & (logy >= -5) & (logy <= np.log10(0.05))
        order = np.argsort(z[keep])
        result[key] = (z[keep][order], 10 ** logy[keep][order])
    with (RUN / "munoz2022_fig6_curves.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["population", "z", "sfrd_msun_yr_cmpc3"])
        for key, (z, y) in result.items():
            writer.writerows(zip([key] * len(z), z, y, strict=True))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSET.mkdir(parents=True, exist_ok=True)
    ref = reference_curves()
    inputs = [ROOT / "data_save/ionizing_sources/sfrd_v1/sfrd.csv", RUN / "sfrd.csv"]
    allrows = np.concatenate([np.genfromtxt(p, delimiter=",", names=True) for p in inputs])
    ours = allrows[allrows["window_myr"] == 10]
    ours.sort(order="z")
    assert np.all(np.diff(ours["z"]) > 0)
    plt.style.use("apj")
    for filename in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / filename)
        )
    plt.rcParams.update({"font.family": "Arial", "text.usetex": False, "font.size": 12})
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.7), sharex=True, sharey=True, layout="constrained"
    )
    labels = {"popii": "Pop II", "popiii": "Pop III", "total": "Pop II + III"}
    for key in ["total", "popii", "popiii"]:
        z, y = ref[key]
        axes[0].semilogy(z, y, color=COLORS[key], lw=3 if key == "total" else 2, label=labels[key])
        axes[1].semilogy(
            ours["z"],
            ours[key],
            color=COLORS[key],
            lw=3 if key == "total" else 2,
            label=labels[key],
        )
    for ax in axes:
        ax.set(xlim=(5, 30), ylim=(1e-5, 5e-2), xlabel="Redshift z", xticks=[5, 10, 15, 20, 25, 30])
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", which="major", alpha=0.2)
    axes[0].set_title("Munoz et al. (2022), Fig. 6")
    axes[1].set_title("AuroraLF: current first-burst model")
    axes[0].set_ylabel(r"SFRD [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{cMpc}^{-3}$]")
    fig.savefig(OUT / "munoz_comparison.png", dpi=160)
    fig.savefig(OUT / "munoz_comparison.pdf")
    fig.set_size_inches(11.5, 3.5)
    fig.savefig(ASSET / "munoz_comparison_slide.pdf")
    plt.close(fig)
    rows = []
    for z0 in [5, 6, 8, 10, 15, 20]:
        row = {"z": z0}
        for key in ["popii", "popiii"]:
            z, y = ref[key]
            assert z.min() - 1e-6 <= z0 <= z.max()
            literature = 10 ** np.interp(z0, z, np.log10(y))
            current = 10 ** np.interp(z0, ours["z"], np.log10(ours[key]))
            row[key] = {"paper": literature, "ours": current, "ratio": current / literature}
        rows.append(row)
    z3, y3 = ref["popiii"]
    peak = int(np.argmax(y3))
    our_peak = int(np.argmax(ours["popiii"]))
    summary = {
        "reference": "Munoz et al. (2022), MNRAS 511, 3657, Fig. 6; arXiv:2110.13919v2",
        "doi": "10.1093/mnras/stac185",
        "method": "Published model vector vertices, not simulation tables; log interpolation only within visible curves. Observational points and alternative feedback curves omitted.",
        "units": "Msun/yr/cMpc^3",
        "paper_popiii_peak": {
            "z": float(z3[peak]),
            "sfrd": float(y3[peak]),
            "z5_over_peak": float(y3[0] / y3[peak]),
        },
        "our_popiii_sampled_peak": {
            "z": float(ours["z"][our_peak]),
            "sfrd": float(ours["popiii"][our_peak]),
            "z5_over_peak": float(ours["popiii"][0] / ours["popiii"][our_peak]),
        },
        "comparisons": rows,
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [PAPER, *inputs]
        },
        "notes": "Paper average-density region with ACG/MCG separation and feedback vs global Reed07 main-branch first-burst expectation averaged over 10 Myr. Parameters have not been refitted.",
    }
    (RUN / "munoz_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
