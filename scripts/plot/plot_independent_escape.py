"""Plot a fixed-Pop-III-escape trial and generate a quantitative slide table."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--asset-prefix",
        default="independent_escape",
        help="Filename prefix for slide assets and the generated values table",
    )
    args = parser.parse_args()
    if not args.asset_prefix or Path(args.asset_prefix).name != args.asset_prefix:
        raise ValueError("Asset prefix must be a filename component")
    plan = json.loads(args.plan.read_text())
    data, out = Path(plan["data_output"]), Path(plan["plot_output"])
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Analysis must be complete")
    arrays = dict(np.load(data / "histories_tau.npz", allow_pickle=False))
    reference = json.loads((ROOT / "data_save/reionization_z5_20260919/summary.json").read_text())
    ref_arrays = np.load(
        ROOT / "data_save/reionization_z5_20260919/histories_tau.npz", allow_pickle=False
    )
    ref = next(r for r in reference["results"] if r["key"] == "popii_popiii_f0.064")
    arrays.update({k: ref_arrays[k] for k in ref_arrays.files if k.startswith(ref["key"] + "_")})
    rows = [dict(ref, fesc_popii=0.064, fesc_popiii=0.064), *summary["results"]]
    styles = [("#2863A5", ":"), ("#C65C28", "-"), ("#74479B", "--")]
    if len(rows) != len(styles):
        raise ValueError("This figure expects one reference and two boundary trials")
    labels = [
        rf"$f_{{esc,II}}={r['fesc_popii']:g},\ f_{{esc,III}}={r['fesc_popiii']:g}$" for r in rows
    ]
    plt.style.use("apj")
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "legend.fontsize": 12})
    deck = ROOT / "slides/popiii_heii_pisn_complete_20260916"

    def save(fig, name):
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight")
        shutil.copy2(
            out / (name + ".pdf"), deck / "assets" / (args.asset_prefix + "_" + name + ".pdf")
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.7), layout="constrained")
    for row, (color, ls), label in zip(rows, styles, labels, strict=True):
        key = row["key"]
        ax.plot(
            arrays[key + "_z"],
            1 - arrays[key + "_q_volume"],
            color=color,
            ls=ls,
            lw=2.4,
            label=label,
        )
    with (ROOT / "external_data/observations/reionization/neutral_fraction.csv").open() as stream:
        obs = list(csv.DictReader(stream))
    for row in obs:
        if row["kind"] != "interval" or row["z_min"] != row["z_max"]:
            continue
        z, x, lo, hi = [float(row[k]) for k in ["z", "xhi", "lower", "upper"]]
        ax.errorbar(
            z,
            x,
            yerr=[[x - lo], [hi - x]],
            fmt="o",
            color="black",
            mfc="white",
            capsize=4,
            label={"mason18_7": r"Mason+18 (68\%)", "davies18_709": r"Davies+18 (68\%)"}.get(
                row["id"]
            ),
        )
    ax.set(
        xlim=(5, 25),
        ylim=(-0.015, 1.025),
        xlabel="Redshift z",
        ylabel=r"Volume-averaged neutral fraction $\langle x_{HI}\rangle_V$",
    )
    ax.legend(loc="lower right")
    save(fig, "history")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    tref = summary["tau_reference"]
    ax.axhspan(
        tref["value"] - tref["sigma"],
        tref["value"] + tref["sigma"],
        color=".85",
        label=r"Planck 2018 total (68\%)",
    )
    ax.axhline(tref["value"], color=".4", ls=":", lw=1)
    for row, (color, ls), label in zip(rows, styles, labels, strict=True):
        key = row["key"]
        ax.plot(
            arrays[key + "_tau_z"],
            arrays[key + "_tau"],
            color=color,
            ls=ls,
            lw=2,
            label=label + rf"; $\tau={row['tau']:.4f}$",
        )
        bx.plot(
            arrays[key + "_tau_z"], arrays[key + "_dtau_dz"], color=color, ls=ls, lw=2, label=label
        )
    ax.set(
        xlim=(0, 40),
        ylim=(0, 1.07 * max(r["tau"] for r in rows)),
        xlabel="Upper integration redshift z",
        ylabel=r"Cumulative Thomson depth $\tau(0,z)$",
    )
    bx.set(xlim=(5, 25), ylim=(0, None), xlabel="Redshift z", ylabel=r"Contribution $d\tau/dz$")
    ax.legend(loc="lower right", fontsize=9)
    bx.legend(fontsize=10)
    save(fig, "tau")
    lines = [
        r"\begin{center}\setlength{\tabcolsep}{4mm}\begin{tabular}{rrrrrr}\toprule",
        r"$f_{\rm esc,II}$ & $f_{\rm esc,III}$ & $x_{\rm HI}(7)$ & $z_{50\%}$ & $\tau$ & $(\tau-0.054)/0.007$ \\",
        r"\midrule",
    ]
    for row in rows:
        residual = (row["tau"] - 0.054) / 0.007
        lines.append(
            f"{row['fesc_popii']:g} & {row['fesc_popiii']:g} & {row['xhi_z7']:.4f} & {row['z50']:.3f} & {row['tau']:.5f} & {residual:.2f}"
            + r" \\"
        )
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    (deck / (args.asset_prefix + "_values.tex")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
