"""Plot actual z=5.01 reionization histories and updated Thomson depths."""

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
    parser.add_argument("--slide-assets", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    data, out = Path(plan["data_output"]), Path(plan["plot_output"])
    summary = json.loads((data / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Analysis incomplete")
    arrays = np.load(data / "histories_tau.npz")
    rows = summary["results"]
    styles = [("#2863A5", "-"), ("#C65C28", "-"), ("#C65C28", "--")]
    if [(r["population"], r["fesc"]) for r in rows] != [
        ("popii", 0.1775),
        ("popii_popiii", 0.064),
        ("popii_popiii", 0.0725),
    ]:
        raise ValueError("Unexpected displayed parameter selections")
    names = {"popii": "Pop II", "popii_popiii": "Pop II+III"}
    labels = [rf"{names[r['population']]}, $f_{{esc}}={r['fesc']:g}$" for r in rows]
    plt.style.use("apj")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "legend.fontsize": 10.5})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ["pdf", "png"]:
            fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_neutral(ax):
        for r, (color, style), label in zip(rows, styles, labels, strict=True):
            key = r["key"]
            ax.plot(
                arrays[key + "_z"],
                1 - arrays[key + "_q_volume"],
                style,
                color=color,
                lw=2.3,
                label=label,
            )
        ax.set(
            xlabel="Redshift z",
            ylabel=r"Volume-averaged neutral fraction $\langle x_{HI}\rangle_V$",
        )

    with (ROOT / "external_data/observations/reionization/neutral_fraction.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    obs_labels = {
        "mason18_7": "Mason+18",
        "davies18_709": "Davies+18",
        "mason26_65": "Mason+26 (wide z bins)",
        "mcgreer15_56": "McGreer+15 limits",
    }
    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout="constrained")
    plot_neutral(ax)
    for r in observations:
        z, x, lo, hi, zlo, zhi = [
            float(r[k]) for k in ["z", "xhi", "lower", "upper", "z_min", "z_max"]
        ]
        if r["kind"] == "upper_limit":
            ax.errorbar(
                z,
                x,
                yerr=0.05,
                uplims=True,
                fmt="none",
                color="0.4",
                capsize=3,
                label=obs_labels.get(r["id"]),
            )
        else:
            ax.errorbar(
                z,
                x,
                yerr=[[x - lo], [hi - x]],
                xerr=[[z - zlo], [zhi - z]],
                fmt="s" if zlo != zhi else "o",
                color="#875294" if zlo != zhi else "black",
                mfc="white",
                ms=6,
                capsize=3,
                lw=1,
                label=obs_labels.get(r["id"]),
            )
    ax.set(xlim=(5, 13.5), ylim=(-0.015, 1.025))
    ax.legend(loc="lower right")
    save(fig, "reionization_history")

    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout="constrained")
    plot_neutral(ax)
    for r in observations:
        if r["kind"] == "upper_limit":
            ax.errorbar(
                float(r["z"]),
                float(r["upper"]),
                yerr=0.022,
                uplims=True,
                fmt="none",
                color="black",
                capsize=5,
                label=obs_labels.get(r["id"]),
            )
    ax.axhline(0.01, color="0.5", lw=1, ls=":")
    ax.text(5.07, 0.016, r"99\% ionized (descriptive crossing)", fontsize=10, color="0.4")
    for r, (color, _), label in zip(rows, styles, labels, strict=True):
        if r["z99"] is not None:
            ax.plot(r["z99"], 0.01, "o", color=color, mfc="white", ms=6)
    ax.set(xlim=(5, 6.5), ylim=(-0.006, 0.35))
    ax.legend(loc="upper left")
    save(fig, "reionization_lowz")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    ref = summary["tau_reference"]
    ax.axhspan(ref["value"] - ref["sigma"], ref["value"] + ref["sigma"], color="0.85", alpha=0.7)
    ax.axhline(ref["value"], color="0.45", ls=":", lw=1)
    for r, (color, style), label in zip(rows, styles, labels, strict=True):
        key = r["key"]
        ax.plot(
            arrays[key + "_tau_z"],
            arrays[key + "_tau"],
            style,
            color=color,
            lw=2,
            label=label + rf"; $\tau={r['tau']:.5f}$",
        )
        bx.plot(
            arrays[key + "_tau_z"], arrays[key + "_dtau_dz"], style, color=color, lw=2, label=label
        )
    ax.axvspan(0, 5.01, color="#9CBACF", alpha=0.14)
    ax.text(0.7, 0.006, "Fully ionized\ncontinuation", fontsize=9, color="#39546A")
    ax.text(8.5, 0.048, "Planck 2018 total optical depth", fontsize=9, color="0.3")
    ax.set(
        xlim=(0, 30),
        ylim=(0, 0.07),
        xlabel="Upper integration redshift z",
        ylabel=r"Cumulative Thomson depth $\tau(0,z)$",
    )
    ax.legend(loc="lower right", fontsize=9.5)
    bx.set(
        xlim=(5, 20),
        ylim=(0, None),
        xlabel="Redshift z",
        ylabel=r"Thomson depth contribution $d\tau/dz$",
    )
    bx.legend(loc="upper right")
    save(fig, "optical_depth")

    if args.slide_assets:
        deck = ROOT / "slides/popiii_heii_pisn_complete_20260916"
        for name, asset in [
            ("reionization_history", "calibrated_reionization_history"),
            ("optical_depth", "calibrated_optical_depth"),
            ("reionization_lowz", "calibrated_lowz"),
        ]:
            shutil.copy2(out / f"{name}.pdf", deck / "assets" / f"{asset}.pdf")
        lines = [
            r"\begin{center}\setlength{\tabcolsep}{4mm}\begin{tabular}{lrrrrr}\toprule",
            r"模型 & $f_{\rm esc}$ & $z_{99\%}$ & $x_{\rm HI}(5.9)$ & $\tau$ & $\Delta\tau$ \\",
            r"\midrule",
        ]
        for r in rows:
            lines.append(
                f"{names[r['population']]} & {r['fesc']:g} & {r['z99']:.3f} & {r['xhi_z59']:.4f} & {r['tau']:.5f} & {r['tau_change']:+.5f}"
                + r" \\"
            )
        lines += [
            r"\bottomrule\end{tabular}\end{center}",
            r"$\Delta\tau$ 相对原先在 $z<6.01$ 假定线性补全至 $z=5.6$ 的结果。",
            r"\par 三组均演化至真实末帧 $z=5.01$，此时体积与密度加权电离比例均为 1。",
        ]
        (deck / "reionization_calibration_values.tex").write_text("\n".join(lines) + "\n")
        ii, mixed = rows[:2]
        (deck / "reionization_tau_tail.tex").write_text(
            f"折中选择下，$z>10$ 贡献的光深：纯 Pop II 为 {ii['tau_above_z']['10']:.4f}，混合模型为 {mixed['tau_above_z']['10']:.4f}。混合模型的高红移电离尾部更长。\n"
        )


if __name__ == "__main__":
    main()
