"""Plot separately calibrated reionization histories and mass-weighted optical depth."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"popii": "#2863A5", "popii_popiii": "#C65C28"}
LABELS = {"popii": "Pop II", "popii_popiii": "Pop II+III"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--slide-assets",
        action="store_true",
        help="Update the calibration slide assets and numeric table",
    )
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    data, out = Path(plan["data_output"]), Path(plan["plot_output"])
    summary = json.loads((data / "summary.json").read_text())
    arrays = np.load(data / "histories_tau.npz")
    rows = {r["key"]: r for r in summary["results"]}
    selected = summary["selected"]
    plt.style.use("apj")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "legend.fontsize": 10})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.4), layout="constrained")
    for pop in LABELS:
        for kind, style in [("combined_score", "-"), ("neutral_score", "--")]:
            key = selected[pop][kind]
            if kind == "neutral_score" and key == selected[pop]["combined_score"]:
                continue
            r = rows[key]
            suffix = "joint diagnostic" if kind == "combined_score" else r"$x_{HI}$ only"
            ax.plot(
                arrays[key + "_z"],
                1 - arrays[key + "_q_volume"],
                style,
                color=COLORS[pop],
                lw=2.4 if style == "-" else 1.6,
                label=rf"{LABELS[pop]}, $f_{{esc}}={r['fesc']:g}$ ({suffix})",
            )
    with (ROOT / "external_data/observations/reionization/neutral_fraction.csv").open() as f:
        observations = list(csv.DictReader(f))
    names = {
        "mason18_7": "Mason+18",
        "davies18_709": "Davies+18",
        "mason26_65": "Mason+26 (wide z bins)",
        "mcgreer15_56": "McGreer+15 limits",
    }
    for r in observations:
        z, x, lo, hi, zlo, zhi = [
            float(r[k]) for k in ["z", "xhi", "lower", "upper", "z_min", "z_max"]
        ]
        wide = zlo != zhi
        if r["kind"] == "upper_limit":
            ax.errorbar(
                z,
                x,
                yerr=0.055,
                uplims=True,
                fmt="none",
                color="0.5",
                capsize=3,
                label=names.get(r["id"]),
            )
        else:
            ax.errorbar(
                z,
                x,
                yerr=[[x - lo], [hi - x]],
                xerr=[[z - zlo], [zhi - z]],
                fmt="s" if wide else "o",
                color="#875294" if wide else "black",
                mfc="white",
                ms=6,
                capsize=3,
                lw=1,
                label=names.get(r["id"]),
            )
    ax.axvspan(5.3, 6.01, color="0.93", zorder=0)
    ax.text(5.65, 0.95, "No spatial\noutputs", ha="center", va="top", fontsize=9, color="0.4")
    ax.set(
        xlim=(5.3, 13.5),
        ylim=(-0.015, 1.025),
        xlabel="Redshift z",
        ylabel=r"Volume-averaged neutral fraction $\langle x_{HI}\rangle_V$",
    )
    ax.legend(loc="lower right", fontsize=11.5)
    save(fig, "reionization_history")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax, bx = axes
    for axis in axes:
        axis.axhspan(0.047, 0.061, color="0.8", alpha=0.5)
        axis.axhline(0.054, color="0.4", lw=1, ls=":")
    for pop in LABELS:
        for kind, style in [("combined_score", "-"), ("neutral_score", "--")]:
            key = selected[pop][kind]
            if kind == "neutral_score" and key == selected[pop]["combined_score"]:
                continue
            r = rows[key]
            ax.plot(
                arrays[key + "_tau_z"],
                arrays[key + "_tau"],
                style,
                color=COLORS[pop],
                lw=2,
                label=rf"{LABELS[pop]}, $f_{{esc}}={r['fesc']:g}$; $\tau={r['tau']:.4f}$",
            )
            if kind == "combined_score":
                ax.fill_between(
                    arrays[key + "_tau_z"],
                    arrays[key + "_tau_late"],
                    arrays[key + "_tau_early"],
                    color=COLORS[pop],
                    alpha=0.18,
                )
        samples = sorted(
            [r for r in rows.values() if r["population"] == pop], key=lambda r: r["fesc"]
        )
        bx.plot(
            [r["fesc"] for r in samples],
            [r["tau"] for r in samples],
            "o-",
            color=COLORS[pop],
            ms=4,
            label=LABELS[pop],
        )
        chosen = rows[selected[pop]["combined_score"]]
        bx.plot(chosen["fesc"], chosen["tau"], "*", color=COLORS[pop], ms=15, mec="black", mew=0.5)
    ax.axvspan(0, 6.01, color="#9CBACF", alpha=0.15)
    ax.text(1, 0.005, "Assumed\ncompletion", fontsize=9, color="#39546A")
    ax.text(10, 0.0485, r"Planck 2018 total $\tau$: $0.054\pm0.007$", fontsize=9)
    ax.set(
        xlim=(0, 30),
        ylim=(0, 0.074),
        xlabel="Upper integration redshift z",
        ylabel=r"Cumulative Thomson optical depth $\tau(0,z)$",
    )
    ax.legend(loc="lower right", fontsize=10.5)
    bx.set(xlabel=r"Escape fraction $f_{esc}$", ylabel=r"Total reionization optical depth $\tau$")
    bx.legend(loc="upper left")
    bx.text(
        0.97,
        0.04,
        "Stars: joint diagnostic selection\nLines connect completed spatial runs",
        transform=bx.transAxes,
        ha="right",
        fontsize=9,
    )
    save(fig, "optical_depth")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    for ax, pop in zip(axes, LABELS, strict=True):
        samples = sorted(
            [r for r in rows.values() if r["population"] == pop], key=lambda r: r["fesc"]
        )
        f = [r["fesc"] for r in samples]
        ax.plot(
            f, [r["neutral_score"] for r in samples], "o--", label=r"Three narrow-z $x_{HI}$ points"
        )
        ax.plot(
            f, [r["combined_score"] for r in samples], "o-", label=r"Add Planck $\tau$ residual"
        )
        for kind, marker in [("neutral_score", "s"), ("combined_score", "*")]:
            r = rows[selected[pop][kind]]
            ax.plot(r["fesc"], r[kind], marker, color="black", ms=10)
        ax.set(
            title=LABELS[pop],
            xlabel=r"Escape fraction $f_{esc}$",
            ylabel="Descriptive residual score",
            ylim=(0, 3),
        )
        if pop == "popii_popiii":
            ax.set_xlim(0.05, 0.085)
        ax.legend(fontsize=10)
    save(fig, "calibration_scan")

    if a.slide_assets:
        deck = ROOT / "slides/popiii_heii_pisn_complete_20260916"
        for name, asset in [
            ("reionization_history", "calibrated_reionization_history"),
            ("optical_depth", "calibrated_optical_depth"),
            ("calibration_scan", "calibrated_scan"),
        ]:
            shutil.copy2(out / f"{name}.pdf", deck / "assets" / f"{asset}.pdf")
        lines = [
            r"\begin{center}\setlength{\tabcolsep}{4mm}\begin{tabular}{llrrrr}\toprule",
            r"模型 & 选择依据 & $f_{\rm esc}$ & $x_{\rm HI}(7)$ & $\tau$ & 分数 \\",
            r"\midrule",
        ]
        changes = []
        for pop in LABELS:
            for kind, label in [
                ("neutral_score", r"$x_{\rm HI}$"),
                ("combined_score", r"$x_{\rm HI}+\tau$"),
            ]:
                r = rows[selected[pop][kind]]
                lines.append(
                    f"{LABELS[pop]} & {label} & {r['fesc']:g} & {r['xhi_z7']:.3f} & "
                    f"{r['tau']:.5f} & {r[kind]:.3f}" + r" \\"
                )
                lo, hi = r["tau_lowz_completion_bounds"]
                changes.append(max(abs(lo - r["tau"]), abs(hi - r["tau"])))
        lines += [
            r"\bottomrule\end{tabular}\end{center}",
            f"共比较 {len(rows)} 组已完成历史。所列参数的低红移补全敏感性为 "
            f"$|\\Delta\\tau|\\leq {max(changes):.5f}$；不代表总系统误差。",
        ]
        ii = rows[selected["popii"]["combined_score"]]
        mixed = rows[selected["popii_popiii"]["combined_score"]]
        lines.append(
            f"\\par 纯 II 在 $z={ii['z99']:.2f}$ 达到 99\\% 电离；"
            f"混合模型在 $z=6.01$ 仍有 {100 * mixed['xhi_zmin']:.1f}\\% 中性体积。"
        )
        (deck / "reionization_calibration_values.tex").write_text("\n".join(lines) + "\n")
        tii, tmix = [rows[selected[pop]["combined_score"]]["tau_above_z"]["10"] for pop in LABELS]
        (deck / "reionization_tau_tail.tex").write_text(
            f"折中选择下，$z>10$ 贡献的光深：纯 Pop II 为 {tii:.4f}，混合模型为 {tmix:.4f}。"
            "混合模型的高红移电离尾部更长。\n"
        )


if __name__ == "__main__":
    main()
