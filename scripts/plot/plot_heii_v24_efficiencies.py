"""Split the completed He II comparison by efficiency, retaining original apertures.

Run from the repository root with PYTHONPATH=. .venv/bin/python.
Only renders existing summaries; never generates formation histories.
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.analysis.heii_v24_comparison import ROOT, digest, observations
from scripts.plot.review_heii_v24_targets import compile_tex

SUMMARY = ROOT / "data_save/heii_v24_targets_20260914/summary.json"
PARENT = ROOT / "slides/popiii_heii_v24_review_20260914"
DECK = ROOT / "slides/popiii_heii_v24_split_20260914"
PREVIEW = ROOT / "outputs/heii_v24_ingest"
EFFICIENCIES = (("0.01", "001"), ("0.03", "003"), ("0.1", "010"))
OBS_STYLE = {
    "LAP1_arclet": ("#97489c", "LAP1: 1$\\sigma$ upper limit\nVanzella23, Table 1"),
    "RXJ2129_z8HeII_A": ("#c34435", "RXJ2129-A\nWang24, Table 3"),
    "GNz11_HeII_clump_IFU": ("#202020", 'GN-z11: IFU 0.24" × 0.24"\nMaiolino24, Tables 1–2'),
    "GNz11_HeII_large_IFU": ("#af7c17", 'GN-z11: IFU 0.48" × 0.60"\nsame system, larger aperture'),
    "GNz11_HeII_MSA": ("#8362ad", 'GN-z11: MSA 0.20" × 0.20"\nsame system, alternative aperture'),
}


def plot(model, catalog, epsilon, tag):
    """Model population quantiles and observed luminosities, at their actual z."""
    quantiles = np.array(
        [
            r["efficiencies"][epsilon]["linear_log_age"]["young_uv_bright"][
                "luminosity_q16_q50_q84"
            ]
            for r in model
        ]
    )
    if (
        not np.isfinite(quantiles).all()
        or np.any(quantiles <= 0)
        or np.any(np.diff(quantiles, axis=1) < 0)
    ):
        raise ValueError("Invalid or nonpositive model quantiles")
    z = np.array([r["z"] for r in model])
    fig, ax = plt.subplots(figsize=(12, 4.9))
    model_color = "#286491"
    ax.errorbar(
        z,
        quantiles[:, 1],
        yerr=[quantiles[:, 1] - quantiles[:, 0], quantiles[:, 2] - quantiles[:, 1]],
        fmt="o",
        ms=8,
        mfc="white",
        mew=1.8,
        color=model_color,
        elinewidth=2.4,
        capsize=7,
        zorder=3,
        label="Model median\n16–84% population range",
    )
    # Explicit record selection includes MSA, which the older IFU-only plot omitted.
    # Keep all three GN-z11 apertures separate; never combine their measurements.
    for r in catalog["sources"]:
        color, label = OBS_STYLE[r["id"]]
        if r["measurement"] == "upper_limit":
            ax.errorbar(
                r["z"],
                r["luminosity"],
                yerr=r["luminosity"] * 0.48,
                uplims=True,
                fmt="_",
                ms=13,
                color=color,
                lw=1.8,
                capsize=5,
                label=label,
                zorder=5,
            )
        else:
            ax.errorbar(
                r["z"],
                r["luminosity"],
                yerr=r["luminosity_error"],
                fmt="*",
                ms=13,
                color=color,
                mec="white",
                mew=0.5,
                lw=1.6,
                capsize=4,
                label=label,
                zorder=6,
            )
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"$L_{\mathrm{He\,II}\,1640}\ [\mathrm{erg\ s^{-1}}]$",
        yscale="log",
        xlim=(6.15, 11.15),
        ylim=(3.3e38, 2.1e43),
        title=rf"Burst efficiency $\epsilon_b={epsilon}$",
    )
    ax.set_xticks([6.639, 8.1623, 10.6], ["6.639", "8.1623", "10.600"])
    ax.grid(axis="y", which="major", alpha=0.16)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),
        fontsize=12,
        frameon=False,
        labelspacing=0.85,
        handlelength=2,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.09, right=0.66, bottom=0.16, top=0.90)
    fig.savefig(DECK / "assets" / f"heii_eps{tag}.pdf")
    fig.savefig(PREVIEW / f"heii_eps{tag}.png", dpi=160)
    plt.close(fig)


def main():
    summary = json.loads(SUMMARY.read_text())
    catalog = observations()
    if catalog["catalog_sha256"] != summary["observations"]["catalog_sha256"]:
        raise ValueError("Observation catalog changed since the completed model summary")
    if {r["id"] for r in catalog["sources"]} != set(OBS_STYLE):
        raise ValueError("Unexpected observation records; review their meaning before plotting")
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
    DECK.mkdir(exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PARENT / "assets", DECK / "assets", dirs_exist_ok=True)
    for epsilon, tag in EFFICIENCIES:
        plot(summary["model"], catalog, epsilon, tag)

    source = (
        (PARENT / "popiii_heii_pisn.tex")
        .read_text()
        .replace(str(PARENT.relative_to(ROOT)), str(DECK.relative_to(ROOT)))
    )
    start = source.index(r"\begin{frame}[c]{目标红移的 He II 对照")
    end = source.index(r"\end{frame}", start) + len(r"\end{frame}")
    frames = []
    for epsilon, tag in EFFICIENCIES:
        frames.append(
            rf"\begin{{frame}}[c]{{He II 光度与原始观测：$\epsilon_b={epsilon}$}}"
            + "\n"
            + rf"\figwide{{heii_eps{tag}.pdf}}{{1.00}}"
            + "\n"
            + r"""模型选本征 $M_{\rm UV}\leq-20$、爆发年龄 $\leq3$ Myr；圆点和竖线表示中位数与 $16$–$84\%$ 分布。
\par 有效质量样本仅 $5$–$13$ 个，且团块与整晕口径不同；本图只比较光度尺度。
\end{frame}
"""
        )
    source = source[:start] + "\n".join(frames) + source[end:]
    tex = DECK / "popiii_heii_pisn.tex"
    tex.write_text(source)
    obsolete = DECK / "assets/heii_v24_luminosity.pdf"
    if obsolete.name not in source:
        obsolete.unlink()
    compile_tex(tex)
    manifest = dict(
        summary_path=str(SUMMARY.relative_to(ROOT)),
        summary_sha256=digest(SUMMARY),
        parent_pdf_sha256=digest(PARENT / "popiii_heii_pisn.pdf"),
        observation_catalog=catalog,
        model_selection="Resolved burst, age <=3 Myr, total intrinsic M1500 <= -20",
        model_interval="HMF-weighted 16–84% population range, not MC uncertainty",
        efficiencies=[e for e, _ in EFFICIENCIES],
        msa_inclusion="Explicit alternative aperture, not an independent galaxy",
        x_coordinates="True redshifts, no jitter or connecting interpolation",
        raw_products="unchanged; no science computation",
        caveats="Only 5–13 effective mass clusters; whole-halo and observed aperture mismatch",
        reviewed_pages=[8, 9, 10],
        visual_review="pending",
        pdf_sha256=digest(tex.with_suffix(".pdf")),
    )
    (DECK / "review.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(tex.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
