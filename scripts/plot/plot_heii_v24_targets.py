"""V24-target luminosity overlay and deck export; no formation-history computation."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.analysis.heii_v24_comparison import ROOT, digest, observations


def plot(catalog, model, output):
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
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    extrema = []
    if model is not None:
        for eps, color, marker in [
            (0.01, "#389077", "v"),
            (0.03, "#286491", "o"),
            (0.1, "#bb7836", "^"),
        ]:
            z = np.array([r["z"] for r in model])
            q = np.array(
                [
                    r["efficiencies"][str(eps)]["linear_log_age"]["young_uv_bright"][
                        "luminosity_q16_q50_q84"
                    ]
                    for r in model
                ]
            )
            if not np.isfinite(q).all() or np.any(q <= 0):
                raise ValueError("Nonpositive model quantile needs an explicitly different plot")
            ax.errorbar(
                z,
                q[:, 1],
                yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt=marker,
                color=color,
                ms=5,
                capsize=4,
                lw=1.5,
                alpha=0.85,
                label=rf"Model: $\epsilon_b={eps:g}$",
            )
            extrema.extend(q.flatten())
    colors = {"LAP1": "#9b4a93", "RXJ2129-z8HeII": "#bd4b42", "GN-z11 halo": "#303030"}
    for r in catalog["sources"]:
        if not r["plot"]:
            continue
        y, z, color = r["luminosity"], r["z"], colors[r["system"]]
        if r["measurement"] == "upper_limit":
            ax.errorbar(
                z,
                y,
                yerr=y * 0.48,
                uplims=True,
                fmt="_",
                color=color,
                ms=10,
                capsize=4,
                lw=1.6,
                label="LAP1: 1$\\sigma$ upper limit (Vanzella23)",
            )
            extrema.append(y * 0.5)
        else:
            name = (
                "RXJ2129-A (Wang24)"
                if r["system"].startswith("RXJ")
                else (
                    "GN-z11: clump (Maiolino24)"
                    if r["primary"]
                    else "GN-z11: larger aperture (same system)"
                )
            )
            ax.errorbar(
                z,
                y,
                yerr=r["luminosity_error"],
                fmt="D" if r["primary"] else "s",
                color=color,
                mfc=color if r["primary"] else "white",
                ms=7,
                capsize=3,
                lw=1.4,
                label=name,
                zorder=10,
            )
            extrema.extend([y - r["luminosity_error"], y + r["luminosity_error"]])
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"$L_{\mathrm{He\,II}\,1640}\ [\mathrm{erg\ s^{-1}}]$",
        yscale="log",
        xlim=(6.15, 11.15),
        ylim=(min(extrema) / 1.5, max(extrema) * 2.3),
    )
    ax.set_xticks([6.639, 8.1623, 10.6], ["6.639", "8.1623", "10.600"])
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11, frameon=False, labelspacing=1.0
    )
    fig.subplots_adjust(left=0.1, right=0.65, bottom=0.18, top=0.96)
    fig.savefig(output)
    preview = ROOT / "outputs/heii_v24_ingest" / (output.parent.parent.name + ".png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=150)
    plt.close(fig)


def new_frames(model):
    if model is None:
        title = "V24 引用的 He II 观测：先统一光度与观测对象"
        detail = "LAP1 保留原文的 $1\\sigma$ 上限；RXJ2129-A 已校正透镜和尘埃。"
    else:
        title = "目标红移的 He II 对照：比较光度尺度"
        detail = "模型点为 $M_{\\rm UV}\\leq-20$、爆发年龄 $\\leq3$ Myr 的样本中位数；竖线为丰度加权 $16$–$84\\%$ 范围。"
    frames = rf"""
\begin{{frame}}[c]{{{title}}}
\figwide{{heii_v24_luminosity.pdf}}{{1.00}}
{detail}
\par GN-z11 两个点来自同一系统的不同 IFU 孔径；外围团块使用自己的谱线通量。
\end{{frame}}

\begin{{frame}}[c]{{观测怎样接入模型：红移、光度和对象必须对应}}
\[ L_{{1640}}=\frac{{4\pi D_L^2(z)F_{{1640}}^{{\rm obs}}}}{{\mu}}. \]
LAP1：$z=6.639$、$\mu=120$；未检出连续谱，$M_{{2000}}>-11.2$。\par
RXJ2129-A：$z=8.1623$、$\mu=2.26$；使用原文已去透镜、去尘的线通量，避免重复校正。\par
GN-z11：$z=10.600$ 的外围 He II 团块；小孔径 $0.24''\times0.24''$，大孔径 $0.48''\times0.60''$。
\gap
模型每条主支历史至多一次爆发，尚无局部团块与孔径模型。LAP1、GN-z11 只作光度尺度对照；RXJ2129 可按 UV 亮度比较。
\par 文献：HEII-OBS-006 Vanzella23；007 Wang24；008 Maiolino24。
\end{{frame}}
"""
    if model is not None:
        rx = next(r for r in model if np.isclose(r["z"], 8.1623, rtol=0, atol=1e-8))
        rows = []
        for eps in (0.01, 0.03, 0.1):
            d = rx["efficiencies"][str(eps)]["linear_log_age"]["rx_uv_matched"]
            q = np.asarray(d["luminosity_q16_q50_q84"]) / 1e41
            rows.append(
                rf"${eps:g}$ & ${q[1]:.2g}\;[{q[0]:.2g},\,{q[2]:.2g}]$ & ${100 * d['fraction_at_or_above_reference']:.1f}\%$ \\"
            )
        frames += (
            r"""
\begin{frame}[c]{RXJ2129-A：在同一红移与 UV 亮度下比较}
取 $z=8.1623$，去尘后的 $M_{\rm UV}\simeqRXCENTER\pm0.25$ mag 窗口；窗口宽度是分析选择。
\par 模型包括尚未爆发的零信号对象；已知历史按晕丰度加权，未施加年轻年龄筛选。
\gap
{\centering\renewcommand{\arraystretch}{1.65}
\begin{tabular}{ccc}
\toprule
$\epsilon_b$ & $L_{1640}$ 中位数 $[16\%,84\%]$ & $L\geq L_{\rm RX}$ 的比例\\
 & $[10^{41}\,\mathrm{erg\,s^{-1}}]$ & \\
\midrule
"""
            + "\n".join(rows)
            + r"""
\bottomrule
\end{tabular}\par}
\gap
观测光度约 $10^{42}\,\mathrm{erg\,s^{-1}}$（Wang24）；表中比例描述模型光度分布，不是检出概率或拟合显著性。
\par UV 匹配采用原文 $A_V=0.12$ 与 Calzetti 去尘；谱线沿用 Case-B、零逃逸和基准年龄插值。
\end{frame}
"""
        )
    if model is not None:
        frames = frames.replace("RXCENTER", f"{rx['rx_intrinsic_muv_center']:.2f}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--base-deck", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    args = parser.parse_args()
    catalog = observations()
    model = None
    if args.summary is not None:
        summary = json.loads(args.summary.read_text())
        if summary["observations"] != catalog:
            raise ValueError("Observation catalog or cosmology changed since model summary")
        model = summary["model"]
    base = (ROOT / args.base_deck).resolve(strict=True)
    deck = (ROOT / args.deck).resolve()
    if deck == base:
        raise ValueError("Use a new deck version; running UVLF job freezes the old deck")
    (deck / "assets").mkdir(parents=True, exist_ok=True)
    for asset in (base / "assets").iterdir():
        if asset.is_file():
            shutil.copy2(asset, deck / "assets" / asset.name)
    plot(catalog, model, deck / "assets/heii_v24_luminosity.pdf")
    source = (base / "popiii_heii_pisn.tex").read_text()
    source = source.replace(
        str(args.base_deck) + "/assets/", str(deck.relative_to(ROOT)) + "/assets/"
    )
    anchor = r"\begin{frame}[c]{He II 比较中的首要数值问题：年轻 SSP 的年龄采样}"
    if source.count(anchor) != 1:
        raise ValueError("Expected unique insertion point after the existing He II/JWST comparison")
    source = source.replace(anchor, new_frames(model) + "\n" + anchor)
    tex = deck / "popiii_heii_pisn.tex"
    tex.write_text(source)
    for _ in range(2):
        process = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(deck),
                str(tex),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        (deck / "compile.stdout.txt").write_text(process.stdout + process.stderr)
        process.check_returncode()
    log = (deck / "popiii_heii_pisn.log").read_text()
    problems = re.findall(r"(?:Overfull .+|.*undefined.*|.*Missing character.*)", log)
    if problems:
        raise RuntimeError("Slide compilation needs review: " + "\n".join(problems))
    provenance = dict(
        base_deck=str(base),
        base_tex_sha256=digest(base / "popiii_heii_pisn.tex"),
        observations=catalog,
        model_summary=None if args.summary is None else str(args.summary.resolve()),
        model_summary_sha256=None if args.summary is None else digest(args.summary),
        new_pages=[8, 9] if model is None else [8, 9, 10],
        visual_review="pending",
        pdf_sha256=digest(deck / "popiii_heii_pisn.pdf"),
    )
    (deck / "heii_v24_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(deck / "popiii_heii_pisn.pdf")


if __name__ == "__main__":
    main()
