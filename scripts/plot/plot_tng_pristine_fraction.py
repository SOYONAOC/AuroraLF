"""Render the TNG resolved-ancestor diagnostic and its resolution limits."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.analysis.tng_pristine_fraction import OUTPUT, ROOT
from scripts.plot.review_heii_v24_targets import compile_tex

DECK = ROOT / "slides/tng_pristine_20260914"
PREVIEW = ROOT / "outputs/tng_pristine_20260914"


def select(summary, particles, delay, branch="all"):
    return sorted(
        [
            r
            for r in summary
            if r["particles"] == particles
            and r["delay_myr"] == delay
            and r["branch"] == branch
            and r["logmass_low"] == 9
            and r["logmass_high"] == 13
        ],
        key=lambda r: r["z"],
    )


def save(fig, name):
    fig.savefig(DECK / "assets" / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=160)
    plt.close(fig)


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Incomplete tree analysis")
    summary = json.loads((OUTPUT / "summary.json").read_text())
    (DECK / "assets").mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
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
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.35), sharex=True, sharey=True)
    for ax, nmin in zip(axes, (20, 100), strict=True):
        short = select(summary, nmin, 10)
        medium = select(summary, nmin, 30)
        if any(a["fraction"] != b["fraction"] for a, b in zip(short, medium, strict=True)):
            raise ValueError("10 and 30 Myr differ; split their legend entries")
        for delay, color, marker, label in (
            (30, "#286491", "o", "Delay = 10 or 30 Myr"),
            (100, "#bb7836", "s", "Delay = 100 Myr"),
        ):
            rows = select(summary, nmin, delay)
            x, y = [r["z"] for r in rows], [100 * r["fraction"] for r in rows]
            ax.plot(x, y, marker=marker, lw=2, ms=7, color=color, label=label)
        ax.set(
            title=f"Ancestor cut: ≥{nmin} DM particles",
            xlabel="Redshift $z$",
            yscale="log",
            xlim=(5.7, 12.3),
            ylim=(0.085, 130),
        )
        ax.set_xticks([6.0108, 8.0122, 9.9966, 11.9802], ["6.01", "8.01", "10.00", "11.98"])
        ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Unvetoed halo fraction [%]")
    axes[1].legend(loc="lower right", fontsize=12)
    fig.subplots_adjust(wspace=0.09, left=0.09, right=0.99, bottom=0.18, top=0.90)
    save(fig, "fraction_redshift")

    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    for nmin, color in ((20, "#286491"), (100, "#bb7836")):
        for delay, style in ((30, "--"), (100, "-")):
            full, main = select(summary, nmin, delay), select(summary, nmin, delay, "main")
            delta = 100 * np.array(
                [m["fraction"] - f["fraction"] for m, f in zip(main, full, strict=True)]
            )
            if np.any(delta < -1e-12):
                raise ValueError("Secondary ancestry cannot remove an existing veto")
            ax.plot(
                [r["z"] for r in full],
                delta,
                style,
                marker="o",
                ms=6,
                color=color,
                label=f"≥{nmin} particles, delay {delay:g} Myr",
            )
    ax.set(
        xlabel="Redshift $z$",
        ylabel="Additional veto\n[percentage points]",
        xlim=(5.7, 12.3),
        ylim=(-0.015, 0.57),
    )
    ax.set_xticks([6.0108, 8.0122, 9.9966, 11.9802], ["6.01", "8.01", "10.00", "11.98"])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=12, ncol=2)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.79)
    save(fig, "secondary_ancestry")

    # Measured from raw SubhaloMass/SubhaloLen, recorded in the independent audit.
    audit = json.loads((OUTPUT / "audit.json").read_text())
    particle_mass = audit["dm_particle_mass_msun"]
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    z = np.array(manifest["snapshot_z"])
    cooling = np.array(manifest["atomic_cooling_mass_msun"])
    ax.plot(z, cooling, "o-", color="#286491", label=r"Atomic cooling: $T_{\rm vir}=10^4$ K")
    for nmin, color in ((20, "#bb7836"), (100, "#8362ad")):
        ax.axhline(
            nmin * particle_mass,
            color=color,
            ls="--",
            lw=2,
            label=rf"{nmin} DM particles: ${nmin * particle_mass / 1e8:.2f}\times10^8\,M_\odot$",
        )
    ax.set(
        xlabel="Redshift $z$",
        ylabel=r"Mass [$M_\odot$]",
        yscale="log",
        xlim=(5.7, 20.5),
        ylim=(2e7, 1.3e9),
    )
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.96)
    save(fig, "mass_resolution")

    rows = []
    for a, b in zip(select(summary, 20, 30), select(summary, 100, 30), strict=True):
        rows.append(
            rf"${a['z']:.2f}$ & ${100 * a['fraction']:.3f}\%$ & ${100 * b['fraction']:.3f}\%$ & ${a['sampled_total']}$ \\"
        )
    tex = DECK / "pristine.tex"
    tex.write_text(
        r"""\documentclass[aspectratio=169,10pt]{beamer}
\geometry{paperwidth=216mm,paperheight=121.5mm}
\input{slides/templates/formula-explained.tex}
\usepackage{graphicx}
\graphicspath{{slides/tng_pristine_20260914/assets/}}
\renewcommand{\AcademicFooter}{}
\renewcommand{\normalsize}{\fontsize{12}{17}\selectfont}
\hypersetup{pdftitle={TNG 祖先污染与 Pop III 候选比例}}
\begin{document}
\begin{frame}[c]{TNG 祖先污染：低红移仍未被排除的候选较少}
\includegraphics[width=\textwidth]{fraction_redshift.pdf}
分母：原子冷却条件满足、$10^9\leq M_{200c}/M_\odot<10^{13}$ 的目标中心晕；恢复原模拟数量权重。
\par 假设可冷却祖先在延迟后充分污染后代；曲线是未被已解析祖先排除的比例，尚不能认定为原初。
\end{frame}

\begin{frame}[c]{30 Myr 延迟：结果明显依赖低粒子数祖先}
\begin{center}\renewcommand{\arraystretch}{1.8}
\begin{tabular}{cccc}
\toprule
红移 & 祖先 $\geq20$ 粒子 & 祖先 $\geq100$ 粒子 & 目标样本数\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{4mm}
同一批 TNG100-1-Dark 树中改变祖先粒子数筛选，不是两套不同分辨率模拟。
\par 快照间隔约 $30$–$100$ Myr，本次 $10$ 与 $30$ Myr 延迟无法区分。
\par 只统计现有四个目标快照；$z=11.98$ 不能标成 $12.5$ 或 $14.5$。
\end{frame}

\begin{frame}[c]{加入旁支祖先：本次判据主要由主分支触发}
\includegraphics[width=\textwidth]{secondary_ancestry.pdf}
纵轴为：只检查主分支的未排除比例，减去检查全部祖先的未排除比例。
\par 本次额外排除量不足 $0.5$ 个百分点；不等于旁支贡献的金属质量很少。
\end{frame}

\begin{frame}[c]{质量分辨率：原子冷却阈值低于祖先筛选尺度}
\includegraphics[width=\textwidth]{mass_resolution.pdf}
暗物质粒子质量约 $8.86\times10^6\,M_\odot$；原子冷却阈值仅相当于约 $3$–$18$ 个粒子。
\par 小祖先与分子冷却成星未被可靠追踪；未找到污染祖先，不能等同于从未受污染。
\end{frame}

\begin{frame}[c]{简化判据与适用范围}
对每个目标晕，沿完整 SubLink 树寻找所有更早的祖先：
\[
M_{\rm sub}(t_i)\geq M_{\rm cool}(z_i),\qquad
N_{{\rm DM},i}\geq N_{\min},\qquad
t_{\rm target}-t_i\geq\tau_{\rm enrich}.
\]
若存在满足以上条件的祖先，就在本次充分混合模型中排除该目标晕。
\par $\tau_{\rm enrich}=10,30,100$ Myr 是冷却、成星、金属释放与混合的总延迟试验值；不是拟合结果。
\par 祖先质量使用束缚子晕质量作为代理；没有计算金属产额、外部污染或局部原初气体存活。
\par 这是目标晕的资格比例，未运行随机临界质量模型，也不是 Pop III 爆发率。
\par 祖先污染的思路参考 Trenti \& Stiavelli09，\S2.5；此处判据为本项目的简化试验。
\end{frame}
\end{document}
"""
    )
    compile_tex(tex)
    print(tex.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
