"""Publish the completed first batch with its measured sampling limitations."""

import json
import re
import shutil
import subprocess

from scripts.analysis.heii_v24_comparison import ROOT, digest


def compile_tex(tex, passes=2):
    for _ in range(passes):
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tex.parent),
                str(tex),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        tex.with_suffix(".compile.txt").write_text(result.stdout + result.stderr)
        result.check_returncode()
    problems = re.findall(
        r"(?:Overfull .+|.*undefined.*|.*Missing character.*)", tex.with_suffix(".log").read_text()
    )
    if problems:
        raise RuntimeError("\n".join(problems))


def main():
    review_path = ROOT / "outputs/heii_v24_ingest/scientific_review.json"
    review = json.loads(review_path.read_text())
    parent = ROOT / "slides/popiii_heii_v24_model_20260914"
    deck = ROOT / "slides/popiii_heii_v24_review_20260914"
    deck.mkdir(exist_ok=True)
    shutil.copytree(parent / "assets", deck / "assets", dirs_exist_ok=True)
    source = (
        (parent / "popiii_heii_pisn.tex")
        .read_text()
        .replace(str(parent.relative_to(ROOT)), str(deck.relative_to(ROOT)))
    )
    start = source.index(r"\begin{frame}[c]{目标红移的 He II 对照")
    end = source.index(r"\end{frame}", start) + len(r"\end{frame}")
    source = (
        source[:start]
        + r"""
\begin{frame}[c]{目标红移的 He II 对照：首批采样的光度尺度}
\figwide{heii_v24_luminosity.pdf}{1.00}
选择本征 $M_{\rm UV}\leq-20$、爆发年龄 $\leq3$ Myr；符号为中位数，竖线为丰度加权 $16$–$84\%$ 分布。
\par 有效独立质量样本仅 $5$–$13$ 个；这是初步结果，尚不能据此约束爆发效率。
\end{frame}
"""
        + source[end:]
    )
    start = source.index(r"\begin{frame}[c]{RXJ2129-A：在同一红移")
    end = source.index(r"\end{frame}", start) + len(r"\end{frame}")
    rows = []
    for row in review["rx"]:
        fraction = f"${100 * row['fraction']:.1f}\\%$" if row["above_count"] else "本批未抽到"
        rows.append(
            rf"${row['epsilon']:g}$ & ${row['above_count']}/{row['known_count']}$ & {fraction} & ${row['effective_clusters']:.1f}$ \\"
        )
    source = (
        source[:start]
        + r"""
\begin{frame}[c]{RXJ2129-A：强谱线尾部的采样仍然不足}
取 $z=8.1623$、本征 $M_{\rm UV}=-19.94\pm0.25$ mag 的分析窗口。\par
包含未爆发和年老爆发的对象；本页不施加 $3$ Myr 年龄筛选。
\gap
{\centering\renewcommand{\arraystretch}{1.7}
\begin{tabular}{cccc}
\toprule
$\epsilon_b$ & 达到观测光度 / 已知样本 & 丰度加权比例 & 有效质量样本\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}\par}
\gap
观测 $L_{1640}\simeq1.02\times10^{42}\,\mathrm{erg\,s^{-1}}$（Wang24，统一宇宙学并去尘）。
\par 另有 $4$–$6\%$ 的所选丰度来自左删失历史，谱线未知，未计入上表分母。
\par 首行比例仅由 $4$ 个强谱线样本贡献；目前不能判断哪档效率更符合观测。
\end{frame}
"""
        + source[end:]
    )
    tex = deck / "popiii_heii_pisn.tex"
    tex.write_text(source)
    compile_tex(tex)
    provenance = dict(
        parent_pdf_sha256=digest(parent / "popiii_heii_pisn.pdf"),
        review_sha256=digest(review_path),
        raw_products="unchanged",
        reviewed_pages=[8, 9, 10],
        visual_review="pending",
        pdf_sha256=digest(tex.with_suffix(".pdf")),
        review=review,
    )
    (deck / "review.json").write_text(json.dumps(provenance, indent=2) + "\n")
    previous = ROOT / "outputs/heii_v24_ingest/comparison.tex"
    text = (
        previous.read_text()
        .replace("我们统一处理的观测（模型补算中）", "我们的模型：年轻 UV 亮子样本（初步）")
        .replace(
            "slides/popiii_heii_v24_observations_20260914/assets",
            str(deck.relative_to(ROOT)) + "/assets",
        )
    )
    text = text.replace(
        "左图为去透镜后的光度；LAP1 采用原文的 $1\\sigma$ 上限。两图纵轴范围不同。右图保留 V24 原图，未改动观测点。",
        "左：本征 $M_{\\rm UV}\\leq-20$、爆发年龄 $\\leq3$ Myr；有效质量样本仅 $5$–$13$ 个，竖线为 $16$–$84\\%$ 分布。右：V24 原图，彩带扫描效率与质量损失模型。两图的样本与纵轴范围不同。",
    )
    comparison = previous.with_name("comparison-model.tex")
    comparison.write_text(text)
    compile_tex(comparison, passes=1)
    print(tex.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
