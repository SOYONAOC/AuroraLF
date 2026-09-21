# Pop III 爆发效率 3% 的文献核查

核查日期：2026-09-13。结论：相对于下述同分母模型基准，3% 是偏高的探索情景，不能表述为文献测定值；现有证据也不支持把它当作被普遍排除的数值。本次未修改生产效率或重新运行 UVLF / 再电离模型。

## 当前参数的含义与来源

`configs/experiments/ionizing_rates.toml` 的 `epsilon_b=0.03` 表示
`MstarIII = epsilon_b * (omega_b/omega_m) * Mh(tb)`，即首次越阈时一次爆发转化的恒星质量相对于宇宙重子比例分配的全晕质量。不是自由落体时间效率，也不是冷气体局部效率。

实现：`auroralf/experiments/random_q.py` 的 `first_crossing` / `burst_light` 保留爆发时质量和当前年龄；`auroralf/experiments/ionizing_rates.py` 通过 `epsilon_b * omega_b/omega_m * fesc_popiii` 乘年龄核。固定历史、SSP、逃逸率时，Pop III 当前产光率随效率线性变化。

项目来源：`docs/visbal-duty-experiment.md` 的 2026-09-09 R032 结果表明，3% 接近当时 z=12.5 的 UVLF 幅度，记录明确说这不是拟合。不是从某篇论文校准得到的 3%。

## 原始文献证据

| 文献 | 原文位置 | 参数及含义 | 与当前模型的关系 |
|---|---|---|---|
| [Zackrisson et al. 2012](https://arxiv.org/abs/1204.0517v3), MNRAS 427, 2212 | §2.4, Eq.1, PDF pp.4–5 | ε=MstarIII/[Mh Ωb/Ωm]；基准 log10 ε=−3，即0.1%；同时研究更广范围 | 同分母、同爆发质量公式；3% 是其基准30倍。基准参考 Safranek-Shrader，不是独立观测证据 |
| [Safranek-Shrader et al. 2012](https://arxiv.org/abs/1205.3835v2), MNRAS 426, 1159 | §7, PDF p.19 | ε=fstar fcold；乐观 J21=10 估算约7e4 Msun冷气体，取10%形成约7000 Msun恒星，Mh≈3e7 Msun，ε≈10^−2.9 | 约0.126%是特定延迟冷却晕的近似估算；不能当作所有晕的上限 |
| [Muñoz et al. 2022](https://arxiv.org/abs/2110.13919v2), MNRAS 511, 3657 | §2.1–2.3, Table1 | EOS: log10 fstar7III=−2.5，即0.316%；OPT: −1.75，即1.78%；两组alphaIII=0 | 同重子归一化，但通过 SFR=Mstar/(0.5 H^-1) 供光，宿主为分子冷却晕；不是首次越阈的一次爆发。3%与EOS仅归一化比为9.49 |

Muñoz OPT 同时改变逃逸率等参数，不可只移植其效率。文献较高效率的探索说明：不能仅凭某一低效率基准就宣布 3% 被排除。上述模型假设、气体估算、当前模型推断应分开。

## 对当前计算的含义

用当前 Ωb/Ωm=0.04897/0.30966，Mh(tb)=1e8 Msun 时：0.1%、0.3%、3% 分别一次形成 1.5814e4、4.7442e4、4.7442e5 Msun Pop III 恒星。这是解析质量换算，并非新增模拟。

固定历史与核函数，3%→0.3%使 Pop III 供光与电离光子率减少10倍，使 III-only UV 绝对星等变暗2.5mag；逐晕总光、LF和再电离历史不能直接同比缩放。

建议将0.1%、0.3%、1%、3%作为敏感性比较，而不是声称其中某值是观测最佳值。冷气体供应、原初气体存活和污染门控仍需独立建模；单纯降低常数效率不会自动获得低红移的物理熄灭过程。具体门控必须避免对已经定义的有效效率重复乘同一气体损失。

## 核查材料与范围

- Zackrisson：现有本地论文库的 PDF 与 `source/Palantir7.tex`，版本身份见 `external_data/literature_sources/popiii_uvlf_library/manifest.json`。已检查 PDF p.5；证据裁剪及哈希存于 `outputs/21cm_map/epsilon_audit/zackrisson-fiducial.*`。
- Safranek-Shrader：通过网页工具打开 arXiv PDF v2，读取 §7；本机 curl 获取 PDF 因 SSL 证书链验证失败，未生成本地 PDF。未绕过证书检查。
- Muñoz：核对 arXiv v2 HTML 与已有 `external_data/literature_sources/munoz2022_eos/source/main.tex`，精确版本与源归档哈希见同目录上一级的 `provenance.json`。新 PDF 获取失败，未声称检查其本地 PDF。
- 数值换算：`outputs/21cm_map/epsilon_audit/scalings.json`。汇报：`slides/21cm_map/popiii_efficiency.tex` / `.pdf`。
