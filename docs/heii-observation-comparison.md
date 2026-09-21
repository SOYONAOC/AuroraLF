# ε=0.03：UV 亮度与 He II 1640 的观测对照

2026-09-10；沿用 [He II 后处理](heii-random-q-study.md) 的模型与真实运行产物。
展示见 [slides 第 16–18 页](../slides/heii_random_q_20260910/heii_random_q.pdf)。

本次完成固定模型的对象级初步对照：在相近红移、相同 UV 亮度区间计算
Pop III He II 通量分布，再叠加实测线通量或上限。没有对 ε、IMF 或气体效率
重新拟合。两对象尚不足以据此确认或排除模型。

2026-09-16 当前结果：作业159041已补算实际观测红移12.342和13.86，
2026-09-17：AUR-S01第9页合并五个红移的效率0.03模型与观测，
高红移仍按目标UV窗口选择，低红移仍使用年轻UV亮样本。效率0.01页面已删除。
下方早期近邻红移分析保留为历史记录；新数值见文末“Exact observed redshifts”。见
[文献图形对照与实现口径](heii-figure-conventions.md)。

## 观测口径

| 对象 | 观测 z | MUV / AB mag | He II 通量 / erg s−1 cm−2 | 模型样本 z |
| --- | --- | --- | --- | --- |
| GHZ2 | 12.342±0.009 | −20.53±0.01 | (2.7±1.6)e−19 | 12.5 |
| JADES-GS-z14-1 | 13.86(+0.04,−0.05) | −19.0±0.4 | <7e−20（3σ） | 14.5 |

GHZ2 数据来自 [Castellano et al. 2024，Table 1、§III.1/2/5](https://arxiv.org/html/2403.10238v2)。
He II 与 O III] 混合，采用双高斯分离；表中 SNR=5 来自直接积分，不能替代
所列拟合误差。MUV 已去透镜放大；谱线通量仍含 μ=1.3，故模型通量乘 1.3。
论文通量已作狭缝及孔径校正；GHZ2 同时有显著金属线。

GS-z14-1 来自 [Wu et al. 2025，Table 2、§III.3/4](https://arxiv.org/html/2507.22858)。
约 56 h NIRSpec PRISM，未分辨高斯线宽固定为仪器分辨率；通量误差考虑谱线
协方差和红移抽样，狭缝损失采用点源几何并经 NIRCam 校准。本分析采用 μ=1。
上限不是“零通量±上限/3”的实测点。本次未读取原始谱线协方差或构造仪器似然。

机器可读原始录入及来源位于 `external_data/observations/heii/jwst_targets.json`
及同目录 README；保持仓库现有 external_data 本地存放约定。
Hebe 为 z≈10.6 的团块目标，未将现有 z=12.5/14.5 星系样本套用到它。

## 模型选择与图形

- 复用 z=12.5 的 R032，以及 z=14.5 的 R024–R027；独立批次取平均，使用原
  HMF 每轨道权重。核验父分析来源、运行 manifest 及全部运行产物 SHA-256。
- 横轴保留原 UV 代理量：Pop II 恒星 1600 Å + Pop III 总 1500 Å，未添加新的
  光谱转换；它尚不是完全匹配观测的合成光度学。
- 纵轴为 μ L1640/(4πDL²)。按目标红移计算 DL，但不改变原样本年龄、质量、
  UV 和 HMF 权重；**这只是近邻红移近似**，尤其 14.5 与 13.86 的差异需补算。
- 固定 ε=0.03，使用匹配 logE SSP 的 Pop III Case-B 贡献、零逃逸、无尘。
  尚无额外 Pop II/AGN/冲击线；气体改变时需要自洽更新连续谱。
- 每 0.5 mag 一个区间，画中位数与 16–84% 丰度加权分布。阴影描述对象散布，
  不是拟合置信区间或 MC 标准误。点间线段只连接离散区间统计。
- 未发生爆发的对象保留为零 Pop III 线；左删失的未知事件排除于线统计，
  同时保存未知权重比例，不能伪装成弱线对象。图示范围内三个基准分位数均正。
- 虚线为 log L 对 log age 插值的中位数，用于展示 SSP 时间采样敏感性；
  三种插值不构成物理不确定度区间。

## 对象窗口的结果

以观测 MUV 为中心、半宽 0.25 mag，通量均以 1e−19 erg s−1 cm−2 为单位：

| 统计 | GHZ2 | GS-z14-1 |
| --- | ---: | ---: |
| 基准中位数 [16%,84%] | 6.07 [0.478,24.91] | 0.920 [0.0685,3.701] |
| 真实模型通量 ≤ 实测中心值/上限的权重比例 | 37.99% | 44.34% |
| L 对 age 插值中位数 | 7.45 | 1.127 |
| log L 对 log age 插值中位数 | 1.84 | 0.269 |
| log L 对 log age：低于参考通量的比例 | 54.53% | 58.10% |
| 未知事件占所选总权重 | 0.0735% | 0.0370% |
| 已知对象中的零 Pop III 线比例 | 0.1514% | 0.0791% |
| 已知对象原始个数（不是有效独立样本量） | 54,747 | 214,380 |
| 参与质量族数 | 489 | 2,376 |
| 权重有效质量族数 (ΣW)²/ΣW² | 84.1 | 378.9 |

基准为 L 对 log age 插值。拓宽窗口至半宽 0.5 mag 后，基准中位数为 5.07、
0.931；低于参考通量的比例为 40.25%、44.84%。有限窗口是分箱敏感性检查，
尚未边缘化 MUV 的观测误差；GS-z14-1 的 ±0.4 mag 误差必须纳入正式比较。

GHZ2 的基准中位数约为测量中心值的 2.25 倍，GS-z14-1 的约为其上限的
1.31 倍，但模型分布宽且对年龄插值敏感。上述 CDF 比例仅计算真实模型通量，
**不是噪声卷积后的未检出概率、p 值、排除置信度或模型后验概率**。

## 复现与下一步

从仓库根目录运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/compare_random_q_heii_observations.py
```

要求父分析 `data_save/heii_random_q_20260910/summary.json` 及其绑定的来源、
SSP 与真实运行产物全部存在且哈希一致。缺失或变化会报错，不合成替代数据。
默认输出为 `data_save/heii_observations_20260910/summary.json`，包含所有分箱、
两个对象窗口、三种插值、未知比例及来源哈希；矢量图位于现有 slides 的
`assets/uv_heii_observations.pdf`，预览及日志位于 `outputs/heii_observations_20260910/`。
CLI 的路径参数相对仓库根目录，原 SSP 路径沿用父分析的配置目录约定。

后续正式检验依赖：目标红移样本；更细的年轻 SSP；UV 测量误差和谱线噪声；
气体响应与各谱线来源；孔径/线宽匹配与观测选择函数。当前图不承担这些尚未
完成的似然计算，也不替代 He II 光度函数的巡天完备性分析。

## JWST 判据更新（2026-09-11）

观测预测按 [波段、模式与谱线噪声判据](heii-jwst-observing-criteria.md) 执行。
固定通量门槛仅保留为分布摘要；新的灵敏度曲线和检出率需真实观测噪声或
指定配置的 ETC 输出。本次未运行 ETC，未更改既有模型结果。
## Exact observed redshifts (2026-09-16; complete)

The high-redshift comparison now uses newly calculated GHZ2 `z=12.342` and
GS-z14-1 `z=13.86` populations. The earlier sections describe historical
`z=12.5,14.5` proxy samples, superseded for these target comparisons.

Plan: `configs/experiments/heii_exact_targets_20260916.json`; typed configurations
are `heii_ghz2_exact_20260916.toml` (R053) and
`heii_gsz14_exact_20260916.toml` (R054). Each uses one batch of 3600 masses,
1000 tracks per mass and 960 time samples. Efficiencies 0.01, 0.03 and 0.1
separately select the total-UV target windows. The previous high-z physics,
mass range and UV proxy (Pop II 1600 A plus Pop III 1500 A) are retained.
There is no young-burst age cut. Known zero emitters and unknown early bursts
remain distinct. Quantiles describe population diversity, not measurement errors.

Job **159041** completed successfully in 747.4 seconds on local DMDE SLURM `node5`, partition `cpu`, with 33 workers
from 34 idle CPUs, leaving one idle CPU at placement. Mail is configured to
`lighmisamisa@agent.qq.com` on END/FAIL. Job 159040 failed before computation
because the submission lacked `PYTHONPATH=.`; 159041 explicitly exports it.
No scientific parameters changed for this repair. The event-only watcher
continues this same task on completion for output checks, three efficiency
plots, current AUR-S01 slides and Zotero synchronization.

Verified output: `data_save/heii_exact_targets_20260916/{manifest.json,summary.json,z12.342.npz,z13.86.npz}`.
The manifest freezes input hashes and archives the implementation. Six focused
tests and SSP/input validation passed before submission. A scheduler exit code
alone does not establish scientific convergence.

The output manifest/product SHA-256 values and raw sample dimensions/redshifts
were checked after completion. Both redshifts have 3,600,000 histories. The
following luminosities are intrinsic, in erg/s, using baseline linear-L/log-age
SSP interpolation and the +/-0.25 mag UV selection.

| Efficiency | Target | L16 | L50 | L84 | Effective mass samples |
| --- | --- | --- | --- | --- | --- |
| 0.01 | GHZ2 | 1.027e+21 | 2.867e+41 | 3.512e+42 | 128.0 |
| 0.01 | JADES-GS-z14-1 | 1.463e+40 | 1.663e+41 | 9.558e+41 | 98.6 |
| 0.03 | GHZ2 | 7.661e+40 | 9.733e+41 | 4.194e+42 | 77.0 |
| 0.03 | JADES-GS-z14-1 | 1.946e+40 | 2.282e+41 | 1.088e+42 | 89.6 |
| 0.1 | GHZ2 | 7.539e+40 | 1.048e+42 | 4.211e+42 | 85.6 |
| 0.1 | JADES-GS-z14-1 | 1.857e+40 | 4.567e+41 | 1.081e+42 | 83.3 |

GHZ2 observation: (4.51 +/- 2.67)e41 erg/s; GS-z14-1: 3-sigma upper
limit 1.97e41 erg/s. Unknown weighted fractions are 0.009%–0.645%;
known-zero fractions are 0.017%–1.351%. The very low GHZ2 L16 at efficiency
0.01 is retained and explicitly annotated below the plotting range. No claim
of full convergence, formal fit or model rejection follows from these intervals.
Current AUR-S01 is 24 pages: exact-redshift high-z comparisons pages9–10,
existing low-z comparisons pages11–13. The high-z efficiency0.1 slide was
removed at user request; its scientific products are retained. Physical assumptions and the separate
young-SSP interpolation diagnostics remain in the frozen plan and summary.

Final PDF compiled twice with XeLaTeX; modified pages9–11 and25 visually
checked with no visible clipping or overlap. Zotero AUR-S01 (DXT5453A) updated
and deep-verified; QA record: outputs/heii_literature_figures_20260916/exact-review.json.

## 展示合并（2026-09-17）

当前讲稿共20页，第9页合并五个红移的epsilon=0.03对照。
两张epsilon=0.01页、独立低红移epsilon=0.03页及epsilon=0.1页均已按用户要求移除。
高、低红移采用不同符号和颜色，并明确列出不同选样条件；未重新计算或改写摘要。
低红移模型只有约5–11个有效独立质量样本，GN-z11不同孔径不是独立星系。
