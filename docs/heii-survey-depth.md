# He II：选样记录

## 当前正文：−20 等且年龄 ≤3 Myr 的年轻模型样本

按用户最新要求，采用 `MUV<=-20` 并筛选爆发年龄 `<=3 Myr`。星等阈值标为
本工作的模型筛选条件，不表示文献标准；五个红移使用相同切选与 1500 Å 总 UV。
正文只画这一母样本的 HMF 加权中位数及 16–84% 分布，不施加谱线通量限。
保留原参数、历史、SSP 与观测数据，仅重做选样后处理。

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/heii_survey_depth.py --muv-limit -20 --max-age-myr 3
PYTHONPATH=. .venv/bin/python scripts/plot/plot_heii_survey_depth.py --summary data_save/heii_uv_bright_20260918/summary.json
```

产物为 `data_save/heii_uv_bright_20260918/{summary,manifest}.json`，图为
`slides/popiii_heii_pisn_complete_20260916/assets/heii_uv_bright.pdf`；图形来源记录
位于 `outputs/heii_uv_bright_20260918/provenance.json`。JSON 内额外的谱线过线统计
仅供辅助比较，未用于正文筛选。只纳入已解析、年龄不超过 3 Myr 的爆发；未发生或年龄未知的爆发不属于这一年轻子样本。

年龄选择的依据：[Schaerer (2002)](https://arxiv.org/abs/astro-ph/0110697) 指出
瞬时爆发的 He II 线在约 3 Myr 后显著衰退；这支持定义年轻子样本，不是普遍
观测年龄限。本模型 Raiter SSP 的单位初始质量线光度在 1、3 Myr 分别为
`3.77268e35`、`6.28e31 erg/s/Msun`，年龄衰退已由 SSP 计算。按用户明确选择，正文使用 `--max-age-myr 3` 展示年轻子样本，
不代表所有 UV 亮星系或观测可检出对象的分布。省略该参数可复现不筛年龄的选样。

## 备查方案：文献参考深度选样

以下为上一版结果，保留在附录 A.5，未用于当前正文选样。

2026-09-18，按用户指定采用 [Vikaeus et al. 2022，表 1](https://arxiv.org/pdf/2107.01230v2#page=6)
的深度 NIRCam/NIRSpec 方案：5σ 点源成像深度 30.6 AB mag，线通量限
`2.9e-19 erg/s/cm²`（该表深度 NIRSpec 约 28 h）。这是论文的参考方案，
不是图中各观测目标的实际选择函数，也不是当前仪器所有模式的统一灵敏度。

## 选择与物理量

- 保持 `q_log10_mean=0`、`q_log10_sigma=1.5`、`epsilon_b=0.03` 与现有 SSP 插值。
- 同时使用 Pop II 与 Pop III 的 1500 Å 光度；按匹配静止系 UV 的 AB 换算，
  `Mlim=30.6-DM(z)+2.5*log10(1+z)`。不做真实滤镜积分，取无尘、无透镜。
- 灰色：总 UV 光度通过成像阈值的已知历史。没有 ±0.25 mag 窗口或年龄切选。
- 蓝色：灰色母样本中，Pop III Case-B 线光度再满足 `L1640>=4πDL²Flim` 的对象。
- 两组圆点和竖线分别为晕丰度加权的中位数与 16–84% 对象分布，不是误差或置信区间。
  下箭头表示区间超出纵轴，下三角表示中位数也低于显示范围；原始值不修改。
- 正文叠加的观测按原文透镜/尘埃口径转换，只作为文献参考。尤其不能据其处于
  无透镜参考线限下方判定它“不应被观测”，不能将其直接视为同一选样样本。

谱线未知的左删失历史没有当成零谱线；其已存 UV 入选权重单独报告。过线比例以
已知历史的 UV 入选样本为分母。JSON 另给出把所选未知历史全判为检出/未检出的
比例界限；此界限不覆盖未知早期爆发可能缺失的 UV 光度。

## 结果

| z | MUV 极限 | 线光度限 / erg s⁻¹ | 已知母样本过线比例 | 过线对象数 | 过线样本有效质量簇 | UV 入选未知权重 |
|---|---:|---:|---:|---:|---:|---:|
| 6.639 | −16.30 | 1.53e41 | 6.31% | 19 | 3.77 | 5.23% |
| 8.1623 | −16.62 | 2.46e41 | 13.82% | 30 | 5.16 | 4.43% |
| 10.600 | −17.01 | 4.47e41 | 8.26% | 52 | 6.30 | 1.88% |
| 12.342 | −17.23 | 6.30e41 | 6.36% | 558 | 59.78 | 0.36% |
| 13.860 | −17.40 | 8.17e41 | 5.06% | 685 | 68.62 | 0.15% |

低红移仍受有限有效质量簇数影响，不据此拟合效率或给出精确的巡天检出率。
这里只施加参考硬阈值，没有模拟噪声散射、完备度、实际曝光与孔径响应。

## 复现与产物

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/heii_survey_depth.py
PYTHONPATH=. .venv/bin/python scripts/plot/plot_heii_survey_depth.py
```

纯后处理，无需新生成历史或重新提交生产任务。低红移读取
`data_save/threshold_zero_all_20260918/heii_v24_targets_20260914/z*/z*.npz`；
高红移读取同批 `heii_uniform_band_20260917/z12.342/` 与 `z13.86/`，
二者均明确使用 1500 Å Pop II 光度，不使用此前 1600 Å 的 exact 目标样本。
校验父 manifest、样本 SHA-256、关键 SSP/重建代码、红移和形成参数。

- 统计与父产物哈希：`data_save/heii_survey_depth_20260918/{summary,manifest}.json`。
- 正文矢量图：`slides/popiii_heii_pisn_complete_20260916/assets/heii_survey_depth.pdf`。
- 图形来源记录：`outputs/heii_survey_depth_20260918/provenance.json`。
- 上一版讲稿 4.2 展示两组选样；当前仅在附录 A.5 保存阈值、加权比例和采样限制。
- 旧目标 UV 窗口诊断保留在附录 A.3/A.4，已标为原选样，不与新图混用。

测试覆盖 AB 换算、阈值方向/边界、不同丰度权重、零谱线、未知历史和空检出集；
绘图显示的量逐项核对 summary，不重新定义分位数或科学参数。
