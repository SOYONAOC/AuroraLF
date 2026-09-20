# 同电离度的电离场和 21 cm 信号

> 2026-09-21 合并说明：新混合模型的默认入口为
> `configs/uvlf/popii_popiii.json` / `run_popii_transition.py`，采用 Pop III 后
> 零有效延迟启动 Pop II、两类源均 100 Myr 供光窗口。下文记录此前独立启动的
> 实验配置和结果，不替代新默认；原始数值、路径及输入哈希保留。
> 当前模型与验证结果见 [Pop II 过渡说明](popii-transition.md)。

使用已完成至 z=5.01 的三套校准模型：纯 Pop II 的 fesc=0.1775，以及混合
Pop II+III 的共享 fesc=0.064、0.0725。这里比较分别校准后的模型，不是只开关
Pop III、其余所有参数和红移完全固定的干预实验。

## 结果

在体积平均电离度约 0.5 时，最近的三个真实快照分别为：

| 模型 | fesc | 红移 | 实际 Q_V | 完全电离体积占比 |
|---|---:|---:|---:|---:|
| Pop II | 0.1775 | 6.95 | 0.502127 | 0.409882 |
| Pop II+III | 0.064 | 6.57 | 0.502736 | 0.290182 |
| Pop II+III | 0.0725 | 6.83 | 0.502559 | 0.281719 |

混合模型在这个阶段的完全电离体积较少，部分电离体积较多；此结论来自完整
三维场的格点统计。切片仅展示形态，没有将其解释为三维气泡半径或连通性测量。

下表为最接近展示尺度 0.1 cMpc^-1 的真实谱箱（中心 **0.101364 cMpc^-1**），
混合模型相对纯 Pop II 的功率比。数值是相邻快照间对 Q_V 线性插值得到的
同阶段统计量，不是人为缩放场得到的目标电离度。

| Q_V | fesc=0.064：电离场功率比 | 21 cm 功率比 | fesc=0.0725：电离场功率比 | 21 cm 功率比 |
|---|---:|---:|---:|---:|
| 0.25 | 0.213 | 0.047 | 0.209 | 0.047 |
| 0.50 | 0.526 | 0.456 | 0.484 | 0.437 |
| 0.75 | 0.744 | 0.617 | 0.726 | 0.630 |
| 0.90 | 0.849 | 0.688 | 0.839 | 0.712 |

Q_V=0.5 的阶段插值平均亮温为 9.36、8.89、9.11 mK，均值相近而空间功率差别
明显。分别去除每个快照的 T0(z)^2 后，两条混合模型在该谱箱的功率比仍为
0.478、0.443；这只能排除显式亮温红移系数作为差异的主要解释，并没有去除
密度演化或密度与电离场的相关性，也不能单独归因于 Pop III。

[十页讲稿](../slides/popiii_heii_pisn_complete_20260916/matched_ionization_21cm.pdf)、
[50% 电离时的两种场](../outputs/matched_reionization_20260919/fields_q0.50.pdf)、
[电离场功率](../outputs/matched_reionization_20260919/ionization_power.pdf)、
[21 cm 功率](../outputs/matched_reionization_20260919/brightness_power.pdf)、
[亮温功率比](../outputs/matched_reionization_20260919/brightness_ratio.pdf)。
25%、75%、90% 的对应切片同样保存在上述输出目录和讲稿中。

## 阶段匹配与物理量

25%、50%、75%、90% 是显示阶段，不是拟合约束或物理阈值。切片选用首次
跨过目标电离度的两个邻接输出中最近的真实快照，标出红移和实际 Q_V。
最大偏差分别为 0.005652、0.002736、0.008510、0.002872。没有对空间场插值、
重新归一化、二值化或强迫其平均值达到目标。每个模型需读取 8 个真实快照。

切片固定为 x=150.5 cMpc；300 cMpc 周期盒子、300³ 网格，同一密度实现。
每阶段同一种物理量使用跨模型统一的线性色标；亮温色标上限为该阶段所展示
三张切片的真实最大值，不截断高亮像素。不同阶段的功率图统一纵轴。

亮温沿用 `smallscale21cm.brightness.saturated_brightness_mk`：

\[
\delta T_b(\mathbf r,z)=T_0(z)[1-x_{\rm HII}(\mathbf r,z)][1+\delta_b(\mathbf r,z)],
\quad
T_0(z)=27\,\mathrm{mK}\,\frac{\Omega_bh^2}{0.023}
\left[\frac{0.15}{\Omega_mh^2}\frac{1+z}{10}\right]^{1/2}.
\]

采用高自旋温度、实空间近似，没有速度畸变、光锥、仪器波束、前景或热噪声。
密度使用空间模型原有的 [-0.95,6] 裁剪及暗物质追踪重子的约定。

三维 FFT 扣除体积均值，使用绝对起伏而非相对均值的起伏：P21 单位为
mK² cMpc³，Δ²21=k³ P21/(2π²) 单位 mK²。电离场功率的 Δ² 无量纲；
k>0 时中性比例与电离比例的功率相等。保留真实 FFT 共轭模计数。
波数箱延续原分析的 24 个对数箱，从 2π/L 至一维 Nyquist π/Δx；没有声称
高波数结果已通过空间分辨率收敛验证。

功率及平均亮温按邻接快照的 Q_V 线性插值；阴影显示两侧快照的范围。功率比
范围保守组合两模型的端点极值。这些范围不是统计置信区间、插值误差界或
模型系统误差。低波数箱模数较少，不能从单盒子起伏推断显著性或可探测性。
既有密度宇宙学来源未核实及重叠区域不严格守恒全局光子数的限制继续适用。

## 复现及验证

配置：`configs/experiments/matched_reionization_20260919.json`。
后处理使用 SmallScale21cm 自身环境，在非 debug 的 SLURM CPU 分配运行：

```bash
cd ../SmallScale21cm
packages/EoRCaLC/.venv/bin/python scripts/analysis/match_aurora_ionization.py \
  --run <completed-spatial-run> --output <new-output-directory> \
  --targets 0.25 0.5 0.75 0.9
```

`--targets` 为新增可选参数，默认仍为 0.25、0.5、0.75；拒绝非法或重复目标。
新增端点产物含直接保存的 `xhii_slice`、`brightness_prefactor_mk`、
`delta2_brightness_over_t0`、谱箱边界及完整场方差/完全电离体积。既有字段保留。

三模型的后处理作业 **160023** 在 node2 完成，耗时 77 秒；共 24 个真实三维
快照、48 次完整三维功率计算。阶段匹配、亮温边界、Parseval 恒等式及奇偶
网格共轭计数相关的 16 项测试通过；另有 48 项真实场功率与全场方差检查通过。
所有读取的密度、场、配置、代码和派生产物有哈希记录；切片来源均值与历史一致。

回到 AuroraLF 根目录绘图：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_calibrated_matched_fields.py \
  --plan configs/experiments/matched_reionization_20260919.json --slide-assets
```

`data_save/matched_reionization_20260919/` 保存各模型端点切片及功率、按阶段
插值的功率、`comparison.json`、`stages.csv` 和 `power_validation.json`。
`outputs/matched_reionization_20260919/` 保存 PDF/PNG 图及作业日志。
这些都是完整三维场的后处理，没有重新运行或改写再电离模拟。
