# 再电离参数扫描与 Thomson 光深

> 2026-09-21 合并说明：新混合模型的默认入口为
> `configs/uvlf/popii_popiii.json` / `run_popii_transition.py`，采用 Pop III 后
> 零有效延迟启动 Pop II、两类源均 100 Myr 供光窗口。下文记录此前独立启动的
> 实验配置和结果，不替代新默认；原始数值、路径及输入哈希保留。
> 当前模型与验证结果见 [Pop II 过渡说明](popii-transition.md)。

本轮沿用 2026-09-18 的真实瞬时源表、密度场及空间求解器，分别校准纯 Pop II
与 Pop II+III 模型的整体逃逸率。混合模型始终满足
`fesc_popii == fesc_popiii`。成星阈值为 `mu=0, sigma=1.5 dex`，效率为
`epsilon_b=0.03`；保留这组配对实验的双 100 Myr 供光时间窗，不替代 Pop III
6 Myr 的生产设置。没有为拟合再电离而改变 UV、He II 或 PISN 参数。

## 延伸结果（z=5.01，当前图与讲稿）

三个完整 GPU 重跑均正常结束，每组 **235 个真实快照**。原有 185 帧的体积
平均及密度加权电离历史逐项一致；低红移新增 50 帧。光子源表已算到 z=5.00，
密度场实际终点为 z=5.01，没有声称存在 z=5.00 的空间快照。末帧三组的两种
电离比例均为 1，光深在更低红移延续完全电离，并保留同一氦二次电离处理。

| 模型 | 整体 fesc | 99% 电离红移 | xHI(5.9) | xHI(5.6) | 总光深 | 相对旧光深变化 |
|---|---:|---:|---:|---:|---:|---:|
| 纯 Pop II | 0.1775 | 6.229 | 0.00000 | 0.00000 | 0.05454635 | +0.00000000 |
| Pop II+III | 0.064 | 5.643 | 0.07577 | 0.00649 | 0.05943637 | +0.00005130 |
| Pop II+III | 0.0725 | 5.873 | 0.01285 | 0.00000 | 0.06229158 | +0.00001887 |

三组均低于 McGreer+15 的 z=5.9、5.6 单侧 1σ 上限（0.11、0.09）；这些点未参与
原来的参数选择。99% 交点仅描述完成时间，不是额外的拟合标准。本轮只延伸
上一轮展示的三组参数，没有重新拟合逃逸率。

[新版再电离图](../outputs/reionization_z5_20260919/reionization_history.pdf)、
[末期放大图](../outputs/reionization_z5_20260919/reionization_lowz.pdf)、
[新版光深图](../outputs/reionization_z5_20260919/optical_depth.pdf)、
[六页讲稿](../slides/popiii_heii_pisn_complete_20260916/reionization_calibration.pdf)。
原始扫描图与旧结果保留用于追溯；讲稿中的扫描页明确标注为上一轮结果。

源表作业 159967–159970，合并作业 159971，空间作业 159973–159975。
z=6 重算源率的最大相对差异为 1.6×10⁻¹¹；原高红移源表完整保留。
质量节点在两个数值环境间有单个 float64 ULP 的表示差异，合并只容许至多
2 ULP 并保留原质量网格，不重新采样物理质量。低红移分析额外直接读取
每组 z=6.01、5.91、5.59、5.01 的三维场，重算均值及密度加权值，与保存历史
相符。光深步长减半差异均小于 10⁻⁹；此数值积分检查不代表模型系统误差。

模型修改后的完整 1227 项测试通过，随后新增质量网格 ULP 检查的 4 项合并
测试也通过；本轮文件 Ruff 检查及格式检查通过。详细统计、输入哈希和
场文件核对见 `data_save/reionization_z5_20260919/summary.json` 与
`source_validation.json`。以下保留初轮数值与假设，不能代替上表的延伸结果。

## 2026-09-19 初轮结果（空间演化止于 z=6.01）

新增两轮、6 组空间计算，连同已有结果比较了 14 组历史。每组均有完整的
185 个快照；新作业 159957、159958、159959、159961、159962、159964 均正常
退出，耗时 12:29–14:06。纯 Pop II 扫描为 0.15、0.175、0.1775、0.2；
混合模型扫描为 0.025、0.05、0.06、0.064、0.0675、0.07、0.0725、0.08、0.1、0.2。

| 模型与选择依据 | 整体 fesc | xHI(z=7) | xHI(z=7.54) | 总光深 | 对应描述性分数 |
|---|---:|---:|---:|---:|---:|
| 纯 Pop II，仅中性氢 | 0.1775 | 0.5331 | 0.7402 | 0.054546 | 0.7978 |
| 纯 Pop II，中性氢与光深 | 0.1775 | 0.5331 | 0.7402 | 0.054546 | 0.8039 |
| 混合模型，仅中性氢 | 0.0725 | 0.5471 | 0.6519 | 0.062273 | 0.2668 |
| 混合模型，中性氢与光深 | 0.064 | 0.6105 | 0.6938 | 0.059385 | 1.1681 |

兼顾两类约束的已算选择为纯 II 的 **0.1775**、混合模型的 **0.064**。
两者在三个窄红移点上均落入所报 68% 区间，光深分别偏离 Planck 中心
0.078σ、0.769σ。这是本次固定模型的候选选择，不意味着逃逸率被测到了
表中小数位的精度，也没有对模型参数给出统计置信区间。

若只按中性氢数据选择，混合模型偏向 0.0725，其光深高于 Planck 中心
1.182σ。混合模型较长的高红移电离尾部提高了光深：折中选择中 `z>10`
的光深贡献为 0.005546，纯 II 为 0.000976。

纯 II 达到 99% 体积电离的红移为 6.229；混合模型 0.064 在最后快照
`z=6.01` 仍有 14.43% 中性体积，模拟尚未覆盖其结束时刻。改变低红移
补全方式给出的混合模型总光深范围为 **0.059136–0.059560**，相对主值
的最大变化为 0.000249。该范围是补全敏感性，不是总误差条。

宽红移高 z 箱仍有差异：在箱中心 z=9.3，纯 II 和混合模型分别给出
0.932、0.840，中性比例高于该箱的 68% 区间上界 0.81。由于未提供箱内选样
权重，这个中心值比较不参与拟合，也不能宣称已经同时解释所有观测。

[再电离图](../outputs/reionization_calibration_20260919/reionization_history.pdf)、
[光深图](../outputs/reionization_calibration_20260919/optical_depth.pdf)、
[扫描图](../outputs/reionization_calibration_20260919/calibration_scan.pdf)与
[当前讲稿](../slides/popiii_heii_pisn_complete_20260916/reionization_calibration.pdf)。

相关 40 项测试通过；全部历史的光深积分步长减半差异小于 10⁻⁸。
实际三维场重算的体积平均与密度加权比例匹配保存历史。
新文件的 Ruff 检查及格式检查通过；仓库整体格式检查仍报告两个本轮未改动的
文件 `analyze_random_q_heii.py`、`analyze_random_q_pisn.py`，保留其原有修改。

## 校准与观测

配置为 `configs/experiments/reionization_calibration_20260919.json`。每个候选值
都对应完整的 185 个空间快照；旧扫描仅在密度哈希、求解器代码、物理设置、
源模型和红移网格一致时复用。插值用于建议新参数，不代替空间重算。

保留两个描述性指标：

\[
S_{\rm HI}=\sum_i [(x_{\rm HI}(z_i)-x_{{\rm HI},i}^{\rm obs})/\sigma_{i,\pm}]^2,
\qquad
S_{\rm HI+\tau}=S_{\rm HI}+[(\tau-0.054)/0.007]^2.
\]

三个窄红移点来自 [Mason et al. 2018](https://arxiv.org/abs/1709.05356)
和 [Davies et al. 2018](https://arxiv.org/abs/1802.06066)，取预测值所在方向的
68% 区间宽度。光深使用 [Planck 2018 VI](https://arxiv.org/abs/1807.06209)
摘要所报的四舍五入值 `0.054 ± 0.007`。这不是原始数据的联合似然，不能把
分数差解释为置信区间或证据比；参与选择的数据也不是独立验证。

[Mason et al. 的 JWST 阻尼翼结果](https://arxiv.org/html/2501.11702v2)
只在宽红移箱中心展示，缺少选样权重时不加入单红移拟合。McGreer et al.
低红移上限超出初轮空间模拟范围，同样不参与初轮拟合。

## 光深定义与补全假设

`auroralf.experiments.reionization_calibration.thomson_depth` 接受降序红移和
**密度加权**氢电离比例，返回累计无量纲光深、`dτ/dz` 和所用电子历史：

\[
\tau(0,z)=c\sigma_T\bar n_{H,0}
\int_0^z \frac{(1+z')^2}{H(z')}x_{e,H}(z')\,dz',
\qquad
Q_M=\frac{\langle(1+\delta_b)x_{\rm HII}\rangle_V}{\langle1+\delta_b\rangle_V}.
\]

重子跟随求解器所用的裁剪后物质密度代理，并归一化到宇宙平均重子密度。
不能把再电离图里的体积平均电离比例直接代入电子数密度。
`X_H=0.75` 与 EoRCaLC 一致，`n_H,0=X_H Ω_b ρ_c,0/m_p`；使用与源表年龄
计算相同的平直 matter+Lambda 背景，`h=0.6766, Ω_m=0.30966, Ω_b=0.04897`。

He I 与 H 同时电离；第二个氦电子采用
`F_HeIII={1+tanh[(3.5-z)/0.5]}/2`，于是
`x_e,H=Q_M [1+f_He+f_He F_HeIII]`，`f_He=(1-X_H)/(4X_H)=1/12`。
中心红移遵循 Planck 的氦处理；宽度 0.5 是本分析显式采用的近似参数。
没有改变空间求解器的复合参数。

空间历史只到 `z=6.01`。主光深结果在最后快照与 `z=5.6` 之间线性补全
`Q_M` 至 1，之后完全电离；另分别计算 `z=5` 完成和最后快照后立即完成。
这些是探索性敏感性选择，不是低红移空间结果或新观测约束。表中给出由这三种
补全造成的范围；它不包含源抽样、氦、密度和空间求解系统误差。
最高模拟红移以前按原求解器的中性初态处理，不额外加入未知前期电离；
再电离光深不包括复合时代的残余自由电子。

## 运行与产物

准备新候选（输出目录必须不存在）：

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/prepare_reionization_calibration.py \
  --population popii_popiii --fesc 0.064
```

准备器只精确缩放已逃逸的源率及其完整协方差，并保存来源哈希和 GPU 运行
脚本；需要通过非 debug 的 SLURM 分配运行该脚本。本地 GPU 分区为 `cpu`，
这里不是超算 cp6 的 UVLF 生产任务。新空间产物位于相邻 SmallScale21cm
项目的 `runs/aurora_maps/reionization_calibration_20260919/`。

所有配置内的空间任务完成后：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/calibrate_reionization.py \
  --plan configs/experiments/reionization_calibration_20260919.json
PYTHONPATH=. .venv/bin/python scripts/plot/plot_reionization_calibration.py \
  --plan configs/experiments/reionization_calibration_20260919.json --slide-assets
```

分析器遇到不完整任务、配对输入变化、缺失快照、无效电离比例或未收敛积分
时会报错。`data_save/reionization_calibration_20260919/` 保存 `summary.json`、
`scores.csv` 及 `histories_tau.npz`。`outputs/reionization_calibration_20260919/`
保存再电离、光深和参数扫描的 PDF/PNG 图及原始作业日志。最终审阅讲稿为
`slides/popiii_heii_pisn_complete_20260916/reionization_calibration.pdf`。
`--slide-assets` 同时更新讲稿所用的矢量图和自动生成数值表。

```bash
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=slides/popiii_heii_pisn_complete_20260916 \
  slides/popiii_heii_pisn_complete_20260916/reionization_calibration.tex
```

从仓库根目录运行两遍，随后检查编译日志并逐页目视检查 PDF。

验证包括纯氢瞬时再电离的解析积分、积分步长减半、补全敏感性、密度加权、
氦电子数和非法输入拒绝。当前模型仍受密度宇宙学来源未核实、区域重叠不
严格保证全局光子守恒等限制；校准结果只能在这套固定模型内解释。

## 延伸到 z≈5：原因与复现流程

原源表最低红移为 `6.0`，空间入口只采用源表覆盖范围内的密度快照，故原结果
止于 `z=6.01`。这不是再电离的物理停止条件。密度数据最低为 `5.01`，没有
`5.00` 的真实快照。本次独立补算源表至 `5.00`，空间演化使用真实末帧 `5.01`。

固定上一轮展示的三个参数：纯 II `0.1775`、混合 `0.064` 和 `0.0725`；不把
延伸后的三点比较描述为重新扫描全参数空间。新的源表配置为
`configs/experiments/reionization_z5_sources_20260919.toml`，含 52 个红移和 111
个质量节点，其中 `z=6.0` 独立重算用于交叉检查。四个源表分片通过已有的
`build_ionizing_rates.py --shard-count 4 --shard-index N` 计算并合并。

延伸暴露了 `threshold_averaged_rate` 的浮点抵消问题：当整个出生区间都在
年龄窗口外时，`r_left + (r_right-r_left)` 可能比右端点小一个浮点单位，
使本应为空的阈值积分仍去求值旧 SSP。修复增加真实出生时间区间与年龄窗口
的交集检查，保持 100 Myr 窗口和首次穿越规则；新增回归测试确保窗口外
旧事件贡献严格为零，且不会重新触发成星。失败与取消的源表尝试保留在
`outputs/reionization_z5_20260919/` 的日志和尝试记录中。

```bash
PYTHONPATH=. srun -p cpu -N1 -n1 -c1 .venv/bin/python scripts/run/build_ionizing_rates.py \
  --config configs/experiments/reionization_z5_sources_20260919.toml \
  --shard-count 4 --merge-shards
PYTHONPATH=. .venv/bin/python scripts/analysis/extend_ionizing_rates_lowz.py \
  --base data_save/threshold_zero_all_20260918/ionizing \
  --low data_save/reionization_z5_20260919/lowz_sources_parallel \
  --output data_save/reionization_z5_20260919/combined_sources
PYTHONPATH=. .venv/bin/python scripts/submit/prepare_reionization_calibration.py \
  --population popii --fesc 0.1775 --tag reionization_z5_20260919 \
  --source-base data_save/reionization_z5_20260919/combined_sources
```

准备器新增可选 `--source-base` 与 `--tag`，默认值保持原扫描接口。混合模型
同样准备 `0.064` 和 `0.0725`。在非 debug SLURM GPU 分配中执行所生成脚本。
空间求解器的多个区域电离状态仅保存在运行内存中，因此必须从高红移重新
演化，不能只拿旧末帧的平均电离比例续算。

全部完成后运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_reionization_extension.py \
  --plan configs/experiments/reionization_z5_20260919.json
PYTHONPATH=. .venv/bin/python scripts/plot/plot_reionization_extension.py \
  --plan configs/experiments/reionization_z5_20260919.json --slide-assets
```

分析器核对高红移重叠历史、真实密度哈希、原模型参数和全部快照。只有末帧
已经完全电离时，才允许光深计算在更低红移使用 `Q_M=1`。因此 `z=5.01–6.01`
将使用计算结果，取代原先的线性补全假设。光深仍包含显式氦近似，并不消除
密度来源与空间光子守恒限制。旧 14 点扫描作为参数选择的历史依据保留；
图中 [McGreer+15](https://arxiv.org/abs/1411.5375) 低红移上限现在可直接检验，
不擅自把单侧上限改作高斯测量参与拟合。

## 同电离度的空间信号

已用延伸后的三套模型完成 25%、50%、75%、90% 阶段的真实电离/21 cm 切片
及完整三维功率比较，详见[同电离度的电离场和 21 cm 信号](matched-reionization.md)。

## 固定 Pop III 逃逸比例的独立试验

用户在 2026-09-20 明确要求固定 `fesc_popiii=0.5` 再调整 Pop II，因此本试验
显式解除此前混合模型的共同逃逸比例条件；原共同逃逸比例结果保留。
配置为 `configs/experiments/popiii_fesc05_20260920.json`。先计算 `fesc_popii=0`
的非负边界及 `0.05` 响应点，其他恒星、密度和复合参数保持不变。

准备器的可选参数 `--fesc-popiii` 指定独立 Pop III 逃逸比例；此时原有
`--fesc` 指定 Pop II。省略新参数仍精确保持共同逃逸比例接口。

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/prepare_reionization_calibration.py \
  --population popii_popiii --fesc 0 --fesc-popiii 0.5 \
  --tag popiii_fesc05_20260920 \
  --source-base data_save/reionization_z5_20260919/combined_sources
```

源表工具 `rescale_popiii_escape.py` 同样支持 `--fesc-popiii 0.5 --fesc-popii 0`；
不能将独立 Pop II 参数与 `--fesc-common` 混用。两个通道分别线性缩放，完整
协方差按两个对应比例的乘积缩放；Pop II 允许为零。Pop III 的未解析上界
附加通道及积分误差随 Pop III 一起缩放。空间计算仍选择 resolved 混合通道，
没有额外打开该附加上界源项。

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_independent_escape.py \
  --plan configs/experiments/popiii_fesc05_20260920.json
PYTHONPATH=. .venv/bin/python scripts/plot/plot_independent_escape.py \
  --plan configs/experiments/popiii_fesc05_20260920.json
```

独立试验产物位于 `data_save/popiii_fesc05_20260920/` 和
`outputs/popiii_fesc05_20260920/`；不会覆盖原共同逃逸比例的参数选择。

本轮结论、三页讲稿及数值验证见[固定 Pop III 逃逸比例为 0.5](popiii-fixed-escape.md)。
