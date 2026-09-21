# z≈6 晕首次越过原子冷却阈值

2026-09-17。下列数值是对本地真实模拟树的重新统计，不是文献直接报告的分布。
统计主祖先分支，冷却定义为 Tvir=10^4 K、mu=0.61，沿用项目完整
Barkana–Loeb/Bryan–Norman 质量阈值。z=5.9935 的阈值为 1.5874e8 Msun。
它不是首次成星、随机爆发阈值或金属污染时间。

## THESAN-HR-large：按质量分箱

使用 `external_data/thesan/thesan-hr-large/postprocessing/trees/LHaloTree/trees_sf1_091.0.hdf5`，
全部 98443 个 Tree 组，目标快照 70、z=5.9935036。目标为中心晕，
Group_M_TopHat200 >= Mcool；主分支也只在中心节点使用该原生维里质量。
采用 h=0.6774、Om=0.3089、Ob=0.0486。单位转换为 1e10/h Msun。

| z≈6 的 Mvir/Msun | 样本数 | 有跨越区间的数量 | 全样本中位越过红移的下–上界 |
|---|---:|---:|---:|
| 1.59e8–1e9 | 5838 | 4915 | 8.27–9.15 |
| 1e9–1e10 | 722 | 593 | 12.68–13.98 |
| 1e10–1e11 | 54 | 38 | 15.62–17.82 |
| 全部达到冷却阈值 | 6614 | 5546 | 8.62–9.58 |
| >=1e9 | 776 | 631 | 12.96–14.36 |

全部所选晕中，在 z>=10 已越过的比例为 32.08%–46.52%，所以至少 53.48%
在 z<10 才越过。>=1e9 Msun 子样本对应比例为 88.53%–94.46%。
这些比例及中位数只描述现有树覆盖的样本，未校正缺失目录对象或宇宙方差。

每条主分支按宇宙时间排序，以第一次高于阈值的快照红移为下界。
如果全部更早节点均有有效中心晕质量，并存在之前的低于阈值节点，前一快照给出红移上界。
若首次树节点已达标，或更早存在卫星/非正质量节点，上界未知（数学上保留为无穷大）。
分别对逐晕下、上界取经验分位数，得到全样本分位数的界限；这不是统计置信区间。
任何后续跌回阈值以下再上升都不能抹除第一次越过。

5546 个有跨越区间的对象，快照时间间隔中位数 10.90 Myr，16–84% 为 9.69–12.13 Myr。
另有 12 个首次出现已达标、1056 个更早历史不完整的对象。
因此整体第 84 百分位的上界不能确定，不能给全样本虚构一个有限中央 68% 区间。
仅在有跨越区间的 5546 个对象上，log(M/Mcool) 对宇宙时间插值得到的
16/50/84 分位为 6.76/8.85/11.97；这只是条件子样本，不能替代全样本结果。

## 数据检查与适用范围

- 树文件目标快照包含 104798 个子晕，Header/TotNsubhalos 为 120418，覆盖 87.03%。
  文件头声明 NumberOfOutputFiles=1；已经扫描所有声明的树组。
  缺失对象的质量未知，该比例不是所选中心晕样本的完备度，不能用统一权重纠正。
- 原始 SubhaloMassType/SubhaloLenType 与标量数组明显不一致。例如 Tree0/node21，
  标量 SubhaloLen=134621，分类型求和为 249；Tree0 中只有约 1.07% 的节点两者一致。
  最初使用分类型质量的作业 159237 报错，结果未保存或使用。本次不修补原始数组，
  改为独立的原生维里质量定义，并且将卫星与缺失质量节点保留为未知。
- 完整目录校验作业 159242 因上述 104798/120418 差异失败，未输出成功产物。
  随后明确收窄统计解释为“树覆盖样本”；作业 159244 在 fat2 使用 8 核，32 秒完成。
  作业 159239 排队期间节点被占用，已取消并重新按实时空闲资源提交。
- 盒长 11.8 cMpc，DM 粒子质量 4.8229e5 Msun。本次没有 >=1e11 Msun 的中心晕。
  维里质量约 1e8 Msun 的分辨率仍非无限；本次没有声称数值收敛。
- 从原始树随机抽取 16 条主分支，独立反算维里温度、检查首次 Tvir>=1e4 K 的
  快照，全部与质量阈值法一致，详见 `audit.json`。这不替代与原始 group catalogs 的字段比对。

## TNG 与其他模拟

重新汇总 `data_save/tng_pristine_20260914/halo_diagnostics.csv`，校验已有 SHA256。
TNG100-1-Dark 目标 z=6.01076，M200c=1e9–1e13 Msun，共 8956 个分层抽样对象。
恢复 available_count/selected_count 数量权重；祖先质量沿用已有诊断的 SubhaloMass。
主分支 >=20 DM 粒子的最早达标记录，加权中位 z≈10；>=100 粒子时为 z≈8。
未找到更早合格祖先的比例分别为 0.132% 和 2.753%，分位数仅对找到者计算。
TNG 的粒子质量约 8.86e6 Msun，z=10 的冷却阈值仅约 9 个粒子，最早达标记录
不能当成真实首次越过。全部祖先的诊断也保存在 summary.json 中；THESAN 主结果仅是主分支。

两套模拟的质量定义、体积、抽样及气体物理不同，不能将差异全部归因于分辨率。
本地 THESAN-Dark-1 树只有 130/192 个分块，本次没有处理其全分布。
Renaissance 最晚的 Void 区域止于 z≈8，不能直接回答 z=6 的后代样本问题。
McBride 参数化属于解析模型；下文新增与模拟终点匹配的比较，模型输出仍与独立模拟数据分开。

来源：

- [Barkana & Loeb (2001)](https://arxiv.org/abs/astro-ph/0010468)：维里温度–质量关系。
- [Garaldi et al. (2024)](https://arxiv.org/abs/2309.06475)：THESAN 数据、分辨率与体积。
- [Borrow et al. (2023)](https://arxiv.org/abs/2212.03255)：THESAN-HR。
- [THESAN 合并树字段文档](https://www.thesan-project.com/thesan/mtrees.html)：主祖先、中心与组质量语义。
- [TNG100-Dark 官方参数](https://www.tng-project.org/data/downloads/TNG100-1-Dark/)。
- [Renaissance 数据范围](https://www.firstgalaxies.physics.gatech.edu/explore-the-data/)。
- [McBride et al. (2009)](https://arxiv.org/abs/0902.3659)：Millennium MAH 参数化。

## 复现与交付

统计脚本 `scripts/analysis/atomic_crossing_z6.py` 要求 SLURM，拒绝覆盖完成的 manifest。
重做时应为新的实验修改输出目录，而不是删除旧证据。
产物 `data_save/atomic_crossing_z6_20260917/` 包含逐晕 CSV、输入元信息、分位汇总和独立复核。
绘图与汇总使用：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/summarize_atomic_crossing_z6.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_atomic_crossing_z6.py -q
```

5 项针对性测试通过；修改的分析文件通过 Ruff 检查和格式检查。
Slides：`slides/atomic_crossing_z6/crossing.pdf`；源文件同目录。
未修改生产模型或科学参数。

六页 PDF 经 XeLaTeX 两遍编译、日志和逐页视觉检查；已通过同步技能上传到
AuroraLF/Slides，标题 AUR-S03 · z=6 暗物质晕的原子冷却越过时间，
Zotero item `5CZJJ46V`，文件 `halo-atomic-cooling-crossing.pdf`，深度校验通过。


## 与当前吸积模型逐个匹配的比较

同日第二阶段：用户要求对照当前吸积模型。使用未经修改的
`sample_parameters(mass_ref=Mh_final, sampler="mcbride")` 与生产 MAH 公式：

M(z) = Mf [(1+z)/(1+zf)]^beta exp[-gamma(z-zf)]。

逐个匹配上述 6614 个 THESAN 的 **实际 Mvir**，zf=5.9935036；每个质量抽样
128 条历史，共 846592 条。每个真实目标赋予相同总权重；未采用质量箱中心替代
质量分布，未重新用 HMF 加权。一个批次，种子 9172601。
保留项目 z_start=50；只判断 q=1 的原子冷却阈值，不运行随机 Pop III 阈值、
SFR、反馈或富集。主结果匹配 THESAN 的 Planck15 宇宙学；以相同 beta/gamma
另算生产默认 Cosmology() 的 Planck18 敏感性。

| 最终 Mvir/Msun | 模拟中位红移界限 | 当前模型中位红移 | 模型中位宇宙时刻提前/Myr |
|---|---:|---:|---:|
| 1.59e8–1e9 | 8.27–9.15 | 15.71 | 282–360 |
| 1e9–1e10 | 12.68–13.98 | 20.27 | 122–166 |
| 1e10–1e11 | 15.62–17.82 | 21.14 | 46–89 |
| 全部样本 | 8.62–9.58 | 16.49 | 266–343 |

时间差是两者中位越过宇宙时刻之差，不是逐个晕配对时间差的中位数。
固定目标样本后，全模型中位数的抽样误差区间为 16.459–16.532（近似 95%）；
此误差不含有限模拟目标样本、缺失目录对象、宇宙方差、质量定义或物理模型系统误差。
生产默认宇宙学下各分箱中位数的变化均小于 0.001。

在 z>=10 已越过的比例：当前模型 76.13%，模拟允许 32.08%–46.52%。
模型整体有 19.56% 的历史在 z=50 已高于阈值；三个质量箱分别为
19.96%、16.89%、12.83%。这些历史保留在分母，以红移 >=50 的删失记录保存，
不外推具体越过时刻；全模型第 84 百分位只有 >=50 下界。
这部分比例不能直接归因于 gamma=0 的 4.66% 混合成分，它还包含联合分布抽出的轨迹。

原论文 Appendix A 的分布来自 z=0、约 M>=1.2e12 Msun 的 Millennium FOF 晕；
论文 §3.3 讨论的是高红移条件下平均历史的关系，未直接验证本项目把低质量 Mf
代入该分布并重设 zf 的完整随机轨迹分布。因此，当前用法属于高红移、低质量外推。
本次比较揭示的偏早说明这一参数分布尚不能用于校准这批模拟晕的冷却起始时间。
不能把公式形式可用等同于其全部参数分布已经在目标质量/红移上得到校准。
FOF 与维里质量定义不同、THESAN 部分样本缺失和小体积限制依然存在，
所以没有把全部偏差唯一归因于某一个代码步骤，也没有修改生产模型。

### 新增产物与验证

- 脚本：`scripts/analysis/compare_atomic_crossing_mcbride.py`；SLURM 作业 159260，fat2，8 核，21 秒完成。
- 原始模型样本：`data_save/atomic_crossing_mcbride_20260917/histories.npz`；包括 beta、gamma、两种宇宙学的越过红移、删失状态和实际最终质量。
- 同目录 manifest.json 保存生产源文件、输入、输出哈希；summary.json 保存对比、尾部比例和误差估计。
- 汇总重绘：`PYTHONPATH=. .venv/bin/python scripts/analysis/summarize_mcbride_crossing.py`。
- 验证：`PYTHONPATH=. .venv/bin/python -m pytest tests/test_mcbride_crossing_comparison.py -q`，3 项参数化测试通过。
  根查找结果与公开 generate_halo_histories 在三组质量、相同随机参数下逐条比较；
  257 点与 2049 点扫描所得根一致，且与公开 API 的 dz=0.025 跨越区间一致。
- 新脚本及测试通过 Ruff 检查和格式检查。
- 在原 AUR-S03 slides 前新增三页模型比较，原模拟统计和限制保留。

九页更新版已完成编译和新增/修改页视觉检查，并更新同一 Zotero item `5CZJJ46V`（AUR-S03）；深度校验通过。生产 MAH 源文件未改动。
