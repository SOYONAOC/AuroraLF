# TNG 已解析祖先污染与 Pop III 候选比例

2026-09-14，作业 158025 在 node6 用 31 核完成，SLURM 耗时 23 秒。
没有修改生产 Pop III、He II 或 UVLF 模型，也没有新增模拟下载。

## 输入与统计对象

使用本地 TNG100-1-Dark 完整 SubLink 树，原始目录为
`external_data/tng/TNG100-1-Dark/raw_sublink_full/`。这是纯暗物质数据；
本实验不含模拟直接测出的气体金属丰度。

目标清单为
`data_save/tng_mah_cache/selection/TNG100-1-Dark_logM9p00_13p00_dlogM0p25_n1000_seed42/selected_subhalos_manifest.csv`，
共 24,549 条不同目标快照/中心晕记录。同一物理后代可能出现在不同快照，
因此各红移结果不是相互独立的宇宙体积实现。

实际红移为 6.0108、8.0122、9.9966、11.9802；没有把后者标成 12.5。
目标质量为 M200c=10^9–10^13 Msun，并要求当前满足原子冷却条件。
质量箱原本以 0.25 dex 分层随机抽样，每箱最多 1000 个；使用
`available_count/selected_count` 恢复原模拟中心晕数量权重。
不可直接使用未加权样本比例，也不把这一结果解释为宇宙全部小晕的比例。

## 简化判据

从目标节点沿 FirstProgenitorID/NextProgenitorID 追溯全部祖先；另算只追踪
FirstProgenitorID 的对照。若某个更早节点满足：

1. 物理束缚子晕质量 SubhaloMass >= 原子冷却质量 Mcool(z)；
2. 暗物质粒子数 >= 20 或 100；
3. 目标时刻距该节点时刻 >= 10、30 或 100 Myr；

则将目标标记为在本次简化模型中被祖先污染、不能形成全新 Pop III 爆发。
SubhaloMass 是祖先引力质量代理，不能把共同宿主的 Group_M_Crit200 分配给每个卫星。
目标分箱仍沿用原抽样使用的 Group_M_Crit200。质量转换用 TNG 的 h=0.6774；
时刻和冷却阈值使用 H0=67.74、Om=0.3089、Ob=0.0486 的平直宇宙学。

延迟覆盖冷却、成星、金属释放和混合，为试验输入，未经拟合。
本模型假设达到条件的祖先能够有效成星并充分污染后代成星气体；
没有计算金属产额、流出与保留、外部污染、局部原初气体、分子冷却成星。
也未对每棵树运行生产随机成星阈值。

祖先污染概率的思路参考
[Trenti & Stiavelli09 §2.5–2.6](https://arxiv.org/html/0901.0711v1#S2.SS5)，
但本文上述规则是本项目的试验，不是该论文概率公式的复现。

## 结果与限制

30 Myr 延迟下，仍未被已解析祖先排除的比例：

| z | 祖先 >=20 粒子 | 祖先 >=100 粒子 |
| --- | --- | --- |
| 6.01 | 0.132% | 2.527% |
| 8.01 | 0.266% | 1.991% |
| 10.00 | 0.169% | 13.153% |
| 11.98 | 13.520% | 82.328% |

这些是**未排除的候选，不是确认原初的晕**。在所选充分混合、有效冷却规则下，
忽略未解析祖先和外部污染会留下乐观的候选比例；对允许局部原初气体存活的更广泛模型，
不能把它当作无条件上限。

原子冷却阈值相当于约 3–18 个 TNG 暗物质粒子，而粒子质量约 8.86e6 Msun。
20/100 粒子试验是在同一模拟中改变分析筛选，不是实际分辨率收敛试验。
两者差异说明结论依赖低粒子数祖先，尚不能当作校准完成的 pristine 概率。
快照间隔约 30–100 Myr，使本次 10 和 30 Myr 曲线完全重合。
树首次解析时已高于冷却质量的节点存在左删失，不能赋予虚构的早期成星时刻。

全部祖先比主分支额外排除的比例，在本次扫描中不足 0.5 个百分点。
这衡量的是布尔资格判据变化，不能解释为旁支带来的金属质量占比。

## 产物与复现

- 逐晕时钟、树输入字段哈希：`data_save/tng_pristine_20260914/halo_diagnostics.csv`
- 180 条分组统计与分层抽样 MC SE：同目录 `summary.json`
- 假设、输入清单哈希、红移与冷却质量网格：同目录 `manifest.json`
- 独立正向 DescendantID 复核：同目录 `audit.json`
- Slides：`slides/tng_pristine_20260914/pristine.pdf`
- 会话图：`outputs/tng_pristine_20260914/fraction_redshift.png`

2026-09-15：Slides 扩充至 8 页，在结果前加入目标样本与祖先追踪、
污染条件与 50 Myr 示例、抽样权重与候选比例公式三页；末页说明充分混合假设
及与生产爆发模型的区别。统计结果不变，已重新编译并逐页检查。

测试：

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_tng_pristine_fraction.py -q
```

计算脚本为 `scripts/analysis/tng_pristine_fraction.py`，只能在 SLURM allocation 中运行，
并拒绝覆盖已有实验 manifest；复现时应显式选择新的实验输出目录。
保存结果的复核与重绘：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/audit_tng_pristine_fraction.py
PYTHONPATH=. .venv/bin/python scripts/plot/plot_tng_pristine_fraction.py
```

审查采用 16 棵原始树的正向后代链接，与主计算反向祖先遍历独立核对；
并复核全部逐晕包含关系及全部 180 条统计权重。
MC SE 不含宇宙方差、物理模型误差，抽样边界的零标准误不代表真实概率已精确确定。
