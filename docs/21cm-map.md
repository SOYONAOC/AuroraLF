# 21 cm 空间预测：瞬时源结果与接入记录

> 2026-09-21 合并说明：新混合模型的默认入口为
> `configs/uvlf/popii_popiii.json` / `run_popii_transition.py`，采用 Pop III 后
> 零有效延迟启动 Pop II、两类源均 100 Myr 供光窗口。下文记录此前独立启动的
> 实验配置和结果，不替代新默认；原始数值、路径及输入哈希保留。
> 当前模型与验证结果见 [Pop II 过渡说明](popii-transition.md)。

## 整体逃逸率：2026-09-18 用户更正

正确比较是纯 Pop II 模型与 Pop II+III 模型分别使用各自的整体逃逸率。
混合模型内部必须 `fesc_popii == fesc_popiii == fesc_common`，共同降低 II+III
源项，不能固定 II 而只降低 III。前述独立 III 扫描及其形态图仅保留为被更正
取代的试验，不能用来回答本次问题，也不应据此声称校准后形态趋近纯 II。

`scripts/analysis/rescale_popiii_escape.py --fesc-common VALUE` 从原双 0.2
源表共同缩放全部通道；同一红移、质量下 III/II 比例保持不变，协方差乘共同
因子的平方。原两类逃逸率不相等时明确报错。纯 II 对照暂沿用其原 fesc=0.2。
整体值 0.025、0.05、0.1 是本轮探索扫描值：均匀模型给出的 z=7 中性比例
约为 0.761、0.521、0.042，仅用于选择扫描范围，不作为空间结果或拟合结论。
用户明确允许不同空闲 GPU 并行重算，各任务保持原 185 快照和全部科学参数。

正确配置为 `configs/experiments/common_escape_scan_20260918.json`；三档任务
159875、159876、159877 分别在 node4、node3、node6 启动。33 项相关测试和
实际源适配器核验通过。`analyze_popiii_escape_scan.py --plan` 根据配置中的
`escape_mode="common_popii_popiii"` 验证两类逃逸率相等，纯 II 作为独立对照，
不再充当混合模型电离度的下界；描述性残差最低值仅在混合模型候选中选择。
产物位于 `data_save/common_escape_scan_20260918/` 及同名 `outputs/` 目录。

`plot_escape_matched_fields.py` 同样识别共同逃逸率模式，要求 `--fesc` 显式
指定三条混合模型曲线（另自动加入纯 II），并可用 `--targets 0.25 0.5`
选择展示阶段。所有指定阶段必须被真实历史覆盖，否则报错；不外推未达到的
电离度。主图仍比较 50% 附近快照，必须包含目标 0.5。

三档粗扫结束后，仅追加一轮 0.06、0.07、0.08：粗扫的实际空间结果在 z=7
给出 `xHI=0.703`（f=0.05）和 `0.237`（f=0.1），夹住所用观测中心 0.59。
细扫任务 159882、159884、159886 全部成功，分别耗时 12:35、12:50、12:24。
最终配置为 `configs/experiments/common_escape_refined_20260918.json`，包含全部
六档新算结果及原双 0.2 对照；各新算档均有 185 个完整三维快照。

已算候选中共同 f=0.07 的窄红移点描述性残差最小；在 z=7、7.54，空间
中性氢比例分别为 0.566、0.664。到最低模拟红移 z=6.01，中性比例仍为
0.0532，尚未达到 99% 电离；不外推到 z=5.6、5.9。此取值是有限扫描的校准
选择，不是联合似然最优解或置信区间。完整曲线、观测残差及校验记录位于
`data_save/common_escape_scan_20260918/refined/summary.json` 和
`outputs/common_escape_scan_20260918/refined/`。三档细扫使用相同求解器代码，
共享源项缩放与完整协方差、密度/配置/源清单哈希均已核验；本次未另做相同
参数的 3090/5080 硬件对照。

## 已被更正取代的独立 Pop III 逃逸率扫描（2026-09-18）

此前误将模型间逃逸率差异理解为混合模型内部两类源的逃逸率差异。
本次固定 `fesc_popii=0.2`，新算 `fesc_popiii=0.005,0.01,0.03,0.06`；
已完成的纯 Pop II 与双 `fesc=0.2` 结果分别提供 0 和 0.2 两个端点。
保持阈值 `mu=0,sigma=1.5`、效率 0.03、双 100 Myr 供光、密度场和复合
处方不变。配置为 `configs/experiments/popiii_escape_scan_20260918.json`。

`scripts/analysis/rescale_popiii_escape.py --source SOURCE --output OUTPUT
--fesc-popiii VALUE` 读取已完成的瞬时源表，按新旧 Pop III 逃逸率之比缩放
已解析及左删失上界源项；标准误线性缩放，完整协方差按两个通道因子乘积
缩放，积分误差同样缩放。Pop II、质量和红移网格不变。原逃逸率必须为
正，新值须在 [0,1] 内；原产品哈希、父清单和派生脚本哈希保留。
这在固定恒星历史和无反馈模型中是源项的精确变换，电离历史仍须重新积分。

作业 159860 在 node3 单 GPU、4 CPU 的一次分配中串行执行四个完整空间
计算，不申请 debug 分区。结束后自动生成独立曲线及逐观测差距：
`data_save/popiii_escape_scan_20260918/summary.json` 和
`outputs/popiii_escape_scan_20260918/escape_scan.pdf`。完成状态以实际清单为准。

比较描述性残差时采用三个窄红移观测点的非对称 68% 区间，不把分数当作
完整似然、置信区间或独立模型验证。宽红移箱仅在图中按中心红移对照，缺少
选样权重时不视为精确单红移测量。纯 Pop II 在 z=7 的中性比例为 0.418，
已低于 Mason18 的中心值；仅降低 Pop III 逃逸率不能保证同时命中全部数据。
氢电离光子的 IGM 逃逸率与 He+ 电离光子逃逸不是同一个参数，本次不改 He II
及星云连续谱。密度场宇宙学来源尚未完全核实的限制继续保留。

扫描已完成，作业 159860 正常退出（51:32），四组各 185 个快照及所有冻结
输入哈希通过检查。各组使用完全相同的空间求解代码、密度场与复合设置；
电离比例有限、在 [0,1] 内，且随 Pop III 逃逸率单调增加。

| fesc,III | xHI(z=7) | xHI(z=7.54) | 99% 电离红移 |
|---:|---:|---:|---:|
| 0 | 0.418 | 0.699 | 6.38 |
| 0.005 | 0.379 | 0.675 | 6.41 |
| 0.01 | 0.340 | 0.652 | 6.45 |
| 0.03 | 0.191 | 0.563 | 6.60 |
| 0.06 | 0.042 | 0.415 | 6.84 |
| 0.2 | 0 | 0 | 8.00 |

三个窄红移点的描述性残差分数在已算的零逃逸端点最小，不能由此给出严格
置信上限或断言实际逃逸率为零。0.005–0.01 是已计算的低逃逸示例，显著减轻
相对于 0.2 的张力；z≈9.3 的宽红移箱倾向更高的逃逸率（0.06 时中性比例
0.792，0.01 时 0.899，观测区间 0.41–0.81）。常数 fesc,III 无法在固定 Pop II
下同时命中所有所示中心值，未为追求吻合额外修改 Pop II、复合或时间窗。
主讲稿采用两页曲线和数值对照，宽箱仅可视化，不纳入上述窄红移分数。

## 指定批次的观测对照图

`scripts/plot/plot_reionization_observations.py` 接受 `--run` 指定完整空间结果。
新增 `--output-dir` 和 `--asset-dir` 分别指定诊断产物和幻灯片资产目录；
省略时保留原有路径。这允许新批次保留自己的图与数值，避免覆盖历史实验。

2026-09-18 的 μ=0 对照保留原双 100 Myr 供光假设，属于历史设置下的配对实验，
不替代下述 Pop III 6 Myr 生产配置。新光子源表含 191 个红移节点、111 个质量节点，
覆盖与旧实验完全相同的 185 个空间快照；输入目录为
`data_save/threshold_zero_all_20260918/ionizing/`。

该配对实验已完成：CPU 作业 11690770 用时 2:06:24，GPU 作业 159856
用时 35:33，均正常退出。三组各 185 个快照完整，电离比例有限且在 [0,1]
内，上界组不低于已解析组；密度输入哈希与旧批次一致，纯 Pop II 历史逐点
完全复现。Pop II+III 达到 99% 电离的红移为 8.002，z=7 和 7.54 的体积平均
中性比例均为 0；阈值中心下移没有消除该双 100 Myr 设置的过早电离。
旧密度场宇宙学来源仍标记为未核验。独立图和数值保存于
`outputs/threshold_zero_all_20260918/reionization/`；主讲稿更新现有结果表。

## 历史独立启动配置：Pop II 100 Myr，Pop III 6 Myr

生产配置与细时间网格配置已分别设置两个年龄上限。主讲稿
[核心公式与思路（5 页）](../slides/21cm_map/21cm_map.pdf)已同步更新；独立的
两页收敛讲稿已清理，以下数值与原始数据继续保留。
新的生产源表输出目录为 `data_save/ionizing_sources/instantaneous_popiii6myr_v1/`，
旧双 100 Myr 产品保留作对照。本次更改设置与讲稿，未重跑完整源表或地图。

## 2026-09-13：再电离历史与观测对照

### 富集门控之前的 z=6、8、10 UVLF

[两页 UVLF 分量讲稿](../slides/21cm_map/uvlf_components.pdf)使用同一套随机首次越阈
爆发模型，epsilon_b=0.03、log10(q)~Normal(0.5,1.5²)、延迟 Pop II 成星、
McBride MAH 和 Reed07 HMF。没有加入富集或红移供光开关。

两类光度统一到 1500 Å。Pop II 是 BPASS 恒星光谱；Pop III 沿用配套
`pop3_ge0_logE_500_001_is5.25` 的 `L_1500`，含表中星云连续谱
（Te=30000 K，LyC 吸收比例为 1）。`.20` 与 `.25` 的 IMF、轨道与
原始 Q0 相同，分别提供电离诊断和 UV；表中的星云条件需明确区分。
未施加尘埃，也未将电离 fesc 直接乘到非电离 UV，未重算星云连续谱的
逃逸率耦合。这是现有 UV 处方的分量诊断，不能当成全部辐射过程已经闭合。

UV 卷积保留两类恒星最近 100 Myr 的贡献；电离光子的 Pop III 6 Myr
收敛结论不自动用于 1500 Å。每个红移抽取 720 个独立均匀 log 质量样本
（10^4–10^15 Msun），每质量 512 条 MAH，960 点时间网格，种子 610228。
三个红移合计 1,105,920 条历史。Total 按同一晕 LII+LIII 重新分箱，
不是两条 LF 相加；分量 LF 也不是按纯 Pop II/III 星系分类。

基础抽样得到的近期 Pop III 明亮样本较稀少，因此在相同 MAH 上按
爆发年龄 0–3–10–30–100 Myr 条件抽取 logq。各层权重是原始正态分布
CDF 的差，并保留 UV 暗态的剩余概率，总概率严格为 1。
每个 MAH 的原始随机 q 光度先逐项复现验证；没有提高物理爆发概率、
把平均光度代替光度分布，或平滑最终 LF。色带为按独立质量抽样聚类的
Monte Carlo 标准误，空心点标记相对标准误超过 30% 的箱。

运行入口（前两步须使用非 debug SLURM CPU 分配）：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/build_current_uvlf.py --workers 8
PYTHONPATH=. .venv/bin/python scripts/analysis/refine_current_uvlf.py --workers 8
PYTHONPATH=. .venv/bin/python scripts/plot/plot_current_uvlf.py
```

基础数据 `data_save/uvlf_current_z6_z8_z10/`，概率分层结果
`data_save/uvlf_current_z6_z8_z10_stratified/`；三个红移各一个 NPZ，
包含样本光度、质量权重、分层概率、phi 与抽样标准误。
预览与摘要 `outputs/21cm_map/uvlf/`，正式矢量资产 `slides/assets/current_uvlf/`。

2026-09-13 完成：SLURM 157709（基础抽样，10:07）和 157710（年龄分层，09:42）
均正常退出。最终 Pop III 的 1500 Å 光度密度占比在 z=6、8、10 分别为
8.8%、24.4%、47.4%；这些不是电离光子占比。MUV<-20 的总数密度相对
Pop II 分量分别增加约 14%、54%、320%。这是未加富集门控的模型结果，
未进行观测拟合。原始 MAH 的随机 q 光度逐项复现、分层概率归一化与
Pop II LF 不变检查均通过；12 项相关回归测试通过。

同日加入 29 个已核对原文表格的观测点：z≈6 的 Bouwens et al. (2021)
表 4，z≈8 的 McLure et al. (2013) 表 2 和 Bowler et al. (2020) 表 6，
z≈10 的 Donnan et al. (2024) 表 2。整理表与源文件 SHA256 保存在
`external_data/observations/uvlf/current_z6_z8_z10.json`；原始 NPZ 保留。
Bowler MUV=-22.90 箱的半宽按原文改为 0.5 mag，Donnan MUV=-20.75
点的下误差按原表恢复为 4e-6；它仍是测量点，不转成上限。下误差达到零的
误差棒在对数图下边界截断，未人为缩短误差。

图聚焦 -24<MUV<-16，保留所有选用观测点。Bouwens 为 1600 Å，其他三篇
为 1500 Å；原文宇宙学均为 H0=70、Om0=0.3。未作颜色、宇宙学或尘埃换算，
没有拟合观测或改模型参数。当前本征模型与实际可见 UVLF 的亮端差异包含
未处理的尘埃问题；特别是 z≈8 的 Pop II 本身已高于亮端观测，不能只归因于 Pop III。

**当前讲稿已切换为开启尘埃的对照图。** 运行
`PYTHONPATH=. .venv/bin/python scripts/plot/plot_current_uvlf_dust.py`，直接读取相同
已完成的本征 LF，调用 `compute_dust_attenuated_uvlf`，使用默认
`c0=2.10, c1=4.85, m0=-19.5` 和生产裁剪
`phi_obs=min(phi_obs_raw,phi_nodust_obs)`；未重算或调整 MAH、SFR、SSP、Pop III
形成概率及电离光子源。三条本征 LF 各自映射；分量曲线是沿用既有 LF 级
经验处方的诊断，不能解释为逐晕共享尘埃屏，也不是经过 Pop III 校准的衰减律。
沿用 A1600≈A1500 的近似；Koprowski+18 的原始校准红移为 3–5，在这里外推。

最终图 `outputs/21cm_map/uvlf/uvlf_components_dust.png` 中灰色点线是本征 Total，
其余三条是尘埃映射结果。只展示模型中心曲线，不将原本征 LF 的误差带冒充
尘埃模型不确定性。所有观测点、波段和宇宙学口径保持前述定义。
模型在原观测星等箱内平均后再作比值；29 个箱的积分网格从 513 增至 1025 点，
结果相差不到 0.1%，所有查询均在已计算的本征 LF 范围内，未外推。
数据及出处哈希记录在 `outputs/21cm_map/uvlf/dust_summary.json`。

每个红移取最接近 MUV=-21 的观测箱：

| z | MUV_obs | A_UV [mag] | Total/观测，无尘埃→有尘埃 | Pop II/观测，有尘埃 |
|---|---|---|---|---|
| 6 | -21.02 | 1.01 | 2.11→1.10 | 1.00 |
| 8 | -21.25 | 0.75 | 8.24→4.05 | 2.27 |
| 10 | -20.75 | 0.25 | 7.46→7.45 | 1.54 |

因此 z≈6 的代表亮端箱基本接近观测；z≈8 的这一区间仍有偏高，并非全部亮端
都已吻合。z=10 的该箱衰减较弱，LF 映射包含 Jacobian 且多数查询点触发
生产 min 裁剪，所以数密度变化很小；不能把 A_UV 直接当作 phi 的乘法因子。
这里的比值是对照诊断，未包含尘埃参数不确定性或作联合观测拟合。

### 当前更正：排除 z<10 的 Pop III 瞬时供光

用户更正后的方向是 **z<10 关闭 Pop III、z>=10 保留 Pop III**。
对应 [两页结果讲稿](../slides/21cm_map/popiii_zge10.pdf)，重新演化输出为
`SmallScale21cm/runs/aurora_maps/popiii_zge10_v1/`。
它保留高红移造成的电离状态，关闭之后仍计算 Pop II 供光与复合；
不把电离历史重置成纯 Pop II，也不删除出生历史。
采用相同的双 100 Myr 源表、密度场及物理参数，单独改变发射红移开关。

```bash
# SmallScale21cm：非 debug SLURM GPU 分配运行
# scripts/submit/run_popiii_zge10.sh 调用 --popiii-min-redshift 10
# AuroraLF：完成后绘图
PYTHONPATH=. .venv/bin/python scripts/plot/plot_reionization_observations.py \
  --gated-run ../SmallScale21cm/runs/aurora_maps/popiii_zge10_v1
```

新图和数值记录位于 `outputs/21cm_map/popiii_zge10/`；
`comparison.json` 核验高红移历史与原 Pop II+III 一致、低红移 Pop III
预算为零，并记录关闭光源后可能出现的电离比例下降。

SLURM 157707 在 node3 的 RTX 5080 上完成全部 185 个快照，耗时 13 分 24 秒。
源表全部 191 个红移预算均通过门控核验，高红移电离历史与原 Pop II+III
逐点完全相同；新历史始终位于纯 Pop II 与原 Pop II+III 之间。

| 指标 | Pop II | Pop II+III | III 仅在 z>=10 |
| --- | ---: | ---: | ---: |
| 半电离红移 | 7.102 | 9.225 | 7.630 |
| 99% 电离红移 | 6.376 | 8.017 | 6.622 |
| z=7.54 中性比例 | 0.6992 | 0.0000 | 0.4653 |
| z=7 中性比例 | 0.418 | 0.000 | 0.123 |

关掉低红移 Pop III 后，再电离显著放缓；z=7.54 的 0.465 落入
Davies+2018 报告的 0.37–0.80 区间。但 z=7 的 0.123 仍低于 Mason+2018
推断的 0.44–0.70，不能称为整条历史已经拟合观测。
关闭后，体积平均电离比例从 z=9.95 的 0.34556 下降至 z=9.38 的 0.33623，
随后重新增长，显示复合消耗与 Pop II 供光共同作用；没有在 z=10 重置状态。
单元检查为 15 通过、1 因测试宿主无 CUDA 跳过，完整科学任务已在 GPU 成功运行。

### 更正前的反向对照：排除 z>10 的 Pop III 瞬时供光

[两页开关实验结果](../slides/21cm_map/popiii_z10.pdf)对比纯 Pop II、原 Pop II+III、
以及只在 z<=10 保留 Pop III 发射的重新演化结果。
实验沿用已完成的双 100 Myr 源表、300³/300 Mpc 密度场及逃逸率，
不同时切换尚未重跑的 Pop III 6 Myr 生产产品。
它不修改出生历史或首次爆发条件，也不会在 z=10 补发此前被排除的光子。

| 指标 | Pop II | Pop II+III | III 仅在 z<=10 |
| --- | ---: | ---: | ---: |
| 半电离红移 | 7.102 | 9.225 | 8.369 |
| 99% 电离红移 | 6.376 | 8.017 | 7.578 |
| z=7.54 中性比例 | 0.699 | 0.000 | 0.00649 |

去掉早期供光能推迟再电离，但新实验在 z=7.54 仍近乎完全电离，
与 Davies+2018 推断的 0.60(-0.23,+0.20) 仍有明显偏差。
原因之一是保留的低红移 Pop III 源依然很强：z=8 时，它占瞬时逃逸
电离光子率的 54.38%。这里不是通过改变污染、逃逸率或成星效率来拟合观测。

运行入口在 SmallScale21cm 的 `scripts/submit/run_popiii_z10.sh`，
完整空间历史输出为 `runs/aurora_maps/popiii_zle10_v1/`。
跨开关时间步分为 z=10.15→10（9.536 Myr，III 关闭）和
z=10→9.95（3.251 Myr，III 打开）；密度按宇宙时间插值，
每侧重新计算 HMF/复合。开关之前与已完成的 Pop II 历史逐快照完全一致。

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_reionization_observations.py \
  --gated-run ../SmallScale21cm/runs/aurora_maps/popiii_zle10_v1
```

绘图脚本核对源表、密度文件 SHA256、宇宙学和空间参数；核验全部 191 个
源红移的门控预算，以及新历史位于两条原对照之间。
输出 `outputs/21cm_map/popiii_z10/`，正式资产在 `slides/assets/reionization/`。
观测集合与推断限制沿用下述记录，结果属于固定参数的诊断实验。

### 原模型的观测对照

[两页观测对照](../slides/21cm_map/reionization_observations.pdf)使用最近完成的
`SmallScale21cm/runs/aurora_maps/instantaneous_100myr_v1/`，按
`1 - mean_xhii` 转换为体积平均中性比例。两类恒星逃逸率均为 0.2，
Pop III 爆发效率为 0.03。没有重跑地图、修改参数，或把旧历史标为 Pop III 6 Myr 的新结果。

观测来源与原文数值位于 `external_data/observations/reionization/`。
收录 McGreer+2015 暗像素上限、Mason+2018 星系 Lyα 等值宽度推断、
Davies+2018 两颗类星体阻尼翼，以及 Mason+2026 JWST 阻尼翼（2025 预印本
2501.11702v2；含 GNz11 与视线方差）。JWST 横线是 5.5–8、8–10.6 的
样本红移分箱；其余方法这里只标出文献代表红移。所有区间保留文献含义，
上限用箭头表示，不当成检测点，也不把重叠/模型依赖的约束合并为独立 χ²。

在 z=7.09，Pop II / Pop II+III 的中性比例为 0.491 / 0，观测为
0.48(-0.26,+0.26)；在 z=7.54 为 0.699 / 0，观测为
0.60(-0.23,+0.20)。当前组合模型在 z≈7–8 的电离进展早于这些观测推断。
Pop II 基线更接近该阶段数据，但并非对所有观测均拟合良好。两组半电离
红移为 7.102 / 9.225，99% 电离红移为 6.376 / 8.017。

地图历史最低红移为 6.01；5.6、5.9 的暗像素观测显示于图中，但未外推模型
至这些点，也未对它们计算符合程度。密度场来源仍标记为
`legacy-cosmology-unverified`。这是固定参数的诊断对照，尚非正式统计排除。

绘图和可复现检查：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_reionization_observations.py
```

输出 `outputs/21cm_map/observations/neutral_history.{png,pdf}`、
`comparison.json`（交叉红移、观测位置的模型值、输入 SHA256 与版本说明）。
讲稿依赖矢量图位于 `slides/assets/reionization/neutral_history.pdf`。

## 2026-09-12：成星率密度与 Venditti (2023) 图 1 对比

[三页结果与定义](../slides/21cm_map/sfrd.pdf)采用 Venditti et al. (2023),
MNRAS 522, 3809, [Fig. 1](https://doi.org/10.1093/mnras/stad1201) 左面板的定义：
最近 10 Myr 新形成的初始恒星质量 / 时间 / 共动体积。论文原始矢量图和
TeX 图注位于 `external_data/literature_sources/popiii_uvlf_library/papers/Venditti2023ANeedleInA/`。
仅提取论文 Pop II、Pop III 与总和的均值曲线，未复制观测点或盒间散布。
矢量坐标由可见轴刻度校准，63 个共同节点的 II+III=total 闭合误差最多
0.013%；这仅检验提取精度，不代表论文模型的物理精度。

当前模型使用生产种子、960 点时间网格和每点 256 条主支，计算 19 个红移、
111 个质量点（10^4--10^15 Msun），随后按当前地图的 Reed07 HMF 权重积分。
Pop II 积分窗口内 SFR；Pop III 只对窗口内首次爆发的出生质量求期望。
不乘 SSP 产光率或逃逸率，不对累计光子数差分。该成星统计窗口与 6 Myr
电离供光上限含义不同。

| z | Pop II SFRD | Pop III SFRD | Pop III 成星质量占比 |
| --- | ---: | ---: | ---: |
| 6 | 2.3951e-2 | 1.2627e-3 | 5.01% |
| 8 | 8.3624e-3 | 1.4365e-3 | 14.66% |
| 10 | 2.6269e-3 | 1.2411e-3 | 32.09% |
| 15 | 1.2525e-4 | 4.7590e-4 | 79.16% |
| 20 | 4.2710e-6 | 1.2599e-4 | 96.72% |

SFRD 单位均为 Msun yr^-1 cMpc^-3。z=6 的 5.01% 是新形成质量占比，
此前 26.69% 是瞬时电离光子占比，二者不可混用。
我们的 Pop III SFRD 在 z=7、10、15 约为该论文均值的 2.8、6.6、55 倍。
比较时在 log(SFRD) 上插值；论文曲线只在已显示的红移范围内使用。
两模型在主支/全盒统计、成星效率、冷却、反馈、IMF 与化学演化方面均有差别；
论文还明确指出 z>~15 成星起点受质量分辨率影响，因此不能把全部差异归因于污染。

同时计算 1、5、10、20 Myr 窗口。z=6 将 10 改为 5 Myr，Pop II 改变
+0.788%，Pop III 改变 -0.0217%。高红移窗口影响更大：整个区间内，
1 相对 10 Myr 的最大差异分别为 58.1%、13.7%；这是不同平均窗口的结果，
不等同于时间网格误差。16/32 阶爆发质量求积的全局差异小于 1e-15；
10 Myr 主曲线的最大 MAH 均值标准误分别为 1.24%、2.33%，未包括物理系统误差。

SLURM 157671 在 amd1 使用 8 核完成，耗时 1 分 50 秒。本次只生成 SFRD
诊断，没有改变生产参数，也没有重跑空间电离历史。复现入口：

```bash
# 通过非 debug SLURM 分配运行；保留默认生产种子和采样规模
PYTHONPATH=. .venv/bin/python scripts/analysis/build_current_sfrd.py
PYTHONPATH=. .venv/bin/python scripts/analysis/summarize_current_sfrd.py
PYTHONPATH=. .venv/bin/python scripts/plot/plot_current_sfrd.py
```

数据、协方差、窗口对照与参考图提取记录：`data_save/ionizing_sources/sfrd_v1/`。
正式矢量图：`slides/assets/sfrd/`；预览与审阅记录：`outputs/21cm_map/sfrd/`。
新增解析基准检查位于 `tests/test_sfrd_diagnostic.py`（3 项通过）。

## 2026-09-12：低红移近期首次爆发诊断

[两页数值说明](../slides/21cm_map/popiii_recent_bursts.pdf)区分单次爆发的衰老、
近期首次爆发的概率和整个晕群体的瞬时光子占比。5 页核心主讲稿保持不变。

复用生产种子、256 条 MAH/点和原阈值分布，在 z=6、10、15 与
终态质量 10^8、10^10、10^12 Msun 的 9 个点计算首次越阈时间分布。
概率对全部 MAH/阈值组合求平均，包含尚未爆发与起点前事件；
已解析爆发年龄中位数仅条件于 z=50 之后的首次爆发。

z=6 的 10^12 Msun 晕最近 6 Myr 首次爆发概率为 0.01877%（MAH 标准误
0.00136 个百分点），已解析爆发年龄中位数 721.1 Myr；群体平均 Pop III
光子占比为 1.533%。10^10 Msun 对应 0.05137%（标准误 0.00325 个百分点）、
719.2 Myr 和 32.93%。小概率不等于同样小的光子占比，后者还按出生恒星质量、
SSP 年龄与 Pop II 供光加权。

使用已有 6 Myr 配对 Pop III 源表、100 Myr Pop II 源表和 Reed07 权重，
全局 Pop III 光子占比在 z=6、8、10 分别为 26.69%、54.22%、76.48%。
z=6 的 Pop III 光子有 70.28% 来自质量小于 10^10 Msun 的晕。
当前 SSP 中，z=10 爆发到 z=6 的年龄是 459.4 Myr，未截断的产光率仅为
1 Myr 时的 2.28e-13。当前 6 Myr 上限将这种旧爆发的供光置零。

这些是当前主支、单次爆发且尚无原初气体/金属污染门控模型的结果；
晚期新爆发的物理可信度仍需额外判据。本次没有重算空间电离历史。
复现：`PYTHONPATH=. .venv/bin/python scripts/analysis/explain_popiii_recent_bursts.py`。
数值和输入哈希：`data_save/ionizing_sources/recent_bursts_v1/summary.json`。

## 2026-09-12：Pop III 年龄窗口收敛检查

以原 100 Myr
已解析 Pop III 通道为基准，复用原始种子、每点 256 条 MAH、960 点时间网格，
检查 111 个质量点（10^4–10^15 Msun）和红移 6、8、10、12.5、15、20、
25.24、30.16、39.8、49。完整首次越阈历史不裁剪，只改变产光年龄上限。
采用原地图的 Reed07 质量函数与 1000 点对数质量积分权重。

单次爆发 SSP 的 99% 累计产光年龄为 5.222624 Myr。群体当前逃逸光子率
相对 100 Myr 的最大损失如下（均出现在 z=6）：

| 年龄上限 / Myr | 最大 Pop III 供光损失 |
| --- | --- |
| 3 | 6.6054% |
| 5 | 1.1317% |
| 5.25 | 0.9852% |
| 6 | 0.6531% |
| 8 | 0.2428% |
| 10 | 0.1082% |

最短已测达标窗口为 5.25 Myr；建议 6 Myr 留出余量。6 Myr 的最大损失加
两倍配对 Monte Carlo 标准误为 0.65347%。32/64 阶求积差异不超过参考
Pop III 总率的 0.000631%；16 阶结果重现原生产表，最大相对差 1.60e-11。
独立数值积分也重现 SSP 累计产额的解析积分。

这里的 1% 指被检验红移的全局 Pop III 瞬时供光，不代表每个晕或空间电离
地图的误差。6 Myr 下，相对误差超过 1% 的质量格点最多只占该红移参考
Pop III 总供光的 0.0000156%。起点前未解析爆发不计入参考；z=49 时其上界
较大，因此该点仅检验已解析通道。该测试未重跑 Pop II 或空间演化；随后生产
设置采用 Pop III 6 Myr，Pop II 保持 100 Myr。

SLURM 作业 157658（fat2，10 核）完成于 9 分 33 秒；配对率、协方差、
HMF 权重与逐红移结果保存在 `data_save/ionizing_sources/popiii_age_window_v1/`。
复现入口为 `scripts/analysis/check_popiii_age_window.py`（需 SLURM），随后运行
`scripts/analysis/summarize_popiii_age_window.py` 和
`scripts/plot/plot_popiii_age_window.py`。渲染检查在
`outputs/21cm_map/popiii_age_window/`。

## 2026-09-12：历史双 100 Myr 结果

主讲稿 [从 Pop III 恒星形成到 21 cm 空间图](../slides/21cm_map/21cm_map.pdf)
已由 51 页精简为 5 页，只保留成星、瞬时供光、质量函数加权、电离演化与亮温的
核心公式和思路；目前已更新为 Pop II 100 Myr、Pop III 6 Myr。原 51 页保存在
`slides/21cm_map/archive/full_51pages_20260912/`。
[100 Myr 结果对照](../slides/21cm_map/lookback_100myr.pdf)仍保留原 7 页，数值记录如下。

主讲稿字体为微软雅黑（中文）、Arial（英文）与 Latin Modern Math（公式）；
标题粗体，正文常规。采用 GALPROP 电子分布讲稿的公式解释风格：白底、深蓝标题、
公式项分色，同色箭头连接物理解释。可复用前导文件为
`slides/templates/formula-explained.tex`；默认模板同时保存在个人 Beamer skill 中。
请从仓库根目录用 XeLaTeX 编译。当前从
`/home/zhuhourui/.local/share/fonts/microsoft-academic/` 按文件名加载字体；
换机器时可在输入模板前设置 `\AcademicLatinFontPath` 和 `\AcademicCJKFontPath`。
微软雅黑常规和粗体文件分别为 `msyh.ttf` 与 `msyhbd.ttf`，
避免旧字体共享内部名称导致字重误选。字体来源与校验值记录在
`external_data/fonts/microsoft/provenance.json`，编译与逐页检查记录在
`outputs/21cm_map/annotated/`。PDF 已嵌入所用字体。

Pop II 和 Pop III 的当前电离光子率统一限制为恒星年龄不超过 100 Myr。
这只裁剪供光积分；z=50 起的完整 MAH、延迟 SFR、首次爆发记录和已有 IGM
电离状态继续保留。窗口以内采用真实 SSP 年龄衰减。配置、边界和单位见
[氢电离光子源说明](ionizing-sources.md)。

与前一轮全历史瞬时模型相比，随机种子、历史网格、SSP、逃逸率、爆发效率、
密度场与空间演化设置均相同。两类 fesc 仍为 0.2，epsilon_b 仍为 0.03。
新源表位于 `data_save/ionizing_sources/instantaneous_100myr_v1/`；
地图位于 `SmallScale21cm/runs/aurora_maps/instantaneous_100myr_v1/`。

源表完整重算 191×111 个点。在 z=6，Pop II 当前逃逸率减少 0.208119%，
Pop III 减少 1.82858e-6%；Pop III 当前供光占比由 26.779213% 变为
26.820083%。占比略增是因为 Pop II 分母减少更多，Pop III 本身没有变亮。
这也说明上一轮瞬时模型已经将古老 Pop III 的当前供光衰减至很低水平。

两组 185 快照完整历史的结果为：

| 源模型 | 全历史瞬时 z50 | 100 Myr z50 | 半电离时刻推迟 [Myr] | 最大电离度变化 [百分点] |
| --- | ---: | ---: | ---: | ---: |
| Pop II | 7.103277 | 7.101573 | 0.236262 | 0.182146 |
| Pop II + 已解析 Pop III | 9.225337 | 9.224889 | 0.034766 | 0.036304 |

两组新历史的平均电离度均未高于原历史。100 Myr 窗口下，Pop II 的
z10/z50/z90 为 8.915994/7.101573/6.654916，Pop II+III 为
13.517985/9.224889/8.357713。以上小数用于配对数值比较，不代表物理预测精度。

回归测试 1154 项通过，包括供光边界裁剪和旧爆发不能在窗口内重新触发。
本轮没有增加金属污染或原初气体反馈判据，也没有重跑 GPU 跨硬件与时间分辨率
比较；下文对应检验仍指前一轮全历史瞬时模型。

三组 300³、185 快照地图已在 RTX 5080 完成（SLURM 157634）。
供光窗口对照数据与输入哈希保存于
`outputs/21cm_map/lookback_100myr/comparison.json`。
25%、50%、75% 同电离度地图和功率谱后处理也已完成（SLURM 157635），
可复用产品为 `SmallScale21cm/data_save/aurora_maps/instantaneous_100myr_v1/matched_ionization/`，
预览位于 `outputs/21cm_map/lookback_100myr/matched/`。
仍用最近真实快照展示地图、两侧快照的谱作阶段插值，不对空间场插值。

重绘新旧对照：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_lookback_comparison.py \
  --reference ../SmallScale21cm/runs/aurora_maps/instantaneous_v1 \
  --run ../SmallScale21cm/runs/aurora_maps/instantaneous_100myr_v1
```

## 历史记录：2026-09-12 全历史瞬时源

本轮审阅材料为 [瞬时电离光子源与再电离演化](../slides/21cm_map/instantaneous.pdf)。
下面 2026-09-11 的累计预算结果仅保留作历史比较，不代表当前默认实现。

Pop II 用 SFR 与年龄依赖 qH 卷积，Pop III 用出生质量乘以当前年龄的 qH；
两类源均返回逃逸 photons/s/halo，不对不同终态主支的累计产额差分。
原首次越阈 q 分布改为条件于每条 MAH 的数值期望积分，保留其原分布参数。
`SmallScale21cm/scripts/run/run_aurora_maps.py` 现转入瞬时演化入口；历史入口
归档为 `scripts/experiments/run_aurora_maps_cumulative.py`。

每个平滑区域推进 dQ/dt = S − RQ，保留气体电离状态；源关闭后可以复合。
每步使用端点源率平均和宇宙时间中点密度，解析推进步内常系数方程。
Q 达到 1 后的多余光子记录为 surplus，不存入后续光子库。
空间标记每步重建，没有永久电离掩膜。局部收支闭合不代表重叠标记的整盒
严格光子守恒；仍使用半解析空间闭合、高自旋温度近似及无 RSD。

| 源模型 | 瞬时 z10 | 瞬时 z50 | 瞬时 z90 | 历史累计 z50 |
| --- | ---: | ---: | ---: | ---: |
| Pop II | 8.917 | 7.103 | 6.657 | 7.313 |
| Pop II + 已解析 Pop III | 13.518 | 9.225 | 8.358 | 10.684 |

本次 Pop III 将半电离宇宙年龄从 750.0 Myr 提前到 529.3 Myr，提前 220.7 Myr；
历史方案为 288.5 Myr。此比较还包含源表加密和阈值求积改变，是完整新旧方案
比较，不是仅改变 SSP 年龄处理的单变量实验。
z=6 当前供光中 Pop II/III 为 73.22%/26.78%；z=10 为 23.42%/76.58%。
尚未加入金属污染、原初气体或光加热反馈门控，不据此宣称低红移 Pop III
形成率已经物理验证。起始密度快照 z=48.52 采用 Q=0，未推断更早的电离状态。

结果与复现路径：

- 源表：`data_save/ionizing_sources/instantaneous_v1/`，191 红移 × 111 质量点，
  每点 256 条 MAH，960 点历史，16 点阈值积分。
- 主地图：`SmallScale21cm/runs/aurora_maps/instantaneous_v1/`，300³ 网格，
  185 个快照，50 个平滑尺度，三种源情景。
- 同电离度后处理：`SmallScale21cm/data_save/aurora_maps/instantaneous_v1/matched_ionization/`；
  运行 `match_aurora_ionization.py --crossing first_upward`，允许非单调历史，
  地图仍取最近真实快照，功率谱在目标两侧快照的谱上插值。
- GPU/步长比较：`SmallScale21cm/outputs/instantaneous_run_validation/`；
  实测 RTX 5080 与 RTX 3080，未在本轮重测 RTX 3090。
- 源时间分辨率：`SmallScale21cm/outputs/instantaneous_source_convergence/summary.json`；
  z=6、10、15 历史网格 960→1920 后 Pop II 当前率最大变化 0.981%，
  Pop III 最大变化 0.0090%，不是全红移空间分辨率收敛证明。

完整历史的 RTX 5080/3080 平均电离度最大差为 3.36e-8。
5 个完整三维快照核对中，z=9.95 存在 1/27,000,000 个完全电离标记不一致，
该像素电离度差为 0.502，整场 RMS 差为 9.66e-5；其余 4 个快照的标记一致。
这是统计量一致而非逐像素或逐比特一致。3080 同硬件加倍步长后 z50=9.2468，
平均电离度最大差为 0.009952；本次正式结果采用原较细步长。

同电离度实际快照对目标的最大偏差为 0.03287（目标 75% 的 Pop III 组，
真实快照为 71.713%），因此地图是最近阶段比较。50% 目标下两组真实快照
分别为 49.500%（z=7.11）和 50.817%（z=9.19）。在插值到 50% 的功率谱上，
k=0.10136 cMpc⁻¹ 分箱的 Pop II+III / Pop II 中性比例功率比为 0.386，
亮温功率比为 0.490；k=1.00934 cMpc⁻¹ 时分别为 1.181 和 1.200。
插值两端的谱范围单独显示，不当作误差界。

重绘时从 AuroraLF 根目录执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_instantaneous_21cm.py \
  --run ../SmallScale21cm/runs/aurora_maps/instantaneous_v1 \
  --legacy ../SmallScale21cm/runs/aurora_maps/mainbranch_v1
PYTHONPATH=. .venv/bin/python scripts/plot/plot_matched_ionization.py \
  --data ../SmallScale21cm/data_save/aurora_maps/instantaneous_v1/matched_ionization \
  --assets slides/21cm_map/assets/instantaneous \
  --previews outputs/21cm_map/instantaneous
```

## 历史记录：2026-09-11 累计预算方案

日期：2026-09-11。新分支 `codex/popiii-21cm-map` 已实现氢电离源与
SmallScale21cm 接入，已完成 εb=0.03 主支近似的 300³ 配对地图。
本次未安装或运行 21cmFAST。

实现、单位、源表与时间边界见 [氢电离光子源说明](ionizing-sources.md)。
审阅用 slides 仍为 `slides/21cm_map/archive/full_51pages_20260912/21cm_map.pdf`；方法之后补充实际结果。
旧密度生成宇宙学/种子尚未独立验证，次支系与 minihalo 气体闭合未包含，
因此首版是同一给定密度实现上的条件响应研究。

本次运行 `SmallScale21cm/runs/aurora_maps/mainbranch_v1`：185 个红移
（48.52→6.01），三种源预算，SLURM 157540，RTX 5080，正常结束耗时 16分39秒。
在 z=12.62，Pop II / Pop II+已解析 Pop III / 再加早期事件上界三组平均亮温分别
为 30.97 / 20.22 / 8.21 mK。源表加密检查见该项目
`outputs/aurora_maps/source_convergence.json`；z=12.5 已解析 Pop III 预算变化 -0.70%。

## 同电离度比较与再电离历史

2026-09-11 补充了体积平均 xHII = 0.25、0.50、0.75 的比较。
原运行的三组历史均单调且覆盖这些阶段；没有重算源或修改电离场。
最近真实快照用于地图，实际 xHII 与红移在图上逐一标出；与目标的最大偏差
为 0.02279（2.28 个百分点）。功率谱由目标两侧的完整三维场分别计算，再在
平均 xHII 上线性插值；浅色范围只表示两侧快照的谱值变化，不是误差界或置信区间。
没有对空间场插值或缩放，也没有对历史外推。

| 源模型 | z10 | z50 | z90 |
| --- | ---: | ---: | ---: |
| Pop II | 9.55 | 7.31 | 6.83 |
| 加已解析 Pop III | 16.21 | 10.68 | 9.27 |
| 再加起点前 Pop III 上界 | 19.59 | 13.83 | 11.75 |

这里的 z10/z50/z90 由相邻输出的平均电离度对红移作线性插值得到。
匹配相同电离度后，不同模型仍对应不同红移和密度、复合历史。
中性比例功率谱（无量纲）直接衡量 xHI 起伏；亮温功率谱（mK²）另外包含
A(z) 和密度权重。两者均先扣除体积均值，不除以均值。
在 xHII=0.50、k=0.10136 comoving Mpc⁻¹ 的分箱，已解析 Pop III 组的中性比例
和亮温功率分别为 Pop II 基线的 0.152 和 0.186；在 k=1.00934 Mpc⁻¹ 处为
1.023 和 1.205。这些是当前条件响应模型的结果，保留上文的物理限制。

后处理 SLURM 157553（amd1）正常结束，耗时 1分54秒，读取 18 个真实快照。
可复用切片、功率谱、输入哈希、匹配权重和历史保存在：
`SmallScale21cm/data_save/aurora_maps/mainbranch_v1/matched_ionization/`。
计算入口为该项目的 `scripts/analysis/match_aurora_ionization.py`，通过计算节点执行：

```bash
packages/EoRCaLC/.venv/bin/python scripts/analysis/match_aurora_ionization.py \
  --run runs/aurora_maps/mainbranch_v1 --output data_save/aurora_maps/mainbranch_v1/matched_ionization
```

输出目录必须不存在，以防覆盖已有结果。匹配逻辑的 10 项测试通过；产物的
输入哈希、三维均值、9 组插值、切片范围均已检查。
AuroraLF 中重绘图像的入口：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_matched_ionization.py \
  --data ../SmallScale21cm/data_save/aurora_maps/mainbranch_v1/matched_ionization
```

slides 第 26 页是电离历史，第 32–41 页是同阶段的方法、地图和功率谱。

## 提前幅度的解释与核查

第 43–44 页补充对提前幅度的审查。已解析 Pop III 组的 z50=10.684，
对应宇宙年龄 433.4 Myr；Pop II 基线 z50=7.313、年龄 721.9 Myr，提前约 288.5 Myr。
这不是仅由图像或亮温转换造成：源表在 z=12.5 已给出 Pop III/Pop II=22.66
的累计逃逸光子比，在 z=15 为 57.90。上述均不含左删失额外上界。

已读实现表明：SSP 产额先对恒星年龄积分；Pop III 每条已解析主支只计一次
越阈爆发；源适配器没有额外乘 fesc 或 duty；空间时间循环没有把累计源预算
逐步重复求和。这个调用链核查不能代替合并树归属与完整光子守恒验证。

当前 εb=0.03、fesc,III=0.2，爆发晕质量 1e8 Msun 对应 Pop III 初始质量
4.744e5 Msun。源是否能普遍这样形成尚无原初气体存活模型验证：
`first_crossing` 只根据 M/Mcool 与 q 判定，不检查已有 Pop II 或金属污染，
也没有自洽 LW/光加热抑制。独立终态主支和缺失早期 Pop II 还可能改变基线。
因此不把本次大幅提前解释为 Pop III 的普遍预期，也不据此直接修改既有参数。

作为物理背景，[Visbal, Haiman & Bryan (2015), §2–3](https://academic.oup.com/mnras/article/453/4/4456/2593745)
说明 LW、富集和光加热对 Pop III 再电离贡献的限制；该文不同的 minihalo 模型与
逃逸率不允许直接套用其效率上限。
[Planck 2018 VI](https://arxiv.org/abs/1807.06209) 的 τ=0.054±0.007
是可使用的积分约束；当前尚未按完整质量加权电子历史完成该项比较，不能仅根据
z50 报告排除显著性。本轮只核查与解释，没有改变模型或重算生产地图。

## 当前方案：复用 SmallScale21cm，分别加入 Pop II/III 光子贡献

### 各红移的累计光子占比

slides 第 45–47 页补充占比表和组成图。两套分母分别为：

- 已解析模型：NII + NIII,res；
- 起点前事件上界情景：NII + NIII,res + NIII,extra,upper。

| 红移 | 已解析模型 Pop II | 已解析模型 Pop III | 上界情景 Pop III 合计 |
| --- | ---: | ---: | ---: |
| 20 | 0.23% | 99.77% | 99.93% |
| 15 | 1.70% | 98.30% | 99.38% |
| 12.5 | 4.23% | 95.77% | 98.32% |
| 10 | 10.57% | 89.43% | 94.87% |
| 8 | 23.32% | 76.68% | 86.68% |
| 6 | 47.38% | 52.62% | 66.88% |

z=6 的 Pop II、已解析 Pop III 和起点前额外上界分别为 2.6870、2.9837 和
2.4419 photons/H。Pop II 在低红移的累计份额已接近一半；此统计包含过去产生的
光子，不能用来判断该时刻的瞬时发射率或拆分电离度。

全部 9 个源表红移（6、8、10、12.5、15、20、30、40、49）的原始光子数及
两套归一化占比保存在 `data_save/ionizing_sources/photon_shares/photon_shares.csv`，
其中 fraction 列为 0–1。配套 manifest 记录输入、脚本、CSV 哈希和分母定义。
低于 z=6 尚无源表，本次不外推。重绘和重建表格：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_photon_shares.py \
  --run ../SmallScale21cm/runs/aurora_maps/mainbranch_v1
```

脚本验证预算非负、分母非零、每种组成之和为 1，并检查占比乘总量恢复原预算。
没有对独立终态主支预算差分以构造未经验证的瞬时产光率。

### 瞬时产光率与参数诊断

针对“低红移应由 Pop II 主导”的问题，新增直接卷积得到的瞬时逃逸光子率：
QII(t)=fesc,II ∫SFRII(tb) qH,II(t−tb) dtb，
QIII(t)=fesc,III Mstar,III qH,III(t−tburst)。
SSP 的 qH 单位为 photons/s/初始 Msun，出生积分用年。
不对不同终态独立采样的累计预算差分；左删失事件的未知当前速率没有被虚构。

`auroralf/experiments/ionizing_audit.py` 复现原始种子、分块、960 点历史、SFR 和
越阈规则，附加当前率与“爆发之前已有 Pop II 成星”诊断。初次 SLURM 157559
对 6 个红移、111 个质量点、每点 256 条历史计算：666 个网格点的累计源复现
相对容差 1e-12，全局 HMF 预算复现相对容差 1e-10。
原样本的低红移当前率存在较大稀有爆发采样噪声，因此 SLURM 157560 将
z=6、8 加密到每点 4096 条；物理参数与时间网格不变，样本包含原始 256 条。

| 红移 | 当前 Pop II 占比 | 当前 Pop III 占比 | 占比 MC 标准误 | 原累计 Pop III 占比 |
| --- | ---: | ---: | ---: | ---: |
| 6 | 71.70% | 28.30% | 3.99 个百分点 | 52.62% |
| 8 | 46.88% | 53.12% | 3.67 个百分点 | 76.68% |

所以当前模型在 z=6 的瞬时供光已由 Pop II 主导，不能把原累计份额当作该时刻的
产光率。z=8 两类的当前贡献接近。误差仅为固定模型 MC 均值的线性传播标准误，
不是反馈/SSP/时间网格误差；没有完成瞬时率的时间网格收敛检验。
Pop II 出生积分的 8 点与16点求积差别在本批次不超过0.023%。

加密样本在 z=6 的 Pop III 累计/当前光子中，分别有 68.49% / 99.33% 来自
爆发前已有模型 Pop II 成星的历史；z=8 为 58.29% / 95.34%。这里依据爆发前
分段线性 SFR 的正积分分类，不是金属污染率，不能证明晕内已无任何原初气体。
它定位了需要补充原初气体分量或存活概率的源模型环节；当前没有删除此类事件。

εb=0.03 在项目路线中明确是尚未联合拟合的参考值。固定历史时源幅度正比于
εb fesc,III；原 z=6 累计 Pop III 份额随 εb=0.03、0.01、0.003 分别为
52.62%、27.01%、9.99%。这只是源幅度的代数敏感性，不是参数拟合或地图重算。

原审查、加密结果分别保存在 `data_save/ionizing_sources/rate_audit_v1/`、
`rate_audit_refined/`，包含每个质量点的均值、速率均值协方差、全局结果和哈希。
计算入口 `scripts/analysis/audit_ionizing_rates.py` 与
`scripts/analysis/refine_ionizing_rates.py` 通过 dmde-compute 提交；前者显式使用
SmallScale 自身虚拟环境重建原 Reed07 权重，不向 AuroraLF 环境添加依赖。
相关解析校验 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ionizing_audit.py`
两项通过。重绘：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_ionizing_rate_audit.py \
  --data data_save/ionizing_sources/rate_audit_v1 \
  --refined data_save/ionizing_sources/rate_audit_refined
```

对应 slides 第48–50页；绘图同时生成 `rate_comparison.csv` 与
`amplitude_sensitivity.csv`，归档于原审查目录。生产源表和地图保持原样。

### 原接入说明

根据后续讨论，优先复用 SmallScale21cm 的现有密度场、条件质量函数与
半解析电离框架。解释性 slides 及本次结果：
[从 Pop III 恒星形成到 21 cm 空间图](../slides/21cm_map/archive/full_51pages_20260912/21cm_map.pdf)。
旧 3 页框架核查保存在 slides/21cm_map/archive/initial_framework_20260911/。

计算顺序为：逐历史成星与 SSP 得到光子率 → 积分逃逸率得到累计光子数 →
固定晕质量、红移下分别统计 Pop II/III → 条件质量函数映射到区域源项 →
与氢原子、复合和 minihalo 消耗比较 → 中性比例与温度转为 21 cm 亮温。
不要求先把样本压缩成总 UVLF，也不要求先构建显式逐晕空间目录。

平均光子数可以保留 Pop III 的平均增强；忽略源分布及其时空相关性会损失
随机起伏信息。第一版假设固定质量和红移后平均源贡献不额外依赖环境。
局部源项、全局归一化、质量积分边界与 minihalo 消耗要共同核对；
不得在已包含爆发概率的统计结果上再乘旧 duty cycle 或重复 HMF 权重。

累计时间仍是待闭合问题：当前独立终态 MC 主支不能自动提供完整祖先贡献
和连续区域光子史。初始越阈事件、合并归属以及区域时间积分须明确处理。
原 SmallScale21cm 采用高自旋温度近似；早期耦合与加热需要另行补充。
当前实现分别保存已解析贡献和左删失 Pop III 的额外上界；这种界限不补全缺失的次支系。

主要文献：Duncan & Conselice (2015), DOI 10.1093/mnras/stv1049；
Park et al. (2019), arXiv:1809.08995，式（15）–（16）；
Nikolić et al. (2024), arXiv:2406.15237，§2 与 Algorithm 1。
书目信息已通过 ADS 核对，公式及方法已读取原文。

以下保留初期框架核查。涉及显式空间晕身份的要求适用于逐晕路线；
它们不是当前条件质量函数统计路线的全部先决条件。

## Muñoz 图 6 对照：补算到 z=5（2026-09-12）

用户指定的“左侧下降”是 Muñoz et al. (2022), MNRAS 511, 3657 的图 6，
2021 年预印本 arXiv:2110.13919；DOI 10.1093/mnras/stac185。
这里是 Pop III 在 z≈10 达峰后下降，不是 Madau–Dickinson 总 SFRD 在 z≈2 的峰。
相应扩展范围采用 z=5–30，不再沿先前误解补算至 z=0。

- 审阅讲稿：`slides/21cm_map/munoz_sfrd.pdf`（2 页）。
- 绘图：`scripts/plot/plot_munoz_sfrd.py`；预览及完整矢量图
  `outputs/21cm_map/sfrd/munoz_comparison.{png,pdf}`。
- 原始论文源包、图和 ADS 元数据：`external_data/literature_sources/munoz2022_eos/`。
  从原始图矢量路径提取三条基准曲线；不外推不可见曲线、不复制观测点。
- 补算：`data_save/ionizing_sources/sfrd_z5_v1/`，SLURM 157672，amd1，8 CPU，
  30 秒完成；z=5、5.25、5.5、5.75，111 个质量点，每点 256 条主支。
  高红移使用原有 `sfrd_v1`，模型参数没有调整。

按原图曲线读取，论文 Pop III 峰值约 6.50e-4（z≈9.95），z=5 为 1.50e-4，
即下降约 77%。我们已计算点的峰值为 1.44e-3（z=8），z=5 为 9.91e-4，
即下降约 31%。z=5、6、8、10、15 的 Pop III SFRD 分别为论文的
6.6、4.1、2.5、1.9、1.4 倍。单位 Msun/yr/cMpc^3；论文数值为读图近似。
补算的 Pop III 采样标准误约 1.5%，5/10 Myr 窗口差异小于 0.09%；
这不代表系统物理误差或全部离散化误差。

论文按平均密度区域中的 ACG/MCG 族群计算，并包括 LW、相对速度与光加热反馈；
我们按全局 Reed07 加权的主支首次爆发期望计算，没有上述自洽反馈与金属污染门控。
族群划分、效率与时间处理也不同，不能把全部差异归结为污染或某一个反馈。
SFRD 只统计新形成质量，不含 SSP/fesc；缩短产光年龄不能改变这些 SFRD。

复现补算时指定新的输出目录，防止覆盖：

```bash
# 在非 debug SLURM 分配内运行
PYTHONPATH=. .venv/bin/python scripts/analysis/build_current_sfrd.py \
  --output data_save/ionizing_sources/sfrd_z5_v1 --redshifts 5 5.25 5.5 5.75
PYTHONPATH=. .venv/bin/python scripts/analysis/summarize_current_sfrd.py \
  --run data_save/ionizing_sources/sfrd_z5_v1
PYTHONPATH=. .venv/bin/python scripts/plot/plot_munoz_sfrd.py
```

## 共享模型与目标

延续 [Pop III 预测路线](popiii-predictions-roadmap.md)：同初始条件下比较
Pop II 基线、Pop II + 随机首次越阈 Pop III，以及两者的亮温差。首轮目标
为共同红移切片、体积平均信号与功率谱；光锥和仪器响应在本征场验证后加入。
红移、盒子大小、分辨率与 X 射线/逃逸参数尚未确定。

`auroralf/experiments/random_q.py` 使用原子冷却阈值，
log10(q)~N(0.5,1.5²)，每条 McBride 历史只抽一次 q；形成质量为
0.03(Ωb/Ωm)Mh(tb)。Pop II 保留延迟 SFR，Pop III 沿用零金属 logE SSP。
不能因后端提供 minihalo 开关，就改为其内置的分子冷却成星模型。

## 框架与接口证据

- 本地 Zeus21 固定于 9f2d2105。`zeus21/maps.py` 的 CoevalMaps 仅实现
  KIND=0/1，从功率谱生成近似场；KIND=2/3 未实现。它没有演化本项目的
  空间爆发与辐射历史。此结论只适用于固定 checkout。
- 已取回 21cmFAST v4.2 干净源码，commit 为
  `2cb6000d61381c658ccbe68028c74ca6b0c46cdc`；来源与许可证哈希见
  `external_data/source_manifests/21cmfast.toml`。目前只作为接入候选。
- 该版本 `src/py21cmfast/wrapper/outputs.py` 的 PerturbedHaloCatalog
  暴露 sfr、ion_emissivity、xray_emissivity；HaloBox 暴露 halo_sfr、
  halo_sfr_mini、n_ion、halo_xray。需沿调用链确认字段是否被重算。
- 同文件 HaloCatalog 有质量、坐标、随机变量和 desc_redshift，没有直接
  导出的逐晕父子 ID 数组。后代条件抽样尚不能直接作为一次爆发继承的完整树。
- `src/py21cmfast/src/SpinTemperatureBox.c` 的 global_reion_properties
  仍调用 EvaluateNionTs / EvaluateNionTs_MINI。只替换局部 SFR 网格会留下
  内置全局源模型；必须统一全局电离史与 X 射线光深链。
- [官方 halo sampler 教程](https://21cmfast.readthedocs.io/en/stable/tutorials/halosampler.html)
  将 SFR 网格标为 Msun/s/Mpc³，AuroraLF 单晕 SFR 为 Msun/yr。接口必须
  显式处理秒、年和共动体积，并以体积积分检查源总量。

固定源码：[字段定义](https://github.com/21cmfast/21cmFAST/blob/2cb6000d61381c658ccbe68028c74ca6b0c46cdc/src/py21cmfast/wrapper/outputs.py)、
[温度源项](https://github.com/21cmfast/21cmFAST/blob/2cb6000d61381c658ccbe68028c74ca6b0c46cdc/src/py21cmfast/src/SpinTemperatureBox.c)。

## 尚缺的物理输入

1. 与密度和速度场共享初始条件的空间晕演化：质量、位置、时间、逐晕身份、
   父子关系及分辨率。现有 R032、R024–R027 无这些空间字段，不能按 HMF
   权重随机放点后声称得到物理一致的地图。
2. 一次爆发的继承规则：何时抽 q，合并后如何携带恒星与爆发状态，初始
   节点已经越阈时如何处理。独立 MC 主支没有唯一决定这些规则。
3. Pop II/III 年龄依赖辐射表、逃逸率、X 射线谱/来源/延迟。单次爆发需
   与 SSP 卷积；UV 和 He II 的匹配不确定上述新增输入。

建议先用有连续身份的空间晕历史承载相同局部成星规则，并重算其 UVLF。
空间后端的 HMF/历史分布与 Reed07/McBride 的差异必须展示，不能无条件沿用
旧 UVLF 结论。这一空间模型选择尚待确定。

## 首轮验收

1. 锁定宇宙学、配置、源表哈希、初始条件种子和后端版本。
2. 逐事件核对质量/时间，合并不重复计数或丢失恒星；记录左删失事件。
3. 源网格积分与逐源总量闭合；局部/全局辐射共用形成历史。
4. 保存密度、中性比例、气体温度、自旋温度和 mK 亮温。初轮实空间计算
   显式记录未加入红移空间畸变，随后独立验证速度处理。
5. 配对运行使用相同初始条件和空间晕；关闭 Pop III 时恢复 Pop II 基线。
6. 检验盒子和时间/空间分辨率，地图与全局平均、功率谱共同审阅。

新增科学批次遵循 SLURM cp6 约束，不使用 debug 分区，不增加独立排队预检。
本次没有提交计算。审阅材料：`slides/21cm_map/archive/full_51pages_20260912/21cm_map.pdf`。

## 独立 Pop III 逃逸率的同阶段切片

`scripts/plot/plot_escape_matched_fields.py` 从已完成运行的完整电离场中选取
体积平均电离度首次向上跨过 0.25、0.50、0.75 的邻接快照，展示最近的真实
快照及其实际红移、电离度；三个目标仅是展示阶段，不是物理判据。固定同一
共动坐标切片和 0–1 色标，不对电离场插值、重标定或二值化。

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_escape_matched_fields.py \
  --plan configs/experiments/popiii_escape_scan_20260918.json \
  --output outputs/popiii_escape_scan_20260918/matched_fields \
  --fesc 0.005 0.01 0.03 0.2
```

纯 Pop II 自动作为基线；显式指定的各档必须已经完成。输出包括各阶段切片、
50% 附近的差值图和另一侧邻接快照、提取切片及带输入哈希的汇总。读取时核验
密度、物理和代码来源一致，完整场均值与历史一致。切片相关系数和均方根差异
仅为描述量，不能当作三维气泡尺度或连通性测量；不同红移及阶段匹配残差必须
与图一起解释。既有密度宇宙学未核实及区域重叠不严格守恒光子的限制继续适用。
