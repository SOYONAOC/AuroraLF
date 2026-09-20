# 氢电离光子源接口

> 2026-09-21 合并说明：新混合模型的默认入口为
> `configs/uvlf/popii_popiii.json` / `run_popii_transition.py`，采用 Pop III 后
> 零有效延迟启动 Pop II、两类源均 100 Myr 供光窗口。下文记录此前独立启动的
> 实验配置和结果，不替代新默认；原始数值、路径及输入哈希保留。
> 当前模型与验证结果见 [Pop II 过渡说明](popii-transition.md)。

历史独立启动入口默认 `max_lookback_myr=100.0`（Pop II）与
`popiii_max_age_myr=6.0`（Pop III）。跨越年龄边界的积分段在出生时间处精确
裁剪；不是将整个时间格点简单删除。晕组装、延迟 SFR 和首次越阈记录仍追踪到
z=50，避免重新触发窗口开始以前已经发生的 Pop III 爆发。
起点前未知 Pop III 事件的额外上界也限定在允许年龄与 `[0, 6 Myr]` 的交集；交集为空
时为零。已有 IGM 电离状态不随年龄窗口删除，继续演化和复合。

瞬时配置类型为 `auroralf.experiments.ionizing_rates.RateConfig`；其
`max_lookback_myr` 和 `popiii_max_age_myr` 均必须为有限正数，分别默认 100 与 6 Myr。
`max_lookback_myr` 现在只控制 Pop II；重现原两类 100 Myr 方案时须显式设置
`popiii_max_age_myr=100.0`。低层
`rate_from_sfh(..., max_age_myr=None)` 与
`threshold_averaged_rate(..., max_age_myr=None)` 保留未截断的历史诊断调用；
生产 `rate_cell` 显式传入配置中的年龄上限。
源 manifest 在 `resolved_model` 中记录两个设置；`max_stellar_age_myr` 改为
`{"popii": 100.0, "popiii": 6.0}`，不再用单一数值表示两类源。
新生产输出目录为 `data_save/ionizing_sources/instantaneous_popiii6myr_v1/`；
`instantaneous_100myr_v1/` 为历史双 100 Myr 结果，`instantaneous_v1/` 为未截断结果。
本次更新生产设置，尚未重新生成完整源表或地图。

6 Myr 的依据：10 个红移、111 个质量点的配对检查中，Reed07 加权的 Pop III
当前供光相对 100 Myr 最大减少 0.6531%；该结论不等同于空间电离场收敛。
数据与判据见 [收敛记录](21cm-map.md)。

2026-09-12：21 cm 生产源改为 `auroralf.experiments.ionizing_rates` 和
`scripts/run/build_ionizing_rates.py --config configs/experiments/ionizing_rates.toml`。
两类恒星都用当前年龄的 SSP 产光率；输出 `mean_rate`、`se_rate` 和
`rate_covariance_of_mean`，单位 escaped photons/s/halo。
`quadrature_error` 记录 Pop III 阈值积分 16 点与 8 点的平均绝对差。
源表单位标记 `instantaneous-rates-v1`，不与历史累计表混用。

Pop III 随机 log10(q) 的物理分布不变；对每条 MAH 的首次越阈记录区间积分，
求取阈值分布的期望，减少罕见年轻爆发带来的抽样噪声。仍保留 MAH 抽样误差。
起点前事件上界使用允许年龄区间内的最大**瞬时**光子率，恒星变老后同样衰减。
没有设硬性的 3 Myr 截断；Pop II 则积分出生 SFR 乘当前年龄的 SSP 率。

可用 `--shard-count N --shard-index i` 在多节点计算同一网格互不重叠的点；
分片格式不能被空间适配器当作完整源表读取。全部完成后，以相同配置执行
`--shard-count N --merge-shards`，检查代码/输入哈希、无重复且完整覆盖后合并。
拆分不会修改随机种子或重复独立科学批次。源表为 CPU float64，空间接口
先除以 `1e50` 再转为 GPU float32。以下累计接口和产物仅保留作历史对照。

原 `codex/popiii-21cm-map` 实验保留当前随机首次越阈 Pop III 和延迟 Pop II
成星模型。`auroralf.ssp.ionizing` 将同一恒星模型转为氢电离光子率和累计产额；
`auroralf.experiments.ionizing_sources` 生成固定终态晕质量、红移下的源预算。

## 恒星模型、单位和时间积分

- Pop II：当前 BPASS binary `imf135_300.BASEL.z001.a+00` 恒星谱。
  按 1 Å 光谱格点，对 1–911 Å 的 `L_lambda / (hc/lambda)` 求和，除以初始
  星团质量 `1e6 Msun`。使用 BPASS 手册的 `Lsun=3.848e33 erg/s`；旧 UV
  loader 使用 Astropy 的名义太阳光度，两种约定相差约 0.52%，没有顺带修改旧 UV。
- Pop III：与当前 UV/He II 一致的 `pop3_ge0_logE_500_001_is5.20`，第三列
  `log(Q_0)`。其与 `.22` 第二列逐行一致。初始质量为 1 Msun；年龄网格沿用
  现有 is5 重建 `0.01,1,...,1000 Myr`，并核对原表打印年龄。
- `IonizingKernel.rate(age_myr)`：本征氢电离光子率，单位 `s^-1 Msun^-1`。
- `IonizingKernel.yield_photons(age_myr)`：从出生至该年龄的本征累计光子数，
  单位 `photons/Msun`。精确积分分段 log-age 线性的光子率。首个年龄以前
  显式保持最年轻 SSP 光子率，超过末端年龄报错。
- `cumulative_photons_from_sfh`：积分 `SFR(t_birth) Y(t_obs-t_birth)`，
  输入时间 Gyr、SFR Msun/yr；每个线性 SFR 区间用 8 点 Gauss 积分。
  无旧 UV 的 100 Myr 截断。此式交换两重积分次序，避免逐发光时刻重复卷积。

光谱单位来源：[BPASS 手册](https://bpass.auckland.ac.nz/8/files/bpassv2_1_manual_accessible_version.pdf)，
[v2.3 数据说明](https://bpass.auckland.ac.nz/14.html)。Pop III：Raiter,
Schaerer & Fosbury (2010), [arXiv:1008.2114](https://arxiv.org/abs/1008.2114)。

## 生成与输出

从仓库根目录使用 `.venv/bin/python scripts/run/build_ionizing_sources.py
--config configs/experiments/ionizing_sources.toml --validate-only` 验证输入。
实际计算必须通过 dmde-compute 提交至非 debug 的 SLURM 分配；程序使用分配内
CPU 数作为进程数，每进程 BLAS 单线程。它是本地源预算研究，未改写现有
cp6 UVLF 生产入口。

`data_save/ionizing_sources/mainbranch_v1/sources.npz` 保存：

- 升序 `redshifts`、`mass_msun`，以及 `(n_z,n_mass,3)` 的 `mean_photons`、
  `se_photons`。三通道为 Pop II、已解析 Pop III、起点前 Pop III 的额外上界。
- `half_means_photons`：两半样本各自均值；`status_counts`：未爆发、已解析、
  左删失事件数；`initial_popii_active_count`：起点已允许 Pop II 成星的历史数。
- `manifest.json`：配置、源代码/SSP/产物 SHA256、SLURM 作业及状态。

输出已经乘各自的逃逸率，单位为物理累计逃逸光子数/晕，CPU float64。
均值包含所有历史的零贡献；不附加 duty cycle、不含 HMF 权重。
SmallScale21cm 的积分器在 CPU 上将这些量除以 `1e50` 后再传到 float32 GPU。

## 当前闭合及限制

首轮采用独立终态晕的主支积分；没有完整合并树和次支系恒星。起点以前的
Pop II 未重建，起点活动数量单独记录，因此不能将此表称为完整宇宙累计光子史。

Pop III 左删失事件没有虚构的爆发时刻。已解析贡献是此主支在已解析时间段内
的贡献；额外上界使用 `Mburst <= q Mcool(z_start)` 和最大可能 SSP 年龄。
这个不等式来自原子冷却质量向高红移降低及连续首次越阈条件，适用于本模型。
上界只约束这些历史的左删失 Pop III，不约束遗漏支系、源表质量下界之外的晕，
也不代表整个物理模型的严格误差条。

首轮范围 `1e4–1e15 Msun` 用于质量积分支持和边界诊断；它包含当前 UVLF
实验范围以外的模型外推，尤其不能声称低质量 McBride 历史已被模拟验证。
首轮 256 条/质量点的样本与加密检查必须区分，结果需附采样误差和边界敏感性。
