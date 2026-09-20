# Pop III 后启动 Pop II：UVLF 与再电离对照

## 已采用的默认模型（2026-09-21）

用户确认将此启动条件落实。新混合模型计算默认使用
`configs/uvlf/popii_popiii.json`，选择 `variants=["delay0"]`：同一条主分支先发生
Pop III，随后才允许 Pop II 形成。零延迟作为最小改动基准，30 Myr 保留为有效延迟
敏感性选项；这不将瞬时富集或 30 Myr 解释为已确定的真实恢复时间。

UVLF 与电离源率共用 `transition_workers` 和该配置。入口
`scripts/run/run_popii_transition.py --kind uvlf|rates` 不再需要显式提供配置，
默认只计算选中的零延迟模型；`--plan` 可选择 30 Myr 或包含 `baseline` 的显式比较。
源表父目录的 `diagnostics.npz` 包含 `popii_quadrature_error`、`popii_censored_upper_extra`；
父级诊断的模型轴按 `popii_diagnostic_variants` 解读。UVLF 输出内也保存 `variants`，
`diagnostic` 列按 manifest 的 `diagnostic_columns` 解读；`paired_difference`
相对于选中的第一组，不自动引入旧模型。

提交入口改为接受 `--release <唯一名称>` 和可选 `--plan`，默认 dry run，
`--prepare` 冻结代码/配置/SSP，`--apply` 核对冻结配置后提交两个 cp6 作业。
没有重复提交已完成的本轮数值计算。配置的 `adoption` 记录已验证的零延迟
UVLF、源表、空间历史路径和源表 manifest 哈希。
`prepare_reionization_calibration.py --population popii_popiii` 默认使用该源表，
并检查选择、源参数、网格与 SSP 一致；改参数后必须显式提供相应新源表。

旧随机 q / 电离率实验入口和带日期的配置保留历史语义，不能作为新混合模型默认
入口。纯 Pop II 的 UVLF-v2 对照没有 Pop III 历史，保持原定义。共同逃逸率 0.064
保持不变；本轮采用默认条件没有重新做参数拟合。

落实验证：全套 1258 项测试通过（413.29 s）；最终默认路由与源表延伸的 19 项
针对性测试通过。源表延伸合并现在显式拒绝混用独立启动、零延迟和 30 Myr 模型。
采用真实 SSP、z=8、1e9 Msun 的 4 条历史，与冻结版本逐项比较三组 UVLF、
源率均值/标准误/协方差及边界诊断，全部精确一致。
混合模型两个 `--validate-only` 与 cp6 提交 dry run 均通过，没有新提交生产作业。
记录位于 `outputs/popii_onset_adoption_20260921/`。本次涉及代码的 Ruff 检查通过；
全仓扫描仍有独立 `miniquench_reproduction` 项目及已有 He II/PISN 分析脚本的
格式问题，保留其现状。幻灯片新增第 9 页默认约定，已更新原 Zotero AUR-S06 条目。

2026-09-20 用户要求同时计算 UVLF 和再电离，并明确选择零延迟与有文献依据的
非零延迟一起比较。本轮固定 MAH、Pop III 随机阈值、效率、SSP 和再电离逃逸率，
比较原模型、有效延迟 0 Myr、有效延迟 30 Myr。配置为
`configs/experiments/popii_transition_20260920.json`。

## 物理定义与范围

沿同一条主分支，Pop III 首次穿越事件的时间为 tIII，原模型的延迟 SFR 为
SFRII,0。新模型仅在恒星出生时间施加条件：

\[
\mathrm{SFR}_{\rm II}(t\mid q)=\mathrm{SFR}_{\rm II,0}(t)
\Theta[t-t_{\rm III}(q)-\Delta t].
\]

原模型的原子冷却条件仍在 SFRII,0 中；没有重设宇宙时间、搬移暗晕历史、
补偿被截断的恒星质量或增加后期 SFR。这里是启动条件的受控试验，尚未包含
气体储库、SN 能量、金属产额/混合、外部富集或合并祖先分支。
假定一次 Pop III 事件足以成功富集。没有触发事件的晕不形成 Pop II。

[Magg et al. 2022, §2.2–2.3](https://arxiv.org/html/2110.15948)
采用 SN 后恢复时间 10/30/100 Myr，并单独处理恒星寿命；
[Hegde & Furlanetto 2023, §5.2](https://arxiv.org/html/2304.03358#S5.SS2)
讨论其 30 Myr 中间情形。因此本轮选择 30 Myr 作为有文献动机的**有效成星后延迟**。
它并非原论文 SN 后 30 Myr 的逐项复现，也不是观测确定的普适时间尺度。
零延迟是瞬时过渡极限，不表示真实恒星能立即释放金属。

## 数值与抽样

使用完整 MAH 的运行最大值确定首次穿越，不允许后期重复穿越重新触发。
UVLF 用同一 q 同时控制 Pop III 和 Pop II；按事件年龄 0/3/10/30/100/130 Myr
划分条件抽样，再将旧事件、起始前事件和未触发事件分别计权。
八个互斥分层的概率之和为 1。Pop III 原来的 100 Myr UV 窗口不变。
总 UVLF 对同一晕的 LII+LIII 分箱，不能相加两个分量的 LF。

Pop II 的 SSP 积分在真实启动时间截断，避免在时间格点之间漏出提前成星。
积分单位为 (Msun/yr) × (observable/Msun) × dt[Gyr] × 1e9。
每个格间采用 16 点 Gauss 积分，与原电离源积分相同；UV 基线也使用此积分，
以便把积分差异与启动条件差异分开。UV 波长为 1500 A；Pop II 为恒星 BPASS，
Pop III 沿用对应 IMF 的 .25 表（包含原表星云连续谱）；先输出本征 UVLF。

再电离源率对 Normal(log10 q) 解析计算过渡概率，并在出生时间积分中使用，
不使用平均光度构造 UVLF。保留 MAH 抽样误差、完整通道协方差，以及 8/16 点
积分差异。源表为相同 242 个红移 × 111 个质量节点，覆盖 z=5–49。
两个星族的电离 SSP 窗口均沿用当前比较的 100 Myr。

起始前事件没有虚构的精确爆发日期：在 t_start+delay 后，其过渡状态已知；
更早时对 Pop II 分别保存保守下界和允许最早宇宙出生的上界。空间主计算采用
下界，源表附加诊断报告这一不确定度。UV 供光窗口距历史起点超过 130 Myr，
因此此处的起始前事件过渡状态确定。

## 再电离比较

先固定共同 fesc=0.064，与现有延伸到 z=5.01 的混合模型配对比较，两个通道
始终共同缩放。纯 Pop II 原校准可以作为参考，但本轮不重拟合逃逸率，也不将
新结果称为重新校准。现有独立 fescIII=0.3/0.5 试验是不同对照，不混作基线。

新源表需要完整重跑空间演化，不能把源率变化直接乘到旧电离历史上。
光深使用真实密度加权电离比例；只在末帧确实完全电离时，才能把更低红移设为
完全电离。若延迟模型到 z=5.01 仍未完成，报告已模拟区间的光深和明确的
补全敏感性，不把补全当作真实低红移计算。

## 执行与验证

入口：`scripts/run/run_popii_transition.py --plan configs/experiments/popii_transition_20260920.json --kind uvlf|rates`。
必须在非 debug SLURM 分配内运行；`--validate-only` 只验证真实配置和 SSP。
`submit_popii_transition.py` 将输入冻结，再分别提交两个 cp6 56 核任务，无独立
预检作业、无时间/内存请求。用户本轮要求两者同步，故这两个计算独立调度。

输出根目录为 `data_save/popii_transition_20260920/`。`rates/{baseline,delay0,delay30}`
是可直接输入现有空间适配器的 `instantaneous-rates-v1`，保持三个既有通道和单位。
`rates/diagnostics.npz` 单独保存 Pop II 积分误差和起始前事件的附加上界。
`uvlf/z*.npz` 保存三种模型、三类光度函数（II、III、总和）、每质量样本贡献和
配对差值误差。所有输入、代码和产品均记录 SHA256。

解析测试覆盖恒定/线性 SFR 的精确截断、非单调历史、未触发事件、起始前事件
上下界、Normal CDF 独立积分和概率守恒。真实 4 条历史的局部回归已验证新
源表基线与现有电离入口的均值、标准误和协方差一致至 1e-13 相对误差。
全量源表也已验证基线均值、标准误和协方差与旧入口一致，最大相对差异分别为
6.10e-10、6.10e-10 和 1.22e-9；Pop III 通道不变。

2026-09-20 执行记录：全套 1245 项测试通过（268.38 s），本轮核心文件与入口
Ruff/格式检查通过。两个 cp6 作业为 UVLF 11713297、源表 11713298，均已完成，
计算分别耗时 2243 s、4067 s，产品 SHA256 核验通过。测试日志、冻结快照与提交记录保存在
`outputs/popii_transition_20260920/` 和 `outputs/deployments/popii-transition-20260920-01/`。

## 源表与 UVLF 结果

HMF 加权的 Pop II 源率 8/16 点积分差异最大为 0.03834%；未知起始前事件的
Pop II 上下界差异，占总源率的最大比例为 4.519e-7。极少高质量节点的 30 Myr
源率比基线高至 7.352e-5（相对量），属于不同积分分段带来的数值差异，低于
上述积分误差；没有为强制单调而裁剪结果。

30 Myr 模型相对于原模型的光度密度比值如下。它们来自完整采样质量范围
1e4–1e15 Msun 的全部光度，不等于任何观测星等切割下的比值。

| z | Pop II 1500 A | 总 1500 A |
|---|---:|---:|
| 6 | 0.9282 | 0.9332 |
| 8 | 0.8858 | 0.9109 |
| 10 | 0.8313 | 0.9088 |
| 12.5 | 0.7486 | 0.9398 |
| 14.5 | 0.6950 | 0.9652 |

在 z=14.5，高红移总光度主要来自本模型的 Pop III，因此 Pop II 光度减少约
30.5%，总光度仅减少约 3.5%。总 UVLF 的比值随星等改变，不能直接用该总光度
比值缩放光度函数。`uvlf_summary.json` 保存配对抽样标准误；它不是模型系统误差。

后处理入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/analysis/validate_transition_sources.py
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_popii_transition.py --stage uvlf
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_popii_transition.py --stage spatial
PYTHONPATH=. .venv/bin/python scripts/plot/plot_popii_transition.py --stage uvlf
PYTHONPATH=. .venv/bin/python scripts/plot/plot_popii_transition.py --stage spatial
```

## 空间再电离结果

两个本地 GPU 作业 160230（零延迟）、160231（30 Myr）均完成，耗时分别为
17 min 36 s、18 min 27 s。三组共享空间网格、全部密度场哈希、求解器代码、
源率以外的参数及共同 fesc=0.064；均输出 235 个真实空间快照，至 z=5.01
完全电离。每组在约 z=15、10、7、5.01 的三维场均通过体积/密度加权均值检查。

| 量 | 原模型 | 零延迟 | 30 Myr |
|---|---:|---:|---:|
| 体积平均 xHI(z=7) | 0.610486 | 0.629802 | 0.630742 |
| z50 | 6.577665 | 6.479530 | 6.474962 |
| z99 | 5.643432 | 5.547018 | 5.543565 |
| tau_e | 0.0594364 | 0.0586111 | 0.0585656 |

z50/z99 定义为体积平均电离比例首次达到 50%/99% 的红移，由相邻输出插值。
tau_e 使用实际密度加权电离比例，并沿用原模型的氦电离和低红移完全电离条件。
步长 0.002 与 0.001 的光深积分比较通过 1e-7 的数值容差。

补上因果启动条件使 z99 降低约 0.10、tau_e 降低约 8.7e-4；额外 30 Myr
相对零延迟使 z99 再降低约 0.00345、tau_e 再降低约 4.55e-5。
这表示本轮参数下主要影响来自“必须先触发 Pop III”，而不是 0/30 Myr 的选择。
这些是固定逃逸率的模型差值，不能解释为重新校准后的观测拟合优劣。
额外延迟造成的 z99 差值小于晚期快照间隔 0.02；尚未针对该小差值单独做
空间时间步长收敛测试，不能将插值后的多位小数视为已验证的物理精度。

空间绝对预测仍继承旧模型的限制：密度场宇宙学来源尚未独立验证，重叠
excursion 电离区域不保证严格光子守恒。全部结果及误差定义保存在
`spatial_summary.json`、`spatial_history.npz`。幻灯片为
`slides/popii_transition_20260920/transition.pdf`；图为 vector PDF，未纳入 Zotero 独立文档。
后处理的三个配对误差极限测试另行通过；不重复运行已通过的完整模型测试。
