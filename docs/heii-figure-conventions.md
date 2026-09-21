# He II 文献画法与本工作（2026-09-16）

## 当前讲稿：−20 等且年龄 ≤3 Myr 的年轻样本（2026-09-18）

按用户最新要求，正文 4.2 采用 `MUV<=-20`、爆发年龄 `<=3 Myr` 的模型选样，
五个红移统一采用这一条件，不再施加成像或谱线通量限。蓝色区间为 HMF 加权
16–84% 对象分布；该星等切选是模型选择，不表示文献检测标准或逐目标匹配。
数据与复现命令见 [He II 选样记录](heii-survey-depth.md)。

## 上一版：文献深度选样（保留为附录对照）

用户指定采用 Vikaeus+22 表 1 的 30.6 AB 成像深度和 `2.9e-19 erg/s/cm²`
谱线限。正文 4.2 改为灰色连续谱母样本、蓝色再过谱线限的子样本、虚线为无透镜
参考线限；使用五个实际红移的 1500 Å 样本重新后处理。详细结果、限制和命令见
[文献参考深度选样](heii-survey-depth.md)。附录 A.3/A.4 的原目标窗口统计仅作备查。

## 历史中间版：撤下无依据的统一切选（2026-09-18）

主讲稿 4.2 不再显示 `MUV<=-20、年龄<=3 Myr` 的绿色模型点；这两个切选
不能作为各观测目标的共同选择函数。LAP1 和 GN-z11 保留观测参考，未构造匹配模型。
RXJ2129-A、GHZ2、GS-z14-1 使用已有真实目标红移的 UV 窗口结果，未重跑科学模型。
`±0.25 mag` 明确标作自定探索窗口，并非文献测量误差，宽度敏感性尚未验证。
RXJ2129-A 的第 16 百分位低于纵轴显示范围，用向下箭头标注；原始分位数完整保存在
`outputs/heii_combined_20260917/provenance.json`，未伪造一个有限下界。
总结页同时撤下基于 `MUV<=-20` 和自定参考通量的 He II 超阈比例。

复现该历史中间版图：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_heii_combined_redshifts.py \
  --low-sample-dir threshold_zero_all_20260918/heii_low \
  --high-sample-dir threshold_zero_all_20260918/exact
```

以下为历史图形与文献核对记录，不代表当前讲稿采用这些切选。

本次核对原论文 PDF 图注并查看原图，选择可由现有真实结果支持的图形。
借鉴坐标与比较方式，不把本工作的样本或统计量称作论文结果的复现。

| 原文与图号 | 图中实际展示的量 | 本工作处理 |
| --- | --- | --- |
| [Venditti+24，图2](https://arxiv.org/html/2405.10940v2#S3.F2) | He II 本征光度与红移；平均 Pop III 质量，效率和质量损失假设形成色带；观测候选体及 NIRSpec 灵敏度 | 采用光度—红移坐标，叠加原始测量和上限。我们的区间仍为固定效率下的丰度加权 16–84% 对象分布。 |
| [Trussler+23，图11](https://academic.oup.com/mnras/article/525/4/5328/7251489) | 瞬时爆发后 He II 等值宽度、光度、He II/Hβ 随年龄变化；固定初始恒星质量，光度面板带 z=8 的灵敏度 | 适合检验 SSP 与年轻爆发窗口。当前这组产品未提供匹配的 Hβ 和总 1640 Å 连续谱，不用 UV 代理量冒充 EW；本次未生成此类模型图。 |
| [Vikaeus+22，图3](https://arxiv.org/abs/2107.01230) | 指定巡天及透镜概率下，探测 He II 所需最低数密度与红移，比较不同成星效率 | 需要透镜放大概率、有效巡天体积及指定灵敏度；不能把已有本征累计数密度直接改名为此量。本次未生成此类模型图。 |

原文来自 `external_data/literature_sources/heii_observability/`：
HEII-OBS-001 Venditti24、002 Trussler23、003 Vikaeus22。
所用 PDF 页码分别为 6、19、9；Vikaeus 的图3是第9页右上图。
原图预览和图注核对产物在 `outputs/heii_literature_figures_20260916/`。

## 讲稿调整

`slides/popiii_heii_pisn_complete_20260916/popiii_heii_pisn.pdf` 仍为 AUR-S01。
第8页保留 V24 图2原图。2026-09-17 按用户要求合并展示：第9页在同一坐标下
叠加五个红移的 epsilon=0.03 模型与观测；低红移 epsilon=0.1 独立页也已按用户要求删除。
删除两张 epsilon=0.01 页及独立低红移 epsilon=0.03 页，当时讲稿共20页，新增独立模型色带页后为21页。
合并图由 `scripts/plot/plot_heii_combined_redshifts.py` 读取两套已完成摘要生成，
没有重新计算或更改科学选择。绿色菱形为年轻UV亮样本（MUV<=-20，年龄<=3 Myr），
蓝圆为两个高红移目标各自的UV窗口，不施加年龄筛选；不能解释为同一母样本的演化。
五个观测系统保留七条测量记录；GN-z11另外两种孔径用紫色星号区分。
LAP1的1sigma上限与GS-z14-1的3sigma上限分别标注。摘要哈希及所绘数值记录于
`outputs/heii_combined_20260917/provenance.json`。
以下为此前分效率作图的历史实现说明；用户删除的页面不恢复。

### 2026-09-17：独立展示本模型的对象分布

用户明确要求保留 epsilon=0.03 的原误差棒图，因此第9页恢复为
`heii_combined_eps003.pdf`，该图文件未经改写。第10页新增
`heii_model_population_band.pdf`，由 `scripts/plot/plot_heii_model_band.py` 生成。
色带为固定 epsilon=0.03 下丰度加权的16–84%对象分布，实线和圆点为中位数；
与前页数值完全相同，不表示效率扫描范围或中位数的统计误差。

低红移三点采用同一年轻UV亮样本选择，分位数间以logL对z线性插值作视觉引导，
未计算新的红移点。高红移两个UV选择窗口不同，分别显示窄色块；色块宽度仅为
显示需要，不是红移误差、红移分箱或连续演化预测。低红移有效质量样本仍仅约5–11。
原始模型、观测、选样均未改动，未重算形成历史。

新图叠加V24的R≈1000、50小时、积分信噪比≈5的IFU/MOS探测阈值，
虚线为500 km/s、点线为50 km/s，不使用V24的模型色带。
仪器阈值来自原图矢量路径，只在原图端点之间展示，不外推至z=12–14；
透镜目标需另行计算检出条件。这些线不是新的ETC计算。

输入哈希、所绘分位数及作图假设记录于
`outputs/heii_model_band_20260917/provenance.json`。
提取源文件为 `external_data/literature_sources/heii_observability/venditti24_fig2_vectors.json`，
完整24条V24阈值及其原文色带的独立复现保留于
`outputs/heii_v24_overlay_20260917/v24_all_thresholds.pdf`，不代表我们的模型。
此前误将V24色带叠加到主图的资产移至outputs保存，不再用于讲稿。

### 统一选样的红移扩展（2026-09-18已完成）

用户要求继续将独立色带延伸至高红移。统一使用本征M1500<=-20、已解析爆发
年龄<=3 Myr、epsilon_b=0.03；复用6.639、8.1623、10.6的原始年轻UV亮样本。
原高红移目标样本的Pop II UV选样采用1600 Å，不能直接拼接，因此在1500 Å下
新增9.5、11.5、12.342、13、13.86、14.5六个红移，每点3600质量×1000条历史。
维持低红移样本的质量范围10^4–10^15 Msun、960时间网格及所有物理参数。
低红移三个旧点的有限采样问题仍然保留，新增红移不构成其收敛验证。

计划：`configs/experiments/heii_uniform_band_20260917.json`；
SLURM作业159307，node5/cpu，3核，不独占；完成/失败通知发至agent邮箱，
同时注册现有任务的事件监听。结果目标为`data_save/heii_uniform_band_20260917/`。
`build_heii_v24_targets.py --data-only`仅产出样本、摘要、哈希及源代码归档，
不会重建已经退役的旧讲稿；`--validate-only`只检查输入，不生成历史。

生成命令为
`PYTHONPATH=. .venv/bin/python scripts/plot/plot_heii_model_band.py --uniform-sample-dir heii_uniform_band_20260917`。
新分支核验物理配置及摘要完成标记后生成`heii_model_uniform_band.pdf`，
仍只将V24仪器阈值画在其原有红移范围。检查结果、有效样本和PDF排版后，
已替换当前讲稿4.4节“模型的中位数与分布”（PDF第14页）；4.3节原始epsilon=0.03
误差棒图的PDF资产逐字节不变。保留任务期间新增的章节页和样式，当前讲稿共42页。

作业用时15:07:51，六个新增红移入选602–1505个对象，有效质量簇数81.4–112.6。
全部数据产品和计划哈希已校验；分位数为正且有序。低红移旧点仍只有5–11个
有效质量簇，尤其z=10.6附近的中位数起伏不应直接解释成物理变化。观测点仅作
光度尺度对照，没有与统一的年轻UV亮选样逐目标匹配。当前图使用9个实际计算红移，
中间连线只是logL对z的线性视觉引导。审阅记录见
`outputs/heii_uniform_band_20260918/review.json`，数值和输入溯源见
`outputs/heii_model_band_20260917/provenance-uniform.json`。

### 历史实现

高红移新图由 `scripts/plot/plot_heii_literature_comparison.py` 生成，读取
`data_save/heii_exact_targets_20260916/summary.json`，并核验该摘要绑定的计划哈希。
SLURM 作业159041在两个实际观测红移重新生成形成历史和HMF权重，每个红移
3600个质量样本、每质量1000条历史；保持原先高红移物理参数，无重新拟合。

- 用已保存的 `flux_per_luminosity = mu/(4*pi*D_L^2)` 逆变换恢复本征光度；
  GHZ2 取 mu=1.3，GS-z14-1 取 mu=1。没有重复去透镜。
- 模型与观测均位于12.342和13.86，不做横向错位或连线。
- 两个模型点各自采用目标 UV 星等 ±0.25 mag 的选择窗口，不施加年轻年龄切选；
  窗口不是观测误差，也不是一个统一 UV 样本的红移演化轨迹。
- 模型误差棒是对象分布，观测误差棒是测量误差；GS-z14-1 是3σ上限，箭头长度仅为显示。
- 高红移现展示 epsilon=0.01、0.03；epsilon=0.1计算产物保留，每档均重新选择总UV窗口。
- 三图使用相同纵轴范围1e40–1e43 erg/s。epsilon=0.01时GHZ2的16%分位数为1.03e21，
  用蓝色向下延伸箭头和数值明确标注图外下端；这不是观测上限，统计中未删除暗源。
- 年龄插值差异从主图移除，完整数值仍在原始摘要和 `docs/heii-observation-comparison.md`。
- 后续三档低红移效率图仍采用年轻、UV亮样本，口径不同，不能与高红移两点拼接成一条演化曲线。
- 文献中的 NIRSpec 灵敏度依赖模式、曝光、线宽等条件，只展示在署名原图中，未移植为我们的检出率。

原始高红移 UV 条件分布资产由历史讲稿 `slides/heii_random_q_20260910/` 保留。
新图的数值、输入哈希和假设记录于
`outputs/heii_literature_figures_20260916/heii_highz_eps001-provenance.json`
（另有003、010两档）。旧近邻红移图及其溯源保留在outputs中的历史预览。
