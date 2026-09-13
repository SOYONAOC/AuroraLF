# 当前随机阈值 Pop III 模型的 PISN 预测

2026-09-11。评阅入口：[PISN slides](../slides/pisn_random_q_20260911/pisn_random_q.pdf)。

本轮复用 εb=0.03 的形成历史，完成两个红移的本征事件率、前身星质量分箱、
天空发生率及 UV 亮宿主比例。尚未完成指定巡天的检出数或观测约束。

## EXP：JWST/PISN 检出预测与观测比较

2026-09-11 经用户确认登记为 **EXP（实验）**，状态：**待开展观测前向计算**。
复用 AUR-EX-0006 的 εb=0.03、logE IMF 模型；已有本征事件率作为计算基础。

目标是回答：在具体 JWST 巡天条件下，该模型预计能检出多少次 PISN，
是否与实际瞬变样本及未检出结果相容。

- 选择有来源、适用于当前前身星的时变光谱，计算真实 NIRCam 滤镜中的光变曲线。
- 核对所选巡天的面积、观测历元、曝光、差分图噪声和筛选规则；
  由这些条件确定检出效率与有效可见时间，不采用无仪器依据的统一通量门槛。
- 补齐所需红移覆盖，将本征率转换为观察者参考系的预期检出数，
  正确处理宇宙学体积和时间膨胀。
- 与同一巡天、同一选择条件的样本比较，保留分类不确定性；
  JADES 的全部超新星数不能直接当作 PISN 数，Capotauro 已排除为高红移 PISN。

完成标准：交付附观测条件与输入来源的预期检出数、观测对照图和 slides，
分别报告抽样误差、光变模型及选择效率的不确定性。具体巡天和光变输入尚待核定；
本次登记未启动新计算，也不表示模型已通过 JWST 观测检验。

## 模型与物理输入

形成模型与 UV、He II 保持一致：随机首次越阈，log10(q)~N(0.5,1.5²)，
每条历史至多一次 Pop III 爆发，Mstar=εb(Ωb/Ωm)Mh(tb)。沿用原有 Pop II
历史，不把 PISN 富集或能量反馈回写到已经生成的形成历史。

IMF 使用 Raiter, Schaerer & Fosbury (2010) Table 1 的 logE/TE：
Mc=60 Msun、σlnM=1、初始质量 1–500 Msun。
[Tumlinson (2006)](https://arxiv.org/abs/astro-ph/0507442) 给出的定义是
dN/dlnM ∝ exp[-ln²(M/Mc)/(2σ²)]，所以 dN/dM 需再除以 M。
不使用旧 PISN 脚本的 Salpeter 幂律产额。

基准 PISN 命运区间为初始质量 140–260 Msun，来自
[Heger & Woosley (2002)](https://doi.org/10.1086/338487) §3/Fig. 2：
零金属、近似无质量损失恒星的 64–133 Msun 氦核映射。它不是普适区间，
也不能由 UV SSP 的 500 Msun 上限推出。

寿命采用 [Marigo, Chiosi & Kudritzki (2003)](https://doi.org/10.1051/0004-6361:20021756)
Table 1 无自转、零金属、辐射驱动质量损失模型的氢燃烧与氦燃烧时长之和。
120、250、500 Msun 三行按 logτ–logM 插值，不外推。
[数据表与来源](../external_data/pisn/README.md)记录具体数值。
这些演化计算至核心碳点火；省略后续阶段，因此是近似爆炸延迟。
Schaerer (2002) §2 的轨道主要包含核心氢燃烧，常用寿命拟合不能直接
当作精确的完整爆炸延迟。

三个输入来自不同演化模型。基准采用经典命运区间，另以 Marigo §2.3
明确给出的 127–252 Msun 初始质量区间计算敏感性，不宣称它们是同一自洽网格。

## 计算及单位

令 ξ(M)=φ(M)/∫Mφ(M)dM，φ=dN/dM。每单位初始形成质量的总产额为
η=∫140^260 ξ(M)dM = **0.001479455 Msun⁻¹**。
这相当于每形成约 676 Msun 恒星，平均一次 PISN；10⁵ Msun 爆发平均约 148 次。

延迟核 k(a)=∫ξ(M)δ[a−τ(M)]dM。质量–寿命关系单调时实现为
ξ(M(a))|dM/da|；基准支持区间为 **2.42964–2.87489 Myr**。
k 的单位是事件 / 初始 Msun / 源参考系年，∫k(a)da=η。
只计氢燃烧的对照用同一 Marigo 表删去氦燃烧项，不是更换为 Schaerer 拟合。

对现有快照的历史加权：R(z)=Σi wi Mstar,i k[t(z)−tb,i]。
wi 为每条历史的共动数密度。各独立运行先保留各自的全局 HMF 权重计算
事件率，再等权平均运行估计；不把四批样本的权重直接累加。
MC 标准误按独立的质量样本分组，同一个质量下的 1000 条轨道不当作
1000 个独立质量抽样。误差不包括恒星演化和反馈模型误差。

源参考系率转换为天空发生率：
dN/(dtobs dz dΩ)=R(z)[dVc/(dz dΩ)]/(1+z)。
平方度换算显式使用 (π/180)²。源率体积单位为共动 Mpc³。

status=0 尚未越阈，当前没有 PISN；status=1 使用实际形成质量和年龄。
status=2 在 z_start=50 前已经越阈，不赋予虚构的爆发质量。
代码验证 t(z)−t(z_start) 大于最长寿命与诊断窗口之和，因此这批事件在
当前时刻及这些窗口内都已结束，可以严格排除其当前 PISN 贡献。

## 结果

| 量 | z=12.5 | z=14.5 |
|---|---:|---:|
| 本征率 [事件 yr_src⁻¹ cMpc⁻³] | (1.294 ± 0.117) × 10⁻⁶ | (8.361 ± 0.374) × 10⁻⁷ |
| 天空发生率 [事件 yr_obs⁻¹ deg⁻² / 单位红移] | 0.4802 ± 0.0435 | 0.2324 ± 0.0104 |
| 来自 MUV≤−20 宿主的事件率份额 | 3.76% | 0.99% |
| “一次爆发期望 PISN 数 < 1”的事件率份额 | 0% | 0.48% |

天空数值是局部微分率。只有两个红移点，不将它们拟合成完整演化曲线，
也不直接积分整个高红移范围。宿主选择使用原有 Pop II+Pop III UV 光度，
没有加入超新星本身的光。

主要结论：在当前模型中，大多数 PISN 事件率来自 UV 亮样本之外。
对 UV 亮星系的定向跟踪与无预选宿主的瞬变搜索检验的是不同样本。

## 稳定性及局限

| 相对于基准的变化 | z=12.5 | z=14.5 |
|---|---:|---:|
| MC 标准误 | 9.1% | 4.5% |
| 过去 0.1 Myr 窗口平均 | −1.4% | −0.7% |
| 过去 0.25 Myr 窗口平均 | −3.6% | −2.5% |
| 过去 0.5 Myr 窗口平均 | −5.6% | −4.1% |
| 只计氢燃烧 | +1.4% | +1.2% |
| Marigo 127–252 Msun 命运区间 | +19.1% | +17.3% |

窗口核为 [C(a)−C(a−Δt)]/Δt，C 是累计爆炸数/形成质量；这里只对同一
快照的历史做时间平均，不能据此重建过去所有红移的真实宇宙事件率。
历史时间步为 0.314、0.246 Myr；上述检查不是重新加密 MAH 后的收敛测试。
仍需检验原始越阈时间分辨率，以及更密的恒星寿命网格。

连续 IMF 计算的是均值，没有按有限质量逐颗生成恒星，不能直接产生
每个星团的整数计数概率。在本样本中，形成质量 <260 Msun 的爆发对
当前本征率贡献为零；z=14.5 仍有约 0.48% 来自预期事件数 <1 的爆发。
该比例只是诊断，不是对任意巡天 Poisson 计数误差或 IMF 随机性的完整验证。

这些结果条件于当前零金属 Pop III 形成假设。PISN 的富集可能终止后续
原初成星；此反馈尚未闭合，不能用本轮计算宣称 εb=0.03 已通过联合观测检验。

## 与 JWST 观测的连接

[STScI NIRCam Filters](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters)
version 7.0 表格（核对日期 2026-09-11）给出 F277W/F356W/F444W 枢轴波长
2.776/3.565/4.402 μm。这些滤镜在 z=12.5、14.5 主要采样静止系紫外，
必须对时间依赖光谱和实际吞吐曲线积分，不能将总光度直接当作宽带亮度。

[Kasen, Woosley & Heger (2011)](https://arxiv.org/abs/1101.3336) 计算的
不同质量与包层模型，光变可持续数百天。以静止系 300 天为例，两个红移
对应观测者约 11.1、12.7 年；一年观测间隔只有约 27、24 天静止系间隔。
PISN 能否被认作瞬变取决于亮度的变化量，而不只是是否存在亮源。

差分成像的信噪比为
|fν,b(t2)−fν,b(t1)| / sqrt(σb,1²+σb,2²−2Covb,12)。
噪声要匹配实际曝光、背景、宿主、孔径和图像处理。对候选源的信噪比、
颜色及多历元筛选，应使用对应的注入恢复效率。

完整检出数为
Ndet=∫dΩ dz [dVc/(dz dΩ)]/(1+z) ∫dM R(z,M) Tctrl,obs(z,M)。
控制时间对可能的爆炸日期积分，包含光变与选择效率；可以包含巡天开始前
爆炸的事件，不等于曝光时长。观测错误相关时保留协方差。

本轮未接入匹配当前零金属、含氢包层前身星的时间光谱库，未执行 NIRCam
ETC，未指定真实巡天的历元/差分深度与恢复效率，也未补齐连续红移率。
[Garching PISN 库](https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/data/Kozyreva/PISN/)
确有公开 STELLA 光变和 SED，但零金属裸氦核、Z=0.001 含氢模型应作为
不同包层/金属丰度情景，不能不加说明地替换基准模型。
因此目前没有给出仪器检出数或与未检出观测的似然约束。

## 复现与验收

### 2026-09-11：加入观测覆盖范围对照

已在 slides 第 7 页加入 HSC 上限与当前预测的同单位、不同红移对照图；
第 8 页说明 JADES 样本与 Capotauro 的后续排除结果。

[Moriya et al. (2021)](https://doi.org/10.3847/1538-4357/abcfc0)
§6（PDF 第 9 页）、Table 2/Fig. 8：HSC 以 g/r/i/z 观测同一 1.75 deg²
视场四季，典型深度约 26 AB mag，没有发现可信的持续超过一年 PISN。
在恒定体积率及所用明亮光变模板下，得到约 100 Gpc⁻³ yr⁻¹ 的数量级
上限；模拟检出事件典型红移为 z≈1–3。它不是逐红移分箱的测量，
论文也没有把 100 指定为严格的 95% 置信上限。

图中统一为源参考系、共动 Gpc³：模型 Mpc⁻³ 数值乘 10⁹，得到
1294±117、836±37 Gpc⁻³ yr⁻¹。HSC 标记仅画在其典型敏感红移区域，
没有延伸到 z>10；其明亮模板群体也不同于当前模型的全部 PISN。
这张图展示现有观测的覆盖范围，不能用于宣称模型符合或违反 HSC 约束。

[DeCoursey et al. (2025)](https://doi.org/10.3847/1538-4357/ad8fab)
JADES 瞬变样本的总超新星检出率不是 PISN 高红移率。
[Ferrara et al. (2026)](https://arxiv.org/html/2601.07374v1) 曾对 Capotauro
提出 z≈15 的 PISN 解释。后续 [Liu et al. (2026)](https://arxiv.org/abs/2608.07461)
（2026-08-07，§3.1.3/§6）测得其在约 3.5 年内移动 132±20 mas，以超过
6σ 排除河外源解释，支持 Y 型褐矮星。因此不能再把它列为当前待确认的
高红移 PISN 候选。此前核查遗漏了这项后续观测，已修正 slides 第 8 页。
Ferrara 的 §VI.2 CEERS 预期数量来自假定 IMF/成星率，也不是观测事件率。
本轮检索没有找到能直接约束当前两个点、具有匹配选择函数的 PISN 率测量。

观测条目与适用条件保存于
`external_data/observations/pisn_rate_constraints.json`，HSC 原文 PDF/source
及 URL/SHA-256 保存于 `external_data/literature_sources/pisn_observations/`。
图的输入摘要及单位转换记录在
`data_save/pisn_random_q_20260911/observation_context.json`。复现无需重跑历史：

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_random_q_pisn_observations.py
```

### 本征预测的复现

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_random_q_pisn.py --config configs/experiments/pisn_random_q.toml
PYTHONPATH=. .venv/bin/python -m pytest tests/test_random_q_pisn.py tests/test_random_q_heii.py tests/test_experiment_workflows.py
.venv/bin/ruff check
.venv/bin/ruff format --check
```

默认路径相对 TOML 所在目录解析。输入为相邻 AuroraLF-visbal-duty 下
`data_save/random_q_20260909/` 的 R032、R024–R027，共 1800 万条轨道。
每批要求 manifest.status=complete、产物 SHA-256 一致、q 参数一致、种子
独立、执行时 UV SSP 一致，以及形成质量/宇宙学定义的源文件哈希一致。
新 checkout 缺少这些实际文件时明确报错，不生成替代样本。

结果：`data_save/pisn_random_q_20260911/summary.json`，包含输入/代码哈希、
原始运行种子与产物哈希、全部数值、前身星质量分箱和误差；图在 PISN slides
的 assets 中。图使用向量 PDF。测试包含 IMF 质量归一化、延迟核积分守恒、
氦燃烧延迟、单位/时间膨胀和非法输入；17 项相关测试通过，Ruff 22 文件通过。
