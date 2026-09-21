# Pop III 单次爆发总恒星质量的分布

核查日期：2026-09-19。对象是一次事件出生的全部 Pop III 恒星质量之和，非单颗恒星 IMF，也非宇宙累计恒星质量。没有修改 AuroraLF 科学模型，没有运行 21cmFAST。

## 文献证据

- **Hazlett, Kulkarni, Visbal & Wise，ApJ 978, 13 (2025)**，DOI 10.3847/1538-4357/ad919e。具体核查版本为 [arXiv:2403.05624v1](https://arxiv.org/html/2403.05624v1)，§3.3.2、Table 1。首次 Pop III 成星事件的总质量近似对数正态，中心约 150 Msun，log10 标准差 0.371 dex；与宿主晕质量、红移相关性很弱。此为 Renaissance 模拟校准，非观测测量，且低质量 minihalo 未充分解析。不能把它解释为所有红移、环境下的通用规律。
- **Hazlett et al.，Aeos + Renaissance 后续校准**：具体核查 [arXiv:2510.11629v1](https://arxiv.org/html/2510.11629v1)，§II.1.2、III.2.1、Eq. (5)。ADS 元数据现列 ApJ 1009, 3 (2026)，DOI 10.3847/1538-4357/ae9a78；本次未核查最终期刊版本。Pop III 改为只校准 Aeos 的 216 个首次成星晕。事件质量由成星单元数与逐颗恒星抽样构造，每单元约 100 Msun，因此产生约 100 Msun 倍数的聚集；不再直接抽取一个光滑的对数正态爆发质量。100 Msun 是成星算法尺度，不能称为自然界已确定的质量量子。Eq. (5) 印为 `F(x)=2.29 exp(-1.46x)+0.06 exp(-0.24x)`，正文称其为 normalized cumulative distribution function，但该表达式递减，不能直接当作通常递增 CDF 求逆；本次未擅自重新解释或采样它。
- **Liu et al.，MNRAS 534, 290 (2024)**，DOI 10.1093/mnras/stae2066，[arXiv:2407.14294v2](https://arxiv.org/html/2407.14294v2)，§3.1、Fig. 9 上图：类星体祖先晕中的星团总质量分布可有多峰和高质量分支，单一对数正态不能概括所有情形。目标是高密度偏置环境；图中 cosmic average 仅对相对流速分布平均，并未解除类星体祖先晕的环境选择。下图是单颗恒星 IMF，本次只用上图。粗灰虚线 Hirano+2015 也是单星参考，不可混作星团数据。

## 对数正态与偏斜

对数正态意味着 log10(M) 正态，而 M 本身右偏；二者不是互斥选项。图按 Hazlett 的中心为中位数的解释重建解析分布，非原始模拟直方图、非新拟合。使用 dP/dlog10(M) 与 dP/dM 的 Jacobian：`p_M = p_log / (M ln(10))`。统计值见 `outputs/popiii_burst_mass/fit_reconstruction.json`。

## 当前代码的含义

`auroralf/experiments/random_q.py:draw_logq,first_crossing,burst_light` 给出随机阈值首次穿越及爆发质量对应的光度：`M_burst = epsilon_b * fb * Mh(t_b)`。阈值 `q=Mcrit/Matomic` 的 log10 宽度为 1.5 dex；`configs/experiments/heii_threshold_zero.toml` 和最新 threshold_zero_all 配置的均值为 0，部分旧配置仍为 0.5。本次不把旧配置均值冒充当前值。

固定红移与效率时，输入阈值映射的 log10 质量宽度等于 1.5 dex，比 0.371 dex 宽约 4 倍，但对象分别是阈值先验映射与事件样本拟合。真实事件分布还受首次穿越、不同出生红移、晕质量函数、年龄选择和未发生事件影响，不能直接认定输出严格对数正态，也不能直接用 0.371 替换 1.5。没有重算或拟合当前事件样本。

## 21cmFAST 源码核查

本地固定版本 v4.2，commit `2cb6000d61381c658ccbe68028c74ca6b0c46cdc`，源码树干净，项目环境未安装 py21cmfast。

`src/py21cmfast/src/scaling_relations.c:get_halo_stellarmass` 在启用 USE_MINI_HALOS 时计算

```
fmini = min(1, f7 * (Mh/1e7)^alpha_mini
              * exp(-Mturn_mini/Mh - Mh/Matomic + G*s - A))
Mstar_mini = fb * Mh * fmini
s = ln(10) * SIGMA_STAR
A = 0 (median convention) or s^2/2 (mean convention)
```

G 为标准正态变量。固定晕质量与局部阈值，非零散布是预设的、受上限裁切的对数正态。`get_halo_sfr` 再用 `Mstar_mini/(t_star*t_H)` 并施加可选 SFR 散布。晕样本总体是质量函数与反馈加权后的混合分布；原生字段不是带起止时刻、金属终止过程的单次 Pop III 爆发质量。任取 SFR × 时间窗会引入一个额外事件定义。本次因此用源码判断适用性，没有为验证输入分布而运行盒子。

公开定位：[源码](https://github.com/21cmfast/21cmFAST/blob/2cb6000d61381c658ccbe68028c74ca6b0c46cdc/src/py21cmfast/src/scaling_relations.c#L310)、[官方 halo sampler 教程](https://21cmfast.readthedocs.io/en/stable/tutorials/halosampler.html)。

## 产物与限制

Slides：`slides/popiii_burst_mass/popiii_burst_mass.pdf`。图形重建脚本：`scripts/plot/plot_popiii_burst_literature.py`。未修改生产参数或运行昂贵模拟。

验证：解析分布两种变量测度下积分归一化通过；XeLaTeX 编译 5 页并逐页视觉检查，原文 Fig. 9 上图裁切保留坐标与图例。Ruff 检查与格式检查通过。自动 Zotero 同步返回 `FALLBACK_REQUIRED`：新目录 `slides/popiii_burst_mass` 不在 AuroraLF 既有批准文档目录中；本次未扩大同步目录范围，PDF 本地可用。

ADS 元数据保存于 `external_data/literature_sources/popiii_burst_mass/ads.json`。两篇 Hazlett 论文通过浏览器读取原文 HTML/PDF；本地 PDF 下载遇到 TLS 链失败及 HTTP 502/超时，未获得可用的本地 PDF 证据裁图。Liu 原文 PDF 从 Cambridge 机构库获取并视觉核查 Fig. 9 上图；来源与文件哈希记入同目录 provenance.json。
