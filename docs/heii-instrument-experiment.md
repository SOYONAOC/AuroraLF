# He II 仪器响应实验

2026-09-18。按用户要求只计算结果和独立图件，不修改 slides。

## 输入和比较范围

状态更新：μ=0 全量种群已重算；本文件的仪器实验仍使用固定的旧 μ=0.5
输入，不能把下述通过比例直接用于新种群。用户同意考虑加入仪器内容，
当前先核对与 V24 的观测设置，尚未把仪器结果加入主讲稿。

### 与 V24 的设置对照

依据 [Venditti et al. 2024 §3.1–3.2](https://arxiv.org/html/2405.10940v2#S3.SS1)。

| 设置 | V24 | 本次独立实验 |
|---|---|---|
| 仪器模式 | NIRSpec IFU、MOS | MOS |
| 分辨率 | PRISM、R≈1000、2700 | PRISM、R≈1000 |
| 曝光 | 约 10、50 h | 10.115、50.186 h |
| 本征线宽 | 50、500 km/s | 50、500 km/s |
| 源形态 | 点源 | 点源及 FWHM=0.1、0.2 arcsec Gaussian |
| 连续谱 | 无 | UV 归一化平坦 fν，并传播扣除误差 |
| 背景 | GN-z11 He II 团块位置，可见期第 50 百分位 | minzodi benchmark |
| 检出门槛 | 主要为积分 S/N≈3；另有 R≈1000、50 h、S/N≈5 | 全部为积分 S/N≥5 |
| MOS 提取 | 三快门、中心源、quadrant 3、full-shutter | 同类三快门居中、full-shutter 设置 |
| 标定版本 | JWST ETC 4.0 | Pandeia 2026.7 |
| 偏心团块 | §3.2 单独计算视场遗漏 | 未模拟团块位置和视场遗漏 |

V24 的 IFU 提取半径为 0.09 arcsec，天空环为 0.3–0.9 arcsec。
居中源的灵敏度阈值与宿主外围团块落在视场外的概率是两项不同计算。
当前结果包含仪器对居中有限大小源的通量损失，不包含未知团块位置的损失。

可比基准应先统一 MOS、R≈1000、50 h、积分 S/N=5、点源、无连续谱和
GN-z11 中等背景，再分别改变源大小、连续谱、线宽与曝光。新标定版本
即使采用相同输入，也不应直接称为 ETC 4.0 的逐数值复现。

使用当前 He II 图对应的五组已完成样本：z=6.639、8.1623、10.6、12.342、13.86。
阈值分布的对数均值为 0.5，标准差为 1.5 dex，爆发效率为 0.03。
这是固定种群、检验仪器影响的实验，与正在进行的 μ=0 全量重算分开。
前三组保留原图的年轻 UV 亮选样；后两组保留目标 UV 星等 ±0.25 mag 选样。
已知零谱线对象保留，历史起点前的未知爆发权重单列。

统一取透镜放大率为 1、无尘。连续谱假设为平坦 fν，即 UV 斜率 β=-2，
每个对象的振幅由其已有 Pop II+III UV 光度给出。高红移输入保留原先的
Pop II 1600 Å / Pop III 1500 Å UV 代理，不宣称增加了新的完整连续谱计算。
这些源不是 LAP1 等真实目标的空间、透镜或观测配置复现。

## 仪器设置

- Pandeia engine 2026.7，配套官方 2026.7 标定和 PSF 数据，下载档案经 SHA1 校验。
- NIRSpec MOS，q3_183_86、1×3 微快门，居中源，full-shutter 提取；相邻快门扣背景。
- PRISM/CLEAR 与中分辨率 G140M/F100LP（低红移）或 G235M/F170LP（高红移）。
- 点源，以及圆对称 Gaussian 源 FWHM=0.1、0.2 arcsec；谱线和连续谱同形态。
- He II Gaussian 本征 FWHM=50、500 km/s。
- nrsirs2，19 groups，1 integration；通过重复曝光实现约 10、50 h，表中保留实际时长。
- Pandeia minzodi benchmark 背景；保留默认暗电流、读噪声、宇宙线与平场噪声。

本次把不同曝光视为独立的有效曝光，遵循 Pandeia 对平场项随曝光次数缩放的
实现；不把无抖动、完全相关的系统误差宣称为可以随时间无限下降。

## 从探测器输出到积分谱线信噪比

Pandeia 输出提取后的电子计数率和逐光谱像素噪声。取累计谱线计数中央 95%
对应的整像素窗口，两侧 3—7 个半窗宽范围拟合平坦 fν 连续谱振幅。
谱线积分与连续谱扣除同时传播噪声。采用官方积分线 S/N 方法的逐像素方差
求和；这不是对真实重采样光谱协方差或红移搜索误差的重建。

为逐对象计算而不逐次重复 ETC，显式测定背景、谱线、连续谱及其二次平场
方差项。独立混合输入验证响应；另用 26 次曝光完整 Pandeia 计算核对时间缩放。
连续谱和背景基准输入也保留零强度谱线，强制沿用相同的内部谱线采样；
避免无谱线与有谱线输入的重采样差异污染线性响应。原始不一致采样的尝试
保存在 `outputs/heii_instrument_20260918/sampling-grid-original/`，不用于最终结果。
求解含源噪声的 S/N=5 方程，得到每个对象连续谱条件下的通量门槛。

对可能超出未饱和近似的极亮对象，用背景、谱线、连续谱各自最亮像素的
饱和比例之和给出保守上界；超过上界的对象权重单列，形成检出比例上下界。
这不是断言所有这些对象均已饱和，也不将其按弱源公式外推为精确结果。

“通过比例”指种群丰度权重下预期积分线 S/N≥5 的比例。另存 Gaussian
测量噪声假设下的触发概率 Φ(S/N−5)，两者均不是实际巡天完备度。
未加入 O III] 混线、红移误差、未知空间偏移或透镜剪切；PRISM 中能检出
孤立谱线不等于真实观测已能分解 He II/O III]。

## 重现与输出

在项目环境中运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/heii_instrument_populations.py
PYTHONPATH=. .venv/bin/python scripts/run/run_heii_instrument.py --local-workers 4
```

`data_save/heii_instrument_20260918/` 保存选样、逐配置 ETC 输入、提取后计数和
方差、响应系数、汇总 JSON 与 CSV。`outputs/heii_instrument_20260918/`
保存运行日志及独立 PDF/PNG 图。

本地 SLURM 作业 159818 因排队未启动而取消；相同仪器后处理在当前工作机
限制为 4 个 CPU 执行。超算 μ=0 作业及其节点设置未改变。

## 方法来源

- [STScI Pandeia 安装与版本配套](https://outerspace.stsci.edu/spaces/PEN/pages/77530136/Pandeia+Engine+Installation)
- [STScI 积分谱线信噪比](https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/jwst-etc-outputs-overview/jwst-etc-integrated-emission-line-snr)
- [STScI NIRSpec 模式与波段](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters)
