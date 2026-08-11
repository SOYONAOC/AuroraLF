---
title: "Pop III：He II 与 PISN 约束文献审计"
subtitle: "从高红移晕到观测似然的前向物理链"
author: "AuroraLF research note"
date: "2026-08-03"
lang: zh-CN
abstract: |
  本文沿高红移暗物质晕、原初气体存续、Pop III 恒星族、He II 星云辐射、
  PISN 爆炸率和巡天选择效应组成的前向链，整理 He II 与 PISN 对 Pop III
  初始质量函数的约束。重点包括不同晕质量中的 Pop III/Pop II 星形成占比、
  Hebe 各运动学分量的线比、随机 IMF、质量依赖的 PISN 水动力命运、寿命延迟、
  宿主晕质量分布、核合成产额、巡天控制时间，以及与 21 cm 信号共享的恒星族
  参数。所有结论均附带适用条件、原始文献和观测边界。
header-includes:
  - '\usepackage{etoolbox}'
  - '\AtBeginEnvironment{longtable}{\small}'
  - '\setlength{\tabcolsep}{3pt}'
  - '\setlength{\emergencystretch}{3em}'
  - '\sloppy'
---

> **一句话结论**
>
> He II 约束最年轻、最硬电离光子的恒星尾部，PISN 约束特定终末
> core-mass 区间，21 cm 约束积分后的 LW、Ly$\alpha$、X-ray 和电离光子历史。
> 三条通道只有共享同一套 IMF、恒星演化、旋转、双星、星云吸收和化学富集历史，
> 才能形成同一物理参数空间上的联合限制。

## 1. 研究问题与完整前向链

本文关注两个问题。

1. 在 $z>10$ 的不同晕质量中，Pop III 与 Pop II 分别贡献多少星形成？
2. He II、PISN 和 21 cm 如何把这些星形成活动投影成可观测量，并进一步限制
   Pop III IMF？

完整物理链可写为

$$
\begin{aligned}
&\left(M_h,\dot M_h,z,J_{\rm LW},v_{\rm bc},p_{\rm pristine}\right)
\longrightarrow \dot M_{\star,\rm III}(M_h,z)
\longrightarrow \theta_\star \\
&\qquad\longrightarrow
\left\{Q_{\rm H},Q_{\rm He^+},L_{1500},N_{\rm LW},p_{\rm term},y_Z\right\}
\\
&\qquad\longrightarrow
\left\{L_{1640},\dot n_{\rm PISN},\delta T_{21}\right\}
\longrightarrow \mathcal L_{\rm obs}.
\end{aligned}
$$

这里 $M_h$ 和 $\dot M_h$ 描述晕质量及其增长；$J_{\rm LW}$ 是 Lyman--Werner
背景；$v_{\rm bc}$ 是重子--暗物质相对速度；$p_{\rm pristine}$ 是气体保持在
临界金属丰度以下的概率；$\theta_\star$ 汇总 IMF、恒星演化、旋转/双星群体
分布和恒星大气参数。

每个观测通道看到前向链的不同部分：

| 通道 | 主要物理灵敏度 |
|---|---|
| UV | 年龄积分后的 $L_{1500}/M_\star$、星云连续谱和尘埃 |
| He II $\lambda1640$ | $Q_{\rm He^+}$、年龄小于数 Myr 的高质量恒星、星云吸收和随机 IMF |
| PISN | 成对不稳定水动力命运、IMF 区间权重、恒星寿命、核合成产额、爆炸光变和巡天选择 |
| 21 cm | LW 抑制、Ly$\alpha$ 耦合、X-ray 加热和电离历史的体积平均 |

## 2. 不同晕质量中的 Pop III 与 Pop II

### 2.1 冷却阈值

分子氢冷却允许 Pop III 在 minihalo 中形成，其有效阈值可写成

$$
M_{\rm mol}=M_{\rm mol}\!\left(z,J_{\rm LW},v_{\rm bc}\right).
$$

$J_{\rm LW}$ 会解离 H$_2$，$v_{\rm bc}$ 会降低重子落入小晕的效率；二者均使
$M_{\rm mol}$ 上升。原子氢冷却对应 $T_{\rm vir}\simeq10^4$ K。标准关系为

$$
M_{\rm atomic}\simeq
10^8 h^{-1}M_\odot
\left(\frac{T_{\rm vir}}{1.98\times10^4\,{\rm K}}\right)^{3/2}
\left(\frac{0.6}{\mu}\right)^{3/2}
\left[
\frac{\Omega_m}{\Omega_m^z}
\frac{\Delta_c}{18\pi^2}
\right]^{-1/2}
\left(\frac{1+z}{10}\right)^{-3/2}.
$$

该尺度控制原子气体能否有效冷却。恒星族身份还取决于局部金属富集、混合时间和
原初气体补给。

### 2.2 星形成率的物理分解

一个便于解释的分解为

$$
\dot M_{\star,\rm III}
=
f_b\dot M_h\,
\epsilon_{\rm III}\,
f_{\rm cool,III}(M_h,z,J_{\rm LW},v_{\rm bc})\,
p_{\rm pristine}(M_h,z,\delta_{\rm env}),
$$

$$
\dot M_{\star,\rm II}
=
f_b\dot M_h\,
\epsilon_{\rm II}\,
f_{\rm cool,II}(M_h,z,Z)\,
\left[1-p_{\rm pristine}(M_h,z,\delta_{\rm env})\right].
$$

$f_b$ 是宇宙重子比例，$\epsilon_{\rm III/II}$ 是从新获得气体转化为恒星的效率，
$\delta_{\rm env}$ 表示环境过密度和邻近金属源。这个写法把冷却与化学状态放在
两个独立因子中。

给定 $M_h$ 和 $z$，瞬时 Pop III 星形成占比为

$$
f_{\rm III}^{\rm SFR}(M_h,z)
=
\frac{\langle\dot M_{\star,\rm III}\mid M_h,z\rangle}
{\langle\dot M_{\star,\rm III}+\dot M_{\star,\rm II}\mid M_h,z\rangle}.
$$

质量箱 $[M_1,M_2]$ 中的 HMF 加权占比为

$$
F_{\rm III}^{\rm SFR}(M_1,M_2,z)
=
\frac{
\displaystyle\int_{M_1}^{M_2}dM_h\,
\frac{dn}{dM_h}
\langle\dot M_{\star,\rm III}\mid M_h,z\rangle
}{
\displaystyle\int_{M_1}^{M_2}dM_h\,
\frac{dn}{dM_h}
\langle\dot M_{\star,\rm III}+\dot M_{\star,\rm II}\mid M_h,z\rangle
}.
$$

宇宙平均 SFRD 则为

$$
\dot\rho_{\star,k}(z)
=
\int dM_h\,\frac{dn}{dM_h}
\langle\dot M_{\star,k}\mid M_h,z\rangle,
\qquad k\in\{{\rm II,III}\}.
$$

这三个量分别回答单个质量处的混合比例、有限质量箱的贡献和全宇宙总量。总 SFRD
无法单独给出晕质量分布。

### 2.3 高红移下的预期形态

- $M_h\sim10^5$--$10^7\,M_\odot$：H$_2$ 冷却和 LW 反馈主导，保持原初状态的
  minihalo 可由 Pop III 占主导。
- $M_h\sim10^7$--$10^8\,M_\odot$：分子冷却、原子冷却、内部富集和外部金属污染
  同时作用，Pop III 与 Pop II 的混合最强。
- $M_h\gtrsim10^8\,M_\odot$：多代恒星形成提高已富集气体的权重，Pop II 通常占主导；
  外围新吸积的原初气体和未混合口袋仍可产生局部 Pop III。
- 红移升高会缩短累计富集时间，因此同一晕质量的 $p_{\rm pristine}$ 通常上升。

因此，Pop III/Pop II 的质量过渡是一条随红移、环境和金属混合变化的宽分布。
原子冷却质量只进入冷却项，$p_{\rm pristine}$ 决定化学身份。

### 2.4 平滑冷却基线给出的条件分布

当前平滑基线可概写为

$$
\dot M_{\star,\rm III}^{\rm mf}
\propto
\dot M_h\epsilon_{\rm III}
\exp\!\left(-\frac{M_{\rm mol}}{M_h}\right)
\exp\!\left(-\frac{M_h}{M_{\rm atomic}}\right),
$$

$$
\dot M_{\star,\rm II}^{\rm mf}
\propto
\dot M_h\epsilon_{\rm II}
\exp\!\left(-\frac{M_{\rm atomic}}{M_h}\right).
$$

两个指数产生平滑质量过渡。金属状态在这一基线中保持固定，因此下表表示冷却和
晕占空权重下的条件结果：

| 晕质量 | $z=10$ | $z\simeq20$ | $z\simeq30$ |
|---|---:|---:|---:|
| $10^5$--$10^6M_\odot$ | 100.00 / 0.00 | 100.00 / 0.00 | 100.00 / 0.00 |
| $10^6$--$10^7M_\odot$ | 99.69 / 0.31 | 98.80 / 1.20 | 98.68 / 1.32 |
| $10^7$--$10^8M_\odot$ | 40.30 / 59.70 | 41.78 / 58.22 | 42.75 / 57.25 |
| $10^8$--$10^9M_\odot$ | 1.59 / 98.41 | 0.41 / 99.59 | 0.106 / 99.894 |
| $>10^9M_\odot$ | $\simeq0$ / 100 | $\simeq0$ / 100 | $\simeq0$ / 100 |

每格依次为 Pop III / Pop II 的瞬时 SFR 百分比。全局 Pop III SFR 百分比为

| 红移 | 10 | 15.1 | 20.0 | 25.2 | 30.2 | 35 |
|---:|---:|---:|---:|---:|---:|---:|
| $F_{\rm III}^{\rm global}$ | 2.78% | 19.72% | 57.19% | 87.54% | 97.16% | 99.34% |

全局占比同时受到质量箱内部比例和各质量箱 SFRD 权重控制。Pop III SFRD 的
质量分布随红移显著迁移：

| 红移 | mode | median | 16--84% 区间 |
|---:|---:|---:|---:|
| 10 | $4.31\times10^7M_\odot$ | $4.47\times10^7M_\odot$ | $1.88\times10^7$--$1.07\times10^8M_\odot$ |
| 20 | $5.70\times10^6M_\odot$ | $5.68\times10^6M_\odot$ | $2.24\times10^6$--$1.57\times10^7M_\odot$ |
| 30.2 | $7.55\times10^5M_\odot$ | $1.06\times10^6M_\odot$ | $4.50\times10^5$--$2.95\times10^6M_\odot$ |

在 $z=10$，$10^5$--$10^6M_\odot$ 质量箱虽然由 Pop III 占满，其总 SFRD
权重约为 $3.7\times10^{-11}\%$。在 $z=30.2$，同一质量箱贡献总 SFRD 的
46.3%，$10^6$--$10^7M_\odot$ 再贡献 50.0%。高红移全局 Pop III 占比的上升
主要来自 SFRD 权重向低质量晕移动。

加入 $p_{\rm pristine}(M_h,z,\delta_{\rm env})$ 后，高质量端会进一步受到累积富集
历史约束，质量过渡的宽度和位置也会出现 halo-to-halo scatter。

## 3. 统一的 Pop III 恒星族

### 3.1 IMF 与共享参数

采用质量归一化 IMF

$$
\xi(M\mid\theta_{\rm IMF})
=A\,M^{-x}\exp\!\left(-\frac{M_{\rm ch}}{M}\right),
\qquad
\int_{M_{\min}}^{M_{\max}}M\xi(M)\,dM=1.
$$

纯幂律由 $M_{\rm ch}\rightarrow0$ 得到。共享恒星族参数定义为

$$
\theta_\star
\equiv
(\theta_{\rm IMF},\theta_{\rm evo},\theta_{\rm pop},\theta_{\rm atm}),
$$

其中 $\theta_{\rm IMF}=(x,M_{\rm ch},M_{\min},M_{\max})$；
$\theta_{\rm evo}$ 描述风、混合、核反应率和恒星轨道；
$\theta_{\rm pop}$ 描述旋转与双星轨道的群体分布；
$\theta_{\rm atm}$ 描述恒星大气与光谱库。出生金属丰度 $Z$ 作为每条恒星轨道的
显式条件变量，不重复收入 $\theta_\star$。

同一个 $\theta_\star$ 应同时给出

$$
\left\{
\ell_{1500}(a),\,
q_{\rm H}(a),\,
q_{\rm He^+}(a),\,
n_{\rm LW}(a),\,
\tau_\star(M),\,
p_{\rm PISN}(M),\,
y_Z(M)
\right\}.
$$

$a$ 是恒星年龄；$\ell_{1500}$ 的单位为
$\,{\rm erg\,s^{-1}\,Hz^{-1}\,M_\odot^{-1}}$，
$q_{\rm H}$ 和 $q_{\rm He^+}$ 的单位为
$\,{\rm s^{-1}\,M_\odot^{-1}}$。

### 3.2 SFH 与恒星族核的卷积

任意线性恒星族量 $X$ 都由

$$
X(t)=\int_0^\infty
\dot M_{\star,\rm III}(t-a)\,
x_{\rm SSP}(a\mid\theta_\star)\,da
$$

给出。年龄核必须与星形成历史在同一时间单位下积分。He II 的年龄核集中于最初
数 Myr；UV 连续谱的有效时间窗更长；PISN 通过质量依赖的寿命连接形成时刻与爆炸时刻。

### 3.3 单团块的随机 IMF

质量归一化 IMF 的平均单星质量为

$$
\langle M\rangle
=
\frac{\int M\xi(M)dM}{\int\xi(M)dM}
=
\frac{1}{\int\xi(M)dM},
$$

形成质量为 $M_{\rm burst}$ 的团块具有期望恒星数

$$
\bar N_\star=\frac{M_{\rm burst}}{\langle M\rangle}.
$$

对 Salpeter 50--500 $M_\odot$，$\langle M\rangle\simeq111.7M_\odot$。
当最近形成质量约为 $116M_\odot$ 时，$\bar N_\star\simeq1.04$。此时单个团块的
$Q_{\rm He^+}$ 和 PISN 数量需要由离散恒星质量之和给出：

$$
Q_{\rm He^+}^{\rm clump}(t)=\sum_{i=1}^{N_\star}q_{\rm He^+}(M_i,t),
\qquad
N_{\rm PISN}^{\rm clump}=\sum_{i=1}^{N_\star}I_{\rm PISN}(M_i).
$$

连续 SSP 描述许多团块的总体均值；Hebe 类单源对应一个宽概率分布。

### 3.4 当前通道中的恒星族假设

| 物理通道 | 当前采用的恒星族假设 |
|---|---|
| Pop III UV | 零金属、Salpeter 1--500 $M_\odot$ |
| He II 基线 | 零金属、Salpeter 1--500 $M_\odot$ |
| Hebe 对照 | 零金属、Salpeter 50--500 $M_\odot$ |
| PISN 形成率 | 幂律 50--500 $M_\odot$，经典 140--260 $M_\odot$ 命运窗 |
| LW 与电离产额 | 固定 intermediate-IMF 产额 |

这些通道当前对应多套 $\theta_\star$。联合约束的第一项物理要求是统一恒星族。

## 4. He II $\lambda1640$ 的前向物理链

### 4.1 从硬电离光子到线光度

He$^+$ 电离阈值为 54.4 eV。Pop III 星形成历史产生的光子率为

$$
Q_{\rm He^+}(t)
=
\int_0^\infty
\dot M_{\star,\rm III}(t-a)
q_{\rm He^+}(a\mid\theta_\star)\,da.
$$

定义 He$^+$ 电离光子的吸收比例 $f_{\rm abs,He^+}$：

$$
Q_{\rm He^+,abs}=f_{\rm abs,He^+}Q_{\rm He^+}.
$$

在 $T_e\simeq3\times10^4$ K、ionization-bounded 和 Case-B 条件下，

$$
\boxed{
L_{1640}^{\rm B}
=
c_{1640}Q_{\rm He^+,abs},
\qquad
c_{1640}=5.67\times10^{-12}\ {\rm erg\,photon^{-1}}.
}
$$

低 ionization parameter 时，部分高能光子被 H 吸收；密度、几何和 two-photon
continuum 也会改变线强与等效宽度。因此 $c_{1640}$ 给出一个条件明确的基准，
完整预测还需 $U$、$n_H$、$T_e$ 和吸收几何。

### 4.2 线与连续谱共同决定等效宽度

1640 Å 连续谱为

$$
L_\lambda(1640,t)
=
\int_0^\infty \dot M_{\star,\rm III}(t-a)
\left[
\ell_{\lambda,\star}(1640,a)
+
\ell_{\lambda,\rm neb}(1640,a)
\right]da,
$$

静止系等效宽度为

$$
\boxed{
W_0(1640)=\frac{L_{1640}}{L_\lambda(1640)}.
}
$$

线、恒星连续谱和星云连续谱必须共享同一个 IMF、年龄分布和光子吸收几何。

### 4.3 Hebe 的分量信息

Maiolino、Übler 和 Rusta 的高分辨率结果给出以下关键量：

| 观测量 | 数值与物理角色 |
|---|---|
| He II 总通量 | $(1.11\pm0.17)\times10^{-19}\ {\rm erg\,s^{-1}\,cm^{-2}}$，已做孔径修正，仍含 $\mu=1.42$ |
| He II 总内禀光度 | $(8.54\pm1.29)\times10^{40}\ {\rm erg\,s^{-1}}$，已做弥散清理和去透镜 |
| C1 He II 光度 | $(5.1\pm0.9)\times10^{40}\ {\rm erg\,s^{-1}}$ |
| C1--C2 速度间隔 | $126\pm17\ {\rm km\,s^{-1}}$ |
| H$\gamma$ | 只随 C2 检出；通量已做孔径和放大修正 |
| 金属丰度 | $Z_{\rm gas}<0.02Z_\odot$ |
| C1 硬度 | Rusta 报告 $3\sigma$ 下 He II/H$\gamma>0.7$ |

统一到去透镜空间后，

$$
R_{\rm total}
=
\frac{F_{{\rm HeII,total}}/\mu}{F_{{\rm H}\gamma,{\rm C2}}}
=0.489,
$$

$$
R_{\rm C2}
=
\frac{F_{{\rm HeII,C2}}/\mu}{F_{{\rm H}\gamma,{\rm C2}}}
=0.224.
$$

$R_{\rm total}$ 把 C1+C2 的 He II 与 C2 的 H$\gamma$ 合在一起，适合作混合系统
诊断。C2 的共同运动学线比由 $R_{\rm C2}$ 表示。C1 的 H$\gamma$ 未检出，对应

$$
F_{{\rm H}\gamma,{\rm C1}}<U_{\rm C1},
\qquad
R_{\rm C1}>
\frac{F_{{\rm HeII,C1}}/\mu}{U_{\rm C1}}.
$$

总比值的统计误差约为 $0.111$；透镜放大、孔径修正和共享通量定标还会引入
协方差。Rusta 预印本正文与结论对 $M_{\rm ch}\sim75M_\odot$ 的置信标签存在差异，
正式推断应依据发布的等似然网格。

### 4.4 He II 的联合谱线似然

适合 Hebe C1 的观测向量为

$$
\mathbf D_{\rm C1}
=
\left\{
R_{\rm C1}^{\rm lower},
L_{1640,\rm C1},
W_{0,\rm C1}^{\rm lower},
F_{\rm metal}^{\rm upper}
\right\}.
$$

联合似然可写为

$$
\mathcal L_{\rm HeII}
=
\mathcal L_{\rm cens}(R_{\rm C1})
\,
\mathcal L(L_{1640,\rm C1})
\,
\mathcal L_{\rm cens}(W_{0,\rm C1})
\prod_j\mathcal L_{\rm cens}(F_{{\rm metal},j}).
$$

对高斯测量和下限 $R_{\min}$，

$$
\mathcal L_{\rm cens}
=
1-\Phi\!\left(
\frac{R_{\min}-R_{\rm pred}}{\sigma_R}
\right),
$$

其中 $\Phi$ 为标准正态累积分布。年龄、$U$、$n_H$、尘埃、旋转、双星和
$f_{\rm abs,He^+}$ 都应作为扰动参数。

He II 的来源分类可写成混合模型

$$
p(\mathbf D)
=
\sum_k\pi_k p(\mathbf D\mid k),
\qquad
k\in
\{
{\rm PopIII,metal\!-\!poor\ stars,AGN,XRB,WR,shock}
\}.
$$

可靠的 Pop III 归因需要同时利用 He II/H recombination lines、EW、UV/optical
metal-line limits、线宽、空间偏移和宿主环境。

## 5. PISN：从恒星命运到巡天事件率

PISN 预测由一条连续的物理链构成：恒星内部的成对不稳定性触发动力学收缩，
爆炸性氧燃烧决定完全解体或脉冲抛射，恒星演化网格给出质量与金属丰度依赖的
命运概率，IMF 和星形成史把单星命运转换为单晕爆炸率，晕质量函数给出宇宙体积率，
辐射输运与巡天选择函数最终给出可探测事件数。下文沿这条链依次定义计算量、
归一化和原始文献依据。

下文以
$Z\equiv Z_{\rm birth,abs}/Z_{\odot,\rm ref}$ 表示恒星出生时的线性金属丰度，
并固定 $Z_{\odot,\rm ref}=0.0134$，对应 Asplund et al. (2009) 的太阳光球总金属
质量分数
([Asplund et al. 2009](https://doi.org/10.1146/annurev.astro.46.060407.145222))。
$Z=0$ 的严格零金属 Pop III 分量作为离散成分处理；$d\log_{10}Z$ 积分仅作用于
$Z>0$ 的富集气体。引用采用绝对金属质量分数时直接除以
$Z_{\odot,\rm ref}$；引用采用源文献太阳单位时，先用该源声明的
$Z_{\odot,\rm src}$ 恢复绝对质量分数，再转换到本文标尺。

第 5 节的解析基准采用“严格 Pop III：$Z=0$；Pop II：$Z>0$”分类。若恒星形成
模型允许 $0<Z<Z_{\rm crit}$ 继续形成 Pop III，则以下替换同时作用于事件数、
寿命、产额和能量卷积：

$$
\begin{aligned}
\dot M_{\star,\rm III}\mathcal K_{\rm III}(\tau\mid0)
\;&\mapsto\;
\dot M_{\star,\rm III}^{(0)}\mathcal K_{\rm III}(\tau\mid0)\\
&\quad+
\int_{0<Z<Z_{\rm crit}}d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm III}(Z)
\mathcal K_{\rm III}(\tau\mid Z),\\
\mathcal R_{\rm II}^{>0}(\tau)
&\equiv
\int_{Z>0}d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm II}(Z)\mathcal K_{\rm II}(\tau\mid Z),\\
\mathcal R_{\rm II}^{>0}(\tau)
\;&\mapsto\;
\int_{Z\ge Z_{\rm crit}}d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm II}(Z)\mathcal K_{\rm II}(\tau\mid Z).
\end{aligned}
\tag{5.0}
$$

### 5.1 Pair instability 与恒星命运判据

在高温氧核中，热光子转化为电子--正电子对会降低辐射压对绝热压缩的响应。
局部绝热指数

$$
\Gamma_1
\equiv
\left(\frac{\partial\ln P}{\partial\ln\rho}\right)_s
<\frac{4}{3}
\tag{5.1}
$$

时，该质量层对绝热压缩的恢复力减弱。当不稳定区域扩展到足以降低全局径向
稳定性指标时，恒星进入动力学收缩。收缩引发爆炸性氧燃烧；当释放的核能
超过恒星束缚能并使全部质量层逸出时，恒星完全解体并且不留下致密残骸。
这条物理链由
[Barkat, Rakavy & Sack (1967)](https://doi.org/10.1103/PhysRevLett.18.379)
和 [Rakavy & Shaviv (1967)](https://doi.org/10.1086/149204) 建立。

现代一维恒星轨道常用压力加权的全局稳定性指标

$$
\langle\Gamma_1\rangle
=
\frac{\displaystyle\int\Gamma_1(P/\rho)\,dm}
{\displaystyle\int(P/\rho)\,dm}
\tag{5.1a}
$$

监测整体失稳。Marchant et al. (2019)、Renzo et al. (2020) 和
Farmer et al. (2020) 的数值轨道采用
$\langle\Gamma_1\rangle-4/3<0.01$ 作为提前切换到水动力演化的判据；
Marchant et al. 还要求中心温度 $T_c>10^9\,{\rm K}$。$0.01$ 是数值切换裕量，
物理失稳边界由全局结构决定。非旋转球对称情形下，水动力阶段的最小示意方程组为

$$
\begin{aligned}
\frac{\partial r}{\partial m}
&=
\frac{1}{4\pi r^2\rho},\\
\frac{\partial^2r}{\partial t^2}
&=
-4\pi r^2\frac{\partial P}{\partial m}
-\frac{Gm}{r^2},\\
\frac{\partial u}{\partial t}
+P\frac{\partial(1/\rho)}{\partial t}
&=
\dot\epsilon_{\rm nuc}
-\dot\epsilon_\nu
-\frac{\partial L}{\partial m}.
\end{aligned}
\tag{5.1b}
$$

其中 $m$ 为拉格朗日质量坐标，$r$ 为半径，$\rho$ 为密度，$P$ 为压强，
$u$ 为比内能，$L$ 为光度，$\dot\epsilon$ 为单位质量能量变化率。
采用 cgs 单位时，$m$ 和 $M_\star$ 以 g 表示，$t$ 以 s 表示，
$v$ 以 $\mathrm{cm\,s^{-1}}$ 表示，
$G$ 的单位为 $\mathrm{cm^3\,g^{-1}\,s^{-2}}$；$r$、$\rho$、$P$、$u$、
$L$ 和 $\dot\epsilon$ 的单位依次为
cm、$\mathrm{g\,cm^{-3}}$、$\mathrm{erg\,cm^{-3}}$、
$\mathrm{erg\,g^{-1}}$、$\mathrm{erg\,s^{-1}}$ 和
$\mathrm{erg\,g^{-1}\,s^{-1}}$。式 (5.1b) 是球对称最小重构，
实际轨道还包括冲击耗散或人工黏性、对流和辐射输运闭合。含 $e^\pm$ 对的状态
方程给出 $P$、$u$ 和 $\Gamma_1$；核反应网络给出
$\dot\epsilon_{\rm nuc}$ 与元素演化；中微子损失进入
$\dot\epsilon_\nu$。爆炸后的全星能量可写为

$$
E_{\rm tot}
=
\int_0^{M_\star}
\left(
\frac{v^2}{2}
+u
-\frac{Gm}{r}
\right)dm.
\tag{5.1c}
$$

$u$ 采用与状态方程一致的能量零点，并包含 $e^\pm$ 对对热力学内能的贡献。
$E_{\rm tot}>0$、全部质量层无回落趋势且渐近径向速度非负，对应完全解体 PISN。
PPISN 轨道在一次或多次脉冲中实际抛出非束缚质量层，同时保留束缚核心并恢复
准静态演化；持续内落且由光致蜕变和中微子损失主导的轨道形成黑洞。这个命运
分类及水动力切换方式见
[Marchant et al. (2019)](https://doi.org/10.3847/1538-4357/ab3426)、
[Renzo et al. (2020)](https://doi.org/10.1051/0004-6361/202037710) 和
[Farmer et al. (2020)](https://doi.org/10.3847/2041-8213/abbadd)；
状态方程能量记账可参见
[Takahashi (2018)](https://doi.org/10.3847/1538-4357/aad2d2)。
因此，ZAMS 质量窗在群体计算中承担轨道网格的代理；单星 PISN 命运由
状态方程、核燃烧和水动力能量共同决定。

[Heger & Woosley (2002)](https://doi.org/10.1086/338487) 从氦主序开始演化
近乎裸露的零金属非旋转氦星；轨道忽略普通稳态风质量损失，同时保留 PPISN
脉冲造成的动力学抛射。该网格给出以下经典命运分区。
表中的 $M_{\rm He,init}$ 表示该轨道网格的初始氦星质量：

| 初始氦星质量 $M_{\rm He,init}$ | 经典终态 |
|---:|---|
| $M_{\rm He,init}\simeq40$--$63\,M_\odot$ | pulsational pair instability；脉冲抛射后形成铁核并坍缩，通常形成黑洞 |
| $M_{\rm He,init}\simeq64$--$133\,M_\odot$ | 完全 PISN；整星解体 |
| $M_{\rm He,init}\gtrsim133.3\,M_\odot$ | 光致蜕变促进坍缩并形成黑洞 |

Heger & Woosley (2002) 的式 (1) 给出近似映射，Pan, Kasen & Loeb (2012)
随后在高红移 PISN 预测中沿用该关系：

$$
M_{\rm He,init}
\simeq
\frac{13}{24}\left(M_{\rm ZAMS}-20M_\odot\right).
\tag{5.2}
$$

于是 $64$--$133\,M_\odot$ 的初始氦星约对应
$M_{\rm ZAMS}\simeq140$--$260\,M_\odot$。式 (5.2) 适用于该组零金属、
非旋转、普通风质量损失可忽略的演化轨道；$M_{\rm ZAMS}$ 表示恒星的初始
零龄主序质量。Umeda & Nomoto (2002) 的 $130$--$300\,M_\odot$ 计算提供
核合成产额网格，其质量端点限定该网格的覆盖范围；经典命运窗继续采用氦核
终态计算给出的边界
([Heger & Woosley 2002](https://doi.org/10.1086/338487);
[Umeda & Nomoto 2002](https://doi.org/10.1086/323946);
[Pan, Kasen & Loeb 2012](https://doi.org/10.1111/j.1365-2966.2012.20837.x))。

有限金属丰度、旋转、风质量损失、双星相互作用和核反应率会改变
$M_{\rm ZAMS}\rightarrow M_{\rm He/CO}$ 的映射。统一的命运概率可写成

$$
p_{\rm PISN}(M,Z\mid\theta_{\rm evo},\theta_{\rm pop})
=
\int d\boldsymbol\lambda_{\rm ast}\,
p(\boldsymbol\lambda_{\rm ast}\mid M,Z,\theta_{\rm pop})
\,
\mathbf 1
\!\left[
\mathcal F_{\rm tr}
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
= {\rm PISN}
\right].
\tag{5.3}
$$

其中 $M$ 为与 IMF 一致的 ZAMS 初始质量，$\mathcal F_{\rm tr}$ 是恒星演化
网格给出的终态。$\boldsymbol\lambda_{\rm ast}$ 表示逐星初始旋转和双星轨道
历史，$\theta_{\rm pop}$ 给出这些变量的群体分布；
$\theta_{\rm evo}$ 汇总风律、混合处方和
$^{12}{\rm C}(\alpha,\gamma)^{16}{\rm O}$ 反应率等全局轨道假设。
全局轨道假设在模型层级比较或边缘化。确定性轨道对应
$p_{\rm PISN}=0$ 或 $1$；边缘化后的 $p_{\rm PISN}$ 位于 $[0,1]$。
Takahashi (2018) 与 Farmer et al. (2019) 表明 core carbon fraction 和
$^{12}{\rm C}(\alpha,\gamma)^{16}{\rm O}$ 会移动不稳定区；
Woosley & Heger (2021) 进一步量化了反应率、旋转和双星带来的质量边界变化；
Umeda & Nagele (2024) 展示了有限金属丰度下旋转与风的竞争
([Takahashi 2018](https://doi.org/10.3847/1538-4357/aad2d2);
[Farmer et al. 2019](https://doi.org/10.3847/1538-4357/ab518b);
[Woosley & Heger 2021](https://doi.org/10.3847/2041-8213/abf2c4);
[Umeda & Nagele 2024](https://doi.org/10.3847/1538-4357/ad140a))。

### 5.2 IMF 质量归一化与单位形成质量效率

令 $\phi(M)=dN/dM$ 为 ZAMS 初始质量上的恒星数 IMF。为了把星形成质量转换为事件数，先定义
质量归一化核

$$
\psi(M\mid\theta_{\rm IMF})
\equiv
\frac{\phi(M\mid\theta_{\rm IMF})}
{\displaystyle\int_{M_{\min}}^{M_{\max}}M\phi(M\mid\theta_{\rm IMF})\,dM},
\qquad
\int_{M_{\min}}^{M_{\max}}M\psi(M)\,dM=1.
\tag{5.4}
$$

$\psi$ 的单位为 $M_\odot^{-2}$。PISN 计算使用共享恒星族参数的子向量

$$
\theta_{\star,\rm PISN}
\equiv
(\theta_{\rm IMF},\theta_{\rm evo},\theta_{\rm pop})
\subset\theta_\star.
$$

单位形成恒星质量产生的 PISN 期望数为

$$
\boxed{
\eta_{\rm PISN}(Z,\theta_{\star,\rm PISN})
=
\int_{M_{\min}}^{M_{\max}}
\psi(M\mid\theta_{\rm IMF})
p_{\rm PISN}(M,Z\mid\theta_{\rm evo},\theta_{\rm pop})\,dM
}
\quad [M_\odot^{-1}].
\tag{5.5}
$$

式 (5.4)--(5.5) 是单星轨道基准。双星相互作用或并合通道以恒星系统为抽样
单位。令 $M_2=qM_1$，先把系统初始质量分布限定在 $(M_1,q)$；轨道周期、
偏心率和其余演化变量全部放入后续的条件分布。系统分布的质量归一化核满足

$$
\psi_{\rm sys}(M_1,q)
=
\frac{\phi_{\rm sys}(M_1,q)}
{\displaystyle
\int dM_1\,dq\,
(M_1+qM_1)\phi_{\rm sys}(M_1,q)},
\qquad
\int(M_1+M_2)\psi_{\rm sys}\,dM_1\,dq=1.
\tag{5.5a}
$$

令 $\boldsymbol\lambda_{\rm ast}=(\boldsymbol\lambda_{\rm orb},
\boldsymbol\lambda_{\rm rem})$。双星版本将式 (5.5) 的 $dM\,\psi(M)$ 替换为
$dM_1\,dq\,\psi_{\rm sys}(M_1,q)$，再乘一次且只乘一次条件分布
$p(\boldsymbol\lambda_{\rm orb},\boldsymbol\lambda_{\rm rem}
\mid M_1,q,Z)$。令 $j$ 枚举该系统产生的终态恒星事件，

$$
N_{\rm PISN,sys}
=
\sum_j\mathcal I_{{\rm PISN},j},
\qquad
\eta_{\rm PISN,sys}
=
\int dM_1\,dq\,d\boldsymbol\lambda_{\rm ast}\,
\psi_{\rm sys}
p(\boldsymbol\lambda_{\rm ast}\mid M_1,q,Z)
N_{\rm PISN,sys}.
\tag{5.5b}
$$

并合成单一终态时该和至多为一；两颗终态星均进入命运窗的系统可贡献两次具有
不同寿命、产额和能量的 PISN。伴星质量进入单位形成总质量的分母，事件按终态
逐次计数，轨道变量也不会在系统 IMF 与演化条件分布中重复积分。

这一定义与 Gabrielli et al. (2024) 的 $dN_{\rm PISN}/dM_{\rm SFR}$
相同；其式 (7) 通过有限金属丰度恒星轨道求出进入和离开质量，并采用
Kroupa IMF 的 $0.1\,M_\odot$ 低质量端。下文数表采用截断 Salpeter IMF 与
经典零金属阶跃命运窗，属于独立解析基准
([Gabrielli et al. 2024](https://doi.org/10.1093/mnras/stae2048))。

对 $\phi(M)=AM^{-\alpha}$ 和经典阶跃命运窗，令

$$
M_a=\max(M_{\min},140M_\odot),
\qquad
M_b=\min(M_{\max},260M_\odot).
$$

当 $M_b>M_a$ 且 $\alpha\ne1,2$ 时，式 (5.5) 有解析结果

$$
\eta_{\rm PISN}
=
\frac{2-\alpha}{1-\alpha}
\,
\frac{M_b^{\,1-\alpha}-M_a^{\,1-\alpha}}
{M_{\max}^{\,2-\alpha}-M_{\min}^{\,2-\alpha}}.
\tag{5.6}
$$

$\alpha=1$ 或 $2$ 时取相应对数极限。式 (5.6) 中各质量以 $M_\odot$ 为
单位代入，结果显式带 $M_\odot^{-1}$。本文对 Salpeter
$\alpha=2.35$ 的解析复算为

| IMF 质量范围 | 经典命运窗与 IMF 交集 | $\eta_{\rm PISN}$ [$M_\odot^{-1}$] |
|---:|---:|---:|
| $1$--$500\,M_\odot$ | $140$--$260\,M_\odot$ | $2.0988\times10^{-4}$ |
| $10$--$500\,M_\odot$ | $140$--$260\,M_\odot$ | $5.5853\times10^{-4}$ |
| $50$--$150\,M_\odot$ | $140$--$150\,M_\odot$ | $3.5982\times10^{-4}$ |
| $50$--$200\,M_\odot$ | $140$--$200\,M_\odot$ | $1.2839\times10^{-3}$ |
| $50$--$260\,M_\odot$ | $140$--$260\,M_\odot$ | $1.6686\times10^{-3}$ |
| $50$--$500\,M_\odot$ | $140$--$260\,M_\odot$ | $1.3221\times10^{-3}$ |

$1.322\times10^{-3}\,M_\odot^{-1}$ 对应 Salpeter
$50$--$500\,M_\odot$ 与经典 $140$--$260\,M_\odot$ 命运窗的条件结果。
$M_{\max}>260\,M_\odot$ 后，PISN 分子保持固定，IMF 总质量继续增加，
从而使 $\eta_{\rm PISN}$ 下降。

### 5.3 单个晕中的连续期望与随机 IMF

晕在时间区间 $[t_1,t_2]$ 内形成的第 $k$ 类恒星质量为

$$
\Delta M_{\star,k}(M_h)
=
\int_{t_1}^{t_2}
\dot M_{\star,k}(M_h,t)\,dt,
\qquad
k\in\{{\rm II,III}\}.
\tag{5.7}
$$

对 $Z>0$ 的 Pop II 成星气体，定义每 dex 出生金属丰度的星形成率

$$
\dot{\mathcal M}_{\star,\rm II}(M_h,t,\log_{10}Z)
\equiv
\frac{d\dot M_{\star,\rm II}}{d\log_{10}Z},
\qquad
\int d\log_{10}Z\,\dot{\mathcal M}_{\star,\rm II}
=
\dot M_{\star,\rm II}.
\tag{5.7a}
$$

$\dot{\mathcal M}_{\star,\rm II}$ 的单位为
$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{dex}^{-1}$。
连续 IMF 给出的 PISN 期望数分别为

$$
\begin{aligned}
\lambda_{{\rm PISN},\rm III}
&=
\int_{t_1}^{t_2}dt\,
\dot M_{\star,\rm III}(M_h,t)
\eta_{{\rm PISN},\rm III}(0),\\
\lambda_{{\rm PISN},\rm II}
&=
\int_{t_1}^{t_2}dt\int d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm II}(M_h,t,\log_{10}Z)
\eta_{{\rm PISN},\rm II}(Z).
\end{aligned}
\tag{5.8}
$$

单一金属丰度且 $\eta_{\rm PISN}$ 在时间区间内固定时，
式 (5.8) 化为
$\lambda_{{\rm PISN},k}=\Delta M_{\star,k}\eta_{{\rm PISN},k}$。
当恒星形成事件采用 Poisson 点过程近似时，整数事件数满足

$$
P(N_{\rm PISN}=n\mid\lambda)
=
\frac{e^{-\lambda}\lambda^n}{n!},
\qquad
P(N_{\rm PISN}\ge1)=1-e^{-\lambda}.
\tag{5.9}
$$

严格固定总质量的逐星 IMF 抽样会引入恒星质量之间的相关性，其计数分布由抽样
算法直接给出。
小星团还需要显式保持质量预算。此时从数目归一化分布

$$
p_N(M)=
\frac{\phi(M)}
{\displaystyle\int_{M_{\min}}^{M_{\max}}\phi(M)\,dM}
$$

逐颗抽取 $M_i$。采用 stop-before 规则时，首次使累计质量越过
$\Delta M_\star$ 的恒星被舍弃，随后终止该次实现，并满足
$\sum_iM_i\le\Delta M_\star$；stop-after 或 reject-redraw 会产生不同的计数
分布。每次实现中的

$$
N_{\rm PISN}
=
\sum_i
\mathbf 1(M_a\le M_i\le M_b)
\tag{5.10}
$$

保持为整数。Wiggins et al. (2024) 系统比较了质量归一化、恒星数归一化和
Monte Carlo 三种 IMF 填充方式，并表明高质量端抽样方式可使高红移 PISN 率产生
显著散布。其有限碎裂基准还采用

$$
N_{\rm frag}
\le
\left\lfloor
9.12
\left(\frac{M_h}{10^6M_\odot}\right)^{4/3}
\left(\frac{1+z}{31}\right)^2
\right\rfloor,
\qquad
N_{\rm III}<6N_{\rm frag},
\qquad
M_{\star,\rm III}\le\epsilon_{\rm III}f_bM_h,
\tag{5.11}
$$

用于限制单晕可形成的 Pop III 恒星数和总质量。式 (5.11) 属于该研究采用的
有限碎裂模型假设；$\epsilon_{\rm III}$ 和宇宙重子分数 $f_b$ 均为无量纲量
$N_{\rm III}$ 为整数，因此严格上限等价于
$N_{{\rm III},\max}=6N_{\rm frag}-1$。
([Wiggins et al. 2024](https://arxiv.org/abs/2402.17076))。

作为数值检查，Salpeter $50$--$500\,M_\odot$ 与经典命运窗对
$\Delta M_\star=116\,M_\odot$ 给出连续期望
$\lambda_{\rm PISN}=0.15337$ 和
$P(N_{\rm PISN}\ge1)=0.14219$。严格质量预算且采用 stop-before 规则时，
$116\,M_\odot<140\,M_\odot$ 使经典 PISN 前身星的抽样概率为零。
两种结果对应不同的星团质量解释和 IMF 填充假设。百太阳质量级团块的 PISN
推断必须声明抽样规则；He II 光子产额也受同一离散高质量尾部控制。

### 5.4 恒星寿命核与单晕爆炸率

恒星在形成后经过从 ZAMS 到爆炸的延迟
$\tau_\star(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})$。
为了保留命运与寿命在同一条演化轨道上的相关性，单位形成质量的 PISN 延迟核
定义为

$$
\begin{aligned}
\mathcal K_{\rm PISN}
(\tau\mid Z,\theta_{\star,\rm PISN})
&=
\int dM\,d\boldsymbol\lambda_{\rm ast}\,
\psi(M\mid\theta_{\rm IMF})\,
p(\boldsymbol\lambda_{\rm ast}\mid M,Z,\theta_{\rm pop})\\
&\quad\times
\mathbf 1\!\left[
\mathcal F_{\rm tr}
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
= {\rm PISN}
\right]
\delta\!\left[
\tau-\tau_\star
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
\right].
\end{aligned}
\tag{5.12}
$$

$\mathcal K_{\rm PISN}$ 的单位为
$M_\odot^{-1}\,{\rm time}^{-1}$，并满足
$\int d\tau\,
\mathcal K_{\rm PISN}
(\tau\mid Z,\theta_{\star,\rm PISN})
=\eta_{\rm PISN}(Z,\theta_{\star,\rm PISN})$。
确定性单星轨道使式 (5.12) 化为
$\int dM\,\psi(M)p_{\rm PISN}(M,Z)
\delta[\tau-\tau_\star(M,Z)]$。

Schaerer (2002) 的 Table 6 给出零金属恒星寿命的解析拟合。令
$x=\log_{10}(M/M_\odot)$，无质量损失轨道满足

$$
\log_{10}\!\left(\frac{\tau_\star}{\rm yr}\right)
=
9.785-3.759x+1.413x^2-0.186x^3,
\tag{5.12a}
$$

强质量损失轨道满足

$$
\log_{10}\!\left(\frac{\tau_\star}{\rm yr}\right)
=
8.795-1.797x+0.332x^2.
\tag{5.12b}
$$

无质量损失拟合的有效质量范围为 $5$--$500\,M_\odot$，强质量损失拟合适用于
$80$--$1000\,M_\odot$。在 $140$--$260\,M_\odot$ 内，两式分别给出
$2.439$--$2.129$ Myr 和 $2.935$--$2.463$ Myr。Schaerer (2002) 将这些拟合
作为总寿命的解析近似，并估计省略 He-burning 带来的修正小于约 $10\%$；
经典无风命运窗采用无质量损失寿命作为一致基准。最终计算使用与
$p_{\rm PISN}$ 相同的恒星轨道寿命
([Schaerer 2002](https://doi.org/10.1051/0004-6361:20011619))。

令 $\mathcal H=\{M_b(t')\}$ 表示爆炸时宿主的完整并合树，$b$ 标记在形成时刻
仍存在的全部祖先支系。Pop III 与 Pop II 的源参考系单晕爆炸率分别为

$$
\begin{aligned}
R_{{\rm PISN},\rm III}(t\mid\mathcal H)
&=
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\dot M_{\star,\rm III}
\!\left[M_b(t-\tau),t-\tau\right]
\mathcal K_{{\rm PISN},\rm III}(\tau\mid0),\\
R_{{\rm PISN},\rm II}(t\mid\mathcal H)
&=
\int_{Z>0}d\log_{10}Z\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}\\
&\quad\times
\dot{\mathcal M}_{\star,\rm II}
\!\left[M_b(t-\tau),t-\tau,\log_{10}Z\right]\\
&\quad\times
\mathcal K_{{\rm PISN},\rm II}(\tau\mid Z).
\end{aligned}
\tag{5.13}
$$

式 (5.13) 对全部祖先支系求和；若 $\mathcal H$ 仅保留主支，该式给出
main-branch 近似。单位为 $\mathrm{yr}^{-1}$ per current halo。群体计算中，形成时质量
$M_{\rm form}$ 与爆炸时宿主质量 $M_{\rm expl}$ 通过后裔映射核联系：

$$
K_{\rm desc}(M_{\rm expl},t\mid M_{\rm form},t-\tau),
\qquad
\int d\log_{10}M_{\rm expl}\,K_{\rm desc}=1.
\tag{5.13a}
$$

$K_{\rm desc}$ 的单位为 dex$^{-1}$，表示一个形成宿主到爆炸时后裔的条件
质量映射；祖先多重性由式 (5.13) 的支系求和或式 (5.14) 的形成晕积分提供。
一般情形下，该核还条件化于完整装配史、环境、星形成率与出生金属丰度。
仅以 $M_{\rm form}$ 为条件的 $K_{\rm desc}$ 是对这些变量边缘化后的压缩表示。
$z>10$ 的单晕星形成可呈 Myr 级爆发，高红移小晕也可在数 Myr 内快速增长；
延迟与后裔修正的大小应由装配历史验证。

### 5.5 晕质量分布、Pop II/Pop III 分解与宇宙率

形成时 HMF 必须与寿命核处于同一时刻。本文采用空间平直的物质加宇宙常数
背景，$\Omega_{\rm m,0}+\Omega_{\Lambda,0}=1$，并在 virial overdensity
拟合中忽略辐射项：

$$
E^2(z)=\Omega_{\rm m,0}(1+z)^3+\Omega_{\Lambda,0},
\qquad
\Omega_{\rm m}(z)
=
\frac{\Omega_{\rm m,0}(1+z)^3}{E^2(z)},
\qquad
\rho_{\rm c}(z)=\rho_{\rm c,0}E^2(z).
\tag{5.13b0}
$$

本文统一采用相对于临界密度的 Bryan--Norman virial mass，记作

$$
M_\Delta\equiv M_{\rm vir}
=
\frac{4\pi}{3}\Delta_{\rm vir}(z)\rho_{\rm c}(z)R_{\rm vir}^3,
\qquad
\Delta_{\rm vir}(z)=18\pi^2+82x-39x^2,
\qquad
x=\Omega_{\rm m}(z)-1.
\tag{5.13b}
$$

$M_{\rm form}$、$M_{\rm expl}$、装配树、后裔核和质量箱均保持该约定
([Bryan & Norman 1998](https://doi.org/10.1086/305262))。若输入 HMF 使用
另一质量定义 $M_{\rm HMF}$，定义数目守恒的条件核

$$
\begin{aligned}
\int d\log_{10}M_{\rm vir}\,
K_{\rm mdef}(M_{\rm vir}\mid M_{\rm HMF},z)
&=1,\\
\frac{dn}{d\log_{10}M_{\rm vir}}
&=
\int d\log_{10}M_{\rm HMF}\,
\frac{dn}{d\log_{10}M_{\rm HMF}}\\
&\quad\times
K_{\rm mdef}(M_{\rm vir}\mid M_{\rm HMF},z).
\end{aligned}
\tag{5.13c}
$$

FoF、$M_{200\rm c}$ 或 $M_{200\rm m}$ 输入均先通过式 (5.13c) 转换；质量转换的
散布随同核传播到每个晕质量箱的 PISN 占比。

令 $\langle\cdot\rangle_{\mathcal H\mid M_{\rm form},t-\tau}$ 表示在固定形成宿主
质量和形成时刻下，对完整装配史及环境的条件平均。对 $Z>0$ 的 Pop II，爆炸时
宿主质量分布为

$$
\begin{aligned}
\frac{d\dot n_{{\rm PISN},\rm II}}
{d\log_{10}M_{\rm expl}}(M_{\rm expl},t)
={}&
\int_0^\infty d\tau\int d\log_{10}M_{\rm form}
\int d\log_{10}Z\,
\frac{dn}{d\log_{10}M_{\rm form}}(M_{\rm form},t-\tau)\\
&\times
\mathcal K_{{\rm PISN},\rm II}(\tau\mid Z)\\
&\times
\left\langle
\dot{\mathcal M}_{\star,\rm II}
(\mathcal H,t-\tau,\log_{10}Z)
K_{\rm desc}(M_{\rm expl},t\mid\mathcal H,t-\tau)
\right\rangle_{\mathcal H\mid M_{\rm form},t-\tau}.
\end{aligned}
\tag{5.14a}
$$

严格零金属 Pop III 分量为

$$
\begin{aligned}
\frac{d\dot n_{{\rm PISN},\rm III}}
{d\log_{10}M_{\rm expl}}(M_{\rm expl},t)
={}&
\int_0^\infty d\tau\int d\log_{10}M_{\rm form}\,
\frac{dn}{d\log_{10}M_{\rm form}}(M_{\rm form},t-\tau)\\
&\times
\mathcal K_{{\rm PISN},\rm III}(\tau\mid0)\\
&\times
\left\langle
\dot M_{\star,\rm III}(\mathcal H,t-\tau)
K_{\rm desc}(M_{\rm expl},t\mid\mathcal H,t-\tau)
\right\rangle_{\mathcal H\mid M_{\rm form},t-\tau}.
\end{aligned}
\tag{5.14b}
$$

$dn/d\log_{10}M_{\rm form}$ 的单位为
$\mathrm{cMpc}^{-3}\,\mathrm{dex}^{-1}$，因此式 (5.14a,b) 的单位为
$\mathrm{yr}_{\rm src}^{-1}\,\mathrm{cMpc}^{-3}\,\mathrm{dex}^{-1}$。
把条件平均分解为
$\langle\dot M_\star K_{\rm desc}\rangle
\simeq\langle\dot M_\star\rangle\langle K_{\rm desc}\rangle$
需要固定 $M_{\rm form}$ 后的条件独立近似。装配偏差、爆发性星形成和富集史会
破坏这一分解；式 (5.14a,b) 的联合条件平均保留这些相关性。
在恒星寿命、晕增长和星形成变化时标之间满足 prompt 近似时，严格
$Z=0$ Pop III 基准定义

$$
\begin{aligned}
\overline{\eta}_{{\rm PISN},\rm III}(M_h,z)
&=\eta_{{\rm PISN},\rm III}(0),\\
\overline{\eta}_{{\rm PISN},\rm II}(M_h,z)
&=
\int_{Z>0}d\log_{10}Z\,
p_{{\rm SF},\rm II}(\log_{10}Z\mid M_h,z)
\eta_{{\rm PISN},\rm II}(Z),
\end{aligned}
\tag{5.14c}
$$

其中
$\int_{Z>0}d\log_{10}Z\,p_{{\rm SF},\rm II}=1$。允许
$0<Z<Z_{\rm crit}$ 形成 Pop III 时，事件效率由成星率加权恒等式给出：

$$
\begin{aligned}
\dot M_{\star,\rm III}\overline\eta_{{\rm PISN},\rm III}
={}&
\dot M_{\star,\rm III}^{(0)}\eta_{{\rm PISN},\rm III}(0)\\
&+
\int_{0<Z<Z_{\rm crit}}d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm III}(\log_{10}Z)
\eta_{{\rm PISN},\rm III}(Z),\\
\dot M_{\star,\rm II}\overline\eta_{{\rm PISN},\rm II}
={}&
\int_{Z\ge Z_{\rm crit}}d\log_{10}Z\,
\dot{\mathcal M}_{\star,\rm II}(\log_{10}Z)
\eta_{{\rm PISN},\rm II}(Z).
\end{aligned}
\tag{5.14c'}
$$

总成星率为零时只使用乘积形式，不单独定义平均效率。式 (5.14a,b) 在 prompt
极限下化为

$$
\boxed{
\frac{d\dot n_{{\rm PISN},k}}{d\log_{10}M_\Delta}
\simeq
\frac{dn}{d\log_{10}M_\Delta}
\dot M_{\star,k}(M_\Delta,z)
\overline{\eta}_{{\rm PISN},k}(M_\Delta,z)
}.
\tag{5.14d}
$$

式 (5.14d) 清楚分离了晕丰度、每晕星形成率和单位形成质量的 PISN 效率；
Myr 级爆发、快速晕增长或窄红移箱采用式 (5.14a,b)。
积分爆炸时宿主质量得到源参考系体积率

$$
\dot n_{{\rm PISN},k}(z)
=
\int d\log_{10}M_{\rm expl}\,
\frac{d\dot n_{{\rm PISN},k}}{d\log_{10}M_{\rm expl}}.
\tag{5.15}
$$

Lazar & Bromm (2022) 的式 (1) 支持 prompt 极限下
“宇宙 SFRD $\times$ IMF 事件效率”的率；Cruz et al. (2025) 的式 (7)--(8)
支持“HMF $\times$ 单晕 Pop II/Pop III SFR”的晕分辨 SFRD。式 (5.14a--d)
把两种结构组合，并加入寿命与后裔映射
([Lazar & Bromm 2022](https://doi.org/10.1093/mnras/stac176);
[Cruz et al. 2025](https://doi.org/10.1103/PhysRevD.111.083503))。

prompt 极限下，单个晕质量处的 Pop III PISN 占比具有直接形式

$$
\boxed{
f_{\rm III}^{\rm PISN}(M_\Delta,z)
=
\frac{
\dot M_{\star,\rm III}\overline{\eta}_{{\rm PISN},\rm III}
}{
\dot M_{\star,\rm III}\overline{\eta}_{{\rm PISN},\rm III}
+
\dot M_{\star,\rm II}\overline{\eta}_{{\rm PISN},\rm II}
}
}.
\tag{5.16a}
$$

爆炸时宿主质量箱 $M_{\rm expl}\in[M_1,M_2]$ 内由第 $k$ 类恒星贡献的 PISN
比例为

$$
F_{{\rm PISN},k}(M_1,M_2,z)
=
\frac{
\displaystyle\int_{\log_{10}M_1}^{\log_{10}M_2}
d\log_{10}M_{\rm expl}\,
\frac{d\dot n_{{\rm PISN},k}}{d\log_{10}M_{\rm expl}}
}{
\displaystyle\sum_{j\in\{{\rm II,III}\}}
\int_{\log_{10}M_1}^{\log_{10}M_2}
d\log_{10}M_{\rm expl}\,
\frac{d\dot n_{{\rm PISN},j}}{d\log_{10}M_{\rm expl}}
}.
\tag{5.16b}
$$

经典 Pop-III-only 命运模型取
$\overline{\eta}_{{\rm PISN},\rm II}=0$。在任意红移的每个非零 PISN 率质量箱中，
式 (5.16a,b) 由定义恒等地给出 $f_{\rm III}^{\rm PISN}=1$。
当两类恒星具有相同的金属丰度平均效率时，PISN 占比化为第 2 节的 SFR 占比。
有限金属丰度 Pop II 通道需要先指定 Pop II IMF、
$p_{{\rm SF},\rm II}(Z\mid M_\Delta,z)$ 和
$\eta_{{\rm PISN},\rm II}(Z)$；完成这三个输入后，式 (5.16b) 才产生与第 2.4 节
相同红移的数值表。第 2.4 节的 SFR 表按成星时宿主质量 $M_{\rm form}$ 分箱，
式 (5.16b) 按爆炸时宿主质量 $M_{\rm expl}$ 分箱；两者只在 prompt 极限或显式
后裔映射后逐箱比较。若需形成宿主的 PISN 分布，应保留式 (5.14a,b) 的
$M_{\rm form}$ 轴并对 $M_{\rm expl}$ 积分。当前尚未给出有限金属 Pop II 模型在
$z>10$ 的数值 PISN 占比表。式 (5.16a,b) 描述固定晕质量或质量箱中的条件期望率占比；
随机小星团中的单次观测结果是整数事件，服从第 5.3 节的抽样分布。
总率为零的质量箱标记为“无预测事件”，占比保持未定义。

各恒星族自身的爆炸时宿主质量分布为

$$
p_k(\log_{10}M_{\rm expl}\mid{\rm PISN},z)
=
\frac{
d\dot n_{{\rm PISN},k}/d\log_{10}M_{\rm expl}
}{
\dot n_{{\rm PISN},k}(z)
},
\tag{5.17}
$$

单位为 dex$^{-1}$。第 2.4 节数表记录 SFR 占比；PISN 数值占比还需要上述
两类恒星各自的命运效率。Gabrielli et al. (2024) 的式 (1) 与式 (7) 对应
寿命核取 prompt 极限、并已经对 HMF 和宿主质量积分后的宇宙率。其恒星轨道
的主要 Pop II/I 网格覆盖绝对金属质量分数 $Z_{\rm abs}\ge10^{-4}$，半经验
星系输入用于 $0\le z\le6$；其第 4.6 节另以
$Z_{\rm abs}=10^{-11}$ 轨道和 A-SLOTH Pop III SFRD 进行独立比较。
$z>10$ 应使用高红移晕形成与富集历史。该研究的模型跨度在 $z=0$ 约为七个
数量级，在 $z=6$ 约为五个数量级，这些范围不直接外推为 $z>10$ 误差
([Gabrielli et al. 2024](https://doi.org/10.1093/mnras/stae2048))。

### 5.6 从 PISN 光谱时序到巡天控制时间

Heger--Woosley 命运网格给出完全解体事件的发生条件。巡天可见性还需要
质量依赖的光谱时序

$$
L_{\nu,e}^{\rm PISN}
(\nu_e,t_e\mid\vartheta_{\rm SN}),
\qquad
\vartheta_{\rm SN}
=
\{k,Z_{\rm birth},M_{\rm prog},R_\star,M_{\rm env},M_{^{56}{\rm Ni}},
E_{\rm kin,\infty},{\rm mixing},{\rm CSM}\},
\tag{5.18}
$$

其中 $L_{\nu,e}$ 的单位为
$\mathrm{erg\,s^{-1}\,Hz^{-1}}$，$\nu_e$ 和 $t_e$ 分别为源参考系频率和相位。
每个模板集合都应记录前身星质量网格、辐射输运版本和包层处理。

巡天预测还需要事件与视线扰动向量

$$
\begin{aligned}
\vartheta_{\rm obs}
&=
\{\vartheta_{\rm SN},\mathcal I_{\rm lens},A_{\rm host},
\tau_{\rm IGM}(\nu),\ldots\},\\
p(\vartheta_{\rm obs}\mid z,\Omega)
&=
p(\vartheta_{\rm SN}\mid z)
p(\mathcal I_{\rm lens},A_{\rm host},\tau_{\rm IGM},\ldots\\
&\hspace{7em}
\mid\vartheta_{\rm SN},z,\Omega).
\end{aligned}
\tag{5.18a}
$$

写成
$\vartheta_{\rm SN}=(k,Z_{\rm birth},\vartheta_{\rm cont})$ 后，其积分采用离散--
连续混合测度

$$
\int d\vartheta_{\rm SN}\,g
\equiv
\sum_{k\in\{\rm II,III\}}
\left[
\left.\int d\vartheta_{\rm cont}\,g\right|_{Z=0}
+
\int_{Z>0}d\log_{10}Z
\int d\vartheta_{\rm cont}\,g
\right],
\tag{5.18b0}
$$

其中不属于给定恒星族的测度分量取零；严格基准的 Pop III 只有 $Z=0$ 原子项。
光变模板权重由同一个 PISN 事件率核确定。令
$d^2\dot n_{\rm PISN}/(d\log_{10}M_{\rm expl}\,d\vartheta_{\rm SN})$
表示在式 (5.14a,b) 中保留命运轨道、出生金属丰度和爆炸参数后的联合率，则

$$
p(\vartheta_{\rm SN}\mid z)
=
\frac{1}{\dot n_{\rm PISN}(z)}
\int d\log_{10}M_{\rm expl}\,
\frac{d^2\dot n_{\rm PISN}}
{d\log_{10}M_{\rm expl}\,d\vartheta_{\rm SN}},
\qquad
\int d\vartheta_{\rm SN}\,p(\vartheta_{\rm SN}\mid z)=1.
\tag{5.18b}
$$

式 (5.18b) 使 IMF、命运概率、寿命、Pop II/Pop III 身份、宿主晕质量与光变
模板保持同一事件率权重。任意外加的模板先验会改变探测率，需作为替代模型单独
列出。

银河系消光 $A_{\rm MW}(\nu,\Omega)$ 由天空位置图给定，其标定误差可继续作为
式 (5.18a) 的扰动量。联合条件分布保留爆炸能量、宿主尘埃、透镜放大和再电离
视线之间可能存在的相关性。

令多像透镜配置为

$$
\mathcal I_{\rm lens}
=
\{(|\mu_i|,\Delta t_i,\Omega_{{\rm img},i})\}_{i=1}^{N_{\rm img}}.
\tag{5.18c}
$$

以下 $\Omega$ 与 $d\Omega$ 采用源平面坐标和源平面立体角。输入巡天遮罩与透镜
图先做射线追踪，并在源平面取遮罩的几何并集；单个像分支局部满足
$d\Omega_{\rm src}=d\Omega_{{\rm img},i}/|\mu_i|$，多重覆盖区只保留一次。
$dV_c/(dz\,d\Omega)$ 按该唯一源平面体积元计算。同一物理爆发的多重像作为
一个源事件进入计数，全部像的放大率、时间延迟、天空位置和季节窗口共同进入
式 (5.21a) 的选择函数。

Kasen, Woosley & Heger (2011) 使用时变多波段辐射输运计算红超巨星和裸氦核
PISN，得到持续数百天且亮度跨越很大的光变族。峰值、颜色和时标随前身星
质量、半径、包层与 $^{56}$Ni 产额共同变化。Whalen et al. (2013) 进一步
计算了周星环境、激波传播和高红移辐射转移
([Kasen, Woosley & Heger 2011](https://doi.org/10.1088/0004-637X/734/2/102);
[Whalen et al. 2013](https://doi.org/10.1088/0004-637X/777/2/110))。

源参考系谱光度投影到观测频率 $\nu_o$ 后为

$$
\begin{aligned}
F_{\nu_o,i}(t_{\rm obs})
&=
\frac{(1+z)|\mu_i|}{4\pi D_L^2(z)}
L_{\nu_e}
\!\left(
\nu_e=(1+z)\nu_o,\,
t_e=\frac{t_{\rm obs}-t_{0,\rm obs}-\Delta t_i}{1+z}
\right)\\
&\quad\times e^{-\tau_{\rm IGM}(\nu_o,z)}\\
&\quad\times
10^{-0.4[A_{\rm host}(\nu_e)+A_{\rm MW}(\nu_o,\Omega_{{\rm img},i})]}.
\end{aligned}
\tag{5.19}
$$

$t_{0,\rm obs}$ 为观测者参考系爆炸历元，$\mu_i$ 和 $\Delta t_i$ 为第 $i$ 幅像的
放大率和相对时间延迟，$A_{\rm host}$ 与 $A_{\rm MW}$ 以星等表示宿主和银河系
消光。$F_{\nu_o,i}$ 的单位为
$\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}$。IGM 平均透射可采用
Inoue et al. (2014) 的统计处方；$z>6$ 的再电离拓扑和视线差异需要额外散布参数
([Inoue et al. 2014](https://doi.org/10.1093/mnras/stu936))。

式 (5.19) 随 photon-counting 系统响应积分后产生

$$
\langle F_{\nu,i}\rangle_b
=
\frac{
\displaystyle\int F_{\nu,i}(\nu)R_b(\nu)\,d\nu/\nu
}{
\displaystyle\int R_b(\nu)\,d\nu/\nu
},
\qquad
m_{{\rm AB},b,i}
=
-2.5\log_{10}
\left(
\frac{\langle F_{\nu,i}\rangle_b}{3631\,{\rm Jy}}
\right).
\tag{5.20}
$$

$R_b(\nu)$ 是包含光学传输、探测器量子效率和带通版本的无量纲完整系统响应。
式 (5.19) 的 luminosity-distance 约定采用
[Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116)，式 (5.20) 的
photon-counting 带通与 AB 约定采用
[Hogg et al. (2002)](https://arxiv.org/abs/astro-ph/0210394)。

给定巡天历元、系统响应和噪声模型
$\mathcal S=\{t_j,R_j,\sigma_j,{\rm reference},{\rm season}\}$，爆炸历元为
$t_{0,\rm obs}$ 的探测与分类效率定义为

$$
\epsilon_{\rm sel}(z,\vartheta_{\rm obs},t_{0,\rm obs},\Omega\mid\mathcal S)
=
\int d\widehat{\mathbf F}\,
p\!\left(
\widehat{\mathbf F}\mid
\mathbf F[z,\vartheta_{\rm obs},t_{0,\rm obs},\Omega;\mathcal S]
\right)
\mathbf 1[{\rm selection\ passed}],
\tag{5.21a}
$$

其中 $\widehat{\mathbf F}$ 是全部透镜像、历元和滤镜的联合测量通量向量；
$\mathbf 1[\mathrm{selection\ passed}]$ 对同一源执行一次分类，包含“至少一幅像
通过”及多像联合条件。观测者参考系控制时间对可能的爆炸历元积分：

$$
T_{\rm ctrl,obs}(z,\vartheta_{\rm obs},\Omega\mid\mathcal S)
=
\int dt_{0,\rm obs}\,
\epsilon_{\rm sel}
(z,\vartheta_{\rm obs},t_{0,\rm obs},\Omega\mid\mathcal S).
\tag{5.21b}
$$

式 (5.21b) 的单位为观测者年。实际效率可由注入--恢复实验的
$N_{\rm rec}/N_{\rm inj}$ 标定，从而同时包含差分成像、参考图、季节窗口、
颜色和分类条件。de Souza et al. (2013)
将宇宙学形成史、辐射输运和观测过程放入同一合成巡天；
Hartwig, Bromm & Loeb (2018) 进一步用可见时间优化 JWST 滤波器和曝光策略
([de Souza et al. 2013](https://doi.org/10.1093/mnras/stt1680);
[Hartwig, Bromm & Loeb 2018](https://doi.org/10.1093/mnras/sty1576))。

### 5.7 观测者事件率、单历元数量与时间膨胀

令 $\dot n_{\rm PISN}(z)$ 为源参考系总共动体积率，单位为
$\mathrm{yr}_{\rm src}^{-1}\,\mathrm{cMpc}^{-3}$。式 (5.18a) 的事件与视线分布
满足
$\int d\vartheta_{\rm obs}\,
p(\vartheta_{\rm obs}\mid z,\Omega)=1$。
单位观测者时间、红移和立体角的新事件率为

$$
\boxed{
\frac{d^3N_{\rm new}}
{dt_{\rm obs}\,dz\,d\Omega}
=
\int d\vartheta_{\rm obs}\,
\frac{\dot n_{\rm PISN}(z)
p(\vartheta_{\rm obs}\mid z,\Omega)}{1+z}
\frac{dV_c}{dz\,d\Omega}
}
\tag{5.22}
$$

其中 $dV_c/(dz\,d\Omega)$ 为共动体积元，单位为
$\mathrm{cMpc}^3\,\mathrm{sr}^{-1}$。因为
$dt_{\rm obs}=(1+z)dt_{\rm src}$，巡天期望探测数为

$$
\boxed{
\mu_{\rm det}
=
\int dz\,d\Omega\,d\vartheta_{\rm obs}\,
\frac{\dot n_{\rm PISN}(z)
p(\vartheta_{\rm obs}\mid z,\Omega)}{1+z}
\frac{dV_c}{dz\,d\Omega}
T_{\rm ctrl,obs}(z,\vartheta_{\rm obs},\Omega\mid\mathcal S)
}
\tag{5.23}
$$

Pan, Kasen & Loeb (2012) 的式 (6)、Weinmann & Lilly (2005) 的式 (4) 和
Lazar & Bromm (2022) 的式 (1) 均给出 $(1+z)^{-1}$ 新事件率
([Weinmann & Lilly 2005](https://doi.org/10.1086/428106))。
对单次观测历元，观测者可见时间
$t_{\rm vis,obs}=(1+z)t_{\rm vis,rest}$ 与事件率中的时间膨胀相消。若
$t_{\rm vis,rest}$ 由式 (5.21a) 的同一选择条件定义，则

$$
N_{\rm snap}
=
\int dz\,d\Omega\,d\vartheta_{\rm obs}\,
\dot n_{\rm PISN}(z)p(\vartheta_{\rm obs}\mid z,\Omega)
t_{\rm vis,rest}(z,\vartheta_{\rm obs},\Omega\mid\mathcal S)
\frac{dV_c}{dz\,d\Omega}.
\tag{5.24}
$$

这对应 Pan, Kasen & Loeb (2012) 的式 (7)
([Pan, Kasen & Loeb 2012](https://doi.org/10.1111/j.1365-2966.2012.20837.x))。
式 (5.24) 还采用 $\dot n_{\rm PISN}$ 在 $t_{\rm vis,rest}$ 内近似恒定的条件；
快速演化时应对爆炸时刻继续卷积。把 $t_{\rm vis,rest}$ 取为空间均匀的单一
阈值只适用于理想 snapshot；真实滤镜、噪声、透镜图和天空遮罩使用
$t_{\rm vis,rest}(z,\vartheta_{\rm obs},\Omega\mid\mathcal S)$。
Wiggins et al. (2024) 第 2.2 节在把源时间率转换为每观测者年时写入乘法
$(1+z)$。相对于式 (5.22) 的规范换算，其绝对观测者事件率高出
$(1+z)^2$。本文仅引用该研究的随机 IMF 方法和同一红移下的相对趋势，不采用
其绝对观测者率。若比较其图示绝对率，需相对原图乘以 $(1+z)^{-2}$；本文尚未
生成这项修正后的曲线或数表。

### 5.8 计数似然与观测边界

把事件划分到互斥的红移或候选类别后，条件独立计数箱的 PISN 似然为

$$
\ln\mathcal L_{\rm PISN}
=
\sum_b
\left[
N_b\ln\mu_b-\mu_b-\ln\Gamma(N_b+1)
\right],
\tag{5.25}
$$

其中 $\mu_b$ 来自式 (5.23)。同一事件的多滤镜测量共同进入式 (5.21a) 的选择
向量，避免在多个滤镜箱中重复计数。若候选包含 PPISN/CSM、Type IIn、magnetar
或测光污染，则

$$
\mu_b
=
\mu_{{\rm PISN},b}
+\mu_{{\rm PPISN/CSM},b}
+\mu_{{\rm IIn},b}
+\mu_{{\rm magnetar},b}
+\mu_{{\rm contamination},b}.
\tag{5.26}
$$

背景率、分类概率、测光红移和选择效率的不确定度作为扰动参数边缘化。
零探测且背景可忽略时，95% Poisson 上限满足
$e^{-\mu_{95}}=0.05$，即 $\mu_{95}=2.996$。体积率上限由
$\dot n_{\rm PISN}<2.996/\langle VT\rangle$ 给出，其中
$\langle VT\rangle$ 是式 (5.23) 中选择函数加权的有效时空体积，单位为
$\mathrm{cMpc}^3\,\mathrm{yr}_{\rm src}$。这一标量上限还要求体积率在敏感
红移域内取常数；对快速红移演化模型，Poisson 似然约束预设率形状的归一化。
Moriya et al. (2021) 用四季 HSC 搜索和巡天模拟，将
$z\lesssim3$、高光度、观测者系持续时间超过一年的 Kasen 模板事件率限制到约
$100\,{\rm Gpc}^{-3}\,{\rm yr}^{-1}$；该数值的适用域由其光变族和选择函数限定
([Moriya et al. 2021](https://doi.org/10.3847/1538-4357/abcfc0))。

高红移预测对 IMF 填充和宿主模型高度敏感。Hartwig et al. (2018) 对最亮的
$225/250\,M_\odot$ Kasen 模板，采用 F200W+F356W、每带 600 s 和双带
$S/N>10$，得到每年至少需要约 $5\times10^4$ 个 JWST 视场才期望发现一个
$z\lesssim7.5$ PISN；
Venditti et al. (2024) 在所研究的 $z\simeq8$ JWST fields 中得到平均探测数
小于一，并在乐观的约 $1\,{\rm deg}^2$ Roman 视场中达到
$\mathcal O(1)$，该结果强烈依赖 IMF 与星形成效率。Wiggins et al. (2024)
支持不同 Pop III IMF 和填充规则之间的相对趋势；其绝对观测者率不进入本文
的数值基准
([Hartwig, Bromm & Loeb 2018](https://doi.org/10.1093/mnras/sty1576);
[Venditti et al. 2024](https://doi.org/10.1093/mnras/stad3513);
[Wiggins et al. 2024](https://arxiv.org/abs/2402.17076))。

这些巡天结果覆盖到 $z\simeq8$ 左右，尚未形成 $z>10$ PISN 率的直接观测上限。

由此，完整 PISN 计算的最小闭环为

$$
\boxed{
\begin{aligned}
\{\dot M_{\star,\rm III},
\dot{\mathcal M}_{\star,\rm II}(\log_{10}Z)\}
\;&\rightarrow
\mathcal F_{\rm tr}(M,Z,\boldsymbol\lambda_{\rm ast})
\rightarrow
\mathcal K_{\rm PISN}(\tau)\\
&\rightarrow
\frac{d\dot n_{{\rm PISN},k}}{d\log_{10}M_{\rm expl}}
\rightarrow
L_{\nu,e}(\nu_e,t_e)\\
&\rightarrow
T_{\rm ctrl,obs}
\rightarrow
\mu_{\rm det}
\rightarrow
\mathcal L_{\rm PISN}.
\end{aligned}
}
\tag{5.27}
$$

## 6. PISN 核合成产额与化学反馈

本节把单星轨道输出重构为群体延迟核和环境反馈闭合。Heger & Woosley (2002)
直接提供零金属单星的整体产额、$^{56}{\rm Ni}$ 质量、奇偶效应和渐近动能；
式 (6.2)--(6.4) 是本文据此定义的群体核，式 (6.11)--(6.12) 是待模拟标定的
环境闭合。上述文献不提供这些群体核或环境参数的通用拟合。

恒星水动力轨道在确认完全解体后，还给出同一条轨道上的元素抛射量和渐近动能。
爆炸前恒星风连续返回物质，完全解体在寿命终点瞬时返回其余非束缚质量层。
令 $X_{a,\rm surf}(\tau)$ 为风物质中核素 $a$ 的质量分数，
$X_a^{\rm post}(m)$ 为爆炸后质量层的质量分数，
$\mathcal U(m)$ 为最终非束缚质量层的指示量。两种返回通道分别为

$$
\dot m_{a,\rm w}(\tau)
=
X_{a,\rm surf}(\tau)\dot M_{\rm w}(\tau),
\qquad
M_{a,\rm ej}^{\rm expl}
=
\int_0^{M_{\star,\rm preSN}}
X_a^{\rm post}(m)\mathcal U(m)\,dm.
\tag{6.1a}
$$

$\dot M_{\rm w}\ge0$ 表示真正离开恒星系统并进入环境的质量损失率；双星内部的
保守 Roche-lobe overflow 不计入该量，非保守传质和共同包层仅计实际逸出部分。
$m$ 和 $M_{\star,\rm preSN}$ 以 $M_\odot$ 表示，$X_a$ 与 $\mathcal U$ 无量纲。
$\dot m_{a,\rm w}$ 的单位为 $M_\odot\,{\rm yr}^{-1}$，
$M_{a,\rm ej}^{\rm expl}$ 的单位为 $M_\odot$。对应的总核素返回量、总质量返回量
和净新生核素产额为

$$
\begin{aligned}
M_{a,\rm ret}
&=
\int_0^{\tau_\star}\dot m_{a,\rm w}(\tau)\,d\tau
+M_{a,\rm ej}^{\rm expl},\\
M_{\rm ret}
&=
\int_0^{\tau_\star}\dot M_{\rm w}(\tau)\,d\tau
+
\int_0^{M_{\star,\rm preSN}}\mathcal U(m)\,dm,\\
p_a^{\rm net}
&=
M_{a,\rm ret}
-X_{a,\rm birth}M_{\rm ret}.
\end{aligned}
\tag{6.1b}
$$

本节的化学守恒采用总返回量 $M_{a,\rm ret}$；用于丰度比较的表格需声明采用
放射性衰变前或衰变后的产额。下文采用衰变后的稳定元素产额，并为光变计算
单独保留 $^{56}{\rm Ni}$。对严格 $Z=0$ 的重元素，
$X_{a,\rm birth}=0$，总返回量与净新生量相同。经典零金属无风基准满足
$\dot M_{\rm w}=0$；有限金属和旋转轨道保留连续风项。

若 $dm_{\rm cgs}$ 以克为单位，爆炸物质的渐近动能为

$$
E_{\rm kin,\infty}
=
\int_{\rm ejecta}
\frac{v_\infty^2(m)}{2}\,dm_{\rm cgs}.
\tag{6.1c}
$$

$E_{\rm kin,\infty}$ 的单位为 erg。它是恒星爆炸轨道的输出；耦合到晕气体的
比例由爆炸位置、宿主晕和环境状态决定，并在式 (6.4) 的反馈注入层施加。

经典一维完全 PISN 对全部预超新星质量层满足 $\mathcal U=1$，并且无致密残骸。
Heger & Woosley (2002) 和 Umeda & Nomoto (2002) 通过爆炸性氧、硅燃烧网络
计算了零金属 PISN 子网格的元素产额；Heger--Woosley 网格在高质量端可合成
约 $57\,M_\odot$ 的 $^{56}{\rm Ni}$，产生显著奇偶效应，并且几乎不产生
锌以上元素
([Heger & Woosley 2002](https://doi.org/10.1086/338487);
[Umeda & Nomoto 2002](https://doi.org/10.1086/323946))。

单星命运、寿命、风、产额和能量由同一
$(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})$ 轨道联合决定。定义

$$
\begin{aligned}
\mathcal I_{\rm PISN}
&\equiv
\mathbf 1\!\left[
\mathcal F_{\rm tr}
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
= {\rm PISN}
\right],\\
d\Pi_k
&\equiv
dM\,d\boldsymbol\lambda_{\rm ast}\,
\psi_k(M\mid\theta_{\rm IMF})
p_k(\boldsymbol\lambda_{\rm ast}\mid
M,Z,\theta_{\rm pop})\mathcal I_{\rm PISN}.
\end{aligned}
\tag{6.1d}
$$

双星通道采用与式 (5.5a) 一致且互斥的系统测度

$$
\begin{aligned}
d\Pi_{k,\rm sys}
\equiv{}&
dM_1\,dq\,d\boldsymbol\lambda_{\rm orb}\,
d\boldsymbol\lambda_{\rm rem}\,
\psi_{\rm sys}(M_1,q)\\
&\times
p_k(\boldsymbol\lambda_{\rm orb},\boldsymbol\lambda_{\rm rem}
\mid M_1,q,Z).
\end{aligned}
\tag{6.1e}
$$

$\psi_{\rm sys}$ 只承载 $(M_1,q)$ 的系统 IMF，轨道与其余演化变量只出现在
条件分布中。定义系统内的 PISN 终态集合
$\mathcal J_{\rm PISN}=\{j:\mathcal I_{{\rm PISN},j}=1\}$。系统事件数、爆炸
核与风核分别采用

$$
\begin{aligned}
\mathcal K_{{\rm PISN},k}^{\rm sys}(\tau\mid Z)
&=
\int d\Pi_{k,\rm sys}
\sum_{j\in\mathcal J_{\rm PISN}}
\delta(\tau-\tau_j),\\
\mathcal K_{a,k}^{{\rm expl,sys}}(\tau\mid Z)
&=
\int d\Pi_{k,\rm sys}
\sum_{j\in\mathcal J_{\rm PISN}}
M_{a,{\rm ej},j}^{\rm expl}\delta(\tau-\tau_j),\\
\mathcal K_{a,k}^{{\rm wind,sys}}(\tau\mid Z)
&=
\int d\Pi_{k,\rm sys}
\sum_{j\in\mathcal J_{\rm PISN}}
\dot m_{a,{\rm w},j}(\tau)\Theta(\tau_j-\tau),\\
\mathcal K_{E,k}^{{\rm expl,sys}}(\tau\mid Z)
&=
\int d\Pi_{k,\rm sys}
\sum_{j\in\mathcal J_{\rm PISN}}
E_{{\rm kin},\infty,j}\delta(\tau-\tau_j).
\end{aligned}
\tag{6.1f}
$$

风能核按式 (6.1f) 的同一事件和，把
$\dot m_{a,{\rm w},j}$ 换成 $\dot E_{{\rm w,yr},j}$。因此宽双星的两次爆炸
保持各自的 $\tau_j$，并合终态自然只保留一个事件。以下先写单星形式；采用
双星通道时用式 (6.1f) 的系统核替换对应单星核。第 $k$ 类恒星的风返回核与
爆炸返回核为

$$
\begin{aligned}
\mathcal K_{a,k}^{\rm wind}
(\tau\mid Z,\theta_{\star,\rm PISN})
&=
\int d\Pi_k\,
\dot m_{a,\rm w}
(\tau\mid M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
\Theta(\tau_\star-\tau),\\
\mathcal K_{a,k}^{\rm expl}
(\tau\mid Z,\theta_{\star,\rm PISN})
&=
\int d\Pi_k\,
M_{a,\rm ej}^{\rm expl}
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})\\
&\quad\times
\delta\!\left[
\tau-\tau_\star
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
\right].
\end{aligned}
\tag{6.2a}
$$

总核素返回核为
$\mathcal K_{a,k}^{\rm ret}=\mathcal K_{a,k}^{\rm wind}
+\mathcal K_{a,k}^{\rm expl}$。将式 (6.2a) 中的
$\dot m_{a,\rm w}$ 和 $M_{a,\rm ej}^{\rm expl}$ 分别替换为
$\dot M_{\rm w}$ 和
$M_{\rm ej}^{\rm expl}=\int\mathcal U(m)\,dm$，得到总质量返回核

$$
\mathcal K_{{\rm ret},k}
=
\mathcal K_{{\rm ret},k}^{\rm wind}
+
\mathcal K_{{\rm ret},k}^{\rm expl}.
\tag{6.2b}
$$

延迟变量 $\tau$ 以下统一以 yr 为单位。若
$\dot M_{\rm w,cgs}$ 以 $\mathrm{g\,s^{-1}}$ 表示，先定义每年的风机械能

$$
\dot E_{\rm w,yr}
=
\frac{1}{2}\dot M_{\rm w,cgs}v_{\rm w,\infty}^2T_{\rm yr},
\qquad
T_{\rm yr}=3.15576\times10^7\ \mathrm{s\,yr^{-1}},
\tag{6.2c0}
$$

其单位为 $\mathrm{erg\,yr^{-1}}$。爆炸渐近动能采用式 (6.1c)。对应的原始
机械能核为

$$
\begin{aligned}
\mathcal K_{E,k}^{\rm wind}
(\tau\mid Z,\theta_{\star,\rm PISN})
&=
\int d\Pi_k\,
\dot E_{\rm w,yr}
(\tau\mid M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
\Theta(\tau_\star-\tau),\\
\mathcal K_{E,k}^{\rm expl}
(\tau\mid Z,\theta_{\star,\rm PISN})
&=
\int d\Pi_k\,
E_{\rm kin,\infty}
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})\\
&\quad\times
\delta\!\left[
\tau-\tau_\star
(M,Z,\boldsymbol\lambda_{\rm ast};\theta_{\rm evo})
\right],\\
\mathcal K_{E,k}^{\rm mech}
&=
\mathcal K_{E,k}^{\rm wind}
+\mathcal K_{E,k}^{\rm expl}.
\end{aligned}
\tag{6.2c}
$$

经典零金属无风基准满足
$\mathcal K_{E,k}^{\rm wind}=0$。有限金属或旋转模型在结果中分别报告
“最终成为 PISN 的前身星风”与“PISN 爆炸”两项，二者之和构成该命运通道的
机械能预算。

环境能量分配在单星积分内施加。对
$c\in\{{\rm wind,expl}\}$ 和
$x\in\{{\rm host,esc}\}$，令
$f_{x,E}^{c}(M,Z,\boldsymbol\lambda_{\rm ast},\mathcal E)$
分别表示留在源晕气体中的能量比例和越过源晕边界的能量比例，并满足

$$
f_{{\rm host},E}^{c}
+f_{{\rm esc},E}^{c}
+f_{{\rm loss},E}^{c}
=1,
\qquad
f_{{\rm host},E}^{c},f_{{\rm esc},E}^{c},f_{{\rm loss},E}^{c}\ge0.
\tag{6.2d}
$$

$f_{{\rm loss},E}^{c}$ 汇总辐射冷却和未进入所选宿主相或外逸流的耗散能量，
从而使每条反馈通道的能量分区闭合。

环境条件核显式写为

$$
\begin{aligned}
\mathcal K_{E,k}^{x}
(\tau\mid Z,\theta_{\star,\rm PISN},\mathcal E)
={}&
\int d\Pi_k\,
f_{x,E}^{\rm wind}(\ldots,\mathcal E)
\dot E_{\rm w,yr}(\tau;\ldots)\Theta(\tau_\star-\tau)\\
&+
\int d\Pi_k\,
f_{x,E}^{\rm expl}(\ldots,\mathcal E)
E_{\rm kin,\infty}(\ldots)
\delta[\tau-\tau_\star(\ldots)],
\quad x\in\{{\rm host,esc}\}.
\end{aligned}
\tag{6.2e}
$$

这一构造保留 $E_{\rm kin,\infty}$、$M_{\rm ej}$、爆炸位置和命运轨道之间的
联合相关性。把平均耦合比例乘到 $\mathcal K_E^{\rm mech}$ 外只适用于
$f_{x,E}^{c}$ 与这些单星量条件独立的可分离近似。

$\mathcal K_{a,k}^{\rm ret}$ 和 $\mathcal K_{{\rm ret},k}$ 的量纲为
$\mathrm{yr}^{-1}$；$\mathcal K_{E,k}^{\rm mech}$ 与
$\mathcal K_{E,k}^{x}$ 的单位为
$\mathrm{erg}\,M_\odot^{-1}\,\mathrm{yr}^{-1}$。事件数核
$\mathcal K_{\rm PISN}$ 的单位为
$M_\odot^{-1}\,\mathrm{yr}^{-1}$。积分延迟核得到每单位形成质量的总量：

$$
\begin{aligned}
y_{a,k}^{\rm ret}(Z)
&=
\int d\tau\,\mathcal K_{a,k}^{\rm ret}(\tau\mid Z),\\
R_k^{\rm PISN}(Z)
&=
\int d\tau\,\mathcal K_{{\rm ret},k}(\tau\mid Z),\\
\varepsilon_{E,k}^{\rm mech}(Z)
&=
\int d\tau\,\mathcal K_{E,k}^{\rm mech}(\tau\mid Z).
\end{aligned}
\tag{6.3}
$$

$y_{a,k}^{\rm ret}$ 和 $R_k^{\rm PISN}$ 无量纲，
$\varepsilon_{E,k}^{\rm mech}$ 的单位为 $\mathrm{erg}\,M_\odot^{-1}$。

沿完整晕装配树 $\mathcal H$，严格零金属 Pop III 在外流分配前的元素注入率和
总质量注入率为

$$
\begin{aligned}
\dot M_{a,{\rm PISN},\rm III}^{\rm inj}(t\mid\mathcal H)
&=
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\dot M_{\star,\rm III}
\!\left[M_b(t-\tau),t-\tau\right]
\mathcal K_{a,\rm III}^{\rm ret}
(\tau\mid0,\theta_{\star,\rm PISN}),\\
\dot M_{{\rm ret},{\rm PISN},\rm III}^{\rm inj}(t\mid\mathcal H)
&=
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\dot M_{\star,\rm III}
\!\left[M_b(t-\tau),t-\tau\right]
\mathcal K_{{\rm ret},\rm III}
(\tau\mid0,\theta_{\star,\rm PISN}).
\end{aligned}
\tag{6.4a}
$$

令
$\mathcal E_b(t)=\{M_{{\rm expl},b}(t),z(t),n_{{\rm amb},b}(t),
r_{{\rm SN},b}(t),\ldots\}$
表示反馈实际注入时刻 $t$ 所处后裔晕的环境。连续风项逐时使用
$\mathcal E_b(t)$；爆炸项的 $\delta(\tau-\tau_\star)$ 自动选取爆炸时环境。
留在源晕所选气体相中的机械能注入率为

$$
\begin{aligned}
\dot E_{{\rm host},{\rm PISN},\rm III}(t\mid\mathcal H)
&=
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\dot M_{\star,\rm III}
\!\left[M_b(t-\tau),t-\tau\right]
\mathcal K_{E,\rm III}^{\rm host}
(\tau\mid0,\theta_{\star,\rm PISN},\mathcal E_b(t)).
\end{aligned}
\tag{6.4b}
$$

定义 Pop II 连续金属丰度积分的下限

$$
Z_{\rm II,min}
=
\begin{cases}
0^+, & \text{严格基准：Pop III 仅含 }Z=0,\\
Z_{\rm crit}, & \text{允许 }0<Z<Z_{\rm crit}\text{ 形成 Pop III}.
\end{cases}
\tag{6.4c0}
$$

Pop II 在外流分配前的元素与总质量注入率为

$$
\mathscr S_{{\rm II},b}(t-\tau,Z)
\equiv
\dot{\mathcal M}_{\star,\rm II}
\!\left[M_b(t-\tau),t-\tau,\log_{10}Z\right],
\qquad
[\mathscr S_{{\rm II},b}]
=M_\odot\,\mathrm{yr}^{-1}\,\mathrm{dex}^{-1}.
\tag{6.4c1}
$$

于是

$$
\begin{aligned}
\dot M_{a,{\rm PISN},\rm II}^{\rm inj}(t\mid\mathcal H)
&=
\int_{Z\ge Z_{\rm II,min}}d\log_{10}Z
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\mathscr S_{{\rm II},b}(t-\tau,Z)
\mathcal K_{a,\rm II}^{\rm ret}
(\tau\mid Z,\theta_{\star,\rm PISN}),\\
\dot M_{{\rm ret},{\rm PISN},\rm II}^{\rm inj}(t\mid\mathcal H)
&=
\int_{Z\ge Z_{\rm II,min}}d\log_{10}Z
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}
\mathscr S_{{\rm II},b}(t-\tau,Z)
\mathcal K_{{\rm ret},\rm II}
(\tau\mid Z,\theta_{\star,\rm PISN}).
\end{aligned}
\tag{6.4c}
$$

留在源晕所选气体相中的机械能注入率为

$$
\begin{aligned}
\dot E_{{\rm host},{\rm PISN},\rm II}(t\mid\mathcal H)
&={}
\int_{Z\ge Z_{\rm II,min}}d\log_{10}Z
\int_0^\infty d\tau\,
\sum_{b\in{\rm Prog}(\mathcal H,t-\tau)}\\
&\quad\times
\mathscr S_{{\rm II},b}(t-\tau,Z)\\
&\quad\times
\mathcal K_{E,\rm II}^{\rm host}
(\tau\mid Z,\theta_{\star,\rm PISN},\mathcal E_b(t)).
\end{aligned}
\tag{6.4d}
$$

式 (6.4a,b) 给出严格 $Z=0$ Pop III 基准。启用亚临界有限金属 Pop III 时，
按照式 (5.0) 和 (5.14c') 向这两式加入
$0<Z<Z_{\rm crit}$ 的 Pop III 金属丰度卷积，同时令式 (6.4c,d) 采用
$Z_{\rm II,min}=Z_{\rm crit}$。式 (6.4a--d) 与事件率式 (5.13) 使用同一
出生金属丰度、寿命和祖先支系，因此爆炸数、元素返回和机械反馈具有一致的
时间归属。质量注入率的单位为 $M_\odot\,{\rm yr}^{-1}$；机械能注入率在
时间核以 yr 计时的单位为 $\mathrm{erg}\,{\rm yr}^{-1}$，除以
$3.15576\times10^7\,{\rm s\,yr}^{-1}$ 后得到 $\mathrm{erg}\,{\rm s}^{-1}$。
对晕内气体中的核素 $a$，质量守恒采用延迟总返回约定：

$$
\begin{aligned}
\frac{dM_{\rm g}}{dt}
=\;&
-\dot M_\star
+\dot M_{\rm ret}^{\rm gross}
+\dot M_{\rm in}
-\dot M_{\rm out}
+\dot M_{\rm reaccr},\\
\frac{dM_{a,\rm g}}{dt}
=\;&
\dot M_{a,\rm ret}^{\rm gross}
+X_{a,\rm in}\dot M_{\rm in}
-X_{a,\rm g}\dot M_\star
-X_{a,\rm out}\dot M_{\rm out}
+\dot M_{a,\rm reaccr},
\end{aligned}
\tag{6.5}
$$

其中
$\dot M_{a,\rm ret}^{\rm gross}
=\sum_k\dot M_{a,{\rm PISN},k}^{\rm inj}
+\dot M_{a,\rm CCSN}
+\dot M_{a,\rm AGB}
+\dot M_{a,\rm other}$
包含所有通道返回的原有核素和新合成核素；
$\dot M_{\rm ret}^{\rm gross}$ 是对应的总返回质量。
$X_{a,\rm in}$、$X_{a,\rm g}$ 和 $X_{a,\rm out}$ 分别为流入、冷气体和流出物质的
核素质量分数。总金属质量由所有重元素的 $M_{a,\rm g}$ 求和。绝对金属质量分数
和太阳单位金属丰度分别定义为

$$
Z_{\rm g,abs}
=
\frac{M_{Z,\rm g}}{M_{\rm g}},
\qquad
Z_{\rm g}
=
\frac{Z_{\rm g,abs}}{Z_{\odot,\rm ref}}.
\tag{6.5a}
$$

式 (6.5) 与式 (6.1a,b) 均采用总返回量约定，保持气体质量与核素质量的一致
记账。除带有“abs”下标的量外，下文 $p_{\rm SF}$、$Z_{\rm crit}$、
$Z_{\rm bub}$、阈值指示函数与目标俘获条件中的 $Z$ 均表示无量纲的
$Z_{\rm abs}/Z_{\odot,\rm ref}$。

源晕与外部富集共享同一个越过晕边界的示踪账本。令 $e$ 标记一个确定的 PISN
前身星，$c\in\{\mathrm{wind},\mathrm{expl}\}$ 标记连续前身星风与最终爆炸。
对每个互斥通道定义单前身星响应窗口
$\mathcal W_{e,\rm wind}=[t_{{\rm form},e},t_{{\rm expl},e})$ 和只包含该次爆炸
响应的 $\mathcal W_{e,\rm expl}$。逃逸质量为

$$
\begin{aligned}
M_{a,\rm esc}^{c,e}
&=
\int_{\mathcal W_{e,c}}dt\,
X_{a,\rm out}^{c,e}(t)\dot M_{\rm out}^{c,e}(t),\\
M_{Z,\rm esc}^{c,e}
&=
\sum_{a>{\rm He}}M_{a,\rm esc}^{c,e},
\qquad
M_{\rm ej,esc}^{c,e}
=
\sum_aM_{a,\rm esc}^{c,e}.
\end{aligned}
\tag{6.5b}
$$

式 (6.5) 的 $-X_{a,\rm out}\dot M_{\rm out}$ 已从源晕气体库扣除式 (6.5b)
中的逃逸核素；外部泡只使用同一通道、同一前身星的 $M_{a,\rm esc}^{c,e}$。
回落物质沿同一示踪标签进入 $\dot M_{a,\rm reaccr}$。轨道总返回先进入注入项，
宿主保留、外逸和再吸积随后由同一质量流分配。爆炸窗口不得包含第二个 PISN；
式 (6.12) 的逐事件率因此只与单次爆炸响应相乘。前身星风保留真实注入时刻，
进入独立的连续风传播项。

Pop III 形成由成星气体金属丰度分布的低端决定。严格零金属成分与连续富集
成分可写成混合测度

$$
p_{\rm SF}(dZ\mid M_h,z,\delta_{\rm env})
=
P_0^{\rm SF}\,\delta_0(dZ)
+
p_+^{\rm SF}(Z)\,dZ,
\qquad
P_0^{\rm SF}
+\int_{0^+}^{\infty}p_+^{\rm SF}(Z)\,dZ
=1.
\tag{6.6}
$$

$P_0^{\rm SF}$ 是严格 pristine 的成星质量分数。允许
$0<Z<Z_{\rm crit}$ 形成 Pop III 时，操作性的 Pop-III-forming fraction 为

$$
\boxed{
P_{\rm III-form}^{\rm SF}
=
P_0^{\rm SF}
+
\int_{0^+}^{Z_{\rm crit}}p_+^{\rm SF}(Z)\,dZ
}.
\tag{6.7}
$$

若模拟给出空间分辨的气体金属丰度，成星权重通过

$$
P_{\rm III-form}^{\rm SF}
=
\frac{
\displaystyle\int dV\,
\dot\rho_\star(\mathbf x)
\mathbf 1[Z(\mathbf x)<Z_{\rm crit}]
}{
\displaystyle\int dV\,\dot\rho_\star(\mathbf x)
}
\tag{6.7a}
$$

映射到式 (6.7)，其中 $\dot\rho_\star$ 已包含密度和局部恒星形成律的权重。
两类恒星的形成率为

$$
\dot M_{\star,\rm III}
=
P_{\rm III-form}^{\rm SF}\dot M_\star,
\qquad
\dot M_{\star,\rm II}
=
\left(1-P_{\rm III-form}^{\rm SF}\right)\dot M_\star.
\tag{6.8}
$$

线性金属丰度密度与每 dex 星形成率之间的 Jacobian 为

$$
\begin{aligned}
p_{+,\log}^{\rm SF}(\log_{10}Z)
&=
(\ln10)Zp_+^{\rm SF}(Z),\\
\dot M_{\star,\rm III}^{(0)}
&=
P_0^{\rm SF}\dot M_\star,\\
\dot{\mathcal M}_{\star,\rm III}(\log_{10}Z)
&=
\dot M_\star p_{+,\log}^{\rm SF}(\log_{10}Z)
\mathbf 1[0<Z<Z_{\rm crit}],\\
\dot{\mathcal M}_{\star,\rm II}(\log_{10}Z)
&=
\dot M_\star p_{+,\log}^{\rm SF}(\log_{10}Z)
\mathbf 1[Z\ge Z_{\rm crit}].
\end{aligned}
\tag{6.8a}
$$

$p_{+,\log}^{\rm SF}$ 的单位为 dex$^{-1}$，并满足
$\int d\log_{10}Z\,p_{+,\log}^{\rm SF}=1-P_0^{\rm SF}$。式 (6.8a) 的积分
恢复式 (6.8)。严格 $Z=0$ Pop III 基准把最后一个指示区间扩展为全部 $Z>0$，
并令有限金属 Pop III 连续项为零。

$Z=0$ 的严格基准取
$P_{\rm III-form}^{\rm SF}=P_0^{\rm SF}$，并直接使用式 (5.8)、(5.13)、
(5.14) 与式 (6.4) 中的 $Z_{\rm II,min}=0^+$。允许亚临界有限金属 Pop III
时，所有这些卷积统一采用式 (5.0)、(5.14c') 和 (6.8a)，并令
$Z_{\rm II,min}=Z_{\rm crit}$。

$Z_{\rm crit}$ 由尘埃、细结构线冷却和碎裂模型共同给定。Schneider et al.
(2003) 给出超新星尘埃冷却触发低质量碎裂的早期模型，Bromm & Loeb (2003)
量化了 C II 和 O I 细结构线冷却的临界丰度。平均
$Z_{\rm g}$ 无法确定式 (6.7) 的低丰度尾部，非均匀混合需要单独演化。
Pan, Scannapieco & Scalo (2013) 的 pristine-fraction 卷积模型写为

$$
\left.\frac{dP}{dt}\right|_{\rm mix}
=
-\frac{n}{\tau_{\rm con}}
P\left(1-P^{1/n}\right).
\tag{6.9}
$$

$\tau_{\rm con}$ 和 $n$ 依赖湍流 Mach 数、污染注入尺度及
$Z_{\rm crit}/\langle Z\rangle$；该式描述已注入被动标量的湍流自卷积。
平流、持续注入、回落、重新吸积和气体质量到成星质量的映射通过独立项处理。
Sarmento et al. (2018) 将这一闭合用于宇宙学 Pop III 模型
([Schneider et al. 2003](https://doi.org/10.1038/nature01579);
[Bromm & Loeb 2003](https://doi.org/10.1038/nature02071);
[Pan, Scannapieco & Scalo 2013](https://doi.org/10.1088/0004-637X/775/2/111);
[Sarmento et al. 2018](https://doi.org/10.3847/1538-4357/aa989a))。

PISN 还可通过外流对邻近晕产生外部富集。单次爆炸越过源晕边界并驱动外部
介质的能量为
$E_{\rm esc}^{\rm expl}=f_{{\rm esc},E}^{\rm expl}
E_{\rm kin,\infty}$。风的外逸机械能通过式 (6.2e) 的连续项预处理周围介质；
下式描述爆炸后的冲击阶段。均匀、静态介质中的
绝热 Sedov--Taylor 阶段给出激波半径和扫掠质量

$$
R_{\rm sh}(\Delta t)
=
\beta_{\rm ST}
\left(
\frac{E_{\rm esc}^{\rm expl}}{\rho_{\rm amb}}
\right)^{1/5}
\Delta t^{2/5},
\qquad
M_{\rm sw}
=
\frac{4\pi}{3}\rho_{\rm amb}R_{\rm sh}^3,
\tag{6.10}
$$

其中 $R_{\rm sh}$ 是物理长度，$\rho_{\rm amb}$ 是物理密度；对绝热指数
$\gamma=5/3$，$\beta_{\rm ST}\simeq1.15$。式 (6.10) 适用于
$\Delta t\ll H^{-1}(z)$、扫掠质量占主导且辐射损失、宇宙膨胀和引力尚未显著
改变激波的阶段。进入辐射冷却和 snowplow 阶段后，半径演化采用相应动量方程
或辐射流体轨道。Scannapieco, Ferrara & Madau (2002) 将这一类外流解用于
高红移金属富集模型
([Scannapieco, Ferrara & Madau 2002](https://doi.org/10.1086/341114))。
采用 cgs 单位时，$E_{\rm esc}^{\rm expl}$、$\rho_{\rm amb}$、$\Delta t$、
$R_{\rm sh}$ 和 $M_{\rm sw}$ 的单位依次为 erg、
$\mathrm{g\,cm^{-3}}$、s、cm 和 g。

激波体积、含金属示踪物的体积和环境气体自身达到 $Z_{\rm crit}$ 的体积具有
不同定义。下文的 PISN 爆炸泡仅使用式 (6.5b) 的单事件爆炸响应
$M_{Z,\rm esc}^{\rm expl,e}$ 与 $M_{\rm ej,esc}^{\rm expl,e}$。
$f_{\rm mix,Z}$ 和 $f_{\rm mix,g}$ 分别描述逃逸金属与逃逸抛射气体进入扫掠介质
的混合比例。若扫掠介质已有无量纲金属丰度 $Z_{\rm amb}$，均匀混合后的总丰度为

$$
\begin{aligned}
Z_{\rm bub}
&\simeq
\frac{
Z_{\rm amb}Z_{\odot,\rm ref}M_{\rm sw}
+f_{\rm mix,Z}M_{Z,\rm esc}^{\rm expl,e}
}{
Z_{\odot,\rm ref}
\left(M_{\rm sw}+f_{\rm mix,g}M_{\rm ej,esc}^{\rm expl,e}\right)
},\\
\Delta Z_{\rm PISN}
&\equiv
\frac{f_{\rm mix,Z}M_{Z,\rm esc}^{\rm expl,e}}
{Z_{\odot,\rm ref}
\left(M_{\rm sw}+f_{\rm mix,g}M_{\rm ej,esc}^{\rm expl,e}\right)}.
\end{aligned}
\tag{6.11a}
$$

$Z_{\rm amb}=0$ 给出 pristine ambient 基准。在
$M_{\rm sw}\gg M_{\rm ej,esc}^{\rm expl,e}$ 的扫掠质量主导极限，增量分母化为
$Z_{\odot,\rm ref}M_{\rm sw}$。质量守恒要求

$$
0\le f_{\rm mix,Z},f_{\rm mix,g}\le1,
\qquad
M_{Z,\rm esc}^{\rm expl,e}\le M_{\rm ej,esc}^{\rm expl,e},
\qquad
f_{\rm mix,Z}M_{Z,\rm esc}^{\rm expl,e}
\le f_{\rm mix,g}M_{\rm ej,esc}^{\rm expl,e}.
\tag{6.11b}
$$

令 $V_{\rm sh,p}=4\pi R_{\rm sh}^3/3$，并令
$V_{\rm metal,p}\subseteq V_{\rm sh,p}$ 为含有可追踪 PISN 金属柱密度的物理
体积。非均匀情形中的 $Z(\mathbf r_{\rm p})$、
$\Sigma_Z^{\rm expl,e}(\mathbf r_{\rm p})$ 和 $V_{\rm metal,p}$ 由金属示踪粒子、
被动标量输运或解析混合核给出。环境气体自身已经超过阈值的充分条件体积为

$$
\begin{aligned}
V_{\rm suff,p}
&=
\int_{V_{\rm metal,p}}d^3r_{\rm p}\,
\mathbf 1[Z(\mathbf r_{\rm p})>Z_{\rm crit}],\\
V_{\rm suff,c}^{\rm uniform}
&=
(1+z)^3V_{\rm sh,p}
\mathbf 1[Z_{\rm bub}>Z_{\rm crit}].
\end{aligned}
\tag{6.11c}
$$

$V_{\rm suff}$ 是保守诊断。目标晕可通过选择性俘获和压缩使局部低于阈值的
外流在成星气体中达到阈值，因此一般目标富集概率在整个
$V_{\rm metal,p}$ 上计算。条件概率定义为

$$
C_{\rm cap}
\equiv
P\!\left[
Z_{\rm SF,target}>Z_{\rm crit}
\,\middle|\,
\substack{
M_{\rm source},E_{\rm esc}^{\rm expl},
M_{Z,\rm esc}^{\rm expl,e},
\Sigma_Z^{\rm expl,e}(\mathbf r_{\rm p}),z',\\
r_{\rm p},M_{\rm target},z,\delta_{\rm env}
}
\right].
\tag{6.11d}
$$

该量同时包含金属俘获、中心输运、气体保留和后续坍缩。Smith et al. (2015)
给出外部富集 minihalo 形成低质量第二代恒星的数值实例，Hicks et al. (2021)
进一步量化首批超新星外流进入邻近 minihalo 中心的条件依赖
([Smith et al. 2015](https://doi.org/10.1093/mnras/stv1509);
[Hicks et al. 2021](https://doi.org/10.3847/1538-4357/abda3a))。

令 $r_{\rm p}$ 为目标红移 $z$ 时的物理源--目标间距。对采用同一物理距离约定的
源--目标两点相关函数 $\xi_{\rm st}$，体积平均聚集与俘获因子和有效共动体积为

$$
\begin{aligned}
\mathcal B
&=
\frac{1}{V_{\rm metal,p}}
\int_{V_{\rm metal,p}}d^3r_{\rm p}\,
[1+\xi_{\rm st}(r_{\rm p},\ldots)]
C_{\rm cap}(r_{\rm p},\ldots),\\
V_{\rm cap,c}
&=
(1+z)^3V_{\rm metal,p}\mathcal B\\
&=
(1+z)^3
\int_{V_{\rm metal,p}}d^3r_{\rm p}\,
[1+\xi_{\rm st}(r_{\rm p},\ldots)]
C_{\rm cap}(r_{\rm p},\ldots).
\end{aligned}
\tag{6.11e}
$$

若相关函数以共动距离给出，其自变量换为
$r_{\rm c}=(1+z)r_{\rm p}$。当 $V_{\rm metal,p}=0$ 时直接取
$V_{\rm cap,c}=0$。式 (6.12) 使用共动 PISN 率，因而与
$V_{\rm cap,c}$ 配对。

环境 $\delta_{\rm env}$ 中，由逐次 PISN 爆炸抛射物单独贡献的平均富集泡覆盖
次数为

$$
\begin{aligned}
Q_{Z,\rm PISN}(M_{\rm target},z\mid\delta_{\rm env})
={}&
\int_z^\infty dz'\,
\left|\frac{dt}{dz'}\right|
\sum_k\int d\log_{10}M_{\rm source}\,
\frac{d\dot n_{{\rm PISN},k}}
{d\log_{10}M_{\rm source}}(M_{\rm source},z')\\
&\times
\int d\vartheta_{\rm PISN}\,
p_k(\vartheta_{\rm PISN}\mid M_{\rm source},z')\\
&\times
V_{\rm cap,c}
(\vartheta_{\rm PISN},M_{\rm source},M_{\rm target},
z',z\mid\delta_{\rm env}),
\end{aligned}
\tag{6.12}
$$

$\vartheta_{\rm PISN}$ 至少包含相关的
$\{E_{\rm esc}^{\rm expl},M_{Z,\rm esc}^{\rm expl,e},
M_{\rm ej,esc}^{\rm expl,e},{\rm geometry}\}$，其条件分布由同一单星轨道和
单事件爆炸外流模型产生，并满足 $\int d\vartheta_{\rm PISN}p_k=1$。
$M_{\rm source}$ 是 $z'$ 时的爆炸宿主质量，采用第 5.5 节的同一 $M_\Delta$
约定。式 (6.12) 因而保留高能量、高产额事件与较大富集泡之间的相关性。
$Q_{Z,\rm PISN}$ 无量纲：体积率、源参考系时间和共动俘获体积的单位依次为
$\mathrm{yr}^{-1}\,\mathrm{cMpc}^{-3}$、yr 和 $\mathrm{cMpc}^3$。

在固定 $\delta_{\rm env}$ 的条件 Poisson 闭合下，$\mathcal B$ 修正平均覆盖次数，
连通的高阶空间相关项暂不保留。此时

$$
P_{\rm no\mbox{-}ext}^{\rm PISN}
\simeq e^{-Q_{Z,\rm PISN}},
\qquad
P_{\rm enriched}^{\rm PISN}
\simeq1-e^{-Q_{Z,\rm PISN}}.
\tag{6.12a}
$$

严格独立的空间基准取 $\xi_{\rm st}=0$；$C_{\rm cap}$ 仍保留目标晕对外来金属
的选择性响应。

环境闭合的扰动参数集合可写为

$$
\theta_{\rm env}
=
\{f_{{\rm host},E}^{c},f_{{\rm esc},E}^{c},
X_{a,\rm out}^{c,e}\dot M_{\rm out}^{c,e},
f_{\rm mix,Z},f_{\rm mix,g},C_{\rm cap}\}.
\tag{6.12b}
$$

这些量目前没有跨质量、红移和环境通用的文献拟合。可复算的数值模型需从同一
组辐射流体或金属示踪模拟中制表，表轴至少包括
$(M_{\rm source},z,n_{\rm amb},r_{\rm SN},E_{\rm kin},M_{\rm ej},M_Z,
r_{\rm p},M_{\rm target})$，并在表格覆盖域内联合插值。超出覆盖域时保持“无数值
预测”状态。若采用固定参数灵敏度试验，需显式声明其先验和能量、质量守恒约束。
Smith et al. (2015) 与 Hicks et al. (2021) 支持俘获过程的存在及环境依赖，
它们没有给出式 (6.12b) 的通用现成拟合。因此，本笔记的式 (6.11)--(6.12)
目前构成物理闭合和模拟提取规范，尚不构成数值外部富集率预测。

式 (6.12) 的 $Q_{Z,\rm PISN}$ 专指最终爆炸抛射物。前身星风从
$\mathcal K^{\rm wind}$ 在每个真实注入时刻产生连续传播轨道，并单独形成
$Q_{Z,\rm PISN\mbox{-}wind}$；该项不得再次使用逐 PISN 爆炸率或爆炸时刻。
完整外部污染采用互斥通道求和，

$$
Q_Z^{\rm all}
=
Q_{Z,\rm PISN\mbox{-}expl}
+Q_{Z,\rm PISN\mbox{-}wind}
+Q_{Z,\rm CCSN\mbox{-}expl}
+Q_{Z,\rm other\mbox{-}wind}
+\cdots,
\qquad
Q_{Z,\rm PISN\mbox{-}expl}\equiv Q_{Z,\rm PISN}.
\tag{6.12c}
$$

解析外流模型可参见 Scannapieco, Ferrara & Madau (2002)；PISN 专属金属传播、
各向异性逃逸和目标晕外部富集由 Greif et al. (2010)、Wise et al. (2012)
和 Hicks et al. (2021) 的模拟支持。Ritter et al. (2015) 的
$10^{51}\,\mathrm{erg}$ 核心坍缩超新星计算说明了一般早期超新星中的差异输运、
回落和再坍缩
([Greif et al. 2010](https://doi.org/10.1088/0004-637X/716/1/510);
[Wise et al. 2012](https://doi.org/10.1088/0004-637X/745/1/50);
[Ritter et al. 2015](https://doi.org/10.1093/mnras/stv982))。

自洽 PISN 化学反馈所需的闭环为

$$
\boxed{
\begin{aligned}
\{\dot M_{\star,\rm III},
\dot{\mathcal M}_{\star,\rm II}\}
\;&\rightarrow
\{\mathcal K_{\rm PISN},
\mathcal K_a^{\rm ret},
\mathcal K_{\rm ret},
\mathcal K_E^{\rm mech},
\mathcal K_E^{\rm host},
\mathcal K_E^{\rm esc}\}\\
&\rightarrow
\{\theta_{\rm env},M_{\rm g},M_{a,\rm g},
{\rm transport},{\rm mixing}\}
\rightarrow
p_{\rm SF}(dZ)\\
&\rightarrow
P_{\rm III-form}^{\rm SF}
\rightarrow
\{\dot M_{\star,\rm III},
\dot M_{\star,\rm II}\}.
\end{aligned}
}
\tag{6.13}
$$

式 (6.13) 定义后续自洽计算需要闭合的物理系统；本笔记完成公式与文献基准，
尚未对该反馈环执行数值迭代。

## 7. 与 21 cm 的共享物理

忽略视线速度梯度时，21 cm 微分亮温近似为

$$
\delta T_b
\simeq
27\,x_{\rm HI}(1+\delta_b)
\left(\frac{\Omega_bh^2}{0.023}\right)
\left[
\frac{0.15}{\Omega_mh^2}
\frac{1+z}{10}
\right]^{1/2}
\left(1-\frac{T_\gamma}{T_s}\right)
{\rm mK}.
$$

Pop III 通过四条路径改变该式：

1. LW 光子提高 $M_{\rm mol}$，改变 minihalo 中的形成率；
2. Ly$\alpha$ 光子驱动 Wouthuysen--Field 耦合，使 $T_s$ 接近气体温度；
3. X-ray 加热提高气体温度；
4. H 电离光子改变 $x_{\rm HI}$。

这些光子产额与 He II 所需的 $Q_{\rm He^+}$ 来自同一个 $\theta_\star$。旋转、双星
和高质量 IMF 尾部会同时改变 He II、LW、电离历史和 PISN 命运。

联合似然写为

$$
\boxed{
\mathcal L_{\rm joint}(\theta_\star,\theta_{\rm form},\theta_Z)
=
\mathcal L_{\rm HeII}
\mathcal L_{\rm PISN}
\mathcal L_{21{\rm cm}}
\mathcal L_{\rm UV}.
}
$$

其中 $\theta_{\rm form}$ 描述晕质量依赖的星形成和 LW 反馈，
$\theta_Z$ 描述富集与混合。四个因子共享相同的恒星族和形成历史。

## 8. 当前可直接使用的物理基准

| 基准 | 数值或关系 | 适用条件 |
|---|---|---|
| 原子冷却 | $T_{\rm vir}\simeq10^4$ K，$M_{\rm atomic}\propto(1+z)^{-3/2}$ | primordial atomic gas |
| He II Case-B | $L_{1640}=5.67\times10^{-12}Q_{\rm He^+,abs}$ | $T_e\simeq3\times10^4$ K、ionization-bounded |
| Hebe C2 | He II/H$\gamma=0.224$ | 两条线统一到去透镜空间 |
| Hebe C1 | He II/H$\gamma>0.7$，Rusta 报告为 $3\sigma$ | 截尾分量似然 |
| 成对不稳定性 | $\langle\Gamma_1\rangle\rightarrow4/3$；爆炸后全星非束缚 | 含 $e^\pm$ 状态方程与水动力轨道 |
| 经典 PISN 命运窗 | $M_{\rm He,init}=64$--$133\,M_\odot$，约映射到 $M_{\rm ZAMS}=140$--$260\,M_\odot$ | 零金属、非旋转基准 |
| 经典 PISN 效率 | $\eta\simeq1.32\times10^{-3}M_\odot^{-1}$ | Salpeter 50--500、经典命运窗 |
| 观测者事件率 | $dN/dt_{\rm obs}\propto\dot n_{\rm src}/(1+z)$ | 源参考系体积率转换 |
| HSC 率尺度 | $\sim100\ {\rm Gpc^{-3}\,yr^{-1}}$ | 高光度、长时标、$z\lesssim3$ PISN-like |

这些基准适合验证代数、单位和方向性。联合参数推断还需随机 IMF、光致电离
网格、质量依赖命运概率、富集概率和选择函数。

## 9. 物理推断顺序

1. 由冷却阈值、LW 背景和
   $P_{\rm III-form}^{\rm SF}(M_h,z,\delta_{\rm env})$ 计算
   $\dot{\mathcal M}_{\star,\rm II}(M_h,t,\log_{10}Z)$ 与
   $\dot M_{\star,\rm III}(M_h,t)$，并输出 $z>10$ 的
   $F_{\rm III}^{\rm SFR}(M_1,M_2,z)$。
2. 对每一类恒星指定 IMF、恒星轨道和命运概率
   $p_{{\rm PISN},k}(M,Z)$，计算
   $\eta_{{\rm PISN},k}$、$\mathcal K_{{\rm PISN},k}$、
   $y_{a,k}^{\rm ret}$、$R_k^{\rm PISN}$ 与 $\varepsilon_{E,k}^{\rm mech}$。
3. 将寿命核与单晕星形成史卷积，通过晕质量函数得到
   $d\dot n_{{\rm PISN},k}/d\log_{10}M_{\rm expl}$、总率
   $\dot n_{{\rm PISN},k}$ 和质量箱占比
   $F_{{\rm PISN},k}(M_1,M_2,z)$。
4. 由 $L_\nu^{\rm PISN}(t)$、宇宙学谱通量变换和巡天选择函数计算
   $T_{\rm ctrl,obs}$ 与 $\mu_{\rm det}$，再构造 Poisson 计数似然。
5. 将 PISN 核素与耦合能量传播到气体质量守恒、外流、回落和湍流混合，再更新
   $p_{\rm SF}(dZ)$ 与 $P_{\rm III-form}^{\rm SF}$。执行式 (6.13) 的数值迭代后，
   Pop III 形成史与化学反馈达到自洽。
6. 用同一 $\theta_\star$ 产生 $L_{1500}$、$Q_{\rm H}$、$Q_{\rm He^+}$ 和
   $N_{\rm LW}$，构造 Hebe C1 的截尾似然，并将 C2 作为共同
   运动学分量检验。
7. 在共享参数空间中联合 He II、PISN、21 cm 和 UV 似然。

## 10. 文献图谱

### 10.1 晕形成、冷却与富集

| 文献 | 物理作用 |
|---|---|
| [Asplund et al. 2009](https://doi.org/10.1146/annurev.astro.46.060407.145222) | 本文采用的 $Z_{\odot,\rm ref}=0.0134$ 太阳金属质量分数标尺 |
| [Bryan & Norman 1998](https://doi.org/10.1086/305262) | 本文采用的红移依赖 virial overdensity 与晕质量定义 |
| [Tegmark et al. 1997](https://doi.org/10.1086/303434) | H$_2$ 冷却与首批成星晕的质量尺度 |
| [Barkana & Loeb 2001](https://doi.org/10.1016/S0370-1573(01)00019-9) | $T_{\rm vir}$、晕质量与红移的标准关系 |
| [Machacek, Bryan & Abel 2001](https://doi.org/10.1086/321717) | LW 背景提高 minihalo 冷却阈值 |
| [Schneider et al. 2003](https://doi.org/10.1038/nature01579) | 超新星尘埃冷却与低质量碎裂的临界条件 |
| [Bromm & Loeb 2003](https://doi.org/10.1038/nature02071) | C II/O I 细结构线冷却的临界丰度 |
| [Scannapieco, Ferrara & Madau 2002](https://doi.org/10.1086/341114) | 高红移超新星外流与外部金属富集的解析模型 |
| [Greif et al. 2010](https://doi.org/10.1088/0004-637X/716/1/510) | PISN 金属泡和 first-galaxy enrichment |
| [Wise et al. 2012](https://doi.org/10.1088/0004-637X/745/1/50) | 辐射反馈、PISN 富集和 Pop III/II 转换 |
| [Ritter et al. 2015](https://doi.org/10.1093/mnras/stv982) | 非均匀金属输运与再坍缩 |
| [Pan, Scannapieco & Scalo 2013](https://doi.org/10.1088/0004-637X/775/2/111) | pristine fraction 的湍流自卷积闭合 |
| [Sarmento et al. 2018](https://doi.org/10.3847/1538-4357/aa989a) | 显式 pristine fraction 与 Pop III-bright galaxies |
| [Smith et al. 2015](https://doi.org/10.1093/mnras/stv1509) | 外部富集 minihalo 中第二代低质量恒星的形成 |
| [Hicks et al. 2021](https://doi.org/10.3847/1538-4357/abda3a) | 首批超新星对 minihalo 的外部富集与中心俘获 |
| [Venditti et al. 2023](https://doi.org/10.1093/mnras/stad1201) | 大宿主外围的 Pop III pockets 与空间条件化 |

### 10.2 He II、恒星族与星云

| 文献 | 物理作用 |
|---|---|
| [Tumlinson & Shull 2000](https://doi.org/10.1086/312432) | 零金属恒星的硬电离谱 |
| [Bromm, Kudritzki & Loeb 2001](https://doi.org/10.1086/320549) | top-heavy IMF 对 H I、He I、He II 光子产额的影响 |
| [Schaerer 2002](https://doi.org/10.1051/0004-6361:20011619) | 年龄、IMF、He II visibility 与星云连续谱 |
| [Schaerer 2003](https://doi.org/10.1051/0004-6361:20021525) | $Q_{\rm He^+}/Q_{\rm H}$ 的金属丰度和 IMF 依赖 |
| [Raiter, Schaerer & Fosbury 2010](https://doi.org/10.1051/0004-6361/201015236) | Case-B、$U/n_H$ 和 two-photon continuum |
| [Zackrisson et al. 2011](https://doi.org/10.1088/0004-637X/740/1/13) | 恒星与星云 SED 的统一几何 |
| [Yoon, Dierks & Langer 2012](https://doi.org/10.1051/0004-6361/201117769) | 旋转和化学均匀演化对硬光子与 PISN 的共同影响 |
| [Mas-Ribas et al. 2016](https://doi.org/10.3847/1538-4357/833/1/65) | 随机 IMF 和低 $U$ 下的 He II 离散 |
| [Stanway, Eldridge & Becker 2016](https://doi.org/10.1093/mnras/stv2661) | 低金属双星对电离光子的影响 |
| [Götberg et al. 2019](https://doi.org/10.1051/0004-6361/201834525) | stripped stars 延长硬电离阶段 |
| [Murphy et al. 2021](https://doi.org/10.1093/mnras/stab2073) | 旋转、对流与 Pop III 电离光子 |
| [Nakajima & Maiolino 2022](https://doi.org/10.1093/mnras/stac1242) | Pop III、极贫金属星族、AGN 和 DCBH 的联合线诊断 |
| [Oskinova & Schaerer 2022](https://doi.org/10.1051/0004-6361/202142520) | 星团风和 superbubble 的 He II 污染 |
| [Katz et al. 2023](https://doi.org/10.1093/mnras/stad1903) | He II 可见期、金属线灵敏度和污染族 |
| [Trussler et al. 2023](https://doi.org/10.1093/mnras/stad2553) | JWST 曝光与 He II 选择函数 |
| [Venditti et al. 2024](https://doi.org/10.3847/2041-8213/ad7387) | He II 光度的 IMF/SFE 不确定度与孔径损失 |
| [Lecroq et al. 2025](https://doi.org/10.1051/0004-6361/202452463) | 双星 Pop III 光谱 |
| [Wasserman et al. 2026](https://doi.org/10.1093/mnras/stag386) | 旋转 Pop III 对 UV、He II 与 21 cm 的共同作用 |

### 10.3 Hebe 与高红移 He II 候选

| 文献 | 物理作用 |
|---|---|
| [Wang et al. 2024](https://doi.org/10.3847/2041-8213/ad4ced) | $z=8.16$ 的 He II 发射体和混合星族检验 |
| [Maiolino et al. 2024](https://doi.org/10.1051/0004-6361/202347087) | GN-z11 周围的初始候选 |
| [Maiolino et al. 2026](https://arxiv.org/abs/2603.20362) | Hebe 两个 He II 分量、光度、EW 和金属线上限 |
| [Übler et al. 2026](https://arxiv.org/abs/2603.20360) | C2 H$\gamma$、H$\delta$、[Ne III] 和气体金属丰度 |
| [Rusta et al. 2026](https://doi.org/10.3847/2041-8213/ae64e1) | Larson IMF、photoionization grid 和 C1 censored likelihood |
| [Jeon et al. 2026](https://doi.org/10.3847/1538-4357/ae7bea) | Hebe 团块质量的模拟可行性和 BH alternative |

### 10.4 PISN 命运、产额与光变

| 文献 | 物理作用 |
|---|---|
| [Barkat, Rakavy & Sack 1967](https://doi.org/10.1103/PhysRevLett.18.379) | 电子--正电子对产生触发动力学不稳定性的经典推导 |
| [Rakavy & Shaviv 1967](https://doi.org/10.1086/149204) | 超大质量恒星的成对不稳定性与爆炸演化 |
| [Heger & Woosley 2002](https://doi.org/10.1086/338487) | 零金属非旋转 PISN 命运与产额 |
| [Umeda & Nomoto 2002](https://doi.org/10.1086/323946) | 130--300 $M_\odot$ 产额网格 |
| [Kasen, Woosley & Heger 2011](https://doi.org/10.1088/0004-637X/734/2/102) | 质量和包层依赖的光变与光谱 |
| [Kozyreva et al. 2014](https://doi.org/10.1051/0004-6361/201423447) | 低质量与高质量 PISN 的亮度和时标 |
| [Kozyreva et al. 2017](https://doi.org/10.1093/mnras/stw2562) | stripping、mixing 与快速 Type-I PISN |
| [Smidt et al. 2015](https://doi.org/10.1088/0004-637X/805/1/44) | 旋转降低 PISN ZAMS 质量边界的示例 |
| [Takahashi, Yoshida & Umeda 2018](https://doi.org/10.3847/1538-4357/aab95f) | 旋转、core range 与 abundance signature |
| [Takahashi 2018](https://doi.org/10.3847/1538-4357/aad2d2) | core carbon fraction 和反应率 |
| [Farmer et al. 2019](https://doi.org/10.3847/1538-4357/ab518b) | $^{12}$C$(\alpha,\gamma)^{16}$O 对边界的系统误差 |
| [Marchant et al. 2019](https://doi.org/10.3847/1538-4357/ab3426) | 成对不稳定区的恒星演化与水动力终态 |
| [Renzo et al. 2020](https://doi.org/10.1051/0004-6361/202037710) | PPISN 脉冲、水动力响应与残骸质量 |
| [Farmer et al. 2020](https://doi.org/10.3847/2041-8213/abbadd) | 核反应率依赖的成对不稳定命运边界 |
| [Woosley & Heger 2021](https://doi.org/10.3847/2041-8213/abf2c4) | 核反应率、旋转、双星与黑洞质量缺口 |
| [Umeda & Nagele 2024](https://doi.org/10.3847/1538-4357/ad140a) | 非零金属旋转 PISN 与风模型 |
| [Gabrielli et al. 2024](https://doi.org/10.1093/mnras/stae2048) | 命运边界、IMF 和金属散布导致的宇宙率跨度 |
| [Wiggins et al. 2024](https://arxiv.org/abs/2402.17076) | Pop III 随机 IMF 填充与同红移相对 PISN 率 |

### 10.5 PISN 巡天与候选分类

| 文献 | 物理作用 |
|---|---|
| [Hogg 1999](https://arxiv.org/abs/astro-ph/9905116) | 谱通量、光度距离与红移变换 |
| [Hogg et al. 2002](https://arxiv.org/abs/astro-ph/0210394) | photon-counting 带通与 K correction 约定 |
| [Weinmann & Lilly 2005](https://doi.org/10.1086/428106) | 源参考系率到观测者率的时间膨胀 |
| [Pan, Kasen & Loeb 2012](https://doi.org/10.1111/j.1365-2966.2012.20837.x) | EoR PISN 率、光变和 JWST 可见性 |
| [Whalen et al. 2013](https://doi.org/10.1088/0004-637X/777/2/110) | CSM、IGM 和高红移可见性 |
| [Inoue et al. 2014](https://doi.org/10.1093/mnras/stu936) | IGM 平均吸收与统计视线处方 |
| [de Souza et al. 2013](https://doi.org/10.1093/mnras/stt1680) | 宇宙学、辐射输运与合成巡天 |
| [Hartwig, Bromm & Loeb 2018](https://doi.org/10.1093/mnras/sty1576) | JWST cadence、滤波器与视场策略 |
| [Lazar & Bromm 2022](https://doi.org/10.1093/mnras/stac176) | 高红移 PISN 宇宙率与 Pop III IMF |
| [Moriya et al. 2019](https://doi.org/10.1093/pasj/psz035) | deep-wide Roman/Subaru mock survey |
| [Wong et al. 2019](https://doi.org/10.1093/pasj/psz037) | cluster lensing 和视线选择 |
| [Liu & Bromm 2020](https://doi.org/10.1093/mnras/staa2143) | Pop III 终止与 LSST/JWST rate scale |
| [Regős, Vinkó & Ziegler 2020](https://doi.org/10.3847/1538-4357/ab8636) | 质量损失、旋转和 JWST horizon |
| [Moriya et al. 2021](https://doi.org/10.3847/1538-4357/abcfc0) | HSC 长时标瞬变搜索和率上限 |
| [Venditti et al. 2024](https://doi.org/10.1093/mnras/stad3513) | 高红移宿主与 JWST/Roman 探测数 |
| [Schulze et al. 2024](https://doi.org/10.1051/0004-6361/202346855) | SN 2018ibb 的 PISN 候选证据与 CSM |
| [Nagele, Umeda & Maeda 2024](https://doi.org/10.3847/1538-4357/ad656c) | SN 2018ibb 光变和温度张力 |
| [Cruz et al. 2025](https://doi.org/10.1103/PhysRevD.111.083503) | 晕质量函数与单晕 Pop II/Pop III SFR 的积分 |
| [Ferrara et al. 2026](https://doi.org/10.33232/001c.162107) | 极高红移 dropout 的瞬变分类退化 |

\newpage

### 10.6 21 cm 宇宙黎明物理

| 文献 | 物理作用 |
|---|---|
| [Furlanetto, Oh & Briggs 2006](https://doi.org/10.1016/j.physrep.2006.08.002) | 21 cm 亮温、spin temperature 与再电离综述 |
| [Pritchard & Loeb 2012](https://doi.org/10.1088/0034-4885/75/8/086901) | Ly$\alpha$ 耦合、X-ray 加热和 global signal |
| [Visbal et al. 2012](https://doi.org/10.1038/nature11129) | streaming velocity 的大尺度 21 cm 印记 |
| [Fialkov et al. 2013](https://doi.org/10.1093/mnras/stt650) | LW 反馈与首星 21 cm 信号 |
| [Muñoz 2019](https://doi.org/10.1103/PhysRevD.100.063538) | velocity-induced acoustic oscillations |

## 11. 最终物理图像

1. 原子冷却阈值控制冷却通道；$p_{\rm pristine}$ 控制 Pop III 化学身份。
   $F_{\rm III}^{\rm SFR}(M_1,M_2,z)$ 才能回答不同晕质量中的 Pop III/Pop II 占比。
2. PISN 的单星判据来自含 $e^\pm$ 状态方程的动力学轨道：
   局部 $\Gamma_1$ 下降削弱恢复力，全局稳定性指标触发水动力收缩，爆炸性氧
   燃烧释放的能量使全星非束缚。经典零金属基准对应
   $M_{\rm He,init}=64$--$133\,M_\odot$。
3. Salpeter 50--500 $M_\odot$ 加经典 140--260 $M_\odot$ 命运窗给出
   $\eta_{\rm PISN}\simeq1.32\times10^{-3}M_\odot^{-1}$。旋转、双星、风和
   核反应率通过 $p_{\rm PISN}(M,Z)$ 传播到率的不确定度。
4. 百太阳质量级团块处于随机 IMF 区域。连续 IMF 对
   $\Delta M_\star=116\,M_\odot$ 给出 $\lambda_{\rm PISN}=0.15337$，
   严格质量预算抽样在经典命运窗下给出零事件；每个预测都需声明 IMF 填充规则。
5. $F_{{\rm PISN},k}(M_1,M_2,z)$ 由
   $dn/d\log M_h$、$\dot M_{\star,k}$、寿命核和命运概率共同决定，可直接给出
   $z>10$ 不同晕质量中 Pop II/Pop III 对 PISN 的相对贡献。经典
   Pop-III-only 模型在全部非零事件质量箱中给出 100% Pop III；两类恒星具有
   相同 PISN 效率时恢复第 2.4 节的 SFR 占比。
6. 可探测 PISN 数量由源参考系体积率、$(1+z)^{-1}$ 时间膨胀、质量依赖光谱时序
   和巡天控制时间共同决定。
7. $M_{a,\rm ret}$、$E_{\rm kin,\infty}$ 与环境参数 $\theta_{\rm env}$ 控制
   金属逃逸、有效俘获体积、非均匀混合和 $P_{\rm III-form}^{\rm SF}$。
   式 (6.13) 给出待数值迭代的化学反馈闭环。
8. He II 对小于数 Myr 的高质量恒星最敏感。Hebe C1 的截尾线比、
   C1 光度、EW 和金属线上限应共同进入似然。
9. He II、PISN、21 cm 和 UV 的联合限制需要共享
   $(\theta_\star,\theta_{\rm form},\theta_Z)$，并为每条观测链保留各自的扰动参数。
