
````markdown
# SFR 计算流程

本文档定义基于 halo growth history 的恒星形成率（SFR）计算流程，供 `realUVLF` 项目中直接实现与调用。

---

## 1. 基本公式

采用带 source-time 延迟的写法：

\[
{\rm SFR}(t)=f_b\,f_*\!\bigl(M_h(t-\tau_{\rm del})\bigr)\,\dot M_h(t-\tau_{\rm del})
\]

其中重子比例取为

\[
f_b=\frac{\Omega_b}{\Omega_m}
\]

恒星形成效率采用

\[
f_*(M_h)=
2\epsilon_0
\left[
\left(\frac{M_h}{M_c}\right)^{-\beta}
+
\left(\frac{M_h}{M_c}\right)^{\gamma}
\right]^{-1}
\]

固定参数为

\[
\epsilon_0=0.12,\qquad
M_c=10^{11.7}M_\odot,\qquad
\beta=0.66,\qquad
\gamma=0.65
\]

---

## 2. 输入来源

SFR 计算函数不直接输入单点形式的 `(t, z, Mh, cosmology)`，而是直接读取
`generate_halo_histories()` 返回的 `HaloHistoryResult.tracks`。

每条轨道至少使用以下字段：

- `halo_id`
- `step`
- `z`
- `t_gyr`
- `Mh`
- `dMh_dt_raw`
- `dMh_dt_sfr`
- `dMh_dt_clipped`

其中：

- `t_gyr` 为宇宙时间，单位 `Gyr`
- `Mh` 为 halo mass
- `dMh_dt_raw` 为原始 halo mass 导数，单位 `Msun/Gyr`，允许为负并用于 MAH 诊断
- `dMh_dt_sfr = maximum(dMh_dt_raw, 0)`，单位 `Msun/Gyr`，只用于 SFR/IMF 路径
- `dMh_dt_clipped = (dMh_dt_raw < 0)` 记录发生裁剪的位置

实际计算时应当：

- 按 `halo_id` 分组
- 在每条轨道内部按 `t_gyr` 升序计算
- 每条轨道独立得到自己的 `tau_del`、`Mh_src`、`fstar_src` 和 `SFR`

---

## 3. 宇宙学参数来源

宇宙学对象必须由调用方通过 keyword-only 参数显式传入。MAH、SFR、Pop III 与
chemistry 路径应复用同一个 `Cosmology` 对象，避免各阶段静默回到不同默认值。

推荐在 pipeline 入口解析一次：

```python
from auroralf.mah import Cosmology

cosmology = Cosmology()
```

并使用其中的：

* `omega_m`
* `omega_b`
* `omega_lambda`
* `h0_km_s_mpc`
* `rhocrit`

重子比例定义为

[
f_b=\frac{\texttt{cosmology.omega_b}}{\texttt{cosmology.omega_m}}
]

---

## 4. halo 物理量计算

对每个时刻 (t_i)（对应红移 (z_i)、质量 (M_{h,i})）计算以下量。

### 4.1 哈勃参数

在平直 (\Lambda)CDM 下：

[
H(z)=H_0\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}
]

其中 (H_0) 由 `cosmo.H0u` 提供。

### 4.2 临界密度

[
\rho_c(z)=\rho_{c,0}\left(\frac{H(z)}{H_0}\right)^2
]

其中 (\rho_{c,0}) 可由 `cosmo.rhocrit` 给出。

### 4.3 物质密度参数

[
\Omega_m(z)=\frac{\Omega_m(1+z)^3}{\Omega_m(1+z)^3+\Omega_\Lambda}
]

### 4.4 virial overdensity

[
\Delta_{\rm vir}(z)=18\pi^2+82d-39d^2,\qquad d=\Omega_m(z)-1
]

### 4.5 virial radius

[
r_{\rm vir}(t)=
\left[
\frac{3M_h(t)}{4\pi \Delta_{\rm vir}(z)\rho_c(z)}
\right]^{1/3}
]

### 4.6 circular velocity

[
V_c(t)=\sqrt{\frac{G M_h(t)}{r_{\rm vir}(t)}}
]

### 4.7 virial temperature

[
T_{\rm vir}(t)=\frac{\mu m_p V_c^2(t)}{2k_B}
]

若无特别说明，可取

[
\mu=0.61
]

---

## 5. 延迟时间定义

延迟时间定义为 halo dynamical / free-fall 时间：

[
\tau_{\rm del}(t)=\frac{r_{\rm vir}(t)}{V_c(t)}
]

这一定义表示：重子被吸积进入 halo 后，需要经过一个动力学时间尺度后才转化为恒星形成活动。

---

## 6. 原子冷却阈值

若不考虑 H(_2) 冷却，则要求 halo 满足原子冷却条件：

[
T_{\rm vir}(t)\ge 10^4,{\rm K}
]

若不满足，则直接设

[
{\rm SFR}(t)=0
]

---

## 7. source-time 取值

对每个时刻 (t_i)，定义源时间

[
t_{{\rm src},i}=t_i-\tau_{{\rm del},i}
]

然后在该 halo 的历史轨道上做线性插值，得到

[
M_{h,{\rm src}} = M_h(t_{{\rm src},i})
]

[
\dot M_{h,{\rm src}} = \dot M_h(t_{{\rm src},i})
]

如果

[
t_{{\rm src},i}<t_{\min}
]

即延迟后的源时间已经落到该轨道历史范围之外，则定义

[
{\rm SFR}(t_i)=0
]

---

## 8. source-time 的恒星形成效率

恒星形成效率在源时间计算，而不是在当前时刻计算：

[
f_{*,{\rm src}}=f_*(M_{h,{\rm src}})
]

即

[
f_{*,{\rm src}}=
2\epsilon_0
\left[
\left(\frac{M_{h,{\rm src}}}{M_c}\right)^{-\beta}
+
\left(\frac{M_{h,{\rm src}}}{M_c}\right)^{\gamma}
\right]^{-1}
]

---

## 9. 最终 SFR

若同时满足：

[
t_{{\rm src},i}\ge t_{\min}
]

且

[
T_{\rm vir}(t_i)\ge 10^4,{\rm K}
]

则

[
{\rm SFR}(t_i)=
f_b,
f_*!\bigl(M_h(t_i-\tau_{\rm del,i})\bigr),
\dot M_h(t_i-\tau_{\rm del,i})
]

否则定义

[
{\rm SFR}(t_i)=0
]

---

## 10. 建议返回字段

对每条轨道，建议至少返回以下字段：

* `halo_id`
* `step`
* `z`
* `t_gyr`
* `Mh`
* `dMh_dt_raw`
* `dMh_dt_sfr`
* `dMh_dt_clipped`
* `r_vir`
* `V_c`
* `T_vir`
* `tau_del`
* `t_src`
* `Mh_src`
* `dMh_dt_sfr_src`
* `fstar_src`
* `SFR`

返回格式建议保持与 `tracks` 一致的扁平表格风格，便于后续直接并入项目数据结构。

---

## 11. 推荐实现接口

建议实现形式为：

```python
compute_sfr_from_tracks(tracks, cosmology=cosmology)
```

如需保留 source-time 延迟，也可以扩展为：

```python
compute_sfr_from_tracks(
    tracks,
    cosmology=cosmology,
    enable_time_delay=True,
)
```

其中 `enable_time_delay=False` 可作为默认行为，表示直接用当前时刻的
`Mh` 和 `dMh_dt_sfr` 计算 SFR；只有在显式开启时才使用
基于 `g(t-t') \propto (t-t') \exp[-(t-t')/(\kappa t_d)]` 的
extended-burst 延迟核，对
\(f_\star[M_h(t')]\,\dot M_h(t')\)
做时间卷积。

原因如下：

1. 项目标准输入已经是 `HaloHistoryResult.tracks`
2. keyword-only 的必填宇宙学对象使科学上下文显式且不可被内部默认值覆盖
3. MAH 与 SFR 可通过对象 identity 复用同一组参数

---

## 12. 实现要求

请在 `realUVLF` 主目录或其子目录中实现 SFR 计算函数，要求如下：

* 输入为 `HaloHistoryResult.tracks` 风格的扁平数组字典
* `cosmology` 是 keyword-only 必填参数，且必须是 `Cosmology` 或其子类
* 禁止在 SFR calculator 内部构造默认宇宙学或对错误对象做 duck-typing fallback
* 按 `halo_id` 分组逐条轨道计算
* 使用 `tracks` 中已有的 `t_gyr, z, Mh, dMh_dt_raw, dMh_dt_sfr, dMh_dt_clipped`
* 计算 `r_vir, V_c, T_vir, tau_del`
* 用线性插值计算 `Mh(t-\tau_del)` 和 `dMh_dt_sfr(t-\tau_del)`
* 若 `T_vir < 1e4 K`，则 `SFR = 0`
* 若源时间超出历史范围，则 `SFR = 0`
* 返回扁平表格风格结果
* 代码使用 `numpy`
* 代码包含注释
* 若合适，可调用 `massfunc` 中已有宇宙学与阈值相关函数以减少重复实现

---

## 13. PopIII 独立 SFR 分支

PopIII 不复用 PopII calibrated `SFRModelParameters`。PopII 的
`epsilon_0, characteristic_mass, beta_star, gamma_star` 保持只描述 PopII。

PopIII 采用 Venditti/Cruz 风格的独立分支：

[
{\rm SFR}_{\rm III}(t)=
f_b\,f_{*,{\rm III}}(M_h)\,f_{{\rm duty},{\rm III}}(M_h,z,J_{\rm LW})\,
\dot M_h / 10^9
]

其中 `dMh_dt_sfr` 的单位是 `Msun/Gyr`，因此除以 `10^9` 后得到
`Msun/yr`。

默认 PopIII SFE 为 Cruz fiducial：

[
f_{*,{\rm III}}(M_h)=
\frac{2\epsilon_{*,{\rm III}}}
{(M_h/M_p)^{\alpha_*}+(M_h/M_p)^{\beta_*}},
\qquad
\epsilon_{*,{\rm III}}=10^{-3},\quad
M_p=10^7 M_\odot,\quad
\alpha_*=\beta_*=0
]

因此默认情况下 `f_star_III` 是常数 `1e-3`。

PopIII duty cycle 为：

[
f_{{\rm duty},{\rm III}}=
\exp(-M_{\rm mol}/M_h)\,\exp(-M_h/M_{\rm up})
]

其中 `M_mol` 使用 Cruz/Venditti/Zeus21 的分子冷却阈值：

[
M_{\rm mol}(z,J_{21}) =
3.3\times10^7(1+z)^{-3/2}
\left(1+2.0\,J_{21}^{0.6}\right)\,M_\odot
]

默认 `J21=0`，即不加入 LW 抑制；v1 也不加入 streaming-velocity
因子。`M_up` 默认使用 `Tvir=1e4 K` 的 atomic cooling mass；也可以通过
fixed upper mass 复现 Venditti 的 Bursty/Heavy 试验。

公开接口：

```python
PopIIISFRParameters(...)
compute_popiii_star_formation_efficiency(...)
compute_popiii_duty_cycle(..., cosmology=cosmology)
compute_popiii_sfr_from_grids(..., cosmology=cosmology)
```

`compute_popiii_sfr_from_grids` 返回 `SFR_popiii`、`fstar_popiii`、
`popiii_duty_cycle`、PopIII lower/upper mass grids 和 source mask。当前
pipeline 不让 PopIII SFR 进入 PopII metallicity regulator 或 PopII IMF gate。

### 13.1 Visbal+2015 PopIII SFH 诊断公式

Visbal, Haiman & Bryan (2015) Eq. 10 可作为单独的诊断型 PopIII SFH
口径使用：

[
{\rm SFR}_{III}^{\rm V15} =
f_{\star,III}\frac{\Omega_b}{\Omega_m}
\frac{M_h}{\eta_{\rm duty}t_H(z)}
]

其中 `t_H(z)=1/H(z)`，代码里 `H(z)` 的单位为 `Gyr^-1`，因此输出
`Msun/yr` 时需要除以 `1e9`。`eta_duty` 是 burst 持续时间占 Hubble
time 的比例，所以单条 halo history 上的瞬时 SFR 按
`eta_duty^-1` 缩放；它不是本节上面的
`exp(-M_mol/M_h) exp(-M_h/M_up)` PopIII occupation/duty weight。

该文的 atomic-cooling mass 使用原文近似：

[
M_{\rm cool}^{\rm V15} =
3\times10^7\left(\frac{1+z}{11}\right)^{1.5}M_\odot
]

并假定 PopIII 主要在 `M_h = (1-2) M_cool` 的 halo 中形成。公开诊断
接口为：

```python
compute_visbal2015_atomic_cooling_mass_msun(...)
compute_popiii_sfr_visbal2015_from_grids(..., cosmology=cosmology)
```

后者返回 `sfr_grid`、`mcool_msun_grid`、`mh_over_mcool_grid` 和
`atomic_window_grid`。该接口不替换默认 PopIII SFR prescription，也不参与
UV/HeII 卷积；需要 HMF 积分时应在调用端显式处理。

```
