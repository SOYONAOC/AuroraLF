# AuroraLF

## Project layout

Core code:

- `auroralf/uvlf/`: UV luminosity function pipeline, HMF weighting, dust mapping, and Pop II IMF gate logic
- `auroralf/mah/`: Monte Carlo halo assembly history generation
- `auroralf/sfr/`: star-formation model utilities
- `auroralf/chemistry/`: stochastic one-zone metal enrichment diagnostics
- `auroralf/ssp/`: SSP UV convolution utilities
- `tests/`: focused regression tests

Workflow code:

- `scripts/run/`: production or batch workflow entry points
- `scripts/submit/`: SLURM submission wrappers
- `scripts/plot/`: plotting and visual comparison scripts
- `scripts/analysis/`: post-processing and result comparison scripts
- `scripts/experiments/`: one-off experiment launchers

Data and generated files:

- `external_data/`: external source data, including observations, SSP spectra, empirical model releases, and literature source packages
- `data_save/`: reusable intermediate products and summary tables; ignored by git
- `outputs/`: logs, progress files, one-off plots, and diagnostics; ignored by git
- `temp_data/`: scratch caches and temporary `.npz` products; ignored by git
- `slides/`: Beamer sources, slide PDFs, and slide assets; ignored by git in this branch
- `archive/`: archived legacy code or notes kept for reference
- `nbody/`: N-body experiment notes and launch documentation

Keep external source data under `external_data/`, with large local libraries ignored by git. Use `data_save/` for reusable computed products and `outputs/` for diagnostics.

## `auroralf.mah.generate_halo_histories()`

导入：

```python
from auroralf.mah import generate_halo_histories
```

输入：

- `n_tracks`
  要生成的 Monte Carlo 轨道条数
- `z_final`
  终止红移
- `Mh_final`
  在 `z_final` 处的 halo mass
- `z_start_max`
  回溯的最高红移，默认 `50.0`
- `M_min`
  最低质量阈值；默认 `None` 时使用 `massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`，也可以传标量、与红移网格同长度的数组，或 `M_min(z)` 形式的可调用对象
- `cosmology`
  `auroralf.mah.Cosmology`；未提供时使用项目默认宇宙学
- `random_seed`
  随机种子
- `time_grid_mode`
  支持：
  - `"uniform_in_z"`
  - `"uniform_in_t"`
  - `"custom"`
- `dt`
  当 `time_grid_mode="uniform_in_t"` 时使用的时间步长，单位 `Gyr`
- `dz`
  当 `time_grid_mode="uniform_in_z"` 时使用的红移步长
- `custom_grid`
  当 `time_grid_mode="custom"` 时使用的自定义红移网格；本轮实现固定按 redshift grid 解释
- `store_inactive_history`
  是否保留低于 `M_min` 之后的历史点
- `sampler`
  `beta, gamma` 的抽样方式，支持 `"mcbride"` 和 `"gaussian"`
- `pilot_samples`
  当 `sampler="gaussian"` 时使用的 pilot sample 数目

输出：

- `HaloHistoryResult`

## `HaloHistoryResult`

字段：

- `tracks`
  扁平表格风格的 `dict[str, np.ndarray]`
- `metadata`
  输入参数回显、宇宙学、采样方式、采样摘要等信息

## `tracks` 字段

- `halo_id`
  轨道编号
- `step`
  该轨道内部的时间步编号
- `z`
  红移，按轨道内部单调下降
- `t_gyr`
  宇宙时间，单位 `Gyr`，按轨道内部单调升序
- `dt_gyr`
  相邻时间步间隔，单位 `Gyr`
- `Mh`
  halo mass
- `dMh_dt`
  halo mass accretion rate
- `active_flag`
  是否仍处于有效区间；当 `Mh < M_min` 后为 `False`
- `termination_flag`
  终止状态标记；当前实现使用：
  - `"active"`
  - `"below_M_min"`
  - `"completed"`

## `auroralf.sfr.compute_sfr_from_tracks()`

导入：

```python
from auroralf.sfr import compute_sfr_from_tracks
```

输入：

- `tracks`
  `HaloHistoryResult.tracks` 风格的扁平 `dict[str, np.ndarray]`；至少需要：
  - `halo_id`
  - `step`
  - `z`
  - `t_gyr`
  - `Mh`
  - `dMh_dt`
- `mu`
  平均分子量，默认 `0.61`
- `atomic_cooling_temperature`
  原子冷却阈值，默认 `1e4 K`
- `enable_time_delay`
  是否启用基于 dynamical time 的 extended-burst 延迟核；默认 `False`

输出：

- `dict[str, np.ndarray]`
  保留输入列，并新增：
  - `r_vir`
  - `V_c`
  - `T_vir`
  - `tau_del`
  - `td_burst`
  - `t_src`
  - `Mh_src`
  - `dMh_dt_src`
  - `fstar_src`
  - `fstar_now`
  - `mdot_burst`
  - `SFR`

说明：

- `SFR` 单位为 `Msun/yr`
- 当 `enable_time_delay=False` 时，直接用当前时刻的 `Mh` 和 `dMh_dt`
- 当 `enable_time_delay=True` 时，使用
  `g(t-t') \propto (t-t') \exp[-(t-t')/(\kappa t_d)]`
  的 extended-burst 核对
  `fstar(Mh(t')) * dMh_dt(t')`
  做时间卷积
- `tau_del/t_src/Mh_src/dMh_dt_src` 仍保留，作为与旧单一延迟时间口径可对照的诊断量
- `mdot_burst` 仍保留，表示只对 `dMh/dt` 做 kernel 卷积后的诊断量；真正进入 delay-SFR 的是
  `kernel * fstar(Mh) * dMh_dt` 的积分
- 若 `T_vir < 1e4 K`，则 `SFR = 0`

最小调用：

```python
from auroralf.mah import generate_halo_histories
from auroralf.sfr import compute_sfr_from_tracks

result = generate_halo_histories(n_tracks=100, z_final=6.0, Mh_final=1e11)
sfr_tracks = compute_sfr_from_tracks(result.tracks, enable_time_delay=True)
```

## `auroralf.chemistry.evolve_stochastic_metallicity()`

导入：

```python
from auroralf.chemistry import MetalEnrichmentParameters, evolve_stochastic_metallicity
```

输入：

- `t_grid_gyr`
  二维宇宙时间网格，单位 `Gyr`
- `z_grid`
  与时间网格同形状的红移网格
- `mh_grid`
  halo mass 网格，单位 `Msun`
- `dmhdt_grid`
  halo mass accretion rate 网格
- `sfr_grid`
  SFR 网格，单位 `Msun/yr`
- `active_grid`
  哪些 source-time 步参与成星/富集
- `baryon_fraction`
  宇宙 baryon fraction，必须在 `(0, 1]`
- `parameters`
  `MetalEnrichmentParameters`；未提供时使用默认参数
- `random_seed`
  stochastic yield、mass loading、birth metallicity scatter 的随机种子
- `topheavy_source_grid`
  可选布尔网格；为 `True` 的 source-time 候选点使用 `topheavy_yield_multiplier * metal_yield`
  作为产额，若同时设置 birth-metallicity gate，则该网格会先被金属丰度阈值筛选
- `topheavy_birth_metallicity_max_zsun`
  可选 `Z/Zsun` 阈值；提供时只允许 `birth_metallicity_zsun_grid <= topheavy_birth_metallicity_max_zsun`
  的候选 source-time 使用 top-heavy 产额

`MetalEnrichmentParameters` 常用字段：

- `gas_fraction_of_baryons`
  诊断 gas reservoir 占 halo baryon mass 的比例
- `metal_yield`
  每形成单位 stellar mass 产生的 metal mass
- `topheavy_yield_multiplier`
  top-heavy source-time 的产额倍增因子；低层 `MetalEnrichmentParameters` 默认 `1.0`
  表示不增强，生产/分析脚本采用 MZR 标定值 `1.28`
- `returned_fraction`
  即时返回比例，必须在 `[0, 1)`
- `mass_loading_norm`
  outflow mass loading 归一化
- `yield_scatter_dex`、`mass_loading_scatter_dex`、`birth_metallicity_scatter_dex`
  对数正态 scatter，单位 dex

输出：

- `MetallicityEvolutionResult`
  包含 `gas_mass_grid`、`metal_mass_grid`、`gas_metallicity_zsun_grid`、`birth_metallicity_zsun_grid`、`mass_loading_grid`、`effective_yield_grid`、`topheavy_source_grid`

说明：

- 该模块是沿固定 MAH/SFR 历史的 stochastic one-zone enrichment 诊断，不改变 SFR；在主 UV pipeline 中，
  其返回的 `topheavy_source_grid` 可用于限制 SSP 选择、UV 卷积和 UVLF 权重
- `birth_metallicity_zsun_grid` 记录本步成星前的 gas metallicity；`gas_metallicity_zsun_grid` 记录本步富集后的 gas metallicity
- birth-metallicity gate 使用本步成星前的 `birth_metallicity_zsun_grid`，避免同一步新产生的 metals 反向决定自己的 IMF
- 金属丰度以 `Z/Zsun` 存储，默认 `Zsun=0.0142`
- 若需要确定性检查，可把所有 scatter 设为 `0.0`

## `auroralf.ssp.load_uv1600_table()`

导入：

```python
from auroralf.ssp import load_uv1600_table
```

输入：

- `file_path`
  SSP 光谱文件路径，例如 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
  或 `external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5`
- `wavelength_a`
  目标波长，单位 `Angstrom`，默认 `1600.0`
- `metallicity`
  仅对 `HDF5` SSP 文件生效，单位是线性 `Z/Zsun`；必须精确匹配文件中的离散金属丰度选项，例如 `0.05`、`0.1`、`0.2`

输出：

- `ages_myr`
  SSP 年龄数组，单位 `Myr`
- `luminosity_per_msun`
  对应波长下的单位恒星质量光度，单位 `erg/s/Hz/Msun`

说明：

- 内部带缓存；同一个文件和波长组合只会实际读取一次
- 对 `.dat` 文件保持现有读取行为不变
- 对 `.hdf5` 文件会直接使用文件内年龄网格，并按 `metallicity=Z/Zsun` 精确选择金属丰度 bin
- 当前这批 BPASS `HDF5` 模板中的 `spectra` 已经是按单位恒星质量归一化的 `Lnu/Msun`
- 读取 `.hdf5` 文件需要 `h5py`

## `auroralf.ssp.interpolate_uv1600_luminosity_per_msun()`

导入：

```python
from auroralf.ssp import interpolate_uv1600_luminosity_per_msun
```

输入：

- `time_myr`
  需要查询的 SSP 年龄，单位 `Myr`；支持标量或 `numpy.ndarray`
- `file_path`
  SSP 光谱文件路径
- `wavelength_a`
  目标波长，单位 `Angstrom`，默认 `1600.0`
- `metallicity`
  仅对 `HDF5` SSP 文件生效，单位是线性 `Z/Zsun`；必须精确匹配文件中的离散金属丰度选项

输出：

- 插值后的单位恒星质量光度，单位 `erg/s/Hz/Msun`
  输入是标量时返回 `float`，输入是数组时返回 `numpy.ndarray`

说明：

- 采用对 `log10(age)` 的一维线性插值
- 超出表格年龄范围时会夹到边界值

最小调用：

```python
from auroralf.ssp import interpolate_uv1600_luminosity_per_msun

lum_1600 = interpolate_uv1600_luminosity_per_msun(
    time_myr=10.0,
    file_path="external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat",
)
```

HDF5 示例：

```python
from auroralf.ssp import load_uv1600_table

ages_myr, luv_per_msun = load_uv1600_table(
    file_path="external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5",
    metallicity=0.05,
)
```

## `auroralf.ssp.compute_halo_uv_luminosity()`

导入：

```python
from auroralf.ssp import compute_halo_uv_luminosity
```

输入：

- `t_obs`
  观测时刻；需与 `t_history`、`ssp_age_grid`、`t_z50` 使用同一时间单位
- `t_history`
  halo 历史时间数组；函数内部会兼容非升序输入
- `mh_history`
  与 `t_history` 对应的 halo mass 历史
- `sfr_history`
  与 `t_history` 对应的恒星形成率历史，单位 `Msun/yr`
- `ssp_age_grid`
  SSP 年龄网格；需与 `t_history` 使用同一时间单位
- `ssp_luv_grid`
  SSP UV 光度核，单位 `erg/s/Hz/Msun`
- `M_min`
  最小 halo 质量阈值
- `t_z50`
  `z=50` 对应的宇宙时间
- `time_unit_in_years`
  时间单位换算到 `yr` 的系数；若时间数组使用 `Gyr`，默认 `1e9`
- `return_details`
  是否额外返回卷积起点和实际积分网格等调试信息

输出：

- 默认返回 `L_uv_halo`
  观测时刻 halo 的总 UV 光度，单位 `erg/s/Hz`
- 当 `return_details=True` 时返回 `dict`
  包含：
  - `L_uv_halo`
  - `ti`
  - `mask_used`
  - `age_used`
  - `t_used`
  - `kernel_used`
  - `integrand_used`
  - `t_cross_Mmin`

说明：

- 卷积公式为 `L_uv = ∫ SFR(t') * L_uv^SSP(t_obs - t') dt'`
- 卷积下限使用 `ti = max(t_z50, t_cross_Mmin)`
- `t_cross_Mmin` 在 `Mh(t)` 穿过 `M_min` 时用线性插值求出
- 若 `ti` 早于 `t_history` 的首个采样点，实际积分会从首个可用历史点开始
- `dt` 的年单位换算已显式通过 `time_unit_in_years` 处理
- SSP 核采用与现有 `auroralf.ssp` 一致的 `log10(age)` 插值风格
- 当年龄小于 SSP 最小年龄时取最小年龄值；大于最大年龄时返回 `0`
- 若 `load_uv1600_table()` 返回的是 `Myr` 年龄网格，而 `auroralf/mah` 和 `auroralf/sfr` 历史是 `Gyr`，请先做 `ssp_age_grid_gyr = ages_myr / 1e3`

最小调用：

```python
from auroralf.mah import generate_halo_histories
from auroralf.sfr import compute_sfr_from_tracks
from auroralf.ssp import compute_halo_uv_luminosity, load_uv1600_table

histories = generate_halo_histories(n_tracks=1, z_final=6.0, Mh_final=1e11)
sfr_tracks = compute_sfr_from_tracks(histories.tracks)

ages_myr, luv_per_msun = load_uv1600_table(
    "external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
)
ssp_age_grid_gyr = ages_myr / 1e3

halo_mask = sfr_tracks["halo_id"] == 0
L_uv = compute_halo_uv_luminosity(
    t_obs=float(sfr_tracks["t_gyr"][halo_mask][-1]),
    t_history=sfr_tracks["t_gyr"][halo_mask],
    mh_history=sfr_tracks["Mh"][halo_mask],
    sfr_history=sfr_tracks["SFR"][halo_mask],
    ssp_age_grid=ssp_age_grid_gyr,
    ssp_luv_grid=luv_per_msun,
    M_min=1e8,
    t_z50=float(sfr_tracks["t_gyr"][halo_mask][0]),
)
```

## `auroralf.uvlf.run_halo_uv_pipeline()`

导入：

```python
from auroralf.uvlf import run_halo_uv_pipeline
```

输入：

- `n_tracks`
  要生成并卷积的 halo 条数
- `z_final`
  观测红移
- `Mh_final`
  在 `z_final` 处的最终 halo mass
- `z_start_max`
  回溯的最高红移，默认 `50.0`
- `n_grid`
  redshift grid 点数，默认 `240`
- `ssp_file`
  canonical Pop II SSP 光谱文件路径；默认使用 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
- `topheavy_ssp_file`
  mild top-heavy Pop II SSP 光谱文件路径；默认使用 `external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5`
- `topheavy_ssp_metallicity`
  读取 HDF5 top-heavy SSP 时使用的金属丰度，单位为 `Z/Zsun`；默认 `0.05`
- `imf_mode`
  Pop II IMF 模式，支持：
  - `"canonical"`：所有源时刻都使用 canonical Pop II SSP
  - `"z10_mild_topheavy"`：源时刻满足 `z >= z_topheavy_min`，且 birth metallicity 不超过阈值时使用 mild top-heavy SSP
  - `"mah_burst_mild_topheavy"`：同时满足 `z >= z_topheavy_min`、`Mh / dMh_dt <= growth_time_threshold_myr`，且 birth metallicity 不超过阈值时使用 mild top-heavy SSP
- `imf_transition_parameters`
  `auroralf.uvlf.IMFTransitionParameters`，默认 `z_topheavy_min=10.0`、`growth_time_threshold_myr=50.0`、
  `metallicity_topheavy_max_zsun=0.05`；若设为 `None`，则关闭金属丰度 gate
- `cosmology`
  `auroralf.mah.Cosmology`；未提供时使用项目默认宇宙学
- `random_seed`
  随机种子
- `sampler`
  `auroralf.mah` 参数抽样方式，默认 `"mcbride"`
- `enable_time_delay`
  是否在 `auroralf.sfr` 计算中启用基于 dynamical time 的 extended-burst 延迟核，默认 `False`
- `workers`
  保留的接口参数；当前实现中 `run_halo_uv_pipeline()` 内部 UV 卷积按串行执行
- `metal_enrichment_parameters`
  可选 `auroralf.chemistry.MetalEnrichmentParameters`；提供时沿 MAH/SFR 历史计算 stochastic one-zone metallicity 诊断。
  非 canonical IMF 模式若启用 `metallicity_topheavy_max_zsun`，必须提供该参数
- `metallicity_random_seed`
  金属演化 stochastic scatter 的随机种子；不影响 MAH 抽样随机种子
- `burst_scatter_dex`
  对源时刻 SFR 施加 lognormal burst scatter 的标准差，单位 dex；默认 `0.0` 表示关闭
- `burst_scatter_timescale_myr`
  burst scatter 在同一 halo 内保持相关的时间尺度，单位 `Myr`；默认 `20.0`
- `burst_scatter_random_seed`
  burst scatter 随机种子；若未提供，则复用该 halo 的 MAH `random_seed`
- `burst_scatter_preserve_mean`
  是否使用 unit-mean lognormal shift，使 ensemble-mean SFR 不因 scatter 系统性升高；默认 `True`

输出：

- `HaloUVPipelineResult`

## `HaloUVPipelineResult`

字段：

- `histories`
  `auroralf.mah.generate_halo_histories()` 返回的原始 `HaloHistoryResult`
- `sfr_tracks`
  `auroralf.sfr.compute_sfr_from_tracks()` 输出的扁平表格
- `uv_luminosities`
  每个 halo 在 `z_final` 的总 UV 光度，单位 `erg/s/Hz`
- `uv_luminosities_canonical`
  canonical Pop II SSP 对总 UV 光度的分量
- `uv_luminosities_topheavy`
  mild top-heavy Pop II SSP 对总 UV 光度的分量
- `redshift_grid`
  这次计算使用的 redshift grid
- `floor_mass`
  从有效历史点反推出的有效 `M_min(z)` 下限，可直接用于画图
- `active_grid`
  每个 halo 每个时间步是否仍处于有效区间
- `imf_topheavy_source_grid`
  每个 halo 每个源时刻是否实际使用 mild top-heavy SSP kernel；若启用金属 gate，该字段已经过
  `birth_metallicity_zsun_grid <= metallicity_topheavy_max_zsun` 筛选
- `gas_metallicity_zsun_grid`
  若启用金属演化，返回每个 halo 每个时间步的富集后 gas metallicity，单位 `Z/Zsun`
- `birth_metallicity_zsun_grid`
  若启用金属演化，返回每个 halo 每个时间步成星前的 birth metallicity，单位 `Z/Zsun`
- `metal_mass_grid`
  若启用金属演化，返回诊断 metal mass，单位 `Msun`
- `gas_mass_grid`
  若启用金属演化，返回诊断 gas reservoir mass，单位 `Msun`
- `metadata`
  包含 `n_tracks`、`steps_per_halo`、`workers`、`canonical_ssp_file`、`topheavy_ssp_file`、`imf_mode`、`topheavy_source_fraction`、`topheavy_candidate_source_fraction`、`stochastic_metallicity_enabled`、`final_gas_metallicity_zsun_median`、`birth_metallicity_zsun_starforming_median`、`burst_scatter_dex`、`burst_sfr_multiplier_median`、`enable_time_delay` 和各阶段耗时

说明：

- 这个函数封装了完整主流程：`auroralf.mah -> auroralf.sfr -> auroralf.ssp UV convolution`
- `auroralf.mah` 部分使用默认 `M_min`，即 `massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`
- UV 卷积只对 `active_flag=True` 的有效历史段进行
- Pop II top-heavy 不是全局替换 SSP，而是按 `imf_mode` 在源时刻选择 canonical 或 mild top-heavy SSP kernel
- 默认 mild top-heavy 还要求本步成星前 `Z_birth <= 0.05 Zsun`；这个阈值位于低金属 IMF 过渡区间内，并与当前 top-heavy SSP 的 `0.05 Zsun` 选择一致
- 启用金属演化时，实际通过红移/MAH 和 birth-metallicity gate 的 top-heavy source-time 会同时决定 SSP kernel 与 `topheavy_yield_multiplier` 金属产额
- 当前标定值 `topheavy_yield_multiplier=1.28` 来自 `z=12.5` 代表性 halo 的 FIRE-2
  高红移 MZR 约束；`3.0` 会使 redshift-gated top-heavy 分支明显过富集
- 可选 burst scatter 使用
  `SFR_burst(t)=SFR_smooth(t) 10^Delta`，其中
  `Delta ~ Normal(-0.5 ln(10) sigma_burst^2, sigma_burst)`；该均值位移保证
  `E[10^Delta]=1`，因此 scatter 默认不整体抬高平均 SFR
- `load_uv1600_table()` 读出的 SSP 年龄网格会自动从 `Myr` 转成 `Gyr` 后再参与卷积

最小调用：

```python
from auroralf.uvlf import run_halo_uv_pipeline

result = run_halo_uv_pipeline(
    n_tracks=10000,
    z_final=6.0,
    Mh_final=1e12,
    workers=32,
)

print(result.uv_luminosities.shape)
print(result.metadata["timing_seconds"])
```

## `auroralf.uvlf.sample_uvlf_from_hmf()`

导入：

```python
from auroralf.uvlf import sample_uvlf_from_hmf
```

输入：

- `z_obs`
  观测红移
- `N_mass`
  外层 Monte Carlo 抽取的 halo 终质量个数，默认 `3000`
- `n_tracks`
  每个质量点内层生成的 luminosity realization 个数，默认 `1000`
- `random_seed`
  随机种子
- `quantity`
  统计对象，支持 `"Muv"` 和 `"luminosity"`；默认 `"Muv"`
- `bins`
  histogram 的 bin 数或 bin edges
- `logM_min`
  外层均匀抽样的最低 `log10 Mh`，默认 `9`
- `logM_max`
  外层均匀抽样的最高 `log10 Mh`，默认 `13`
- `z_start_max`
  内层 `auroralf.mah` 回溯的最高红移，默认 `50.0`
- `n_grid`
  内层 `auroralf/mah` 和 `auroralf/sfr` 使用的 redshift grid 点数，默认 `240`
- `sampler`
  `auroralf.mah` 参数抽样方式，默认 `"mcbride"`
- `enable_time_delay`
  是否在 `auroralf.sfr` 中启用时间延迟，默认 `False`
- `pipeline_workers`
  外层 `N_mass` 质量点采样使用的并行 worker 数
- `ssp_file`
  canonical Pop II SSP 文件路径；默认使用 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
- `topheavy_ssp_file`
  mild top-heavy Pop II SSP 文件路径；仅非 canonical IMF 模式实际读取
- `topheavy_ssp_metallicity`
  HDF5 mild top-heavy SSP 的金属丰度，单位为 `Z/Zsun`；默认 `0.05`
- `imf_mode`
  同 `run_halo_uv_pipeline()`，默认 `"canonical"`
- `imf_transition_parameters`
  mild top-heavy IMF 的源时刻触发参数；默认包含 `metallicity_topheavy_max_zsun=0.05`
- `progress_path`
  可选进度文件路径；若提供，会把外层 `N_mass` 循环进度持续写入该 txt 文件
- `mass_function_model`
  外层 halo mass function 权重模型；当前生产接口只支持 `"hmf_reed07"`，使用 `hmf` 包中的 Reed07 fitting function。旧的 `"massfunc_st"` 和 Watson13 分支已禁用。
- `metal_enrichment_parameters`
  可选 `auroralf.chemistry.MetalEnrichmentParameters`；会透传给每个质量点的 `run_halo_uv_pipeline()`
- `metallicity_random_seed`
  金属演化随机种子；外层每个质量点会使用 `metallicity_random_seed + mass_index`
- `burst_scatter_dex`
  透传给每个质量点的 SFR burst scatter 标准差，单位 dex；默认 `0.0`
- `burst_scatter_timescale_myr`
  burst scatter 时间相关尺度，单位 `Myr`；默认 `20.0`
- `burst_scatter_random_seed`
  burst scatter 随机种子；外层每个质量点会使用 `burst_scatter_random_seed + mass_index`
- `burst_scatter_preserve_mean`
  是否保持 lognormal multiplier 的 ensemble mean 为 1；默认 `True`

输出：

- `UVLFSamplingResult`

## `UVLFSamplingResult`

字段：

- `samples`
  样本表，包含：
  - `logMh`
  - `Mh`
  - `mass_weight`
  - `track_index`
  - `luminosity`
  - `topheavy_light_fraction`
  - `Muv`
  - `sample_weight`
- `auroralf.uvlf`
  UVLF histogram 结果，包含：
  - `quantity`
  - `bin_edges`
  - `bin_centers`
  - `bin_width`
  - `raw_counts`
  - `weighted_counts`
  - `weight_squared_counts`
  - `weighted_count_sigma`
  - `effective_counts`
  - `phi`
  - `phi_sigma`
- `metadata`
  运行参数、耗时信息、top-heavy 诊断和可选金属演化诊断；启用金属演化时包含 `final_gas_metallicity_zsun_median_by_mass` 和 `birth_metallicity_zsun_starforming_median_by_mass`

说明：

- 外层在 `log10 Mh in [9, 13]` 上均匀抽样
- 外层权重默认使用 `hmf` 包的 Reed07 halo mass function：
  - `dn/dlogM = M ln(10) dn/dM`
- `hmf` 的质量单位从 `Msun/h` 转成项目内部使用的 `Msun`，`dn/dM` 从 `h^4 Mpc^-3 Msun^-1` 转成 `Mpc^-3 Msun^-1`
- 若传入旧的 `mass_function_model="massfunc_st"` 或 `"hmf_watson13_fof"`，接口会显式报错，避免误用历史分支
- 每个质量点的总权重会平均分配给其 `n_tracks` 个 luminosity realization
- 内层条件采样器直接复用 `auroralf.uvlf.run_halo_uv_pipeline()`
- 当前并行层级放在外层 `N_mass` 循环；`run_halo_uv_pipeline()` 内部 UV 卷积保持串行，避免嵌套进程池
- 若设置 `progress_path`，外层 `N_mass` 进度条会实时写入文本文件
- 非 canonical IMF 模式默认使用 birth-metallicity gate，因此需要传入 `metal_enrichment_parameters`；
  关闭该 gate 时可把 `IMFTransitionParameters.metallicity_topheavy_max_zsun` 设为 `None`
- `burst_scatter_dex > 0` 时，金属演化和 UV 卷积使用同一条 burst 后的 SFR 历史；
  canonical 与 top-heavy mode 若使用同一个 `burst_scatter_random_seed`，会共享同一组 burst realization

最小调用：

```python
from auroralf.uvlf import sample_uvlf_from_hmf

result = sample_uvlf_from_hmf(
    z_obs=6.0,
    N_mass=3000,
    n_tracks=1000,
    pipeline_workers=32,
)

print(result.samples["Muv"].shape)
print(result.uvlf["phi"])
```

## 生产 UVLF 脚本中的金属演化选项

生产脚本 `scripts/run/run_uvlf_compare_imf_no_delay_all_z.py` 必须通过 SLURM wrapper 提交。
生产脚本默认启用 `enable_time_delay=True`；若要做历史 no-delay 对照，需要显式传入 `--disable-time-delay`。
启用 stochastic metallicity 的 dry-run 示例：

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --enable-stochastic-metallicity --metallicity-random-seed 123 --metallicity-topheavy-max-zsun 0.05 --metal-topheavy-yield-multiplier 1.28
```

no-delay 对照示例：

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --disable-time-delay --enable-stochastic-metallicity --metallicity-random-seed 123 --metallicity-topheavy-max-zsun 0.05 --metal-topheavy-yield-multiplier 1.28
```

常用金属演化参数：

- `--enable-stochastic-metallicity`
  启用 one-zone metallicity 诊断
- `--metallicity-random-seed`
  stochastic scatter 随机种子
- `--metal-gas-fraction-of-baryons`
  诊断 gas reservoir 占 halo baryon mass 的比例
- `--metal-yield`
  canonical source-time 的 metal yield
- `--metal-topheavy-yield-multiplier`
  top-heavy source-time 的 metal yield 倍增因子，默认使用 MZR 标定值 `1.28`
- `--metallicity-topheavy-max-zsun`
  mild top-heavy IMF 的 birth-metallicity 上限，单位 `Z/Zsun`，默认 `0.05`
- `--disable-metallicity-topheavy-gate`
  关闭 birth-metallicity gate，用于历史分支对照；默认不关闭
- `--metal-returned-fraction`
  即时返回比例
- `--metal-mass-loading-norm`
  mass loading 归一化
- `--metal-yield-scatter-dex`、`--metal-mass-loading-scatter-dex`、`--metal-birth-scatter-dex`
  stochastic scatter，单位 dex

脚本输出的 `.npz` 和 summary txt 会记录金属演化开关、金属阈值、参数、每个红移/IMF mode 的 `final_gas_metallicity_zsun_median` 和 `birth_metallicity_zsun_starforming_median`。若默认金属 gate 开启且运行非 canonical IMF mode，必须同时启用 `--enable-stochastic-metallicity`。

## 生产 UVLF 脚本中的 burst scatter 选项

生产脚本默认不启用 burst scatter。启用 `0.5 dex`、`20 Myr` 相关尺度的示例：

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --burst-scatter-dex 0.5 --burst-scatter-timescale-myr 20 --burst-scatter-random-seed 777
```

常用参数：

- `--burst-scatter-dex`
  SFR lognormal scatter 的标准差，单位 dex；`0.3 dex` 约对应瞬时 SFR 的因子 `2`
- `--burst-scatter-timescale-myr`
  同一个随机 multiplier 在 halo 历史中保持相关的时间尺度；默认 `20 Myr`
- `--burst-scatter-random-seed`
  burst scatter 随机种子；生产脚本对不同红移加 `1000*z_index`，对不同 IMF mode 保持相同 seed，
  因此 canonical 与 top-heavy 分支使用同一组 burst realization
- `--disable-burst-scatter-mean-preservation`
  关闭 unit-mean lognormal 修正；默认不关闭

脚本输出的 `.npz` 会记录 `burst_scatter_dex`、`burst_scatter_timescale_myr`、`burst_scatter_random_seed` 和 `burst_scatter_preserve_mean`。

## `auroralf.uvlf.compute_dust_attenuated_uvlf()`

导入：

```python
from auroralf.uvlf import compute_dust_attenuated_uvlf
```

输入：

- `intrinsic_muv`
  intrinsic UVLF 的绝对星等网格
- `intrinsic_phi`
  intrinsic UVLF，单位通常为 `Mpc^-3 mag^-1`
- `z`
  观测红移
- `muv_obs`
  输出时使用的 observed magnitude 网格；未提供时默认使用 `intrinsic_muv`
- `c0`, `c1`, `m0`
  尘埃修正公式
  `A_UV = max(c1 + c0 * beta, 0)` 和 `beta = beta0 + dbeta * (M_UV^obs - m0)` 中的系数
- `clip_to_bounds`
  是否把映射后的 intrinsic magnitude 截断到输入网格边界内，默认 `True`
- `match_faint_end_after_intersection`
  保留的兼容接口参数；当前实现中不再使用旧的交点拼接逻辑
- `insert_transition_point`
  保留的兼容接口参数；当前实现中不再使用旧的交点插点逻辑

输出：

- 返回一个字典，常用字段包括：
  - `Muv_obs`
  - `Muv_intrinsic`
  - `A_uv`
  - `dMuv_dMuv_obs`
  - `phi_nodust_obs`
  - `phi_intrinsic_interp`
  - `phi_obs`
  - `phi_obs_eval`
  - `transition_index`

说明：

- 先按公式计算
  `phi_obs_raw(M_UV^obs) = phi_int(M_UV) * dM_UV / dM_UV^obs`
- 当前最终返回的 dust UVLF 采用物理裁剪：
  `phi_obs = min(phi_obs_raw, phi_nodust_obs)`
- 因此最终的 dust 曲线不会高于 no-dust 曲线
- `phi_obs_eval` 保留未经裁剪的原始 dust 结果，便于调试

最小调用：

```python
from auroralf.uvlf import sample_uvlf_from_hmf, compute_dust_attenuated_uvlf
import numpy as np

result = sample_uvlf_from_hmf(
    z_obs=6.0,
    N_mass=3000,
    n_tracks=1000,
    bins=np.linspace(-28.0, -10.0, 21),
    pipeline_workers=32,
)

dust_result = compute_dust_attenuated_uvlf(
    intrinsic_muv=result.uvlf["bin_centers"],
    intrinsic_phi=result.uvlf["phi"],
    z=6.0,
    muv_obs=np.linspace(-28.0, -10.0, 400),
)

print(dust_result["phi_obs"])
```

## `auroralf.uvlf` 尘埃修正辅助函数

导入：

```python
from auroralf.uvlf import (
    intrinsic_muv_from_observed,
    intrinsic_muv_jacobian,
    uv_continuum_slope_beta,
    uv_dust_attenuation,
)
```

说明：

- `uv_continuum_slope_beta(muv_obs, z)`
  返回 Bouwens 型 UV continuum slope `beta`
- `uv_dust_attenuation(muv_obs, z, c0=2.10, c1=4.85, m0=-19.5)`
  返回 `A_UV`
- `intrinsic_muv_from_observed(muv_obs, z, ...)`
  返回 `M_UV = M_UV^obs - A_UV`
- `intrinsic_muv_jacobian(muv_obs, z, ...)`
  返回 `dM_UV / dM_UV^obs`

最小调用：

```python
from auroralf.uvlf import uv_dust_attenuation, intrinsic_muv_from_observed

muv_obs = [-22.0, -20.0, -18.0]
auv = uv_dust_attenuation(muv_obs, z=6.0)
muv_intrinsic = intrinsic_muv_from_observed(muv_obs, z=6.0)
```

## `auroralf.uvlf.compute_reed07_halo_mass_function_dndm()`
导入：

```python
from auroralf.uvlf import compute_reed07_halo_mass_function_dndm
```

输入：

- `halo_mass_msun`
  halo mass；支持标量或 `numpy.ndarray`
- `z_obs`
  红移

输出：

- `dndm`
  Reed07 halo mass function `dn/dM`，单位为 `Mpc^-3 Msun^-1`

说明：

- 该接口是项目当前唯一的 HMF 生产接口
- 底层调用 `hmf.MassFunction(hmf_model="Reed07")`
- 质量从 `hmf` 的 `Msun/h` 转为项目内部的 `Msun`
- `dn/dM` 从 `h^4 Mpc^-3 Msun^-1` 转为 `Mpc^-3 Msun^-1`

最小调用：

```python
import numpy as np
from auroralf.uvlf import compute_reed07_halo_mass_function_dndm

masses = np.logspace(8, 12, 100)
dndm = compute_reed07_halo_mass_function_dndm(masses, 6.0)
```
