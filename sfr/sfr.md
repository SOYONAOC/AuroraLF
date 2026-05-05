# SFR Module Notes

This file is the module-level scientific note for `sfr.compute_sfr_from_tracks`.
The short public API summary lives in `README.md`; keep this file for the
formulae, implementation details, and caveats that are useful when editing
`sfr/calculator.py`.

## Entry Point

```python
from sfr import compute_sfr_from_tracks

sfr_tracks = compute_sfr_from_tracks(
    tracks,
    enable_time_delay=False,
)
```

The input `tracks` should follow `mah.HaloHistoryResult.tracks` and must contain
flat arrays with the same length for:

- `halo_id`
- `step`
- `z`
- `t_gyr`
- `Mh`
- `dMh_dt`

The function groups rows by `halo_id` and sorts each halo by `t_gyr` when the
input is not already grouped and time-ordered. The returned arrays follow that
grouped/sorted order.

## Model Parameters

The baryon fraction is taken from the project cosmology:

```text
f_b = Omega_b / Omega_m
```

The stellar formation efficiency is

```text
f_star(M_h) = 2 epsilon_0 /
              [(M_h / M_c)^(-beta_star) + (M_h / M_c)^(gamma_star)]
```

The default parameter values are stored in `DEFAULT_SFR_MODEL_PARAMETERS`:

```text
epsilon_0 = 0.12
M_c       = 10^11.7 Msun
beta_star = 0.66
gamma_star = 0.65
```

Pass a `SFRModelParameters` instance to change these values explicitly.

## Virial Quantities

For each row, the function computes:

- `r_vir`
- `V_c`
- `T_vir`
- `tau_del = r_vir / V_c`
- `td_burst`, the free-fall/dynamical burst time used by the delay kernel

Rows below the atomic-cooling threshold are inactive:

```text
T_vir < atomic_cooling_temperature  ->  SFR = 0
```

The default threshold is `1e4 K`.

## No-Delay Branch

The default branch is `enable_time_delay=False`. It uses the current halo state:

```text
SFR(t) = f_b * f_star[M_h(t)] * dM_h/dt(t) / 1e9
```

The division by `1e9` converts the project halo accretion rate convention from
`Msun/Gyr` to `Msun/yr`.

## Delay Branch

When `enable_time_delay=True`, the current implementation uses an extended
burst kernel rather than a single source-time sample. The source term is:

```text
q(t') = f_star[M_h(t')] * dM_h/dt(t')
```

The kernel is:

```text
g(Delta t) = Delta t / (kappa^2 t_d^2) * exp[-Delta t / (kappa t_d)]
```

where `Delta t = t - t'`, `t_d = td_burst`, and the default
`EXTENDED_BURST_KAPPA` is `0.1`.

The delayed source rate is:

```text
q_delay(t) = integral g(t - t') q(t') dt'
```

and the final SFR is:

```text
SFR_delay(t) = f_b * q_delay(t) / 1e9
```

The integral is limited to `burst_lookback_max_myr`; the default
`EXTENDED_BURST_LOOKBACK_MAX_MYR` is `100.0`.

For shared regular time grids, `sfr/calculator.py` uses a vectorized
matrix-kernel path. For irregular or differently sampled histories, it uses the
grouped loop implementation. Both paths are intended to return the same model
quantity.

## Source-Time Diagnostics

The function still computes the legacy one-time-delay diagnostic columns:

- `t_src = t_gyr - tau_del`
- `Mh_src`
- `dMh_dt_src`
- `fstar_src`

These are useful for comparing to the older source-time formulation, but they
are not the source of `SFR` when `enable_time_delay=True`. In the delay branch,
`SFR` comes from the extended-burst convolution above.

The `mdot_burst` column is also diagnostic: it is the same kernel applied to
`dMh_dt` alone. The physical delayed SFR uses the kernel applied to
`f_star(M_h) * dMh_dt`.

## Returned Columns

The returned dictionary contains the sorted input columns plus:

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

## UVLF Pipeline Use

`uvlf.run_halo_uv_pipeline()` forwards these controls:

- `enable_time_delay`
- `burst_lookback_max_myr`
- `sfr_model_parameters`

Use the default no-delay branch for baseline UVLF runs. Enable the delay branch
only when comparing burst-delay effects or when the run metadata explicitly
needs the extended-burst model.
