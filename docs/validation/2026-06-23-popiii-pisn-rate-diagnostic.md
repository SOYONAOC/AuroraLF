# Pop III PISN Rate Diagnostic

Date: 2026-06-23

Main merge: `4da1477 Add Pop III PISN rate diagnostics`

Slide deck: `slides/popiii_pisn_summary_20260623/`

## Scope

This note records the Pop III pair-instability supernova (PISN) diagnostic used
to compare the current Pop III halo upper-mass boundary with a fixed
`M_up = 1e10 Msun` model.

The direct diagnostic is

```text
n_dot_PISN(z) = rho_SFR,III(z) * eta_PISN
```

where `rho_SFR,III` is the HMF-weighted Pop III star-formation-rate density from
`sample_uvlf_from_hmf()`. For the fiducial extreme Pop III IMF used here,

```text
phi(M) proportional to M^-2.35, 50 <= M/Msun <= 500
PISN window: 140 <= M/Msun <= 260
eta_PISN = integral_140^260 phi(M) dM / integral_50^500 M phi(M) dM
         = 1.3221293020e-3 Msun^-1
```

The comparison keeps all other parameters fixed and changes only the Pop III
upper halo-mass mode:

- current: `M_up = M_vir(T_vir = 1e4 K)`
- fixed: `M_up = 1e10 Msun`

The rate is PISN-only for the adopted IMF window; it is not the total Pop III
supernova rate.

## Direct Rate Results

Source CSV:
`outputs/popiii_mup_pisn_rate_from_sfr.csv`

| z | current n_dot_PISN [yr^-1 Mpc^-3] | fixed 1e10 n_dot_PISN [yr^-1 Mpc^-3] | fixed/current |
|---:|---:|---:|---:|
| 6.0 | 2.5625e-7 | 1.0230e-6 | 3.99 |
| 8.0 | 3.7167e-7 | 1.1922e-6 | 3.21 |
| 10.0 | 4.1593e-7 | 1.0634e-6 | 2.56 |
| 12.5 | 3.7477e-7 | 7.3023e-7 | 1.95 |
| 14.5 | 2.9954e-7 | 4.8731e-7 | 1.63 |

The fixed `M_up = 1e10 Msun` model reaches the commonly discussed
`~1e-6 yr^-1 Mpc^-3` level at `z = 6-10`. In the same units converted to
`Gpc^-3 yr^-1`, the fixed model gives about `490-1190 Gpc^-3 yr^-1` over
`z = 6-14.5`.

## Why The Boost Is Larger At Lower Redshift

Source CSV:
`outputs/popiii_pisn_sfrd_boost_decomposition.csv`

The direct reason is not `eta_PISN`; that factor is fixed for both models. The
ratio is set by how much Pop III SFRD is added when the upper halo-mass boundary
is extended from the atomic-cooling threshold to `1e10 Msun`.

| z | M_atomic [Msun] | total SFRD ratio | delta from below M_atomic | delta from M_atomic to 1e10 | delta above 1e10 |
|---:|---:|---:|---:|---:|---:|
| 6.0 | 1.58e8 | 3.99 | 10.9% | 84.1% | 4.9% |
| 8.0 | 1.08e8 | 3.21 | 14.9% | 82.1% | 2.9% |
| 10.0 | 8.02e7 | 2.56 | 19.6% | 79.4% | 1.1% |
| 12.5 | 5.90e7 | 1.95 | 27.6% | 72.2% | 0.2% |
| 14.5 | 4.80e7 | 1.63 | 36.8% | 63.2% | 0.0% |

So the low-redshift enhancement is large because the newly allowed
`M_atomic <= M_h < 1e10 Msun` interval contributes most of the additional Pop III
SFRD. At higher redshift, the same interval still dominates the increment, but
the total fixed/current ratio is smaller because the current model already
captures a larger fraction of the Pop III SFRD and the absolute added
high-mass contribution is lower.

This also explains why the high-redshift UVLF bright end and the integrated
Pop III SFRD/PISN rate are not the same sensitivity test. A UVLF bright-end
comparison weights rare luminous systems and dust/visibility effects, while the
direct PISN diagnostic integrates Pop III SFRD over all sampled halos.

## Observational Interpretation

The present result is an intrinsic comoving rate-density diagnostic. It is not
yet a survey detection prediction.

Existing long-duration supernova searches provide useful scale checks, but they
do not directly exclude the `z > 6` Pop III PISN rates here. For example, HSC
constraints on bright, long-timescale PISN-like events are mostly relevant at
lower redshift and depend on light curves, survey cadence, limiting magnitude,
and selection efficiency. The fixed `M_up = 1e10 Msun` model is therefore best
read as bringing the Pop III PISN rate into the literature-scale
`~1e-6 yr^-1 Mpc^-3` regime, not as already ruled out.

A direct observational comparison should compute

```text
N_det = integral dz dOmega [n_dot_PISN(z) / (1 + z)]
        [dV / dz / dOmega] P_det(z, light_curve, cadence, m_lim)
```

The next useful step is a JWST/Roman/HSC-like detection-number estimate using a
chosen PISN light-curve model and survey selection function.

## Reproduction Commands

Direct recomputation:

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_popiii_mup_pisn_rate_from_sfr.py
```

Redraw the two-panel slide figure from the saved CSV:

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_popiii_mup_pisn_rate_from_sfr.py \
  --input-csv outputs/popiii_mup_pisn_rate_from_sfr.csv \
  --no-ratio-panel \
  --output-prefix slides/popiii_pisn_summary_20260623/assets/popiii_mup_pisn_rate_from_sfr_two_panel
```

Compile the slide deck:

```bash
cd slides/popiii_pisn_summary_20260623
xelatex -interaction=nonstopmode -halt-on-error popiii_pisn_summary_20260623.tex
xelatex -interaction=nonstopmode -halt-on-error popiii_pisn_summary_20260623.tex
```

## Verification

Post-merge focused checks on `main`:

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  auroralf/uvlf/hmf_sampling.py \
  scripts/plot/plot_popiii_mup_pisn_proxy.py \
  scripts/plot/plot_popiii_mup_pisn_rate_from_sfr.py
```

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_hmf_sampling.py \
  tests/test_popiii_model.py \
  tests/test_popiii_pisn_proxy_plot.py
```

Result: `33 passed`.

The Beamer deck compiled successfully with XeLaTeX as an 8-page PDF, and the
compiled pages were visually inspected for the PISN formula, two-panel rate
figure, numerical table, literature-scale slide, observational-limit slide, and
summary slide.

## Files Added Or Updated

- `auroralf/uvlf/hmf_sampling.py`: records sample-level `sfr` and `popiii_sfr`
  plus metadata `sfrd_msun_yr_mpc3` and `popiii_sfrd_msun_yr_mpc3`.
- `auroralf/uvlf/uvlf.md`: documents the SFRD metadata.
- `scripts/plot/plot_popiii_mup_pisn_rate_from_sfr.py`: direct PISN
  rate-density calculation from Pop III SFRD.
- `scripts/plot/plot_popiii_mup_pisn_proxy.py`: exploratory luminosity-density
  proxy helper.
- `slides/popiii_pisn_summary_20260623/`: compact slide deck summarizing the
  PISN result.
- `tests/test_hmf_sampling.py` and `tests/test_popiii_pisn_proxy_plot.py`:
  focused regression coverage for the new SFRD metadata and IMF-window helper.
