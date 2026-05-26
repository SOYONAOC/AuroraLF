# AGENTS.md

## Scope

These instructions apply to the whole AuroraLF repository.

Work from the repository root unless a task explicitly says otherwise. Treat
this repository as a research codebase: preserve scientific meaning, data
provenance, and reproducibility ahead of cosmetic cleanup.

## Agent-Owned Workflow

Agents are responsible for the full technical workflow: reading code, planning,
editing, running jobs, checking outputs, debugging failures, and summarizing the
result. Do not ask the user to inspect code, logs, or implementation details as
the primary review path.

The user's review surface is slides. When a result needs human inspection,
prepare or update the relevant slide deck under `slides/` and keep the chat
summary focused on conclusions, assumptions, and remaining decisions rather than
code-level detail.

## Project Map

- `auroralf/mah/`: Monte Carlo halo assembly history generation.
- `auroralf/sfr/`: star-formation model utilities.
- `auroralf/chemistry/`: stochastic one-zone metal enrichment diagnostics
  along fixed MAH/SFR histories.
- `auroralf/ssp/`: SSP loading and UV convolution utilities.
- `auroralf/uvlf/`: UV luminosity function sampling, HMF weighting, dust
  correction, and Pop II IMF mode logic.
- `tests/`: focused regression tests.
- `scripts/run/`: production or batch workflow entry points.
- `scripts/submit/`: SLURM submission wrappers.
- `scripts/plot/`: plotting and visual comparison scripts.
- `scripts/analysis/`: post-processing and result comparison scripts.
- `scripts/experiments/`: one-off diagnostics and exploratory scripts; do not
  treat these as stable public APIs.

## Environment

Use the project virtual environment explicitly:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

For focused checks, run only the affected tests, for example:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_imf_modes.py tests/test_hmf_sampling.py
```

Run project scripts from the repository root so relative data paths and package
imports resolve consistently:

```bash
PYTHONPATH=. .venv/bin/python scripts/experiments/<script>.py
```

Do not silently switch to the system Python when `.venv/bin/python` is expected.
If a dependency is missing, fix the project environment deliberately with `uv`
or report the setup problem.

This branch does not currently define a tracked `pyproject.toml`,
`requirements.txt`, or `uv.lock`; treat the existing `.venv` as the working
environment record. If the environment must be rebuilt, document the exact
packages and versions you installed.

## Data And Outputs

- `external_data/`: external source data, observational constraints, SSP
  spectra, empirical model releases, and literature source packages. Preserve
  source-data provenance and avoid editing these files unless the task is
  explicitly about data ingestion or correction.
- `data_save/`: reusable intermediate products, long-running `.npz` outputs,
  and summary tables.
- `outputs/`: logs, progress files, quick-look plots, and one-off diagnostics.
- `temp_data/`: scratch caches and temporary `.npz` products.
- `slides/`: Beamer sources, slide PDFs, and slide-dependent assets.
- `nbody/`: N-body experiment notes and launch documentation.

When adding new generated outputs, choose `data_save/` for reusable products and
`outputs/` for temporary diagnostics. Keep formal slide material under `slides/`.
For slide figures, prefer vector `.pdf` assets under `slides/assets/`; keep
preview rasters and draft figures in `outputs/` unless the user requests a
tracked raster asset.

Large external libraries and raw source packages may be ignored by git,
including `external_data/ssp_spectra/`,
`external_data/empirical_models/universemachine_dr1/tarballs/`, and literature
source tarballs. A fresh clone may not contain these files. If required data are
missing, report the exact missing path instead of creating fake data, changing
default paths, or silently skipping the calculation.

## Scientific Constraints

- Fail fast. Do not hide missing data, failed imports, invalid parameters, or
  failed jobs with broad fallbacks, placeholder arrays, synthetic data, cached
  plots, or silent default values.
- `main` is the unified production branch. Do not maintain a separate old
  `topheavyIMF` path in parallel; use switches in the unified model for
  historical comparisons.
- Preserve units and conversions:
  - halo mass: `Msun`
  - SFR: `Msun/yr`
  - UV luminosity: `erg/s/Hz`
  - HMF: `Mpc^-3 Msun^-1`
  - metallicity gates and chemistry summaries: linear `Z/Zsun`
  - SSP tables commonly provide ages in `Myr`; convert to `Gyr` before
    convolving with MAH/SFR histories stored in `Gyr`.
- The current production HMF path is `hmf` Reed07. Deprecated model names such
  as `massfunc_st` and `hmf_watson13_fof` intentionally raise errors; do not
  restore them unless the user explicitly asks for a historical comparison.
- Pop II top-heavy IMF variants are source-time gated. Do not replace the SSP
  globally when the intended behavior is `z10_mild_topheavy` or
  `mah_burst_mild_topheavy`.
- Current IMF modes are `canonical`, `z10_mild_topheavy`, and
  `mah_burst_mild_topheavy`. The default transition parameters are
  `source_redshift_gate_enabled=False`,
  `growth_time_threshold_myr=50.0`, and
  `metallicity_topheavy_max_zsun=0.05`. The old `z_topheavy_min=10.0`
  threshold is retained only for explicit historical comparisons.
- Do not re-enable the source-time `z>=10` top-heavy gate by default. Use
  `source_redshift_gate_enabled=True` or
  `--enable-source-redshift-topheavy-gate` only when reproducing old z-gated
  runs.
- `IMFTransitionParameters.metallicity_topheavy_max_zsun=None` disables the
  birth-metallicity gate and recovers the historical top-heavy behavior. The
  production CLI equivalent is `--disable-metallicity-topheavy-gate`.
- Non-canonical IMF modes with a non-`None` metallicity gate require
  `MetalEnrichmentParameters` / `--enable-stochastic-metallicity`; keep this as
  an explicit error rather than silently dropping the gate.
- `topheavy_ssp_metallicity` selects the HDF5 SSP template metallicity. It is
  not the gas birth-metallicity gate, even though the default value is also
  `0.05 Zsun`.
- Stochastic metallicity evolution is diagnostic along fixed MAH/SFR tracks and
  must not feed back into the SFR model. `birth_metallicity_zsun_grid` is the
  pre-star-formation metallicity used for IMF gating; `gas_metallicity_zsun_grid`
  is the post-step gas metallicity.
- The calibrated top-heavy metal yield multiplier is
  `CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER = 1.28`. Use larger values only for an
  explicit parameter sweep or historical comparison.
- The SFR delay model is controlled by `enable_time_delay`. Low-level Python
  APIs keep `False` as their backward-compatible default; the production UVLF
  script defaults to delay enabled and uses `--disable-time-delay` only for
  historical no-delay comparisons.
- `burst_scatter_dex > 0` applies a correlated lognormal multiplier to the SFR
  after the delay-SFR calculation and before metallicity/UV convolution. The
  default correlation timescale is `20 Myr`.
- The default burst scatter is mass-conserving per halo:
  `SFR_burst(t) = SFR_0(t) B(t) integral SFR_0 dt / integral SFR_0(t) B(t) dt`.
  This is recorded as `burst_scatter_mass_conserving=True`. Do not switch to
  non-conserving scatter unless the user explicitly asks for that comparison.
- UV convolution should use valid active history segments only. Do not adjust
  scientific parameters, mass cuts, redshift cuts, or units just to make a test
  or run finish.
- HDF5 top-heavy SSP loading requires an exact linear `Z/Zsun` bin, for example
  `0.05`. If required SSP data or `h5py` are missing, surface the exact missing
  dependency/path.
- The dust-corrected UVLF currently applies the physical clipping
  `phi_obs = min(phi_obs_raw, phi_nodust_obs)`.

## Production Runs

Large UVLF comparisons are compute jobs. Submit them through the SLURM wrapper
instead of running the production target directly on a login node:

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --canonical-only
```

The target script `scripts/run/run_uvlf_compare_imf_no_delay_all_z.py` is the
unified production entry point for canonical, top-heavy, metallicity-gated, and
burst-scatter UVLF runs. It requires a SLURM allocation and defaults to
`enable_time_delay=True`. The older
`scripts/run/run_uvlf_mass_function_compare_full.py` entry point is intentionally
disabled and points users to the Reed07 HMF workflow.

When running non-canonical IMF modes with the default metallicity gate, include
the stochastic-metallicity switch, for example:

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --enable-stochastic-metallicity --metallicity-random-seed 123
```

Use `scripts/analysis/run_metallicity_history_grid.py` for representative halo
metallicity histories and `scripts/analysis/sweep_metal_yield_multiplier_mzr.py`
for MZR yield-multiplier checks. Plotting scripts under `scripts/plot/` and
`scripts/analysis/` expect real observational/source files under
`external_data/`; do not synthesize missing observations.

## Verification

Run the full focused suite after model changes:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

For narrower edits, use the relevant tests:

- IMF and source-time gate behavior: `tests/test_imf_modes.py`
- stochastic metallicity and pipeline metadata: `tests/test_chemistry.py`
- mass-conserving SFR burst scatter and production CLI defaults:
  `tests/test_burst_scatter.py`
- HMF validation and Reed07 unit handling: `tests/test_hmf_sampling.py`
- metallicity history summaries: `tests/test_metallicity_history_summary.py`
- MZR calibration helpers: `tests/test_mzr_constraints.py`

## Editing Rules

- Check `git status --short` before modifying files. This repository may have
  large moves, generated assets, or conflict markers in progress; do not revert
  unrelated user changes.
- Keep changes narrowly scoped to the requested behavior.
- Do not add temporary Python scripts at the repository root. Root-level `*.py`
  files are ignored here; place scripts under `scripts/run/`, `scripts/plot/`,
  `scripts/analysis/`, or `scripts/experiments/` according to their purpose.
- Update README/API documentation when changing public functions, returned
  fields, accepted modes, units, or output paths.
- Prefer structured data loading and explicit validation over ad hoc parsing.
- Add or update focused tests for behavioral changes. If tests cannot be run,
  state the exact reason and the command that should be run later.
