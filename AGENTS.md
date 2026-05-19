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
- Preserve units and conversions:
  - halo mass: `Msun`
  - SFR: `Msun/yr`
  - UV luminosity: `erg/s/Hz`
  - HMF: `Mpc^-3 Msun^-1`
  - SSP tables commonly provide ages in `Myr`; convert to `Gyr` before
    convolving with MAH/SFR histories stored in `Gyr`.
- The current production HMF path is `hmf` Reed07. Deprecated model names such
  as `massfunc_st` and `hmf_watson13_fof` intentionally raise errors; do not
  restore them unless the user explicitly asks for a historical comparison.
- Pop II top-heavy IMF variants are source-time gated. Do not replace the SSP
  globally when the intended behavior is `z10_mild_topheavy` or
  `mah_burst_mild_topheavy`.
- UV convolution should use valid active history segments only. Do not adjust
  scientific parameters, mass cuts, redshift cuts, or units just to make a test
  or run finish.
- The dust-corrected UVLF currently applies the physical clipping
  `phi_obs = min(phi_obs_raw, phi_nodust_obs)`.

## Production Runs

Large UVLF comparisons are compute jobs. Submit them through the SLURM wrapper
instead of running the production target directly on a login node:

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_imf_compare.py --dry-run -- --canonical-only
```

The target script `scripts/run/run_uvlf_compare_imf_no_delay_all_z.py` requires
a SLURM allocation. The older
`scripts/run/run_uvlf_mass_function_compare_full.py` entry point is intentionally
disabled and points users to the Reed07 HMF workflow.

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
