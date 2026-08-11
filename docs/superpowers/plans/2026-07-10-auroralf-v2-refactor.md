# AuroraLF v2 Refactor Implementation Plan

> Execute this plan task by task with the normal project workflow and track progress with checkbox (`- [ ]`) syntax.

**Goal:** Replace the current loosely coupled UVLF production path with a scientifically explicit, deterministic, typed, resumable v2 pipeline whose only production artifact is a provenance-complete HDF5 file.

**Architecture:** Separate physical history construction, source-time stellar-population evaluation, HMF aggregation, and artifact persistence. A frozen `UVLFRunConfig` is the single source of physical and runtime settings; `run_uvlf(config)` returns typed results and the CLI serializes them through an atomic HDF5 artifact writer. Shared halo histories and metallicity tracks are computed once per redshift/mass batch and reused across IMF modes, while streaming accumulators avoid retaining all samples by default.

**Tech Stack:** Python 3.12, NumPy, SciPy, Astropy, h5py, hmf, pytest, TOML (`tomllib`), SLURM, uv.

---

## File map

- `auroralf/config.py`: frozen v2 configuration dataclasses and TOML loading.
- `auroralf/results.py`: typed track, mode, redshift, diagnostic, and run results.
- `auroralf/mah/generator.py`: raw and positive-only halo accretion-rate columns.
- `auroralf/mah/sampling.py`: strict MAH backend/model validation.
- `auroralf/sfr/calculator.py`: consume only the effective non-negative accretion rate.
- `auroralf/sfr/popiii.py`: Visbal raw scaling plus explicitly gated SFR.
- `auroralf/ssp/convolution.py`: common final-time SSP convolution engine.
- `auroralf/ssp/uv1600.py`, `auroralf/ssp/heii1640.py`: table loading and thin observable wrappers.
- `auroralf/uvlf/dust.py`: branch-correct dust Jacobian.
- `auroralf/uvlf/pipeline.py`: shared cosmology, burst normalization, deterministic component RNGs, typed pipeline results.
- `auroralf/uvlf/streaming.py`: online weighted histograms and uncertainty accumulators.
- `auroralf/uvlf/runner.py`: v2 redshift-batch orchestration and public `run_uvlf` API.
- `auroralf/io/schema.py`, `auroralf/io/hdf5.py`: schema constants, validation, atomic shard/final artifact I/O.
- `auroralf/mah/tng.py`, `auroralf/mah/thesan.py`: strict cache schema and provenance validation.
- `scripts/run/run_uvlf_v2.py`: TOML-only production entry point.
- `scripts/submit/submit_uvlf_v2.py`: SLURM submission and closure validation.
- `scripts/data/convert_uvlf_npz_to_v2_hdf5.py`: explicit one-way migration utility, never a production reader fallback.
- `tests/test_*.py`: focused unit, integration, schema, resume, and CLI regression tests.
- `pyproject.toml`, `uv.lock`: reproducible environment definition.
- `.gitignore`, `external_data/source_manifests/meraxes.toml`: source-tree policy and exact external-source provenance.
- `slides/auroralf_v2_validation_20260710/`: review deck and vector validation assets.

## Task 1: Correct the dust mapping Jacobian

**Files:**
- Modify: `auroralf/uvlf/dust.py`
- Create: `tests/test_dust.py`

- [x] **Step 1: Write failing tests for active attenuation, custom `c0`, and the attenuation floor**

```python
def test_intrinsic_muv_jacobian_matches_finite_difference():
    z, magnitude, c0, c1, step = 6.0, -22.0, 4.7, 10.0, 1.0e-5
    numerical = (
        intrinsic_muv_from_observed(magnitude + step, z, c0=c0, c1=c1)
        - intrinsic_muv_from_observed(magnitude - step, z, c0=c0, c1=c1)
    ) / (2.0 * step)
    assert intrinsic_muv_jacobian(magnitude, z, c0=c0, c1=c1) == pytest.approx(numerical, rel=1e-8)

def test_intrinsic_muv_jacobian_is_unity_when_attenuation_is_floored():
    assert intrinsic_muv_jacobian(-14.0, 6.0, c0=4.7) == pytest.approx(1.0)
```

- [x] **Step 2: Run the focused test and observe the active-branch and floor failures**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_dust.py -q`

- [x] **Step 3: Compute the derivative from the same attenuation branch used by the forward transform**

```python
attenuation_raw = c1 + c0 * uv_continuum_slope_beta(magnitude, redshift, m0=m0)
active = attenuation_raw > 0.0
active_jacobian = 1.0 - c0 * (-0.007 * redshift - 0.09)
return np.where(active, active_jacobian, 1.0)
```

- [x] **Step 4: Run `tests/test_dust.py` and `tests/test_hmf_sampling.py` and require all tests to pass**

## Task 2: Make source-time gating and accretion-rate semantics explicit

**Files:**
- Modify: `auroralf/mah/generator.py`
- Modify: `auroralf/mah/tng.py`
- Modify: `auroralf/mah/thesan.py`
- Modify: `auroralf/mah/sampling.py`
- Modify: `auroralf/sfr/calculator.py`
- Modify: `auroralf/sfr/popiii.py`
- Modify: `auroralf/uvlf/pipeline.py`
- Modify: `auroralf/results.py`
- Modify: `tests/test_popiii_model.py`
- Create: `tests/test_accretion_rate_semantics.py`

- [x] **Step 1: Add failing tests that require both raw/effective rates, clipping diagnostics, and strict sampling model names**

```python
def test_negative_raw_accretion_is_preserved_but_sfr_rate_is_clipped():
    tracks = generate_halo_histories(...)
    assert np.any(tracks["dMh_dt_raw"] < 0.0)
    assert np.all(tracks["dMh_dt_sfr"] >= 0.0)
    np.testing.assert_array_equal(
        tracks["dMh_dt_clipped"],
        tracks["dMh_dt_raw"] < 0.0,
    )

def test_unknown_mah_parameter_model_fails():
    with pytest.raises(ValueError, match="mcbride"):
        sample_parameters(1.0e9, 4, np.random.default_rng(1), model="mcbride_typo")
```

- [x] **Step 2: Add a failing Visbal test requiring zero physical SFR outside `1 <= Mh/Mcool <= 2` while retaining the raw scaling**

```python
result = compute_popiii_sfr_visbal2015_from_grids(...)
np.testing.assert_array_equal(result.atomic_window, [False, True, True, False])
assert np.all(result.sfr_msun_per_yr[[0, 3]] == 0.0)
assert np.all(result.raw_sfr_scaling_msun_per_yr > 0.0)
```

- [x] **Step 3: Introduce track columns `dMh_dt_raw`, `dMh_dt_sfr`, and `dMh_dt_clipped` (rates in `Msun/Gyr`); the later typed result wraps them with unit-explicit attributes**

```python
dmh_dt_raw = np.asarray(dmh_dt, dtype=float)
dmh_dt_sfr = np.maximum(dmh_dt_raw, 0.0)
accretion_rate_clipped = dmh_dt_raw < 0.0
```

- [x] **Step 4: Make every SFR path consume only `dMh_dt_sfr` and raise if its final SFR contains a negative finite value**

- [x] **Step 5: Validate MAH parameter models with an explicit dispatch table containing only `mcbride` and `gaussian_approximation`**

- [x] **Step 6: Run the focused MAH, SFR, Pop III, TNG, and THESAN test files and require all to pass**

## Task 3: Unify cosmology, J_LW, burst normalization, and component RNGs

**Files:**
- Modify: `auroralf/config.py`
- Modify: `auroralf/sfr/calculator.py`
- Modify: `auroralf/sfr/popiii.py`
- Modify: `auroralf/uvlf/hmf_sampling.py`
- Modify: `auroralf/uvlf/pipeline.py`
- Modify: `tests/test_burst_scatter.py`
- Create: `tests/test_pipeline_context.py`
- Create: `tests/test_deterministic_rng.py`

- [x] **Step 1: Write a failing burst test whose positive SFR islands are separated by internal zeros and integrate over the full time grid**

```python
original_mass = np.trapezoid(sfr, time_gyr * 1.0e9)
scattered = _apply_burst_scatter_to_sfr_grid(...)
new_mass = np.trapezoid(scattered, time_gyr * 1.0e9)
assert new_mass == pytest.approx(original_mass, rel=1e-12)
assert np.all(scattered[sfr == 0.0] == 0.0)
```

- [x] **Step 2: Write failing tests showing a custom cosmology reaches the SFR delay calculation and one `lw_background_j21` reaches both cooling/HMF labels and Pop III SFR**

- [x] **Step 3: Normalize burst scatter on the full monotonic grid with masked zero samples left in place**

```python
formed_before = np.trapezoid(sfr_grid, time_grid_gyr * 1.0e9, axis=-1)
candidate = sfr_grid * multiplier
formed_after = np.trapezoid(candidate, time_grid_gyr * 1.0e9, axis=-1)
scale = np.divide(formed_before, formed_after, out=np.ones_like(formed_before), where=formed_after > 0)
candidate *= scale[..., None]
candidate[sfr_grid == 0.0] = 0.0
```

- [x] **Step 4: Pass one explicit cosmology object through MAH, SFR delay, regulator, and luminosity paths; remove internal default construction**

- [x] **Step 5: Put J_LW only in the run configuration and derive downstream Pop III/cooling settings from it**

- [x] **Step 6: Require a production base seed and derive stable component seeds independent of IMF order**

```python
def component_rng(base_seed: int, redshift_index: int, mass_index: int, component: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, redshift_index, mass_index, component]))
```

- [x] **Step 7: Run focused tests twice with reversed IMF mode order and require bitwise-identical paired histories and metallicity draws**

## Task 4: Consolidate SSP convolution and close the LW horizon

**Files:**
- Modify: `auroralf/ssp/convolution.py`
- Modify: `auroralf/ssp/uv1600.py`
- Modify: `auroralf/ssp/heii1640.py`
- Modify: `scripts/analysis/plot_sfrd_lw_background_from_model.py`
- Modify: `tests/test_popiii_heii1640.py`
- Modify: `tests/test_sfrd_lw_background_plot.py`
- Create: `tests/test_ssp_convolution.py`

- [x] **Step 1: Add parameterized failing tests that feed the same delta-burst and constant-SFR histories to UV1600, HeII1640, and Q2 kernels**

- [x] **Step 2: Implement one observable-neutral final-time convolution**

```python
def convolve_final_ssp_observable(
    time_gyr: np.ndarray,
    sfr_msun_per_yr: np.ndarray,
    ssp_age_myr: np.ndarray,
    observable_per_msun: np.ndarray,
) -> float:
    ...
```

- [x] **Step 3: Keep UV, HeII, and Q2 modules responsible only for loading/validating tables and calling the common engine**

- [x] **Step 4: Extend the internal SFRD redshift grid through the complete Lyman-Werner light-cone horizon before evaluating requested output redshifts**

- [x] **Step 5: Raise a precise error when the supplied model cannot evaluate the required horizon instead of returning a zero endpoint**

- [x] **Step 6: Run SSP, Pop III HeII, and LW-background focused tests and require all to pass**

## Task 5: Define the v2 typed API and strict configuration

**Files:**
- Create: `auroralf/config.py`
- Create: `auroralf/results.py`
- Modify: `auroralf/__init__.py`
- Modify: `auroralf/uvlf/pipeline.py`
- Create: `tests/test_v2_config.py`
- Create: `tests/test_v2_results.py`

- [x] **Step 1: Write failing construction/validation tests for valid config, missing seed, invalid units/ranges, incompatible metallicity gates, and unknown TOML keys**

- [x] **Step 2: Implement frozen nested dataclasses with explicit physical units in every dimensional field name**

```python
@dataclass(frozen=True)
class UVLFRunConfig:
    schema_version: str
    run_id: str
    redshifts: tuple[float, ...]
    base_seed: int
    cosmology: CosmologyConfig
    mah: MAHConfig
    star_formation: StarFormationConfig
    stellar_population: StellarPopulationConfig
    sampling: SamplingConfig
    output: OutputConfig
```

- [x] **Step 3: Implement typed `HaloTrackResult`, `IMFModeResult`, `RedshiftResult`, `RunDiagnostics`, and `UVLFRunResult` without `.get()`-style optional schema access**

- [x] **Step 4: Implement strict `UVLFRunConfig.from_toml(path)` that rejects missing and unknown keys and resolves relative data paths against the TOML file**

- [x] **Step 5: Export only the v2 public API `run_uvlf(config: UVLFRunConfig) -> UVLFRunResult`; migrate internal callers rather than retaining compatibility shims**

- [x] **Step 6: Run config/result tests plus the full existing suite**

## Task 6: Build streaming shared-batch UVLF execution

**Files:**
- Create: `auroralf/uvlf/streaming.py`
- Create: `auroralf/uvlf/runner.py`
- Modify: `auroralf/uvlf/hmf_sampling.py`
- Modify: `auroralf/uvlf/pipeline.py`
- Create: `tests/test_streaming_uvlf.py`
- Create: `tests/test_shared_batch_runner.py`

- [x] **Step 1: Write failing tests comparing streaming weighted histograms and online variance with the exact in-memory reference calculation**

- [x] **Step 2: Implement a bounded-memory accumulator that stores bin sums, squared contributions, counts, and rejection diagnostics only**

```python
@dataclass
class WeightedHistogramAccumulator:
    edges: np.ndarray
    weighted_sum: np.ndarray
    weighted_square_sum: np.ndarray
    sample_count: np.ndarray

    def update(self, values: np.ndarray, weights: np.ndarray) -> None:
        ...
```

- [x] **Step 3: Write a failing spy-based test proving each redshift/mass halo batch, simulation cache, and canonical SFR track is constructed once regardless of IMF mode count**

- [x] **Step 4: Implement the batch order `redshift -> bounded mass chunks -> shared histories/SFR/metallicity -> IMF observables -> online histograms`**

- [x] **Step 5: Initialize SSP tables and simulation caches once per worker and bound outstanding futures to at most `2 * worker_count`**

- [x] **Step 6: Keep per-halo samples disabled by default; when enabled, stream them directly to HDF5 shard datasets**

- [x] **Step 7: Run numerical-equivalence tests and record peak resident memory for a reduced deterministic benchmark**

## Task 7: Implement the HDF5 artifact and atomic resume protocol

**Files:**
- Create: `auroralf/io/__init__.py`
- Create: `auroralf/io/schema.py`
- Create: `auroralf/io/hdf5.py`
- Create: `scripts/data/convert_uvlf_npz_to_v2_hdf5.py`
- Create: `tests/test_hdf5_artifact.py`
- Create: `tests/test_hdf5_resume.py`
- Modify: all production/analysis consumers returned by `rg -l '\.npz|np\.load|np\.savez' scripts auroralf tests`

- [x] **Step 1: Write failing schema round-trip tests for `/config`, `/provenance`, `/axes`, `/results/<z>/<mode>`, `/diagnostics`, and optional `/samples`**

- [x] **Step 2: Define `SCHEMA_NAME = "auroralf.uvlf"` and `SCHEMA_VERSION = "2.0.0"`; validate required groups, datasets, dtypes, shapes, units, and finite-value rules on read and write**

- [x] **Step 3: Write shards to a same-filesystem temporary path, flush/fsync, validate, and rename atomically; create a completion marker only after rename**

- [x] **Step 4: Resume only shards whose config hash, code revision, schema version, seed namespace, and source-cache checksums exactly match the current run**

- [x] **Step 5: Merge validated shards into `data_save/<run_id>.h5` atomically and refuse conflicting duplicate redshift/mode data**

- [x] **Step 6: Implement a one-way explicit NPZ conversion command; remove NPZ from all v2 production readers**

- [x] **Step 7: Migrate repository analysis and plotting consumers to strict HDF5 reads and test representative scripts**

## Task 8: Enforce simulation-cache and external-source provenance

**Files:**
- Modify: `auroralf/mah/tng.py`
- Modify: `auroralf/mah/thesan.py`
- Modify: `scripts/data/` cache builders and smoke validators selected by `rg -l 'tng|thesan' scripts/data`
- Modify: `.gitignore`
- Create: `external_data/source_manifests/meraxes.toml`
- Modify: `tests/test_tng_mah_backend.py`
- Modify: `tests/test_thesan_mah_backend.py`
- Modify: `tests/test_tng_selection_script.py`
- Modify: `tests/test_thesan_download_plan.py`

- [x] **Step 1: Add failing tests that reject caches missing real source identifiers, simulation name, snapshot, units, selection description, creator version, and source checksums**

- [x] **Step 2: Remove fabricated identifiers and `unknown` metadata fallbacks from TNG/THESAN cache readers**

- [x] **Step 3: Validate THESAN tree structure and required datasets in smoke checks; validate downloaded file hashes or authoritative sizes rather than non-emptiness alone**

- [x] **Step 4: Add `third_party/` and `runs/` to `.gitignore`; record Meraxes URL, exact commit, license, retrieval date, build flags, and local patch hashes in the tracked manifest**

- [x] **Step 5: Run every TNG/THESAN selection, cache, merger-event, and download-plan test**

## Task 9: Replace the production CLI and close the SLURM workflow

**Files:**
- Create: `scripts/run/run_uvlf_v2.py`
- Create: `scripts/submit/submit_uvlf_v2.py`
- Create: `configs/uvlf/production.toml`
- Modify: `scripts/run/run_uvlf_compare_imf_no_delay_all_z.py`
- Modify: `scripts/submit/submit_uvlf_imf_compare.py`
- Create: `tests/test_uvlf_v2_cli.py`
- Create: `tests/test_uvlf_v2_submit.py`

- [x] **Step 1: Write failing CLI tests for config validation, SLURM requirement, dry-run rendering, explicit overwrite/resume policy, and nonzero exit on incomplete artifacts**

- [x] **Step 2: Implement `run_uvlf_v2.py --config <toml> [--resume | --overwrite]` with mutually exclusive output policies**

- [x] **Step 3: Implement the submit wrapper so requested CPU/memory/time, config checksum, command, job ID, logs, exit code, and final artifact validation are recorded in provenance**

- [x] **Step 4: Disable old production entry points with a precise migration message; do not leave parallel v1 execution paths**

- [x] **Step 5: Run CLI integration tests and a reduced local artifact run; use SLURM only for production-sized validation**

## Task 10: Lock dependencies and finish scientific validation

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `slides/auroralf_v2_validation_20260710/auroralf_v2_validation_20260710.tex`
- Create: vector figures under `slides/auroralf_v2_validation_20260710/assets/`
- Modify: public API and run documentation selected by `rg -l 'run_uvlf|UVLF|NPZ|npz' README.md auroralf docs scripts`

- [x] **Step 1: Inventory actual imports and installed versions from `.venv`; declare runtime, analysis, test, and slide-build dependency groups without silently changing the active interpreter**

- [x] **Step 2: Generate `uv.lock`, create a clean temporary environment from it, and run `PYTHONPATH=. .venv/bin/python -m pytest tests` in the project environment**

- [x] **Step 3: Run `git diff --check`, strict HDF5 validation, deterministic rerun comparison, reduced performance benchmark, and all focused scientific regressions**

- [x] **Step 4: Produce a Beamer validation deck showing dust finite differences, Visbal gating, negative-rate clipping fractions, burst mass conservation, paired-seed equality, LW horizon closure, v1/v2 UVLF comparison, memory scaling, and artifact provenance**

- [x] **Step 5: Compile the deck, inspect every page visually, and keep review-facing conclusions and remaining scientific decisions on the slides**

- [x] **Step 6: Perform final spec and code-quality reviews, resolve every Critical/Important issue, and repeat the full verification suite before reporting completion**

Completion evidence (2026-07-11): a brand-new environment synchronized from
the frozen lock and the existing project `.venv` both passed all 1057 tests;
the validation metrics were byte-identical across reruns; strict HDF5/source
readback and `git diff --check` passed; the 17-page deck compiled twice without
warnings and every page was inspected visually. The final review was performed
locally, without invoking or depending on any superpowers skill.

## Execution constraints

- Work on `codex/auroralf-v2`; preserve the dirty working tree that existed when the branch was created.
- Use `PYTHONPATH=. .venv/bin/python` for every Python command.
- Do not create synthetic simulation or observational data and do not introduce fallback paths.
- Do not run production UVLF jobs on a login node; submit them through the v2 SLURM wrapper.
- Do not stage or commit pre-existing user work implicitly. Each implementation task records its exact changed paths; commits remain deferred until ownership can be separated safely.
- A task is complete only after its RED test was observed, its GREEN verification passed, spec review passed, and code-quality review has no unresolved Critical or Important issue.
