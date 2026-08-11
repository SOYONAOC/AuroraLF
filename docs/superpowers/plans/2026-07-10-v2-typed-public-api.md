# AuroraLF v2 Typed Public API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, typed, immutable v2 configuration/result boundary and an in-memory `run_uvlf` API while preserving existing low-level scientific APIs.

**Architecture:** `auroralf.config` owns validated frozen dataclasses and strict TOML decoding, `auroralf.results` owns immutable typed result objects, and `auroralf.api` adapts those types to the existing HMF sampler and dust transform. The package root exports only the three v2 entry points; low-level modules remain available through their existing module paths.

**Tech Stack:** Python 3 dataclasses, pathlib, stdlib `tomllib`, NumPy, pytest, existing AuroraLF MAH/SFR/chemistry/SSP/UVLF modules.

---

### Task 1: Strict configuration dataclasses and TOML schema

**Files:**
- Create: `auroralf/config.py`
- Create: `tests/test_v2_config.py`

- [ ] **Step 1: Write failing construction tests**

  Cover frozen dataclasses, exact numeric/string/bool types, finite and physical ranges, unit-bearing names, model conversion, supported validators, required backend caches, metallicity-source nesting, IMF cross-gates, fixed Pop III upper mass, sampling edges/ranges, absolute `.h5` output, run id/schema/redshift/base-seed rules, and exact nested types.

- [ ] **Step 2: Run config tests and verify RED**

  Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_v2_config.py -q`; expect import failure because `auroralf.config` does not exist.

- [ ] **Step 3: Implement frozen schemas and explicit validators**

  Define `CONFIG_SCHEMA_VERSION = "2.0.0"`, `CosmologyConfig`, `MAHConfig`, `MZRConfig`, `RegulatorConfig`, `StarFormationConfig`, `StellarPopulationConfig`, `SamplingConfig`, `OutputConfig`, and `UVLFRunConfig`. Use shared strict helpers that reject booleans, non-real numerics, non-finite values, implicit string conversion, unknown modes, and invalid optional-field combinations. Convert only through explicit `to_model()` methods.

- [ ] **Step 4: Add strict TOML tests and parser**

  Test required/unknown root, nested, and inactive-table keys; precise missing `base_seed`; all other required scalars; optional omissions; and resolution of all paths relative to the TOML parent. Parse with `tomllib`, recursively compare exact key sets, then construct typed nested dataclasses without `.get()`.

- [ ] **Step 5: Run config tests and verify GREEN**

  Re-run the focused config command and require zero warnings or failures.

### Task 2: Immutable typed result objects

**Files:**
- Create: `auroralf/results.py`
- Create: `tests/test_v2_results.py`

- [ ] **Step 1: Write failing result tests**

  Cover exact frozen types, defensive array copies with `writeable=False`, halo-track shape/time/physics/clipping invariants, optional metallicity fields, IMF bin/count/sigma invariants, exact track types, unique mode/redshift diagnostics, exact lookup methods, and top-level config/result matching.

- [ ] **Step 2: Run result tests and verify RED**

  Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_v2_results.py -q`; expect import failure because `auroralf.results` does not exist.

- [ ] **Step 3: Implement immutable result dataclasses**

  Define `HaloTrackResult`, `IMFModeResult`, `RedshiftResult`, `ModeRunDiagnostics`, `RunDiagnostics`, and `UVLFRunResult`. Every array field is validated from a private copy and marked non-writeable before assignment; sigma arrays permit NaN but reject infinity and negative finite values. Lookup methods perform exact equality and raise `KeyError`.

- [ ] **Step 4: Run result tests and verify GREEN**

  Re-run the focused result command and require zero warnings or failures.

### Task 3: In-memory v2 API adapter and root exports

**Files:**
- Create: `auroralf/api.py`
- Modify: `auroralf/__init__.py`
- Create: `tests/test_v2_api.py`

- [ ] **Step 1: Write failing API tests**

  Add a sampler spy that proves one call per redshift/mode, identical `base_seed`, exact parameter mapping, explicit bins, disabled progress output, typed UVLF/diagnostic mapping, no dust copy behavior, dust fractional-sigma scaling, and no output artifact creation. Test exact missing SSP/backend-cache paths and exact root `__all__`.

- [ ] **Step 2: Run API tests and verify RED**

  Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_v2_api.py -q`; expect import failure because `auroralf.api` does not exist.

- [ ] **Step 3: Implement `run_uvlf(config)`**

  Require exact `UVLFRunConfig`; verify all active input paths exist before sampling; build current `Cosmology`, SFR, Pop III, IMF transition, MZR/regulator parameter types; call `sample_uvlf_from_hmf` for every configured redshift/mode with `progress_path=None` and `print_progress=False`; map required `uvlf[...]` and `metadata[...]` keys into typed results; apply current dust transform and fractional-sigma rule only when requested; leave `halo_tracks=()` and never write the output path.

- [ ] **Step 4: Add minimal real canonical smoke test if practical**

  Use one halo-mass sample, one track, one redshift, one worker, the real canonical SSP, and no dust. If the real environment/data prevents execution, retain the exact failure evidence; the spy test remains mandatory.

- [ ] **Step 5: Run API tests and verify GREEN**

  Re-run the focused API command and require zero warnings or failures.

### Task 4: Documentation and structural checks

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the minimal v2 example**

  Document only `from auroralf import UVLFRunConfig, UVLFRunResult, run_uvlf`, `UVLFRunConfig.from_toml(...)`, and that `run_uvlf` returns an in-memory result without writing `OutputConfig.artifact_path`. Do not document a nonexistent CLI.

- [ ] **Step 2: Add AST and public-surface tests**

  Assert `auroralf/config.py`, `auroralf/results.py`, and `auroralf/api.py` contain no `.get(...)` calls and package-root `__all__` is exactly the three v2 names.

- [ ] **Step 3: Run all v2 tests**

  Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_v2_config.py tests/test_v2_results.py tests/test_v2_api.py -q`.

### Task 5: Full verification

**Files:**
- Verify all modified files; make no commit or staging operation per task instruction.

- [ ] **Step 1: Run full test suite**

  Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests -q` and require zero failures/warnings.

- [ ] **Step 2: Compile and inspect diffs**

  Run `PYTHONPATH=. .venv/bin/python -m compileall -q auroralf tests`, `git diff --check`, inspect the scoped diff, confirm the active branch, and confirm no generated output artifact exists.

- [ ] **Step 3: Report evidence and remaining risks**

  Report created/modified files, the observed RED failures and GREEN counts, whether the real smoke run executed, exact verification commands, and any remaining integration risk. Do not stage or commit.
