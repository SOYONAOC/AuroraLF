# Task 6C IPC And File-Version Cache Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve strict immutable dataclass invariants across multiprocessing IPC and invalidate all scientific file caches whenever the underlying file version changes.

**Architecture:** Every IPC result is reconstructed through normal strict dataclass constructors before scheduling, observation, or histogram consumption. A shared frozen `FileVersion` value captures resolved-path stat identity and becomes the file-bearing part of all MAH and SSP cache keys; loaders stat once per public call and open `version.path` only on misses.

**Tech Stack:** Python dataclasses, NumPy, multiprocessing `ForkingPickler`, protocol 4/5 pickle, `functools.lru_cache`, h5py, pytest.

---

### Task 1: Strict IPC reconstruction

**Files:**
- Modify: `auroralf/uvlf/runner.py`
- Modify: `auroralf/uvlf/pipeline.py`
- Test: `tests/test_shared_batch_runner.py`

- [ ] Add protocol 4/5 and `ForkingPickler` failing round-trip tests for `_MassModeTaskResult`, `_MassTaskResult`, `LoadedSSPKernels`, and `_WorkerContext`; assert every array and its base cannot become writeable.
- [ ] Add failing scheduler tests that inject a mutated exact-type result and require a fresh strict normalized result before observer or histogram consumption.
- [ ] Add top-level reconstruction functions and `__reduce_ex__` methods that call normal constructors, plus a `_normalize_mass_task_result` deep reconstruction boundary used by serial and parallel execution.
- [ ] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_shared_batch_runner.py -q -x` and require all tests to pass.

### Task 2: Strict version value

**Files:**
- Create: `auroralf/file_version.py`
- Create: `tests/test_file_version.py`

- [ ] Add failing tests for exact field types, absolute resolved paths, regular-file enforcement, and version changes after atomic replacement even with restored size and mtime.
- [ ] Implement frozen `FileVersion(path, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` and `FileVersion.from_path` using one resolved-file stat.
- [ ] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_file_version.py -q -x` and require all tests to pass.

### Task 3: TNG and THESAN cache versioning

**Files:**
- Modify: `auroralf/mah/tng.py`
- Modify: `auroralf/mah/thesan.py`
- Modify: `tests/test_tng_mah_backend.py`
- Modify: `tests/test_thesan_mah_backend.py`

- [ ] Add failing same-path in-place and atomic-replacement tests for both backends; assert unchanged calls do not reread datasets and failed loads are not cached.
- [ ] Replace path keys with `(FileVersion, exact_z)` and open `version.path`; keep preload return values as resolved `Path`.
- [ ] Run both backend test files with `-W error` and require all tests to pass.

### Task 4: UV1600 and Pop III UV cache versioning

**Files:**
- Modify: `auroralf/ssp/uv1600.py`
- Modify: `tests/test_ssp_convolution.py`
- Modify: `tests/test_popiii_model.py`

- [ ] Add failing NPZ canonical and Pop III same-path replacement tests plus unchanged-read-count and failed-then-repair coverage.
- [ ] Change cached-loader keys to `FileVersion` plus wavelength, metallicity, or UV column and open only `version.path`.
- [ ] Run the UV SSP focused tests with `-W error` and require all tests to pass.

### Task 5: HeII cache versioning

**Files:**
- Modify: `auroralf/ssp/heii1640.py`
- Modify: `tests/test_popiii_heii1640.py`

- [ ] Add failing same-path replacement tests for both HeII luminosity and He+ loaders and unchanged-read-count coverage.
- [ ] Change both cached-loader keys to `FileVersion` and read only `version.path`.
- [ ] Run `tests/test_popiii_heii1640.py` with `-W error` and require all tests to pass.

### Task 6: Integrated verification

**Files:**
- Verify: `README.md`

- [ ] Run real spawn, cache, sequential-config, and public API focused suites with `-W error`.
- [ ] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests -q`.
- [ ] Run `PYTHONPATH=. .venv/bin/python -m compileall -q auroralf tests` and `git diff --check`.
- [ ] Confirm README does not claim content checksums and leave all changes unstaged.
