# Streaming UVLF Sample Sink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit non-root API that streams exact per-halo samples into strict atomic HDF5 shards without retaining the full sample population in memory or changing the default UVLF runner payload.

**Architecture:** `auroralf.uvlf.runner` conditionally exposes one immutable `HaloSampleTable` per redshift, halo-mass index, and IMF mode only when a private observer is supplied. `auroralf.io.sample_sink` appends each table into a same-directory owned HDF5 spool, then publishes one strict shard per configured redshift and mode through the existing marker/flock/rollback transaction. `auroralf.io.hdf5` performs bounded spool-to-shard dataset copies and validates every published shard through the public lazy reader.

**Tech Stack:** Python frozen dataclasses, NumPy, h5py, multiprocessing spawn, UUID same-directory temporary files, pytest.

---

### Task 1: Conditional runner sample observer

**Files:**
- Modify: `auroralf/uvlf/runner.py`
- Create: `tests/test_hdf5_sample_sink.py`

- [x] Add a failing default-path test that captures `_MassTaskResult` through `_mass_result_observer` and asserts both optional per-track SFR fields are `None`.
- [x] Add a failing enabled-path test that calls `run_uvlf_streaming(config, _halo_sample_observer=observer)` and checks the exact callback order `(redshift, mass_index, imf_mode)`, mass-major/track-major indices, per-track weight `mass_weight_per_mpc3 / n_tracks`, luminosity, MUV, final Pop II SFR, and final Pop III SFR.
- [x] Add failing tests requiring a non-callable observer to fail before science work and an observer exception to propagate unchanged.
- [x] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_hdf5_sample_sink.py -q -x` and confirm the first failure is the missing per-track sample payload.
- [x] Add `include_samples: bool` to `_WorkerContext`, optional immutable final-SFR vectors to `_MassTaskResult`, and preserve those vectors only when `include_samples=True`.
- [x] Build one `HaloSampleTable` immediately before each callback and do not add samples to `UVLFRunResult` or `IMFModeResult.halo_tracks`.
- [x] Re-run the focused tests and `tests/test_shared_batch_runner.py`; require all tests to pass.

### Task 2: Owned append-only sample spool

**Files:**
- Create: `auroralf/io/sample_sink.py`
- Modify: `auroralf/io/__init__.py`
- Test: `tests/test_hdf5_sample_sink.py`

- [x] Add failing tests constructing the sink in a shard directory and checking a UUID spool exists in that directory with mode `0600`.
- [x] Add failing append tests that require one extensible, chunked, compressed dataset per `HaloSampleTable` field under `samples/z_<redshift>/<mode>`, exact config/provenance metadata, and strict configured mass/mode/redshift order.
- [x] Add failing tamper and lifecycle tests for a missing spool, mutated metadata, changed source checksums, closed-sink append rejection, and idempotent abort that removes only the owned spool.
- [x] Run the new spool subset and confirm it fails because the sink class does not exist.
- [x] Implement a private `_HDF5SampleSink` that creates its spool with `os.open(..., O_CREAT | O_EXCL, 0o600)`, verifies `ArtifactProvenance.config_sha256`, calls `provenance.verify_sources()`, validates every incoming exact `HaloSampleTable`, and appends via `dataset[old_size:new_size]` without storing tables.
- [x] Re-run the spool tests and require all tests to pass.

### Task 3: Bounded spool-to-shard atomic writer

**Files:**
- Modify: `auroralf/io/hdf5.py`
- Modify: `auroralf/io/sample_sink.py`
- Test: `tests/test_hdf5_sample_sink.py`

- [x] Add failing spy tests that record every copy slice length and require append/copy lengths to stay at or below the configured chunk bound; implementation contains no full sample-dataset reads.
- [x] Add failing rollback tests that pre-create valid shards, inject a marker failure, and require the old shard and marker bytes to remain exact while owned spool/temp/backup files are removed.
- [x] Add failing overwrite and concurrent-writer tests requiring `overwrite=False` rejection and existing marker/flock serialization.
- [x] Implement `_write_uvlf_shard_from_spool_atomic(...)` using `_write_payload_atomic`, existing config/provenance/result/diagnostic writers, extensible sample datasets, and repeated bounded `dataset[start:stop]` copies reconstructed through `HaloSampleTable` validation.
- [x] After each write, call public `read_uvlf_shard(path, load_samples=False)` and compare config, provenance, result, diagnostic, and descriptor exactly; never construct a full sample table during finalize.
- [x] Re-run the bounded-copy and rollback tests and require all tests to pass.

### Task 4: Explicit non-root API

**Files:**
- Modify: `auroralf/io/sample_sink.py`
- Modify: `auroralf/io/__init__.py`
- Test: `tests/test_hdf5_sample_sink.py`

- [x] Add a failing import/API test for `auroralf.io.run_uvlf_to_sample_shards(config, provenance, shard_directory, overwrite=False) -> tuple[UVLFRunResult, tuple[Path, ...]]` and assert `auroralf.__all__` is unchanged.
- [x] Add failing serial and two-worker tests requiring the enabled result to be bitwise identical to disabled `run_uvlf_streaming`, with exact sample count/order/content in every lazy-readable shard.
- [x] Add failing cleanup tests for observer failure, spool tampering, source replacement, overwrite marker failure, existing shards, and concurrent locks.
- [x] Implement the API with exact type/path validation, an owned sink, `run_uvlf_streaming(..., _halo_sample_observer=sink.append)`, axis-ordered finalize, and cleanup on every unsuccessful path.
- [x] Export only from `auroralf.io`; do not modify `auroralf/__init__.py`, root `run_uvlf`, config, or production CLI.
- [x] Re-run the sample-sink tests and require all tests to pass.

### Task 5: Integrated verification

**Files:**
- Verify: `auroralf/uvlf/runner.py`
- Verify: `auroralf/io/sample_sink.py`
- Verify: `auroralf/io/hdf5.py`
- Verify: `auroralf/io/__init__.py`
- Verify: `tests/test_hdf5_sample_sink.py`

- [x] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_hdf5_sample_sink.py tests/test_shared_batch_runner.py tests/test_hdf5_artifact.py -q` after final spool-schema hardening (115 passed).
- [x] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests -q` after final spool-schema hardening (934 passed).
- [x] Run `PYTHONPATH=. .venv/bin/python -m compileall -q auroralf tests` and `git diff --check`.
- [x] Confirm no sample arrays are present in default `_MassTaskResult`, all shard reads used for finalize verification are lazy, no owned spool/temp/backup remains, root exports are unchanged, and all changes remain unstaged in the shared worktree.
