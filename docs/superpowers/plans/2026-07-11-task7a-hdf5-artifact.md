# Task 7A Strict HDF5 Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, versioned, atomically committed HDF5 representation for one complete typed UVLF run.

**Architecture:** `auroralf/io/schema.py` owns canonical config JSON, provenance, checksums, samples, and the aggregate artifact invariant. `auroralf/io/hdf5.py` owns the exact HDF tree, typed reconstruction, corruption rejection, artifact hashing, completion markers, fsync, and atomic replacement.

**Tech Stack:** Python frozen dataclasses, JSON/SHA-256, NumPy, h5py, pytest.

---

### Task 1: Strict schema values and canonical config

**Files:**
- Create: `auroralf/io/schema.py`
- Create: `auroralf/io/__init__.py`
- Modify: `auroralf/config.py`
- Test: `tests/test_hdf5_artifact.py`

- [ ] Write failing tests constructing strict `SourceChecksum`, `ArtifactProvenance`, `HaloSampleTable`, and `UVLFArtifact`, including writable-input mutation and duplicate-axis rejection.
- [ ] Write failing canonical-config tests requiring sorted compact JSON, explicit null optionals, absolute path strings, unknown/missing-key rejection, and exact `UVLFRunConfig` reconstruction.
- [ ] Implement `canonical_config_mapping`, `decode_canonical_config_mapping`, JSON/hash helpers, SHA-256 file checks, UTC provenance, and irreversible sample arrays.
- [ ] Run `PYTHONPATH=. .venv/bin/python -W error -m pytest tests/test_hdf5_artifact.py -q -x` and require the schema tests to pass.

### Task 2: Exact HDF5 tree and typed roundtrip

**Files:**
- Create: `auroralf/io/hdf5.py`
- Modify: `auroralf/io/__init__.py`
- Test: `tests/test_hdf5_artifact.py`

- [ ] Write a failing two-redshift/two-mode roundtrip test that checks exact config, provenance, result arrays, diagnostics, axes, dtypes, units, group names, and optional compressed samples.
- [ ] Write failing corruption tests for unknown/missing objects, attrs, dtype, shape, nonfinite values, axes, config hash, and source checksum mismatch.
- [ ] Implement exact-schema writers and readers that construct `UVLFRunResult`, `ArtifactProvenance`, and optional `HaloSampleTable` through their strict constructors.
- [ ] Run the HDF roundtrip and corruption subset and require all tests to pass.

### Task 3: Atomic artifact and completion marker

**Files:**
- Modify: `auroralf/io/hdf5.py`
- Test: `tests/test_hdf5_artifact.py`

- [ ] Write failing tests for missing/invalid markers, artifact and marker tampering, overwrite false/true, same-directory temp files, fsync/replace ordering, pre-rename cleanup, and post-rename marker failure.
- [ ] Implement exclusive same-directory temp creation, strict temp readback, file/directory fsync, artifact replace, and SHA-bound canonical marker replacement.
- [ ] Run `tests/test_hdf5_artifact.py`, then relevant v2 tests, then the full suite with `-W error`; run compileall and diff checks without staging.
