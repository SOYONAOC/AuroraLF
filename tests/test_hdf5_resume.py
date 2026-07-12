from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

import h5py
import numpy as np
import pytest

from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.io import (
    ArtifactProvenance,
    HaloSampleDescriptor,
    HaloSampleTable,
    SourceChecksum,
    UVLFShard,
    UVLFShardDescriptor,
    merge_uvlf_shards,
    read_uvlf_artifact,
    read_uvlf_shard,
    uvlf_shard_filename,
    validate_uvlf_resume_shards,
    write_uvlf_shard_atomic,
)
from auroralf.results import IMFModeResult, ModeRunDiagnostics


MODES = ("canonical", "mah_burst_mild_topheavy")
REDSHIFTS = (6.0, 8.0)


def _config(tmp_path: Path, *, base_seed: int = 123) -> UVLFRunConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    canonical = (tmp_path / "canonical.dat").resolve()
    topheavy = (tmp_path / "topheavy.hdf5").resolve()
    popiii = (tmp_path / "popiii.dat").resolve()
    canonical.write_bytes(b"canonical-source")
    topheavy.write_bytes(b"topheavy-source")
    popiii.write_bytes(b"popiii-source")
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="resume-test",
        redshifts=REDSHIFTS,
        base_seed=base_seed,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=8),
        star_formation=StarFormationConfig(metallicity_source="none"),
        stellar_population=StellarPopulationConfig(
            imf_modes=MODES,
            canonical_ssp_path=canonical,
            topheavy_ssp_path=topheavy,
            popiii_ssp_path=popiii,
            birth_metallicity_topheavy_max_zsun=None,
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=2,
            n_tracks_per_halo_mass=2,
            muv_bin_edges=(-24.0, -20.0, -16.0),
            workers=1,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / "merged.h5").resolve()),
    )


def _provenance(
    config: UVLFRunConfig,
    *,
    revision: str = "a" * 40,
    dirty: bool = False,
    namespace: str = "auroralf.pipeline.v1",
    created: str = "2026-07-11T12:00:00Z",
) -> ArtifactProvenance:
    return ArtifactProvenance.for_config(
        config,
        code_revision=revision,
        code_dirty=dirty,
        seed_namespace=namespace,
        source_paths=(
            ("canonical_ssp", config.stellar_population.canonical_ssp_path),
            ("topheavy_ssp", config.stellar_population.topheavy_ssp_path),
        ),
        created_utc=created,
    )


def _mode(mode: str, scale: float = 1.0) -> IMFModeResult:
    return IMFModeResult(
        imf_mode=mode,
        bin_edges_muv=np.array([-24.0, -20.0, -16.0]),
        bin_centers_muv=np.array([-22.0, -18.0]),
        bin_width_mag=np.array([4.0, 4.0]),
        raw_counts=np.array([2, 1], dtype=np.int64),
        weighted_counts_per_mpc3=np.array([2.0e-5, 1.0e-4]) * scale,
        weight_squared_counts_per_mpc6=np.array([2.0e-10, 1.0e-8]) * scale,
        weighted_count_sigma_per_mpc3=np.array([1.4e-5, 1.0e-4]) * scale,
        effective_counts=np.array([2.0, 1.0]) * scale,
        phi_intrinsic_per_mpc3_per_mag=np.array([5.0e-6, 2.5e-5]) * scale,
        phi_intrinsic_sigma_per_mpc3_per_mag=np.array([3.5e-6, 2.5e-5]) * scale,
        phi_observed_per_mpc3_per_mag=np.array([4.0e-6, 2.0e-5]) * scale,
        phi_observed_sigma_per_mpc3_per_mag=np.array([2.8e-6, 2.0e-5]) * scale,
        halo_tracks=(),
    )


def _diagnostic(
    redshift: float,
    mode: str,
    *,
    seconds: float = 1.0,
) -> ModeRunDiagnostics:
    return ModeRunDiagnostics(
        redshift=redshift,
        imf_mode=mode,
        sampling_seconds=seconds,
        sample_count=3,
        valid_sample_count=2,
        topheavy_source_fraction=0.0 if mode == "canonical" else 0.25,
        popiii_source_fraction=0.0,
        sfrd_msun_per_yr_per_mpc3=1.0e-3,
        popiii_sfrd_msun_per_yr_per_mpc3=0.0,
    )


def _sample(redshift: float, mode: str, *, offset: float = 0.0) -> HaloSampleTable:
    return HaloSampleTable(
        redshift=redshift,
        imf_mode=mode,
        mass_index=np.array([0, 1], dtype=np.int64),
        track_index=np.array([0, 0], dtype=np.int64),
        halo_mass_msun=np.array([1.0e9, 2.0e9]),
        mass_weight_per_mpc3=np.array([1.0e-4, 2.0e-4]),
        uv_luminosity_erg_per_s_hz=np.array([1.0e27, 2.0e27]),
        muv=np.array([-18.0 + offset, -20.0]),
        sfr_msun_per_yr=np.array([0.1, 0.2]),
        popiii_sfr_msun_per_yr=np.array([0.0, 0.0]),
    )


def _shard(
    config: UVLFRunConfig,
    redshift: float,
    mode: str,
    *,
    scale: float = 1.0,
    seconds: float = 1.0,
    provenance: ArtifactProvenance | None = None,
    sample: HaloSampleTable | None = None,
) -> UVLFShard:
    descriptor = (
        None
        if sample is None
        else HaloSampleDescriptor(redshift, mode, sample.mass_index.size)
    )
    return UVLFShard(
        config=config,
        provenance=_provenance(config) if provenance is None else provenance,
        result=_mode(mode, scale),
        diagnostic=_diagnostic(redshift, mode, seconds=seconds),
        sample_descriptor=descriptor,
        sample=sample,
    )


def _write_grid(
    tmp_path: Path,
    config: UVLFRunConfig,
    *,
    created_by_key: dict[tuple[float, str], str] | None = None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, (redshift, mode) in enumerate(
        (r, m) for r in config.redshifts for m in config.stellar_population.imf_modes
    ):
        created = (
            "2026-07-11T12:00:00Z"
            if created_by_key is None
            else created_by_key[(redshift, mode)]
        )
        shard = _shard(
            config,
            redshift,
            mode,
            scale=1.0 + index,
            seconds=1.0 + index,
            provenance=_provenance(config, created=created),
            sample=_sample(redshift, mode) if index == 0 else None,
        )
        path = tmp_path / uvlf_shard_filename(config, redshift, mode)
        paths.append(write_uvlf_shard_atomic(shard, path=path.resolve()))
    return tuple(paths)


def _refresh_marker(path: Path) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    marker = path.with_name(path.name + ".complete")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = artifact_hdf5._sha256_file(path)
    payload["size_bytes"] = path.stat().st_size
    marker.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_shard_types_filename_and_axis_invariants(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical", sample=_sample(6.0, "canonical"))
    path = (tmp_path / "one.h5").resolve()
    descriptor = UVLFShardDescriptor(path, 6.0, "canonical", 2)

    assert shard.key == (6.0, "canonical")
    assert descriptor.key == shard.key
    assert descriptor.path == path
    assert uvlf_shard_filename(config, 6.0, "canonical") == (
        "resume-test.z=6.canonical.shard.h5"
    )
    with pytest.raises(ValueError, match="configured axes"):
        replace(shard, diagnostic=_diagnostic(7.0, "canonical"))
    with pytest.raises(ValueError, match="configured axes|mode"):
        uvlf_shard_filename(config, 6.0, "../../escape")
    with pytest.raises(ValueError, match="absolute"):
        UVLFShardDescriptor(Path("../escape.h5"), 6.0, "canonical", None)


def test_shard_roundtrip_has_exact_kind_full_axes_and_single_axis_tree(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(6.0, "canonical")
    shard = _shard(config, 6.0, "canonical", sample=sample)
    path = write_uvlf_shard_atomic(shard, path=(tmp_path / "one.h5").resolve())

    lazy = read_uvlf_shard(path)
    loaded = read_uvlf_shard(path, load_samples=True)

    assert lazy.key == shard.key
    assert lazy.sample is None
    assert lazy.sample_descriptor == HaloSampleDescriptor(6.0, "canonical", 2)
    np.testing.assert_array_equal(loaded.result.phi_intrinsic_per_mpc3_per_mag, shard.result.phi_intrinsic_per_mpc3_per_mag)
    np.testing.assert_array_equal(loaded.sample.muv, sample.muv)
    with h5py.File(path, "r") as handle:
        assert dict(handle.attrs) == {
            "schema_name": "auroralf.uvlf",
            "schema_version": "2.0.0",
            "artifact_kind": "shard",
        }
        np.testing.assert_array_equal(handle["axes/redshifts"], np.array(REDSHIFTS))
        assert tuple(handle["axes/imf_modes"].asstr()[()]) == MODES
        assert set(handle["results"]) == {"z=6"}
        assert set(handle["results/z=6"]) == {"canonical"}
        assert set(handle["diagnostics"]) == {"z=6"}
        assert set(handle["diagnostics/z=6"]) == {"canonical"}


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("artifact_kind", "artifact_kind"),
        ("schema_version", "schema_version"),
        ("extra_axis", "exactly one redshift"),
        ("wrong_dtype", "raw_counts.*wrong dtype"),
        ("wrong_units", "wrong units"),
        ("soft_link", "SoftLink"),
    ],
)
def test_shard_reader_strictly_rejects_kind_schema_tree_dtype_units_and_links(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    config = _config(tmp_path)
    path = write_uvlf_shard_atomic(
        _shard(config, 6.0, "canonical"),
        path=(tmp_path / f"strict-{corruption}.h5").resolve(),
    )
    with h5py.File(path, "r+") as handle:
        if corruption == "artifact_kind":
            handle.attrs["artifact_kind"] = "final"
        elif corruption == "schema_version":
            handle.attrs["schema_version"] = "9.9.9"
        elif corruption == "extra_axis":
            handle["results"].create_group("z=8")
        elif corruption == "wrong_dtype":
            group = handle["results/z=6/canonical"]
            values = np.asarray(group["raw_counts"], dtype=np.float64)
            del group["raw_counts"]
            dataset = group.create_dataset("raw_counts", data=values, dtype=np.float64)
            dataset.attrs["units"] = "count"
        elif corruption == "wrong_units":
            handle["results/z=6/canonical/raw_counts"].attrs["units"] = "kg"
        else:
            del handle["config/sha256"]
            handle["config/sha256"] = h5py.SoftLink(
                "/provenance/config_sha256"
            )
    _refresh_marker(path)

    with pytest.raises((TypeError, ValueError), match=match):
        validate_uvlf_resume_shards(config, _provenance(config), (path,))


def test_shard_public_reader_rejects_marker_tamper_and_artifact_fd_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    path = write_uvlf_shard_atomic(
        _shard(config, 6.0, "canonical"),
        path=(tmp_path / "fd-bound.h5").resolve(),
    )
    marker = path.with_name(path.name + ".complete")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker.write_text(
        json.dumps(
            {**marker_payload, "artifact_sha256": "0" * 64},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker.*checksum"):
        read_uvlf_shard(path)

    marker.write_text(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement-shard.h5"
    replacement.write_bytes(path.read_bytes())
    real_read_handle = artifact_hdf5._read_uvlf_shard_handle

    def replacing_read_handle(
        handle: h5py.File,
        *,
        load_samples: bool,
    ) -> UVLFShard:
        os.replace(replacement, path)
        return real_read_handle(handle, load_samples=load_samples)

    monkeypatch.setattr(
        artifact_hdf5,
        "_read_uvlf_shard_handle",
        replacing_read_handle,
    )
    with pytest.raises(ValueError, match="artifact.*identity|changed"):
        read_uvlf_shard(path)


def test_resume_validation_is_fail_fast_and_returns_config_axis_order(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    provenance = _provenance(config)
    paths = _write_grid(tmp_path, config)

    descriptors = validate_uvlf_resume_shards(
        config,
        provenance,
        tuple(reversed(paths)),
    )

    assert tuple(descriptor.key for descriptor in descriptors) == tuple(
        (redshift, mode)
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )
    assert all(type(item) is UVLFShardDescriptor for item in descriptors)
    paths[0].with_name(paths[0].name + ".complete").unlink()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        validate_uvlf_resume_shards(config, provenance, paths)


@pytest.mark.parametrize(
    ("field", "replacement_value"),
    [
        ("code_revision", "b" * 40),
        ("code_dirty", True),
        ("seed_namespace", "auroralf.pipeline.v2"),
    ],
)
def test_resume_rejects_provenance_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement_value: object,
) -> None:
    config = _config(tmp_path)
    expected = _provenance(config)
    changed = replace(expected, **{field: replacement_value})
    shard = _shard(config, 6.0, "canonical", provenance=changed)
    path = write_uvlf_shard_atomic(shard, path=(tmp_path / f"{field}.h5").resolve())

    with pytest.raises(ValueError, match=field):
        validate_uvlf_resume_shards(config, expected, (path,))


@pytest.mark.parametrize("field", ["label", "path"])
def test_resume_rejects_source_checksum_label_and_path_identity(
    tmp_path: Path,
    field: str,
) -> None:
    config = _config(tmp_path)
    expected = _provenance(config)
    original = expected.source_checksums[0]
    if field == "label":
        changed_source = SourceChecksum.from_path("canonical_alt", original.path)
    else:
        alternate_path = (tmp_path / "alternate-canonical.dat").resolve()
        alternate_path.write_bytes(original.path.read_bytes())
        changed_source = SourceChecksum.from_path(original.label, alternate_path)
    changed = replace(
        expected,
        source_checksums=(changed_source, *expected.source_checksums[1:]),
    )
    path = write_uvlf_shard_atomic(
        _shard(config, 6.0, "canonical", provenance=changed),
        path=(tmp_path / f"source-{field}.h5").resolve(),
    )

    with pytest.raises(ValueError, match="source_checksums"):
        validate_uvlf_resume_shards(config, expected, (path,))


@pytest.mark.parametrize("field", ["sha256", "size_bytes"])
def test_resume_rejects_tampered_source_hash_and_size_fields(
    tmp_path: Path,
    field: str,
) -> None:
    config = _config(tmp_path)
    expected = _provenance(config)
    path = write_uvlf_shard_atomic(
        _shard(config, 6.0, "canonical"),
        path=(tmp_path / f"source-{field}.h5").resolve(),
    )
    with h5py.File(path, "r+") as handle:
        source = handle["provenance/sources/canonical_ssp"]
        if field == "sha256":
            source["sha256"][()] = "0" * 64
        else:
            source["size_bytes"][()] = np.int64(
                int(source["size_bytes"][()]) + 1
            )
    _refresh_marker(path)

    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        validate_uvlf_resume_shards(config, expected, (path,))


def test_resume_rejects_config_and_source_identity_or_disk_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    expected = _provenance(config)
    other_config = _config(tmp_path / "other", base_seed=124)
    other = _shard(other_config, 6.0, "canonical")
    other_path = write_uvlf_shard_atomic(
        other,
        path=(tmp_path / "other-config.h5").resolve(),
    )
    with pytest.raises(ValueError, match="config"):
        validate_uvlf_resume_shards(config, expected, (other_path,))

    path = write_uvlf_shard_atomic(
        _shard(config, 6.0, "canonical"),
        path=(tmp_path / "source-change.h5").resolve(),
    )
    config.stellar_population.canonical_ssp_path.write_bytes(b"modified-source!")
    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        validate_uvlf_resume_shards(config, expected, (path,))


def test_merge_complete_grid_orders_axes_sums_unique_seconds_and_uses_earliest_utc(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    created = {
        (6.0, "canonical"): "2026-07-11T12:04:00Z",
        (6.0, "mah_burst_mild_topheavy"): "2026-07-11T12:03:00Z",
        (8.0, "canonical"): "2026-07-11T12:02:00Z",
        (8.0, "mah_burst_mild_topheavy"): "2026-07-11T12:01:00Z",
    }
    paths = _write_grid(tmp_path, config, created_by_key=created)
    output = (tmp_path / "final.h5").resolve()

    merged_path = merge_uvlf_shards(
        (paths[2], paths[0], paths[3], paths[1], paths[0]),
        output_path=output,
    )
    artifact = read_uvlf_artifact(merged_path, load_samples=True)

    assert tuple(redshift.redshift for redshift in artifact.result.redshifts) == REDSHIFTS
    assert tuple(
        mode.imf_mode
        for redshift in artifact.result.redshifts
        for mode in redshift.imf_modes
    ) == MODES + MODES
    assert artifact.result.diagnostics.total_seconds == 10.0
    assert artifact.provenance.created_utc == "2026-07-11T12:01:00Z"
    assert artifact.sample_keys == ((6.0, "canonical"),)


def test_merge_can_append_checksummed_execution_provenance_without_changing_shards(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = _write_grid(tmp_path, config)
    scientific = _provenance(config)
    execution_manifest = (tmp_path / "slurm-execution.json").resolve()
    execution_manifest.write_text(
        json.dumps(
            {
                "job_id": "12345",
                "requested_cpus": 8,
                "requested_memory": "32G",
                "requested_time": "02:00:00",
                "command": ["run_uvlf_v2.py", "--config", "production.toml"],
                "stdout_path": "outputs/job.out",
                "stderr_path": "outputs/job.err",
                "exit_code": 0,
                "final_artifact_validated": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    final = replace(
        scientific,
        source_checksums=(
            *scientific.source_checksums,
            SourceChecksum.from_path("slurm_execution", execution_manifest),
        ),
    )
    output = (tmp_path / "final-with-execution.h5").resolve()

    merge_uvlf_shards(
        paths,
        output_path=output,
        final_provenance=final,
    )
    artifact = read_uvlf_artifact(output, load_samples=False)

    assert artifact.provenance.source_checksums == final.source_checksums
    assert read_uvlf_shard(paths[0], load_samples=False).provenance == scientific


def test_merge_rejects_missing_axis_and_preserves_existing_final(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = _write_grid(tmp_path, config)
    output = (tmp_path / "final.h5").resolve()
    merge_uvlf_shards(paths, output_path=output)
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    marker_before = output.with_name(output.name + ".complete").read_bytes()

    with pytest.raises(ValueError, match="missing.*axis|complete coverage"):
        merge_uvlf_shards(paths[:-1], output_path=output, overwrite=True)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == before
    assert output.with_name(output.name + ".complete").read_bytes() == marker_before


def test_merge_uses_config_output_by_default_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = _write_grid(tmp_path, config)

    assert merge_uvlf_shards(paths) == config.output.artifact_path
    with pytest.raises(FileExistsError):
        merge_uvlf_shards(paths)
    assert merge_uvlf_shards(paths, overwrite=True) == config.output.artifact_path


@pytest.mark.parametrize("conflict", ["result", "diagnostic", "sample", "provenance"])
def test_merge_rejects_conflicting_duplicate_key(
    tmp_path: Path,
    conflict: str,
) -> None:
    config = _config(tmp_path)
    paths = list(_write_grid(tmp_path, config))
    base = _shard(
        config,
        6.0,
        "canonical",
        sample=_sample(6.0, "canonical"),
    )
    if conflict == "result":
        duplicate = replace(base, result=_mode("canonical", 99.0))
    elif conflict == "diagnostic":
        duplicate = replace(base, diagnostic=_diagnostic(6.0, "canonical", seconds=99.0))
    elif conflict == "sample":
        duplicate = replace(base, sample=_sample(6.0, "canonical", offset=1.0))
    else:
        duplicate = replace(base, provenance=_provenance(config, revision="b" * 40))
    duplicate_path = write_uvlf_shard_atomic(
        duplicate,
        path=(tmp_path / f"duplicate-{conflict}.h5").resolve(),
    )

    with pytest.raises(ValueError, match="conflicting duplicate|provenance"):
        merge_uvlf_shards(tuple(paths) + (duplicate_path,), output_path=(tmp_path / "bad.h5").resolve())


def test_shard_writer_reuses_atomic_concurrency_and_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = (tmp_path / "concurrent.h5").resolve()
    outcomes: list[object] = []
    barrier = threading.Barrier(2)

    def writer() -> None:
        barrier.wait(timeout=30.0)
        try:
            outcomes.append(write_uvlf_shard_atomic(shard, path=path))
        except BaseException as error:
            outcomes.append(error)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert sum(isinstance(item, Path) for item in outcomes) == 1
    assert sum(isinstance(item, FileExistsError) for item in outcomes) == 1

    failed = (tmp_path / "failed.h5").resolve()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected shard failure")

    monkeypatch.setattr(artifact_hdf5, "_write_uvlf_shard_hdf5_file", fail_write)
    with pytest.raises(RuntimeError, match="injected shard failure"):
        write_uvlf_shard_atomic(shard, path=failed)
    assert not failed.exists()
    assert not failed.with_name(failed.name + ".complete").exists()


def test_shard_filename_normalizes_configured_negative_zero_axis(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), redshifts=(-0.0, 6.0))

    assert uvlf_shard_filename(config, -0.0, "canonical") == (
        "resume-test.z=0.canonical.shard.h5"
    )
    assert uvlf_shard_filename(config, 0.0, "canonical") == (
        "resume-test.z=0.canonical.shard.h5"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "marker_write",
        "source_reverify",
        "final_validation",
        "final_target_validation",
    ],
)
def test_shard_overwrite_failure_restores_old_pair_bytes_modes_and_readability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = write_uvlf_shard_atomic(
        shard,
        path=(tmp_path / "rollback-shard.h5").resolve(),
    )
    marker = path.with_name(path.name + ".complete")
    path.chmod(0o640)
    marker.chmod(0o640)
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    old_modes = (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode))
    marker_written = False
    real_marker = artifact_hdf5._write_completion_marker_atomic

    def marker_hook(*args: object, **kwargs: object) -> tuple[int, int]:
        nonlocal marker_written
        if fault == "marker_write":
            raise RuntimeError("injected marker write failure")
        owner = real_marker(*args, **kwargs)
        marker_written = True
        return owner

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        marker_hook,
    )
    if fault == "source_reverify":
        real_verify = ArtifactProvenance.verify_sources

        def fail_after_marker(self: ArtifactProvenance) -> None:
            if marker_written:
                raise RuntimeError("injected source reverify failure")
            real_verify(self)

        monkeypatch.setattr(ArtifactProvenance, "verify_sources", fail_after_marker)
    elif fault == "final_validation":
        monkeypatch.setattr(
            artifact_hdf5,
            "_require_completion_marker_bound",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("injected final validation failure")
            ),
        )
    elif fault == "final_target_validation":
        real_target_validation = artifact_hdf5._require_committed_artifact_bound
        validation_count = 0

        def fail_final_target_validation(*args: object, **kwargs: object) -> None:
            nonlocal validation_count
            validation_count += 1
            if validation_count == 2:
                raise RuntimeError("injected final target validation failure")
            real_target_validation(*args, **kwargs)

        monkeypatch.setattr(
            artifact_hdf5,
            "_require_committed_artifact_bound",
            fail_final_target_validation,
        )

    with pytest.raises(RuntimeError, match="injected"):
        write_uvlf_shard_atomic(shard, path=path, overwrite=True)

    marker_written = False
    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode)) == old_modes
    assert read_uvlf_shard(path).key == shard.key


def test_overwrite_rollback_never_deletes_foreign_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = write_uvlf_shard_atomic(
        shard,
        path=(tmp_path / "foreign-rollback.h5").resolve(),
    )
    marker = path.with_name(path.name + ".complete")
    foreign_payload = b"foreign-target-must-survive"

    def replace_with_foreign_then_fail(*args: object, **kwargs: object) -> None:
        foreign = tmp_path / "foreign-target"
        foreign.write_bytes(foreign_payload)
        os.replace(foreign, path)
        raise RuntimeError("injected foreign replacement")

    monkeypatch.setattr(
        artifact_hdf5,
        "_require_completion_marker_bound",
        replace_with_foreign_then_fail,
    )
    with pytest.raises(RuntimeError, match="foreign replacement"):
        write_uvlf_shard_atomic(shard, path=path, overwrite=True)

    assert path.read_bytes() == foreign_payload
    assert not marker.exists()
    assert list(tmp_path.glob(".foreign-rollback.h5.*.backup"))
    assert list(tmp_path.glob(".foreign-rollback.h5.complete.*.backup"))


def test_successful_shard_overwrite_removes_transaction_backups(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = write_uvlf_shard_atomic(
        shard,
        path=(tmp_path / "successful-overwrite.h5").resolve(),
    )

    assert write_uvlf_shard_atomic(shard, path=path, overwrite=True) == path
    assert list(tmp_path.glob(".successful-overwrite.h5.*.backup")) == []
    assert list(tmp_path.glob(".successful-overwrite.h5.complete.*.backup")) == []
    assert read_uvlf_shard(path).key == shard.key


def test_shard_committed_pair_survives_partial_backup_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = write_uvlf_shard_atomic(
        shard,
        path=(tmp_path / "cleanup-shard.h5").resolve(),
    )
    marker = path.with_name(path.name + ".complete")
    real_unlink_owned = artifact_hdf5._unlink_owned_file

    def fail_owned_marker_backup_cleanup(
        candidate: Path,
        owner: tuple[int, int] | None,
    ) -> None:
        if (
            ".complete." in candidate.name
            and candidate.name.endswith(".backup")
            and owner is not None
            and artifact_hdf5._path_matches_owner(candidate, owner)
        ):
            raise OSError("injected shard marker backup cleanup failure")
        real_unlink_owned(candidate, owner)

    monkeypatch.setattr(
        artifact_hdf5,
        "_unlink_owned_file",
        fail_owned_marker_backup_cleanup,
    )

    with pytest.raises(OSError, match="shard marker backup cleanup failure"):
        write_uvlf_shard_atomic(shard, path=path, overwrite=True)

    assert read_uvlf_shard(path).key == shard.key
    assert marker.is_file()
    assert [
        candidate
        for candidate in tmp_path.glob(".cleanup-shard.h5.*.backup")
        if ".complete." not in candidate.name
    ] == []
    assert len(list(tmp_path.glob(".cleanup-shard.h5.complete.*.backup"))) == 1


@pytest.mark.parametrize("mutation", ["missing_target", "foreign_marker"])
def test_shard_rollback_validates_all_backups_before_removing_new_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    shard = _shard(config, 6.0, "canonical")
    path = write_uvlf_shard_atomic(
        shard,
        path=(tmp_path / "validate-backups.h5").resolve(),
    )
    marker = path.with_name(path.name + ".complete")
    foreign_payload = b"foreign-shard-marker-backup"

    def mutate_backup_then_fail(*args: object, **kwargs: object) -> None:
        if mutation == "missing_target":
            backup = next(
                candidate
                for candidate in tmp_path.glob(".validate-backups.h5.*.backup")
                if ".complete." not in candidate.name
            )
            backup.unlink()
        else:
            backup = next(
                tmp_path.glob(".validate-backups.h5.complete.*.backup")
            )
            foreign = tmp_path / "foreign-shard-marker-backup"
            foreign.write_bytes(foreign_payload)
            os.replace(foreign, backup)
        raise RuntimeError("injected shard backup mutation")

    monkeypatch.setattr(
        artifact_hdf5,
        "_require_completion_marker_bound",
        mutate_backup_then_fail,
    )

    with pytest.raises(RuntimeError, match="shard backup mutation"):
        write_uvlf_shard_atomic(shard, path=path, overwrite=True)

    assert path.is_file()
    assert marker.is_file()
    assert read_uvlf_shard(path).key == shard.key
    if mutation == "foreign_marker":
        backup = next(tmp_path.glob(".validate-backups.h5.complete.*.backup"))
        assert backup.read_bytes() == foreign_payload


@pytest.mark.parametrize("window", ["before_final_write", "after_final_marker"])
def test_merge_rejects_changed_input_snapshot_and_preserves_old_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    paths = _write_grid(tmp_path, config)
    output = (tmp_path / "snapshot-final.h5").resolve()
    merge_uvlf_shards(paths, output_path=output)
    output_marker = output.with_name(output.name + ".complete")
    old_final = output.read_bytes()
    old_marker = output_marker.read_bytes()
    changed_path = paths[0]

    if window == "before_final_write":
        real_read = artifact_hdf5._read_uvlf_shard_with_snapshot
        read_count = 0

        def replace_after_last_read(*args: object, **kwargs: object) -> object:
            nonlocal read_count
            result = real_read(*args, **kwargs)
            read_count += 1
            if read_count == len(paths):
                replacement = tmp_path / "replacement-input.h5"
                replacement.write_bytes(changed_path.read_bytes())
                os.replace(replacement, changed_path)
            return result

        monkeypatch.setattr(
            artifact_hdf5,
            "_read_uvlf_shard_with_snapshot",
            replace_after_last_read,
        )
    else:
        real_marker_write = artifact_hdf5._write_completion_marker_atomic

        def change_input_after_final_marker(
            marker_path: Path,
            payload: dict[str, object],
            *,
            file_mode: int = 0o600,
        ) -> tuple[int, int]:
            owner = real_marker_write(marker_path, payload, file_mode=file_mode)
            if marker_path == output_marker:
                input_marker = changed_path.with_name(changed_path.name + ".complete")
                replacement = tmp_path / "replacement-input.complete"
                replacement.write_bytes(input_marker.read_bytes())
                os.replace(replacement, input_marker)
            return owner

        monkeypatch.setattr(
            artifact_hdf5,
            "_write_completion_marker_atomic",
            change_input_after_final_marker,
        )

    with pytest.raises(ValueError, match="input shard.*changed|snapshot"):
        merge_uvlf_shards(paths, output_path=output, overwrite=True)

    assert output.read_bytes() == old_final
    assert output_marker.read_bytes() == old_marker
    assert read_uvlf_artifact(output).result.config == config


def test_merge_holds_shared_input_locks_until_exit_and_releases_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    paths = _write_grid(tmp_path, config)
    output = (tmp_path / "guarded-merge.h5").resolve()
    entered_final_write = threading.Event()
    release_final_write = threading.Event()
    real_final_write = artifact_hdf5._write_uvlf_artifact_atomic_with_validator

    def blocking_final_write(*args: object, **kwargs: object) -> Path:
        entered_final_write.set()
        assert release_final_write.wait(timeout=30.0)
        return real_final_write(*args, **kwargs)

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_uvlf_artifact_atomic_with_validator",
        blocking_final_write,
    )
    outcomes: list[object] = []

    def run_merge() -> None:
        try:
            outcomes.append(merge_uvlf_shards(paths, output_path=output))
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=run_merge)
    thread.start()
    assert entered_final_write.wait(timeout=30.0)
    shard = read_uvlf_shard(paths[0], load_samples=True)
    try:
        with pytest.raises(FileExistsError, match="commit lock|held"):
            write_uvlf_shard_atomic(shard, path=paths[0], overwrite=True)
    finally:
        release_final_write.set()
        thread.join(timeout=30.0)

    assert not thread.is_alive()
    assert outcomes == [output]
    assert write_uvlf_shard_atomic(shard, path=paths[0], overwrite=True) == paths[0]
