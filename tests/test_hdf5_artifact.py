from __future__ import annotations

from dataclasses import replace
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import weakref

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
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArtifactProvenance,
    HaloSampleDescriptor,
    HaloSampleTable,
    SourceChecksum,
    UVLFArtifact,
    canonical_config_json,
    canonical_config_sha256,
    decode_canonical_config_json,
    read_uvlf_artifact,
    write_uvlf_artifact_atomic,
)
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)


MODES = ("canonical", "mah_burst_mild_topheavy")
REDSHIFTS = (6.0, 8.0)


def _config(tmp_path: Path) -> UVLFRunConfig:
    canonical = (tmp_path / "canonical.dat").resolve()
    topheavy = (tmp_path / "topheavy.hdf5").resolve()
    popiii = (tmp_path / "popiii.dat").resolve()
    canonical.write_bytes(b"canonical-source")
    topheavy.write_bytes(b"topheavy-source")
    popiii.write_bytes(b"popiii-source")
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="artifact-test",
        redshifts=REDSHIFTS,
        base_seed=123,
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
        output=OutputConfig((tmp_path / "artifact.h5").resolve()),
    )


def _mode(mode: str, scale: float) -> IMFModeResult:
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


def _result(config: UVLFRunConfig) -> UVLFRunResult:
    redshift_results = tuple(
        RedshiftResult(
            redshift=redshift,
            imf_modes=tuple(
                _mode(mode, 1.0 + redshift / 100.0 + mode_index)
                for mode_index, mode in enumerate(MODES)
            ),
        )
        for redshift in REDSHIFTS
    )
    diagnostics = tuple(
        ModeRunDiagnostics(
            redshift=redshift,
            imf_mode=mode,
            sampling_seconds=1.0 + redshift / 10.0 + mode_index,
            sample_count=4,
            valid_sample_count=3,
            topheavy_source_fraction=0.0 if mode == "canonical" else 0.25,
            popiii_source_fraction=0.0,
            sfrd_msun_per_yr_per_mpc3=1.0e-3,
            popiii_sfrd_msun_per_yr_per_mpc3=0.0,
        )
        for redshift in REDSHIFTS
        for mode_index, mode in enumerate(MODES)
    )
    return UVLFRunResult(
        config=config,
        redshifts=redshift_results,
        diagnostics=RunDiagnostics(total_seconds=9.5, mode_runs=diagnostics),
    )


def _provenance(config: UVLFRunConfig) -> ArtifactProvenance:
    return ArtifactProvenance.for_config(
        config,
        code_revision="a" * 40,
        code_dirty=False,
        seed_namespace="auroralf.pipeline.v1",
        source_paths=(
            ("canonical_ssp", config.stellar_population.canonical_ssp_path),
            ("topheavy_ssp", config.stellar_population.topheavy_ssp_path),
        ),
        created_utc="2026-07-11T12:34:56Z",
    )


def _samples(redshift: float = 6.0, mode: str = "canonical") -> HaloSampleTable:
    return HaloSampleTable(
        redshift=redshift,
        imf_mode=mode,
        mass_index=np.array([0, 0, 1], dtype=np.int64),
        track_index=np.array([0, 1, 0], dtype=np.int64),
        halo_mass_msun=np.array([1.0e9, 1.0e9, 2.0e9]),
        mass_weight_per_mpc3=np.array([1.0e-4, 1.0e-4, 2.0e-4]),
        uv_luminosity_erg_per_s_hz=np.array([1.0e27, 2.0e27, 3.0e27]),
        muv=np.array([-18.0, np.nan, -20.0]),
        sfr_msun_per_yr=np.array([0.1, 0.2, 0.3]),
        popiii_sfr_msun_per_yr=np.array([0.0, 0.0, 0.01]),
    )


def test_schema_constants_and_canonical_config_roundtrip_are_exact(tmp_path: Path) -> None:
    config = _config(tmp_path)

    encoded = canonical_config_json(config)
    decoded = decode_canonical_config_json(encoded)

    assert SCHEMA_NAME == "auroralf.uvlf"
    assert SCHEMA_VERSION == "2.0.0"
    assert encoded == json.dumps(
        json.loads(encoded),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert '"tng_cache_path":null' in encoded
    assert decoded == config
    assert canonical_config_json(decoded) == encoded
    assert canonical_config_sha256(decoded) == canonical_config_sha256(config)

    root = json.loads(encoded)
    root["unknown"] = 1
    with pytest.raises(ValueError, match="unknown.*unknown"):
        decode_canonical_config_json(json.dumps(root))
    del root["unknown"]
    del root["run_id"]
    with pytest.raises(ValueError, match="missing.*run_id"):
        decode_canonical_config_json(json.dumps(root))
    bool_numeric = json.loads(encoded)
    bool_numeric["base_seed"] = True
    with pytest.raises(TypeError, match="base_seed.*boolean|base_seed.*int"):
        decode_canonical_config_json(json.dumps(bool_numeric))


def test_source_checksum_and_provenance_are_strict_and_disk_bound(tmp_path: Path) -> None:
    config = _config(tmp_path)
    provenance = _provenance(config)

    assert provenance.config_sha256 == canonical_config_sha256(config)
    assert tuple(item.label for item in provenance.source_checksums) == (
        "canonical_ssp",
        "topheavy_ssp",
    )
    assert all(len(item.sha256) == 64 for item in provenance.source_checksums)
    with pytest.raises(ValueError, match="40.*hex"):
        replace(provenance, code_revision="unknown")
    with pytest.raises(TypeError, match="code_dirty.*boolean"):
        replace(provenance, code_dirty=1)
    with pytest.raises(ValueError, match="versioned"):
        replace(provenance, seed_namespace="auroralf")
    with pytest.raises(ValueError, match="duplicate.*label"):
        replace(
            provenance,
            source_checksums=(
                provenance.source_checksums[0],
                replace(
                    provenance.source_checksums[1],
                    label=provenance.source_checksums[0].label,
                ),
            ),
        )
    with pytest.raises(ValueError, match="lowercase.*64"):
        replace(provenance.source_checksums[0], sha256="A" * 64)
    with pytest.raises(TypeError, match="size_bytes.*integer non-boolean"):
        replace(provenance.source_checksums[0], size_bytes=True)


def test_sample_table_and_artifact_are_strict_immutable_and_axis_bound(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = _result(config)
    provenance = _provenance(config)
    source_mass = np.array([0, 0, 1], dtype=np.int64)
    samples = replace(_samples(), mass_index=source_mass)
    artifact = UVLFArtifact(
        result=result,
        provenance=provenance,
        sample_descriptors=(HaloSampleDescriptor(6.0, "canonical", 3),),
        samples=(samples,),
    )

    source_mass[0] = 99
    assert artifact.samples[0].mass_index[0] == 0
    assert artifact.samples[0].mass_index.flags.writeable is False
    with pytest.raises(ValueError):
        artifact.samples[0].mass_index.flags.writeable = True
    with pytest.raises(ValueError, match="muv.*infinity"):
        replace(samples, muv=np.array([-18.0, np.inf, -20.0]))
    with pytest.raises(ValueError, match="descriptors.*samples"):
        replace(
            artifact,
            sample_descriptors=(HaloSampleDescriptor(8.0, "canonical", 3),),
        )
    with pytest.raises(ValueError, match="config hash"):
        replace(artifact, provenance=replace(provenance, config_sha256="b" * 64))


def _assert_result_exact(actual: UVLFRunResult, expected: UVLFRunResult) -> None:
    assert actual.config == expected.config
    assert actual.diagnostics == expected.diagnostics
    assert tuple(item.redshift for item in actual.redshifts) == tuple(
        item.redshift for item in expected.redshifts
    )
    fields_to_compare = (
        "bin_edges_muv",
        "bin_centers_muv",
        "bin_width_mag",
        "raw_counts",
        "weighted_counts_per_mpc3",
        "weight_squared_counts_per_mpc6",
        "weighted_count_sigma_per_mpc3",
        "effective_counts",
        "phi_intrinsic_per_mpc3_per_mag",
        "phi_intrinsic_sigma_per_mpc3_per_mag",
        "phi_observed_per_mpc3_per_mag",
        "phi_observed_sigma_per_mpc3_per_mag",
    )
    for actual_redshift, expected_redshift in zip(
        actual.redshifts,
        expected.redshifts,
        strict=True,
    ):
        for actual_mode, expected_mode in zip(
            actual_redshift.imf_modes,
            expected_redshift.imf_modes,
            strict=True,
        ):
            assert actual_mode.imf_mode == expected_mode.imf_mode
            assert actual_mode.halo_tracks == ()
            for field_name in fields_to_compare:
                np.testing.assert_array_equal(
                    getattr(actual_mode, field_name),
                    getattr(expected_mode, field_name),
                )


def test_hdf5_artifact_roundtrip_tree_dtypes_units_and_samples_on_off(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = _result(config)
    provenance = _provenance(config)
    samples = _samples()
    artifact = UVLFArtifact(
        result=result,
        provenance=provenance,
        sample_descriptors=(HaloSampleDescriptor(6.0, "canonical", 3),),
        samples=(samples,),
    )

    path = write_uvlf_artifact_atomic(artifact)
    without_samples = read_uvlf_artifact(path)
    with_samples = read_uvlf_artifact(path, load_samples=True)

    assert path == config.output.artifact_path
    assert path.with_name(path.name + ".complete").is_file()
    _assert_result_exact(without_samples.result, result)
    assert without_samples.provenance == provenance
    assert without_samples.sample_keys == (samples.key,)
    assert without_samples.sample_descriptors == (
        HaloSampleDescriptor(6.0, "canonical", 3),
    )
    assert without_samples.samples == ()
    _assert_result_exact(with_samples.result, result)
    assert with_samples.sample_keys == (samples.key,)
    assert len(with_samples.samples) == 1
    for field_name in (
        "mass_index",
        "track_index",
        "halo_mass_msun",
        "mass_weight_per_mpc3",
        "uv_luminosity_erg_per_s_hz",
        "muv",
        "sfr_msun_per_yr",
        "popiii_sfr_msun_per_yr",
    ):
        np.testing.assert_array_equal(
            getattr(with_samples.samples[0], field_name),
            getattr(samples, field_name),
        )

    with h5py.File(path, "r") as handle:
        assert dict(handle.attrs) == {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
        }
        assert set(handle) == {
            "config",
            "provenance",
            "axes",
            "results",
            "diagnostics",
            "samples",
        }
        assert handle["axes/redshifts"].dtype == np.dtype("float64")
        assert handle["axes/redshifts"].attrs["units"] == "dimensionless"
        assert handle["axes/muv_bin_edges"].dtype == np.dtype("float64")
        assert handle["axes/muv_bin_edges"].attrs["units"] == "mag"
        assert list(handle["results"]) == ["z=6", "z=8"]
        mode_group = handle["results/z=6/canonical"]
        assert set(mode_group.attrs) == {"redshift", "imf_mode"}
        assert mode_group["raw_counts"].dtype == np.dtype("int64")
        assert mode_group["phi_intrinsic_per_mpc3_per_mag"].dtype == np.dtype(
            "float64"
        )
        assert (
            mode_group["phi_intrinsic_per_mpc3_per_mag"].attrs["units"]
            == "Mpc^-3 mag^-1"
        )
        sample_dataset = handle["samples/z=6/canonical/halo_mass_msun"]
        assert sample_dataset.chunks is not None
        assert sample_dataset.maxshape == (None,)
        assert sample_dataset.compression is not None
        assert sample_dataset.shuffle
        assert sample_dataset.fletcher32
        assert sample_dataset.attrs["units"] == "Msun"


def test_hdf5_strict_reader_rejects_unknown_tree_object(tmp_path: Path) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    old_modes = (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode))
    with h5py.File(path, "r+") as handle:
        handle.create_group("unknown")

    with pytest.raises(ValueError, match="unknown.*root.*unknown"):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_public_reader_requires_valid_marker_and_detects_tamper(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))

    marker.unlink()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        read_uvlf_artifact(path)

    marker.write_text(
        json.dumps(
            {**marker_payload, "artifact_sha256": "0" * 64},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker.*checksum"):
        read_uvlf_artifact(path)

    marker.write_text(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact.*size|artifact.*checksum"):
        read_uvlf_artifact(path)


def test_atomic_write_overwrite_and_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match=str(path)):
        write_uvlf_artifact_atomic(artifact)
    assert write_uvlf_artifact_atomic(artifact, overwrite=True) == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_sha

    failed_path = (tmp_path / "failed.h5").resolve()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(artifact_hdf5, "_write_hdf5_file", fail_write)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_uvlf_artifact_atomic(artifact, path=failed_path)
    assert not failed_path.exists()
    assert not failed_path.with_name(failed_path.name + ".complete").exists()
    assert list(tmp_path.glob(".failed.h5.*.tmp")) == []


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("missing", "missing required result mode object.*effective_counts"),
        ("unknown_dataset", "unknown result mode object.*unknown"),
        ("unknown_attr", "unknown result mode attribute.*unknown"),
        ("dtype", "raw_counts.*wrong dtype"),
        ("shape", "sampling_seconds must be scalar"),
        ("nan", "phi_intrinsic.*finite"),
        ("inf", "phi_intrinsic_sigma.*infinity"),
        ("axes", "axes.redshifts.*config"),
        ("config_hash", "config hash mismatch"),
    ],
)
def test_unmarked_strict_reader_rejects_corrupt_schema_payloads(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        mode = handle["results/z=6/canonical"]
        if corruption == "missing":
            del mode["effective_counts"]
        elif corruption == "unknown_dataset":
            mode.create_dataset("unknown", data=np.array([1.0]))
        elif corruption == "unknown_attr":
            mode.attrs["unknown"] = "bad"
        elif corruption == "dtype":
            values = np.asarray(mode["raw_counts"], dtype=np.float64)
            del mode["raw_counts"]
            dataset = mode.create_dataset("raw_counts", data=values, dtype=np.float64)
            dataset.attrs["units"] = "count"
        elif corruption == "shape":
            diagnostics = handle["diagnostics/z=6/canonical"]
            del diagnostics["sampling_seconds"]
            dataset = diagnostics.create_dataset(
                "sampling_seconds", data=np.array([1.0]), dtype=np.float64
            )
            dataset.attrs["units"] = "s"
        elif corruption == "nan":
            mode["phi_intrinsic_per_mpc3_per_mag"][0] = np.nan
        elif corruption == "inf":
            mode["phi_intrinsic_sigma_per_mpc3_per_mag"][0] = np.inf
        elif corruption == "axes":
            handle["axes/redshifts"][0] = 7.0
        elif corruption == "config_hash":
            handle["config/sha256"][()] = "0" * 64
        else:  # pragma: no cover
            raise AssertionError(corruption)

    with pytest.raises((TypeError, ValueError), match=match):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_writer_and_reader_reverify_source_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    provenance = _provenance(config)
    artifact = UVLFArtifact(result=_result(config), provenance=provenance)
    config.stellar_population.canonical_ssp_path.write_bytes(b"changed-source")

    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        write_uvlf_artifact_atomic(artifact)
    assert not config.output.artifact_path.exists()

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    config.stellar_population.canonical_ssp_path.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        read_uvlf_artifact(path)


def test_post_rename_marker_failure_leaves_unblessed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))

    def fail_marker(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected marker failure")

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        fail_marker,
    )
    with pytest.raises(RuntimeError, match="injected marker failure"):
        write_uvlf_artifact_atomic(artifact)

    path = config.output.artifact_path
    assert path.is_file()
    assert not path.with_name(path.name + ".complete").exists()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        read_uvlf_artifact(path)


@pytest.mark.parametrize(
    "fault",
    ["source_reverify", "final_validation", "final_target_validation"],
)
def test_overwrite_late_failure_restores_old_artifact_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    old_modes = (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode))
    marker_written = False
    real_marker = artifact_hdf5._write_completion_marker_atomic

    def tracking_marker(*args: object, **kwargs: object) -> tuple[int, int]:
        nonlocal marker_written
        owner = real_marker(*args, **kwargs)
        marker_written = True
        return owner

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        tracking_marker,
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
    else:
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
        write_uvlf_artifact_atomic(artifact, overwrite=True)

    marker_written = False
    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode)) == old_modes
    assert read_uvlf_artifact(path).provenance == artifact.provenance


def test_overwrite_marker_failure_cannot_bless_replaced_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    old_modes = (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode))

    def fail_marker(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected overwrite marker failure")

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        fail_marker,
    )
    with pytest.raises(RuntimeError, match="overwrite marker failure"):
        write_uvlf_artifact_atomic(artifact, overwrite=True)

    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(marker.stat().st_mode)) == old_modes
    assert read_uvlf_artifact(path).provenance == artifact.provenance


def test_artifact_committed_pair_survives_partial_backup_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
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
            raise OSError("injected marker backup cleanup failure")
        real_unlink_owned(candidate, owner)

    monkeypatch.setattr(
        artifact_hdf5,
        "_unlink_owned_file",
        fail_owned_marker_backup_cleanup,
    )

    with pytest.raises(OSError, match="marker backup cleanup failure"):
        write_uvlf_artifact_atomic(artifact, overwrite=True)

    assert read_uvlf_artifact(path).provenance == artifact.provenance
    assert marker.is_file()
    assert [
        candidate
        for candidate in tmp_path.glob(".artifact.h5.*.backup")
        if ".complete." not in candidate.name
    ] == []
    assert len(list(tmp_path.glob(".artifact.h5.complete.*.backup"))) == 1


@pytest.mark.parametrize("mutation", ["missing_target", "foreign_marker"])
def test_artifact_rollback_validates_all_backups_before_removing_new_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    foreign_payload = b"foreign-marker-backup"

    def mutate_backup_then_fail(*args: object, **kwargs: object) -> None:
        if mutation == "missing_target":
            backup = next(
                candidate
                for candidate in tmp_path.glob(".artifact.h5.*.backup")
                if ".complete." not in candidate.name
            )
            backup.unlink()
        else:
            backup = next(tmp_path.glob(".artifact.h5.complete.*.backup"))
            foreign = tmp_path / "foreign-marker-backup"
            foreign.write_bytes(foreign_payload)
            os.replace(foreign, backup)
        raise RuntimeError("injected backup mutation")

    monkeypatch.setattr(
        artifact_hdf5,
        "_require_completion_marker_bound",
        mutate_backup_then_fail,
    )

    with pytest.raises(RuntimeError, match="backup mutation"):
        write_uvlf_artifact_atomic(artifact, overwrite=True)

    assert path.is_file()
    assert marker.is_file()
    assert read_uvlf_artifact(path).provenance == artifact.provenance
    if mutation == "foreign_marker":
        backup = next(tmp_path.glob(".artifact.h5.complete.*.backup"))
        assert backup.read_bytes() == foreign_payload


def test_atomic_write_uses_same_directory_temp_and_orders_fsync_before_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    events: list[tuple[str, Path, Path | None]] = []
    real_write = artifact_hdf5._write_hdf5_file
    real_fsync_file = artifact_hdf5._fsync_file
    real_fsync_directory = artifact_hdf5._fsync_directory
    real_replace = artifact_hdf5.os.replace

    def write_spy(received: UVLFArtifact, owned_file: object) -> None:
        events.append(("write", Path(getattr(owned_file, "path")), None))
        real_write(received, owned_file)

    def fsync_file_spy(owned_file: object) -> None:
        events.append(("fsync_file", Path(getattr(owned_file, "path")), None))
        real_fsync_file(owned_file)

    def fsync_directory_spy(path: Path) -> None:
        events.append(("fsync_directory", path, None))
        real_fsync_directory(path)

    def replace_spy(source: Path, destination: Path) -> None:
        events.append(("replace", Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(artifact_hdf5, "_write_hdf5_file", write_spy)
    monkeypatch.setattr(artifact_hdf5, "_fsync_file", fsync_file_spy)
    monkeypatch.setattr(artifact_hdf5, "_fsync_directory", fsync_directory_spy)
    monkeypatch.setattr(artifact_hdf5.os, "replace", replace_spy)

    path = write_uvlf_artifact_atomic(artifact)

    write_event = next(event for event in events if event[0] == "write")
    artifact_replace_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and event[2] == path
    )
    marker_replace_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and event[2] == path.with_name(path.name + ".complete")
    )
    fsync_file_index = next(
        index for index, event in enumerate(events) if event[0] == "fsync_file"
    )
    fsync_parent_before_replace = any(
        event[0] == "fsync_directory" and event[1] == path.parent
        for event in events[:artifact_replace_index]
    )
    assert write_event[1].parent == path.parent
    assert write_event[1].name.startswith(f".{path.name}.")
    assert fsync_parent_before_replace
    assert fsync_file_index < artifact_replace_index < marker_replace_index


def _large_samples(sample_count: int) -> HaloSampleTable:
    indices = np.arange(sample_count, dtype=np.int64)
    return HaloSampleTable(
        redshift=6.0,
        imf_mode="canonical",
        mass_index=indices % 7,
        track_index=indices % 11,
        halo_mass_msun=np.full(sample_count, 1.0e9),
        mass_weight_per_mpc3=np.full(sample_count, 1.0e-4),
        uv_luminosity_erg_per_s_hz=np.full(sample_count, 1.0e27),
        muv=np.full(sample_count, -18.0),
        sfr_msun_per_yr=np.full(sample_count, 0.1),
        popiii_sfr_msun_per_yr=np.zeros(sample_count),
    )


def test_lazy_sample_validation_uses_bounded_slices_and_returns_descriptors_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    samples = _large_samples(70_000)
    artifact = UVLFArtifact(
        result=_result(config),
        provenance=_provenance(config),
        sample_descriptors=(HaloSampleDescriptor(6.0, "canonical", 70_000),),
        samples=(samples,),
    )
    path = write_uvlf_artifact_atomic(artifact)
    real_getitem = h5py.Dataset.__getitem__
    accesses: list[tuple[object, int]] = []
    array_references: list[weakref.ReferenceType[np.ndarray]] = []

    def getitem_spy(dataset: h5py.Dataset, key: object) -> object:
        result = real_getitem(dataset, key)
        if dataset.name.startswith("/samples/"):
            assert dataset.chunks is not None
            accesses.append((key, dataset.chunks[0]))
            if isinstance(result, np.ndarray):
                array_references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem_spy)

    lazy = read_uvlf_artifact(path, load_samples=False)
    gc.collect()

    assert lazy.samples == ()
    assert lazy.sample_descriptors == (
        HaloSampleDescriptor(6.0, "canonical", 70_000),
    )
    assert accesses
    for key, dataset_chunk in accesses:
        assert isinstance(key, slice)
        assert key.start is not None and key.stop is not None
        assert key.step is None
        assert 0 < key.stop - key.start <= min(65_536, dataset_chunk)
    assert all(reference() is None for reference in array_references)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("unequal_shape", "same length"),
        ("dtype", "mass_index.*wrong dtype"),
        ("units", "halo_mass_msun.*wrong units"),
    ],
)
def test_lazy_sample_validation_rejects_structural_corruption_without_full_reads(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    samples = _large_samples(20)
    artifact = UVLFArtifact(
        result=_result(config),
        provenance=_provenance(config),
        sample_descriptors=(HaloSampleDescriptor(6.0, "canonical", 20),),
        samples=(samples,),
    )
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        group = handle["samples/z=6/canonical"]
        if corruption in ("unequal_shape", "dtype"):
            values = np.asarray(group["mass_index"])
            del group["mass_index"]
            if corruption == "unequal_shape":
                values = values[:-1]
                dtype = np.int64
            else:
                values = values.astype(np.float64)
                dtype = np.float64
            dataset = group.create_dataset(
                "mass_index",
                data=values,
                dtype=dtype,
                chunks=True,
                maxshape=(None,),
                compression="gzip",
                shuffle=True,
                fletcher32=True,
            )
            dataset.attrs["units"] = "index"
        else:
            group["halo_mass_msun"].attrs["units"] = "kg"

    with pytest.raises(ValueError, match=match):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


@pytest.mark.parametrize(
    ("field_name", "position", "value", "match"),
    [
        ("mass_index", "first", -1, "indices.*non-negative"),
        ("halo_mass_msun", "middle", 0.0, "halo_mass_msun.*positive"),
        ("mass_weight_per_mpc3", "last", -1.0, "mass_weight.*non-negative"),
        ("uv_luminosity_erg_per_s_hz", "middle", np.inf, "infinity"),
        ("sfr_msun_per_yr", "first", np.nan, "finite"),
        ("muv", "last", np.inf, "infinity"),
    ],
)
def test_lazy_sample_validation_rejects_first_middle_last_chunk_data_corruption(
    tmp_path: Path,
    field_name: str,
    position: str,
    value: float,
    match: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    sample_count = 70_000
    samples = _large_samples(sample_count)
    artifact = UVLFArtifact(
        result=_result(config),
        provenance=_provenance(config),
        sample_descriptors=(
            HaloSampleDescriptor(6.0, "canonical", sample_count),
        ),
        samples=(samples,),
    )
    path = write_uvlf_artifact_atomic(artifact)
    index = {"first": 0, "middle": sample_count // 2, "last": sample_count - 1}[
        position
    ]
    with h5py.File(path, "r+") as handle:
        handle[f"samples/z=6/canonical/{field_name}"][index] = value

    with pytest.raises((TypeError, ValueError), match=match):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_lazy_sample_validation_allows_nan_muv(tmp_path: Path) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    samples = _large_samples(70_000)
    artifact = UVLFArtifact(
        result=_result(config),
        provenance=_provenance(config),
        sample_descriptors=(HaloSampleDescriptor(6.0, "canonical", 70_000),),
        samples=(samples,),
    )
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        handle["samples/z=6/canonical/muv"][35_000] = np.nan

    lazy = artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)

    assert lazy.samples == ()
    assert lazy.sample_descriptors[0].sample_count == 70_000


@pytest.mark.parametrize("link_kind", ["soft", "external"])
def test_strict_reader_recursively_rejects_soft_and_external_links(
    tmp_path: Path,
    link_kind: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        del handle["config/sha256"]
        if link_kind == "soft":
            handle["config/sha256"] = h5py.SoftLink("/provenance/config_sha256")
        else:
            external_path = tmp_path / "external.h5"
            with h5py.File(external_path, "w") as external:
                external.create_dataset(
                    "payload",
                    data=artifact.provenance.config_sha256,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            handle["config/sha256"] = h5py.ExternalLink(
                str(external_path),
                "/payload",
            )

    with pytest.raises(ValueError, match=rf"{link_kind.capitalize()}Link|{link_kind} link"):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_strict_reader_rejects_hard_link_alias_between_required_paths(
    tmp_path: Path,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        group = handle["results/z=6/canonical"]
        del group["phi_observed_per_mpc3_per_mag"]
        group["phi_observed_per_mpc3_per_mag"] = group[
            "phi_intrinsic_per_mpc3_per_mag"
        ]

    with pytest.raises(ValueError, match="hard link alias|multiple paths"):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_public_reader_rejects_atomic_artifact_replacement_during_hdf_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    replacement = tmp_path / "replacement.h5"
    replacement.write_bytes(path.read_bytes())
    real_read_handle = artifact_hdf5._read_uvlf_artifact_handle
    replaced = False

    def replacing_read_handle(
        handle: h5py.File,
        *,
        load_samples: bool,
    ) -> UVLFArtifact:
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return real_read_handle(handle, load_samples=load_samples)

    monkeypatch.setattr(
        artifact_hdf5,
        "_read_uvlf_artifact_handle",
        replacing_read_handle,
    )

    with pytest.raises(ValueError, match="artifact.*changed|identity"):
        read_uvlf_artifact(path)


def test_source_checksum_verify_rejects_same_size_atomic_replace_during_fd_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.schema as artifact_schema

    source = tmp_path / "source.dat"
    replacement = tmp_path / "replacement.dat"
    source.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    checksum = SourceChecksum.from_path("source", source)
    real_hash_fd = artifact_schema._hash_open_file_descriptor
    replaced = False

    def replacing_hash_fd(descriptor: int) -> str:
        nonlocal replaced
        digest = real_hash_fd(descriptor)
        if not replaced:
            replaced = True
            os.replace(replacement, source)
        return digest

    monkeypatch.setattr(
        artifact_schema,
        "_hash_open_file_descriptor",
        replacing_hash_fd,
    )

    with pytest.raises(ValueError, match="source.*changed|identity"):
        checksum.verify()


def test_concurrent_overwrite_false_writers_have_exactly_one_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    real_write = artifact_hdf5._write_hdf5_file
    first_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def blocking_first_write(received: UVLFArtifact, path: Path) -> None:
        nonlocal call_count
        with call_lock:
            call_count += 1
            is_first = call_count == 1
        if is_first:
            first_entered.set()
            assert release_first.wait(timeout=30.0)
        real_write(received, path)

    monkeypatch.setattr(artifact_hdf5, "_write_hdf5_file", blocking_first_write)
    first_outcome: list[object] = []

    def first_writer() -> None:
        try:
            first_outcome.append(write_uvlf_artifact_atomic(artifact))
        except BaseException as error:  # captured for exact cross-thread assertion
            first_outcome.append(error)

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert first_entered.wait(timeout=30.0)
    second_outcome: object
    try:
        second_outcome = write_uvlf_artifact_atomic(artifact)
    except BaseException as error:
        second_outcome = error
    finally:
        release_first.set()
        thread.join(timeout=30.0)

    assert not thread.is_alive()
    outcomes = (*first_outcome, second_outcome)
    assert sum(isinstance(item, Path) for item in outcomes) == 1
    assert sum(isinstance(item, FileExistsError) for item in outcomes) == 1


class _FixedUUID:
    def __init__(self, value: str) -> None:
        self.hex = value


def test_artifact_temp_uuid_collision_never_deletes_unowned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    fixed_hex = "1" * 32
    collision = tmp_path / f".{config.output.artifact_path.name}.{fixed_hex}.tmp"
    collision.write_bytes(b"belongs-to-another-writer")
    monkeypatch.setattr(
        artifact_hdf5.uuid,
        "uuid4",
        lambda: _FixedUUID(fixed_hex),
    )

    with pytest.raises(FileExistsError):
        write_uvlf_artifact_atomic(artifact)

    assert collision.read_bytes() == b"belongs-to-another-writer"


def test_marker_temp_uuid_collision_never_deletes_unowned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    values = iter(("1" * 32, "2" * 32))
    monkeypatch.setattr(
        artifact_hdf5.uuid,
        "uuid4",
        lambda: _FixedUUID(next(values)),
    )
    marker = config.output.artifact_path.with_name(
        config.output.artifact_path.name + ".complete"
    )
    collision = tmp_path / f".{marker.name}.{'2' * 32}.tmp"
    collision.write_bytes(b"another-marker-temp")

    with pytest.raises(FileExistsError):
        write_uvlf_artifact_atomic(artifact)

    assert collision.read_bytes() == b"another-marker-temp"
    assert config.output.artifact_path.is_file()
    assert not marker.exists()


@pytest.mark.parametrize("file_mode", [0o600, 0o640])
def test_atomic_write_uses_secure_permissions_and_preserves_overwrite_modes(
    tmp_path: Path,
    file_mode: int,
) -> None:
    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600

    path.chmod(file_mode)
    marker.chmod(file_mode)
    write_uvlf_artifact_atomic(artifact, overwrite=True)

    assert stat.S_IMODE(path.stat().st_mode) == file_mode
    assert stat.S_IMODE(marker.stat().st_mode) == file_mode


def test_result_shape_is_rejected_before_dataset_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    target_name = "/results/z=6/canonical/effective_counts"
    with h5py.File(path, "r+") as handle:
        group = handle["results/z=6/canonical"]
        del group["effective_counts"]
        dataset = group.create_dataset(
            "effective_counts",
            data=np.array([1.0, 2.0, 3.0]),
            dtype=np.float64,
        )
        dataset.attrs["units"] = "count"
    real_getitem = h5py.Dataset.__getitem__
    target_reads: list[object] = []

    def getitem_spy(dataset: h5py.Dataset, key: object) -> object:
        if dataset.name == target_name:
            target_reads.append(key)
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem_spy)

    with pytest.raises(ValueError, match="effective_counts.*shape"):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)
    assert target_reads == []


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("fixed_ascii_dataset", "UTF-8.*variable|variable.*UTF-8"),
        ("fixed_ascii_attr", "imf_mode.*UTF-8"),
        ("float32_redshift_attr", "redshift.*float64"),
    ],
)
def test_strict_reader_rejects_noncanonical_string_and_scalar_attr_dtypes(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    with h5py.File(path, "r+") as handle:
        if corruption == "fixed_ascii_dataset":
            group = handle["config"]
            value = artifact.provenance.config_sha256.encode("ascii")
            del group["sha256"]
            group.create_dataset("sha256", data=np.bytes_(value), dtype="S64")
        else:
            group = handle["results/z=6/canonical"]
            if corruption == "fixed_ascii_attr":
                del group.attrs["imf_mode"]
                group.attrs.create("imf_mode", np.bytes_(b"canonical"), dtype="S9")
            else:
                del group.attrs["redshift"]
                group.attrs.create("redshift", np.float32(6.0), dtype=np.float32)

    with pytest.raises(ValueError, match=match):
        artifact_hdf5._read_uvlf_artifact_file(path, load_samples=False)


def test_writer_reverifies_sources_after_marker_creation_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    source = config.stellar_population.canonical_ssp_path
    replacement = tmp_path / "replacement-source.dat"
    replacement.write_bytes(b"modified-source!")
    assert replacement.stat().st_size == source.stat().st_size
    real_write_marker = artifact_hdf5._write_completion_marker_atomic

    def replacing_marker_write(
        marker: Path,
        payload: dict[str, object],
        *,
        file_mode: int = 0o600,
    ) -> tuple[int, int]:
        owner = real_write_marker(marker, payload, file_mode=file_mode)
        os.replace(replacement, source)
        return owner

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        replacing_marker_write,
    )

    with pytest.raises(ValueError, match="source.*checksum|source.*identity"):
        write_uvlf_artifact_atomic(artifact)

    path = config.output.artifact_path
    assert path.is_file()
    assert not path.with_name(path.name + ".complete").exists()


def test_reader_reverifies_sources_after_hdf_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    source = config.stellar_population.canonical_ssp_path
    replacement = tmp_path / "replacement-source.dat"
    replacement.write_bytes(b"modified-source!")
    assert replacement.stat().st_size == source.stat().st_size
    real_read_handle = artifact_hdf5._read_uvlf_artifact_handle

    def replacing_read_handle(
        handle: h5py.File,
        *,
        load_samples: bool,
    ) -> UVLFArtifact:
        result = real_read_handle(handle, load_samples=load_samples)
        os.replace(replacement, source)
        return result

    monkeypatch.setattr(
        artifact_hdf5,
        "_read_uvlf_artifact_handle",
        replacing_read_handle,
    )

    with pytest.raises(ValueError, match="source.*checksum|source.*identity"):
        read_uvlf_artifact(path)


def test_public_reader_rejects_external_link_even_when_target_changes_outside_artifact_sha(
    tmp_path: Path,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    external_path = tmp_path / "external.h5"
    with h5py.File(external_path, "w") as external:
        external.create_dataset(
            "payload",
            data=artifact.provenance.config_sha256,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with h5py.File(path, "r+") as handle:
        del handle["config/sha256"]
        handle["config/sha256"] = h5py.ExternalLink(
            str(external_path),
            "/payload",
        )
    marker = path.with_name(path.name + ".complete")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = artifact_hdf5._sha256_file(path)
    payload["size_bytes"] = path.stat().st_size
    marker.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    artifact_sha = artifact_hdf5._sha256_file(path)
    with h5py.File(external_path, "r+") as external:
        external["payload"][()] = "0" * 64
    assert artifact_hdf5._sha256_file(path) == artifact_sha

    with pytest.raises(ValueError, match="ExternalLink"):
        read_uvlf_artifact(path)


def test_public_reader_rejects_marker_atomic_replacement_during_hdf_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    path = write_uvlf_artifact_atomic(artifact)
    marker = path.with_name(path.name + ".complete")
    replacement = tmp_path / "replacement.complete"
    replacement.write_bytes(marker.read_bytes())
    real_read_handle = artifact_hdf5._read_uvlf_artifact_handle
    replaced = False

    def replacing_read_handle(
        handle: h5py.File,
        *,
        load_samples: bool,
    ) -> UVLFArtifact:
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, marker)
        return real_read_handle(handle, load_samples=load_samples)

    monkeypatch.setattr(
        artifact_hdf5,
        "_read_uvlf_artifact_handle",
        replacing_read_handle,
    )

    with pytest.raises(ValueError, match="completion marker.*identity"):
        read_uvlf_artifact(path)


def test_artifact_temp_replacement_is_never_opened_or_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    real_write = artifact_hdf5._write_hdf5_file
    foreign_payload = b"foreign-artifact-temp"
    replaced_path: list[Path] = []

    def replacing_write(received: UVLFArtifact, owned_file: object) -> None:
        path = Path(getattr(owned_file, "path", owned_file))
        replacement = tmp_path / "foreign-artifact-replacement"
        replacement.write_bytes(foreign_payload)
        os.replace(replacement, path)
        replaced_path.append(path)
        real_write(received, owned_file)

    monkeypatch.setattr(artifact_hdf5, "_write_hdf5_file", replacing_write)

    with pytest.raises(ValueError, match="temporary artifact.*identity|ownership"):
        write_uvlf_artifact_atomic(artifact)

    assert len(replaced_path) == 1
    assert replaced_path[0].read_bytes() == foreign_payload
    assert not config.output.artifact_path.exists()
    assert not config.output.artifact_path.with_name(
        config.output.artifact_path.name + ".complete"
    ).exists()


def test_marker_temp_replacement_is_never_opened_or_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    real_create = artifact_hdf5._create_owned_file
    foreign_payload = b"foreign-marker-temp"
    replaced_path: list[Path] = []

    def replacing_create(path: Path, *, mode: int) -> object:
        owned_file = real_create(path, mode=mode)
        if ".complete." in path.name and path.name.endswith(".tmp"):
            replacement = tmp_path / "foreign-marker-replacement"
            replacement.write_bytes(foreign_payload)
            os.replace(replacement, path)
            replaced_path.append(path)
        return owned_file

    monkeypatch.setattr(artifact_hdf5, "_create_owned_file", replacing_create)

    with pytest.raises(ValueError, match="temporary completion marker.*identity|ownership"):
        write_uvlf_artifact_atomic(artifact)

    assert len(replaced_path) == 1
    assert replaced_path[0].read_bytes() == foreign_payload
    assert config.output.artifact_path.is_file()
    assert not config.output.artifact_path.with_name(
        config.output.artifact_path.name + ".complete"
    ).exists()


def test_crashed_writer_leaves_stale_unlocked_lock_that_does_not_block(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    lock_path = config.output.artifact_path.with_name(
        "." + config.output.artifact_path.name + ".commit.lock"
    )
    stale_payload = b"stale-lock-file"
    crash = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys; "
                "fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT, 0o600); "
                "fcntl.flock(fd, fcntl.LOCK_EX); "
                "os.write(fd, b'stale-lock-file'); os.fsync(fd); os._exit(23)"
            ),
            str(lock_path),
        ],
        check=False,
    )
    assert crash.returncode == 23

    path = write_uvlf_artifact_atomic(artifact)

    assert path.is_file()
    assert lock_path.read_bytes() == stale_payload
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_active_commit_lock_strictly_rejects_concurrent_writer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    lock_path = config.output.artifact_path.with_name(
        "." + config.output.artifact_path.name + ".commit.lock"
    )
    lock_path.touch(mode=0o600)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(FileExistsError, match="commit.*progress|lock"):
            write_uvlf_artifact_atomic(artifact)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert not config.output.artifact_path.exists()
    assert lock_path.is_file()


def test_temp_path_replacement_in_final_rename_window_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    target = config.output.artifact_path
    marker = target.with_name(target.name + ".complete")
    foreign_payload = b"foreign-target-in-final-rename-window"
    real_replace = artifact_hdf5.os.replace
    replaced = False

    def replacing_artifact_temp(source: Path, destination: Path) -> None:
        nonlocal replaced
        source_path = Path(source)
        destination_path = Path(destination)
        if not replaced and destination_path == target:
            replaced = True
            foreign = tmp_path / "foreign-final-window"
            foreign.write_bytes(foreign_payload)
            real_replace(foreign, source_path)
        real_replace(source_path, destination_path)

    monkeypatch.setattr(artifact_hdf5.os, "replace", replacing_artifact_temp)

    with pytest.raises(ValueError, match="committed artifact.*identity|checksum"):
        write_uvlf_artifact_atomic(artifact)

    assert replaced
    assert target.read_bytes() == foreign_payload
    assert not marker.exists()


def test_target_replacement_after_marker_write_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    target = config.output.artifact_path
    marker = target.with_name(target.name + ".complete")
    foreign_payload = b"foreign-target-after-marker"
    real_write_marker = artifact_hdf5._write_completion_marker_atomic

    def replacing_target_after_marker(
        marker_path: Path,
        payload: dict[str, object],
        *,
        file_mode: int = 0o600,
    ) -> tuple[int, int]:
        owner = real_write_marker(marker_path, payload, file_mode=file_mode)
        foreign = tmp_path / "foreign-after-marker"
        foreign.write_bytes(foreign_payload)
        os.replace(foreign, target)
        return owner

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        replacing_target_after_marker,
    )

    with pytest.raises(ValueError, match="committed artifact.*identity|checksum"):
        write_uvlf_artifact_atomic(artifact)

    assert target.read_bytes() == foreign_payload
    assert not marker.exists()


def test_marker_content_change_after_write_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as artifact_hdf5

    config = _config(tmp_path)
    artifact = UVLFArtifact(result=_result(config), provenance=_provenance(config))
    target = config.output.artifact_path
    marker = target.with_name(target.name + ".complete")
    real_write_marker = artifact_hdf5._write_completion_marker_atomic

    def changing_marker_content(
        marker_path: Path,
        payload: dict[str, object],
        *,
        file_mode: int = 0o600,
    ) -> tuple[int, int]:
        owner = real_write_marker(marker_path, payload, file_mode=file_mode)
        marker_path.write_bytes(b"changed-marker-content")
        return owner

    monkeypatch.setattr(
        artifact_hdf5,
        "_write_completion_marker_atomic",
        changing_marker_content,
    )

    with pytest.raises(ValueError, match="completion marker.*payload|JSON"):
        write_uvlf_artifact_atomic(artifact)

    assert target.is_file()
    assert not marker.exists()
