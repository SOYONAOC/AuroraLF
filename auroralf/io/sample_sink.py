from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid

import h5py
import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.results import UVLFRunResult
from .hdf5 import (
    _RESULT_FIELDS,
    _SAMPLE_FIELDS,
    _SAMPLE_UNITS,
    _capture_spool_snapshot,
    _require_spool_owner,
    _write_uvlf_shard_from_spool_atomic,
    read_uvlf_shard,
    _write_config_provenance_axes,
    _z_name,
)
from .schema import (
    ArtifactProvenance,
    HaloSampleDescriptor,
    HaloSampleTable,
    UVLFShard,
    canonical_config_sha256,
    uvlf_shard_filename,
)


_SPOOL_CHUNK_SIZE = 65_536


class _HDF5SampleSink:
    def __init__(
        self,
        config: UVLFRunConfig,
        provenance: ArtifactProvenance,
        shard_directory: Path,
    ) -> None:
        if type(config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
        if type(provenance) is not ArtifactProvenance:
            raise TypeError("provenance must be exactly ArtifactProvenance")
        if not isinstance(shard_directory, Path):
            raise TypeError("shard_directory must be a Path")
        directory = shard_directory.resolve(strict=True)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        if provenance.config_sha256 != canonical_config_sha256(config):
            raise ValueError("provenance config hash does not match config")
        provenance.verify_sources()

        self._config = config
        self._provenance = provenance
        self._spool_path = directory / (
            f".auroralf-samples-{uuid.uuid4().hex}.spool.h5"
        )
        self._descriptor = -1
        self._identity: tuple[int, int] | None = None
        self._redshift_position = 0
        self._mass_position = 0
        self._mode_position = 0
        self._closed = False

        descriptor = os.open(
            self._spool_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        try:
            file_stat = os.fstat(descriptor)
            self._descriptor = descriptor
            self._identity = (file_stat.st_dev, file_stat.st_ino)
        except BaseException:
            os.close(descriptor)
            raise
        try:
            self._require_owned_spool()
            with os.fdopen(
                os.dup(self._descriptor), "r+b", closefd=True
            ) as file_object, h5py.File(file_object, "w") as handle:
                _write_config_provenance_axes(
                    handle,
                    config,
                    provenance,
                    artifact_kind="sample_spool",
                )
                samples_group = handle.create_group("samples", track_order=True)
                for redshift in config.redshifts:
                    redshift_group = samples_group.create_group(
                        _z_name(redshift),
                        track_order=True,
                    )
                    for mode in config.stellar_population.imf_modes:
                        mode_group = redshift_group.create_group(mode, track_order=True)
                        mode_group.attrs["redshift"] = np.float64(redshift)
                        mode_group.attrs["imf_mode"] = mode
                        mode_group.attrs["sample_count"] = np.int64(0)
                        mode_group.attrs["next_mass_index"] = np.int64(0)
                        for field_name in _SAMPLE_FIELDS:
                            is_index = field_name in ("mass_index", "track_index")
                            dataset = mode_group.create_dataset(
                                field_name,
                                shape=(0,),
                                maxshape=(None,),
                                chunks=(_SPOOL_CHUNK_SIZE,),
                                compression="gzip",
                                shuffle=True,
                                fletcher32=True,
                                dtype=np.int64 if is_index else np.float64,
                            )
                            dataset.attrs["units"] = _SAMPLE_UNITS[field_name]
                handle.flush()
            self._fsync_spool()
        except BaseException as error:
            try:
                self.abort()
            except BaseException as cleanup_error:
                error.add_note(f"sample spool cleanup also failed: {cleanup_error}")
            raise

    @property
    def spool_path(self) -> Path:
        return self._spool_path

    def _require_owned_spool(self) -> None:
        if self._identity is None:
            raise RuntimeError("sample spool ownership is unavailable")
        _require_spool_owner(
            self._descriptor,
            self._spool_path,
            self._identity,
        )

    def _fsync_spool(self) -> None:
        self._require_owned_spool()
        os.fsync(self._descriptor)
        self._require_owned_spool()

    def _expected_key(self) -> tuple[float, int, str]:
        if self._redshift_position >= len(self._config.redshifts):
            raise RuntimeError("sample spool already contains every configured batch")
        return (
            self._config.redshifts[self._redshift_position],
            self._mass_position,
            self._config.stellar_population.imf_modes[self._mode_position],
        )

    def _advance(self) -> None:
        self._mode_position += 1
        if self._mode_position == len(self._config.stellar_population.imf_modes):
            self._mode_position = 0
            self._mass_position += 1
            if self._mass_position == self._config.sampling.n_halo_mass_samples:
                self._mass_position = 0
                self._redshift_position += 1

    def append(self, sample: HaloSampleTable) -> None:
        if self._closed:
            raise RuntimeError("sample spool is closed")
        if type(sample) is not HaloSampleTable:
            raise TypeError("sample must be exactly HaloSampleTable")
        self._require_owned_spool()
        expected_redshift, expected_mass_index, expected_mode = self._expected_key()
        actual_mass_indices = sample.mass_index
        if (
            sample.redshift != expected_redshift
            or sample.imf_mode != expected_mode
            or actual_mass_indices.size != self._config.sampling.n_tracks_per_halo_mass
            or not np.array_equal(
                actual_mass_indices,
                np.full(actual_mass_indices.size, expected_mass_index, dtype=np.int64),
            )
            or not np.array_equal(
                sample.track_index,
                np.arange(actual_mass_indices.size, dtype=np.int64),
            )
        ):
            raise RuntimeError(
                "sample batch order/content does not match configured redshift, mass, mode, and tracks"
            )

        with os.fdopen(
            os.dup(self._descriptor), "r+b", closefd=True
        ) as file_object, h5py.File(file_object, "r+") as handle:
            mode_group = handle["samples"][_z_name(sample.redshift)][sample.imf_mode]
            old_size = int(mode_group.attrs["sample_count"])
            new_size = old_size + actual_mass_indices.size
            for field_name in _SAMPLE_FIELDS:
                dataset = mode_group[field_name]
                dataset.resize((new_size,))
                dataset[old_size:new_size] = getattr(sample, field_name)
            mode_group.attrs["sample_count"] = np.int64(new_size)
            mode_group.attrs["next_mass_index"] = np.int64(expected_mass_index + 1)
            handle.flush()
        self._fsync_spool()
        self._advance()

    def finalize(
        self,
        result: UVLFRunResult,
        *,
        overwrite: bool,
    ) -> tuple[Path, ...]:
        if self._closed:
            raise RuntimeError("sample spool is closed")
        if type(result) is not UVLFRunResult:
            raise TypeError("result must be exactly UVLFRunResult")
        if type(overwrite) is not bool:
            raise TypeError("overwrite must be exactly boolean")
        if result.config != self._config:
            raise ValueError("result config does not match sample spool config")
        if self._redshift_position != len(self._config.redshifts):
            raise RuntimeError("sample spool is missing configured batches")
        self._require_owned_spool()
        self._provenance.verify_sources()
        if self._identity is None:
            raise RuntimeError("sample spool ownership is unavailable")
        spool_snapshot = _capture_spool_snapshot(
            self._descriptor,
            self._spool_path,
            self._identity,
        )
        sample_count = (
            self._config.sampling.n_halo_mass_samples
            * self._config.sampling.n_tracks_per_halo_mass
        )
        diagnostic_by_key = {
            (item.redshift, item.imf_mode): item
            for item in result.diagnostics.mode_runs
        }
        paths: list[Path] = []
        for redshift in self._config.redshifts:
            redshift_result = result.for_redshift(redshift)
            for mode in self._config.stellar_population.imf_modes:
                mode_result = redshift_result.for_mode(mode)
                descriptor = HaloSampleDescriptor(redshift, mode, sample_count)
                shard = UVLFShard(
                    config=self._config,
                    provenance=self._provenance,
                    result=mode_result,
                    diagnostic=diagnostic_by_key[(redshift, mode)],
                    sample_descriptor=descriptor,
                    sample=None,
                )
                target = self._spool_path.parent / uvlf_shard_filename(
                    self._config,
                    redshift,
                    mode,
                )
                path = _write_uvlf_shard_from_spool_atomic(
                    shard,
                    self._descriptor,
                    self._spool_path,
                    self._identity,
                    spool_snapshot,
                    target,
                    overwrite=overwrite,
                )
                validated = read_uvlf_shard(path, load_samples=False)
                if (
                    validated.config != shard.config
                    or validated.provenance != shard.provenance
                    or validated.diagnostic != shard.diagnostic
                    or validated.sample_descriptor != shard.sample_descriptor
                    or validated.sample is not None
                    or validated.key != shard.key
                    or validated.result.imf_mode != shard.result.imf_mode
                    or any(
                        not np.array_equal(
                            getattr(validated.result, field_name),
                            getattr(shard.result, field_name),
                        )
                        for field_name in _RESULT_FIELDS
                    )
                ):
                    raise RuntimeError("published sample shard failed exact lazy readback")
                paths.append(path)
        self.abort()
        return tuple(paths)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        identity = self._identity
        self._identity = None
        descriptor = self._descriptor
        self._descriptor = -1
        cleanup_error: BaseException | None = None
        try:
            if identity is not None:
                try:
                    file_stat = os.lstat(self._spool_path)
                except FileNotFoundError:
                    pass
                else:
                    if (
                        not stat.S_ISREG(file_stat.st_mode)
                        or (file_stat.st_dev, file_stat.st_ino) != identity
                    ):
                        cleanup_error = RuntimeError(
                            "refusing to remove a replaced sample spool"
                        )
                    else:
                        self._spool_path.unlink()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if cleanup_error is not None:
            raise cleanup_error


def run_uvlf_to_sample_shards(
    config: UVLFRunConfig,
    provenance: ArtifactProvenance,
    shard_directory: Path,
    overwrite: bool = False,
) -> tuple[UVLFRunResult, tuple[Path, ...]]:
    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if type(provenance) is not ArtifactProvenance:
        raise TypeError("provenance must be exactly ArtifactProvenance")
    if not isinstance(shard_directory, Path):
        raise TypeError("shard_directory must be a Path")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    if provenance.config_sha256 != canonical_config_sha256(config):
        raise ValueError("provenance config hash does not match config")
    provenance.verify_sources()
    from auroralf.uvlf.runner import run_uvlf_streaming

    sink = _HDF5SampleSink(config, provenance, shard_directory)
    try:
        result = run_uvlf_streaming(
            config,
            _halo_sample_observer=sink.append,
        )
        paths = sink.finalize(result, overwrite=overwrite)
        return result, paths
    except BaseException as error:
        try:
            sink.abort()
        except BaseException as cleanup_error:
            error.add_note(f"sample spool cleanup also failed: {cleanup_error}")
        raise


__all__ = ["run_uvlf_to_sample_shards"]
