from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import stat as stat_module
import uuid

import h5py
import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)
from ._file_ops import (
    file_identity as _file_identity,
    sha256_open_descriptor as _hash_open_file_descriptor,
)
from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArtifactProvenance,
    HaloSampleDescriptor,
    HaloSampleTable,
    SourceChecksum,
    UVLFArtifact,
    UVLFShard,
    UVLFShardDescriptor,
    canonical_config_json,
    canonical_config_sha256,
    decode_canonical_config_json,
)


_ROOT_GROUPS = {"config", "provenance", "axes", "results", "diagnostics"}
_ROOT_ATTRS = {"schema_name", "schema_version"}
_RESULT_FIELDS = (
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
_RESULT_UNITS = {
    "bin_edges_muv": "mag",
    "bin_centers_muv": "mag",
    "bin_width_mag": "mag",
    "raw_counts": "count",
    "weighted_counts_per_mpc3": "Mpc^-3",
    "weight_squared_counts_per_mpc6": "Mpc^-6",
    "weighted_count_sigma_per_mpc3": "Mpc^-3",
    "effective_counts": "count",
    "phi_intrinsic_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_intrinsic_sigma_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_observed_per_mpc3_per_mag": "Mpc^-3 mag^-1",
    "phi_observed_sigma_per_mpc3_per_mag": "Mpc^-3 mag^-1",
}
_DIAGNOSTIC_FIELDS = (
    "sampling_seconds",
    "sample_count",
    "valid_sample_count",
    "topheavy_source_fraction",
    "popiii_source_fraction",
    "sfrd_msun_per_yr_per_mpc3",
    "popiii_sfrd_msun_per_yr_per_mpc3",
)
_DIAGNOSTIC_UNITS = {
    "sampling_seconds": "s",
    "sample_count": "count",
    "valid_sample_count": "count",
    "topheavy_source_fraction": "dimensionless",
    "popiii_source_fraction": "dimensionless",
    "sfrd_msun_per_yr_per_mpc3": "Msun yr^-1 Mpc^-3",
    "popiii_sfrd_msun_per_yr_per_mpc3": "Msun yr^-1 Mpc^-3",
}
_SAMPLE_FIELDS = (
    "mass_index",
    "track_index",
    "halo_mass_msun",
    "mass_weight_per_mpc3",
    "uv_luminosity_erg_per_s_hz",
    "muv",
    "sfr_msun_per_yr",
    "popiii_sfr_msun_per_yr",
)
_SAMPLE_UNITS = {
    "mass_index": "index",
    "track_index": "index",
    "halo_mass_msun": "Msun",
    "mass_weight_per_mpc3": "Mpc^-3",
    "uv_luminosity_erg_per_s_hz": "erg s^-1 Hz^-1",
    "muv": "mag",
    "sfr_msun_per_yr": "Msun yr^-1",
    "popiii_sfr_msun_per_yr": "Msun yr^-1",
}
_MARKER_KEYS = {
    "schema_name",
    "schema_version",
    "config_sha256",
    "artifact_sha256",
    "size_bytes",
}


@dataclass(slots=True)
class _OwnedFile:
    path: Path
    descriptor: int
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _SpoolSnapshot:
    file_identity: tuple[int, int, int, int, int]
    sha256: str


def _z_name(redshift: float) -> str:
    return f"z={redshift:.17g}"


def _string_dtype() -> np.dtype:
    return h5py.string_dtype(encoding="utf-8")


def _write_string(group: h5py.Group, name: str, value: str) -> h5py.Dataset:
    return group.create_dataset(name, data=value, dtype=_string_dtype())


def _write_array(
    group: h5py.Group,
    name: str,
    value: np.ndarray,
    *,
    units: str,
) -> h5py.Dataset:
    dtype = np.int64 if name == "raw_counts" else np.float64
    dataset = group.create_dataset(name, data=np.asarray(value, dtype=dtype), dtype=dtype)
    dataset.attrs["units"] = units
    return dataset


def _write_config_provenance_axes(
    handle: h5py.File,
    config: UVLFRunConfig,
    provenance: ArtifactProvenance,
    *,
    artifact_kind: str | None,
) -> None:
    handle.attrs["schema_name"] = SCHEMA_NAME
    handle.attrs["schema_version"] = SCHEMA_VERSION
    if artifact_kind is not None:
        handle.attrs["artifact_kind"] = artifact_kind
    config_json = canonical_config_json(config)
    config_hash = canonical_config_sha256(config)
    config_group = handle.create_group("config", track_order=True)
    _write_string(config_group, "canonical_json", config_json)
    _write_string(config_group, "sha256", config_hash)

    provenance_group = handle.create_group("provenance", track_order=True)
    _write_string(provenance_group, "config_sha256", provenance.config_sha256)
    _write_string(provenance_group, "code_revision", provenance.code_revision)
    provenance_group.create_dataset(
        "code_dirty", data=provenance.code_dirty, dtype=np.bool_
    )
    _write_string(provenance_group, "seed_namespace", provenance.seed_namespace)
    _write_string(provenance_group, "created_utc", provenance.created_utc)
    sources_group = provenance_group.create_group("sources", track_order=True)
    for source in provenance.source_checksums:
        source_group = sources_group.create_group(source.label, track_order=True)
        _write_string(source_group, "path", str(source.path))
        _write_string(source_group, "sha256", source.sha256)
        source_group.create_dataset(
            "size_bytes", data=source.size_bytes, dtype=np.int64
        )

    axes_group = handle.create_group("axes", track_order=True)
    redshifts = axes_group.create_dataset(
        "redshifts",
        data=np.asarray(config.redshifts, dtype=np.float64),
        dtype=np.float64,
    )
    redshifts.attrs["units"] = "dimensionless"
    axes_group.create_dataset(
        "imf_modes",
        data=np.asarray(config.stellar_population.imf_modes, dtype=object),
        dtype=_string_dtype(),
    )
    edges = axes_group.create_dataset(
        "muv_bin_edges",
        data=np.asarray(config.sampling.muv_bin_edges, dtype=np.float64),
        dtype=np.float64,
    )
    edges.attrs["units"] = "mag"


def _write_result_mode(
    results_group: h5py.Group,
    redshift: float,
    result: IMFModeResult,
) -> None:
    z_name = _z_name(redshift)
    z_group = (
        results_group[z_name]
        if z_name in results_group
        else results_group.create_group(z_name, track_order=True)
    )
    mode_group = z_group.create_group(result.imf_mode, track_order=True)
    mode_group.attrs["redshift"] = np.float64(redshift)
    mode_group.attrs["imf_mode"] = result.imf_mode
    for field_name in _RESULT_FIELDS:
        _write_array(
            mode_group,
            field_name,
            getattr(result, field_name),
            units=_RESULT_UNITS[field_name],
        )


def _write_diagnostic_mode(
    diagnostics_group: h5py.Group,
    diagnostic: ModeRunDiagnostics,
) -> None:
    z_name = _z_name(diagnostic.redshift)
    z_group = (
        diagnostics_group[z_name]
        if z_name in diagnostics_group
        else diagnostics_group.create_group(z_name, track_order=True)
    )
    mode_group = z_group.create_group(diagnostic.imf_mode, track_order=True)
    mode_group.attrs["redshift"] = np.float64(diagnostic.redshift)
    mode_group.attrs["imf_mode"] = diagnostic.imf_mode
    for field_name in _DIAGNOSTIC_FIELDS:
        is_count = field_name in ("sample_count", "valid_sample_count")
        dataset = mode_group.create_dataset(
            field_name,
            data=(
                np.int64(getattr(diagnostic, field_name))
                if is_count
                else np.float64(getattr(diagnostic, field_name))
            ),
            dtype=np.int64 if is_count else np.float64,
        )
        dataset.attrs["units"] = _DIAGNOSTIC_UNITS[field_name]


def _write_sample_table(
    samples_group: h5py.Group,
    sample: HaloSampleTable,
) -> None:
    z_name = _z_name(sample.redshift)
    z_group = (
        samples_group[z_name]
        if z_name in samples_group
        else samples_group.create_group(z_name, track_order=True)
    )
    mode_group = z_group.create_group(sample.imf_mode, track_order=True)
    mode_group.attrs["redshift"] = np.float64(sample.redshift)
    mode_group.attrs["imf_mode"] = sample.imf_mode
    for field_name in _SAMPLE_FIELDS:
        is_index = field_name in ("mass_index", "track_index")
        dataset = mode_group.create_dataset(
            field_name,
            data=np.asarray(
                getattr(sample, field_name),
                dtype=np.int64 if is_index else np.float64,
            ),
            dtype=np.int64 if is_index else np.float64,
            chunks=True,
            maxshape=(None,),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        dataset.attrs["units"] = _SAMPLE_UNITS[field_name]


def _write_hdf5_file(artifact: UVLFArtifact, owned_file: _OwnedFile) -> None:
    if type(artifact) is not UVLFArtifact:
        raise TypeError("artifact must be exactly UVLFArtifact")
    if artifact.sample_descriptors and len(artifact.samples) != len(
        artifact.sample_descriptors
    ):
        raise ValueError("writing samples requires one loaded table for every sample key")
    config = artifact.result.config
    if artifact.provenance.config_sha256 != canonical_config_sha256(config):
        raise ValueError("provenance config hash does not match canonical config")
    artifact.provenance.verify_sources()
    for redshift_result in artifact.result.redshifts:
        for mode_result in redshift_result.imf_modes:
            if mode_result.halo_tracks:
                raise ValueError("halo_tracks persistence is not supported by schema v2.0.0")

    _require_owned_path(owned_file, label="temporary artifact")
    with os.fdopen(
        os.dup(owned_file.descriptor), "r+b", closefd=True
    ) as file_object, h5py.File(file_object, "w") as handle:
        _write_config_provenance_axes(
            handle,
            config,
            artifact.provenance,
            artifact_kind=None,
        )
        results_group = handle.create_group("results", track_order=True)
        for redshift_result in artifact.result.redshifts:
            for result in redshift_result.imf_modes:
                _write_result_mode(results_group, redshift_result.redshift, result)

        diagnostics_group = handle.create_group("diagnostics", track_order=True)
        total = diagnostics_group.create_dataset(
            "total_seconds",
            data=np.float64(artifact.result.diagnostics.total_seconds),
            dtype=np.float64,
        )
        total.attrs["units"] = "s"
        for diagnostic in artifact.result.diagnostics.mode_runs:
            _write_diagnostic_mode(diagnostics_group, diagnostic)

        if artifact.samples:
            samples_group = handle.create_group("samples", track_order=True)
            for sample in artifact.samples:
                _write_sample_table(samples_group, sample)
        handle.flush()
    _require_owned_path(owned_file, label="temporary artifact")


def _write_uvlf_shard_hdf5_file(
    shard: UVLFShard,
    owned_file: _OwnedFile,
) -> None:
    if type(shard) is not UVLFShard:
        raise TypeError("shard must be exactly UVLFShard")
    if shard.sample_descriptor is not None and shard.sample is None:
        raise ValueError("writing a shard sample requires its loaded sample table")
    if shard.result.halo_tracks:
        raise ValueError("halo_tracks persistence is not supported by schema v2.0.0")
    if shard.provenance.config_sha256 != canonical_config_sha256(shard.config):
        raise ValueError("provenance config hash does not match canonical config")
    shard.provenance.verify_sources()

    _require_owned_path(owned_file, label="temporary shard")
    with os.fdopen(
        os.dup(owned_file.descriptor), "r+b", closefd=True
    ) as file_object, h5py.File(file_object, "w") as handle:
        _write_config_provenance_axes(
            handle,
            shard.config,
            shard.provenance,
            artifact_kind="shard",
        )
        results_group = handle.create_group("results", track_order=True)
        _write_result_mode(results_group, shard.key[0], shard.result)
        diagnostics_group = handle.create_group("diagnostics", track_order=True)
        _write_diagnostic_mode(diagnostics_group, shard.diagnostic)
        if shard.sample is not None:
            samples_group = handle.create_group("samples", track_order=True)
            _write_sample_table(samples_group, shard.sample)
        handle.flush()
    _require_owned_path(owned_file, label="temporary shard")


def _require_exact_names(container: object, expected: set[str], *, name: str) -> None:
    actual = set(container.keys())
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"unknown {name} object: {unknown[0]}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing required {name} object: {missing[0]}")


def _require_exact_attrs(container: object, expected: set[str], *, name: str) -> None:
    actual = set(container.attrs.keys())
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"unknown {name} attribute: {unknown[0]}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing required {name} attribute: {missing[0]}")


def _reject_links_and_hard_link_aliases(handle: h5py.File) -> None:
    seen_addresses: dict[int, str] = {
        int(h5py.h5o.get_info(handle.id).addr): "/"
    }

    def visit(group: h5py.Group, parent_path: str) -> None:
        for name in group.keys():
            child_path = f"/{name}" if parent_path == "/" else f"{parent_path}/{name}"
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                raise ValueError(f"SoftLink is forbidden in artifact: {child_path}")
            if isinstance(link, h5py.ExternalLink):
                raise ValueError(f"ExternalLink is forbidden in artifact: {child_path}")
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"unsupported HDF5 link type at {child_path}")
            child = group[name]
            address = int(h5py.h5o.get_info(child.id).addr)
            if address in seen_addresses:
                raise ValueError(
                    "hard link alias exposes one HDF5 object at multiple paths: "
                    f"{seen_addresses[address]} and {child_path}"
                )
            seen_addresses[address] = child_path
            if isinstance(child, h5py.Group):
                visit(child, child_path)

    visit(handle, "/")


def _read_string(dataset: h5py.Dataset, *, name: str) -> str:
    if dataset.shape != ():
        raise ValueError(f"{name} must be a scalar UTF-8 dataset")
    _require_variable_utf8_dtype(dataset.dtype, name=name)
    _require_exact_attrs(dataset, set(), name=name)
    value = dataset.asstr()[()]
    if type(value) is not str:
        raise TypeError(f"{name} must decode to a string")
    return value


def _require_variable_utf8_dtype(dtype: np.dtype, *, name: str) -> None:
    string_info = h5py.check_string_dtype(dtype)
    if (
        string_info is None
        or string_info.encoding != "utf-8"
        or string_info.length is not None
    ):
        raise ValueError(f"{name} must use variable-length UTF-8 string dtype")


def _read_utf8_attr(container: object, attr_name: str, *, name: str) -> str:
    attribute = container.attrs.get_id(attr_name)
    if attribute.shape != ():
        raise ValueError(f"{name}.{attr_name} must be a scalar UTF-8 attribute")
    _require_variable_utf8_dtype(attribute.dtype, name=f"{name}.{attr_name}")
    value = container.attrs[attr_name]
    if type(value) is not str:
        raise ValueError(f"{name}.{attr_name} must decode to a strict UTF-8 string")
    return value


def _read_float64_attr(container: object, attr_name: str, *, name: str) -> float:
    attribute = container.attrs.get_id(attr_name)
    if attribute.shape != () or attribute.dtype != np.dtype(np.float64):
        raise ValueError(f"{name}.{attr_name} must be an exact scalar float64 attribute")
    value = container.attrs[attr_name]
    if type(value) is not np.float64:
        raise ValueError(f"{name}.{attr_name} must be an exact np.float64 scalar")
    return float(value)


def _read_numeric(
    dataset: h5py.Dataset,
    *,
    name: str,
    dtype: np.dtype,
    units: str,
    scalar: bool,
    expected_shape: tuple[int, ...] | None = None,
) -> object:
    _validate_numeric_dataset(
        dataset,
        name=name,
        dtype=dtype,
        units=units,
        scalar=scalar,
        expected_shape=expected_shape,
    )
    return dataset[()]


def _validate_numeric_dataset(
    dataset: h5py.Dataset,
    *,
    name: str,
    dtype: np.dtype,
    units: str,
    scalar: bool,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    if dataset.dtype != np.dtype(dtype):
        raise ValueError(f"{name} has wrong dtype; expected {np.dtype(dtype)}")
    if scalar and dataset.shape != ():
        raise ValueError(f"{name} must be scalar")
    if not scalar and (len(dataset.shape) != 1 or dataset.shape[0] == 0):
        raise ValueError(f"{name} must be a non-empty 1D dataset")
    if expected_shape is not None and dataset.shape != expected_shape:
        raise ValueError(
            f"{name} has wrong shape; expected {expected_shape}, got {dataset.shape}"
        )
    _require_exact_attrs(dataset, {"units"}, name=name)
    if _read_utf8_attr(dataset, "units", name=name) != units:
        raise ValueError(f"{name} has wrong units")


def _read_provenance(group: h5py.Group) -> ArtifactProvenance:
    _require_exact_attrs(group, set(), name="provenance")
    _require_exact_names(
        group,
        {
            "config_sha256",
            "code_revision",
            "code_dirty",
            "seed_namespace",
            "created_utc",
            "sources",
        },
        name="provenance",
    )
    dirty = group["code_dirty"]
    if dirty.shape != () or dirty.dtype != np.dtype(bool):
        raise ValueError("provenance.code_dirty must be a scalar bool dataset")
    _require_exact_attrs(dirty, set(), name="provenance.code_dirty")
    sources_group = group["sources"]
    _require_exact_attrs(sources_group, set(), name="provenance.sources")
    sources: list[SourceChecksum] = []
    for label in sources_group:
        source_group = sources_group[label]
        _require_exact_attrs(source_group, set(), name=f"source {label}")
        _require_exact_names(
            source_group, {"path", "sha256", "size_bytes"}, name=f"source {label}"
        )
        size_dataset = source_group["size_bytes"]
        if size_dataset.shape != () or size_dataset.dtype != np.dtype(np.int64):
            raise ValueError(f"source {label}.size_bytes must be scalar int64")
        _require_exact_attrs(size_dataset, set(), name=f"source {label}.size_bytes")
        sources.append(
            SourceChecksum(
                label=label,
                path=Path(_read_string(source_group["path"], name=f"source {label}.path")),
                sha256=_read_string(
                    source_group["sha256"], name=f"source {label}.sha256"
                ),
                size_bytes=int(size_dataset[()]),
            )
        )
    provenance = ArtifactProvenance(
        config_sha256=_read_string(
            group["config_sha256"], name="provenance.config_sha256"
        ),
        code_revision=_read_string(
            group["code_revision"], name="provenance.code_revision"
        ),
        code_dirty=bool(dirty[()]),
        seed_namespace=_read_string(
            group["seed_namespace"], name="provenance.seed_namespace"
        ),
        created_utc=_read_string(group["created_utc"], name="provenance.created_utc"),
        source_checksums=tuple(sources),
    )
    provenance.verify_sources()
    return provenance


def _validate_axis_group(group: h5py.Group, config: object) -> None:
    _require_exact_attrs(group, set(), name="axes")
    _require_exact_names(group, {"redshifts", "imf_modes", "muv_bin_edges"}, name="axes")
    redshifts = np.asarray(
        _read_numeric(
            group["redshifts"],
            name="axes.redshifts",
            dtype=np.float64,
            units="dimensionless",
            scalar=False,
            expected_shape=(len(config.redshifts),),
        )
    )
    edges = np.asarray(
        _read_numeric(
            group["muv_bin_edges"],
            name="axes.muv_bin_edges",
            dtype=np.float64,
            units="mag",
            scalar=False,
            expected_shape=(len(config.sampling.muv_bin_edges),),
        )
    )
    modes_dataset = group["imf_modes"]
    if modes_dataset.shape != (len(config.stellar_population.imf_modes),):
        raise ValueError("axes.imf_modes must be a 1D UTF-8 dataset")
    _require_variable_utf8_dtype(modes_dataset.dtype, name="axes.imf_modes")
    _require_exact_attrs(modes_dataset, set(), name="axes.imf_modes")
    modes = tuple(modes_dataset.asstr()[()].tolist())
    if not np.array_equal(redshifts, np.asarray(config.redshifts, dtype=float)):
        raise ValueError("axes.redshifts does not exactly match config")
    if modes != config.stellar_population.imf_modes:
        raise ValueError("axes.imf_modes does not exactly match config")
    if not np.array_equal(edges, np.asarray(config.sampling.muv_bin_edges, dtype=float)):
        raise ValueError("axes.muv_bin_edges does not exactly match config")


def _read_results(group: h5py.Group, config: object) -> tuple[RedshiftResult, ...]:
    _require_exact_attrs(group, set(), name="results")
    _require_exact_names(
        group, {_z_name(redshift) for redshift in config.redshifts}, name="results"
    )
    redshift_results: list[RedshiftResult] = []
    bin_count = len(config.sampling.muv_bin_edges) - 1
    for redshift in config.redshifts:
        z_group = group[_z_name(redshift)]
        _require_exact_attrs(z_group, set(), name=f"results/{_z_name(redshift)}")
        _require_exact_names(
            z_group, set(config.stellar_population.imf_modes), name="result modes"
        )
        modes: list[IMFModeResult] = []
        for mode in config.stellar_population.imf_modes:
            mode_group = z_group[mode]
            _require_exact_attrs(mode_group, {"redshift", "imf_mode"}, name="result mode")
            stored_redshift = _read_float64_attr(
                mode_group,
                "redshift",
                name="result mode",
            )
            stored_mode = _read_utf8_attr(
                mode_group,
                "imf_mode",
                name="result mode",
            )
            if stored_redshift != redshift or stored_mode != mode:
                raise ValueError("result mode attributes do not match axes")
            _require_exact_names(mode_group, set(_RESULT_FIELDS), name="result mode")
            values: dict[str, object] = {"imf_mode": mode, "halo_tracks": ()}
            for field_name in _RESULT_FIELDS:
                values[field_name] = np.asarray(
                    _read_numeric(
                        mode_group[field_name],
                        name=f"result.{field_name}",
                        dtype=np.int64 if field_name == "raw_counts" else np.float64,
                        units=_RESULT_UNITS[field_name],
                        scalar=False,
                        expected_shape=(
                            (bin_count + 1,)
                            if field_name == "bin_edges_muv"
                            else (bin_count,)
                        ),
                    )
                )
            modes.append(IMFModeResult(**values))
        redshift_results.append(RedshiftResult(redshift=redshift, imf_modes=tuple(modes)))
    return tuple(redshift_results)


def _read_diagnostics(group: h5py.Group, config: object) -> RunDiagnostics:
    _require_exact_attrs(group, set(), name="diagnostics")
    expected = {"total_seconds"} | {_z_name(redshift) for redshift in config.redshifts}
    _require_exact_names(group, expected, name="diagnostics")
    total = float(
        _read_numeric(
            group["total_seconds"],
            name="diagnostics.total_seconds",
            dtype=np.float64,
            units="s",
            scalar=True,
        )
    )
    mode_runs: list[ModeRunDiagnostics] = []
    for redshift in config.redshifts:
        z_group = group[_z_name(redshift)]
        _require_exact_attrs(z_group, set(), name="diagnostic redshift")
        _require_exact_names(
            z_group, set(config.stellar_population.imf_modes), name="diagnostic modes"
        )
        for mode in config.stellar_population.imf_modes:
            mode_group = z_group[mode]
            _require_exact_attrs(
                mode_group, {"redshift", "imf_mode"}, name="diagnostic mode"
            )
            stored_redshift = _read_float64_attr(
                mode_group,
                "redshift",
                name="diagnostic mode",
            )
            stored_mode = _read_utf8_attr(
                mode_group,
                "imf_mode",
                name="diagnostic mode",
            )
            if stored_redshift != redshift or stored_mode != mode:
                raise ValueError("diagnostic mode attributes do not match axes")
            _require_exact_names(mode_group, set(_DIAGNOSTIC_FIELDS), name="diagnostic mode")
            values: dict[str, object] = {"redshift": redshift, "imf_mode": mode}
            for field_name in _DIAGNOSTIC_FIELDS:
                is_count = field_name in ("sample_count", "valid_sample_count")
                raw = _read_numeric(
                    mode_group[field_name],
                    name=f"diagnostic.{field_name}",
                    dtype=np.int64 if is_count else np.float64,
                    units=_DIAGNOSTIC_UNITS[field_name],
                    scalar=True,
                )
                values[field_name] = int(raw) if is_count else float(raw)
            mode_runs.append(ModeRunDiagnostics(**values))
    return RunDiagnostics(total_seconds=total, mode_runs=tuple(mode_runs))


def _read_samples(
    group: h5py.Group | None,
    config: object,
    *,
    load_samples: bool,
) -> tuple[tuple[HaloSampleDescriptor, ...], tuple[HaloSampleTable, ...]]:
    if group is None:
        return (), ()
    _require_exact_attrs(group, set(), name="samples")
    allowed_z = {_z_name(redshift) for redshift in config.redshifts}
    unknown_z = set(group) - allowed_z
    if unknown_z:
        raise ValueError(f"unknown samples redshift group: {sorted(unknown_z)[0]}")
    descriptors: list[HaloSampleDescriptor] = []
    samples: list[HaloSampleTable] = []
    for redshift in config.redshifts:
        z_name = _z_name(redshift)
        if z_name not in group:
            continue
        z_group = group[z_name]
        _require_exact_attrs(z_group, set(), name="samples redshift")
        unknown_modes = set(z_group) - set(config.stellar_population.imf_modes)
        if unknown_modes:
            raise ValueError(f"unknown samples mode group: {sorted(unknown_modes)[0]}")
        for mode in config.stellar_population.imf_modes:
            if mode not in z_group:
                continue
            mode_group = z_group[mode]
            _require_exact_attrs(mode_group, {"redshift", "imf_mode"}, name="sample mode")
            stored_redshift = _read_float64_attr(
                mode_group,
                "redshift",
                name="sample mode",
            )
            stored_mode = _read_utf8_attr(
                mode_group,
                "imf_mode",
                name="sample mode",
            )
            if stored_redshift != redshift or stored_mode != mode:
                raise ValueError("sample mode attributes do not match axes")
            _require_exact_names(mode_group, set(_SAMPLE_FIELDS), name="sample mode")
            datasets: dict[str, h5py.Dataset] = {}
            lengths: dict[str, int] = {}
            chunk_lengths: list[int] = []
            for field_name in _SAMPLE_FIELDS:
                dataset = mode_group[field_name]
                is_index = field_name in ("mass_index", "track_index")
                _validate_numeric_dataset(
                    dataset,
                    name=f"sample.{field_name}",
                    dtype=np.int64 if is_index else np.float64,
                    units=_SAMPLE_UNITS[field_name],
                    scalar=False,
                )
                if dataset.chunks is None or dataset.maxshape != (None,):
                    raise ValueError(f"sample.{field_name} must be chunked and extensible")
                if dataset.compression is None or not dataset.shuffle or not dataset.fletcher32:
                    raise ValueError(f"sample.{field_name} storage filters are incomplete")
                datasets[field_name] = dataset
                lengths[field_name] = dataset.shape[0]
                chunk_lengths.append(dataset.chunks[0])
            sample_count = lengths[_SAMPLE_FIELDS[0]]
            if any(length != sample_count for length in lengths.values()):
                raise ValueError("all sample datasets must have the same length")
            descriptor = HaloSampleDescriptor(redshift, mode, sample_count)
            validation_chunk = min(65_536, *chunk_lengths)
            for start in range(0, sample_count, validation_chunk):
                stop = min(start + validation_chunk, sample_count)
                sample_slice = slice(start, stop)
                chunk_values = {
                    field_name: np.asarray(dataset[sample_slice])
                    for field_name, dataset in datasets.items()
                }
                HaloSampleTable(
                    redshift=redshift,
                    imf_mode=mode,
                    **chunk_values,
                )
                del chunk_values
            descriptors.append(descriptor)
            if load_samples:
                values = {
                    field_name: np.asarray(dataset[:])
                    for field_name, dataset in datasets.items()
                }
                samples.append(
                    HaloSampleTable(
                        redshift=redshift,
                        imf_mode=mode,
                        **values,
                    )
                )
    if not descriptors:
        raise ValueError("samples group must contain at least one sample table")
    return tuple(descriptors), tuple(samples)


def _read_uvlf_artifact_handle(
    handle: h5py.File,
    *,
    load_samples: bool,
) -> UVLFArtifact:
    _reject_links_and_hard_link_aliases(handle)
    _require_exact_attrs(handle, _ROOT_ATTRS, name="root")
    root_schema_name = _read_utf8_attr(handle, "schema_name", name="root")
    root_schema_version = _read_utf8_attr(handle, "schema_version", name="root")
    if root_schema_name != SCHEMA_NAME:
        raise ValueError("root schema_name is unsupported")
    if root_schema_version != SCHEMA_VERSION:
        raise ValueError("root schema_version is unsupported")
    expected_groups = set(_ROOT_GROUPS)
    if "samples" in handle:
        expected_groups.add("samples")
    _require_exact_names(handle, expected_groups, name="root")

    config_group = handle["config"]
    _require_exact_attrs(config_group, set(), name="config")
    _require_exact_names(config_group, {"canonical_json", "sha256"}, name="config")
    config_json = _read_string(config_group["canonical_json"], name="config.canonical_json")
    stored_hash = _read_string(config_group["sha256"], name="config.sha256")
    config = decode_canonical_config_json(config_json)
    actual_hash = canonical_config_sha256(config)
    if stored_hash != actual_hash:
        raise ValueError("config hash mismatch")
    provenance = _read_provenance(handle["provenance"])
    if provenance.config_sha256 != actual_hash:
        raise ValueError("provenance config hash mismatch")
    _validate_axis_group(handle["axes"], config)
    redshifts = _read_results(handle["results"], config)
    diagnostics = _read_diagnostics(handle["diagnostics"], config)
    result = UVLFRunResult(config=config, redshifts=redshifts, diagnostics=diagnostics)
    sample_descriptors, samples = _read_samples(
        handle["samples"] if "samples" in handle else None,
        config,
        load_samples=load_samples,
    )
    return UVLFArtifact(
        result=result,
        provenance=provenance,
        sample_descriptors=sample_descriptors,
        samples=samples,
    )


def _read_uvlf_shard_handle(
    handle: h5py.File,
    *,
    load_samples: bool,
) -> UVLFShard:
    _reject_links_and_hard_link_aliases(handle)
    _require_exact_attrs(
        handle,
        _ROOT_ATTRS | {"artifact_kind"},
        name="root",
    )
    if _read_utf8_attr(handle, "schema_name", name="root") != SCHEMA_NAME:
        raise ValueError("root schema_name is unsupported")
    if _read_utf8_attr(handle, "schema_version", name="root") != SCHEMA_VERSION:
        raise ValueError("root schema_version is unsupported")
    if _read_utf8_attr(handle, "artifact_kind", name="root") != "shard":
        raise ValueError("root artifact_kind must equal 'shard'")
    expected_groups = set(_ROOT_GROUPS)
    if "samples" in handle:
        expected_groups.add("samples")
    _require_exact_names(handle, expected_groups, name="root")

    config_group = handle["config"]
    _require_exact_attrs(config_group, set(), name="config")
    _require_exact_names(config_group, {"canonical_json", "sha256"}, name="config")
    config_json = _read_string(
        config_group["canonical_json"], name="config.canonical_json"
    )
    stored_hash = _read_string(config_group["sha256"], name="config.sha256")
    config = decode_canonical_config_json(config_json)
    actual_hash = canonical_config_sha256(config)
    if stored_hash != actual_hash:
        raise ValueError("config hash mismatch")
    provenance = _read_provenance(handle["provenance"])
    if provenance.config_sha256 != actual_hash:
        raise ValueError("provenance config hash mismatch")
    _validate_axis_group(handle["axes"], config)

    results_group = handle["results"]
    _require_exact_attrs(results_group, set(), name="results")
    if len(results_group) != 1:
        raise ValueError("shard results must contain exactly one redshift")
    z_name = next(iter(results_group))
    redshift_by_name = {_z_name(value): value for value in config.redshifts}
    if z_name not in redshift_by_name:
        raise ValueError("shard result redshift is not on configured axes")
    redshift = redshift_by_name[z_name]
    result_z_group = results_group[z_name]
    _require_exact_attrs(result_z_group, set(), name="shard result redshift")
    if len(result_z_group) != 1:
        raise ValueError("shard results must contain exactly one mode")
    mode = next(iter(result_z_group))
    if mode not in config.stellar_population.imf_modes:
        raise ValueError("shard result mode is not on configured axes")
    result_group = result_z_group[mode]
    _require_exact_attrs(result_group, {"redshift", "imf_mode"}, name="result mode")
    if (
        _read_float64_attr(result_group, "redshift", name="result mode") != redshift
        or _read_utf8_attr(result_group, "imf_mode", name="result mode") != mode
    ):
        raise ValueError("result mode attributes do not match shard key")
    _require_exact_names(result_group, set(_RESULT_FIELDS), name="result mode")
    bin_count = len(config.sampling.muv_bin_edges) - 1
    result_values: dict[str, object] = {"imf_mode": mode, "halo_tracks": ()}
    for field_name in _RESULT_FIELDS:
        result_values[field_name] = np.asarray(
            _read_numeric(
                result_group[field_name],
                name=f"result.{field_name}",
                dtype=np.int64 if field_name == "raw_counts" else np.float64,
                units=_RESULT_UNITS[field_name],
                scalar=False,
                expected_shape=(
                    (bin_count + 1,)
                    if field_name == "bin_edges_muv"
                    else (bin_count,)
                ),
            )
        )
    result = IMFModeResult(**result_values)

    diagnostics_group = handle["diagnostics"]
    _require_exact_attrs(diagnostics_group, set(), name="diagnostics")
    _require_exact_names(diagnostics_group, {z_name}, name="shard diagnostics")
    diagnostic_z_group = diagnostics_group[z_name]
    _require_exact_attrs(
        diagnostic_z_group,
        set(),
        name="shard diagnostic redshift",
    )
    _require_exact_names(diagnostic_z_group, {mode}, name="shard diagnostic modes")
    diagnostic_group = diagnostic_z_group[mode]
    _require_exact_attrs(
        diagnostic_group,
        {"redshift", "imf_mode"},
        name="diagnostic mode",
    )
    if (
        _read_float64_attr(diagnostic_group, "redshift", name="diagnostic mode")
        != redshift
        or _read_utf8_attr(diagnostic_group, "imf_mode", name="diagnostic mode")
        != mode
    ):
        raise ValueError("diagnostic mode attributes do not match shard key")
    _require_exact_names(
        diagnostic_group,
        set(_DIAGNOSTIC_FIELDS),
        name="diagnostic mode",
    )
    diagnostic_values: dict[str, object] = {
        "redshift": redshift,
        "imf_mode": mode,
    }
    for field_name in _DIAGNOSTIC_FIELDS:
        is_count = field_name in ("sample_count", "valid_sample_count")
        raw = _read_numeric(
            diagnostic_group[field_name],
            name=f"diagnostic.{field_name}",
            dtype=np.int64 if is_count else np.float64,
            units=_DIAGNOSTIC_UNITS[field_name],
            scalar=True,
        )
        diagnostic_values[field_name] = int(raw) if is_count else float(raw)
    diagnostic = ModeRunDiagnostics(**diagnostic_values)

    sample_descriptors, samples = _read_samples(
        handle["samples"] if "samples" in handle else None,
        config,
        load_samples=load_samples,
    )
    if sample_descriptors and (
        len(sample_descriptors) != 1 or sample_descriptors[0].key != (redshift, mode)
    ):
        raise ValueError("shard samples must contain only the shard key")
    return UVLFShard(
        config=config,
        provenance=provenance,
        result=result,
        diagnostic=diagnostic,
        sample_descriptor=(sample_descriptors[0] if sample_descriptors else None),
        sample=(samples[0] if samples else None),
    )


def _require_spool_owner(
    descriptor: int,
    path: Path,
    owner: tuple[int, int],
) -> os.stat_result:
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("sample spool descriptor must be an open non-negative integer")
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("sample spool path must be an absolute Path")
    if (
        type(owner) is not tuple
        or len(owner) != 2
        or any(type(value) is not int for value in owner)
    ):
        raise TypeError("sample spool owner must be an exact (st_dev, st_ino) tuple")
    descriptor_stat = os.fstat(descriptor)
    if not stat_module.S_ISREG(descriptor_stat.st_mode):
        raise ValueError("sample spool descriptor is not a regular file")
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError("sample spool path identity changed") from error
    if not stat_module.S_ISREG(path_stat.st_mode):
        raise ValueError("sample spool path is not a regular file")
    if (
        (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != owner
        or (path_stat.st_dev, path_stat.st_ino) != owner
    ):
        raise ValueError("sample spool path identity changed")
    return descriptor_stat


def _capture_spool_snapshot(
    descriptor: int,
    path: Path,
    owner: tuple[int, int],
) -> _SpoolSnapshot:
    before = _require_spool_owner(descriptor, path, owner)
    digest = _hash_open_file_descriptor(descriptor)
    after = _require_spool_owner(descriptor, path, owner)
    if _file_identity(before) != _file_identity(after):
        raise ValueError("sample spool changed while its snapshot was captured")
    return _SpoolSnapshot(file_identity=_file_identity(after), sha256=digest)


def _require_spool_snapshot(
    descriptor: int,
    path: Path,
    owner: tuple[int, int],
    snapshot: _SpoolSnapshot,
) -> None:
    if type(snapshot) is not _SpoolSnapshot:
        raise TypeError("snapshot must be exactly _SpoolSnapshot")
    before = _require_spool_owner(descriptor, path, owner)
    if _file_identity(before) != snapshot.file_identity:
        raise ValueError("sample spool file identity changed")
    digest = _hash_open_file_descriptor(descriptor)
    after = _require_spool_owner(descriptor, path, owner)
    if _file_identity(after) != snapshot.file_identity:
        raise ValueError("sample spool file identity changed")
    if digest != snapshot.sha256:
        raise ValueError("sample spool content changed")


def _read_hdf5_from_open_descriptor(
    descriptor: int,
    *,
    load_samples: bool,
) -> UVLFArtifact:
    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as file_object:
        with h5py.File(file_object, "r") as handle:
            return _read_uvlf_artifact_handle(handle, load_samples=load_samples)


def _read_uvlf_shard_from_open_descriptor(
    descriptor: int,
    *,
    load_samples: bool,
) -> UVLFShard:
    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as file_object:
        with h5py.File(file_object, "r") as handle:
            return _read_uvlf_shard_handle(handle, load_samples=load_samples)


def _require_open_file_stable(
    descriptor: int,
    path: Path,
    before: os.stat_result,
    digest: str,
    *,
    label: str,
) -> None:
    after = os.fstat(descriptor)
    current = path.stat()
    if not (
        _file_identity(before)
        == _file_identity(after)
        == _file_identity(current)
    ):
        raise ValueError(f"{label} changed identity while being read: {path}")
    if _hash_open_file_descriptor(descriptor) != digest:
        raise ValueError(f"{label} changed content while being read: {path}")


def _read_uvlf_artifact_file(path: str | Path, *, load_samples: bool) -> UVLFArtifact:
    resolved = Path(path).expanduser().resolve(strict=True)
    descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        digest = _hash_open_file_descriptor(descriptor)
        artifact = _read_hdf5_from_open_descriptor(
            descriptor,
            load_samples=load_samples,
        )
        _require_open_file_stable(
            descriptor,
            resolved,
            before,
            digest,
            label="artifact",
        )
        artifact.provenance.verify_sources()
        return artifact
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        digest = _hash_open_file_descriptor(descriptor)
        _require_open_file_stable(
            descriptor,
            path,
            before,
            digest,
            label="artifact",
        )
        return digest
    finally:
        os.close(descriptor)


def _marker_path(path: Path) -> Path:
    return path.with_name(path.name + ".complete")


def _canonical_marker_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fsync_file(owned_file: _OwnedFile) -> None:
    os.fsync(owned_file.descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_owned_file(path: Path, *, mode: int) -> _OwnedFile:
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        file_stat = os.fstat(descriptor)
        os.fchmod(descriptor, mode)
        return _OwnedFile(
            path=path,
            descriptor=descriptor,
            identity=(file_stat.st_dev, file_stat.st_ino),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _close_owned_file(owned_file: _OwnedFile | None) -> None:
    if owned_file is not None and owned_file.descriptor >= 0:
        os.close(owned_file.descriptor)
        owned_file.descriptor = -1


def _require_owned_path(owned_file: _OwnedFile, *, label: str) -> None:
    if owned_file.descriptor < 0:
        raise ValueError(f"{label} ownership descriptor is closed")
    descriptor_stat = os.fstat(owned_file.descriptor)
    try:
        path_stat = os.stat(owned_file.path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} changed identity after exclusive creation") from error
    if not (
        (descriptor_stat.st_dev, descriptor_stat.st_ino)
        == owned_file.identity
        == (path_stat.st_dev, path_stat.st_ino)
    ):
        raise ValueError(f"{label} changed identity after exclusive creation")
    if not stat_module.S_ISREG(path_stat.st_mode):
        raise ValueError(f"{label} is no longer a regular file")


def _acquire_commit_lock(path: Path) -> _OwnedFile:
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        file_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(file_stat.st_mode):
            raise ValueError(f"commit lock must be a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError(f"artifact commit lock is held: {path}") from error
        owned_file = _OwnedFile(
            path=path,
            descriptor=descriptor,
            identity=(file_stat.st_dev, file_stat.st_ino),
        )
        _require_owned_path(owned_file, label="commit lock")
        return owned_file
    except BaseException:
        os.close(descriptor)
        raise


def _acquire_shared_commit_lock(path: Path) -> _OwnedFile:
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        file_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(file_stat.st_mode):
            raise ValueError(f"commit lock must be a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError(
                f"input shard commit lock is held exclusively: {path}"
            ) from error
        owned_file = _OwnedFile(
            path=path,
            descriptor=descriptor,
            identity=(file_stat.st_dev, file_stat.st_ino),
        )
        _require_owned_path(owned_file, label="input shard commit lock")
        return owned_file
    except BaseException:
        os.close(descriptor)
        raise


def _path_matches_owner(path: Path, owner: tuple[int, int]) -> bool:
    try:
        file_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (file_stat.st_dev, file_stat.st_ino) == owner


def _path_exists_nofollow(path: Path) -> bool:
    try:
        os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_owned_file(path: Path, owner: tuple[int, int] | None) -> None:
    if owner is not None and _path_matches_owner(path, owner):
        path.unlink()


def _require_committed_artifact_bound(
    owned_file: _OwnedFile,
    target: Path,
    payload: dict[str, object],
) -> None:
    if owned_file.descriptor < 0:
        raise ValueError("committed artifact ownership descriptor is closed")
    before = os.fstat(owned_file.descriptor)
    if (before.st_dev, before.st_ino) != owned_file.identity:
        raise ValueError("committed artifact file descriptor changed identity")
    try:
        target_before = os.stat(target, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("committed artifact path changed identity") from error
    if (
        (target_before.st_dev, target_before.st_ino) != owned_file.identity
        or not stat_module.S_ISREG(target_before.st_mode)
    ):
        raise ValueError("committed artifact path changed identity")
    digest = _hash_open_file_descriptor(owned_file.descriptor)
    after = os.fstat(owned_file.descriptor)
    try:
        target_after = os.stat(target, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("committed artifact path changed identity") from error
    if not (
        _file_identity(before)
        == _file_identity(after)
        == _file_identity(target_after)
    ):
        raise ValueError("committed artifact changed while being verified")
    if after.st_size != payload["size_bytes"]:
        raise ValueError("committed artifact size does not match completion payload")
    if digest != payload["artifact_sha256"]:
        raise ValueError("committed artifact checksum does not match completion payload")


def _require_completion_marker_bound(
    marker: Path,
    owner: tuple[int, int] | None,
    payload: dict[str, object],
) -> None:
    if owner is None or not _path_matches_owner(marker, owner):
        raise ValueError("completion marker changed identity after commit")
    try:
        text = _read_stable_bytes(marker, label="completion marker").decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("completion marker is not valid UTF-8") from error
    if not _path_matches_owner(marker, owner):
        raise ValueError("completion marker changed identity after commit")
    marker_payload = _parse_marker_text(text)
    if marker_payload != payload:
        raise ValueError("completion marker does not match committed payload")


def _write_completion_marker_atomic(
    marker: Path,
    payload: dict[str, object],
    *,
    file_mode: int = 0o600,
) -> tuple[int, int]:
    temp = marker.parent / f".{marker.name}.{uuid.uuid4().hex}.tmp"
    owned_file: _OwnedFile | None = None
    marker_owner: tuple[int, int] | None = None
    try:
        owned_file = _create_owned_file(temp, mode=file_mode)
        _require_owned_path(owned_file, label="temporary completion marker")
        content = _canonical_marker_json(payload).encode("utf-8")
        offset = 0
        while offset < len(content):
            written = os.write(owned_file.descriptor, content[offset:])
            if written <= 0:
                raise OSError("completion marker write made no progress")
            offset += written
        os.fsync(owned_file.descriptor)
        _require_owned_path(owned_file, label="temporary completion marker")
        os.replace(temp, marker)
        marker_owner = owned_file.identity
        if not _path_matches_owner(marker, marker_owner):
            raise ValueError("completion marker changed identity during commit")
        _fsync_directory(marker.parent)
        return marker_owner
    except BaseException:
        _unlink_owned_file(marker, marker_owner)
        raise
    finally:
        if owned_file is not None:
            _unlink_owned_file(temp, owned_file.identity)
            _close_owned_file(owned_file)


def _read_open_file_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        block = os.pread(descriptor, 64 * 1024, offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    return b"".join(chunks)


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        content = _read_open_file_descriptor(descriptor)
        after = os.fstat(descriptor)
        current = path.stat()
        if not (
            _file_identity(before)
            == _file_identity(after)
            == _file_identity(current)
        ):
            raise ValueError(f"{label} changed identity while being read: {path}")
        return content
    finally:
        os.close(descriptor)


def _parse_marker_text(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("completion marker is not valid JSON") from error
    if type(payload) is not dict or set(payload) != _MARKER_KEYS:
        raise ValueError("completion marker has unknown or missing fields")
    if text != _canonical_marker_json(payload):
        raise ValueError("completion marker JSON is not canonical")
    if payload["schema_name"] != SCHEMA_NAME or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("completion marker schema mismatch")
    if type(payload["size_bytes"]) is not int or payload["size_bytes"] < 0:
        raise TypeError("completion marker size_bytes must be a non-negative integer")
    if type(payload["config_sha256"]) is not str or type(payload["artifact_sha256"]) is not str:
        raise TypeError("completion marker hashes must be strings")
    return payload


@dataclass(frozen=True, slots=True)
class _MarkedPayloadSnapshot:
    path: Path
    artifact_identity: tuple[int, int, int, int, int]
    artifact_sha256: str
    marker_path: Path
    marker_identity: tuple[int, int, int, int, int]
    marker_sha256: str
    marker_json: str
    artifact_descriptor: int | None = None
    marker_descriptor: int | None = None


@dataclass(slots=True)
class _MergeInputGuard:
    locks: tuple[_OwnedFile, ...]
    snapshots: list[_MarkedPayloadSnapshot]

    def close(self) -> None:
        descriptors = {
            descriptor
            for snapshot in self.snapshots
            for descriptor in (
                snapshot.artifact_descriptor,
                snapshot.marker_descriptor,
            )
            if descriptor is not None
        }
        for descriptor in descriptors:
            os.close(descriptor)
        for lock in reversed(self.locks):
            _close_owned_file(lock)


def _acquire_merge_input_guard(
    shard_paths: tuple[Path, ...],
) -> _MergeInputGuard:
    if type(shard_paths) is not tuple or not shard_paths:
        raise TypeError("shard_paths must be a non-empty tuple")
    resolved_paths: set[Path] = set()
    for candidate in shard_paths:
        if not isinstance(candidate, Path):
            raise TypeError("shard_paths entries must be pathlib.Path")
        resolved_paths.add(candidate.expanduser().resolve(strict=True))
    locks: list[_OwnedFile] = []
    try:
        for path in sorted(resolved_paths, key=str):
            lock_path = path.parent / f".{path.name}.commit.lock"
            locks.append(_acquire_shared_commit_lock(lock_path))
        return _MergeInputGuard(locks=tuple(locks), snapshots=[])
    except BaseException:
        for lock in reversed(locks):
            _close_owned_file(lock)
        raise


def _read_marked_payload_with_snapshot(
    resolved: Path,
    *,
    load_samples: bool,
    reader: Callable[..., object],
    keep_descriptors: bool = False,
) -> tuple[object, _MarkedPayloadSnapshot]:
    marker = _marker_path(resolved)
    if not marker.is_file():
        raise FileNotFoundError(f"completion marker does not exist: {marker}")
    retain_descriptors = False
    marker_descriptor = os.open(
        marker,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        marker_before = os.fstat(marker_descriptor)
        marker_digest = _hash_open_file_descriptor(marker_descriptor)
        try:
            marker_text = _read_open_file_descriptor(marker_descriptor).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("completion marker is not valid UTF-8") from error
        payload = _parse_marker_text(marker_text)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            digest = _hash_open_file_descriptor(descriptor)
            if before.st_size != payload["size_bytes"]:
                raise ValueError("artifact size does not match completion marker")
            if digest != payload["artifact_sha256"]:
                raise ValueError("marker artifact checksum does not match artifact")
            artifact = reader(
                descriptor,
                load_samples=load_samples,
            )
            _require_open_file_stable(
                descriptor,
                resolved,
                before,
                digest,
                label="artifact",
            )
            _require_open_file_stable(
                marker_descriptor,
                marker,
                marker_before,
                marker_digest,
                label="completion marker",
            )
            artifact.provenance.verify_sources()
            if payload["config_sha256"] != artifact.provenance.config_sha256:
                raise ValueError("completion marker config hash mismatch")
            snapshot = _MarkedPayloadSnapshot(
                path=resolved,
                artifact_identity=_file_identity(before),
                artifact_sha256=digest,
                marker_path=marker,
                marker_identity=_file_identity(marker_before),
                marker_sha256=marker_digest,
                marker_json=marker_text,
                artifact_descriptor=(descriptor if keep_descriptors else None),
                marker_descriptor=(
                    marker_descriptor if keep_descriptors else None
                ),
            )
            retain_descriptors = keep_descriptors
            return artifact, snapshot
        finally:
            if not retain_descriptors:
                os.close(descriptor)
    finally:
        if not retain_descriptors:
            os.close(marker_descriptor)


def _read_marked_payload(
    resolved: Path,
    *,
    load_samples: bool,
    reader: Callable[..., object],
) -> object:
    payload, _ = _read_marked_payload_with_snapshot(
        resolved,
        load_samples=load_samples,
        reader=reader,
    )
    return payload


def _require_marked_payload_snapshot(
    snapshot: _MarkedPayloadSnapshot,
) -> None:
    marker_descriptor = snapshot.marker_descriptor
    close_marker = marker_descriptor is None
    try:
        if marker_descriptor is None:
            marker_descriptor = os.open(
                snapshot.marker_path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        marker_before = os.fstat(marker_descriptor)
        if _file_identity(marker_before) != snapshot.marker_identity:
            raise ValueError("input shard completion marker snapshot changed identity")
        marker_digest = _hash_open_file_descriptor(marker_descriptor)
        if marker_digest != snapshot.marker_sha256:
            raise ValueError("input shard completion marker snapshot changed content")
        marker_text = _read_open_file_descriptor(marker_descriptor).decode("utf-8")
        if marker_text != snapshot.marker_json:
            raise ValueError("input shard completion marker snapshot changed payload")
        marker_payload = _parse_marker_text(marker_text)
        descriptor = snapshot.artifact_descriptor
        close_artifact = descriptor is None
        if descriptor is None:
            descriptor = os.open(
                snapshot.path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        try:
            before = os.fstat(descriptor)
            if _file_identity(before) != snapshot.artifact_identity:
                raise ValueError("input shard artifact snapshot changed identity")
            digest = _hash_open_file_descriptor(descriptor)
            if digest != snapshot.artifact_sha256:
                raise ValueError("input shard artifact snapshot changed content")
            if (
                before.st_size != marker_payload["size_bytes"]
                or digest != marker_payload["artifact_sha256"]
            ):
                raise ValueError("input shard snapshot no longer matches its marker")
            _require_open_file_stable(
                descriptor,
                snapshot.path,
                before,
                digest,
                label="input shard artifact snapshot",
            )
            _require_open_file_stable(
                marker_descriptor,
                snapshot.marker_path,
                marker_before,
                marker_digest,
                label="input shard completion marker snapshot",
            )
        finally:
            if close_artifact:
                os.close(descriptor)
    except (FileNotFoundError, UnicodeDecodeError) as error:
        raise ValueError("input shard snapshot changed or disappeared") from error
    finally:
        if close_marker and marker_descriptor is not None:
            os.close(marker_descriptor)


def read_uvlf_artifact(
    path: str | Path,
    *,
    load_samples: bool = False,
) -> UVLFArtifact:
    if type(load_samples) is not bool:
        raise TypeError("load_samples must be exactly boolean")
    resolved = Path(path).expanduser().resolve(strict=True)
    artifact = _read_marked_payload(
        resolved,
        load_samples=load_samples,
        reader=_read_hdf5_from_open_descriptor,
    )
    if type(artifact) is not UVLFArtifact:
        raise RuntimeError("artifact reader returned the wrong payload type")
    return artifact


def read_uvlf_shard(
    path: str | Path,
    *,
    load_samples: bool = False,
) -> UVLFShard:
    if type(load_samples) is not bool:
        raise TypeError("load_samples must be exactly boolean")
    resolved = Path(path).expanduser().resolve(strict=True)
    shard = _read_marked_payload(
        resolved,
        load_samples=load_samples,
        reader=_read_uvlf_shard_from_open_descriptor,
    )
    if type(shard) is not UVLFShard:
        raise RuntimeError("shard reader returned the wrong payload type")
    return shard


def _read_uvlf_shard_with_snapshot(
    path: str | Path,
    *,
    load_samples: bool,
    keep_descriptors: bool = False,
) -> tuple[UVLFShard, _MarkedPayloadSnapshot]:
    resolved = Path(path).expanduser().resolve(strict=True)
    shard, snapshot = _read_marked_payload_with_snapshot(
        resolved,
        load_samples=load_samples,
        reader=_read_uvlf_shard_from_open_descriptor,
        keep_descriptors=keep_descriptors,
    )
    if type(shard) is not UVLFShard:
        raise RuntimeError("shard reader returned the wrong payload type")
    return shard, snapshot


def _move_existing_to_owned_backup(
    source: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    backup = source.parent / f".{source.name}.{uuid.uuid4().hex}.backup"
    reservation = _create_owned_file(backup, mode=0o600)
    try:
        _require_owned_path(reservation, label=f"{label} backup reservation")
        os.replace(source, backup)
        backup_stat = os.stat(backup, follow_symlinks=False)
        owner = (backup_stat.st_dev, backup_stat.st_ino)
        if not stat_module.S_ISREG(backup_stat.st_mode):
            raise ValueError(f"{label} backup must remain a regular file")
        return backup, owner
    finally:
        _unlink_owned_file(reservation.path, reservation.identity)
        _close_owned_file(reservation)


def _rollback_overwrite_pair(
    *,
    target: Path,
    target_backup: Path | None,
    target_backup_owner: tuple[int, int] | None,
    new_target_owner: tuple[int, int] | None,
    marker: Path,
    marker_backup: Path | None,
    marker_backup_owner: tuple[int, int] | None,
    new_marker_owner: tuple[int, int] | None,
) -> bool:
    target_blocked = _path_exists_nofollow(target) and not (
        new_target_owner is not None
        and _path_matches_owner(target, new_target_owner)
    )
    marker_blocked = _path_exists_nofollow(marker) and not (
        new_marker_owner is not None
        and _path_matches_owner(marker, new_marker_owner)
    )
    if target_backup is not None and (
        target_backup_owner is None
        or not _path_matches_owner(target_backup, target_backup_owner)
    ):
        _fsync_directory(target.parent)
        return False
    if marker_backup is not None and (
        marker_backup_owner is None
        or not _path_matches_owner(marker_backup, marker_backup_owner)
    ):
        _fsync_directory(target.parent)
        return False
    if target_blocked or marker_blocked:
        if target_blocked:
            _unlink_owned_file(marker, new_marker_owner)
        if marker_blocked:
            _unlink_owned_file(target, new_target_owner)
        _fsync_directory(target.parent)
        return False
    _unlink_owned_file(marker, new_marker_owner)
    _unlink_owned_file(target, new_target_owner)
    if _path_exists_nofollow(target) or _path_exists_nofollow(marker):
        _fsync_directory(target.parent)
        return False
    if target_backup is not None:
        os.replace(target_backup, target)
    if marker_backup is not None:
        os.replace(marker_backup, marker)
    _fsync_directory(target.parent)
    return True


def _write_payload_atomic(
    payload_object: object,
    *,
    config: UVLFRunConfig,
    provenance: ArtifactProvenance,
    target: Path,
    overwrite: bool,
    writer: Callable[..., None],
    reader: Callable[..., object],
    final_validator: Callable[[], None] | None = None,
    reader_load_samples: bool = True,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(target)
    claim = target.parent / f".{target.name}.commit.lock"
    claim_file: _OwnedFile | None = None
    temp_file: _OwnedFile | None = None
    target_backup: Path | None = None
    target_backup_owner: tuple[int, int] | None = None
    marker_backup: Path | None = None
    marker_backup_owner: tuple[int, int] | None = None
    marker_owner: tuple[int, int] | None = None
    renamed = False
    commit_validated = False
    try:
        claim_file = _acquire_commit_lock(claim)
        _fsync_directory(target.parent)
        target_exists = _path_exists_nofollow(target)
        marker_exists = _path_exists_nofollow(marker)
        if not overwrite and (target_exists or marker_exists):
            raise FileExistsError(f"artifact or completion marker already exists: {target}")
        for existing_path, exists, label in (
            (target, target_exists, "artifact"),
            (marker, marker_exists, "completion marker"),
        ):
            if exists and not stat_module.S_ISREG(
                os.stat(existing_path, follow_symlinks=False).st_mode
            ):
                raise ValueError(f"existing {label} path must be a regular file")
        target_mode = (
            stat_module.S_IMODE(os.stat(target, follow_symlinks=False).st_mode)
            if target_exists
            else 0o600
        )
        marker_mode = (
            stat_module.S_IMODE(os.stat(marker, follow_symlinks=False).st_mode)
            if marker_exists
            else 0o600
        )

        provenance.verify_sources()
        config_hash = canonical_config_sha256(config)
        if config_hash != provenance.config_sha256:
            raise ValueError("provenance config hash does not match canonical config")
        temp_path = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        temp_file = _create_owned_file(temp_path, mode=0o600)
        writer(payload_object, temp_file)
        os.fchmod(temp_file.descriptor, target_mode)
        _fsync_file(temp_file)
        _require_owned_path(temp_file, label="temporary artifact")
        temp_before = os.fstat(temp_file.descriptor)
        artifact_sha = _hash_open_file_descriptor(temp_file.descriptor)
        validated = reader(
            temp_file.descriptor,
            load_samples=reader_load_samples,
        )
        _require_owned_path(temp_file, label="temporary artifact")
        temp_after = os.fstat(temp_file.descriptor)
        if _file_identity(temp_before) != _file_identity(temp_after):
            raise ValueError("temporary artifact changed while being validated")
        if _hash_open_file_descriptor(temp_file.descriptor) != artifact_sha:
            raise ValueError("temporary artifact content changed while being validated")
        validated.provenance.verify_sources()
        if validated.provenance.config_sha256 != config_hash:
            raise RuntimeError("temporary artifact validation changed config hash")
        payload = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config_hash,
            "artifact_sha256": artifact_sha,
            "size_bytes": temp_after.st_size,
        }
        _fsync_directory(target.parent)
        if overwrite and _path_exists_nofollow(target):
            target_backup, target_backup_owner = _move_existing_to_owned_backup(
                target,
                label="artifact",
            )
        if overwrite and _path_exists_nofollow(marker):
            marker_backup, marker_backup_owner = _move_existing_to_owned_backup(
                marker,
                label="completion marker",
            )
        if target_backup is not None or marker_backup is not None:
            _fsync_directory(target.parent)

        _require_owned_path(claim_file, label="commit lock")
        _require_owned_path(temp_file, label="temporary artifact")
        os.replace(temp_file.path, target)
        renamed = True
        _require_committed_artifact_bound(temp_file, target, payload)
        _fsync_directory(target.parent)
        _require_owned_path(claim_file, label="commit lock")
        marker_owner = _write_completion_marker_atomic(
            marker,
            payload,
            file_mode=marker_mode,
        )
        provenance.verify_sources()
        _require_committed_artifact_bound(temp_file, target, payload)
        _require_completion_marker_bound(marker, marker_owner, payload)
        if final_validator is not None:
            final_validator()
        _fsync_directory(target.parent)
        commit_validated = True
        if target_backup is not None:
            _unlink_owned_file(target_backup, target_backup_owner)
        if marker_backup is not None:
            _unlink_owned_file(marker_backup, marker_backup_owner)
        if target_backup is not None or marker_backup is not None:
            _fsync_directory(target.parent)
        return target
    except BaseException:
        if (
            not commit_validated
            and (target_backup is not None or marker_backup is not None)
        ):
            _rollback_overwrite_pair(
                target=target,
                target_backup=target_backup,
                target_backup_owner=target_backup_owner,
                new_target_owner=(
                    temp_file.identity
                    if renamed and temp_file is not None
                    else None
                ),
                marker=marker,
                marker_backup=marker_backup,
                marker_backup_owner=marker_backup_owner,
                new_marker_owner=marker_owner,
            )
        elif not commit_validated and renamed:
            _unlink_owned_file(marker, marker_owner)
            _fsync_directory(target.parent)
        raise
    finally:
        if temp_file is not None:
            _unlink_owned_file(temp_file.path, temp_file.identity)
            _close_owned_file(temp_file)
        _close_owned_file(claim_file)


def _write_uvlf_artifact_atomic_with_validator(
    artifact: UVLFArtifact,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
    final_validator: Callable[[], None] | None = None,
) -> Path:
    if type(artifact) is not UVLFArtifact:
        raise TypeError("artifact must be exactly UVLFArtifact")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    target = (
        artifact.result.config.output.artifact_path
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not target.is_absolute() or target.suffix != ".h5":
        raise ValueError("artifact path must be an absolute .h5 path")
    return _write_payload_atomic(
        artifact,
        config=artifact.result.config,
        provenance=artifact.provenance,
        target=target,
        overwrite=overwrite,
        writer=_write_hdf5_file,
        reader=_read_hdf5_from_open_descriptor,
        final_validator=final_validator,
    )


def write_uvlf_artifact_atomic(
    artifact: UVLFArtifact,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    return _write_uvlf_artifact_atomic_with_validator(
        artifact,
        path=path,
        overwrite=overwrite,
    )


def write_uvlf_shard_atomic(
    shard: UVLFShard,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    if type(shard) is not UVLFShard:
        raise TypeError("shard must be exactly UVLFShard")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    target = Path(path).expanduser().resolve()
    if not target.is_absolute() or target.suffix != ".h5":
        raise ValueError("shard path must be an absolute .h5 path")
    return _write_payload_atomic(
        shard,
        config=shard.config,
        provenance=shard.provenance,
        target=target,
        overwrite=overwrite,
        writer=_write_uvlf_shard_hdf5_file,
        reader=_read_uvlf_shard_from_open_descriptor,
    )


def _write_uvlf_shard_from_spool_atomic(
    shard: UVLFShard,
    spool_descriptor: int,
    spool_path: Path,
    spool_owner: tuple[int, int],
    spool_snapshot: _SpoolSnapshot,
    path: Path,
    *,
    overwrite: bool,
    copy_chunk_size: int = 65_536,
    _copy_observer: Callable[[int], None] | None = None,
) -> Path:
    if type(shard) is not UVLFShard:
        raise TypeError("shard must be exactly UVLFShard")
    if shard.sample_descriptor is None or shard.sample is not None:
        raise ValueError("spool shard requires a descriptor and no loaded sample")
    if type(spool_descriptor) is not int or spool_descriptor < 0:
        raise ValueError("spool_descriptor must be an open non-negative integer")
    if not isinstance(spool_path, Path):
        raise TypeError("spool_path must be a Path")
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    if type(copy_chunk_size) is not int or copy_chunk_size <= 0:
        raise ValueError("copy_chunk_size must be a positive integer")
    if _copy_observer is not None and not callable(_copy_observer):
        raise TypeError("_copy_observer must be callable or None")
    chunk_size = copy_chunk_size
    if not spool_path.is_absolute():
        raise ValueError("spool_path must be absolute")
    target = path.expanduser().resolve()
    if target.suffix != ".h5":
        raise ValueError("shard path must be an absolute .h5 path")

    def writer(payload: object, owned_file: _OwnedFile) -> None:
        if type(payload) is not UVLFShard:
            raise TypeError("spool writer payload must be exactly UVLFShard")
        descriptor = payload.sample_descriptor
        if descriptor is None:
            raise RuntimeError("validated spool shard lost its sample descriptor")
        _require_owned_path(owned_file, label="temporary shard")
        _require_spool_snapshot(
            spool_descriptor,
            spool_path,
            spool_owner,
            spool_snapshot,
        )
        with os.fdopen(
            os.dup(spool_descriptor), "rb", closefd=True
        ) as source_file, h5py.File(source_file, "r") as source_handle:
            _reject_links_and_hard_link_aliases(source_handle)
            _require_exact_attrs(
                source_handle,
                _ROOT_ATTRS | {"artifact_kind"},
                name="sample spool root",
            )
            if _read_utf8_attr(
                source_handle,
                "schema_name",
                name="sample spool root",
            ) != SCHEMA_NAME:
                raise ValueError("sample spool schema_name mismatch")
            if _read_utf8_attr(
                source_handle,
                "schema_version",
                name="sample spool root",
            ) != SCHEMA_VERSION:
                raise ValueError("sample spool schema_version mismatch")
            if _read_utf8_attr(
                source_handle,
                "artifact_kind",
                name="sample spool root",
            ) != "sample_spool":
                raise ValueError("sample spool artifact_kind mismatch")
            _require_exact_names(
                source_handle,
                {"config", "provenance", "axes", "samples"},
                name="sample spool root",
            )
            config_group = source_handle["config"]
            _require_exact_attrs(config_group, set(), name="sample spool config")
            _require_exact_names(
                config_group,
                {"canonical_json", "sha256"},
                name="sample spool config",
            )
            stored_json = _read_string(
                config_group["canonical_json"],
                name="sample spool config.canonical_json",
            )
            stored_hash = _read_string(
                config_group["sha256"],
                name="sample spool config.sha256",
            )
            if (
                stored_json != canonical_config_json(payload.config)
                or stored_hash != canonical_config_sha256(payload.config)
            ):
                raise ValueError("sample spool config mismatch")
            stored_provenance = _read_provenance(source_handle["provenance"])
            if stored_provenance != payload.provenance:
                raise ValueError("sample spool provenance mismatch")
            _validate_axis_group(source_handle["axes"], payload.config)
            source_samples = source_handle["samples"]
            _require_exact_attrs(source_samples, set(), name="sample spool samples")
            _require_exact_names(
                source_samples,
                {_z_name(redshift) for redshift in payload.config.redshifts},
                name="sample spool redshifts",
            )
            for configured_redshift in payload.config.redshifts:
                configured_z_group = source_samples[_z_name(configured_redshift)]
                _require_exact_attrs(
                    configured_z_group,
                    set(),
                    name="sample spool redshift",
                )
                _require_exact_names(
                    configured_z_group,
                    set(payload.config.stellar_population.imf_modes),
                    name="sample spool modes",
                )
            z_name = _z_name(descriptor.redshift)
            if z_name not in source_samples or descriptor.imf_mode not in source_samples[z_name]:
                raise ValueError("sample spool is missing shard sample key")
            source_group = source_samples[z_name][descriptor.imf_mode]
            _require_exact_attrs(
                source_group,
                {"redshift", "imf_mode", "sample_count", "next_mass_index"},
                name="sample spool mode",
            )
            sample_count_attr = np.asarray(source_group.attrs["sample_count"])
            next_mass_attr = np.asarray(source_group.attrs["next_mass_index"])
            if (
                sample_count_attr.shape != ()
                or sample_count_attr.dtype != np.dtype(np.int64)
                or next_mass_attr.shape != ()
                or next_mass_attr.dtype != np.dtype(np.int64)
            ):
                raise ValueError("sample spool count attributes must be scalar int64")
            if (
                _read_float64_attr(source_group, "redshift", name="sample spool mode")
                != descriptor.redshift
                or _read_utf8_attr(source_group, "imf_mode", name="sample spool mode")
                != descriptor.imf_mode
                or int(sample_count_attr) != descriptor.sample_count
                or int(next_mass_attr)
                != payload.config.sampling.n_halo_mass_samples
            ):
                raise ValueError("sample spool mode metadata mismatch")
            _require_exact_names(source_group, set(_SAMPLE_FIELDS), name="sample spool mode")
            source_datasets: dict[str, h5py.Dataset] = {}
            for field_name in _SAMPLE_FIELDS:
                source_dataset = source_group[field_name]
                is_index = field_name in ("mass_index", "track_index")
                _validate_numeric_dataset(
                    source_dataset,
                    name=f"sample spool.{field_name}",
                    dtype=np.int64 if is_index else np.float64,
                    units=_SAMPLE_UNITS[field_name],
                    scalar=False,
                )
                if source_dataset.shape != (descriptor.sample_count,):
                    raise ValueError("sample spool dataset length mismatch")
                if source_dataset.chunks is None or source_dataset.maxshape != (None,):
                    raise ValueError("sample spool datasets must be chunked and extensible")
                if (
                    source_dataset.compression is None
                    or not source_dataset.shuffle
                    or not source_dataset.fletcher32
                ):
                    raise ValueError("sample spool dataset storage filters are incomplete")
                source_datasets[field_name] = source_dataset

            with os.fdopen(
                os.dup(owned_file.descriptor), "r+b", closefd=True
            ) as file_object, h5py.File(file_object, "w") as target_handle:
                _write_config_provenance_axes(
                    target_handle,
                    payload.config,
                    payload.provenance,
                    artifact_kind="shard",
                )
                results_group = target_handle.create_group("results", track_order=True)
                _write_result_mode(results_group, payload.key[0], payload.result)
                diagnostics_group = target_handle.create_group(
                    "diagnostics", track_order=True
                )
                _write_diagnostic_mode(diagnostics_group, payload.diagnostic)
                samples_group = target_handle.create_group("samples", track_order=True)
                target_z_group = samples_group.create_group(z_name, track_order=True)
                target_group = target_z_group.create_group(
                    descriptor.imf_mode,
                    track_order=True,
                )
                target_group.attrs["redshift"] = np.float64(descriptor.redshift)
                target_group.attrs["imf_mode"] = descriptor.imf_mode
                target_datasets: dict[str, h5py.Dataset] = {}
                target_chunk_size = min(chunk_size, descriptor.sample_count)
                for field_name in _SAMPLE_FIELDS:
                    is_index = field_name in ("mass_index", "track_index")
                    target_dataset = target_group.create_dataset(
                        field_name,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(target_chunk_size,),
                        compression="gzip",
                        shuffle=True,
                        fletcher32=True,
                        dtype=np.int64 if is_index else np.float64,
                    )
                    target_dataset.attrs["units"] = _SAMPLE_UNITS[field_name]
                    target_datasets[field_name] = target_dataset
                for start in range(0, descriptor.sample_count, chunk_size):
                    stop = min(start + chunk_size, descriptor.sample_count)
                    sample_slice = slice(start, stop)
                    values = {
                        field_name: np.asarray(dataset[sample_slice])
                        for field_name, dataset in source_datasets.items()
                    }
                    validated_chunk = HaloSampleTable(
                        redshift=descriptor.redshift,
                        imf_mode=descriptor.imf_mode,
                        **values,
                    )
                    if _copy_observer is not None:
                        _copy_observer(stop - start)
                    for field_name, target_dataset in target_datasets.items():
                        target_dataset.resize((stop,))
                        target_dataset[start:stop] = getattr(validated_chunk, field_name)
                    del values, validated_chunk
                target_handle.flush()
        _require_spool_snapshot(
            spool_descriptor,
            spool_path,
            spool_owner,
            spool_snapshot,
        )
        _require_owned_path(owned_file, label="temporary shard")

    return _write_payload_atomic(
        shard,
        config=shard.config,
        provenance=shard.provenance,
        target=target,
        overwrite=overwrite,
        writer=writer,
        reader=_read_uvlf_shard_from_open_descriptor,
        final_validator=lambda: _require_spool_snapshot(
            spool_descriptor,
            spool_path,
            spool_owner,
            spool_snapshot,
        ),
        reader_load_samples=False,
    )


def _require_provenance_identity(
    actual: ArtifactProvenance,
    expected: ArtifactProvenance,
) -> None:
    for field_name in (
        "config_sha256",
        "code_revision",
        "code_dirty",
        "seed_namespace",
        "source_checksums",
    ):
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise ValueError(f"resume shard provenance {field_name} mismatch")


def validate_uvlf_resume_shards(
    config: UVLFRunConfig,
    provenance: ArtifactProvenance,
    shard_paths: tuple[Path, ...],
) -> tuple[UVLFShardDescriptor, ...]:
    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if type(provenance) is not ArtifactProvenance:
        raise TypeError("provenance must be exactly ArtifactProvenance")
    if type(shard_paths) is not tuple:
        raise TypeError("shard_paths must be a tuple")
    if provenance.config_sha256 != canonical_config_sha256(config):
        raise ValueError("resume provenance config hash mismatch")
    provenance.verify_sources()
    descriptors_by_key: dict[tuple[float, str], UVLFShardDescriptor] = {}
    for candidate in shard_paths:
        if not isinstance(candidate, Path):
            raise TypeError("shard_paths entries must be pathlib.Path")
        path = candidate.expanduser().resolve(strict=True)
        shard = read_uvlf_shard(path, load_samples=False)
        if canonical_config_json(shard.config) != canonical_config_json(config):
            raise ValueError(f"resume shard config mismatch: {path}")
        _require_provenance_identity(shard.provenance, provenance)
        shard.provenance.verify_sources()
        if shard.key in descriptors_by_key:
            raise ValueError(f"duplicate resume shard axis: {shard.key}")
        descriptors_by_key[shard.key] = UVLFShardDescriptor(
            path,
            shard.key[0],
            shard.key[1],
            (
                None
                if shard.sample_descriptor is None
                else shard.sample_descriptor.sample_count
            ),
        )
    configured_axes = tuple(
        (redshift, mode)
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )
    return tuple(
        descriptors_by_key[key] for key in configured_axes if key in descriptors_by_key
    )


def _result_payload_equal(left: IMFModeResult, right: IMFModeResult) -> bool:
    return (
        left.imf_mode == right.imf_mode
        and left.halo_tracks == right.halo_tracks == ()
        and all(
            np.array_equal(getattr(left, name), getattr(right, name), equal_nan=True)
            for name in _RESULT_FIELDS
        )
    )


def _sample_payload_equal(
    left: HaloSampleTable | None,
    right: HaloSampleTable | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return left.key == right.key and all(
        np.array_equal(getattr(left, name), getattr(right, name), equal_nan=True)
        for name in _SAMPLE_FIELDS
    )


def _duplicate_shards_equal(left: UVLFShard, right: UVLFShard) -> bool:
    return (
        _result_payload_equal(left.result, right.result)
        and left.diagnostic == right.diagnostic
        and left.sample_descriptor == right.sample_descriptor
        and _sample_payload_equal(left.sample, right.sample)
    )


def _merge_uvlf_shards_guarded(
    shard_paths: tuple[Path, ...],
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    final_provenance: ArtifactProvenance | None = None,
    guard: _MergeInputGuard,
) -> Path:
    """Merge a complete shard grid while cooperative writers are shared-locked.

    The commit point is the final validator under these locks. Direct filesystem
    replacement after that validator is outside the cooperative atomic protocol.
    ``total_seconds`` is the sum over unique axes.
    """
    if type(shard_paths) is not tuple or not shard_paths:
        raise TypeError("shard_paths must be a non-empty tuple")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    if final_provenance is not None and type(final_provenance) is not ArtifactProvenance:
        raise TypeError("final_provenance must be exactly ArtifactProvenance or None")
    shards: list[UVLFShard] = []
    snapshots: list[_MarkedPayloadSnapshot] = []
    for candidate in shard_paths:
        if not isinstance(candidate, Path):
            raise TypeError("shard_paths entries must be pathlib.Path")
        shard, snapshot = _read_uvlf_shard_with_snapshot(
            candidate,
            load_samples=True,
            keep_descriptors=True,
        )
        shards.append(shard)
        snapshots.append(snapshot)
        guard.snapshots.append(snapshot)
    first = shards[0]
    first_config_json = canonical_config_json(first.config)
    first_provenance = first.provenance
    by_key: dict[tuple[float, str], UVLFShard] = {}
    created_timestamps: list[str] = []
    for shard in shards:
        if canonical_config_json(shard.config) != first_config_json:
            raise ValueError("shard config conflict during merge")
        try:
            _require_provenance_identity(shard.provenance, first_provenance)
        except ValueError as error:
            raise ValueError("shard provenance identity conflict during merge") from error
        shard.provenance.verify_sources()
        created_timestamps.append(shard.provenance.created_utc)
        existing = by_key.get(shard.key)
        if existing is None:
            by_key[shard.key] = shard
        elif not _duplicate_shards_equal(existing, shard):
            raise ValueError(f"conflicting duplicate shard payload for axis {shard.key}")

    configured_axes = tuple(
        (redshift, mode)
        for redshift in first.config.redshifts
        for mode in first.config.stellar_population.imf_modes
    )
    missing = tuple(key for key in configured_axes if key not in by_key)
    if missing:
        raise ValueError(f"shards do not provide complete coverage; missing axis {missing[0]}")
    redshift_results = tuple(
        RedshiftResult(
            redshift=redshift,
            imf_modes=tuple(by_key[(redshift, mode)].result for mode in first.config.stellar_population.imf_modes),
        )
        for redshift in first.config.redshifts
    )
    mode_runs = tuple(by_key[key].diagnostic for key in configured_axes)
    total_seconds = sum(item.sampling_seconds for item in mode_runs)
    result = UVLFRunResult(
        config=first.config,
        redshifts=redshift_results,
        diagnostics=RunDiagnostics(
            total_seconds=total_seconds,
            mode_runs=mode_runs,
        ),
    )
    sample_shards = tuple(by_key[key] for key in configured_axes if by_key[key].sample is not None)
    provenance_base = first_provenance
    if final_provenance is not None:
        for field_name in (
            "config_sha256",
            "code_revision",
            "code_dirty",
            "seed_namespace",
        ):
            if getattr(final_provenance, field_name) != getattr(first_provenance, field_name):
                raise ValueError(f"final provenance {field_name} mismatch")
        scientific_sources = first_provenance.source_checksums
        if final_provenance.source_checksums[: len(scientific_sources)] != scientific_sources:
            raise ValueError("final provenance must preserve shard scientific source checksums")
        final_provenance.verify_sources()
        provenance_base = final_provenance
    merged_provenance = replace(
        provenance_base,
        created_utc=min(
            created_timestamps,
            key=lambda value: datetime.fromisoformat(value[:-1] + "+00:00"),
        ),
    )
    artifact = UVLFArtifact(
        result=result,
        provenance=merged_provenance,
        sample_descriptors=tuple(
            shard.sample_descriptor for shard in sample_shards
        ),
        samples=tuple(shard.sample for shard in sample_shards),
    )
    target = (
        first.config.output.artifact_path
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if not target.is_absolute() or target.suffix != ".h5":
        raise ValueError("output_path must be an absolute .h5 path")

    def validate_input_snapshots() -> None:
        for snapshot in snapshots:
            _require_marked_payload_snapshot(snapshot)

    validate_input_snapshots()

    def validate_merge_commit() -> None:
        validate_input_snapshots()
        read_uvlf_artifact(target, load_samples=True)

    written = _write_uvlf_artifact_atomic_with_validator(
        artifact,
        path=target,
        overwrite=overwrite,
        final_validator=validate_merge_commit,
    )
    return written


def merge_uvlf_shards(
    shard_paths: tuple[Path, ...],
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    final_provenance: ArtifactProvenance | None = None,
) -> Path:
    guard = _acquire_merge_input_guard(shard_paths)
    try:
        return _merge_uvlf_shards_guarded(
            shard_paths,
            output_path=output_path,
            overwrite=overwrite,
            final_provenance=final_provenance,
            guard=guard,
        )
    finally:
        guard.close()


__all__ = [
    "merge_uvlf_shards",
    "read_uvlf_artifact",
    "read_uvlf_shard",
    "validate_uvlf_resume_shards",
    "write_uvlf_artifact_atomic",
    "write_uvlf_shard_atomic",
]
