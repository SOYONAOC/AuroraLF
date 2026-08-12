"""Strict HDF5 encoding and decoding for AuroraLF UVLF payloads."""

from __future__ import annotations

import h5py
import numpy as np
import os
from pathlib import Path

from auroralf.config import UVLFRunConfig
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)
from .hdf5_layout import (
    _DIAGNOSTIC_FIELDS,
    _DIAGNOSTIC_UNITS,
    _RESULT_FIELDS,
    _RESULT_UNITS,
    _ROOT_ATTRS,
    _ROOT_GROUPS,
    _SAMPLE_FIELDS,
    _SAMPLE_UNITS,
)
from .hdf5_ownership import _OwnedFile, _require_owned_path
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
