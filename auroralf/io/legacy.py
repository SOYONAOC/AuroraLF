from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat as stat_module
import tomllib
from typing import BinaryIO
import zipfile

import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)
from auroralf.uvlf.dust import compute_dust_attenuated_uvlf
from ._file_ops import (
    file_identity as _file_identity,
    sha256_open_descriptor as _sha256_descriptor,
)
from .hdf5 import read_uvlf_artifact, write_uvlf_artifact_atomic
from .schema import (
    ArtifactProvenance,
    SourceChecksum,
    UVLFArtifact,
    canonical_config_sha256,
)


_MANIFEST_SCHEMA = "auroralf.legacy_uvlf_manifest.v1"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 10_000
_MAX_STRING_ITEMSIZE_BYTES = 4096
_SUPPORTED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
# These limits are deliberately above normal legacy UVLF products while keeping
# one malformed array or archive from reserving multi-gigabyte memory.
_MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
# Compression ratios are checked only for multi-megabyte members so ordinary
# scalar/vector zero arrays in compressed NPZ files remain valid.
_COMPRESSION_RATIO_MIN_MEMBER_BYTES = 4 * 1024 * 1024
_MAX_NPZ_COMPRESSION_RATIO = 200.0
_MANIFEST_KEYS = {
    "schema_version",
    "code_revision",
    "code_dirty",
    "seed_namespace",
    "created_utc",
    "sources",
    "diagnostics",
}
_SOURCE_KEYS = {"label", "path"}
_DIAGNOSTIC_KEYS = {
    "redshift",
    "imf_mode",
    "sampling_seconds",
    "sample_count",
    "valid_sample_count",
    "topheavy_source_fraction",
    "popiii_source_fraction",
    "sfrd_msun_per_yr_per_mpc3",
    "popiii_sfrd_msun_per_yr_per_mpc3",
}
_RESERVED_SOURCE_LABELS = {
    "legacy_uvlf_npz",
    "conversion_config_toml",
    "conversion_manifest",
}
_GLOBAL_ARRAY_KEYS = {
    "z_values",
    "mode_names",
    "variant_mode_names",
    "shared_bin_edges",
}
_GLOBAL_INT_SCALARS = {
    "workers",
    "N_mass",
    "n_tracks",
    "n_grid",
    "tng_min_candidates",
    "thesan_min_candidates",
    "bins_count",
}
_GLOBAL_UINT_SCALARS = {"base_seed"}
_GLOBAL_BOOL_SCALARS = {
    "apply_dust",
    "enable_time_delay",
    "burst_scatter_preserve_mean",
    "burst_scatter_mass_conserving",
    "mzr_metallicity_enabled",
    "regulator_metallicity_enabled",
    "enable_popiii",
    "source_redshift_gate_enabled",
}
_GLOBAL_STRING_SCALARS = {
    "mah_backend",
    "sampler",
    "tng_mah_cache_path",
    "tng_time_grid_mode",
    "thesan_mah_cache_path",
    "thesan_time_grid_mode",
    "mass_function_model",
    "metallicity_source",
    "mzr_relation",
    "canonical_ssp_file",
    "topheavy_ssp_file",
    "popiii_ssp_file",
    "popiii_upper_mass_mode",
}
_GLOBAL_FLOAT_SCALARS = {
    "z_start_max",
    "tng_mass_bin_width_dex",
    "tng_smoothing_myr",
    "thesan_mass_bin_width_dex",
    "thesan_smoothing_myr",
    "muv_min",
    "muv_max",
    "logM_min",
    "logM_max",
    "epsilon_0",
    "fstar_characteristic_mass",
    "fstar_beta",
    "fstar_gamma",
    "burst_scatter_dex",
    "burst_scatter_timescale_myr",
    "mzr_stellar_mass_floor",
    "mzr_scatter_dex",
    "mzr_returned_fraction",
    "regulator_gas_fraction_norm",
    "regulator_gas_fraction_mass_slope",
    "regulator_gas_fraction_redshift_slope",
    "regulator_yield",
    "regulator_returned_fraction",
    "regulator_inflow_metallicity_zsun",
    "regulator_metal_loading_norm",
    "regulator_metal_loading_mass_slope",
    "regulator_metal_loading_redshift_slope",
    "regulator_metallicity_scatter_dex",
    "metallicity_topheavy_max_zsun",
    "topheavy_ssp_metallicity",
    "popiii_epsilon_star",
    "popiii_mp",
    "popiii_alpha_star",
    "popiii_beta_star",
    "popiii_upper_mass_msun",
    "lw_background_j21",
    "z_topheavy_min",
    "growth_time_threshold_myr",
    "total_seconds",
}
_MODE_VECTOR_SUFFIXES = {
    "intrinsic_phi",
    "intrinsic_phi_sigma",
    "phi",
    "phi_sigma_mc",
    "raw_counts",
    "weighted_counts",
    "intrinsic_weighted_counts",
    "weight_squared_counts",
    "effective_counts",
}
_MODE_SCALAR_SUFFIXES = {
    "sampling_seconds",
    "topheavy_source_fraction",
    "topheavy_light_fraction_median",
    "popiii_source_fraction",
    "popiii_light_fraction_median",
}
_MODE_MASS_SUFFIXES = {
    "final_gas_metallicity_zsun_median_by_mass",
    "birth_metallicity_zsun_starforming_median_by_mass",
}


@dataclass(frozen=True, slots=True)
class _Manifest:
    code_revision: str
    code_dirty: bool
    seed_namespace: str
    created_utc: str
    source_paths: tuple[tuple[str, Path], ...]
    diagnostics: tuple[ModeRunDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class _NpySpec:
    shape: tuple[int, ...]
    dtype: np.dtype | None


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"manifest JSON contains non-standard constant: {value}")


def _read_bounded_utf8(path: Path, *, format_name: str) -> str:
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"manifest {format_name} is too large; maximum is "
            f"{_MAX_MANIFEST_BYTES} bytes"
        )
    content = path.read_bytes()
    if len(content) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"manifest {format_name} is too large; maximum is "
            f"{_MAX_MANIFEST_BYTES} bytes"
        )
    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest {format_name} must be strict UTF-8") from error


def _read_strict_json(path: Path) -> object:
    text = _read_bounded_utf8(path, format_name="JSON")
    return json.loads(
        text,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object/table")
    actual = set(value)
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"unknown {name} key: {unknown[0]}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing {name} key: {missing[0]}")
    return value


def _strict_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer non-boolean value")
    return int(value)


def _read_manifest(path: Path, config: UVLFRunConfig) -> _Manifest:
    if path.suffix == ".json":
        root = _read_strict_json(path)
    elif path.suffix == ".toml":
        root = tomllib.loads(_read_bounded_utf8(path, format_name="TOML"))
    else:
        raise ValueError("manifest path must have .toml or .json suffix")
    table = _require_exact_keys(root, _MANIFEST_KEYS, name="manifest")
    if table["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema_version must equal {_MANIFEST_SCHEMA!r}")
    if type(table["code_revision"]) is not str:
        raise TypeError("manifest code_revision must be a string")
    if type(table["code_dirty"]) is not bool:
        raise TypeError("manifest code_dirty must be boolean")
    if type(table["seed_namespace"]) is not str:
        raise TypeError("manifest seed_namespace must be a string")
    if type(table["created_utc"]) is not str:
        raise TypeError("manifest created_utc must be a string")
    sources_value = table["sources"]
    if type(sources_value) is not list or not sources_value:
        raise TypeError("manifest sources must be a non-empty list/array of tables")
    source_paths: list[tuple[str, Path]] = []
    for index, source_value in enumerate(sources_value):
        source = _require_exact_keys(
            source_value,
            _SOURCE_KEYS,
            name=f"manifest source[{index}]",
        )
        if type(source["label"]) is not str or not source["label"]:
            raise TypeError("manifest source label must be a non-empty string")
        if source["label"] in _RESERVED_SOURCE_LABELS:
            raise ValueError(f"manifest source label is reserved: {source['label']}")
        if type(source["path"]) is not str or not source["path"]:
            raise TypeError("manifest source path must be a non-empty string")
        source_path = Path(source["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_paths.append((source["label"], source_path.resolve(strict=True)))
    labels = tuple(label for label, _ in source_paths)
    paths = tuple(source_path for _, source_path in source_paths)
    if len(set(labels)) != len(labels):
        raise ValueError("manifest sources contains a duplicate label")
    if len(set(paths)) != len(paths):
        raise ValueError("manifest sources contains a duplicate path")
    required_active_sources: list[tuple[str, Path]] = [
        ("canonical SSP", config.stellar_population.canonical_ssp_path)
    ]
    if any(
        mode != "canonical" for mode in config.stellar_population.imf_modes
    ):
        required_active_sources.append(
            ("top-heavy SSP", config.stellar_population.topheavy_ssp_path)
        )
    if config.stellar_population.enable_popiii:
        required_active_sources.append(
            ("Pop III SSP", config.stellar_population.popiii_ssp_path)
        )
    if config.mah.backend == "tng":
        if config.mah.tng_cache_path is None:
            raise RuntimeError("active TNG backend has no cache path")
        required_active_sources.append(("TNG MAH cache", config.mah.tng_cache_path))
    elif config.mah.backend == "thesan":
        if config.mah.thesan_cache_path is None:
            raise RuntimeError("active THESAN backend has no cache path")
        required_active_sources.append(
            ("THESAN MAH cache", config.mah.thesan_cache_path)
        )
    source_path_set = set(paths)
    for source_name, required_path in required_active_sources:
        if required_path.resolve() not in source_path_set:
            raise ValueError(
                f"manifest sources must cover required active source: {source_name} "
                f"({required_path.resolve()})"
            )

    diagnostics_value = table["diagnostics"]
    if type(diagnostics_value) is not list:
        raise TypeError("manifest diagnostics must be a list/array of tables")
    diagnostics_by_key: dict[tuple[float, str], ModeRunDiagnostics] = {}
    for index, diagnostic_value in enumerate(diagnostics_value):
        diagnostic = _require_exact_keys(
            diagnostic_value,
            _DIAGNOSTIC_KEYS,
            name=f"manifest diagnostic[{index}]",
        )
        redshift = _strict_float(diagnostic["redshift"], name="manifest redshift")
        if type(diagnostic["imf_mode"]) is not str:
            raise TypeError("manifest imf_mode must be a string")
        item = ModeRunDiagnostics(
            redshift=redshift,
            imf_mode=diagnostic["imf_mode"],
            sampling_seconds=_strict_float(
                diagnostic["sampling_seconds"],
                name="manifest sampling_seconds",
                nonnegative=True,
            ),
            sample_count=_strict_int(
                diagnostic["sample_count"], name="manifest sample_count"
            ),
            valid_sample_count=_strict_int(
                diagnostic["valid_sample_count"],
                name="manifest valid_sample_count",
            ),
            topheavy_source_fraction=_strict_float(
                diagnostic["topheavy_source_fraction"],
                name="manifest topheavy_source_fraction",
            ),
            popiii_source_fraction=_strict_float(
                diagnostic["popiii_source_fraction"],
                name="manifest popiii_source_fraction",
            ),
            sfrd_msun_per_yr_per_mpc3=_strict_float(
                diagnostic["sfrd_msun_per_yr_per_mpc3"],
                name="manifest sfrd_msun_per_yr_per_mpc3",
                nonnegative=True,
            ),
            popiii_sfrd_msun_per_yr_per_mpc3=_strict_float(
                diagnostic["popiii_sfrd_msun_per_yr_per_mpc3"],
                name="manifest popiii_sfrd_msun_per_yr_per_mpc3",
                nonnegative=True,
            ),
        )
        expected_sample_count = (
            config.sampling.n_halo_mass_samples
            * config.sampling.n_tracks_per_halo_mass
        )
        if item.sample_count != expected_sample_count:
            raise ValueError(
                "manifest sample_count must equal legacy N_mass * n_tracks "
                f"({expected_sample_count})"
            )
        key = (item.redshift, item.imf_mode)
        if key in diagnostics_by_key:
            raise ValueError("manifest diagnostics contains a duplicate axis")
        diagnostics_by_key[key] = item
    axes = tuple(
        (redshift, mode)
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )
    if set(diagnostics_by_key) != set(axes):
        raise ValueError("manifest diagnostics must exactly cover every config axis")
    return _Manifest(
        code_revision=table["code_revision"],
        code_dirty=table["code_dirty"],
        seed_namespace=table["seed_namespace"],
        created_utc=table["created_utc"],
        source_paths=tuple(source_paths),
        diagnostics=tuple(diagnostics_by_key[key] for key in axes),
    )


def _z_tag(redshift: float) -> str:
    return f"z{str(float(redshift)).replace('.', 'p')}"


def _expected_npz_keys(config: UVLFRunConfig) -> set[str]:
    expected = (
        set(_GLOBAL_ARRAY_KEYS)
        | set(_GLOBAL_INT_SCALARS)
        | set(_GLOBAL_UINT_SCALARS)
        | set(_GLOBAL_BOOL_SCALARS)
        | set(_GLOBAL_STRING_SCALARS)
        | set(_GLOBAL_FLOAT_SCALARS)
    )
    for redshift in config.redshifts:
        z_tag = _z_tag(redshift)
        expected |= {
            f"{z_tag}_bin_edges",
            f"{z_tag}_bin_centers",
            f"{z_tag}_bin_width",
            f"{z_tag}_base_seed",
        }
        for mode in config.stellar_population.imf_modes:
            prefix = f"{z_tag}_{mode}"
            expected |= {f"{prefix}_{suffix}" for suffix in _MODE_VECTOR_SUFFIXES}
            expected |= {f"{prefix}_{suffix}" for suffix in _MODE_SCALAR_SUFFIXES}
            expected |= {f"{prefix}_{suffix}" for suffix in _MODE_MASS_SUFFIXES}
            if mode != "canonical":
                expected.add(f"{prefix}_phi_ratio_over_canonical")
    return expected


def _expected_npz_specs(config: UVLFRunConfig) -> dict[str, _NpySpec]:
    string = None
    specs = {
        "z_values": _NpySpec((len(config.redshifts),), np.dtype(np.float64)),
        "mode_names": _NpySpec(
            (len(config.stellar_population.imf_modes),), string
        ),
        "variant_mode_names": _NpySpec(
            (len(config.stellar_population.imf_modes) - 1,), string
        ),
        "shared_bin_edges": _NpySpec(
            (len(config.sampling.muv_bin_edges),), np.dtype(np.float64)
        ),
    }
    specs.update(
        {key: _NpySpec((1,), np.dtype(np.int64)) for key in _GLOBAL_INT_SCALARS}
    )
    specs.update(
        {key: _NpySpec((1,), np.dtype(np.uint64)) for key in _GLOBAL_UINT_SCALARS}
    )
    specs.update(
        {key: _NpySpec((1,), np.dtype(np.bool_)) for key in _GLOBAL_BOOL_SCALARS}
    )
    specs.update({key: _NpySpec((1,), string) for key in _GLOBAL_STRING_SCALARS})
    specs.update(
        {key: _NpySpec((1,), np.dtype(np.float64)) for key in _GLOBAL_FLOAT_SCALARS}
    )
    bin_count = len(config.sampling.muv_bin_edges) - 1
    for redshift in config.redshifts:
        z_tag = _z_tag(redshift)
        specs.update(
            {
                f"{z_tag}_bin_edges": _NpySpec(
                    (bin_count + 1,), np.dtype(np.float64)
                ),
                f"{z_tag}_bin_centers": _NpySpec(
                    (bin_count,), np.dtype(np.float64)
                ),
                f"{z_tag}_bin_width": _NpySpec(
                    (bin_count,), np.dtype(np.float64)
                ),
                f"{z_tag}_base_seed": _NpySpec((1,), np.dtype(np.uint64)),
            }
        )
        for mode in config.stellar_population.imf_modes:
            prefix = f"{z_tag}_{mode}"
            for suffix in _MODE_VECTOR_SUFFIXES:
                dtype = np.dtype(np.int64) if suffix == "raw_counts" else np.dtype(np.float64)
                specs[f"{prefix}_{suffix}"] = _NpySpec((bin_count,), dtype)
            for suffix in _MODE_SCALAR_SUFFIXES:
                specs[f"{prefix}_{suffix}"] = _NpySpec((1,), np.dtype(np.float64))
            for suffix in _MODE_MASS_SUFFIXES:
                specs[f"{prefix}_{suffix}"] = _NpySpec(
                    (config.sampling.n_halo_mass_samples,), np.dtype(np.float64)
                )
            if mode != "canonical":
                specs[f"{prefix}_phi_ratio_over_canonical"] = _NpySpec(
                    (bin_count,), np.dtype(np.float64)
                )
    if set(specs) != _expected_npz_keys(config):
        raise RuntimeError("internal legacy NPZ specification is incomplete")
    return specs


def _validate_expected_npz_layout(config: UVLFRunConfig) -> None:
    expected_total = 0
    for key, spec in _expected_npz_specs(config).items():
        itemsize = (
            _MAX_STRING_ITEMSIZE_BYTES
            if spec.dtype is None
            else int(spec.dtype.itemsize)
        )
        data_bytes = math.prod(spec.shape) * itemsize
        member_bytes = _MAX_NPY_HEADER_BYTES + data_bytes
        if member_bytes > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"config-derived legacy NPZ layout exceeds per-member limit for {key}"
            )
        expected_total += member_bytes
    if expected_total > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("config-derived legacy NPZ layout exceeds total limit")


def _read_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    try:
        with archive.open(info, "r") as handle:
            magic = handle.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError("invalid NPY magic")
            version_bytes = handle.read(2)
            if len(version_bytes) != 2:
                raise ValueError("truncated NPY version")
            version = tuple(version_bytes)
            if version == (1, 0):
                length_width = 2
            elif version == (2, 0):
                length_width = 4
            else:
                raise ValueError(
                    f"legacy NPZ member {info.filename} uses unsupported NPY version {version}"
                )
            length_bytes = handle.read(length_width)
            if len(length_bytes) != length_width:
                raise ValueError("truncated NPY header length")
            header_length = int.from_bytes(length_bytes, "little")
            if header_length > _MAX_NPY_HEADER_BYTES:
                raise ValueError(
                    f"legacy NPZ member {info.filename} NPY header is too large: "
                    f"{header_length} bytes"
                )
        with archive.open(info, "r") as handle:
            parsed_version = np.lib.format.read_magic(handle)
            if parsed_version != version:
                raise ValueError("NPY version changed between header reads")
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle,
                    max_header_size=_MAX_NPY_HEADER_BYTES,
                )
            else:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle,
                    max_header_size=_MAX_NPY_HEADER_BYTES,
                )
            header_bytes = handle.tell()
    except (EOFError, OSError, RuntimeError, UnicodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"corrupt legacy NPZ/NPY member: {info.filename}") from error
    except ValueError as error:
        if "unsupported NPY version" in str(error) or "NPY header is too large" in str(error):
            raise
        raise ValueError(f"corrupt legacy NPZ/NPY member: {info.filename}") from error
    return tuple(shape), bool(fortran_order), np.dtype(dtype), header_bytes


def _preflight_npz(source: BinaryIO, config: UVLFRunConfig) -> None:
    specs = _expected_npz_specs(config)
    expected_members = {f"{key}.npy" for key in specs}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("legacy NPZ total uncompressed bytes exceed total limit")
            for info in infos:
                if info.file_size > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        f"legacy NPZ member exceeds uncompressed member limit: {info.filename}"
                    )
                if info.file_size > _COMPRESSION_RATIO_MIN_MEMBER_BYTES:
                    compression_ratio = info.file_size / max(info.compress_size, 1)
                    if compression_ratio > _MAX_NPZ_COMPRESSION_RATIO:
                        raise ValueError(
                            f"legacy NPZ member exceeds compression ratio limit: {info.filename}"
                        )
            names = [info.filename for info in infos]
            if any(info.is_dir() for info in infos):
                raise ValueError("legacy NPZ must not contain directory members")
            if len(names) != len(set(names)):
                raise ValueError("legacy NPZ contains a duplicate member")
            actual_members = set(names)
            unknown = sorted(actual_members - expected_members)
            if unknown:
                raise ValueError(f"unknown legacy NPZ key/member: {unknown[0]}")
            missing = sorted(expected_members - actual_members)
            if missing:
                raise ValueError(f"missing legacy NPZ key/member: {missing[0]}")
            for info in infos:
                key = info.filename[:-4]
                spec = specs[key]
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"legacy NPZ member must not be encrypted: {info.filename}"
                    )
                if info.compress_type not in _SUPPORTED_ZIP_COMPRESSION:
                    raise ValueError(
                        f"unsupported legacy NPZ compression for member: {info.filename}"
                    )
                shape, fortran_order, dtype, header_bytes = _read_npy_header(
                    archive,
                    info,
                )
                if dtype.hasobject:
                    raise TypeError(
                        f"legacy NPZ member must not use object dtype: {info.filename}"
                    )
                if shape != spec.shape:
                    raise ValueError(
                        f"legacy NPZ member {info.filename} has wrong shape {shape}; "
                        f"expected {spec.shape}"
                    )
                if fortran_order:
                    raise ValueError(
                        f"legacy NPZ member must not use Fortran order: {info.filename}"
                    )
                if spec.dtype is None:
                    if dtype.kind not in ("U", "S"):
                        raise TypeError(
                            f"legacy NPZ member {info.filename} must use string dtype"
                        )
                    if dtype.itemsize <= 0 or dtype.itemsize > _MAX_STRING_ITEMSIZE_BYTES:
                        raise ValueError(
                            f"legacy NPZ member {info.filename} string dtype is too large"
                        )
                elif dtype != spec.dtype:
                    raise TypeError(
                        f"legacy NPZ member {info.filename} has wrong dtype {dtype}; "
                        f"expected {spec.dtype}"
                    )
                expected_data_bytes = math.prod(shape) * int(dtype.itemsize)
                if info.file_size - header_bytes != expected_data_bytes:
                    raise ValueError(
                        f"corrupt legacy NPZ member {info.filename}: "
                        "inconsistent NPY data bytes"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError("corrupt legacy NPZ archive") from error


def _array(
    payload: dict[str, np.ndarray],
    key: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool = True,
    allow_nan: bool = False,
) -> np.ndarray:
    value = payload[key]
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"legacy NPZ {key} has wrong dtype {value.dtype}; expected {dtype}")
    if value.shape != shape:
        raise ValueError(f"legacy NPZ {key} has wrong shape {value.shape}; expected {shape}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"legacy NPZ {key} must contain finite values")
    if allow_nan and np.any(np.isinf(value)):
        raise ValueError(f"legacy NPZ {key} must not contain infinity")
    return value


def _string_array(payload: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    value = payload[key]
    if value.shape != shape or value.dtype.kind not in ("U", "S"):
        raise TypeError(f"legacy NPZ {key} must be a strict string array with shape {shape}")
    decoded = value.astype(str)
    return tuple(str(item) for item in decoded.tolist())


def _scalar_float(payload: dict[str, np.ndarray], key: str, *, allow_nan: bool = False) -> float:
    value = _array(
        payload,
        key,
        dtype=np.float64,
        shape=(1,),
        finite=not allow_nan,
        allow_nan=allow_nan,
    )
    return float(value[0])


def _scalar_int(payload: dict[str, np.ndarray], key: str) -> int:
    return int(_array(payload, key, dtype=np.int64, shape=(1,))[0])


def _scalar_bool(payload: dict[str, np.ndarray], key: str) -> bool:
    return bool(_array(payload, key, dtype=np.bool_, shape=(1,), finite=False)[0])


def _scalar_string(payload: dict[str, np.ndarray], key: str) -> str:
    return _string_array(payload, key, (1,))[0]


def _require_equal(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"legacy NPZ {name} does not match v2 config")


def _optional_float_matches(actual: float, expected: float | None, *, name: str) -> None:
    if expected is None:
        if not np.isnan(actual):
            raise ValueError(f"legacy NPZ {name} does not match disabled v2 config value")
    elif actual != expected:
        raise ValueError(f"legacy NPZ {name} does not match v2 config")


def _validate_global_identity(payload: dict[str, np.ndarray], config: UVLFRunConfig) -> None:
    for key in _GLOBAL_INT_SCALARS:
        _scalar_int(payload, key)
    for key in _GLOBAL_BOOL_SCALARS:
        _scalar_bool(payload, key)
    for key in _GLOBAL_STRING_SCALARS:
        _scalar_string(payload, key)
    optional_float_keys = {
        "metallicity_topheavy_max_zsun",
        "topheavy_ssp_metallicity",
        "popiii_upper_mass_msun",
    }
    for key in _GLOBAL_FLOAT_SCALARS:
        _scalar_float(payload, key, allow_nan=key in optional_float_keys)
    redshifts = _array(
        payload,
        "z_values",
        dtype=np.float64,
        shape=(len(config.redshifts),),
    )
    if not np.array_equal(redshifts, np.asarray(config.redshifts, dtype=np.float64)):
        raise ValueError("legacy NPZ redshift axes do not match v2 config")
    modes = _string_array(payload, "mode_names", (len(config.stellar_population.imf_modes),))
    if modes != config.stellar_population.imf_modes:
        raise ValueError("legacy NPZ mode axes do not match v2 config")
    variants = _string_array(payload, "variant_mode_names", (len(modes) - 1,))
    if variants != modes[1:]:
        raise ValueError("legacy NPZ variant mode axes are inconsistent")
    edges = _array(
        payload,
        "shared_bin_edges",
        dtype=np.float64,
        shape=(len(config.sampling.muv_bin_edges),),
    )
    if not np.array_equal(edges, np.asarray(config.sampling.muv_bin_edges)):
        raise ValueError("legacy NPZ bin axes do not match v2 config")
    if payload["base_seed"].dtype != np.dtype(np.uint64) or payload["base_seed"].shape != (1,):
        raise TypeError("legacy NPZ base_seed must be scalar uint64")
    _require_equal(int(payload["base_seed"][0]), config.base_seed, name="base_seed")
    comparisons = (
        (_scalar_int(payload, "workers"), config.sampling.workers, "workers"),
        (_scalar_int(payload, "N_mass"), config.sampling.n_halo_mass_samples, "N_mass"),
        (_scalar_int(payload, "n_tracks"), config.sampling.n_tracks_per_halo_mass, "n_tracks"),
        (_scalar_float(payload, "z_start_max"), config.mah.z_start_max, "z_start_max"),
        (_scalar_int(payload, "n_grid"), config.mah.n_time_steps, "n_grid"),
        (_scalar_string(payload, "mah_backend"), config.mah.backend, "mah_backend"),
        (_scalar_string(payload, "sampler"), config.mah.sampler, "sampler"),
        (_scalar_float(payload, "tng_mass_bin_width_dex"), config.mah.tng_mass_bin_width_dex, "tng_mass_bin_width_dex"),
        (_scalar_int(payload, "tng_min_candidates"), config.mah.tng_min_candidates, "tng_min_candidates"),
        (_scalar_float(payload, "tng_smoothing_myr"), config.mah.tng_smoothing_myr, "tng_smoothing_myr"),
        (_scalar_string(payload, "tng_time_grid_mode"), config.mah.tng_time_grid_mode, "tng_time_grid_mode"),
        (_scalar_float(payload, "thesan_mass_bin_width_dex"), config.mah.thesan_mass_bin_width_dex, "thesan_mass_bin_width_dex"),
        (_scalar_int(payload, "thesan_min_candidates"), config.mah.thesan_min_candidates, "thesan_min_candidates"),
        (_scalar_float(payload, "thesan_smoothing_myr"), config.mah.thesan_smoothing_myr, "thesan_smoothing_myr"),
        (_scalar_string(payload, "thesan_time_grid_mode"), config.mah.thesan_time_grid_mode, "thesan_time_grid_mode"),
        (_scalar_int(payload, "bins_count"), len(config.sampling.muv_bin_edges) - 1, "bins_count"),
        (_scalar_float(payload, "muv_min"), config.sampling.muv_bin_edges[0], "muv_min"),
        (_scalar_float(payload, "muv_max"), config.sampling.muv_bin_edges[-1], "muv_max"),
        (_scalar_float(payload, "logM_min"), config.sampling.log10_halo_mass_min_msun, "logM_min"),
        (_scalar_float(payload, "logM_max"), config.sampling.log10_halo_mass_max_msun, "logM_max"),
        (_scalar_bool(payload, "apply_dust"), config.sampling.apply_dust, "apply_dust"),
        (_scalar_bool(payload, "enable_time_delay"), config.star_formation.enable_time_delay, "enable_time_delay"),
        (_scalar_string(payload, "mass_function_model"), config.sampling.mass_function_model, "mass_function_model"),
        (_scalar_float(payload, "epsilon_0"), config.star_formation.efficiency_normalization, "epsilon_0/efficiency_normalization"),
        (_scalar_float(payload, "fstar_characteristic_mass"), config.star_formation.characteristic_halo_mass_msun, "fstar_characteristic_mass"),
        (_scalar_float(payload, "fstar_beta"), config.star_formation.low_mass_slope, "fstar_beta"),
        (_scalar_float(payload, "fstar_gamma"), config.star_formation.high_mass_slope, "fstar_gamma"),
        (_scalar_float(payload, "burst_scatter_dex"), config.star_formation.burst_scatter_dex, "burst_scatter_dex"),
        (_scalar_float(payload, "burst_scatter_timescale_myr"), config.star_formation.burst_scatter_correlation_timescale_myr, "burst_scatter_timescale_myr"),
        (_scalar_bool(payload, "burst_scatter_mass_conserving"), config.star_formation.burst_scatter_mass_conserving, "burst_scatter_mass_conserving"),
        (_scalar_string(payload, "metallicity_source"), config.star_formation.metallicity_source, "metallicity_source"),
        (_scalar_string(payload, "canonical_ssp_file"), str(config.stellar_population.canonical_ssp_path), "canonical_ssp_file"),
        (_scalar_string(payload, "topheavy_ssp_file"), str(config.stellar_population.topheavy_ssp_path), "topheavy_ssp_file"),
        (_scalar_bool(payload, "enable_popiii"), config.stellar_population.enable_popiii, "enable_popiii"),
        (_scalar_string(payload, "popiii_ssp_file"), str(config.stellar_population.popiii_ssp_path), "popiii_ssp_file"),
        (_scalar_float(payload, "popiii_epsilon_star"), config.stellar_population.popiii_efficiency, "popiii_epsilon_star"),
        (_scalar_float(payload, "popiii_mp"), config.stellar_population.popiii_pivot_halo_mass_msun, "popiii_mp"),
        (_scalar_float(payload, "popiii_alpha_star"), config.stellar_population.popiii_low_mass_slope, "popiii_alpha_star"),
        (_scalar_float(payload, "popiii_beta_star"), config.stellar_population.popiii_high_mass_slope, "popiii_beta_star"),
        (_scalar_string(payload, "popiii_upper_mass_mode"), config.stellar_population.popiii_upper_mass_mode, "popiii_upper_mass_mode"),
        (_scalar_float(payload, "lw_background_j21"), config.stellar_population.lw_background_j21, "lw_background_j21"),
        (_scalar_float(payload, "z_topheavy_min"), config.stellar_population.historical_topheavy_redshift_min, "z_topheavy_min"),
        (_scalar_bool(payload, "source_redshift_gate_enabled"), config.stellar_population.source_redshift_gate_enabled, "source_redshift_gate_enabled"),
        (_scalar_float(payload, "growth_time_threshold_myr"), config.stellar_population.growth_time_threshold_myr, "growth_time_threshold_myr"),
    )
    for actual, expected, name in comparisons:
        _require_equal(actual, expected, name=name)
    _require_equal(
        _scalar_bool(payload, "burst_scatter_preserve_mean"),
        _scalar_bool(payload, "burst_scatter_mass_conserving"),
        name="burst_scatter_preserve_mean",
    )
    _require_equal(
        _scalar_bool(payload, "mzr_metallicity_enabled"),
        config.star_formation.mzr is not None,
        name="mzr_metallicity_enabled",
    )
    _require_equal(
        _scalar_bool(payload, "regulator_metallicity_enabled"),
        config.star_formation.regulator is not None,
        name="regulator_metallicity_enabled",
    )
    _optional_float_matches(
        _scalar_float(payload, "metallicity_topheavy_max_zsun", allow_nan=True),
        config.stellar_population.birth_metallicity_topheavy_max_zsun,
        name="metallicity_topheavy_max_zsun",
    )
    _optional_float_matches(
        _scalar_float(payload, "topheavy_ssp_metallicity", allow_nan=True),
        config.stellar_population.topheavy_ssp_template_metallicity_zsun,
        name="topheavy_ssp_metallicity",
    )
    _optional_float_matches(
        _scalar_float(payload, "popiii_upper_mass_msun", allow_nan=True),
        config.stellar_population.popiii_upper_mass_msun,
        name="popiii_upper_mass_msun",
    )
    _require_equal(_scalar_string(payload, "tng_mah_cache_path"), "" if config.mah.tng_cache_path is None else str(config.mah.tng_cache_path), name="tng_mah_cache_path")
    _require_equal(_scalar_string(payload, "thesan_mah_cache_path"), "" if config.mah.thesan_cache_path is None else str(config.mah.thesan_cache_path), name="thesan_mah_cache_path")
    if config.star_formation.mzr is not None:
        mzr = config.star_formation.mzr
        for actual, expected, name in (
            (_scalar_string(payload, "mzr_relation"), mzr.relation, "mzr_relation"),
            (_scalar_float(payload, "mzr_stellar_mass_floor"), mzr.stellar_mass_floor_msun, "mzr_stellar_mass_floor"),
            (_scalar_float(payload, "mzr_scatter_dex"), mzr.scatter_dex, "mzr_scatter_dex"),
            (_scalar_float(payload, "mzr_returned_fraction"), mzr.returned_fraction, "mzr_returned_fraction"),
        ):
            _require_equal(actual, expected, name=name)
    if config.star_formation.regulator is not None:
        regulator = config.star_formation.regulator
        for key, expected in (
            ("regulator_gas_fraction_norm", regulator.gas_fraction_norm),
            ("regulator_gas_fraction_mass_slope", regulator.gas_fraction_mass_slope),
            ("regulator_gas_fraction_redshift_slope", regulator.gas_fraction_redshift_slope),
            ("regulator_yield", regulator.metal_yield),
            ("regulator_returned_fraction", regulator.returned_fraction),
            ("regulator_inflow_metallicity_zsun", regulator.inflow_metallicity_zsun),
            ("regulator_metal_loading_norm", regulator.metal_loading_norm),
            ("regulator_metal_loading_mass_slope", regulator.metal_loading_mass_slope),
            ("regulator_metal_loading_redshift_slope", regulator.metal_loading_redshift_slope),
            ("regulator_metallicity_scatter_dex", regulator.metallicity_scatter_dex),
        ):
            _require_equal(_scalar_float(payload, key), expected, name=key)


def _verify_open_npz_stability(
    descriptor: int,
    path: Path,
    initial_stat: os.stat_result,
) -> None:
    current_descriptor_stat = os.fstat(descriptor)
    try:
        current_path_stat = path.stat()
    except FileNotFoundError as error:
        raise ValueError("legacy NPZ path identity changed during conversion") from error
    if not (
        _file_identity(initial_stat)
        == _file_identity(current_descriptor_stat)
        == _file_identity(current_path_stat)
    ):
        raise ValueError("legacy NPZ identity or metadata changed during conversion")


def _duplicate_descriptor_file(descriptor: int) -> BinaryIO:
    os.lseek(descriptor, 0, os.SEEK_SET)
    return os.fdopen(os.dup(descriptor), "rb", closefd=True)


def _load_npz(
    path: Path,
    config: UVLFRunConfig,
    expected_checksum: SourceChecksum,
) -> dict[str, np.ndarray]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        initial_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(initial_stat.st_mode):
            raise ValueError(f"legacy NPZ path must be a regular file: {path}")
        if initial_stat.st_size != expected_checksum.size_bytes:
            raise ValueError("legacy NPZ size changed before preflight")
        initial_sha256 = _sha256_descriptor(descriptor)
        if initial_sha256 != expected_checksum.sha256:
            raise ValueError("legacy NPZ checksum changed before preflight")
        _verify_open_npz_stability(descriptor, path, initial_stat)

        with _duplicate_descriptor_file(descriptor) as preflight_source:
            _preflight_npz(preflight_source, config)
        _verify_open_npz_stability(descriptor, path, initial_stat)

        with _duplicate_descriptor_file(descriptor) as materialization_source:
            with np.load(materialization_source, allow_pickle=False) as archive:
                expected = _expected_npz_keys(config)
                actual = set(archive.files)
                unknown = sorted(actual - expected)
                if unknown:
                    raise ValueError(f"unknown legacy NPZ key: {unknown[0]}")
                missing = sorted(expected - actual)
                if missing:
                    raise ValueError(f"missing legacy NPZ key: {missing[0]}")
                try:
                    payload = {
                        key: np.array(archive[key], copy=True)
                        for key in archive.files
                    }
                except (OSError, ValueError, zipfile.BadZipFile) as error:
                    raise ValueError("corrupt legacy NPZ array payload") from error

        _verify_open_npz_stability(descriptor, path, initial_stat)
        if _sha256_descriptor(descriptor) != initial_sha256:
            raise ValueError("legacy NPZ checksum changed during conversion")
        return payload
    finally:
        os.close(descriptor)


def _build_result(
    payload: dict[str, np.ndarray],
    config: UVLFRunConfig,
    manifest: _Manifest,
) -> UVLFRunResult:
    _validate_global_identity(payload, config)
    diagnostics_by_key = {
        (item.redshift, item.imf_mode): item for item in manifest.diagnostics
    }
    redshift_results: list[RedshiftResult] = []
    mode_diagnostics: list[ModeRunDiagnostics] = []
    bin_count = len(config.sampling.muv_bin_edges) - 1
    for redshift in config.redshifts:
        z_tag = _z_tag(redshift)
        edges = _array(payload, f"{z_tag}_bin_edges", dtype=np.float64, shape=(bin_count + 1,))
        centers = _array(payload, f"{z_tag}_bin_centers", dtype=np.float64, shape=(bin_count,))
        widths = _array(payload, f"{z_tag}_bin_width", dtype=np.float64, shape=(bin_count,))
        if not np.array_equal(edges, np.asarray(config.sampling.muv_bin_edges)):
            raise ValueError(f"legacy NPZ bin edges mismatch at redshift {redshift}")
        if not np.array_equal(centers, 0.5 * (edges[:-1] + edges[1:])):
            raise ValueError(f"legacy NPZ bin centers are inconsistent at redshift {redshift}")
        if not np.array_equal(widths, np.diff(edges)):
            raise ValueError(f"legacy NPZ bin widths are inconsistent at redshift {redshift}")
        seed_key = f"{z_tag}_base_seed"
        if payload[seed_key].dtype != np.dtype(np.uint64) or payload[seed_key].shape != (1,):
            raise TypeError(f"legacy NPZ {seed_key} must be scalar uint64")
        _require_equal(int(payload[seed_key][0]), config.base_seed, name=seed_key)
        modes: list[IMFModeResult] = []
        canonical_phi: np.ndarray | None = None
        for mode in config.stellar_population.imf_modes:
            prefix = f"{z_tag}_{mode}"
            raw_counts = _array(payload, f"{prefix}_raw_counts", dtype=np.int64, shape=(bin_count,))
            vectors = {
                suffix: _array(
                    payload,
                    f"{prefix}_{suffix}",
                    dtype=np.float64,
                    shape=(bin_count,),
                    finite=suffix != "phi_sigma_mc",
                    allow_nan=suffix == "phi_sigma_mc",
                )
                for suffix in _MODE_VECTOR_SUFFIXES - {"raw_counts"}
            }
            if np.any(raw_counts < 0) or any(np.any(value < 0.0) for value in vectors.values()):
                raise ValueError(f"legacy NPZ result values must be non-negative for {prefix}")
            if not np.array_equal(vectors["weighted_counts"], vectors["phi"] * widths):
                raise ValueError(f"legacy NPZ observed weighted_counts is inconsistent for {prefix}")
            if not np.array_equal(
                vectors["intrinsic_weighted_counts"],
                vectors["intrinsic_phi"] * widths,
            ):
                raise ValueError(f"legacy NPZ intrinsic_weighted_counts is inconsistent for {prefix}")
            if not np.array_equal(
                np.sqrt(vectors["weight_squared_counts"]),
                vectors["intrinsic_phi_sigma"] * widths,
            ):
                raise ValueError(
                    f"legacy NPZ weight_squared_counts is inconsistent for {prefix}"
                )
            expected_effective_counts = np.divide(
                vectors["intrinsic_weighted_counts"] ** 2,
                vectors["weight_squared_counts"],
                out=np.zeros(bin_count),
                where=vectors["weight_squared_counts"] > 0.0,
            )
            if not np.array_equal(
                vectors["effective_counts"],
                expected_effective_counts,
            ):
                raise ValueError(
                    f"legacy NPZ effective_counts is inconsistent for {prefix}"
                )
            if config.sampling.apply_dust:
                expected_observed_phi = np.asarray(
                    compute_dust_attenuated_uvlf(
                        intrinsic_muv=centers,
                        intrinsic_phi=vectors["intrinsic_phi"],
                        z=redshift,
                        muv_obs=centers,
                    )["phi_obs"],
                    dtype=np.float64,
                )
                if not np.array_equal(vectors["phi"], expected_observed_phi):
                    raise ValueError(
                        f"legacy NPZ dust observed phi is inconsistent for {prefix}"
                    )
            else:
                if not np.array_equal(vectors["phi"], vectors["intrinsic_phi"]):
                    raise ValueError(
                        f"legacy NPZ observed phi must equal intrinsic phi when apply_dust=False for {prefix}"
                    )
                if not np.array_equal(
                    vectors["phi_sigma_mc"],
                    vectors["intrinsic_phi_sigma"],
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"legacy NPZ phi_sigma_mc is inconsistent for {prefix}: "
                        "observed sigma must equal intrinsic sigma when apply_dust=False"
                    )
            expected_observed_sigma = vectors["phi"] * np.divide(
                vectors["intrinsic_phi_sigma"],
                vectors["intrinsic_phi"],
                out=np.full(bin_count, np.nan),
                where=vectors["intrinsic_phi"] > 0.0,
            )
            if not np.array_equal(
                vectors["phi_sigma_mc"],
                expected_observed_sigma,
                equal_nan=True,
            ):
                raise ValueError(f"legacy NPZ phi_sigma_mc is inconsistent for {prefix}")
            for suffix in _MODE_SCALAR_SUFFIXES:
                scalar = _scalar_float(payload, f"{prefix}_{suffix}")
                if "fraction" in suffix and not 0.0 <= scalar <= 1.0:
                    raise ValueError(f"legacy NPZ {prefix}_{suffix} must lie in [0, 1]")
            for suffix in _MODE_MASS_SUFFIXES:
                metallicity = _array(
                    payload,
                    f"{prefix}_{suffix}",
                    dtype=np.float64,
                    shape=(config.sampling.n_halo_mass_samples,),
                    finite=False,
                    allow_nan=True,
                )
                if np.any(metallicity[np.isfinite(metallicity)] < 0.0):
                    raise ValueError(
                        f"legacy NPZ metallicity values must be non-negative for {prefix}_{suffix}"
                    )
            if mode == "canonical":
                canonical_phi = vectors["phi"]
            else:
                ratio = _array(
                    payload,
                    f"{prefix}_phi_ratio_over_canonical",
                    dtype=np.float64,
                    shape=(bin_count,),
                    finite=False,
                    allow_nan=True,
                )
                if canonical_phi is None:
                    raise ValueError(
                        "v2 config must place canonical before every variant IMF mode"
                    )
                expected_ratio = np.divide(
                    vectors["phi"],
                    canonical_phi,
                    out=np.full(bin_count, np.nan),
                    where=canonical_phi > 0.0,
                )
                if not np.array_equal(ratio, expected_ratio, equal_nan=True):
                    raise ValueError(f"legacy NPZ phi ratio is inconsistent for {prefix}")
            modes.append(
                IMFModeResult(
                    imf_mode=mode,
                    bin_edges_muv=edges,
                    bin_centers_muv=centers,
                    bin_width_mag=widths,
                    raw_counts=raw_counts,
                    weighted_counts_per_mpc3=vectors["intrinsic_weighted_counts"],
                    weight_squared_counts_per_mpc6=vectors["weight_squared_counts"],
                    weighted_count_sigma_per_mpc3=np.sqrt(vectors["weight_squared_counts"]),
                    effective_counts=vectors["effective_counts"],
                    phi_intrinsic_per_mpc3_per_mag=vectors["intrinsic_phi"],
                    phi_intrinsic_sigma_per_mpc3_per_mag=vectors["intrinsic_phi_sigma"],
                    phi_observed_per_mpc3_per_mag=vectors["phi"],
                    phi_observed_sigma_per_mpc3_per_mag=vectors["phi_sigma_mc"],
                    halo_tracks=(),
                )
            )
            diagnostic = diagnostics_by_key[(redshift, mode)]
            raw_sample_count = sum(int(value) for value in raw_counts)
            if raw_sample_count > diagnostic.valid_sample_count:
                raise ValueError(
                    f"legacy NPZ sum(raw_counts) exceeds manifest valid_sample_count for {prefix}"
                )
            for suffix, expected in (
                ("sampling_seconds", diagnostic.sampling_seconds),
                ("topheavy_source_fraction", diagnostic.topheavy_source_fraction),
                ("popiii_source_fraction", diagnostic.popiii_source_fraction),
            ):
                actual = _scalar_float(payload, f"{prefix}_{suffix}")
                if actual != expected:
                    raise ValueError(f"manifest {suffix} does not match legacy NPZ for {prefix}")
            mode_diagnostics.append(diagnostic)
        redshift_results.append(RedshiftResult(redshift=redshift, imf_modes=tuple(modes)))
    total_seconds = _scalar_float(payload, "total_seconds")
    if total_seconds < 0.0:
        raise ValueError("legacy NPZ total_seconds must be non-negative")
    return UVLFRunResult(
        config=config,
        redshifts=tuple(redshift_results),
        diagnostics=RunDiagnostics(
            total_seconds=total_seconds,
            mode_runs=tuple(mode_diagnostics),
        ),
    )


def convert_legacy_uvlf_npz(
    npz_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be exactly boolean")
    npz = Path(npz_path).expanduser().resolve(strict=True)
    config_source = Path(config_path).expanduser().resolve(strict=True)
    manifest_source = Path(manifest_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    if output.suffix != ".h5":
        raise ValueError("output_path must be an absolute .h5 path")
    bound_inputs = (
        SourceChecksum.from_path("legacy_uvlf_npz", npz),
        SourceChecksum.from_path("conversion_config_toml", config_source),
        SourceChecksum.from_path("conversion_manifest", manifest_source),
    )
    config = UVLFRunConfig.from_toml(config_source)
    _validate_expected_npz_layout(config)
    manifest = _read_manifest(manifest_source, config)
    payload = _load_npz(npz, config, bound_inputs[0])
    result = _build_result(payload, config, manifest)
    for source in bound_inputs:
        source.verify()
    scientific_sources = tuple(
        SourceChecksum.from_path(label, path)
        for label, path in manifest.source_paths
    )
    provenance = ArtifactProvenance(
        config_sha256=canonical_config_sha256(config),
        code_revision=manifest.code_revision,
        code_dirty=manifest.code_dirty,
        seed_namespace=manifest.seed_namespace,
        created_utc=manifest.created_utc,
        source_checksums=(*scientific_sources, *bound_inputs),
    )
    artifact = UVLFArtifact(result=result, provenance=provenance)
    written = write_uvlf_artifact_atomic(
        artifact,
        path=output,
        overwrite=overwrite,
    )
    read_uvlf_artifact(written, load_samples=False)
    return written


__all__ = ["convert_legacy_uvlf_npz"]
