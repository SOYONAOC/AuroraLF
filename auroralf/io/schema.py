from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat as stat_module
from typing import Any, Mapping

import numpy as np

from auroralf.config import (
    CosmologyConfig,
    MAHConfig,
    MZRConfig,
    OutputConfig,
    RegulatorConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
    _decode_run_config,
)
from auroralf.results import IMFModeResult, ModeRunDiagnostics, UVLFRunResult
from auroralf.uvlf.imf import validate_imf_mode
from ._file_ops import (
    file_identity as _file_identity,
    sha256_open_descriptor as _hash_open_file_descriptor,
)


SCHEMA_NAME = "auroralf.uvlf"
SCHEMA_VERSION = "2.0.0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSIONED_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.v[1-9][0-9]*")


def _strict_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean value")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer non-boolean value")
    return int(value)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.flags.writeable = False
    return result


def _strict_float_array(name: str, value: object, *, allow_nan: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype == np.dtype(bool) or np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real non-boolean values")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinity")
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_array(array)


def _strict_int_array(name: str, value: object) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype == np.dtype(bool) or not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{name} must contain integer non-boolean values")
    array = np.array(value, dtype=np.int64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return _immutable_array(array)


def _stable_file_checksum(path: Path) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise ValueError(f"source path must be a regular file: {path}")
        digest = _hash_open_file_descriptor(descriptor)
        after = os.fstat(descriptor)
        current = path.stat()
        if not (
            _file_identity(before)
            == _file_identity(after)
            == _file_identity(current)
        ):
            raise ValueError(f"source file changed identity during checksum: {path}")
        return digest, after.st_size
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return _stable_file_checksum(path)[0]


def _config_value(value: object) -> object:
    if isinstance(value, Path):
        if not value.is_absolute():
            raise ValueError("config Paths must be absolute")
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _config_value(getattr(value, field.name)) for field in fields(value)}
    if type(value) is tuple:
        return [_config_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"unsupported canonical config value: {type(value).__name__}")


def canonical_config_mapping(config: UVLFRunConfig) -> dict[str, object]:
    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    mapping = _config_value(config)
    if type(mapping) is not dict:
        raise RuntimeError("canonical config mapping must be a dictionary")
    return mapping


def canonical_config_json(config: UVLFRunConfig) -> str:
    return json.dumps(
        canonical_config_mapping(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_config_sha256(config: UVLFRunConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


_CONFIG_TYPES = {
    "root": UVLFRunConfig,
    "cosmology": CosmologyConfig,
    "mah": MAHConfig,
    "star_formation": StarFormationConfig,
    "stellar_population": StellarPopulationConfig,
    "sampling": SamplingConfig,
    "output": OutputConfig,
    "star_formation.mzr": MZRConfig,
    "star_formation.regulator": RegulatorConfig,
}


def _require_exact_keys(value: object, *, name: str, model: type[object]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    expected = {field.name for field in fields(model)}
    actual = set(value)
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"unknown canonical config key: {name}.{unknown[0]}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing canonical config key: {name}.{missing[0]}")
    return value


def _strict_same_json_types(actual: object, expected: object, *, name: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(
            f"canonical config {name} must have type {type(expected).__name__}"
        )
    if type(actual) is dict:
        for key in actual:
            _strict_same_json_types(actual[key], expected[key], name=f"{name}.{key}")
    elif type(actual) is list:
        if len(actual) != len(expected):
            raise ValueError(f"canonical config {name} length changed during decode")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _strict_same_json_types(left, right, name=f"{name}[{index}]")


def decode_canonical_config_mapping(root: Mapping[str, object]) -> UVLFRunConfig:
    root_table = _require_exact_keys(root, name="root", model=UVLFRunConfig)
    for name in (
        "cosmology",
        "mah",
        "star_formation",
        "stellar_population",
        "sampling",
        "output",
    ):
        _require_exact_keys(root_table[name], name=name, model=_CONFIG_TYPES[name])
    star = root_table["star_formation"]
    assert type(star) is dict
    for nested in ("mzr", "regulator"):
        if star[nested] is not None:
            _require_exact_keys(
                star[nested],
                name=f"star_formation.{nested}",
                model=_CONFIG_TYPES[f"star_formation.{nested}"],
            )

    normalized = json.loads(json.dumps(root_table, allow_nan=False))
    mah = normalized["mah"]
    for name in ("tng_cache_path", "thesan_cache_path"):
        if mah[name] is None:
            del mah[name]
    star = normalized["star_formation"]
    for name in ("mzr", "regulator"):
        if star[name] is None:
            del star[name]
    population = normalized["stellar_population"]
    for name in (
        "topheavy_ssp_template_metallicity_zsun",
        "birth_metallicity_topheavy_max_zsun",
        "popiii_upper_mass_msun",
    ):
        if population[name] is None:
            del population[name]
    config = _decode_run_config(normalized, Path("/"))
    expected = canonical_config_mapping(config)
    _strict_same_json_types(root_table, expected, name="root")
    if root_table != expected:
        raise ValueError("canonical config values changed during strict decode")
    return config


def decode_canonical_config_json(value: str) -> UVLFRunConfig:
    text = _strict_string("canonical_json", value)

    def reject_constant(token: str) -> object:
        raise ValueError(f"canonical config JSON contains non-finite token {token}")

    root = json.loads(text, parse_constant=reject_constant)
    return decode_canonical_config_mapping(root)


@dataclass(frozen=True, slots=True)
class SourceChecksum:
    label: str
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        label = _strict_string("label", self.label)
        if _LABEL_PATTERN.fullmatch(label) is None:
            raise ValueError("label must contain only letters, digits, dot, underscore, or hyphen")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        path = self.path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("path must be a regular file")
        sha256 = _strict_string("sha256", self.sha256)
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError("sha256 must be lowercase 64-character hexadecimal")
        size = _strict_int("size_bytes", self.size_bytes)
        if size < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "size_bytes", size)

    @classmethod
    def from_path(cls, label: str, path: str | Path) -> SourceChecksum:
        resolved = Path(path).expanduser().resolve(strict=True)
        sha256, size = _stable_file_checksum(resolved)
        result = cls(
            label=label,
            path=resolved,
            sha256=sha256,
            size_bytes=size,
        )
        result.verify()
        return result

    def verify(self) -> None:
        current_sha, current_size = _stable_file_checksum(self.path)
        if current_size != self.size_bytes:
            raise ValueError(f"source size mismatch for {self.label}: {self.path}")
        if current_sha != self.sha256:
            raise ValueError(f"source checksum mismatch for {self.label}: {self.path}")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    config_sha256: str
    code_revision: str
    code_dirty: bool
    seed_namespace: str
    created_utc: str
    source_checksums: tuple[SourceChecksum, ...]

    def __post_init__(self) -> None:
        config_sha = _strict_string("config_sha256", self.config_sha256)
        if _SHA256_PATTERN.fullmatch(config_sha) is None:
            raise ValueError("config_sha256 must be lowercase 64-character hexadecimal")
        revision = _strict_string("code_revision", self.code_revision)
        if _REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError("code_revision must be a lowercase 40-character hexadecimal revision")
        if type(self.code_dirty) is not bool:
            raise TypeError("code_dirty must be exactly boolean")
        namespace = _strict_string("seed_namespace", self.seed_namespace)
        if _VERSIONED_NAMESPACE_PATTERN.fullmatch(namespace) is None:
            raise ValueError("seed_namespace must be a non-empty versioned namespace ending in .vN")
        created = _strict_string("created_utc", self.created_utc)
        if not created.endswith("Z"):
            raise ValueError("created_utc must be an ISO-8601 UTC timestamp ending in Z")
        try:
            parsed = datetime.fromisoformat(created[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("created_utc must be a valid ISO-8601 UTC timestamp") from error
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_utc must use UTC")
        if type(self.source_checksums) is not tuple or not self.source_checksums:
            raise TypeError("source_checksums must be a non-empty tuple")
        if any(type(item) is not SourceChecksum for item in self.source_checksums):
            raise TypeError("source_checksums entries must be exactly SourceChecksum")
        labels = tuple(item.label for item in self.source_checksums)
        paths = tuple(item.path for item in self.source_checksums)
        if len(set(labels)) != len(labels):
            raise ValueError("source_checksums contains duplicate label")
        if len(set(paths)) != len(paths):
            raise ValueError("source_checksums contains duplicate path")
        object.__setattr__(self, "config_sha256", config_sha)
        object.__setattr__(self, "code_revision", revision)
        object.__setattr__(self, "seed_namespace", namespace)
        object.__setattr__(self, "created_utc", created)

    @classmethod
    def for_config(
        cls,
        config: UVLFRunConfig,
        *,
        code_revision: str,
        code_dirty: bool,
        seed_namespace: str,
        source_paths: tuple[tuple[str, Path], ...],
        created_utc: str | None = None,
    ) -> ArtifactProvenance:
        if type(source_paths) is not tuple:
            raise TypeError("source_paths must be a tuple")
        timestamp = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if created_utc is None
            else created_utc
        )
        return cls(
            config_sha256=canonical_config_sha256(config),
            code_revision=code_revision,
            code_dirty=code_dirty,
            seed_namespace=seed_namespace,
            created_utc=timestamp,
            source_checksums=tuple(
                SourceChecksum.from_path(label, path) for label, path in source_paths
            ),
        )

    def verify_sources(self) -> None:
        for source in self.source_checksums:
            source.verify()


@dataclass(frozen=True, slots=True)
class HaloSampleTable:
    redshift: float
    imf_mode: str
    mass_index: np.ndarray
    track_index: np.ndarray
    halo_mass_msun: np.ndarray
    mass_weight_per_mpc3: np.ndarray
    uv_luminosity_erg_per_s_hz: np.ndarray
    muv: np.ndarray
    sfr_msun_per_yr: np.ndarray
    popiii_sfr_msun_per_yr: np.ndarray

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        mass_index = _strict_int_array("mass_index", self.mass_index)
        track_index = _strict_int_array("track_index", self.track_index)
        halo_mass = _strict_float_array("halo_mass_msun", self.halo_mass_msun)
        mass_weight = _strict_float_array(
            "mass_weight_per_mpc3", self.mass_weight_per_mpc3
        )
        luminosity = _strict_float_array(
            "uv_luminosity_erg_per_s_hz", self.uv_luminosity_erg_per_s_hz
        )
        muv = _strict_float_array("muv", self.muv, allow_nan=True)
        sfr = _strict_float_array("sfr_msun_per_yr", self.sfr_msun_per_yr)
        popiii_sfr = _strict_float_array(
            "popiii_sfr_msun_per_yr", self.popiii_sfr_msun_per_yr
        )
        arrays = (
            track_index,
            halo_mass,
            mass_weight,
            luminosity,
            muv,
            sfr,
            popiii_sfr,
        )
        if any(array.size != mass_index.size for array in arrays):
            raise ValueError("all HaloSampleTable arrays must have the same length")
        if np.any(mass_index < 0) or np.any(track_index < 0):
            raise ValueError("sample indices must be non-negative")
        if np.any(halo_mass <= 0.0):
            raise ValueError("halo_mass_msun must be positive")
        for name, array in (
            ("mass_weight_per_mpc3", mass_weight),
            ("uv_luminosity_erg_per_s_hz", luminosity),
            ("sfr_msun_per_yr", sfr),
            ("popiii_sfr_msun_per_yr", popiii_sfr),
        ):
            if np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("redshift", redshift),
            ("imf_mode", mode),
            ("mass_index", mass_index),
            ("track_index", track_index),
            ("halo_mass_msun", halo_mass),
            ("mass_weight_per_mpc3", mass_weight),
            ("uv_luminosity_erg_per_s_hz", luminosity),
            ("muv", muv),
            ("sfr_msun_per_yr", sfr),
            ("popiii_sfr_msun_per_yr", popiii_sfr),
        ):
            object.__setattr__(self, name, value)

    @property
    def key(self) -> tuple[float, str]:
        return (self.redshift, self.imf_mode)


@dataclass(frozen=True, slots=True)
class HaloSampleDescriptor:
    redshift: float
    imf_mode: str
    sample_count: int

    def __post_init__(self) -> None:
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        sample_count = _strict_int("sample_count", self.sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        object.__setattr__(self, "redshift", redshift)
        object.__setattr__(self, "imf_mode", mode)
        object.__setattr__(self, "sample_count", sample_count)

    @property
    def key(self) -> tuple[float, str]:
        return (self.redshift, self.imf_mode)


@dataclass(frozen=True, slots=True)
class UVLFArtifact:
    result: UVLFRunResult
    provenance: ArtifactProvenance
    sample_descriptors: tuple[HaloSampleDescriptor, ...] = ()
    samples: tuple[HaloSampleTable, ...] = ()

    def __post_init__(self) -> None:
        if type(self.result) is not UVLFRunResult:
            raise TypeError("result must be exactly UVLFRunResult")
        if type(self.provenance) is not ArtifactProvenance:
            raise TypeError("provenance must be exactly ArtifactProvenance")
        if self.provenance.config_sha256 != canonical_config_sha256(self.result.config):
            raise ValueError("provenance config hash does not match result.config")
        if type(self.sample_descriptors) is not tuple:
            raise TypeError("sample_descriptors must be a tuple")
        if type(self.samples) is not tuple:
            raise TypeError("samples must be a tuple")
        configured_axes = tuple(
            (redshift, mode)
            for redshift in self.result.config.redshifts
            for mode in self.result.config.stellar_population.imf_modes
        )
        if any(
            type(descriptor) is not HaloSampleDescriptor
            for descriptor in self.sample_descriptors
        ):
            raise TypeError(
                "sample_descriptors entries must be exactly HaloSampleDescriptor"
            )
        normalized_keys: list[tuple[float, str]] = []
        for descriptor in self.sample_descriptors:
            key = descriptor.key
            if key not in configured_axes:
                raise ValueError("sample descriptors must lie on configured result axes")
            normalized_keys.append(key)
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("sample descriptors must have unique keys")
        expected_order = tuple(key for key in configured_axes if key in normalized_keys)
        if tuple(normalized_keys) != expected_order:
            raise ValueError("sample descriptors must follow configured axis order")
        if any(type(sample) is not HaloSampleTable for sample in self.samples):
            raise TypeError("samples entries must be exactly HaloSampleTable")
        if self.samples:
            sample_descriptors = tuple(
                HaloSampleDescriptor(
                    sample.redshift,
                    sample.imf_mode,
                    sample.mass_index.size,
                )
                for sample in self.samples
            )
            if sample_descriptors != self.sample_descriptors:
                raise ValueError("sample descriptors must exactly match samples")

    @property
    def sample_keys(self) -> tuple[tuple[float, str], ...]:
        return tuple(descriptor.key for descriptor in self.sample_descriptors)


@dataclass(frozen=True, slots=True)
class UVLFShardDescriptor:
    path: Path
    redshift: float
    imf_mode: str
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("shard descriptor path must be an absolute pathlib.Path")
        path = self.path.resolve()
        if path.suffix != ".h5":
            raise ValueError("shard descriptor path must be an absolute .h5 path")
        redshift = _strict_float("redshift", self.redshift)
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if type(self.imf_mode) is not str:
            raise TypeError("imf_mode must be a string")
        mode = validate_imf_mode(self.imf_mode)
        sample_count = self.sample_count
        if sample_count is not None:
            sample_count = _strict_int("sample_count", sample_count)
            if sample_count <= 0:
                raise ValueError("sample_count must be positive when provided")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "redshift", redshift)
        object.__setattr__(self, "imf_mode", mode)
        object.__setattr__(self, "sample_count", sample_count)

    @property
    def key(self) -> tuple[float, str]:
        return (self.redshift, self.imf_mode)


@dataclass(frozen=True, slots=True)
class UVLFShard:
    config: UVLFRunConfig
    provenance: ArtifactProvenance
    result: IMFModeResult
    diagnostic: ModeRunDiagnostics
    sample_descriptor: HaloSampleDescriptor | None = None
    sample: HaloSampleTable | None = None

    def __post_init__(self) -> None:
        if type(self.config) is not UVLFRunConfig:
            raise TypeError("config must be exactly UVLFRunConfig")
        if type(self.provenance) is not ArtifactProvenance:
            raise TypeError("provenance must be exactly ArtifactProvenance")
        if type(self.result) is not IMFModeResult:
            raise TypeError("result must be exactly IMFModeResult")
        if type(self.diagnostic) is not ModeRunDiagnostics:
            raise TypeError("diagnostic must be exactly ModeRunDiagnostics")
        if self.provenance.config_sha256 != canonical_config_sha256(self.config):
            raise ValueError("provenance config hash does not match shard config")
        key = (self.diagnostic.redshift, self.diagnostic.imf_mode)
        configured_axes = tuple(
            (redshift, mode)
            for redshift in self.config.redshifts
            for mode in self.config.stellar_population.imf_modes
        )
        if key not in configured_axes:
            raise ValueError("shard key must lie on configured axes")
        if self.result.imf_mode != self.diagnostic.imf_mode:
            raise ValueError("shard result and diagnostic mode must match")
        if not np.array_equal(
            self.result.bin_edges_muv,
            np.asarray(self.config.sampling.muv_bin_edges, dtype=np.float64),
        ):
            raise ValueError("shard result bin edges must match configured axes")
        descriptor = self.sample_descriptor
        sample = self.sample
        if descriptor is not None:
            if type(descriptor) is not HaloSampleDescriptor:
                raise TypeError(
                    "sample_descriptor must be exactly HaloSampleDescriptor or None"
                )
            if descriptor.key != key:
                raise ValueError("sample descriptor must match the shard key")
        if sample is not None:
            if type(sample) is not HaloSampleTable:
                raise TypeError("sample must be exactly HaloSampleTable or None")
            expected = HaloSampleDescriptor(
                sample.redshift,
                sample.imf_mode,
                sample.mass_index.size,
            )
            if descriptor != expected:
                raise ValueError("sample descriptor must exactly match the shard sample")

    @property
    def key(self) -> tuple[float, str]:
        return (self.diagnostic.redshift, self.diagnostic.imf_mode)


def uvlf_shard_filename(
    config: UVLFRunConfig,
    redshift: float,
    imf_mode: str,
) -> str:
    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    normalized_redshift = _strict_float("redshift", redshift)
    if type(imf_mode) is not str:
        raise TypeError("imf_mode must be a string")
    mode = validate_imf_mode(imf_mode)
    configured_redshift = next(
        (value for value in config.redshifts if value == normalized_redshift),
        None,
    )
    if configured_redshift is None or mode not in config.stellar_population.imf_modes:
        raise ValueError("shard key must lie on configured axes")
    filename_redshift = 0.0 if configured_redshift == 0.0 else configured_redshift
    filename = (
        f"{config.run_id}.z={filename_redshift:.17g}.{mode}.shard.h5"
    )
    if Path(filename).name != filename:
        raise ValueError("shard filename must not contain path traversal")
    return filename


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "ArtifactProvenance",
    "HaloSampleDescriptor",
    "HaloSampleTable",
    "SourceChecksum",
    "UVLFArtifact",
    "UVLFShard",
    "UVLFShardDescriptor",
    "canonical_config_json",
    "canonical_config_mapping",
    "canonical_config_sha256",
    "decode_canonical_config_json",
    "decode_canonical_config_mapping",
    "uvlf_shard_filename",
]
