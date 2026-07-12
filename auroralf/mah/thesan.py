from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from auroralf.file_version import FileVersion

from .models import Cosmology, HaloHistoryResult
from .tng import MAH_BACKEND_THESAN


THESAN_MAH_CACHE_SCHEMA_VERSION = "auroralf_thesan_mah_cache_v1"
THESAN_TIME_GRID_SNAPSHOT = "snapshot"
THESAN_TIME_GRID_UNIFORM_IN_T = "uniform_in_t"
THESAN_TIME_GRID_MODES = (THESAN_TIME_GRID_SNAPSHOT, THESAN_TIME_GRID_UNIFORM_IN_T)
_THESAN_MAH_CACHE: dict[tuple[FileVersion, float], dict[str, Any]] = {}
_PROVENANCE_PLACEHOLDERS = frozenset({"unknown", "none", "null", "n/a"})
_SHA256_HEXDIGITS = frozenset("0123456789abcdefABCDEF")


def validate_thesan_time_grid_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in THESAN_TIME_GRID_MODES:
        choices = ", ".join(THESAN_TIME_GRID_MODES)
        raise ValueError(f"thesan_time_grid_mode must be one of: {choices}")
    return normalized


def _read_required_dataset(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"THESAN MAH cache is missing required dataset '{name}'")
    return np.asarray(handle[name])


def _read_required_bool_dataset(handle: h5py.File, name: str) -> np.ndarray:
    values = _read_required_dataset(handle, name)
    if values.dtype.kind != "b":
        raise ValueError(f"THESAN MAH cache {name} must use a native bool dtype")
    return values


def _read_required_text_attr(handle: h5py.File, name: str) -> str:
    if name not in handle.attrs:
        raise KeyError(f"THESAN MAH cache is missing required provenance attribute '{name}'")
    raw_value = handle.attrs[name]
    if isinstance(raw_value, bytes):
        try:
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"THESAN MAH cache provenance attribute '{name}' must be UTF-8 text"
            ) from exc
    elif isinstance(raw_value, str):
        value = raw_value
    else:
        raise ValueError(f"THESAN MAH cache provenance attribute '{name}' must be text")
    value = value.strip()
    if not value or value.lower() in _PROVENANCE_PLACEHOLDERS:
        raise ValueError(
            f"THESAN MAH cache provenance attribute '{name}' must contain a real value; got {value!r}"
        )
    return value


def _read_required_nonnegative_int_attr(handle: h5py.File, name: str) -> int:
    if name not in handle.attrs:
        raise KeyError(f"THESAN MAH cache is missing required provenance attribute '{name}'")
    raw_value = handle.attrs[name]
    if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, (int, np.integer)):
        raise ValueError(f"THESAN MAH cache provenance attribute '{name}' must be an integer")
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"THESAN MAH cache provenance attribute '{name}' must be non-negative")
    return value


def _read_required_identifier_dataset(handle: h5py.File, name: str) -> np.ndarray:
    values = _read_required_dataset(handle, name)
    if values.dtype.kind not in "iu":
        raise ValueError(f"THESAN MAH cache {name} must contain integer identifiers")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"THESAN MAH cache {name} values must fit in signed 64-bit integers")
    identifiers = np.asarray(values, dtype=np.int64)
    if np.any(identifiers < 0):
        raise ValueError(f"THESAN MAH cache {name} must contain real non-negative identifiers")
    return identifiers


def _read_required_text_dataset(handle: h5py.File, name: str) -> np.ndarray:
    raw_values = _read_required_dataset(handle, name)
    if raw_values.ndim != 1 or raw_values.size == 0:
        raise ValueError(f"THESAN MAH cache {name} must be a non-empty 1D text array")
    decoded: list[str] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bytes):
            try:
                value = raw_value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"THESAN MAH cache {name} must contain UTF-8 text") from exc
        elif isinstance(raw_value, str):
            value = raw_value
        else:
            raise ValueError(f"THESAN MAH cache {name} must contain text values")
        value = value.strip()
        if not value or value.lower() in _PROVENANCE_PLACEHOLDERS:
            raise ValueError(
                f"THESAN MAH cache {name} must contain real source values; got {value!r}"
            )
        decoded.append(value)
    max_length = max(len(value) for value in decoded)
    return np.asarray(decoded, dtype=f"U{max_length}")


def _validate_source_file_checksums(
    source_file_identifier: np.ndarray,
    source_file_sha256: np.ndarray,
) -> None:
    if source_file_sha256.shape != source_file_identifier.shape:
        raise ValueError(
            "THESAN MAH cache source_file_sha256 must match source_file_identifier shape"
        )
    identifier_to_checksum: dict[str, str] = {}
    for identifier, checksum in zip(
        source_file_identifier.tolist(),
        source_file_sha256.tolist(),
        strict=True,
    ):
        if len(checksum) != 64 or any(character not in _SHA256_HEXDIGITS for character in checksum):
            raise ValueError(
                "THESAN MAH cache source_file_sha256 must contain 64-character hexadecimal SHA-256 values"
            )
        previous = identifier_to_checksum.setdefault(identifier, checksum.lower())
        if previous != checksum.lower():
            raise ValueError(
                "THESAN MAH cache source_file_identifier cannot map to multiple SHA-256 values"
            )


def _read_required_cosmology_attr(handle: h5py.File, name: str) -> float:
    if name not in handle.attrs:
        raise KeyError(f"THESAN MAH cache is missing required cosmology attribute '{name}'")
    try:
        value = float(handle.attrs[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"THESAN MAH cache cosmology attribute '{name}' must be numeric"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"THESAN MAH cache cosmology attribute '{name}' must be finite")
    return value


def _validate_cache_cosmology(cache: dict[str, Any], cosmology: Cosmology) -> None:
    expected = {
        "hubble": cosmology.h0_km_s_mpc / 100.0,
        "omega_m": cosmology.omega_m,
        "omega_b": cosmology.omega_b,
    }
    for field, expected_value in expected.items():
        cache_value = float(cache[field])
        if not np.isclose(cache_value, float(expected_value), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"THESAN MAH cache cosmology {field}={cache_value:.16g} does not match "
                f"requested cosmology {field}={float(expected_value):.16g}"
            )


def _read_cache_z_final(handle: h5py.File, z_grid: np.ndarray) -> float:
    if "z_final" in handle.attrs:
        try:
            return float(handle.attrs["z_final"])
        except (TypeError, ValueError) as exc:
            raise ValueError("THESAN MAH cache attribute 'z_final' must be numeric") from exc
    return float(z_grid[-1])


def _version_from_open_file(version: FileVersion, file_descriptor: int) -> FileVersion:
    file_stat = os.fstat(file_descriptor)
    return FileVersion(
        path=version.path,
        st_dev=file_stat.st_dev,
        st_ino=file_stat.st_ino,
        st_size=file_stat.st_size,
        st_mtime_ns=file_stat.st_mtime_ns,
        st_ctime_ns=file_stat.st_ctime_ns,
    )


def _require_current_file_version(version: FileVersion) -> None:
    try:
        current = FileVersion.from_path(version.path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"THESAN MAH cache changed during preload: {version.path}") from exc
    if current != version:
        raise RuntimeError(f"THESAN MAH cache changed during preload: {version.path}")


def _resolve_cache_path(cache_path: str | Path, z_final: float) -> Path:
    path = Path(cache_path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"THESAN MAH cache path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"THESAN MAH cache path must be a file or directory: {path}")

    matches: list[Path] = []
    for candidate in sorted(path.glob("*.hdf5")):
        with h5py.File(candidate, "r") as handle:
            try:
                schema_version = _read_required_text_attr(handle, "schema_version")
            except (KeyError, ValueError):
                continue
            if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
                continue
            z_grid = _read_required_dataset(handle, "z_grid")
            cache_z_final = _read_cache_z_final(handle, np.asarray(z_grid, dtype=float))
        if np.isclose(cache_z_final, float(z_final), rtol=0.0, atol=1.0e-3):
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"no THESAN MAH cache file in {path} matches z_final={z_final:g}")
    if len(matches) > 1:
        names = ", ".join(str(match) for match in matches)
        raise RuntimeError(f"multiple THESAN MAH cache files match z_final={z_final:g}: {names}")
    return matches[0].resolve()


def _read_thesan_cache_file(
    version: FileVersion,
    z_final: float,
) -> tuple[Path, dict[str, Any]]:
    if type(version) is not FileVersion:
        raise TypeError("version must be exactly FileVersion")
    resolved = version.path
    with resolved.open("rb") as raw_handle:
        opened_version = _version_from_open_file(version, raw_handle.fileno())
        if opened_version != version:
            raise RuntimeError(f"THESAN MAH cache changed during preload: {resolved}")
        with h5py.File(raw_handle, "r") as handle:
            schema_version = _read_required_text_attr(handle, "schema_version")
            if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    "THESAN MAH cache schema_version must be "
                    f"{THESAN_MAH_CACHE_SCHEMA_VERSION!r}; got {schema_version!r}"
                )
            z_grid = np.asarray(_read_required_dataset(handle, "z_grid"), dtype=float)
            t_gyr_grid = np.asarray(_read_required_dataset(handle, "t_gyr_grid"), dtype=float)
            mass_ratio = np.asarray(_read_required_dataset(handle, "mass_ratio"), dtype=float)
            resolved_mask = _read_required_bool_dataset(handle, "resolved_mask")
            logm_final = np.asarray(_read_required_dataset(handle, "logM_final"), dtype=float)
            source_subhalo_id = _read_required_identifier_dataset(handle, "source_subhalo_id")
            source_group_index = _read_required_identifier_dataset(handle, "source_group_index")
            source_tree_file = _read_required_identifier_dataset(handle, "source_tree_file")
            source_tree_num = _read_required_identifier_dataset(handle, "source_tree_num")
            source_tree_index = _read_required_identifier_dataset(handle, "source_tree_index")
            source_snapshot = _read_required_identifier_dataset(handle, "source_snapshot")
            source_file_identifier = _read_required_text_dataset(handle, "source_file_identifier")
            source_file_sha256 = _read_required_text_dataset(handle, "source_file_sha256")
            _validate_source_file_checksums(source_file_identifier, source_file_sha256)
            source_simulation = _read_required_text_attr(handle, "source_simulation")
            source_tree = _read_required_text_attr(handle, "source_tree")
            snapshot = _read_required_nonnegative_int_attr(handle, "snapshot")
            mass_unit = _read_required_text_attr(handle, "mass_unit")
            time_unit = _read_required_text_attr(handle, "time_unit")
            redshift_unit = _read_required_text_attr(handle, "redshift_unit")
            mass_ratio_unit = _read_required_text_attr(handle, "mass_ratio_unit")
            selection_description = _read_required_text_attr(handle, "selection_description")
            creator_version = _read_required_text_attr(handle, "creator_version")
            cache_z_final = _read_cache_z_final(handle, z_grid)
            hubble = _read_required_cosmology_attr(handle, "hubble")
            omega_m = _read_required_cosmology_attr(handle, "omega_m")
            omega_b = _read_required_cosmology_attr(handle, "omega_b")
    _require_current_file_version(version)

    if mass_unit != "Msun":
        raise ValueError(f"THESAN MAH cache mass_unit must be 'Msun'; got {mass_unit!r}")
    if time_unit != "Gyr":
        raise ValueError(f"THESAN MAH cache time_unit must be 'Gyr'; got {time_unit!r}")
    if redshift_unit != "dimensionless":
        raise ValueError(
            "THESAN MAH cache redshift_unit must be 'dimensionless'; "
            f"got {redshift_unit!r}"
        )
    if mass_ratio_unit != "dimensionless":
        raise ValueError(
            "THESAN MAH cache mass_ratio_unit must be 'dimensionless'; "
            f"got {mass_ratio_unit!r}"
        )
    if hubble <= 0.0:
        raise ValueError("THESAN MAH cache cosmology attribute 'hubble' must be positive")
    if omega_m <= 0.0:
        raise ValueError("THESAN MAH cache cosmology attribute 'omega_m' must be positive")
    if omega_b <= 0.0 or omega_b > omega_m:
        raise ValueError("THESAN MAH cache cosmology attribute 'omega_b' must lie in (0, omega_m]")
    if z_grid.ndim != 1 or z_grid.size < 2:
        raise ValueError("THESAN MAH cache z_grid must be a 1D array with at least two entries")
    if t_gyr_grid.ndim != 1 or t_gyr_grid.shape != z_grid.shape:
        raise ValueError("THESAN MAH cache t_gyr_grid must be 1D and match z_grid")
    if mass_ratio.ndim != 2 or mass_ratio.shape[1] != z_grid.size:
        raise ValueError("THESAN MAH cache mass_ratio must have shape (n_halos, n_steps)")
    if resolved_mask.shape != mass_ratio.shape:
        raise ValueError("THESAN MAH cache resolved_mask must match mass_ratio shape")
    if logm_final.ndim != 1 or logm_final.size != mass_ratio.shape[0]:
        raise ValueError("THESAN MAH cache logM_final must be 1D and match mass_ratio rows")
    for name, values in (
        ("source_subhalo_id", source_subhalo_id),
        ("source_group_index", source_group_index),
        ("source_tree_file", source_tree_file),
        ("source_tree_num", source_tree_num),
        ("source_tree_index", source_tree_index),
        ("source_snapshot", source_snapshot),
    ):
        if values.ndim != 1 or values.size != mass_ratio.shape[0]:
            raise ValueError(f"THESAN MAH cache {name} must be 1D and match mass_ratio rows")
    if np.any(source_snapshot != snapshot):
        raise ValueError("THESAN MAH cache source_snapshot values must match snapshot attribute")
    source_identity = np.column_stack(
        (source_snapshot, source_tree_file, source_tree_num, source_tree_index)
    )
    if np.unique(source_identity, axis=0).shape[0] != source_identity.shape[0]:
        raise ValueError("THESAN MAH cache source halo identities must be unique")
    if np.any(np.diff(z_grid) >= 0.0):
        raise ValueError("THESAN MAH cache z_grid must be strictly decreasing")
    if np.any(np.diff(t_gyr_grid) <= 0.0):
        raise ValueError("THESAN MAH cache t_gyr_grid must be strictly increasing")
    if not np.isclose(cache_z_final, float(z_final), rtol=0.0, atol=1.0e-3):
        raise ValueError(
            f"THESAN MAH cache z_final={cache_z_final:g} does not match requested z_final={z_final:g}"
        )
    if not np.all(np.isfinite(mass_ratio)) or np.any(mass_ratio <= 0.0):
        raise ValueError("THESAN MAH cache mass_ratio values must be finite and positive")
    if not np.all(resolved_mask[:, -1]):
        raise ValueError("THESAN MAH cache mass_ratio final column must be resolved for every halo")
    if not np.all(np.isfinite(mass_ratio[:, -1])) or not np.allclose(
        mass_ratio[:, -1],
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("THESAN MAH cache mass_ratio final resolved column must equal 1")
    if not np.all(np.isfinite(logm_final)):
        raise ValueError("THESAN MAH cache logM_final values must be finite")

    return resolved, {
        "z_grid": z_grid,
        "t_gyr_grid": t_gyr_grid,
        "mass_ratio": mass_ratio,
        "resolved_mask": resolved_mask,
        "logM_final": logm_final,
        "source_subhalo_id": source_subhalo_id,
        "source_group_index": source_group_index,
        "source_tree_file": source_tree_file,
        "source_tree_num": source_tree_num,
        "source_tree_index": source_tree_index,
        "source_snapshot": source_snapshot,
        "source_file_identifier": source_file_identifier,
        "source_file_sha256": source_file_sha256,
        "source_simulation": source_simulation,
        "source_tree": source_tree,
        "snapshot": snapshot,
        "mass_unit": mass_unit,
        "time_unit": time_unit,
        "redshift_unit": redshift_unit,
        "mass_ratio_unit": mass_ratio_unit,
        "selection_description": selection_description,
        "creator_version": creator_version,
        "hubble": hubble,
        "omega_m": omega_m,
        "omega_b": omega_b,
    }


def _immutable_cache_array(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    immutable.flags.writeable = False
    return immutable


def _freeze_thesan_cache(cache: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _immutable_cache_array(value) if isinstance(value, np.ndarray) else value
        for key, value in cache.items()
    }


def _clear_thesan_mah_cache_for_tests() -> None:
    _THESAN_MAH_CACHE.clear()


def preload_thesan_mah_cache(
    cache_path: str | Path,
    *,
    z_final: float,
    cosmology: Cosmology,
) -> Path:
    return _preload_thesan_mah_cache_version(
        cache_path,
        z_final=z_final,
        cosmology=cosmology,
    ).path


def _preload_thesan_mah_cache_version(
    cache_path: str | Path,
    *,
    z_final: float,
    cosmology: Cosmology,
) -> FileVersion:
    if type(cosmology) is not Cosmology:
        raise TypeError("cosmology must be exactly Cosmology")
    resolved = _resolve_cache_path(cache_path, z_final=float(z_final))
    version = FileVersion.from_path(resolved)
    key = (version, float(z_final))
    cached = _THESAN_MAH_CACHE.get(key)
    if cached is not None:
        _require_current_file_version(version)
        _validate_cache_cosmology(cached, cosmology)
        return version
    loaded_path, loaded = _read_thesan_cache_file(
        version,
        z_final=float(z_final),
    )
    if loaded_path != version.path:
        raise RuntimeError("resolved THESAN cache path changed during preload")
    _validate_cache_cosmology(loaded, cosmology)
    frozen = _freeze_thesan_cache(loaded)
    _THESAN_MAH_CACHE[key] = frozen
    return version


def _load_thesan_cache(
    cache_path: str | Path,
    z_final: float,
    *,
    cosmology: Cosmology,
) -> tuple[Path, dict[str, Any]]:
    version = _preload_thesan_mah_cache_version(
        cache_path,
        z_final=float(z_final),
        cosmology=cosmology,
    )
    return version.path, _THESAN_MAH_CACHE[(version, float(z_final))]


def _slice_grid(
    z_grid: np.ndarray,
    t_gyr_grid: np.ndarray,
    mass_ratio: np.ndarray,
    resolved_mask: np.ndarray,
    *,
    z_final: float,
    z_start_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if z_start_max is None:
        mask = z_grid >= float(z_final) - 1.0e-3
    else:
        if float(z_start_max) <= float(z_final):
            raise ValueError("z_start_max must be greater than z_final")
        mask = (z_grid <= float(z_start_max) + 1.0e-3) & (z_grid >= float(z_final) - 1.0e-3)
    if np.count_nonzero(mask) < 2:
        raise ValueError("THESAN MAH cache does not contain at least two snapshots in the requested redshift range")
    sliced_z = z_grid[mask]
    sliced_t = t_gyr_grid[mask]
    sliced_ratio = mass_ratio[:, mask]
    sliced_resolved = resolved_mask[:, mask]
    if not np.isclose(sliced_z[-1], float(z_final), rtol=0.0, atol=1.0e-3):
        raise ValueError("THESAN MAH cache redshift slice must end at requested z_final")
    if not np.all(sliced_resolved[:, -1]):
        raise ValueError("THESAN MAH cache final snapshot must be resolved for every selected halo")
    return sliced_z, sliced_t, sliced_ratio, sliced_resolved


def _smooth_mass_ratio(mass_ratio: np.ndarray, t_gyr_grid: np.ndarray, smoothing_myr: float) -> np.ndarray:
    if float(smoothing_myr) < 0.0:
        raise ValueError("thesan_smoothing_myr must be non-negative")
    if float(smoothing_myr) == 0.0:
        return mass_ratio.copy()

    sigma_gyr = float(smoothing_myr) / 1.0e3
    smoothed = np.empty_like(mass_ratio, dtype=float)
    for step, t_value in enumerate(t_gyr_grid):
        dt = np.asarray(t_gyr_grid, dtype=float) - float(t_value)
        weights = np.exp(-0.5 * np.square(dt / sigma_gyr))
        weights /= np.sum(weights)
        smoothed[:, step] = mass_ratio @ weights

    final_ratio = smoothed[:, -1]
    if np.any(final_ratio <= 0.0) or not np.all(np.isfinite(final_ratio)):
        raise RuntimeError("smoothed THESAN MAH ratios have invalid final values")
    smoothed = smoothed / final_ratio[:, None]
    smoothed[:, -1] = 1.0
    return smoothed


def _regrid_mass_ratio_uniform_in_t(
    *,
    z_grid: np.ndarray,
    t_gyr_grid: np.ndarray,
    mass_ratio: np.ndarray,
    resolved_mask: np.ndarray,
    target_n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(target_n_grid) < 2:
        raise ValueError("target_n_grid must be at least 2 for THESAN uniform_in_t regridding")
    if mass_ratio.shape != resolved_mask.shape:
        raise ValueError("mass_ratio and resolved_mask must have the same shape for THESAN regridding")
    if mass_ratio.shape[1] != t_gyr_grid.size:
        raise ValueError("THESAN regridding source arrays have inconsistent time dimensions")

    target_t = np.linspace(float(t_gyr_grid[0]), float(t_gyr_grid[-1]), int(target_n_grid))
    target_z = np.interp(target_t, np.asarray(t_gyr_grid, dtype=float), np.asarray(z_grid, dtype=float))
    target_z[0] = float(z_grid[0])
    target_z[-1] = float(z_grid[-1])

    regridded_ratio = np.empty((mass_ratio.shape[0], target_t.size), dtype=float)
    regridded_resolved = np.zeros_like(regridded_ratio, dtype=bool)
    for row_index in range(mass_ratio.shape[0]):
        source_indices = np.flatnonzero(resolved_mask[row_index])
        if source_indices.size < 2:
            raise ValueError(
                "THESAN uniform_in_t regridding requires at least two resolved snapshots "
                f"for every selected track; row {row_index} has {source_indices.size}"
            )
        source_t = np.asarray(t_gyr_grid[source_indices], dtype=float)
        source_ratio = np.asarray(mass_ratio[row_index, source_indices], dtype=float)
        if np.any(source_ratio <= 0.0) or not np.all(np.isfinite(source_ratio)):
            raise ValueError("resolved THESAN mass ratios must be finite and positive for log interpolation")

        first_resolved_t = float(source_t[0])
        resolved_target = target_t >= first_resolved_t - 1.0e-12
        regridded_resolved[row_index, resolved_target] = True
        regridded_ratio[row_index, ~resolved_target] = float(source_ratio[0])
        regridded_ratio[row_index, resolved_target] = np.exp(
            np.interp(target_t[resolved_target], source_t, np.log(source_ratio))
        )

    regridded_ratio /= regridded_ratio[:, -1][:, None]
    regridded_ratio[:, -1] = 1.0
    regridded_resolved[:, -1] = True
    return target_z, target_t, regridded_ratio, regridded_resolved


def _flatten_tracks(
    *,
    redshift: np.ndarray,
    time_gyr: np.ndarray,
    mass: np.ndarray,
    dmhdt_raw: np.ndarray,
    dmhdt_sfr: np.ndarray,
    dmhdt_clipped: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    n_halos, n_steps = mass.shape
    dt_gyr = np.diff(time_gyr, prepend=time_gyr[0])
    halo_id = np.repeat(np.arange(n_halos, dtype=int), n_steps)
    step = np.tile(np.arange(n_steps, dtype=int), n_halos)
    active_flag = active_mask.reshape(-1).astype(bool)
    termination = np.full(n_halos * n_steps, "active", dtype="<U10")
    termination[~active_flag] = "unresolved"
    termination[np.arange(n_steps - 1, n_halos * n_steps, n_steps)] = "completed"

    return {
        "halo_id": halo_id,
        "step": step,
        "z": np.tile(redshift, n_halos),
        "t_gyr": np.tile(time_gyr, n_halos),
        "dt_gyr": np.tile(dt_gyr, n_halos),
        "Mh": mass.reshape(-1),
        "dMh_dt_raw": dmhdt_raw.reshape(-1),
        "dMh_dt_sfr": dmhdt_sfr.reshape(-1),
        "dMh_dt_clipped": dmhdt_clipped.reshape(-1),
        "active_flag": active_flag,
        "termination_flag": termination,
    }


def _compute_dmhdt(
    mass: np.ndarray,
    time_gyr: np.ndarray,
    resolved_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    dt_gyr = np.diff(time_gyr)
    if np.any(dt_gyr <= 0.0):
        raise ValueError("THESAN MAH time grid must be strictly increasing")
    if resolved_mask.shape != mass.shape:
        raise ValueError("resolved_mask must match mass shape")
    raw = np.diff(mass, axis=1) / dt_gyr[None, :]
    resolved_transition = resolved_mask[:, 1:] & resolved_mask[:, :-1]
    negative = (raw < 0.0) & resolved_transition
    negative_count = int(np.count_nonzero(negative))
    total_count = int(np.count_nonzero(resolved_transition))
    raw = raw.copy()
    raw[~resolved_transition] = 0.0
    dmhdt_raw = np.zeros_like(mass, dtype=float)
    dmhdt_raw[:, 1:] = raw
    dmhdt_sfr = np.maximum(dmhdt_raw, 0.0)
    dmhdt_clipped = dmhdt_raw < 0.0
    return dmhdt_raw, dmhdt_sfr, dmhdt_clipped, negative_count, total_count


def generate_thesan_halo_histories(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    cosmology: Cosmology,
    cache_path: str | Path,
    z_start_max: float | None = None,
    mass_bin_width_dex: float = 0.15,
    min_candidates: int = 5,
    smoothing_myr: float = 0.0,
    random_seed: int | None = None,
    time_grid_mode: str = THESAN_TIME_GRID_SNAPSHOT,
    target_n_grid: int | None = None,
) -> HaloHistoryResult:
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    if int(n_tracks) <= 0:
        raise ValueError("n_tracks must be positive")
    if float(Mh_final) <= 0.0:
        raise ValueError("Mh_final must be positive")
    if float(mass_bin_width_dex) <= 0.0:
        raise ValueError("thesan_mass_bin_width_dex must be positive")
    if int(min_candidates) <= 0:
        raise ValueError("thesan_min_candidates must be positive")
    time_grid_mode = validate_thesan_time_grid_mode(time_grid_mode)
    if time_grid_mode == THESAN_TIME_GRID_UNIFORM_IN_T and target_n_grid is None:
        raise ValueError("target_n_grid is required when thesan_time_grid_mode='uniform_in_t'")

    resolved_cache_path, cache = _load_thesan_cache(
        cache_path,
        z_final=float(z_final),
        cosmology=cosmology,
    )
    z_grid, t_grid, mass_ratio, resolved_mask = _slice_grid(
        np.asarray(cache["z_grid"], dtype=float),
        np.asarray(cache["t_gyr_grid"], dtype=float),
        np.asarray(cache["mass_ratio"], dtype=float),
        np.asarray(cache["resolved_mask"], dtype=bool),
        z_final=float(z_final),
        z_start_max=z_start_max,
    )
    mass_ratio = _smooth_mass_ratio(mass_ratio, t_grid, smoothing_myr=float(smoothing_myr))
    target_logm = float(np.log10(float(Mh_final)))
    logm_final = np.asarray(cache["logM_final"], dtype=float)
    candidate_mask = np.abs(logm_final - target_logm) <= float(mass_bin_width_dex)
    raw_candidate_count = int(np.count_nonzero(candidate_mask))
    if time_grid_mode == THESAN_TIME_GRID_UNIFORM_IN_T:
        candidate_mask &= np.count_nonzero(resolved_mask, axis=1) >= 2
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_count = int(candidate_indices.size)
    if candidate_count < int(min_candidates):
        raise ValueError(
            "THESAN MAH candidate count "
            f"{candidate_count} is below thesan_min_candidates={int(min_candidates)} "
            f"for log10(Mh_final)={target_logm:.3f} within {float(mass_bin_width_dex):.3f} dex"
        )

    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(candidate_indices, size=int(n_tracks), replace=True)
    selected_ratio = mass_ratio[selected_indices]
    selected_resolved_mask = resolved_mask[selected_indices]
    if time_grid_mode == THESAN_TIME_GRID_UNIFORM_IN_T:
        z_grid, t_grid, selected_ratio, selected_resolved_mask = _regrid_mass_ratio_uniform_in_t(
            z_grid=z_grid,
            t_gyr_grid=t_grid,
            mass_ratio=selected_ratio,
            resolved_mask=selected_resolved_mask,
            target_n_grid=int(target_n_grid),
        )
    mass = selected_ratio * float(Mh_final)
    dmhdt_raw, dmhdt_sfr, dmhdt_clipped, negative_count, total_dmhdt_count = _compute_dmhdt(
        mass, t_grid, selected_resolved_mask
    )
    tracks = _flatten_tracks(
        redshift=z_grid,
        time_gyr=t_grid,
        mass=mass,
        dmhdt_raw=dmhdt_raw,
        dmhdt_sfr=dmhdt_sfr,
        dmhdt_clipped=dmhdt_clipped,
        active_mask=selected_resolved_mask,
    )
    negative_fraction = float(negative_count / total_dmhdt_count) if total_dmhdt_count > 0 else 0.0
    unresolved_step_count = int(np.count_nonzero(~selected_resolved_mask))
    unresolved_step_total = int(selected_resolved_mask.size)
    unresolved_step_fraction = (
        float(unresolved_step_count / unresolved_step_total) if unresolved_step_total > 0 else 0.0
    )

    metadata: dict[str, Any] = {
        "mah_backend": MAH_BACKEND_THESAN,
        "cache_path": str(resolved_cache_path),
        "thesan_mah_cache_path": str(resolved_cache_path),
        "source_simulation": str(cache["source_simulation"]),
        "source_tree": str(cache["source_tree"]),
        "snapshot": int(cache["snapshot"]),
        "mass_unit": str(cache["mass_unit"]),
        "time_unit": str(cache["time_unit"]),
        "redshift_unit": str(cache["redshift_unit"]),
        "mass_ratio_unit": str(cache["mass_ratio_unit"]),
        "selection_description": str(cache["selection_description"]),
        "creator_version": str(cache["creator_version"]),
        "source_file_identifier": np.asarray(cache["source_file_identifier"]).copy(),
        "source_file_sha256": np.asarray(cache["source_file_sha256"]).copy(),
        "schema_version": THESAN_MAH_CACHE_SCHEMA_VERSION,
        "cache_hubble": float(cache["hubble"]),
        "cache_omega_m": float(cache["omega_m"]),
        "cache_omega_b": float(cache["omega_b"]),
        "n_tracks": int(n_tracks),
        "z_final": float(z_final),
        "Mh_final": float(Mh_final),
        "z_start_max": None if z_start_max is None else float(z_start_max),
        "time_grid_mode": "thesan_snapshot_grid"
        if time_grid_mode == THESAN_TIME_GRID_SNAPSHOT
        else "thesan_uniform_in_t",
        "grid_size": int(z_grid.size),
        "random_seed": random_seed,
        "target_logM_final": target_logm,
        "thesan_mass_bin_width_dex": float(mass_bin_width_dex),
        "thesan_min_candidates": int(min_candidates),
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": candidate_count,
        "selected_cache_indices": selected_indices.astype(np.int64),
        "selected_source_subhalo_id": np.asarray(cache["source_subhalo_id"], dtype=np.int64)[selected_indices],
        "selected_source_group_index": np.asarray(cache["source_group_index"], dtype=np.int64)[selected_indices],
        "selected_source_tree_file": np.asarray(cache["source_tree_file"], dtype=np.int64)[selected_indices],
        "selected_source_tree_num": np.asarray(cache["source_tree_num"], dtype=np.int64)[selected_indices],
        "selected_source_tree_index": np.asarray(cache["source_tree_index"], dtype=np.int64)[selected_indices],
        "thesan_smoothing_myr": float(smoothing_myr),
        "thesan_time_grid_mode": time_grid_mode,
        "thesan_target_n_grid": None if target_n_grid is None else int(target_n_grid),
        "negative_dmhdt_clip_count": negative_count,
        "negative_dmhdt_total_count": total_dmhdt_count,
        "negative_dmhdt_clip_fraction": negative_fraction,
        "unresolved_step_count": unresolved_step_count,
        "unresolved_step_total_count": unresolved_step_total,
        "unresolved_step_fraction": unresolved_step_fraction,
        "dt_gyr_median": float(np.median(np.diff(t_grid))),
    }
    return HaloHistoryResult(tracks=tracks, metadata=metadata)


__all__ = [
    "THESAN_MAH_CACHE_SCHEMA_VERSION",
    "THESAN_TIME_GRID_MODES",
    "THESAN_TIME_GRID_SNAPSHOT",
    "THESAN_TIME_GRID_UNIFORM_IN_T",
    "generate_thesan_halo_histories",
    "preload_thesan_mah_cache",
    "validate_thesan_time_grid_mode",
]
