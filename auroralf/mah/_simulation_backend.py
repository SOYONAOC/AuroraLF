"""Strict shared primitives for cache-backed simulation MAH backends."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np

from auroralf._array_utils import immutable_array
from auroralf.file_version import FileVersion

from .models import Cosmology


_PROVENANCE_PLACEHOLDERS = frozenset({"unknown", "none", "null", "n/a"})
_SHA256_HEXDIGITS = frozenset("0123456789abcdefABCDEF")
_Z_MATCH_ATOL = 1.0e-3


def normalize_choice(value: str, *, choices: tuple[str, ...], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{field_name} must be one of: {', '.join(choices)}")
    return normalized


def read_required_dataset(
    handle: h5py.File, name: str, *, cache_label: str
) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"{cache_label} MAH cache is missing required dataset '{name}'")
    return np.asarray(handle[name])


class CacheReader:
    """Backend-labelled strict HDF5 readers using the caller's dataset seam."""

    __slots__ = ("handle", "cache_label", "dataset_reader")

    def __init__(
        self,
        handle: h5py.File,
        cache_label: str,
        dataset_reader: Callable[[h5py.File, str], np.ndarray],
    ) -> None:
        self.handle = handle
        self.cache_label = cache_label
        self.dataset_reader = dataset_reader

    def _decode_text(
        self,
        raw_value: object,
        *,
        utf8_message: str,
        type_message: str,
        placeholder_message: str,
    ) -> str:
        if isinstance(raw_value, bytes):
            try:
                value = raw_value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{self.cache_label} MAH cache {utf8_message}") from exc
        elif isinstance(raw_value, str):
            value = raw_value
        else:
            raise ValueError(f"{self.cache_label} MAH cache {type_message}")
        value = value.strip()
        if not value or value.lower() in _PROVENANCE_PLACEHOLDERS:
            raise ValueError(
                f"{self.cache_label} MAH cache {placeholder_message}; got {value!r}"
            )
        return value

    def bool_dataset(self, name: str) -> np.ndarray:
        values = self.dataset_reader(self.handle, name)
        if values.dtype.kind != "b":
            raise ValueError(
                f"{self.cache_label} MAH cache {name} must use a native bool dtype"
            )
        return values

    def text_attr(self, name: str) -> str:
        if name not in self.handle.attrs:
            raise KeyError(
                f"{self.cache_label} MAH cache is missing required provenance attribute '{name}'"
            )
        subject = f"provenance attribute '{name}'"
        return self._decode_text(
            self.handle.attrs[name],
            utf8_message=f"{subject} must be UTF-8 text",
            type_message=f"{subject} must be text",
            placeholder_message=f"{subject} must contain a real value",
        )

    def nonnegative_int_attr(self, name: str) -> int:
        if name not in self.handle.attrs:
            raise KeyError(
                f"{self.cache_label} MAH cache is missing required provenance attribute '{name}'"
            )
        raw_value = self.handle.attrs[name]
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, np.integer)
        ):
            raise ValueError(
                f"{self.cache_label} MAH cache provenance attribute '{name}' must be an integer"
            )
        value = int(raw_value)
        if value < 0:
            raise ValueError(
                f"{self.cache_label} MAH cache provenance attribute '{name}' must be non-negative"
            )
        return value

    def identifier_dataset(self, name: str) -> np.ndarray:
        values = self.dataset_reader(self.handle, name)
        if values.dtype.kind not in "iu":
            raise ValueError(
                f"{self.cache_label} MAH cache {name} must contain integer identifiers"
            )
        if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
            raise ValueError(
                f"{self.cache_label} MAH cache {name} values must fit in signed 64-bit integers"
            )
        identifiers = np.asarray(values, dtype=np.int64)
        if np.any(identifiers < 0):
            raise ValueError(
                f"{self.cache_label} MAH cache {name} must contain real non-negative identifiers"
            )
        return identifiers

    def text_dataset(self, name: str) -> np.ndarray:
        raw_values = self.dataset_reader(self.handle, name)
        if raw_values.ndim != 1 or raw_values.size == 0:
            raise ValueError(
                f"{self.cache_label} MAH cache {name} must be a non-empty 1D text array"
            )
        decoded: list[str] = []
        for raw_value in raw_values:
            decoded.append(
                self._decode_text(
                    raw_value,
                    utf8_message=f"{name} must contain UTF-8 text",
                    type_message=f"{name} must contain text values",
                    placeholder_message=f"{name} must contain real source values",
                )
            )
        max_length = max(len(value) for value in decoded)
        return np.asarray(decoded, dtype=f"U{max_length}")

    def validate_source_file_checksums(
        self,
        source_file_identifier: np.ndarray,
        source_file_sha256: np.ndarray,
    ) -> None:
        if source_file_sha256.shape != source_file_identifier.shape:
            raise ValueError(
                f"{self.cache_label} MAH cache source_file_sha256 must match "
                "source_file_identifier shape"
            )
        identifier_to_checksum: dict[str, str] = {}
        for identifier, checksum in zip(
            source_file_identifier.tolist(),
            source_file_sha256.tolist(),
            strict=True,
        ):
            if len(checksum) != 64 or any(
                character not in _SHA256_HEXDIGITS for character in checksum
            ):
                raise ValueError(
                    f"{self.cache_label} MAH cache source_file_sha256 must contain "
                    "64-character hexadecimal SHA-256 values"
                )
            normalized = checksum.lower()
            previous = identifier_to_checksum.setdefault(identifier, normalized)
            if previous != normalized:
                raise ValueError(
                    f"{self.cache_label} MAH cache source_file_identifier cannot map "
                    "to multiple SHA-256 values"
                )

    def cosmology_attr(self, name: str) -> float:
        if name not in self.handle.attrs:
            raise KeyError(
                f"{self.cache_label} MAH cache is missing required cosmology attribute '{name}'"
            )
        try:
            value = float(self.handle.attrs[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.cache_label} MAH cache cosmology attribute '{name}' must be numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"{self.cache_label} MAH cache cosmology attribute '{name}' must be finite"
            )
        return value

    def z_final(self, z_grid: np.ndarray) -> float:
        if "z_final" in self.handle.attrs:
            try:
                return float(self.handle.attrs["z_final"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.cache_label} MAH cache attribute 'z_final' must be numeric"
                ) from exc
        return float(z_grid[-1])


def validate_cache_cosmology(
    cache: dict[str, Any], cosmology: Cosmology, *, cache_label: str
) -> None:
    expected = {
        "hubble": cosmology.h0_km_s_mpc / 100.0,
        "omega_m": cosmology.omega_m,
        "omega_b": cosmology.omega_b,
    }
    for field, expected_value in expected.items():
        cache_value = float(cache[field])
        if not np.isclose(cache_value, float(expected_value), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{cache_label} MAH cache cosmology {field}={cache_value:.16g} "
                f"does not match requested cosmology {field}={float(expected_value):.16g}"
            )


def version_from_open_file(version: FileVersion, file_descriptor: int) -> FileVersion:
    file_stat = os.fstat(file_descriptor)
    return FileVersion(
        path=version.path,
        st_dev=file_stat.st_dev,
        st_ino=file_stat.st_ino,
        st_size=file_stat.st_size,
        st_mtime_ns=file_stat.st_mtime_ns,
        st_ctime_ns=file_stat.st_ctime_ns,
    )


def require_current_file_version(version: FileVersion, *, cache_label: str) -> None:
    try:
        current = FileVersion.from_path(version.path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"{cache_label} MAH cache changed during preload: {version.path}"
        ) from exc
    if current != version:
        raise RuntimeError(f"{cache_label} MAH cache changed during preload: {version.path}")


def freeze_cache(cache: dict[str, Any]) -> dict[str, Any]:
    return {
        key: immutable_array(value) if isinstance(value, np.ndarray) else value
        for key, value in cache.items()
    }


def validate_cache_contents(
    *,
    cache_label: str,
    z_grid: np.ndarray,
    t_gyr_grid: np.ndarray,
    mass_ratio: np.ndarray,
    resolved_mask: np.ndarray,
    logm_final: np.ndarray,
    source_snapshot: np.ndarray,
    identifier_fields: tuple[tuple[str, np.ndarray], ...],
    source_identity_columns: tuple[np.ndarray, ...],
    snapshot: int,
    mass_unit: str,
    time_unit: str,
    redshift_unit: str,
    mass_ratio_unit: str,
    hubble: float,
    omega_m: float,
    omega_b: float,
    cache_z_final: float,
    requested_z_final: float,
) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"{cache_label} MAH cache {message}")

    for name, value, expected in (
        ("mass_unit", mass_unit, "Msun"),
        ("time_unit", time_unit, "Gyr"),
        ("redshift_unit", redshift_unit, "dimensionless"),
        ("mass_ratio_unit", mass_ratio_unit, "dimensionless"),
    ):
        if value != expected:
            fail(f"{name} must be {expected!r}; got {value!r}")
    if hubble <= 0.0:
        fail("cosmology attribute 'hubble' must be positive")
    if omega_m <= 0.0:
        fail("cosmology attribute 'omega_m' must be positive")
    if omega_b <= 0.0 or omega_b > omega_m:
        fail("cosmology attribute 'omega_b' must lie in (0, omega_m]")
    if z_grid.ndim != 1 or z_grid.size < 2:
        fail("z_grid must be a 1D array with at least two entries")
    if t_gyr_grid.ndim != 1 or t_gyr_grid.shape != z_grid.shape:
        fail("t_gyr_grid must be 1D and match z_grid")
    if mass_ratio.ndim != 2 or mass_ratio.shape[1] != z_grid.size:
        fail("mass_ratio must have shape (n_halos, n_steps)")
    if resolved_mask.shape != mass_ratio.shape:
        fail("resolved_mask must match mass_ratio shape")
    if logm_final.ndim != 1 or logm_final.size != mass_ratio.shape[0]:
        fail("logM_final must be 1D and match mass_ratio rows")
    for name, values in identifier_fields:
        if values.ndim != 1 or values.size != mass_ratio.shape[0]:
            fail(f"{name} must be 1D and match mass_ratio rows")
    if np.any(source_snapshot != snapshot):
        fail("source_snapshot values must match snapshot attribute")
    source_identity = np.column_stack(source_identity_columns)
    if np.unique(source_identity, axis=0).shape[0] != source_identity.shape[0]:
        fail("source halo identities must be unique")
    if np.any(np.diff(z_grid) >= 0.0):
        fail("z_grid must be strictly decreasing")
    if np.any(np.diff(t_gyr_grid) <= 0.0):
        fail("t_gyr_grid must be strictly increasing")
    if not np.isclose(cache_z_final, float(requested_z_final), rtol=0.0, atol=_Z_MATCH_ATOL):
        fail(
            f"z_final={cache_z_final:g} does not match requested "
            f"z_final={requested_z_final:g}"
        )
    if not np.all(np.isfinite(mass_ratio)) or np.any(mass_ratio <= 0.0):
        fail("mass_ratio values must be finite and positive")
    if not np.all(resolved_mask[:, -1]):
        fail("mass_ratio final column must be resolved for every halo")
    if not np.all(np.isfinite(mass_ratio[:, -1])) or not np.allclose(
        mass_ratio[:, -1],
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        fail("mass_ratio final resolved column must equal 1")
    if not np.all(np.isfinite(logm_final)):
        fail("logM_final values must be finite")


def validate_generation_parameters(
    *,
    field_prefix: str,
    time_grid_modes: tuple[str, ...], uniform_time_grid_mode: str,
    cosmology: Cosmology, n_tracks: int,
    halo_mass_final: float,
    mass_bin_width_dex: float,
    min_candidates: int,
    time_grid_mode: str,
    target_n_grid: int | None,
) -> str:
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    if int(n_tracks) <= 0:
        raise ValueError("n_tracks must be positive")
    if float(halo_mass_final) <= 0.0:
        raise ValueError("Mh_final must be positive")
    if float(mass_bin_width_dex) <= 0.0:
        raise ValueError(f"{field_prefix}_mass_bin_width_dex must be positive")
    if int(min_candidates) <= 0:
        raise ValueError(f"{field_prefix}_min_candidates must be positive")
    normalized_mode = normalize_choice(
        time_grid_mode,
        choices=time_grid_modes,
        field_name=f"{field_prefix}_time_grid_mode",
    )
    if normalized_mode == uniform_time_grid_mode and target_n_grid is None:
        raise ValueError(
            f"target_n_grid is required when "
            f"{field_prefix}_time_grid_mode='{uniform_time_grid_mode}'"
        )
    return normalized_mode


def slice_grid(
    z_grid: np.ndarray, t_gyr_grid: np.ndarray,
    mass_ratio: np.ndarray, resolved_mask: np.ndarray,
    *,
    cache_label: str,
    z_final: float,
    z_start_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if z_start_max is None:
        mask = z_grid >= float(z_final) - _Z_MATCH_ATOL
    else:
        if float(z_start_max) <= float(z_final):
            raise ValueError("z_start_max must be greater than z_final")
        mask = (z_grid <= float(z_start_max) + _Z_MATCH_ATOL) & (
            z_grid >= float(z_final) - _Z_MATCH_ATOL
        )
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"{cache_label} MAH cache does not contain at least two snapshots "
            "in the requested redshift range"
        )
    sliced_z = z_grid[mask]
    sliced_t = t_gyr_grid[mask]
    sliced_ratio = mass_ratio[:, mask]
    sliced_resolved = resolved_mask[:, mask]
    if not np.isclose(sliced_z[-1], float(z_final), rtol=0.0, atol=_Z_MATCH_ATOL):
        raise ValueError(
            f"{cache_label} MAH cache redshift slice must end at requested z_final"
        )
    if not np.all(sliced_resolved[:, -1]):
        raise ValueError(
            f"{cache_label} MAH cache final snapshot must be resolved for every selected halo"
        )
    return sliced_z, sliced_t, sliced_ratio, sliced_resolved


def smooth_mass_ratio(
    mass_ratio: np.ndarray, t_gyr_grid: np.ndarray, smoothing_myr: float,
    *,
    cache_label: str,
    field_prefix: str,
) -> np.ndarray:
    if float(smoothing_myr) < 0.0:
        raise ValueError(f"{field_prefix}_smoothing_myr must be non-negative")
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
        raise RuntimeError(f"smoothed {cache_label} MAH ratios have invalid final values")
    smoothed = smoothed / final_ratio[:, None]
    smoothed[:, -1] = 1.0
    return smoothed


def regrid_mass_ratio_uniform_in_t(
    *,
    cache_label: str,
    z_grid: np.ndarray,
    t_gyr_grid: np.ndarray,
    mass_ratio: np.ndarray,
    resolved_mask: np.ndarray,
    target_n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(target_n_grid) < 2:
        raise ValueError(
            f"target_n_grid must be at least 2 for {cache_label} uniform_in_t regridding"
        )
    if mass_ratio.shape != resolved_mask.shape:
        raise ValueError(
            "mass_ratio and resolved_mask must have the same shape for "
            f"{cache_label} regridding"
        )
    if mass_ratio.shape[1] != t_gyr_grid.size:
        raise ValueError(
            f"{cache_label} regridding source arrays have inconsistent time dimensions"
        )

    target_t = np.linspace(float(t_gyr_grid[0]), float(t_gyr_grid[-1]), int(target_n_grid))
    target_z = np.interp(
        target_t,
        np.asarray(t_gyr_grid, dtype=float),
        np.asarray(z_grid, dtype=float),
    )
    target_z[0] = float(z_grid[0])
    target_z[-1] = float(z_grid[-1])

    regridded_ratio = np.empty((mass_ratio.shape[0], target_t.size), dtype=float)
    regridded_resolved = np.zeros_like(regridded_ratio, dtype=bool)
    for row_index in range(mass_ratio.shape[0]):
        source_indices = np.flatnonzero(resolved_mask[row_index])
        if source_indices.size < 2:
            raise ValueError(
                f"{cache_label} uniform_in_t regridding requires at least two "
                "resolved snapshots for every selected track; "
                f"row {row_index} has {source_indices.size}"
            )
        source_t = np.asarray(t_gyr_grid[source_indices], dtype=float)
        source_ratio = np.asarray(mass_ratio[row_index, source_indices], dtype=float)
        if np.any(source_ratio <= 0.0) or not np.all(np.isfinite(source_ratio)):
            raise ValueError(
                f"resolved {cache_label} mass ratios must be finite and positive "
                "for log interpolation"
            )

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


def flatten_tracks(
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


def compute_dmhdt(
    mass: np.ndarray, time_gyr: np.ndarray, resolved_mask: np.ndarray,
    *,
    cache_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    dt_gyr = np.diff(time_gyr)
    if np.any(dt_gyr <= 0.0):
        raise ValueError(f"{cache_label} MAH time grid must be strictly increasing")
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
