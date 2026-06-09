from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .models import HaloHistoryResult
from .tng import MAH_BACKEND_THESAN


THESAN_MAH_CACHE_SCHEMA_VERSION = "auroralf_thesan_mah_cache_v0"
THESAN_TIME_GRID_SNAPSHOT = "snapshot"
THESAN_TIME_GRID_UNIFORM_IN_T = "uniform_in_t"
THESAN_TIME_GRID_MODES = (THESAN_TIME_GRID_SNAPSHOT, THESAN_TIME_GRID_UNIFORM_IN_T)


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


def _read_cache_z_final(handle: h5py.File, z_grid: np.ndarray) -> float:
    if "z_final" in handle.attrs:
        return float(handle.attrs["z_final"])
    return float(z_grid[-1])


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
            schema_version = str(handle.attrs.get("schema_version", ""))
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


def _load_thesan_cache(cache_path: str | Path, z_final: float) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_cache_path(cache_path, z_final=z_final)
    with h5py.File(resolved, "r") as handle:
        schema_version = str(handle.attrs.get("schema_version", ""))
        if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "THESAN MAH cache schema_version must be "
                f"{THESAN_MAH_CACHE_SCHEMA_VERSION!r}; got {schema_version!r}"
            )
        z_grid = np.asarray(_read_required_dataset(handle, "z_grid"), dtype=float)
        t_gyr_grid = np.asarray(_read_required_dataset(handle, "t_gyr_grid"), dtype=float)
        mass_ratio = np.asarray(_read_required_dataset(handle, "mass_ratio"), dtype=float)
        resolved_mask = np.asarray(_read_required_dataset(handle, "resolved_mask"), dtype=bool)
        logm_final = np.asarray(_read_required_dataset(handle, "logM_final"), dtype=float)
        source_subhalo_id = (
            np.asarray(handle["source_subhalo_id"], dtype=np.int64)
            if "source_subhalo_id" in handle
            else np.arange(logm_final.size, dtype=np.int64)
        )
        source_group_index = (
            np.asarray(handle["source_group_index"], dtype=np.int64)
            if "source_group_index" in handle
            else np.full(logm_final.size, -1, dtype=np.int64)
        )
        source_tree_file = (
            np.asarray(handle["source_tree_file"], dtype=np.int64)
            if "source_tree_file" in handle
            else np.full(logm_final.size, -1, dtype=np.int64)
        )
        source_simulation = str(handle.attrs.get("source_simulation", "unknown"))
        source_tree = str(handle.attrs.get("source_tree", "unknown"))
        mass_unit = str(handle.attrs.get("mass_unit", ""))
        cache_z_final = _read_cache_z_final(handle, z_grid)

    if mass_unit != "Msun":
        raise ValueError(f"THESAN MAH cache mass_unit must be 'Msun'; got {mass_unit!r}")
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
    ):
        if values.ndim != 1 or values.size != mass_ratio.shape[0]:
            raise ValueError(f"THESAN MAH cache {name} must be 1D and match mass_ratio rows")
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
        "source_simulation": source_simulation,
        "source_tree": source_tree,
    }


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
    dmhdt: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    n_halos, n_steps = mass.shape
    dt_gyr = np.diff(time_gyr, prepend=time_gyr[0])
    halo_id = np.repeat(np.arange(n_halos, dtype=int), n_steps)
    step = np.tile(np.arange(n_steps, dtype=int), n_halos)
    active_flag = active_mask.reshape(-1).astype(bool)
    termination = np.full(n_halos * n_steps, "active", dtype=object)
    termination[~active_flag] = "unresolved"
    termination[np.arange(n_steps - 1, n_halos * n_steps, n_steps)] = "completed"

    return {
        "halo_id": halo_id,
        "step": step,
        "z": np.tile(redshift, n_halos),
        "t_gyr": np.tile(time_gyr, n_halos),
        "dt_gyr": np.tile(dt_gyr, n_halos),
        "Mh": mass.reshape(-1),
        "dMh_dt": dmhdt.reshape(-1),
        "active_flag": active_flag,
        "termination_flag": termination,
    }


def _compute_dmhdt(mass: np.ndarray, time_gyr: np.ndarray, resolved_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
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
    raw[negative] = 0.0
    dmhdt = np.zeros_like(mass, dtype=float)
    dmhdt[:, 1:] = raw
    return dmhdt, negative_count, total_count


def generate_thesan_halo_histories(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    cache_path: str | Path,
    z_start_max: float | None = None,
    mass_bin_width_dex: float = 0.15,
    min_candidates: int = 5,
    smoothing_myr: float = 0.0,
    random_seed: int | None = None,
    time_grid_mode: str = THESAN_TIME_GRID_SNAPSHOT,
    target_n_grid: int | None = None,
) -> HaloHistoryResult:
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

    resolved_cache_path, cache = _load_thesan_cache(cache_path, z_final=float(z_final))
    z_grid, t_grid, mass_ratio, resolved_mask = _slice_grid(
        np.asarray(cache["z_grid"], dtype=float),
        np.asarray(cache["t_gyr_grid"], dtype=float),
        np.asarray(cache["mass_ratio"], dtype=float),
        np.asarray(cache["resolved_mask"], dtype=bool),
        z_final=float(z_final),
        z_start_max=z_start_max,
    )
    mass_ratio = _smooth_mass_ratio(mass_ratio, t_grid, smoothing_myr=float(smoothing_myr))
    final_ratio = mass_ratio[:, -1]
    if np.any(final_ratio <= 0.0) or not np.all(np.isfinite(final_ratio)):
        raise ValueError("THESAN MAH cache mass_ratio final values must be finite and positive")
    mass_ratio = mass_ratio / final_ratio[:, None]
    mass_ratio[:, -1] = 1.0

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
    dmhdt, negative_count, total_dmhdt_count = _compute_dmhdt(mass, t_grid, selected_resolved_mask)
    tracks = _flatten_tracks(
        redshift=z_grid,
        time_gyr=t_grid,
        mass=mass,
        dmhdt=dmhdt,
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
        "schema_version": THESAN_MAH_CACHE_SCHEMA_VERSION,
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
    "validate_thesan_time_grid_mode",
]
