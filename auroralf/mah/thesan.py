from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from auroralf.file_version import FileVersion

from . import _simulation_backend as shared
from .models import Cosmology, HaloHistoryResult
from .tng import MAH_BACKEND_THESAN


THESAN_MAH_CACHE_SCHEMA_VERSION = "auroralf_thesan_mah_cache_v1"
THESAN_TIME_GRID_SNAPSHOT = "snapshot"
THESAN_TIME_GRID_UNIFORM_IN_T = "uniform_in_t"
THESAN_TIME_GRID_MODES = (THESAN_TIME_GRID_SNAPSHOT, THESAN_TIME_GRID_UNIFORM_IN_T)
_THESAN_MAH_CACHE: dict[tuple[FileVersion, float], dict[str, Any]] = {}
_CACHE_LABEL = "THESAN"


def validate_thesan_time_grid_mode(mode: str) -> str:
    return shared.normalize_choice(
        mode,
        choices=THESAN_TIME_GRID_MODES,
        field_name="thesan_time_grid_mode",
    )


def _read_required_dataset(handle: h5py.File, name: str) -> np.ndarray:
    return shared.read_required_dataset(handle, name, cache_label=_CACHE_LABEL)


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
            reader = shared.CacheReader(handle, _CACHE_LABEL, _read_required_dataset)
            try:
                schema_version = reader.text_attr("schema_version")
            except (KeyError, ValueError):
                continue
            if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
                continue
            z_grid = _read_required_dataset(handle, "z_grid")
            cache_z_final = reader.z_final(np.asarray(z_grid, dtype=float))
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
        opened_version = shared.version_from_open_file(version, raw_handle.fileno())
        if opened_version != version:
            raise RuntimeError(f"THESAN MAH cache changed during preload: {resolved}")
        with h5py.File(raw_handle, "r") as handle:
            reader = shared.CacheReader(handle, _CACHE_LABEL, _read_required_dataset)
            schema_version = reader.text_attr("schema_version")
            if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    "THESAN MAH cache schema_version must be "
                    f"{THESAN_MAH_CACHE_SCHEMA_VERSION!r}; got {schema_version!r}"
                )
            z_grid = np.asarray(_read_required_dataset(handle, "z_grid"), dtype=float)
            t_gyr_grid = np.asarray(_read_required_dataset(handle, "t_gyr_grid"), dtype=float)
            mass_ratio = np.asarray(_read_required_dataset(handle, "mass_ratio"), dtype=float)
            resolved_mask = reader.bool_dataset("resolved_mask")
            logm_final = np.asarray(_read_required_dataset(handle, "logM_final"), dtype=float)
            source_subhalo_id = reader.identifier_dataset("source_subhalo_id")
            source_group_index = reader.identifier_dataset("source_group_index")
            source_tree_file = reader.identifier_dataset("source_tree_file")
            source_tree_num = reader.identifier_dataset("source_tree_num")
            source_tree_index = reader.identifier_dataset("source_tree_index")
            source_snapshot = reader.identifier_dataset("source_snapshot")
            source_file_identifier = reader.text_dataset("source_file_identifier")
            source_file_sha256 = reader.text_dataset("source_file_sha256")
            reader.validate_source_file_checksums(source_file_identifier, source_file_sha256)
            source_simulation = reader.text_attr("source_simulation")
            source_tree = reader.text_attr("source_tree")
            snapshot = reader.nonnegative_int_attr("snapshot")
            mass_unit = reader.text_attr("mass_unit")
            time_unit = reader.text_attr("time_unit")
            redshift_unit = reader.text_attr("redshift_unit")
            mass_ratio_unit = reader.text_attr("mass_ratio_unit")
            selection_description = reader.text_attr("selection_description")
            creator_version = reader.text_attr("creator_version")
            cache_z_final = reader.z_final(z_grid)
            hubble = reader.cosmology_attr("hubble")
            omega_m = reader.cosmology_attr("omega_m")
            omega_b = reader.cosmology_attr("omega_b")
    shared.require_current_file_version(version, cache_label=_CACHE_LABEL)

    shared.validate_cache_contents(
        cache_label=_CACHE_LABEL,
        z_grid=z_grid,
        t_gyr_grid=t_gyr_grid,
        mass_ratio=mass_ratio,
        resolved_mask=resolved_mask,
        logm_final=logm_final,
        source_snapshot=source_snapshot,
        identifier_fields=(
            ("source_subhalo_id", source_subhalo_id),
            ("source_group_index", source_group_index),
            ("source_tree_file", source_tree_file),
            ("source_tree_num", source_tree_num),
            ("source_tree_index", source_tree_index),
            ("source_snapshot", source_snapshot),
        ),
        source_identity_columns=(
            source_snapshot,
            source_tree_file,
            source_tree_num,
            source_tree_index,
        ),
        snapshot=snapshot,
        mass_unit=mass_unit,
        time_unit=time_unit,
        redshift_unit=redshift_unit,
        mass_ratio_unit=mass_ratio_unit,
        hubble=hubble,
        omega_m=omega_m,
        omega_b=omega_b,
        cache_z_final=cache_z_final,
        requested_z_final=float(z_final),
    )

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
        shared.require_current_file_version(version, cache_label=_CACHE_LABEL)
        shared.validate_cache_cosmology(cached, cosmology, cache_label=_CACHE_LABEL)
        return version
    loaded_path, loaded = _read_thesan_cache_file(
        version,
        z_final=float(z_final),
    )
    if loaded_path != version.path:
        raise RuntimeError("resolved THESAN cache path changed during preload")
    shared.validate_cache_cosmology(loaded, cosmology, cache_label=_CACHE_LABEL)
    frozen = shared.freeze_cache(loaded)
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
    time_grid_mode = shared.validate_generation_parameters(
        field_prefix="thesan",
        time_grid_modes=THESAN_TIME_GRID_MODES,
        uniform_time_grid_mode=THESAN_TIME_GRID_UNIFORM_IN_T,
        cosmology=cosmology,
        n_tracks=n_tracks,
        halo_mass_final=Mh_final,
        mass_bin_width_dex=mass_bin_width_dex,
        min_candidates=min_candidates,
        time_grid_mode=time_grid_mode,
        target_n_grid=target_n_grid,
    )

    resolved_cache_path, cache = _load_thesan_cache(
        cache_path,
        z_final=float(z_final),
        cosmology=cosmology,
    )
    z_grid, t_grid, mass_ratio, resolved_mask = shared.slice_grid(
        np.asarray(cache["z_grid"], dtype=float),
        np.asarray(cache["t_gyr_grid"], dtype=float),
        np.asarray(cache["mass_ratio"], dtype=float),
        np.asarray(cache["resolved_mask"], dtype=bool),
        cache_label=_CACHE_LABEL,
        z_final=float(z_final),
        z_start_max=z_start_max,
    )
    mass_ratio = shared.smooth_mass_ratio(
        mass_ratio,
        t_grid,
        smoothing_myr=float(smoothing_myr),
        cache_label=_CACHE_LABEL,
        field_prefix="thesan",
    )
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
        z_grid, t_grid, selected_ratio, selected_resolved_mask = (
            shared.regrid_mass_ratio_uniform_in_t(
                cache_label=_CACHE_LABEL,
                z_grid=z_grid,
                t_gyr_grid=t_grid,
                mass_ratio=selected_ratio,
                resolved_mask=selected_resolved_mask,
                target_n_grid=int(target_n_grid),
            )
        )
    mass = selected_ratio * float(Mh_final)
    dmhdt_raw, dmhdt_sfr, dmhdt_clipped, negative_count, total_dmhdt_count = (
        shared.compute_dmhdt(
            mass,
            t_grid,
            selected_resolved_mask,
            cache_label=_CACHE_LABEL,
        )
    )
    tracks = shared.flatten_tracks(
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
