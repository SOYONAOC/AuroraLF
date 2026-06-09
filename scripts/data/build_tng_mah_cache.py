#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import requests
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.mah.tng import TNG_MAH_CACHE_SCHEMA_VERSION


DEFAULT_SIMULATION = "TNG100-1-Dark"
DEFAULT_API_BASE = "https://www.tng-project.org/api"
DEFAULT_HUBBLE = 0.6774
DEFAULT_OMEGA_M = 0.3089
DEFAULT_OMEGA_B = 0.0486
DEFAULT_MASS_FIELD = "Group_M_Crit200"
DEFAULT_MISSING_MASS_RATIO_FLOOR = 1.0e-6


def _tag_from_z(z_value: float) -> str:
    return f"z{float(z_value):.3f}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AuroraLF TNG MAH cache from selected TNG SubLink main-progenitor branches. "
            "The script downloads raw mpb.hdf5 files with TNG_API_KEY and writes a compact "
            "mass-ratio cache consumed by mah_backend='tng'."
        )
    )
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--z-final", type=float, required=True)
    parser.add_argument("--subhalo-ids", nargs="+", type=int, default=None)
    parser.add_argument("--subhalo-id-file", type=str, default=None)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mass-field", default=DEFAULT_MASS_FIELD)
    parser.add_argument("--hubble", type=float, default=DEFAULT_HUBBLE)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--omega-b", type=float, default=DEFAULT_OMEGA_B)
    parser.add_argument(
        "--snapshot-grid",
        choices=("common", "union"),
        default="common",
        help=(
            "Use the common positive-mass snapshots shared by every selected MPB, "
            "or the union of available snapshots with unresolved prehistory filled by a small mass floor."
        ),
    )
    parser.add_argument(
        "--missing-mass-ratio-floor",
        type=float,
        default=DEFAULT_MISSING_MASS_RATIO_FLOOR,
        help="For --snapshot-grid union, mass assigned before the first resolved MPB point as a ratio of final mass.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Number of parallel TNG MPB downloads. Existing raw files are reused.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=0,
        help="Number of retries for transient MPB download failures before raising an error.",
    )
    parser.add_argument(
        "--drop-invalid-mpb",
        action="store_true",
        help=(
            "Drop downloaded MPB files that cannot provide at least two positive finite "
            "mass samples. Dropped subhalo IDs and reasons are written to the cache."
        ),
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError("TNG_API_KEY must be set to download TNG API data")
    return api_key


def _load_subhalo_ids(args: argparse.Namespace) -> np.ndarray:
    ids: list[int] = []
    if args.subhalo_ids is not None:
        ids.extend(int(value) for value in args.subhalo_ids)
    if args.subhalo_id_file is not None:
        path = Path(args.subhalo_id_file).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"subhalo id file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            ids.append(int(stripped.split()[0]))
    if len(ids) == 0:
        raise ValueError("provide at least one subhalo id via --subhalo-ids or --subhalo-id-file")
    unique = np.unique(np.asarray(ids, dtype=np.int64))
    if unique.size != len(ids):
        raise ValueError("subhalo ids must be unique")
    return unique


def _request_json(url: str, api_key: str) -> object:
    response = requests.get(url, headers={"api-key": api_key}, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"TNG API request failed with status {response.status_code}: {url}")
    return response.json()


def _download_file(url: str, destination: Path, api_key: str, *, force: bool, retries: int) -> None:
    if destination.exists() and not force:
        print(f"using_existing_raw_subtree={destination}", flush=True)
        return
    if int(retries) < 0:
        raise ValueError("download-retries must be non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    attempts = int(retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, headers={"api-key": api_key}, timeout=120, stream=True) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"TNG API download failed with status {response.status_code}: {url}")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                tmp.replace(destination)
            print(f"downloaded_raw_subtree={destination}", flush=True)
            return
        except (OSError, RuntimeError, requests.RequestException) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt >= attempts:
                raise RuntimeError(
                    f"TNG MPB download failed after {attempts} attempt(s): {destination}"
                ) from exc
            print(
                f"retrying_raw_subtree={destination} attempt={attempt + 1}/{attempts} "
                f"reason={exc.__class__.__name__}",
                flush=True,
            )
            time.sleep(min(2.0 ** (attempt - 1), 30.0))


def _download_raw_subtrees(
    *,
    jobs: list[tuple[str, Path]],
    api_key: str,
    force: bool,
    workers: int,
    retries: int,
) -> None:
    if int(workers) < 1:
        raise ValueError("download-workers must be positive")
    if int(workers) == 1:
        for url, raw_path in jobs:
            _download_file(url, raw_path, api_key, force=force, retries=int(retries))
        return
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        future_to_path = {
            executor.submit(_download_file, url, raw_path, api_key, force=force, retries=int(retries)): raw_path
            for url, raw_path in jobs
        }
        for future in as_completed(future_to_path):
            future.result()


def _snapshot_redshift_map(api_base: str, simulation: str, api_key: str) -> dict[int, float]:
    url = f"{api_base.rstrip('/')}/{simulation}/snapshots/"
    payload = _request_json(url, api_key)
    if not isinstance(payload, list):
        raise RuntimeError(f"TNG snapshots endpoint returned non-list payload: {url}")
    mapping: dict[int, float] = {}
    for item in payload:
        if not isinstance(item, dict) or "number" not in item or "redshift" not in item:
            raise RuntimeError("TNG snapshots endpoint payload is missing number/redshift fields")
        mapping[int(item["number"])] = float(item["redshift"])
    if len(mapping) == 0:
        raise RuntimeError("TNG snapshots endpoint returned no snapshots")
    return mapping


def _read_mpb(path: Path, *, mass_field: str, hubble: float) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if "SnapNum" not in handle:
            raise KeyError(f"{path} is missing SnapNum")
        if mass_field not in handle:
            raise KeyError(f"{path} is missing requested mass field {mass_field!r}")
        snap = np.asarray(handle["SnapNum"], dtype=np.int64)
        mass_code = np.asarray(handle[mass_field], dtype=float)
    if snap.ndim != 1 or mass_code.ndim != 1 or snap.size != mass_code.size:
        raise ValueError(f"{path} SnapNum and {mass_field} must be matching 1D arrays")
    if float(hubble) <= 0.0:
        raise ValueError("hubble must be positive")
    mass_msun = mass_code * 1.0e10 / float(hubble)
    valid = np.isfinite(mass_msun) & (mass_msun > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"{path} contains fewer than two positive finite mass samples")
    return snap[valid], mass_msun[valid]


def _build_cache_arrays(
    *,
    raw_paths: list[Path],
    source_subhalo_ids: np.ndarray,
    snapshot: int,
    snapshot_redshift: dict[int, float],
    mass_field: str,
    hubble: float,
    cosmology: FlatLambdaCDM,
    drop_invalid_mpb: bool = False,
    snapshot_grid: str = "common",
    missing_mass_ratio_floor: float = DEFAULT_MISSING_MASS_RATIO_FLOOR,
) -> dict[str, np.ndarray]:
    per_tree: list[dict[int, float]] = []
    valid_source_subhalo_ids: list[int] = []
    dropped_source_subhalo_ids: list[int] = []
    dropped_reasons: list[str] = []
    common_snaps: set[int] | None = None
    union_snaps: set[int] = set()
    for raw_path, source_subhalo_id in zip(raw_paths, source_subhalo_ids, strict=True):
        try:
            snap, mass = _read_mpb(raw_path, mass_field=mass_field, hubble=hubble)
        except (KeyError, ValueError) as exc:
            if not drop_invalid_mpb:
                raise
            dropped_source_subhalo_ids.append(int(source_subhalo_id))
            dropped_reasons.append(str(exc))
            continue
        tree_map = {int(snap_value): float(mass_value) for snap_value, mass_value in zip(snap, mass, strict=True)}
        if int(snapshot) not in tree_map:
            if not drop_invalid_mpb:
                raise ValueError(f"{raw_path} does not contain requested final snapshot {snapshot}")
            dropped_source_subhalo_ids.append(int(source_subhalo_id))
            dropped_reasons.append(f"{raw_path} does not contain requested final snapshot {snapshot}")
            continue
        snap_set = set(tree_map)
        common_snaps = snap_set if common_snaps is None else common_snaps & snap_set
        union_snaps |= snap_set
        per_tree.append(tree_map)
        valid_source_subhalo_ids.append(int(source_subhalo_id))

    if snapshot_grid not in {"common", "union"}:
        raise ValueError("snapshot_grid must be 'common' or 'union'")
    if float(missing_mass_ratio_floor) <= 0.0 or not np.isfinite(float(missing_mass_ratio_floor)):
        raise ValueError("missing_mass_ratio_floor must be finite and positive")

    if common_snaps is None or len(common_snaps) < 2:
        if snapshot_grid == "common":
            raise RuntimeError("selected TNG MPB trees have fewer than two common snapshots")
    if snapshot_grid == "common":
        grid_snaps = set(common_snaps or set())
        if int(snapshot) not in grid_snaps:
            raise RuntimeError("selected TNG MPB trees do not share the requested final snapshot")
    else:
        grid_snaps = set(union_snaps)
        if int(snapshot) not in grid_snaps:
            raise RuntimeError("selected TNG MPB trees do not include the requested final snapshot")
    if len(grid_snaps) < 2:
        raise RuntimeError("selected TNG MPB trees have fewer than two snapshots on the requested grid")
    missing_redshift = sorted(snap for snap in grid_snaps if snap not in snapshot_redshift)
    if missing_redshift:
        raise KeyError(f"snapshot redshift table is missing snapshots: {missing_redshift}")

    grid_snap_array = np.array(sorted(grid_snaps, key=lambda snap: snapshot_redshift[snap], reverse=True), dtype=np.int64)
    z_grid = np.array([snapshot_redshift[int(snap)] for snap in grid_snap_array], dtype=float)
    t_gyr_grid = np.asarray(cosmology.age(z_grid).value, dtype=float)
    if np.any(np.diff(z_grid) >= 0.0):
        raise RuntimeError("constructed z_grid is not strictly decreasing")
    if np.any(np.diff(t_gyr_grid) <= 0.0):
        raise RuntimeError("constructed t_gyr_grid is not strictly increasing")

    mass_history = np.empty((len(per_tree), grid_snap_array.size), dtype=float)
    resolved_mask = np.empty((len(per_tree), grid_snap_array.size), dtype=bool)
    resolved_snap_count = np.empty(len(per_tree), dtype=np.int64)
    filled_snap_count = np.empty(len(per_tree), dtype=np.int64)
    for tree_index, tree_map in enumerate(per_tree):
        if snapshot_grid == "common":
            mass_history[tree_index] = np.array([tree_map[int(snap)] for snap in grid_snap_array], dtype=float)
            resolved_mask[tree_index] = True
            resolved_snap_count[tree_index] = grid_snap_array.size
            filled_snap_count[tree_index] = 0
            continue

        known_snaps = np.array(sorted(tree_map, key=lambda snap: snapshot_redshift[snap], reverse=True), dtype=np.int64)
        known_z = np.array([snapshot_redshift[int(snap)] for snap in known_snaps], dtype=float)
        known_t = np.asarray(cosmology.age(known_z).value, dtype=float)
        known_mass = np.array([tree_map[int(snap)] for snap in known_snaps], dtype=float)
        if np.any(np.diff(known_t) <= 0.0):
            raise RuntimeError("known MPB time grid must be strictly increasing")
        final_mass_for_floor = float(tree_map[int(snapshot)])
        floor_mass = min(
            float(missing_mass_ratio_floor) * final_mass_for_floor,
            0.999999 * float(np.min(known_mass)),
        )
        if floor_mass <= 0.0 or not np.isfinite(floor_mass):
            raise RuntimeError("constructed missing-snapshot mass floor is invalid")
        interpolated = np.exp(np.interp(t_gyr_grid, known_t, np.log(known_mass)))
        before_first = t_gyr_grid < known_t[0]
        interpolated[before_first] = floor_mass
        known_by_snap = {int(snap): float(mass) for snap, mass in zip(known_snaps, known_mass, strict=True)}
        is_resolved = np.zeros(grid_snap_array.size, dtype=bool)
        for step, snap_value in enumerate(grid_snap_array):
            if int(snap_value) in known_by_snap:
                interpolated[step] = known_by_snap[int(snap_value)]
                is_resolved[step] = True
        mass_history[tree_index] = interpolated
        resolved_mask[tree_index] = is_resolved
        resolved_snap_count[tree_index] = int(np.count_nonzero(is_resolved))
        filled_snap_count[tree_index] = int(grid_snap_array.size - np.count_nonzero(is_resolved))
    final_mass = mass_history[:, -1]
    if np.any(final_mass <= 0.0) or not np.all(np.isfinite(final_mass)):
        raise RuntimeError("final masses must be finite and positive")

    return {
        "source_subhalo_id": np.asarray(valid_source_subhalo_ids, dtype=np.int64),
        "source_snapshot": np.full(len(valid_source_subhalo_ids), int(snapshot), dtype=np.int64),
        "snap_grid": grid_snap_array,
        "z_grid": z_grid,
        "t_gyr_grid": t_gyr_grid,
        "mass_ratio": mass_history / final_mass[:, None],
        "resolved_mask": resolved_mask,
        "logM_final": np.log10(final_mass),
        "resolved_snap_count": resolved_snap_count,
        "filled_snap_count": filled_snap_count,
        "dropped_source_subhalo_id": np.asarray(dropped_source_subhalo_ids, dtype=np.int64),
        "dropped_reason": np.asarray(dropped_reasons, dtype=h5py.string_dtype(encoding="utf-8")),
    }


def main() -> None:
    args = _parse_args()
    api_key = _require_api_key()
    subhalo_ids = _load_subhalo_ids(args)
    if int(args.download_workers) < 1:
        raise ValueError("download-workers must be positive")
    if int(args.download_retries) < 0:
        raise ValueError("download-retries must be non-negative")
    if float(args.missing_mass_ratio_floor) <= 0.0 or not np.isfinite(float(args.missing_mass_ratio_floor)):
        raise ValueError("missing-mass-ratio-floor must be finite and positive")
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir is not None else (
        PROJECT_ROOT / "external_data" / "tng" / args.simulation / "raw_subtrees"
    )
    if not raw_dir.is_absolute():
        raw_dir = (PROJECT_ROOT / raw_dir).resolve()
    else:
        raw_dir = raw_dir.resolve()

    output = Path(args.output).expanduser() if args.output is not None else (
        PROJECT_ROOT
        / "data_save"
        / "tng_mah_cache"
        / f"{args.simulation}_sublink_mpb_{_tag_from_z(float(args.z_final))}.hdf5"
    )
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    else:
        output = output.resolve()
    if output.exists() and not args.force_output:
        raise FileExistsError(f"output cache already exists: {output}; pass --force-output to overwrite")

    snapshot_redshift = _snapshot_redshift_map(args.api_base, args.simulation, api_key)
    if int(args.snapshot) not in snapshot_redshift:
        raise KeyError(f"snapshot {args.snapshot} is not present in TNG API snapshot list")
    api_z_final = float(snapshot_redshift[int(args.snapshot)])
    if not np.isclose(api_z_final, float(args.z_final), rtol=0.0, atol=1.0e-3):
        raise ValueError(
            f"--z-final {float(args.z_final):g} does not match API snapshot redshift "
            f"{api_z_final:g} for snapshot {int(args.snapshot)}"
        )

    raw_paths: list[Path] = []
    download_jobs: list[tuple[str, Path]] = []
    for subhalo_id in subhalo_ids:
        raw_path = raw_dir / f"snap_{int(args.snapshot):03d}_subhalo_{int(subhalo_id)}_sublink_mpb.hdf5"
        url = (
            f"{args.api_base.rstrip('/')}/{args.simulation}/snapshots/{int(args.snapshot)}"
            f"/subhalos/{int(subhalo_id)}/sublink/mpb.hdf5"
        )
        raw_paths.append(raw_path)
        download_jobs.append((url, raw_path))
    _download_raw_subtrees(
        jobs=download_jobs,
        api_key=api_key,
        force=bool(args.force_download),
        workers=int(args.download_workers),
        retries=int(args.download_retries),
    )

    cosmology = FlatLambdaCDM(
        H0=100.0 * float(args.hubble),
        Om0=float(args.omega_m),
        Ob0=float(args.omega_b),
    )
    arrays = _build_cache_arrays(
        raw_paths=raw_paths,
        source_subhalo_ids=subhalo_ids,
        snapshot=int(args.snapshot),
        snapshot_redshift=snapshot_redshift,
        mass_field=str(args.mass_field),
        hubble=float(args.hubble),
        cosmology=cosmology,
        drop_invalid_mpb=bool(args.drop_invalid_mpb),
        snapshot_grid=str(args.snapshot_grid),
        missing_mass_ratio_floor=float(args.missing_mass_ratio_floor),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = TNG_MAH_CACHE_SCHEMA_VERSION
        handle.attrs["source_simulation"] = str(args.simulation)
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["source_mass_field"] = str(args.mass_field)
        handle.attrs["source_tree"] = "SubLink/mpb"
        handle.attrs["z_final"] = float(args.z_final)
        handle.attrs["snapshot"] = int(args.snapshot)
        handle.attrs["hubble"] = float(args.hubble)
        handle.attrs["omega_m"] = float(args.omega_m)
        handle.attrs["omega_b"] = float(args.omega_b)
        handle.attrs["drop_invalid_mpb"] = bool(args.drop_invalid_mpb)
        handle.attrs["dropped_mpb_count"] = int(arrays["dropped_source_subhalo_id"].size)
        handle.attrs["snapshot_grid"] = str(args.snapshot_grid)
        handle.attrs["missing_mass_ratio_floor"] = float(args.missing_mass_ratio_floor)
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)
    tmp.replace(output)

    print(f"saved_tng_mah_cache={output}", flush=True)
    print(f"n_halos={arrays['source_subhalo_id'].size}", flush=True)
    print(f"n_dropped_mpb={arrays['dropped_source_subhalo_id'].size}", flush=True)
    print(f"n_snapshots={arrays['z_grid'].size}", flush=True)


if __name__ == "__main__":
    main()
