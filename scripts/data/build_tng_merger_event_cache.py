#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION = "TNG100-1-Dark"
DEFAULT_HUBBLE = 0.6774
DEFAULT_OMEGA_M = 0.3089
DEFAULT_OMEGA_B = 0.0486
SCHEMA_VERSION = "auroralf_tng_merger_event_cache_v1"
REQUIRED_FULL_TREE_FIELDS = (
    "DescendantID",
    "FirstProgenitorID",
    "Group_M_Crit200",
    "Mass",
    "NextProgenitorID",
    "SnapNum",
    "SubfindID",
    "SubhaloID",
    "SubhaloMass",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact TNG SubLink merger event cache from downloaded full.hdf5 trees. "
            "Events are direct secondary progenitors merging into the selected halo's main progenitor branch."
        )
    )
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--selection-manifest", type=str, default=None)
    parser.add_argument(
        "--snapshot-id-file",
        nargs=2,
        action="append",
        metavar=("SNAPSHOT", "ID_FILE"),
        default=None,
        help="Snapshot/id-file pair. May be passed more than once.",
    )
    parser.add_argument("--raw-full-dir", type=str, default=None)
    parser.add_argument("--mah-cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hubble", type=float, default=DEFAULT_HUBBLE)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--omega-b", type=float, default=DEFAULT_OMEGA_B)
    parser.add_argument("--limit-per-snapshot", type=int, default=None)
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_id_file(path: Path, *, limit: int | None) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"subhalo id file not found: {path}")
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        ids.append(int(stripped.split()[0]))
        if limit is not None and len(ids) >= int(limit):
            break
    if len(ids) == 0:
        raise ValueError(f"subhalo id file contains no IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"subhalo id file contains duplicate IDs: {path}")
    return ids


def _pairs_from_manifest(path: Path) -> list[tuple[int, Path]]:
    if not path.exists():
        raise FileNotFoundError(f"selection manifest not found: {path}")
    pairs: dict[tuple[int, Path], None] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot", "all_id_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"selection manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            all_id_file = str(row["all_id_file"]).strip()
            if all_id_file == "":
                continue
            pairs[(int(row["snapshot"]), _resolve_path(all_id_file))] = None
    if len(pairs) == 0:
        raise ValueError(f"selection manifest contains no all_id_file rows: {path}")
    return list(pairs)


def _collect_targets(args: argparse.Namespace) -> list[tuple[int, int]]:
    snapshot_pairs: list[tuple[int, Path]] = []
    if args.selection_manifest is not None:
        snapshot_pairs.extend(_pairs_from_manifest(_resolve_path(args.selection_manifest)))
    if args.snapshot_id_file is not None:
        for snapshot_value, id_file in args.snapshot_id_file:
            snapshot_pairs.append((int(snapshot_value), _resolve_path(id_file)))
    if len(snapshot_pairs) == 0:
        raise ValueError("provide --selection-manifest and/or at least one --snapshot-id-file SNAPSHOT ID_FILE")

    targets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for snapshot, id_file in snapshot_pairs:
        ids = _read_id_file(id_file, limit=args.limit_per_snapshot)
        for subhalo_id in ids:
            key = (int(snapshot), int(subhalo_id))
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    if len(targets) == 0:
        raise RuntimeError("no merger-cache targets were constructed")
    return targets


def _load_snapshot_redshifts(mah_cache_dir: Path) -> dict[int, float]:
    if not mah_cache_dir.exists():
        raise FileNotFoundError(f"MAH cache directory not found: {mah_cache_dir}")
    mapping: dict[int, float] = {}
    for path in sorted(mah_cache_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            if "snap_grid" not in handle or "z_grid" not in handle:
                continue
            snap_grid = np.asarray(handle["snap_grid"], dtype=np.int64)
            z_grid = np.asarray(handle["z_grid"], dtype=float)
        if snap_grid.shape != z_grid.shape:
            raise ValueError(f"snap_grid and z_grid shape mismatch in {path}")
        for snap_value, z_value in zip(snap_grid, z_grid, strict=True):
            snap = int(snap_value)
            z = float(z_value)
            if snap in mapping and not np.isclose(mapping[snap], z, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"conflicting redshift for snapshot {snap}: {mapping[snap]} vs {z}")
            mapping[snap] = z
    if len(mapping) == 0:
        raise RuntimeError(f"no snap_grid/z_grid datasets found in {mah_cache_dir}")
    return mapping


def _load_required_tree_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"full SubLink tree not found: {path}")
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_FULL_TREE_FIELDS if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing required full-tree fields: {missing}")
        for name in REQUIRED_FULL_TREE_FIELDS:
            arrays[name] = np.asarray(handle[name])
    n_rows = int(arrays["SubhaloID"].size)
    if n_rows == 0:
        raise ValueError(f"full SubLink tree contains no rows: {path}")
    for name, values in arrays.items():
        if values.shape[0] != n_rows:
            raise ValueError(f"{path} field {name!r} has inconsistent first dimension")
    return arrays


def _follow_first_progenitor_branch(
    *,
    start_index: int,
    first_progenitor_id: np.ndarray,
    subhalo_id: np.ndarray,
    id_to_index: dict[int, int],
) -> list[int]:
    branch: list[int] = []
    seen: set[int] = set()
    index = int(start_index)
    while True:
        if index in seen:
            raise RuntimeError("SubLink first-progenitor branch contains a loop")
        seen.add(index)
        branch.append(index)
        next_id = int(first_progenitor_id[index])
        if next_id < 0:
            break
        if next_id not in id_to_index:
            raise KeyError(f"FirstProgenitorID {next_id} is missing from the full tree")
        index = id_to_index[next_id]
    return branch


def _branch_peak_mass(
    *,
    start_index: int,
    mass_msun: np.ndarray,
    first_progenitor_id: np.ndarray,
    subhalo_id: np.ndarray,
    id_to_index: dict[int, int],
) -> tuple[float, int]:
    branch = _follow_first_progenitor_branch(
        start_index=start_index,
        first_progenitor_id=first_progenitor_id,
        subhalo_id=subhalo_id,
        id_to_index=id_to_index,
    )
    values = np.asarray(mass_msun[branch], dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if not np.any(valid):
        return np.nan, -1
    valid_indices = np.asarray(branch, dtype=int)[valid]
    peak_local = int(np.nanargmax(values[valid]))
    return float(values[valid][peak_local]), int(valid_indices[peak_local])


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return float(numerator / denominator)


def _ordered_ratio(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second) or first <= 0.0 or second <= 0.0:
        return np.nan
    return float(min(first, second) / max(first, second))


def _extract_tree_events(
    path: Path,
    *,
    final_snapshot: int,
    final_subhalo_id: int,
    hubble: float,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    arrays = _load_required_tree_arrays(path)
    if float(hubble) <= 0.0:
        raise ValueError("hubble must be positive")

    subhalo_id = np.asarray(arrays["SubhaloID"], dtype=np.int64)
    descendant_id = np.asarray(arrays["DescendantID"], dtype=np.int64)
    first_progenitor_id = np.asarray(arrays["FirstProgenitorID"], dtype=np.int64)
    next_progenitor_id = np.asarray(arrays["NextProgenitorID"], dtype=np.int64)
    snap = np.asarray(arrays["SnapNum"], dtype=np.int64)
    subfind = np.asarray(arrays["SubfindID"], dtype=np.int64)
    subhalo_mass_msun = np.asarray(arrays["SubhaloMass"], dtype=float) * 1.0e10 / float(hubble)
    mass_msun = np.asarray(arrays["Mass"], dtype=float) * 1.0e10 / float(hubble)
    mcrit200_msun = np.asarray(arrays["Group_M_Crit200"], dtype=float) * 1.0e10 / float(hubble)

    if len(set(int(value) for value in subhalo_id)) != subhalo_id.size:
        raise ValueError(f"full SubLink tree has duplicate SubhaloID entries: {path}")
    id_to_index = {int(value): int(index) for index, value in enumerate(subhalo_id)}
    matches = np.flatnonzero((snap == int(final_snapshot)) & (subfind == int(final_subhalo_id)))
    if matches.size != 1:
        raise ValueError(
            f"{path} must contain exactly one final node for snapshot={final_snapshot}, "
            f"subhalo={final_subhalo_id}; found {matches.size}"
        )
    final_index = int(matches[0])
    main_branch = _follow_first_progenitor_branch(
        start_index=final_index,
        first_progenitor_id=first_progenitor_id,
        subhalo_id=subhalo_id,
        id_to_index=id_to_index,
    )

    events: list[dict[str, float | int]] = []
    for descendant_index in main_branch:
        primary_id = int(first_progenitor_id[descendant_index])
        if primary_id < 0:
            continue
        if primary_id not in id_to_index:
            raise KeyError(f"FirstProgenitorID {primary_id} is missing from {path}")
        primary_index = id_to_index[primary_id]
        secondary_id = int(next_progenitor_id[primary_index])
        while secondary_id >= 0:
            if secondary_id not in id_to_index:
                raise KeyError(f"NextProgenitorID {secondary_id} is missing from {path}")
            secondary_index = id_to_index[secondary_id]
            if int(descendant_id[secondary_index]) != int(subhalo_id[descendant_index]):
                raise ValueError(
                    "secondary progenitor descendant does not match the main-branch descendant "
                    f"in {path}: secondary_id={secondary_id}"
                )

            primary_peak_mass, primary_peak_index = _branch_peak_mass(
                start_index=primary_index,
                mass_msun=subhalo_mass_msun,
                first_progenitor_id=first_progenitor_id,
                subhalo_id=subhalo_id,
                id_to_index=id_to_index,
            )
            secondary_peak_mass, secondary_peak_index = _branch_peak_mass(
                start_index=secondary_index,
                mass_msun=subhalo_mass_msun,
                first_progenitor_id=first_progenitor_id,
                subhalo_id=subhalo_id,
                id_to_index=id_to_index,
            )

            primary_direct_mass = float(subhalo_mass_msun[primary_index])
            secondary_direct_mass = float(subhalo_mass_msun[secondary_index])
            events.append(
                {
                    "final_snapshot": int(final_snapshot),
                    "final_subhalo_id": int(final_subhalo_id),
                    "descendant_snap": int(snap[descendant_index]),
                    "primary_snap": int(snap[primary_index]),
                    "secondary_snap": int(snap[secondary_index]),
                    "descendant_subfind_id": int(subfind[descendant_index]),
                    "primary_subfind_id": int(subfind[primary_index]),
                    "secondary_subfind_id": int(subfind[secondary_index]),
                    "descendant_tree_id": int(subhalo_id[descendant_index]),
                    "primary_tree_id": int(subhalo_id[primary_index]),
                    "secondary_tree_id": int(subhalo_id[secondary_index]),
                    "primary_subhalo_mass_msun": primary_direct_mass,
                    "secondary_subhalo_mass_msun": secondary_direct_mass,
                    "descendant_subhalo_mass_msun": float(subhalo_mass_msun[descendant_index]),
                    "primary_mcrit200_msun": float(mcrit200_msun[primary_index]),
                    "secondary_mcrit200_msun": float(mcrit200_msun[secondary_index]),
                    "descendant_mcrit200_msun": float(mcrit200_msun[descendant_index]),
                    "primary_tree_mass_msun": float(mass_msun[primary_index]),
                    "secondary_tree_mass_msun": float(mass_msun[secondary_index]),
                    "primary_peak_subhalo_mass_msun": primary_peak_mass,
                    "secondary_peak_subhalo_mass_msun": secondary_peak_mass,
                    "primary_peak_snap": -1 if primary_peak_index < 0 else int(snap[primary_peak_index]),
                    "secondary_peak_snap": -1 if secondary_peak_index < 0 else int(snap[secondary_peak_index]),
                    "mass_ratio_direct": _ratio(secondary_direct_mass, primary_direct_mass),
                    "mass_ratio_direct_ordered": _ordered_ratio(secondary_direct_mass, primary_direct_mass),
                    "mass_ratio_peak": _ratio(secondary_peak_mass, primary_peak_mass),
                    "mass_ratio_peak_ordered": _ordered_ratio(secondary_peak_mass, primary_peak_mass),
                }
            )
            secondary_id = int(next_progenitor_id[secondary_index])

    summary = {
        "final_snapshot": int(final_snapshot),
        "final_subhalo_id": int(final_subhalo_id),
        "final_subhalo_mass_msun": float(subhalo_mass_msun[final_index]),
        "final_mcrit200_msun": float(mcrit200_msun[final_index]),
        "main_branch_length": int(len(main_branch)),
        "tree_node_count": int(subhalo_id.size),
        "event_count": int(len(events)),
    }
    return events, summary


def _empty_event_columns() -> dict[str, list[float | int]]:
    return {
        "final_snapshot": [],
        "final_subhalo_id": [],
        "descendant_snap": [],
        "primary_snap": [],
        "secondary_snap": [],
        "descendant_subfind_id": [],
        "primary_subfind_id": [],
        "secondary_subfind_id": [],
        "descendant_tree_id": [],
        "primary_tree_id": [],
        "secondary_tree_id": [],
        "event_z": [],
        "event_t_gyr": [],
        "primary_subhalo_mass_msun": [],
        "secondary_subhalo_mass_msun": [],
        "descendant_subhalo_mass_msun": [],
        "primary_mcrit200_msun": [],
        "secondary_mcrit200_msun": [],
        "descendant_mcrit200_msun": [],
        "primary_tree_mass_msun": [],
        "secondary_tree_mass_msun": [],
        "primary_peak_subhalo_mass_msun": [],
        "secondary_peak_subhalo_mass_msun": [],
        "primary_peak_snap": [],
        "secondary_peak_snap": [],
        "mass_ratio_direct": [],
        "mass_ratio_direct_ordered": [],
        "mass_ratio_peak": [],
        "mass_ratio_peak_ordered": [],
    }


def _append_event(columns: dict[str, list[float | int]], event: dict[str, float | int], z_map: dict[int, float], cosmology: FlatLambdaCDM) -> None:
    event_snap = int(event["descendant_snap"])
    if event_snap not in z_map:
        raise KeyError(f"snapshot redshift map is missing event snapshot {event_snap}")
    event_z = float(z_map[event_snap])
    event_t_gyr = float(cosmology.age(event_z).value)
    for name in columns:
        if name == "event_z":
            columns[name].append(event_z)
        elif name == "event_t_gyr":
            columns[name].append(event_t_gyr)
        else:
            columns[name].append(event[name])


def _write_cache(
    output: Path,
    *,
    args: argparse.Namespace,
    targets: list[tuple[int, int]],
    halo_columns: dict[str, list[float | int]],
    event_columns: dict[str, list[float | int]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["source_simulation"] = str(args.simulation)
        handle.attrs["source_tree"] = "SubLink/full"
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["hubble"] = float(args.hubble)
        handle.attrs["omega_m"] = float(args.omega_m)
        handle.attrs["omega_b"] = float(args.omega_b)
        handle.attrs["target_count"] = int(len(targets))
        event_group = handle.create_group("events")
        for name, values in event_columns.items():
            array = np.asarray(values)
            event_group.create_dataset(name, data=array)
        halo_group = handle.create_group("halos")
        for name, values in halo_columns.items():
            array = np.asarray(values)
            halo_group.create_dataset(name, data=array)
    tmp.replace(output)


def main() -> None:
    args = _parse_args()
    if float(args.hubble) <= 0.0:
        raise ValueError("hubble must be positive")
    if args.limit_per_snapshot is not None and int(args.limit_per_snapshot) <= 0:
        raise ValueError("limit-per-snapshot must be positive when provided")

    raw_full_dir = (
        _resolve_path(args.raw_full_dir)
        if args.raw_full_dir is not None
        else PROJECT_ROOT / "external_data" / "tng" / str(args.simulation) / "raw_sublink_full"
    )
    mah_cache_dir = (
        _resolve_path(args.mah_cache_dir)
        if args.mah_cache_dir is not None
        else PROJECT_ROOT / "data_save" / "tng_mah_cache"
    )
    output = (
        _resolve_path(args.output)
        if args.output is not None
        else PROJECT_ROOT
        / "data_save"
        / "tng_merger_event_cache"
        / f"{args.simulation}_sublink_full_merger_events.hdf5"
    )
    if output.exists() and not args.force_output:
        raise FileExistsError(f"output cache already exists: {output}; pass --force-output to overwrite")

    targets = _collect_targets(args)
    z_map = _load_snapshot_redshifts(mah_cache_dir)
    cosmology = FlatLambdaCDM(H0=100.0 * float(args.hubble), Om0=float(args.omega_m), Ob0=float(args.omega_b))

    event_columns = _empty_event_columns()
    halo_columns: dict[str, list[float | int]] = {
        "final_snapshot": [],
        "final_subhalo_id": [],
        "final_subhalo_mass_msun": [],
        "final_mcrit200_msun": [],
        "main_branch_length": [],
        "tree_node_count": [],
        "event_start": [],
        "event_count": [],
        "major_1to4_peak_count": [],
        "major_1to10_peak_count": [],
    }

    for target_index, (snapshot, subhalo_id) in enumerate(targets, start=1):
        path = raw_full_dir / f"snap_{int(snapshot):03d}" / f"subhalo_{int(subhalo_id)}_sublink_full.hdf5"
        events, summary = _extract_tree_events(
            path,
            final_snapshot=int(snapshot),
            final_subhalo_id=int(subhalo_id),
            hubble=float(args.hubble),
        )
        event_start = len(event_columns["final_snapshot"])
        for event in events:
            _append_event(event_columns, event, z_map, cosmology)
        ratios = np.asarray([event["mass_ratio_peak_ordered"] for event in events], dtype=float)
        finite = np.isfinite(ratios)

        for name in (
            "final_snapshot",
            "final_subhalo_id",
            "final_subhalo_mass_msun",
            "final_mcrit200_msun",
            "main_branch_length",
            "tree_node_count",
        ):
            halo_columns[name].append(summary[name])
        halo_columns["event_start"].append(event_start)
        halo_columns["event_count"].append(int(len(events)))
        halo_columns["major_1to4_peak_count"].append(int(np.count_nonzero(finite & (ratios >= 0.25))))
        halo_columns["major_1to10_peak_count"].append(int(np.count_nonzero(finite & (ratios >= 0.10))))

        if target_index == 1 or target_index % 100 == 0 or target_index == len(targets):
            print(
                f"progress={target_index}/{len(targets)} events={len(event_columns['final_snapshot'])} "
                f"last_snap={snapshot} last_subhalo={subhalo_id}",
                flush=True,
            )

    _write_cache(output, args=args, targets=targets, halo_columns=halo_columns, event_columns=event_columns)
    print(f"saved_tng_merger_event_cache={output}", flush=True)
    print(f"n_halos={len(targets)}", flush=True)
    print(f"n_events={len(event_columns['final_snapshot'])}", flush=True)


if __name__ == "__main__":
    main()
