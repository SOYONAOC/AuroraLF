#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


THESAN_MAH_CACHE_SCHEMA_VERSION = "auroralf_thesan_mah_cache_v0"
DEFAULT_THESAN_ROOT = PROJECT_ROOT / "external_data/thesan/thesan-dark-1"
DEFAULT_SELECTED_HALOS = PROJECT_ROOT / "outputs/thesan_discovery/selected_halos.csv"
DEFAULT_HUBBLE = 0.6774
DEFAULT_OMEGA_M = 0.3089
DEFAULT_OMEGA_B = 0.0486
DEFAULT_MASS_FIELD = "Group_M_Crit200"
DEFAULT_UNRESOLVED_MASS_RATIO_FILL = 1.0e-6


def _tag_from_z(z_value: float) -> str:
    return f"z{float(z_value):.3f}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact AuroraLF THESAN-dark-1 MAH smoke cache from selected "
            "LHaloTree main-progenitor branches. This script never downloads data."
        )
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_THESAN_ROOT))
    parser.add_argument("--selected-halos", type=str, default=str(DEFAULT_SELECTED_HALOS))
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument(
        "--tree-file",
        default="0",
        help="LHaloTree chunk number to use, or 'all' to use every tree_file listed in --selected-halos.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mass-field", default=DEFAULT_MASS_FIELD)
    parser.add_argument(
        "--branch-start",
        choices=("first_fof", "subhalo"),
        default="first_fof",
        help=(
            "Start the MPB from the final snapshot FoF first-halo node for halo masses, "
            "or from the selected subhalo offset node."
        ),
    )
    parser.add_argument("--hubble", type=float, default=DEFAULT_HUBBLE)
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--omega-b", type=float, default=DEFAULT_OMEGA_B)
    parser.add_argument("--min-logm", type=float, default=None)
    parser.add_argument("--max-logm", type=float, default=None)
    parser.add_argument("--max-halos", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--unresolved-mass-ratio-fill", type=float, default=DEFAULT_UNRESOLVED_MASS_RATIO_FILL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _read_selected_rows(
    selected_path: Path,
    *,
    snapshot: int,
    tree_file: int | None,
    min_logm: float | None,
    max_logm: float | None,
    max_halos: int | None,
    random_seed: int,
) -> list[dict[str, str]]:
    if not selected_path.exists():
        raise FileNotFoundError(f"selected halo table not found: {selected_path}")
    rows: list[dict[str, str]] = []
    with selected_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "snapshot",
            "redshift",
            "group_index",
            "source_subhalo_id",
            "logM_final",
            "tree_file",
            "tree_num",
            "tree_index",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"selected halo table is missing required columns: {missing}")
        for row in reader:
            if int(row["snapshot"]) != int(snapshot):
                continue
            if tree_file is not None and int(row["tree_file"]) != int(tree_file):
                continue
            logm = float(row["logM_final"])
            if min_logm is not None and logm < float(min_logm):
                continue
            if max_logm is not None and logm >= float(max_logm):
                continue
            rows.append(row)

    if not rows:
        raise ValueError(
            f"no selected halos match snapshot={snapshot}, tree_file={tree_file}, "
            f"min_logm={min_logm}, max_logm={max_logm}"
        )
    if max_halos is not None:
        if int(max_halos) <= 0:
            raise ValueError("--max-halos must be positive when provided")
        if len(rows) > int(max_halos):
            rng = np.random.default_rng(int(random_seed))
            indices = np.sort(rng.choice(len(rows), size=int(max_halos), replace=False))
            rows = [rows[int(index)] for index in indices]
    return rows


def _parse_tree_file_filter(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return None
    try:
        tree_file = int(normalized)
    except ValueError as exc:
        raise ValueError("--tree-file must be an integer chunk number or 'all'") from exc
    if tree_file < 0:
        raise ValueError("--tree-file chunk number must be non-negative")
    return tree_file


def _read_branch(
    tree_group: h5py.Group,
    *,
    start_index: int,
    final_snapshot: int,
    mass_field: str,
    hubble: float,
) -> tuple[dict[int, float], int, int]:
    required = {"SnapNum", "FirstProgenitor", mass_field}
    missing = sorted(required.difference(tree_group.keys()))
    if missing:
        raise KeyError(f"{tree_group.name} is missing required datasets: {missing}")
    if float(hubble) <= 0.0:
        raise ValueError("hubble must be positive")

    snap_data = tree_group["SnapNum"]
    first_prog = tree_group["FirstProgenitor"]
    mass_data = tree_group[mass_field]
    n_nodes = snap_data.shape[0]
    current = int(start_index)
    branch: dict[int, float] = {}
    zero_mass_nodes = 0
    visited = 0
    seen_indices: set[int] = set()
    while current >= 0:
        if current >= n_nodes:
            raise IndexError(f"{tree_group.name} index {current} is outside dataset length {n_nodes}")
        if current in seen_indices:
            raise RuntimeError(f"{tree_group.name} FirstProgenitor loop detected at index {current}")
        seen_indices.add(current)

        snap = int(snap_data[current])
        if snap > int(final_snapshot):
            raise ValueError(
                f"{tree_group.name} branch reached snapshot {snap}, later than final snapshot {final_snapshot}"
            )
        if snap in branch:
            raise ValueError(f"{tree_group.name} branch has duplicate snapshot {snap}")
        mass_code = float(mass_data[current])
        if np.isfinite(mass_code) and mass_code > 0.0:
            branch[snap] = mass_code * 1.0e10 / float(hubble)
        else:
            branch[snap] = np.nan
            zero_mass_nodes += 1
        current = int(first_prog[current])
        visited += 1

    if int(final_snapshot) not in branch:
        raise ValueError(f"{tree_group.name} branch does not include final snapshot {final_snapshot}")
    final_mass = branch[int(final_snapshot)]
    if not np.isfinite(final_mass) or final_mass <= 0.0:
        raise ValueError(f"{tree_group.name} branch has invalid final mass at snapshot {final_snapshot}")
    return branch, visited, zero_mass_nodes


def _build_cache(args: argparse.Namespace) -> Path:
    root = _resolve_project_path(args.root)
    selected_path = _resolve_project_path(args.selected_halos)
    tree_file_filter = _parse_tree_file_filter(str(args.tree_file))

    rows = _read_selected_rows(
        selected_path,
        snapshot=int(args.snapshot),
        tree_file=tree_file_filter,
        min_logm=args.min_logm,
        max_logm=args.max_logm,
        max_halos=args.max_halos,
        random_seed=int(args.random_seed),
    )

    if float(args.unresolved_mass_ratio_fill) <= 0.0:
        raise ValueError("--unresolved-mass-ratio-fill must be positive")

    cosmology = FlatLambdaCDM(H0=100.0 * float(args.hubble), Om0=float(args.omega_m), Ob0=float(args.omega_b))
    requested_tree_files = sorted({int(row["tree_file"]) for row in rows})
    tree_paths = {
        tree_file: root / "postprocessing/trees/LHaloTree" / f"trees_sf1_190.{tree_file}.hdf5"
        for tree_file in requested_tree_files
    }
    missing_tree_paths = [path for path in tree_paths.values() if not path.exists()]
    if missing_tree_paths:
        names = ", ".join(str(path) for path in missing_tree_paths[:10])
        if len(missing_tree_paths) > 10:
            names += f", ... ({len(missing_tree_paths)} total)"
        raise FileNotFoundError(f"THESAN LHaloTree file(s) not found: {names}")

    header_tree_file = requested_tree_files[0]
    with h5py.File(tree_paths[header_tree_file], "r") as handle:
        if "Header/Redshifts" not in handle:
            raise KeyError(f"{tree_paths[header_tree_file]} is missing Header/Redshifts")
        redshift_table = np.asarray(handle["Header/Redshifts"], dtype=float)
        if int(args.snapshot) >= redshift_table.size:
            raise ValueError(
                f"requested snapshot {args.snapshot} is outside redshift table length {redshift_table.size}"
            )
        snap_grid = np.arange(0, int(args.snapshot) + 1, dtype=np.int64)
        z_grid = redshift_table[snap_grid]
        if np.any(np.diff(z_grid) >= 0.0):
            raise ValueError("THESAN redshift grid must be strictly decreasing with snapshot number")
        t_gyr_grid = np.asarray(cosmology.age(z_grid).value, dtype=float)
        if np.any(np.diff(t_gyr_grid) <= 0.0):
            raise ValueError("THESAN time grid must be strictly increasing")

        n_halos = len(rows)
        n_steps = snap_grid.size
        mass_ratio = np.full((n_halos, n_steps), float(args.unresolved_mass_ratio_fill), dtype=float)
        resolved_mask = np.zeros((n_halos, n_steps), dtype=bool)
        mass_msun = np.full((n_halos, n_steps), np.nan, dtype=float)
        source_subhalo_id = np.empty(n_halos, dtype=np.int64)
        source_group_index = np.empty(n_halos, dtype=np.int64)
        source_tree_file = np.empty(n_halos, dtype=np.int64)
        source_tree_num = np.empty(n_halos, dtype=np.int64)
        source_tree_index = np.empty(n_halos, dtype=np.int64)
        branch_start_tree_index = np.empty(n_halos, dtype=np.int64)
        logm_final_catalog = np.empty(n_halos, dtype=float)
        logm_final_tree = np.empty(n_halos, dtype=float)
        branch_length = np.empty(n_halos, dtype=np.int64)
        zero_mass_node_count = np.empty(n_halos, dtype=np.int64)
        resolved_snap_count = np.empty(n_halos, dtype=np.int64)

        snap_to_column = {int(snap): column for column, snap in enumerate(snap_grid)}
        rows_by_tree_file: dict[int, list[tuple[int, dict[str, str]]]] = {}
        for row_index, row in enumerate(rows):
            rows_by_tree_file.setdefault(int(row["tree_file"]), []).append((row_index, row))

        for tree_file in requested_tree_files:
            tree_path = tree_paths[tree_file]
            with h5py.File(tree_path, "r") as handle:
                for row_index, row in rows_by_tree_file[tree_file]:
                    tree_name = f"Tree{int(row['tree_num'])}"
                    if tree_name not in handle:
                        raise KeyError(f"{tree_path} is missing {tree_name}")
                    tree_group = handle[tree_name]
                    selected_tree_index = int(row["tree_index"])
                    if args.branch_start == "first_fof":
                        if "FirstHaloInFOFGroup" not in tree_group:
                            raise KeyError(f"{tree_group.name} is missing FirstHaloInFOFGroup")
                        start_index = int(tree_group["FirstHaloInFOFGroup"][selected_tree_index])
                        if start_index < 0:
                            raise ValueError(
                                f"{tree_group.name} selected index {selected_tree_index} has no FirstHaloInFOFGroup"
                            )
                        start_snap = int(tree_group["SnapNum"][start_index])
                        start_group = int(tree_group["SubhaloGrNr"][start_index])
                        if start_snap != int(args.snapshot):
                            raise ValueError(
                                f"{tree_group.name} FirstHaloInFOFGroup index {start_index} has snapshot "
                                f"{start_snap}, expected {args.snapshot}"
                            )
                        if start_group != int(row["group_index"]):
                            raise ValueError(
                                f"{tree_group.name} FirstHaloInFOFGroup index {start_index} has SubhaloGrNr "
                                f"{start_group}, expected {row['group_index']}"
                            )
                    else:
                        start_index = selected_tree_index
                    branch, n_branch, n_zero = _read_branch(
                        tree_group,
                        start_index=start_index,
                        final_snapshot=int(args.snapshot),
                        mass_field=str(args.mass_field),
                        hubble=float(args.hubble),
                    )
                    final_mass = branch[int(args.snapshot)]
                    source_subhalo_id[row_index] = int(row["source_subhalo_id"])
                    source_group_index[row_index] = int(row["group_index"])
                    source_tree_file[row_index] = tree_file
                    source_tree_num[row_index] = int(row["tree_num"])
                    source_tree_index[row_index] = selected_tree_index
                    branch_start_tree_index[row_index] = start_index
                    logm_final_catalog[row_index] = float(row["logM_final"])
                    logm_final_tree[row_index] = float(np.log10(final_mass))
                    branch_length[row_index] = int(n_branch)
                    zero_mass_node_count[row_index] = int(n_zero)

                    for snap, mass in branch.items():
                        column = snap_to_column.get(int(snap))
                        if column is None:
                            continue
                        if np.isfinite(mass) and mass > 0.0:
                            mass_msun[row_index, column] = float(mass)
                            mass_ratio[row_index, column] = float(mass) / final_mass
                            resolved_mask[row_index, column] = True
                    resolved_snap_count[row_index] = int(np.count_nonzero(resolved_mask[row_index]))

    if not np.all(resolved_mask[:, -1]):
        raise RuntimeError("final snapshot must be resolved for every selected THESAN track")
    if not np.all(np.isfinite(mass_ratio)) or np.any(mass_ratio <= 0.0):
        raise RuntimeError("mass_ratio must be finite and positive after unresolved fill")
    if args.output is None:
        z_tag = _tag_from_z(float(z_grid[-1]))
        tree_tag = "allchunks" if tree_file_filter is None else f"file{tree_file_filter}"
        output = PROJECT_ROOT / "data_save/thesan_mah_cache" / (
            f"thesan-dark-1_LHaloTree_{tree_tag}_{z_tag}_n{len(rows)}_smoke.hdf5"
        )
    else:
        output = _resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise FileExistsError(f"output cache already exists: {output}; pass --force to overwrite")

    with h5py.File(output, "w") as handle:
        handle.attrs["schema_version"] = THESAN_MAH_CACHE_SCHEMA_VERSION
        handle.attrs["source_simulation"] = "Thesan-Dark-1"
        handle.attrs["source_tree"] = "LHaloTree"
        handle.attrs["source_tree_file_pattern"] = str(
            (root / "postprocessing/trees/LHaloTree/trees_sf1_190.{tree_file}.hdf5").relative_to(PROJECT_ROOT)
        )
        handle.attrs["source_selection_table"] = str(selected_path.relative_to(PROJECT_ROOT))
        handle.attrs["tree_file_subset"] = "all" if tree_file_filter is None else int(tree_file_filter)
        handle.attrs["source_tree_file_count"] = int(len(requested_tree_files))
        handle.attrs["branch_start"] = str(args.branch_start)
        handle.attrs["publication_complete"] = False
        handle.attrs["snapshot"] = int(args.snapshot)
        handle.attrs["z_final"] = float(z_grid[-1])
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["source_mass_field"] = str(args.mass_field)
        handle.attrs["hubble"] = float(args.hubble)
        handle.attrs["omega_m"] = float(args.omega_m)
        handle.attrs["omega_b"] = float(args.omega_b)
        handle.attrs["unresolved_mass_ratio_fill"] = float(args.unresolved_mass_ratio_fill)
        handle.attrs["n_selected"] = int(len(rows))
        handle.attrs["min_logm_filter"] = np.nan if args.min_logm is None else float(args.min_logm)
        handle.attrs["max_logm_filter"] = np.nan if args.max_logm is None else float(args.max_logm)
        handle.create_dataset("snap_grid", data=snap_grid)
        handle.create_dataset("z_grid", data=z_grid)
        handle.create_dataset("t_gyr_grid", data=t_gyr_grid)
        handle.create_dataset("mass_ratio", data=mass_ratio)
        handle.create_dataset("mass_msun", data=mass_msun)
        handle.create_dataset("resolved_mask", data=resolved_mask)
        handle.create_dataset("source_snapshot", data=np.full(len(rows), int(args.snapshot), dtype=np.int64))
        handle.create_dataset("source_subhalo_id", data=source_subhalo_id)
        handle.create_dataset("source_group_index", data=source_group_index)
        handle.create_dataset("source_tree_file", data=source_tree_file)
        handle.create_dataset("source_tree_num", data=source_tree_num)
        handle.create_dataset("source_tree_index", data=source_tree_index)
        handle.create_dataset("branch_start_tree_index", data=branch_start_tree_index)
        handle.create_dataset("logM_final", data=logm_final_tree)
        handle.create_dataset("logM_final_catalog", data=logm_final_catalog)
        handle.create_dataset("branch_length", data=branch_length)
        handle.create_dataset("zero_mass_node_count", data=zero_mass_node_count)
        handle.create_dataset("resolved_snap_count", data=resolved_snap_count)

    print(f"wrote_thesan_mah_cache={output}", flush=True)
    print(f"n_selected={len(rows)}", flush=True)
    print(f"z_final={float(z_grid[-1]):.9f}", flush=True)
    print(f"resolved_snap_count_min={int(resolved_snap_count.min())}", flush=True)
    print(f"resolved_snap_count_median={float(np.median(resolved_snap_count)):.1f}", flush=True)
    print(f"tracks_with_zero_mass_nodes={int(np.count_nonzero(zero_mass_node_count > 0))}", flush=True)
    return output


def main() -> None:
    args = _parse_args()
    _build_cache(args)


if __name__ == "__main__":
    main()
