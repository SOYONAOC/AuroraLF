#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_THESAN_ROOT = PROJECT_ROOT / "external_data/thesan/thesan-dark-1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/thesan_discovery/selected_halos_corrected.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "outputs/thesan_discovery/selection_summary_corrected.csv"
DEFAULT_MASS_FIELD = "Group_M_Crit200"
DEFAULT_HUBBLE = 0.6774


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select THESAN-dark-1 central halos for MAH extraction. Uses GroupFirstSub "
            "as the snapshot-global central subhalo index and maps it to LHaloTree offsets."
        )
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_THESAN_ROOT))
    parser.add_argument("--snapshots", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", type=str, default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--mass-field", default=DEFAULT_MASS_FIELD)
    parser.add_argument("--hubble", type=float, default=DEFAULT_HUBBLE)
    parser.add_argument("--min-logm", type=float, default=8.5)
    parser.add_argument("--max-logm", type=float, default=11.5)
    parser.add_argument("--bin-width", type=float, default=0.25)
    parser.add_argument("--max-per-bin", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _snapshot_group_files(root: Path, snapshot: int) -> list[Path]:
    group_dir = root / "output" / f"groups_{snapshot:03d}"
    if not group_dir.exists():
        raise FileNotFoundError(f"THESAN group catalog directory not found: {group_dir}")
    files = sorted(
        group_dir.glob(f"fof_subhalo_tab_{snapshot:03d}.*.hdf5"),
        key=lambda path: int(path.stem.split(".")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"no THESAN group catalog chunks found in {group_dir}")
    with h5py.File(files[0], "r") as handle:
        expected_files = int(handle["Header"].attrs["NumFiles"])
    if len(files) != expected_files:
        raise FileNotFoundError(
            f"snapshot {snapshot:03d} group catalog is incomplete: found {len(files)} of {expected_files} chunks"
        )
    return files


def _read_group_catalog(
    files: list[Path],
    *,
    snapshot: int,
    mass_field: str,
    hubble: float,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    if float(hubble) <= 0.0:
        raise ValueError("hubble must be positive")
    group_counts: list[int] = []
    total_groups: int | None = None
    total_subhalos: int | None = None
    redshift: float | None = None
    for path in files:
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            group_counts.append(int(header["Ngroups_ThisFile"]))
            if total_groups is None:
                total_groups = int(header["Ngroups_Total"])
                total_subhalos = int(header["Nsubgroups_Total"])
                redshift = float(header["Redshift"])
            elif (
                total_groups != int(header["Ngroups_Total"])
                or total_subhalos != int(header["Nsubgroups_Total"])
                or not np.isclose(float(redshift), float(header["Redshift"]), rtol=0.0, atol=1.0e-12)
            ):
                raise ValueError(f"inconsistent Header totals/redshift in snapshot {snapshot:03d} group catalog")

    if total_groups is None or total_subhalos is None or redshift is None:
        raise RuntimeError(f"no group catalog chunks were read for snapshot {snapshot:03d}")
    if sum(group_counts) != total_groups:
        raise ValueError(
            f"snapshot {snapshot:03d} group count mismatch: chunks sum to {sum(group_counts)}, "
            f"Header says {total_groups}"
        )

    group_offsets = np.cumsum([0] + group_counts[:-1])
    first_subhalo = np.empty(total_groups, dtype=np.int64)
    group_mass_code = np.empty(total_groups, dtype=float)
    for file_index, path in enumerate(files):
        offset = int(group_offsets[file_index])
        count = int(group_counts[file_index])
        with h5py.File(path, "r") as handle:
            if f"Group/{mass_field}" not in handle:
                raise KeyError(f"{path} is missing Group/{mass_field}")
            first_subhalo[offset : offset + count] = np.asarray(handle["Group/GroupFirstSub"], dtype=np.int64)
            group_mass_code[offset : offset + count] = np.asarray(handle[f"Group/{mass_field}"], dtype=float)

    occupied = first_subhalo >= 0
    if np.any(first_subhalo[occupied] >= int(total_subhalos)):
        raise ValueError(
            f"snapshot {snapshot:03d} GroupFirstSub contains values outside Nsubgroups_Total={total_subhalos}"
        )
    group_mass_msun = group_mass_code * 1.0e10 / float(hubble)
    return float(redshift), int(total_subhalos), first_subhalo, group_mass_msun


def _read_offsets(root: Path, snapshot: int, expected_subhalo_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset_path = root / "postprocessing" / "offsets" / f"offsets_{snapshot:03d}.hdf5"
    if not offset_path.exists():
        raise FileNotFoundError(f"THESAN offsets file not found: {offset_path}")
    with h5py.File(offset_path, "r") as handle:
        required = {
            "Subhalo/LHaloTree/File",
            "Subhalo/LHaloTree/Num",
            "Subhalo/LHaloTree/Index",
        }
        missing = sorted(name for name in required if name not in handle)
        if missing:
            raise KeyError(f"{offset_path} is missing required datasets: {missing}")
        tree_file = np.asarray(handle["Subhalo/LHaloTree/File"], dtype=np.int64)
        tree_num = np.asarray(handle["Subhalo/LHaloTree/Num"], dtype=np.int64)
        tree_index = np.asarray(handle["Subhalo/LHaloTree/Index"], dtype=np.int64)
    if tree_file.shape != (expected_subhalo_count,):
        raise ValueError(
            f"{offset_path} tree offset length {tree_file.size} does not match Nsubgroups_Total={expected_subhalo_count}"
        )
    if tree_num.shape != tree_file.shape or tree_index.shape != tree_file.shape:
        raise ValueError(f"{offset_path} LHaloTree offset arrays have inconsistent shapes")
    return tree_file, tree_num, tree_index


def _select_snapshot(
    root: Path,
    *,
    snapshot: int,
    mass_field: str,
    hubble: float,
    bins: np.ndarray,
    max_per_bin: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    files = _snapshot_group_files(root, snapshot)
    redshift, total_subhalos, first_subhalo, group_mass_msun = _read_group_catalog(
        files,
        snapshot=snapshot,
        mass_field=mass_field,
        hubble=hubble,
    )
    tree_file, tree_num, tree_index = _read_offsets(root, snapshot, expected_subhalo_count=total_subhalos)

    central_group = np.flatnonzero(
        (first_subhalo >= 0) & np.isfinite(group_mass_msun) & (group_mass_msun > 0.0)
    )
    central_subhalo = first_subhalo[central_group]
    if np.any(central_subhalo < 0):
        raise RuntimeError("central_subhalo unexpectedly contains negative values after filtering")
    logm = np.log10(group_mass_msun[central_group])

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = np.flatnonzero((logm >= float(lo)) & (logm < float(hi)))
        available = int(in_bin.size)
        if available > int(max_per_bin):
            chosen_local = np.sort(rng.choice(in_bin, size=int(max_per_bin), replace=False))
        else:
            chosen_local = in_bin
        status = "ok"
        if available == 0:
            status = "empty"
        elif available < int(max_per_bin):
            status = "insufficient"
        summary.append(
            {
                "snapshot": int(snapshot),
                "redshift": float(redshift),
                "logM_lo": float(lo),
                "logM_hi": float(hi),
                "available": available,
                "selected": int(chosen_local.size),
                "status": status,
            }
        )
        for local_index in chosen_local:
            group_index = int(central_group[int(local_index)])
            subhalo_index = int(central_subhalo[int(local_index)])
            rows.append(
                {
                    "snapshot": int(snapshot),
                    "redshift": float(redshift),
                    "group_index": group_index,
                    "source_subhalo_id": subhalo_index,
                    "logM_final": float(logm[int(local_index)]),
                    "tree_file": int(tree_file[subhalo_index]),
                    "tree_num": int(tree_num[subhalo_index]),
                    "tree_index": int(tree_index[subhalo_index]),
                }
            )
    return rows, summary


def main() -> None:
    args = _parse_args()
    root = _resolve_project_path(args.root)
    output = _resolve_project_path(args.output)
    summary_output = _resolve_project_path(args.summary_output)
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output}; pass --force to overwrite")
    if summary_output.exists() and not args.force:
        raise FileExistsError(f"summary output already exists: {summary_output}; pass --force to overwrite")
    if float(args.bin_width) <= 0.0:
        raise ValueError("--bin-width must be positive")
    if int(args.max_per_bin) <= 0:
        raise ValueError("--max-per-bin must be positive")
    bins = np.arange(float(args.min_logm), float(args.max_logm) + 0.5 * float(args.bin_width), float(args.bin_width))
    if bins.size < 2:
        raise ValueError("mass bin configuration produced fewer than two bin edges")

    rng = np.random.default_rng(int(args.random_seed))
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for snapshot in args.snapshots:
        selected_rows, selected_summary = _select_snapshot(
            root,
            snapshot=int(snapshot),
            mass_field=str(args.mass_field),
            hubble=float(args.hubble),
            bins=bins,
            max_per_bin=int(args.max_per_bin),
            rng=rng,
        )
        rows.extend(selected_rows)
        summary_rows.extend(selected_summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snapshot",
        "redshift",
        "group_index",
        "source_subhalo_id",
        "logM_final",
        "tree_file",
        "tree_num",
        "tree_index",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["snapshot", "redshift", "logM_lo", "logM_hi", "available", "selected", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote_selected_halos={output}", flush=True)
    print(f"wrote_selection_summary={summary_output}", flush=True)
    print(f"selected_rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
