#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION = "thesan-dark-1"
DEFAULT_GLOBUS_COLLECTION_ID = "784b5949-5dd6-41c9-8de6-2cae0844501b"
DEFAULT_SOURCE_ROOT = "/Thesan-Dark-1"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "external_data" / "thesan" / DEFAULT_SIMULATION
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_save" / "thesan_mah_cache" / "download_manifests"
SMOKE_TARGET_REDSHIFT = 11.98
SAMPLE_TARGET_REDSHIFTS = (6.0, 8.0, 10.0, 12.0)
SAMPLE_SELECTION_POLICY = "central_only; logM=8.5-11.5; dlogM=0.25; max_per_bin=50; min_per_bin=20"
TREE_MASS_FIELD_CANDIDATES = (
    "Group_M_Crit200",
    "GroupMass",
    "SubhaloMass",
    "SubhaloMassType",
    "SubhaloMassInRadType",
    "Mass",
)


def _tag_from_float(value: float, *, precision: int = 3) -> str:
    return f"{float(value):.{precision}f}".replace(".", "p").replace("-", "m")


def _resolve_path(path_value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _format_snapshot(snapshot: int | str) -> str:
    if isinstance(snapshot, str):
        return snapshot
    return f"{int(snapshot):03d}"


def _source_path(source_root: str, local_path: str) -> str:
    root = str(source_root).rstrip("/")
    return f"{root}/{local_path.lstrip('/')}"


def _nearest_snapshot(target_redshift: float, snapshot_redshift: dict[int, float]) -> tuple[int, float]:
    if not snapshot_redshift:
        raise ValueError("snapshot_redshift must not be empty")
    snapshot = min(snapshot_redshift, key=lambda snap: abs(float(snapshot_redshift[snap]) - float(target_redshift)))
    return int(snapshot), float(snapshot_redshift[snapshot])


def _load_snapshot_redshift_file(path: Path) -> dict[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"snapshot redshift file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.items()
        elif isinstance(payload, list):
            items = ((row["snapshot"], row["redshift"]) for row in payload)
        else:
            raise ValueError("snapshot redshift JSON must be an object or list of rows")
        return {int(snapshot): float(redshift) for snapshot, redshift in items}

    mapping: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot", "redshift"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("snapshot redshift CSV must contain snapshot and redshift columns")
        for row in reader:
            mapping[int(row["snapshot"])] = float(row["redshift"])
    if not mapping:
        raise ValueError(f"snapshot redshift file contains no rows: {path}")
    return mapping


def _manifest_row(
    *,
    stage: str,
    product: str,
    local_path: str,
    source_root: str,
    required: bool,
    target_redshift: float | str,
    snapshot: int | str,
    snapshot_redshift: float | str,
    notes: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "product": product,
        "required": bool(required),
        "target_redshift": target_redshift,
        "snapshot": snapshot,
        "snapshot_redshift": snapshot_redshift,
        "source_path": _source_path(source_root, local_path),
        "local_path": local_path,
        "notes": notes,
    }


def _smoke_rows(
    *,
    snapshot_redshift: dict[int, float] | None,
    source_root: str,
) -> list[dict[str, Any]]:
    if snapshot_redshift:
        snapshot, redshift = _nearest_snapshot(SMOKE_TARGET_REDSHIFT, snapshot_redshift)
    else:
        snapshot, redshift = "NNN", ""
    snap = _format_snapshot(snapshot)
    notes = "Step 1 smoke test; <=3 GB; verify groupcat, offset, and one tree chunk schema"
    return [
        _manifest_row(
            stage="smoke",
            product="offset",
            local_path=f"postprocessing/offsets/offsets_{snap}.hdf5",
            source_root=source_root,
            required=True,
            target_redshift=SMOKE_TARGET_REDSHIFT,
            snapshot=snapshot,
            snapshot_redshift=redshift,
            notes=notes,
        ),
        _manifest_row(
            stage="smoke",
            product="groupcat_chunk",
            local_path=f"output/groups_{snap}/fof_subhalo_tab_{snap}.0.hdf5",
            source_root=source_root,
            required=True,
            target_redshift=SMOKE_TARGET_REDSHIFT,
            snapshot=snapshot,
            snapshot_redshift=redshift,
            notes=notes,
        ),
        _manifest_row(
            stage="smoke",
            product="sub_desc",
            local_path="postprocessing/trees/LHaloTree/sub_desc_sf1_080",
            source_root=source_root,
            required=True,
            target_redshift=SMOKE_TARGET_REDSHIFT,
            snapshot=snapshot,
            snapshot_redshift=redshift,
            notes=notes,
        ),
    ]


def _sample_rows(
    *,
    snapshot_redshift: dict[int, float] | None,
    source_root: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected: list[tuple[float, int | str, float | str]] = []
    if snapshot_redshift:
        seen: set[int] = set()
        for target in SAMPLE_TARGET_REDSHIFTS:
            snapshot, redshift = _nearest_snapshot(target, snapshot_redshift)
            if snapshot in seen:
                raise ValueError(f"multiple sample target redshifts map to snapshot {snapshot}")
            seen.add(snapshot)
            selected.append((target, snapshot, redshift))
    else:
        for target in SAMPLE_TARGET_REDSHIFTS:
            selected.append((target, f"NNN_for_z{_tag_from_float(target)}", ""))

    for target, snapshot, redshift in selected:
        snap = _format_snapshot(snapshot)
        notes = f"Step 2 usable sample; {SAMPLE_SELECTION_POLICY}; download full groupcat for selection"
        rows.append(
            _manifest_row(
                stage="sample",
                product="offset",
                local_path=f"postprocessing/offsets/offsets_{snap}.hdf5",
                source_root=source_root,
                required=True,
                target_redshift=target,
                snapshot=snapshot,
                snapshot_redshift=redshift,
                notes=notes,
            )
        )
        rows.append(
            _manifest_row(
                stage="sample",
                product="groupcat_all_chunks",
                local_path=f"output/groups_{snap}/fof_subhalo_tab_{snap}.*.hdf5",
                source_root=source_root,
                required=True,
                target_redshift=target,
                snapshot=snapshot,
                snapshot_redshift=redshift,
                notes=notes,
            )
        )

    rows.append(
        _manifest_row(
            stage="sample",
            product="tree_chunk_selection",
            local_path="postprocessing/trees/LHaloTree/trees_sf1_190.C.hdf5",
            source_root=source_root,
            required=True,
            target_redshift="6,8,10,12",
            snapshot="selected_by_offsets",
            snapshot_redshift="",
            notes=(
                f"Step 2 usable sample; {SAMPLE_SELECTION_POLICY}; expand C to only tree chunks "
                "needed by selected central subhalos"
            ),
        )
    )
    return rows


def _publication_rows(
    *,
    snapshot_redshift: dict[int, float] | None,
    source_root: str,
) -> list[dict[str, Any]]:
    snapshots = sorted(snapshot_redshift) if snapshot_redshift else list(range(80))
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        snap = _format_snapshot(snapshot)
        redshift: float | str = float(snapshot_redshift[snapshot]) if snapshot_redshift else ""
        notes = "Step 3 publication-grade MAH-only dataset; all offsets and group catalogs"
        rows.append(
            _manifest_row(
                stage="publication",
                product="offset",
                local_path=f"postprocessing/offsets/offsets_{snap}.hdf5",
                source_root=source_root,
                required=True,
                target_redshift="all",
                snapshot=snapshot,
                snapshot_redshift=redshift,
                notes=notes,
            )
        )
        rows.append(
            _manifest_row(
                stage="publication",
                product="groupcat_all_chunks",
                local_path=f"output/groups_{snap}/fof_subhalo_tab_{snap}.*.hdf5",
                source_root=source_root,
                required=True,
                target_redshift="all",
                snapshot=snapshot,
                snapshot_redshift=redshift,
                notes=notes,
            )
        )

    rows.append(
        _manifest_row(
            stage="publication",
            product="tree_all_chunks",
            local_path="postprocessing/trees/LHaloTree/trees_sf1_190.*.hdf5",
            source_root=source_root,
            required=True,
            target_redshift="all",
            snapshot="all",
            snapshot_redshift="",
            notes="Step 3 publication-grade MAH-only dataset; all merger tree chunks",
        )
    )
    rows.append(
        _manifest_row(
            stage="publication",
            product="cross_link",
            local_path="postprocessing/cross_link.hdf5",
            source_root=source_root,
            required=False,
            target_redshift="all",
            snapshot="all",
            snapshot_redshift="",
            notes="Optional but recommended for later dark-to-hydro matching",
        )
    )
    return rows


def _build_stage_rows(
    *,
    stage: str,
    snapshot_redshift: dict[int, float] | None,
    source_root: str,
    local_root: Path,
) -> list[dict[str, Any]]:
    stage = str(stage).strip().lower()
    if stage == "smoke":
        return _smoke_rows(snapshot_redshift=snapshot_redshift, source_root=source_root)
    if stage == "sample":
        return _sample_rows(snapshot_redshift=snapshot_redshift, source_root=source_root)
    if stage == "publication":
        return _publication_rows(snapshot_redshift=snapshot_redshift, source_root=source_root)
    if stage == "all":
        rows: list[dict[str, Any]] = []
        for substage in ("smoke", "sample", "publication"):
            rows.extend(
                _build_stage_rows(
                    stage=substage,
                    snapshot_redshift=snapshot_redshift,
                    source_root=source_root,
                    local_root=local_root,
                )
            )
        return rows
    raise ValueError("stage must be one of: smoke, sample, publication, all")


def _write_manifest_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    stage: str,
    source_root: str,
    local_root: Path,
    write_globus_batch: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"thesan-dark-1_{stage}_download_manifest"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    fieldnames = [
        "stage",
        "product",
        "required",
        "target_redshift",
        "snapshot",
        "snapshot_redshift",
        "source_path",
        "local_path",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "simulation": DEFAULT_SIMULATION,
        "globus_collection_id": DEFAULT_GLOBUS_COLLECTION_ID,
        "source_root": source_root,
        "local_root": str(local_root),
        "stage": stage,
        "row_count": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# THESAN-dark-1 {stage} download manifest",
        "",
        f"- Globus collection id: `{DEFAULT_GLOBUS_COLLECTION_ID}`",
        f"- Source root: `{source_root}`",
        f"- Local root: `{local_root}`",
        f"- Rows: `{len(rows)}`",
        "",
        "| stage | product | required | target z | snapshot | source path | local path |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['product']} | {row['required']} | {row['target_redshift']} | "
            f"{row['snapshot']} | `{row['source_path']}` | `{row['local_path']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    paths = {"csv": csv_path, "json": json_path, "markdown": md_path}
    if write_globus_batch:
        batch_path = output_dir / f"{stem}.globus_batch.tsv"
        with batch_path.open("w", encoding="utf-8") as handle:
            handle.write("# source_path\tdestination_path\n")
            handle.write("# Expand wildcard rows against a real Globus listing before executing transfer.\n")
            for row in rows:
                destination = str(local_root / str(row["local_path"]))
                handle.write(f"{row['source_path']}\t{destination}\n")
        paths["globus_batch"] = batch_path
    return paths


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required THESAN file not found: {path}")
    if not path.is_file():
        raise ValueError(f"required THESAN path is not a file: {path}")


def _dataset_leaf_names(handle: h5py.File) -> list[str]:
    names: list[str] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    handle.visititems(visitor)
    return names


def _count_offset_mapping_datasets(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        dataset_names = _dataset_leaf_names(handle)
    mapping_names = {
        "treefile",
        "treefilenum",
        "treechunk",
        "treenumber",
        "treeindex",
        "treeoffset",
        "subhalotreefile",
        "subhalotreeindex",
    }
    count = 0
    for name in dataset_names:
        lower_name = name.lower()
        leaf = name.split("/")[-1].lower()
        if (
            leaf in mapping_names
            or ("tree" in leaf and ("file" in leaf or "index" in leaf or "number" in leaf))
            or ("lhalotree" in lower_name and leaf in {"file", "num", "index"})
        ):
            count += 1
    return count


def _find_tree_mass_field(path: Path) -> tuple[bool, str]:
    with h5py.File(path, "r") as handle:
        found_snapnum = False
        found_mass = ""

        def visitor(name: str, obj: h5py.Dataset) -> None:
            nonlocal found_snapnum, found_mass
            if not isinstance(obj, h5py.Dataset):
                return
            leaf = name.split("/")[-1]
            if leaf == "SnapNum":
                found_snapnum = True
            if found_mass == "" and leaf in TREE_MASS_FIELD_CANDIDATES:
                found_mass = leaf

        handle.visititems(visitor)
    return found_snapnum, found_mass


def _read_sub_desc_summary(path: Path) -> dict[str, Any]:
    _require_file(path)
    size = path.stat().st_size
    if size < 4 or size % 4 != 0:
        raise ValueError(f"sub_desc file must be int32 binary with at least one entry: {path}")
    with path.open("rb") as handle:
        first = int.from_bytes(handle.read(4), byteorder="little", signed=True)
    count = int(size // 4)
    if first < 0:
        raise ValueError(f"sub_desc first int32 entry must be non-negative; got {first}: {path}")
    return {
        "sub_desc_path": str(path),
        "sub_desc_dtype": "int32",
        "sub_desc_file_size_bytes": int(size),
        "sub_desc_entry_count": int(first),
        "sub_desc_int32_count": count,
    }


def _validate_smoke_files(*, root: Path, snapshot: int, tree_chunk: int) -> dict[str, Any]:
    snap = f"{int(snapshot):03d}"
    del tree_chunk
    groupcat_path = root / "output" / f"groups_{snap}" / f"fof_subhalo_tab_{snap}.0.hdf5"
    offset_path = root / "postprocessing" / "offsets" / f"offsets_{snap}.hdf5"
    sub_desc_path = root / "postprocessing" / "trees" / "LHaloTree" / "sub_desc_sf1_080"
    for path in (groupcat_path, offset_path, sub_desc_path):
        _require_file(path)

    with h5py.File(groupcat_path, "r") as handle:
        has_header = "Header" in handle
        has_group = "Group" in handle
        has_subhalo = "Subhalo" in handle
    if not has_header or not has_group or not has_subhalo:
        raise KeyError(f"group catalog is missing Header/Group/Subhalo: {groupcat_path}")

    mapping_count = _count_offset_mapping_datasets(offset_path)
    if mapping_count == 0:
        raise KeyError(f"offset file has no tree-mapping datasets: {offset_path}")
    sub_desc_summary = _read_sub_desc_summary(sub_desc_path)

    report = {
        "root": str(root),
        "snapshot": int(snapshot),
        "groupcat_path": str(groupcat_path),
        "offset_path": str(offset_path),
        "groupcat_has_header": has_header,
        "groupcat_has_group": has_group,
        "groupcat_has_subhalo": has_subhalo,
        "offset_mapping_dataset_count": mapping_count,
    }
    report.update(sub_desc_summary)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare reproducible THESAN-dark-1 download manifests for AuroraLF MAH work. "
            "This script writes manifests and validates existing smoke-test files; it does not "
            "silently download data or synthesize missing files."
        )
    )
    parser.add_argument("--stage", choices=("smoke", "sample", "publication", "all"), default="all")
    parser.add_argument("--snapshot-redshift-file", type=str, default=None)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--local-root", type=str, default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-globus-batch", action="store_true")
    parser.add_argument("--validate-smoke", action="store_true")
    parser.add_argument("--smoke-snapshot", type=int, default=None)
    parser.add_argument("--smoke-tree-chunk", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    local_root = _resolve_path(args.local_root)
    output_dir = _resolve_path(args.output_dir)
    snapshot_redshift = (
        _load_snapshot_redshift_file(_resolve_path(args.snapshot_redshift_file))
        if args.snapshot_redshift_file is not None
        else None
    )
    rows = _build_stage_rows(
        stage=str(args.stage),
        snapshot_redshift=snapshot_redshift,
        source_root=str(args.source_root),
        local_root=local_root,
    )
    paths = _write_manifest_outputs(
        rows=rows,
        output_dir=output_dir,
        stage=str(args.stage),
        source_root=str(args.source_root),
        local_root=local_root,
        write_globus_batch=bool(args.write_globus_batch),
    )
    for key, path in paths.items():
        print(f"saved_{key}={path}")

    if args.validate_smoke:
        if args.smoke_snapshot is None:
            if snapshot_redshift is None:
                raise ValueError("--validate-smoke requires --smoke-snapshot or --snapshot-redshift-file")
            smoke_snapshot, _ = _nearest_snapshot(SMOKE_TARGET_REDSHIFT, snapshot_redshift)
        else:
            smoke_snapshot = int(args.smoke_snapshot)
        report = _validate_smoke_files(
            root=local_root,
            snapshot=smoke_snapshot,
            tree_chunk=int(args.smoke_tree_chunk),
        )
        report_path = output_dir / f"thesan-dark-1_smoke_snapshot_{smoke_snapshot:03d}_validation.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"saved_validation={report_path}")


if __name__ == "__main__":
    main()
