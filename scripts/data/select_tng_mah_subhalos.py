#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

from auroralf.constants import PLANCK15_H

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION = "TNG100-1-Dark"
DEFAULT_API_BASE = "https://www.tng-project.org/api"
DEFAULT_HUBBLE = PLANCK15_H
DEFAULT_TARGET_REDSHIFTS = (6.0, 8.0, 10.0, 12.5)
DEFAULT_LOGM_MIN = 9.0
DEFAULT_LOGM_MAX = 13.0
DEFAULT_LOGM_BIN_WIDTH = 0.25
DEFAULT_MAX_PER_BIN = 200
DEFAULT_RANDOM_SEED = 42
EXPECTED_MPB_BYTES = 65536
DEFAULT_BUILD_MISSING_MASS_RATIO_FLOOR = 1.0e-6


@dataclass(frozen=True)
class SnapshotSelection:
    target_z: float
    snapshot: int
    snapshot_z: float
    selected_ids: np.ndarray
    bin_rows: list[dict[str, Any]]


def _tag_from_float(value: float, *, precision: int = 3) -> str:
    return f"{float(value):.{precision}f}".replace(".", "p").replace("-", "m")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select central TNG subhalos for AuroraLF TNG MAH cache construction. "
            "The script downloads compact groupcat field files, bins Group_M_Crit200 "
            "in Msun, writes per-bin and per-snapshot subhalo ID lists, and emits a "
            "manifest plus cache-build commands. It does not download MPB trees."
        )
    )
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--target-redshifts", nargs="+", type=float, default=list(DEFAULT_TARGET_REDSHIFTS))
    parser.add_argument("--snapshots", nargs="+", type=int, default=None)
    parser.add_argument("--logM-min", type=float, default=DEFAULT_LOGM_MIN)
    parser.add_argument("--logM-max", type=float, default=DEFAULT_LOGM_MAX)
    parser.add_argument("--logM-bin-width", type=float, default=DEFAULT_LOGM_BIN_WIDTH)
    parser.add_argument("--max-per-bin", type=int, default=DEFAULT_MAX_PER_BIN)
    parser.add_argument("--min-per-bin", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--build-download-workers",
        type=int,
        default=8,
        help="download-workers value written into generated build_tng_mah_cache.py commands.",
    )
    parser.add_argument(
        "--build-download-retries",
        type=int,
        default=3,
        help="download-retries value written into generated build_tng_mah_cache.py commands.",
    )
    parser.add_argument(
        "--build-drop-invalid-mpb",
        action="store_true",
        help="Write --drop-invalid-mpb into generated build_tng_mah_cache.py commands.",
    )
    parser.add_argument(
        "--build-snapshot-grid",
        choices=("common", "union"),
        default="common",
        help="snapshot-grid value written into generated build_tng_mah_cache.py commands.",
    )
    parser.add_argument(
        "--build-missing-mass-ratio-floor",
        type=float,
        default=DEFAULT_BUILD_MISSING_MASS_RATIO_FLOOR,
        help="missing-mass-ratio-floor value written into generated build_tng_mah_cache.py commands.",
    )
    parser.add_argument("--hubble", type=float, default=DEFAULT_HUBBLE)
    parser.add_argument("--groupcat-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def _resolve_path(path_value: str | None, default: Path) -> Path:
    path = Path(path_value).expanduser() if path_value is not None else default
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    return path.resolve()


def _require_api_key() -> str:
    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError("TNG_API_KEY must be set to download TNG groupcat field files")
    return api_key


def _request_json(url: str, api_key: str) -> object:
    response = requests.get(url, headers={"api-key": api_key}, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"TNG API request failed with status {response.status_code}: {url}")
    return response.json()


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
    if not mapping:
        raise RuntimeError("TNG snapshots endpoint returned no snapshots")
    return mapping


def _nearest_snapshot(target_z: float, snapshot_redshift: dict[int, float]) -> tuple[int, float]:
    snapshot = min(snapshot_redshift, key=lambda snap: abs(snapshot_redshift[snap] - float(target_z)))
    return int(snapshot), float(snapshot_redshift[snapshot])


def _field_filename(snapshot: int, field: str) -> str:
    return f"fof_subhalo_tab_{int(snapshot):03d}.Group.{field}.hdf5"


def _download_groupcat_field(
    *,
    api_base: str,
    simulation: str,
    snapshot: int,
    field: str,
    destination: Path,
    api_key: str,
    force: bool,
) -> None:
    if destination.exists() and not force:
        print(f"using_existing_groupcat_field={destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{api_base.rstrip('/')}/{simulation}/files/groupcat-{int(snapshot)}/?Group={field}"
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, headers={"api-key": api_key}, timeout=120, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(f"TNG groupcat field download failed with status {response.status_code}: {url}")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)
    print(f"downloaded_groupcat_field={destination}", flush=True)


def _read_groupcat_field(path: Path, field: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"groupcat field file not found: {path}")
    with h5py.File(path, "r") as handle:
        if f"Group/{field}" in handle:
            return np.asarray(handle[f"Group/{field}"])
        if field in handle:
            return np.asarray(handle[field])
        matches: list[str] = []

        def visitor(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == field:
                matches.append(name)

        handle.visititems(visitor)
        if len(matches) == 1:
            return np.asarray(handle[matches[0]])
        if len(matches) > 1:
            raise KeyError(f"{path} contains multiple datasets named {field!r}: {matches}")
        raise KeyError(f"{path} is missing groupcat field {field!r}")


def _mass_bin_edges(logm_min: float, logm_max: float, bin_width: float) -> np.ndarray:
    if float(logm_max) <= float(logm_min):
        raise ValueError("logM_max must be larger than logM_min")
    if float(bin_width) <= 0.0:
        raise ValueError("logM-bin-width must be positive")
    n_bins = int(round((float(logm_max) - float(logm_min)) / float(bin_width)))
    edges = float(logm_min) + np.arange(n_bins + 1, dtype=float) * float(bin_width)
    if not np.isclose(edges[-1], float(logm_max), rtol=0.0, atol=1.0e-8):
        raise ValueError("logM range must be an integer multiple of logM-bin-width")
    return edges


def _selection_status(selected_count: int, available_count: int, min_per_bin: int, max_per_bin: int) -> str:
    if available_count == 0:
        return "empty"
    if selected_count < int(min_per_bin):
        return "insufficient"
    if selected_count < int(max_per_bin):
        return "partial"
    return "ok"


def _write_id_file(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for subhalo_id in np.asarray(ids, dtype=np.int64):
            handle.write(f"{int(subhalo_id)}\n")


def _select_snapshot_subhalos(
    *,
    target_z: float,
    snapshot: int,
    snapshot_z: float,
    group_first_sub_path: Path,
    group_mcrit200_path: Path,
    output_dir: Path,
    simulation: str,
    logm_edges: np.ndarray,
    max_per_bin: int,
    min_per_bin: int,
    hubble: float,
    rng: np.random.Generator,
) -> SnapshotSelection:
    first_sub = np.asarray(_read_groupcat_field(group_first_sub_path, "GroupFirstSub"), dtype=np.int64)
    mass_code = np.asarray(_read_groupcat_field(group_mcrit200_path, "Group_M_Crit200"), dtype=float)
    if first_sub.ndim != 1 or mass_code.ndim != 1 or first_sub.size != mass_code.size:
        raise ValueError(
            f"GroupFirstSub and Group_M_Crit200 must be matching 1D arrays for snapshot {int(snapshot)}"
        )
    if float(hubble) <= 0.0:
        raise ValueError("hubble must be positive")
    mass_msun = mass_code * 1.0e10 / float(hubble)
    valid = (first_sub >= 0) & np.isfinite(mass_msun) & (mass_msun > 0.0)
    logm = np.full(first_sub.shape, np.nan, dtype=float)
    logm[valid] = np.log10(mass_msun[valid])

    ids_dir = output_dir / "ids"
    selected_all: list[np.ndarray] = []
    bin_rows: list[dict[str, Any]] = []
    z_tag = _tag_from_float(snapshot_z)
    for bin_index, (lo, hi) in enumerate(zip(logm_edges[:-1], logm_edges[1:], strict=True)):
        in_bin = valid & (logm >= float(lo)) & (logm < float(hi))
        candidates = first_sub[in_bin]
        available_count = int(candidates.size)
        if available_count > int(max_per_bin):
            selected = np.sort(rng.choice(candidates, size=int(max_per_bin), replace=False).astype(np.int64))
        else:
            selected = np.sort(candidates.astype(np.int64))
        selected_count = int(selected.size)
        if selected_count:
            selected_all.append(selected)
        logm_tag = f"logM{_tag_from_float(lo, precision=2)}_{_tag_from_float(hi, precision=2)}"
        id_path = ids_dir / f"{simulation}_snap{int(snapshot):03d}_z{z_tag}_{logm_tag}_n{selected_count}.txt"
        _write_id_file(id_path, selected)
        status = _selection_status(
            selected_count=selected_count,
            available_count=available_count,
            min_per_bin=int(min_per_bin),
            max_per_bin=int(max_per_bin),
        )
        bin_rows.append(
            {
                "target_z": float(target_z),
                "snapshot": int(snapshot),
                "snapshot_z": float(snapshot_z),
                "logM_low": float(lo),
                "logM_high": float(hi),
                "bin_index": int(bin_index),
                "available_count": available_count,
                "selected_count": selected_count,
                "status": status,
                "id_file": str(id_path),
                "expected_mpb_bytes": int(selected_count * EXPECTED_MPB_BYTES),
            }
        )

    selected_ids = np.unique(np.concatenate(selected_all).astype(np.int64)) if selected_all else np.array([], dtype=np.int64)
    all_id_path = ids_dir / f"{simulation}_snap{int(snapshot):03d}_z{z_tag}_all_selected_n{selected_ids.size}.txt"
    _write_id_file(all_id_path, selected_ids)
    for row in bin_rows:
        row["all_id_file"] = str(all_id_path)
    return SnapshotSelection(
        target_z=float(target_z),
        snapshot=int(snapshot),
        snapshot_z=float(snapshot_z),
        selected_ids=selected_ids,
        bin_rows=bin_rows,
    )


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_z",
        "snapshot",
        "snapshot_z",
        "logM_low",
        "logM_high",
        "bin_index",
        "available_count",
        "selected_count",
        "status",
        "id_file",
        "all_id_file",
        "expected_mpb_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_build_commands(
    path: Path,
    *,
    selections: list[SnapshotSelection],
    simulation: str,
    output_dir: Path,
    raw_dir: Path,
    download_workers: int,
    download_retries: int,
    drop_invalid_mpb: bool,
    snapshot_grid: str,
    missing_mass_ratio_floor: float,
) -> None:
    lines = [
        "# Commands generated by scripts/data/select_tng_mah_subhalos.py",
        "# Run from the AuroraLF repository root.",
        "set -euo pipefail",
        'source "$HOME/.config/auroralf/tng.env"',
        "",
    ]
    for selection in selections:
        z_tag = _tag_from_float(selection.snapshot_z)
        id_file = output_dir / "ids" / (
            f"{simulation}_snap{selection.snapshot:03d}_z{z_tag}_all_selected_n{selection.selected_ids.size}.txt"
        )
        cache_output = (
            PROJECT_ROOT
            / "data_save"
            / "tng_mah_cache"
            / f"{simulation}_sublink_mpb_z{z_tag}_n{selection.selected_ids.size}.hdf5"
        )
        command = (
            "PYTHONPATH=. .venv/bin/python scripts/data/build_tng_mah_cache.py "
            f"--simulation {simulation} "
            f"--snapshot {selection.snapshot} "
            f"--z-final {selection.snapshot_z:.12g} "
            f"--subhalo-id-file {id_file} "
            f"--raw-dir {raw_dir} "
            f"--download-workers {int(download_workers)} "
            f"--download-retries {int(download_retries)} "
            f"--snapshot-grid {snapshot_grid} "
            f"--missing-mass-ratio-floor {float(missing_mass_ratio_floor):.12g} "
        )
        if drop_invalid_mpb:
            command += "--drop-invalid-mpb "
        command += f"--output {cache_output}"
        lines.append(command)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, *, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TNG MAH Subhalo Selection",
        "",
        f"- simulation: `{payload['simulation']}`",
        f"- target_redshifts: `{payload['target_redshifts']}`",
        f"- snapshots: `{payload['snapshots']}`",
        f"- logM range: `{payload['logM_min']} - {payload['logM_max']}` dex",
        f"- bin width: `{payload['logM_bin_width']}` dex",
        f"- max_per_bin: `{payload['max_per_bin']}`",
        f"- build_download_workers: `{payload['build_download_workers']}`",
        f"- build_download_retries: `{payload['build_download_retries']}`",
        f"- build_drop_invalid_mpb: `{payload['build_drop_invalid_mpb']}`",
        f"- build_snapshot_grid: `{payload['build_snapshot_grid']}`",
        f"- build_missing_mass_ratio_floor: `{payload['build_missing_mass_ratio_floor']}`",
        f"- random_seed: `{payload['random_seed']}`",
        f"- total_selected: `{payload['total_selected']}`",
        f"- expected_mpb_mib: `{payload['expected_mpb_bytes'] / 1024**2:.2f}`",
        "",
        "| snapshot | z | selected | ok bins | partial | insufficient | empty | expected MPB MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for snapshot_info in payload["per_snapshot"]:
        lines.append(
            f"| {snapshot_info['snapshot']} | {snapshot_info['snapshot_z']:.6f} | "
            f"{snapshot_info['selected_count']} | {snapshot_info['ok_bins']} | "
            f"{snapshot_info['partial_bins']} | {snapshot_info['insufficient_bins']} | "
            f"{snapshot_info['empty_bins']} | {snapshot_info['expected_mpb_bytes'] / 1024**2:.2f} |"
        )
    lines.extend(
        [
            "",
            "Bins marked `empty` or `insufficient` are recorded explicitly; they are not replaced by McBride samples.",
            "",
            "Detailed bin rows are in `selected_subhalos_manifest.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if int(args.max_per_bin) < 1:
        raise ValueError("max-per-bin must be positive")
    if int(args.min_per_bin) < 1:
        raise ValueError("min-per-bin must be positive")
    if int(args.min_per_bin) > int(args.max_per_bin):
        raise ValueError("min-per-bin must be <= max-per-bin")
    if int(args.build_download_workers) < 1:
        raise ValueError("build-download-workers must be positive")
    if int(args.build_download_retries) < 0:
        raise ValueError("build-download-retries must be non-negative")
    if float(args.build_missing_mass_ratio_floor) <= 0.0 or not np.isfinite(float(args.build_missing_mass_ratio_floor)):
        raise ValueError("build-missing-mass-ratio-floor must be finite and positive")

    api_key = _require_api_key()
    groupcat_dir = _resolve_path(
        args.groupcat_dir,
        PROJECT_ROOT / "external_data" / "tng" / str(args.simulation) / "groupcat_fields",
    )
    output_dir = _resolve_path(
        args.output_dir,
        PROJECT_ROOT
        / "data_save"
        / "tng_mah_cache"
        / "selection"
        / (
            f"{args.simulation}_"
            f"logM{_tag_from_float(args.logM_min, precision=2)}_{_tag_from_float(args.logM_max, precision=2)}_"
            f"dlogM{_tag_from_float(args.logM_bin_width, precision=2)}_"
            f"n{int(args.max_per_bin)}_seed{int(args.random_seed)}"
        ),
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force_output:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --force-output to overwrite files")
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_redshift = _snapshot_redshift_map(str(args.api_base), str(args.simulation), api_key)
    target_redshifts = [float(value) for value in args.target_redshifts]
    if args.snapshots is None:
        snapshot_pairs = [_nearest_snapshot(target_z, snapshot_redshift) for target_z in target_redshifts]
    else:
        snapshots = [int(value) for value in args.snapshots]
        if len(snapshots) != len(target_redshifts):
            raise ValueError("--snapshots must have the same length as --target-redshifts")
        snapshot_pairs = []
        for snapshot in snapshots:
            if snapshot not in snapshot_redshift:
                raise KeyError(f"snapshot {snapshot} is not present in TNG API snapshot list")
            snapshot_pairs.append((snapshot, float(snapshot_redshift[snapshot])))

    seen_snapshots: set[int] = set()
    for snapshot, _snapshot_z in snapshot_pairs:
        if snapshot in seen_snapshots:
            raise ValueError(f"target redshifts map to duplicate snapshot {snapshot}")
        seen_snapshots.add(snapshot)

    logm_edges = _mass_bin_edges(float(args.logM_min), float(args.logM_max), float(args.logM_bin_width))
    rng = np.random.default_rng(int(args.random_seed))
    selections: list[SnapshotSelection] = []
    rows: list[dict[str, Any]] = []
    for target_z, (snapshot, snapshot_z) in zip(target_redshifts, snapshot_pairs, strict=True):
        first_sub_path = groupcat_dir / _field_filename(snapshot, "GroupFirstSub")
        mcrit_path = groupcat_dir / _field_filename(snapshot, "Group_M_Crit200")
        _download_groupcat_field(
            api_base=str(args.api_base),
            simulation=str(args.simulation),
            snapshot=snapshot,
            field="GroupFirstSub",
            destination=first_sub_path,
            api_key=api_key,
            force=bool(args.force_download),
        )
        _download_groupcat_field(
            api_base=str(args.api_base),
            simulation=str(args.simulation),
            snapshot=snapshot,
            field="Group_M_Crit200",
            destination=mcrit_path,
            api_key=api_key,
            force=bool(args.force_download),
        )
        selection = _select_snapshot_subhalos(
            target_z=float(target_z),
            snapshot=snapshot,
            snapshot_z=snapshot_z,
            group_first_sub_path=first_sub_path,
            group_mcrit200_path=mcrit_path,
            output_dir=output_dir,
            simulation=str(args.simulation),
            logm_edges=logm_edges,
            max_per_bin=int(args.max_per_bin),
            min_per_bin=int(args.min_per_bin),
            hubble=float(args.hubble),
            rng=rng,
        )
        selections.append(selection)
        rows.extend(selection.bin_rows)

    manifest_csv = output_dir / "selected_subhalos_manifest.csv"
    manifest_json = output_dir / "selected_subhalos_manifest.json"
    build_commands = output_dir / "build_cache_commands.sh"
    summary_md = output_dir / "selection_summary.md"

    per_snapshot: list[dict[str, Any]] = []
    total_selected = 0
    for selection in selections:
        status_counts = {status: sum(1 for row in selection.bin_rows if row["status"] == status) for status in ("ok", "partial", "insufficient", "empty")}
        selected_count = int(selection.selected_ids.size)
        total_selected += selected_count
        per_snapshot.append(
            {
                "target_z": float(selection.target_z),
                "snapshot": int(selection.snapshot),
                "snapshot_z": float(selection.snapshot_z),
                "selected_count": selected_count,
                "expected_mpb_bytes": int(selected_count * EXPECTED_MPB_BYTES),
                "ok_bins": int(status_counts["ok"]),
                "partial_bins": int(status_counts["partial"]),
                "insufficient_bins": int(status_counts["insufficient"]),
                "empty_bins": int(status_counts["empty"]),
            }
        )

    payload = {
        "simulation": str(args.simulation),
        "api_base": str(args.api_base),
        "target_redshifts": target_redshifts,
        "snapshots": [int(selection.snapshot) for selection in selections],
        "snapshot_redshifts": [float(selection.snapshot_z) for selection in selections],
        "logM_min": float(args.logM_min),
        "logM_max": float(args.logM_max),
        "logM_bin_width": float(args.logM_bin_width),
        "max_per_bin": int(args.max_per_bin),
        "min_per_bin": int(args.min_per_bin),
        "random_seed": int(args.random_seed),
        "build_download_workers": int(args.build_download_workers),
        "build_download_retries": int(args.build_download_retries),
        "build_drop_invalid_mpb": bool(args.build_drop_invalid_mpb),
        "build_snapshot_grid": str(args.build_snapshot_grid),
        "build_missing_mass_ratio_floor": float(args.build_missing_mass_ratio_floor),
        "hubble": float(args.hubble),
        "groupcat_dir": str(groupcat_dir),
        "output_dir": str(output_dir),
        "expected_mpb_bytes_per_halo": EXPECTED_MPB_BYTES,
        "total_selected": int(total_selected),
        "expected_mpb_bytes": int(total_selected * EXPECTED_MPB_BYTES),
        "per_snapshot": per_snapshot,
        "manifest_csv": str(manifest_csv),
        "build_commands": str(build_commands),
    }
    _write_manifest_csv(manifest_csv, rows)
    _write_json(manifest_json, payload)
    _write_build_commands(
        build_commands,
        selections=selections,
        simulation=str(args.simulation),
        output_dir=output_dir,
        raw_dir=PROJECT_ROOT / "external_data" / "tng" / str(args.simulation) / "raw_subtrees",
        download_workers=int(args.build_download_workers),
        download_retries=int(args.build_download_retries),
        drop_invalid_mpb=bool(args.build_drop_invalid_mpb),
        snapshot_grid=str(args.build_snapshot_grid),
        missing_mass_ratio_floor=float(args.build_missing_mass_ratio_floor),
    )
    _write_summary(summary_md, payload=payload, rows=rows)

    print(f"selection_output_dir={output_dir}", flush=True)
    print(f"selection_manifest_csv={manifest_csv}", flush=True)
    print(f"selection_manifest_json={manifest_json}", flush=True)
    print(f"build_cache_commands={build_commands}", flush=True)
    print(f"selection_summary={summary_md}", flush=True)
    print(f"total_selected={total_selected}", flush=True)
    print(f"expected_mpb_mib={total_selected * EXPECTED_MPB_BYTES / 1024**2:.3f}", flush=True)


if __name__ == "__main__":
    main()
