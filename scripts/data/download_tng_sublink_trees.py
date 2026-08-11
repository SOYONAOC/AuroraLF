#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SCRIPT_DATA_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DATA_DIR))

from _tng_selection_inputs import (  # noqa: E402
    _pairs_from_manifest,
    _read_id_file,
    _resolve_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION = "TNG100-1-Dark"
DEFAULT_API_BASE = "https://www.tng-project.org/api"
TREE_ENDPOINTS = {
    "simple": ("simple.json", ".json"),
    "full": ("full.hdf5", ".hdf5"),
    "mpb": ("mpb.hdf5", ".hdf5"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected TNG SubLink tree products from the API. "
            "Raw files are cached under external_data/tng and reused unless --force-download is set."
        )
    )
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--tree-kind", choices=tuple(TREE_ENDPOINTS), required=True)
    parser.add_argument(
        "--selection-manifest",
        type=str,
        default=None,
        help="CSV manifest written by select_tng_mah_subhalos.py; all unique all_id_file rows are used.",
    )
    parser.add_argument(
        "--snapshot-id-file",
        nargs=2,
        action="append",
        metavar=("SNAPSHOT", "ID_FILE"),
        default=None,
        help="Additional or alternative snapshot/id-file pair. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--limit-per-snapshot", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError("TNG_API_KEY must be set to download TNG API data")
    return api_key


def _collect_jobs(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, Path]]:
    endpoint, suffix = TREE_ENDPOINTS[str(args.tree_kind)]
    snapshot_pairs: list[tuple[int, Path]] = []
    if args.selection_manifest is not None:
        snapshot_pairs.extend(_pairs_from_manifest(_resolve_path(args.selection_manifest)))
    if args.snapshot_id_file is not None:
        for snapshot_value, id_file in args.snapshot_id_file:
            snapshot_pairs.append((int(snapshot_value), _resolve_path(id_file)))
    if len(snapshot_pairs) == 0:
        raise ValueError("provide --selection-manifest and/or at least one --snapshot-id-file SNAPSHOT ID_FILE")

    jobs: list[tuple[str, Path]] = []
    seen: set[tuple[int, int]] = set()
    for snapshot, id_file in snapshot_pairs:
        ids = _read_id_file(id_file, limit=args.limit_per_snapshot)
        for subhalo_id in ids:
            key = (int(snapshot), int(subhalo_id))
            if key in seen:
                continue
            seen.add(key)
            url = (
                f"{args.api_base.rstrip('/')}/{args.simulation}/snapshots/{int(snapshot)}"
                f"/subhalos/{int(subhalo_id)}/sublink/{endpoint}"
            )
            destination = (
                output_dir
                / f"snap_{int(snapshot):03d}"
                / f"subhalo_{int(subhalo_id)}_sublink_{args.tree_kind}{suffix}"
            )
            jobs.append((url, destination))
    if len(jobs) == 0:
        raise RuntimeError("no download jobs were constructed")
    return jobs


def _validate_download(path: Path, tree_kind: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"downloaded file is empty or missing: {path}")
    if tree_kind == "simple":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError(f"simple SubLink payload must be a JSON object: {path}")


def _download_one(
    url: str,
    destination: Path,
    *,
    api_key: str,
    tree_kind: str,
    force: bool,
    retries: int,
) -> tuple[Path, int, bool]:
    if destination.exists() and not force:
        _validate_download(destination, tree_kind)
        return destination, int(destination.stat().st_size), True
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
            _validate_download(destination, tree_kind)
            return destination, int(destination.stat().st_size), False
        except (OSError, RuntimeError, requests.RequestException, json.JSONDecodeError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt >= attempts:
                raise RuntimeError(f"TNG SubLink download failed after {attempts} attempt(s): {destination}") from exc
            print(
                f"retrying={destination} attempt={attempt + 1}/{attempts} reason={exc.__class__.__name__}",
                flush=True,
            )
            time.sleep(min(2.0 ** (attempt - 1), 30.0))
    raise RuntimeError(f"unreachable download failure for {destination}")


def _download_all(args: argparse.Namespace, jobs: list[tuple[str, Path]], api_key: str) -> tuple[int, int, int]:
    if int(args.download_workers) < 1:
        raise ValueError("download-workers must be positive")
    downloaded = 0
    reused = 0
    total_bytes = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=int(args.download_workers)) as executor:
        futures = [
            executor.submit(
                _download_one,
                url,
                destination,
                api_key=api_key,
                tree_kind=str(args.tree_kind),
                force=bool(args.force_download),
                retries=int(args.download_retries),
            )
            for url, destination in jobs
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                path, size, was_reused = future.result()
            except Exception as exc:
                failures.append(f"{exc.__class__.__name__}: {exc}")
                path = Path("<failed>")
                size = 0
                was_reused = False
            else:
                total_bytes += int(size)
                reused += int(was_reused)
                downloaded += int(not was_reused)
            if index == 1 or index % 100 == 0 or index == len(futures):
                print(
                    f"progress={index}/{len(futures)} downloaded={downloaded} "
                    f"reused={reused} failed={len(failures)} bytes={total_bytes} last={path}",
                    flush=True,
                )
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(f"{len(failures)} TNG SubLink download(s) failed:\n{preview}")
    return downloaded, reused, total_bytes


def main() -> None:
    args = _parse_args()
    if int(args.download_workers) < 1:
        raise ValueError("download-workers must be positive")
    if int(args.download_retries) < 0:
        raise ValueError("download-retries must be non-negative")
    if args.limit_per_snapshot is not None and int(args.limit_per_snapshot) <= 0:
        raise ValueError("limit-per-snapshot must be positive when provided")

    api_key = _require_api_key()
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "external_data" / "tng" / args.simulation / f"raw_sublink_{args.tree_kind}"
    )
    jobs = _collect_jobs(args, output_dir)
    downloaded, reused, total_bytes = _download_all(args, jobs, api_key)
    print(f"tree_kind={args.tree_kind}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"n_jobs={len(jobs)}", flush=True)
    print(f"n_downloaded={downloaded}", flush=True)
    print(f"n_reused={reused}", flush=True)
    print(f"total_bytes={total_bytes}", flush=True)


if __name__ == "__main__":
    main()
