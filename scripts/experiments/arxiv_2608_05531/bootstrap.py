#!/usr/bin/env python3
"""Plan, download, and verify the public inputs for arXiv:2608.05531.

This script uses only the Python standard library so that input acquisition can
be audited before the paper's unpinned scientific environment is installed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[2]
MANIFEST_PATH = EXPERIMENT_DIR / "zenodo_manifest.json"
DEFAULT_TARGET = (
    REPOSITORY_ROOT
    / "external_data"
    / "literature_sources"
    / "arxiv_2608_05531"
    / "zenodo_20695048"
)

STAGE_FILES: dict[str, tuple[str, ...]] = {
    "ml-min": (
        "README.md",
        "code.zip",
        "Model_training.csv",
        "Tobe_predicted.csv",
    ),
    "ml-verify": (
        "README.md",
        "code.zip",
        "Model_training.csv",
        "Tobe_predicted.csv",
        "regalade_knownSFR.csv",
        "Allgalaxy_final_version.csv",
    ),
    "maps": (
        "README.md",
        "code.zip",
        "Allgalaxy_final_version.csv",
        "MASS.zip",
        "SFR.zip",
    ),
    "all": (
        "README.md",
        "SFR.zip",
        "MASS.zip",
        "Model_training.csv",
        "regalade_knownSFR.csv",
        "code.zip",
        "Tobe_predicted.csv",
        "Allgalaxy_final_version.csv",
    ),
}


class InputVerificationError(RuntimeError):
    """A required public input is absent or does not match its archive record."""


@contextmanager
def exclusive_target_lock(target: Path):
    """Prevent concurrent processes from appending to the same partial file."""
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / ".bootstrap.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InputVerificationError(
                f"Another bootstrap process holds the target lock: {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise InputVerificationError(
            f"Unsupported manifest schema in {path}: {manifest.get('schema_version')!r}"
        )
    return manifest


def index_files(manifest: dict) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for entry in manifest["files"]:
        name = entry["name"]
        if name in indexed:
            raise InputVerificationError(f"Duplicate manifest entry: {name}")
        indexed[name] = entry
    return indexed


def selected_entries(manifest: dict, stage: str) -> list[dict]:
    indexed = index_files(manifest)
    missing = [name for name in STAGE_FILES[stage] if name not in indexed]
    if missing:
        raise InputVerificationError(
            f"Stage {stage!r} references missing manifest files: {missing}"
        )
    return [indexed[name] for name in STAGE_FILES[stage]]


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise InputVerificationError(f"Missing required input: {path}")
    actual_size = path.stat().st_size
    expected_size = entry["size_bytes"]
    if actual_size != expected_size:
        raise InputVerificationError(
            f"Size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_md5 = md5sum(path)
    expected_md5 = entry["md5"]
    if actual_md5 != expected_md5:
        raise InputVerificationError(
            f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
        )


def download_file(target: Path, entry: dict, chunk_size: int = 1024 * 1024) -> None:
    if target.exists():
        verify_file(target, entry)
        print(f"verified existing  {target.name}")
        return

    partial = target.with_name(f"{target.name}.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    expected_size = entry["size_bytes"]
    if offset > expected_size:
        raise InputVerificationError(
            f"Partial download is larger than the archive record for {target.name}: "
            f"expected at most {expected_size}, got {offset}"
        )
    if offset == expected_size:
        verify_file(partial, entry)
        os.replace(partial, target)
        print(f"verified partial   {target.name}")
        return

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(entry["url"], headers=headers)
    action = "resuming" if offset else "downloading"
    print(f"{action:<19}{target.name}")
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {offset}-"
            if response.status != 206 or not content_range.startswith(expected_prefix):
                raise InputVerificationError(
                    f"Server did not honor the resume request for {target.name}: "
                    f"HTTP {response.status}, Content-Range={content_range!r}"
                )
        mode = "ab" if offset else "xb"
        with partial.open(mode) as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)

    verify_file(partial, entry)
    os.replace(partial, target)
    print(f"downloaded+verified {target.name}")


def gibibytes(size_bytes: int) -> float:
    return size_bytes / 1024**3


def print_plan(stage: str, target: Path, entries: list[dict]) -> None:
    print(f"stage:  {stage}")
    print(f"target: {target}")
    print("files:")
    for entry in entries:
        size_mib = entry["size_bytes"] / 1024**2
        print(f"  {entry['name']:<30} {size_mib:8.2f} MiB")
    total = sum(entry["size_bytes"] for entry in entries)
    print(f"archive total: {total:,} bytes ({total / 1024**2:.2f} MiB)")
    print(f"minimum free space for archives only: {gibibytes(total):.3f} GiB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_FILES), default="ml-verify")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--download",
        action="store_true",
        help="download missing files and verify every selected file",
    )
    action.add_argument(
        "--verify-only",
        action="store_true",
        help="verify selected local files and fail if any are absent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.target.resolve()
    manifest = load_manifest()
    entries = selected_entries(manifest, args.stage)
    print_plan(args.stage, target, entries)

    if args.download:
        with exclusive_target_lock(target):
            for entry in entries:
                download_file(target / entry["name"], entry)
        print("all selected files downloaded and verified")
    elif args.verify_only:
        with exclusive_target_lock(target):
            for entry in entries:
                verify_file(target / entry["name"], entry)
                print(f"verified           {entry['name']}")
        print("all selected files verified")
    else:
        print("plan only; pass --download or --verify-only to perform an action")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputVerificationError as error:
        print(f"input verification failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
