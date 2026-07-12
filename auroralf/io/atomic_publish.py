from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import stat
import uuid

import h5py


HDF5Writer = Callable[[h5py.File], None]


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_owned_path(path: Path, descriptor: int, identity: tuple[int, int]) -> None:
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("owned temporary HDF5 file disappeared before publication") from exc
    if _identity(descriptor_stat) != identity or _identity(path_stat) != identity:
        raise ValueError("owned temporary HDF5 file changed identity before publication")
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("owned temporary HDF5 path is not a regular file")


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _identity(path_stat) == identity and stat.S_ISREG(path_stat.st_mode):
        path.unlink()


def publish_hdf5_atomic(
    target: str | Path,
    writer: HDF5Writer,
    *,
    overwrite: bool,
) -> Path:
    if not callable(writer):
        raise TypeError("writer must be callable")
    destination = Path(target).expanduser()
    if not destination.is_absolute():
        destination = destination.resolve()
    else:
        destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"output cache already exists: {destination}")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    identity = _identity(os.fstat(descriptor))
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(os.dup(descriptor), "w+b") as raw_file:
            with h5py.File(raw_file, "w") as handle:
                writer(handle)
                handle.flush()
            raw_file.flush()
        os.fsync(descriptor)
        _require_owned_path(temporary, descriptor, identity)
        _fsync_directory(destination.parent)

        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError:
                raise FileExistsError(f"output cache already exists: {destination}") from None
            _require_owned_path(temporary, descriptor, identity)
            destination_stat = os.stat(destination, follow_symlinks=False)
            if _identity(destination_stat) != identity:
                raise RuntimeError("published HDF5 target is not bound to the owned temporary file")
            temporary.unlink()
        published = True
        _fsync_directory(destination.parent)
        destination_stat = os.stat(destination, follow_symlinks=False)
        if _identity(destination_stat) != identity or not stat.S_ISREG(destination_stat.st_mode):
            raise RuntimeError("published HDF5 target changed identity during commit")
        return destination
    finally:
        if not published:
            _unlink_if_owned(temporary, identity)
            _fsync_directory(destination.parent)
        os.close(descriptor)


__all__ = ["publish_hdf5_atomic"]
