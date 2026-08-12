"""Owned-descriptor identity checks used by HDF5 transactions and codecs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat as stat_module


@dataclass(slots=True)
class _OwnedFile:
    path: Path
    descriptor: int
    identity: tuple[int, int]


def _require_owned_path(owned_file: _OwnedFile, *, label: str) -> None:
    if owned_file.descriptor < 0:
        raise ValueError(f"{label} ownership descriptor is closed")
    descriptor_stat = os.fstat(owned_file.descriptor)
    try:
        path_stat = os.stat(owned_file.path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} changed identity after exclusive creation") from error
    if not (
        (descriptor_stat.st_dev, descriptor_stat.st_ino)
        == owned_file.identity
        == (path_stat.st_dev, path_stat.st_ino)
    ):
        raise ValueError(f"{label} changed identity after exclusive creation")
    if not stat_module.S_ISREG(path_stat.st_mode):
        raise ValueError(f"{label} is no longer a regular file")


__all__ = []
