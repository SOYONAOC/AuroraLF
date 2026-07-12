from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from auroralf.file_version import FileVersion


def test_file_version_from_path_is_stable_for_unchanged_regular_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.dat"
    path.write_bytes(b"abcd")

    first = FileVersion.from_path(path)
    second = FileVersion.from_path(path)

    assert first == second
    assert first.path == path.resolve()
    assert first.path.is_absolute()
    assert first.st_size == 4


def test_file_version_requires_existing_regular_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.dat"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        FileVersion.from_path(missing)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        FileVersion.from_path(directory)


def test_file_version_constructor_strictly_validates_fields(tmp_path: Path) -> None:
    path = tmp_path / "input.dat"
    path.write_bytes(b"abcd")
    version = FileVersion.from_path(path)

    with pytest.raises(TypeError, match="st_size.*integer non-boolean"):
        replace(version, st_size=True)
    with pytest.raises(ValueError, match="path.*absolute.*resolved"):
        replace(version, path=Path("relative.dat"))
    with pytest.raises(ValueError, match="st_mtime_ns.*non-negative"):
        replace(version, st_mtime_ns=-1)


def test_file_version_changes_after_atomic_replace_with_same_size_and_restored_mtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.dat"
    path.write_bytes(b"old!")
    first = FileVersion.from_path(path)
    replacement = tmp_path / "replacement.dat"
    replacement.write_bytes(b"new!")
    os.utime(replacement, ns=(first.st_mtime_ns, first.st_mtime_ns))

    os.replace(replacement, path)
    os.utime(path, ns=(first.st_mtime_ns, first.st_mtime_ns))
    second = FileVersion.from_path(path)

    assert second.path == first.path
    assert second.st_size == first.st_size
    assert second.st_mtime_ns == first.st_mtime_ns
    assert second != first
    assert (second.st_ino, second.st_ctime_ns) != (first.st_ino, first.st_ctime_ns)
