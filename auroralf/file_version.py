from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import os
from pathlib import Path
import stat as stat_module


def _strict_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer non-boolean value")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class FileVersion:
    path: Path
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if not self.path.is_absolute() or self.path != self.path.resolve():
            raise ValueError("path must be an absolute resolved Path")
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            object.__setattr__(
                self,
                name,
                _strict_nonnegative_integer(name, getattr(self, name)),
            )

    @classmethod
    def from_path(cls, path: str | Path) -> FileVersion:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        candidate = Path(path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"file does not exist: {candidate}") from None
        file_stat = resolved.stat()
        if not stat_module.S_ISREG(file_stat.st_mode):
            raise ValueError(f"path must be a regular file: {resolved}")
        return cls(
            path=resolved,
            st_dev=file_stat.st_dev,
            st_ino=file_stat.st_ino,
            st_size=file_stat.st_size,
            st_mtime_ns=file_stat.st_mtime_ns,
            st_ctime_ns=file_stat.st_ctime_ns,
        )


def _version_matches_stat(version: FileVersion, file_stat: os.stat_result) -> bool:
    return (
        version.st_dev == file_stat.st_dev
        and version.st_ino == file_stat.st_ino
        and version.st_size == file_stat.st_size
        and version.st_mtime_ns == file_stat.st_mtime_ns
        and version.st_ctime_ns == file_stat.st_ctime_ns
    )


def _sha256_open_descriptor(descriptor: int, *, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError("source file changed while its SHA-256 was being computed")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFileProvenance:
    version: FileVersion
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.version, FileVersion):
            raise TypeError("version must be a FileVersion")
        checksum = str(self.sha256).strip().lower()
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal digest")
        object.__setattr__(self, "sha256", checksum)

    @property
    def identifier(self) -> str:
        return str(self.version.path)


def capture_source_file_provenance(path: str | Path) -> SourceFileProvenance:
    version = FileVersion.from_path(path)
    descriptor = os.open(
        version.path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or not _version_matches_stat(version, before):
            raise ValueError(f"source file changed before it could be hashed: {version.path}")
        checksum = _sha256_open_descriptor(descriptor, size=version.st_size)
        after = os.fstat(descriptor)
        if not _version_matches_stat(version, after):
            raise ValueError(f"source file changed while it was being hashed: {version.path}")
        try:
            current = FileVersion.from_path(version.path)
        except FileNotFoundError as exc:
            raise ValueError(f"source file changed while it was being hashed: {version.path}") from exc
        if current != version:
            raise ValueError(f"source file changed while it was being hashed: {version.path}")
        return SourceFileProvenance(version=version, sha256=checksum)
    finally:
        os.close(descriptor)


def verify_source_file_provenance(provenance: SourceFileProvenance) -> None:
    if not isinstance(provenance, SourceFileProvenance):
        raise TypeError("provenance must be a SourceFileProvenance")
    try:
        current = capture_source_file_provenance(provenance.version.path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"source file changed after scientific data were read: {provenance.identifier}") from exc
    if current != provenance:
        raise ValueError(f"source file changed after scientific data were read: {provenance.identifier}")


__all__ = [
    "FileVersion",
    "SourceFileProvenance",
    "capture_source_file_provenance",
    "verify_source_file_provenance",
]
