from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import h5py
import pytest

from auroralf.file_version import (
    capture_source_file_provenance,
    verify_source_file_provenance,
)
from auroralf.io.atomic_publish import publish_hdf5_atomic


def test_source_provenance_rejects_atomic_path_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original source bytes")
    provenance = capture_source_file_provenance(source)

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement source bytes")
    os.replace(replacement, source)

    with pytest.raises(ValueError, match="source file changed"):
        verify_source_file_provenance(provenance)


def test_source_provenance_rejects_in_place_content_change(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"first content")
    provenance = capture_source_file_provenance(source)
    source.write_bytes(b"second content")

    with pytest.raises(ValueError, match="source file changed"):
        verify_source_file_provenance(provenance)


def test_atomic_hdf5_publish_is_concurrent_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "cache.hdf5"

    def publish(value: int) -> Path:
        def writer(handle: h5py.File) -> None:
            handle.create_dataset("value", data=value)

        return publish_hdf5_atomic(target, writer, overwrite=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, value) for value in (1, 2)]

    successes: list[Path] = []
    failures: list[BaseException] = []
    for future in futures:
        try:
            successes.append(future.result())
        except BaseException as exc:
            failures.append(exc)

    assert successes == [target.resolve()]
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    with h5py.File(target, "r") as handle:
        assert int(handle["value"][()]) in {1, 2}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_hdf5_publish_cleans_owned_temp_after_writer_failure(tmp_path: Path) -> None:
    target = tmp_path / "cache.hdf5"

    def writer(handle: h5py.File) -> None:
        handle.create_dataset("partial", data=1)
        raise RuntimeError("injected writer failure")

    with pytest.raises(RuntimeError, match="injected writer failure"):
        publish_hdf5_atomic(target, writer, overwrite=False)

    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_hdf5_force_replaces_only_after_complete_write(tmp_path: Path) -> None:
    target = tmp_path / "cache.hdf5"
    target.write_bytes(b"old target")

    def failing_writer(handle: h5py.File) -> None:
        handle.create_dataset("partial", data=1)
        raise RuntimeError("injected writer failure")

    with pytest.raises(RuntimeError, match="injected writer failure"):
        publish_hdf5_atomic(target, failing_writer, overwrite=True)
    assert target.read_bytes() == b"old target"

    def successful_writer(handle: h5py.File) -> None:
        handle.create_dataset("complete", data=2)

    assert publish_hdf5_atomic(target, successful_writer, overwrite=True) == target.resolve()
    with h5py.File(target, "r") as handle:
        assert int(handle["complete"][()]) == 2
    assert list(tmp_path.glob(".*.tmp")) == []
