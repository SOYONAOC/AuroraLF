from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import stat

import numpy as np
import pytest

import auroralf
from auroralf.io import (
    ArtifactProvenance,
    HaloSampleTable,
    read_uvlf_shard,
    run_uvlf_to_sample_shards,
)
from auroralf.io.sample_sink import _HDF5SampleSink
from auroralf.uvlf.hmf_sampling import uv_luminosity_to_muv
from auroralf.uvlf.runner import run_uvlf_streaming
from tests.test_shared_batch_runner import (
    _assert_run_science_bitwise_equal,
    _config,
    _install_fake_science,
    _real_config,
)


def _provenance(config: object) -> ArtifactProvenance:
    return ArtifactProvenance.for_config(
        config,  # type: ignore[arg-type]
        code_revision="a" * 40,
        code_dirty=False,
        seed_namespace="auroralf.sample-sink.v1",
        source_paths=(
            ("canonical_ssp", config.stellar_population.canonical_ssp_path),  # type: ignore[attr-defined]
            ("topheavy_ssp", config.stellar_population.topheavy_ssp_path),  # type: ignore[attr-defined]
            ("popiii_ssp", config.stellar_population.popiii_ssp_path),  # type: ignore[attr-defined]
        ),
        created_utc="2026-07-11T00:00:00Z",
    )


def _sample(*, mass_index: int, mode: str = "canonical") -> HaloSampleTable:
    luminosity = np.array([1.0e28, 2.0e28])
    return HaloSampleTable(
        redshift=6.0,
        imf_mode=mode,
        mass_index=np.repeat(mass_index, 2),
        track_index=np.arange(2),
        halo_mass_msun=np.repeat(1.0e10 + mass_index, 2),
        mass_weight_per_mpc3=np.repeat(5.0e-5, 2),
        uv_luminosity_erg_per_s_hz=luminosity,
        muv=uv_luminosity_to_muv(luminosity),
        sfr_msun_per_yr=np.array([0.2, 0.3]),
        popiii_sfr_msun_per_yr=np.array([0.0, 0.1]),
    )


def test_default_runner_mass_payload_has_no_per_track_sfr_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    _install_fake_science(monkeypatch, tmp_path, [])
    observed: list[object] = []

    run_uvlf_streaming(config, _mass_result_observer=observed.append)

    assert len(observed) == 1
    payload = observed[0]
    assert payload.final_sfr_msun_per_yr is None  # type: ignore[attr-defined]
    assert payload.final_popiii_sfr_msun_per_yr is None  # type: ignore[attr-defined]


def test_runner_halo_sample_observer_receives_exact_mass_mode_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=2,
        n_tracks=2,
    )
    _install_fake_science(monkeypatch, tmp_path, [])
    samples: list[HaloSampleTable] = []

    run_uvlf_streaming(config, _halo_sample_observer=samples.append)

    assert [(sample.mass_index[0], sample.imf_mode) for sample in samples] == [
        (0, "canonical"),
        (0, "mah_burst_mild_topheavy"),
        (1, "canonical"),
        (1, "mah_burst_mild_topheavy"),
    ]
    for sample in samples:
        assert type(sample) is HaloSampleTable
        np.testing.assert_array_equal(sample.mass_index, np.repeat(sample.mass_index[0], 2))
        np.testing.assert_array_equal(sample.track_index, np.arange(2, dtype=np.int64))
        np.testing.assert_array_equal(sample.halo_mass_msun, np.repeat(sample.halo_mass_msun[0], 2))
        np.testing.assert_array_equal(sample.sfr_msun_per_yr, np.repeat(0.2, 2))
        np.testing.assert_array_equal(sample.popiii_sfr_msun_per_yr, np.zeros(2))
        np.testing.assert_array_equal(
            sample.mass_weight_per_mpc3,
            np.repeat(sample.mass_weight_per_mpc3[0], 2),
        )
        assert sample.mass_weight_per_mpc3[0] > 0.0
        np.testing.assert_array_equal(
            sample.muv,
            uv_luminosity_to_muv(sample.uv_luminosity_erg_per_s_hz),
        )
        for array_name in (
            "mass_index",
            "track_index",
            "halo_mass_msun",
            "mass_weight_per_mpc3",
            "uv_luminosity_erg_per_s_hz",
            "muv",
            "sfr_msun_per_yr",
            "popiii_sfr_msun_per_yr",
        ):
            assert getattr(sample, array_name).flags.writeable is False


def test_runner_rejects_noncallable_halo_sample_observer_before_science_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=1,
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)

    with pytest.raises(TypeError, match="_halo_sample_observer must be callable"):
        run_uvlf_streaming(config, _halo_sample_observer=object())  # type: ignore[arg-type]

    assert events == []


def test_runner_propagates_halo_sample_observer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=1,
    )
    _install_fake_science(monkeypatch, tmp_path, [])
    expected = RuntimeError("sample append failed")

    def fail(sample: HaloSampleTable) -> None:
        del sample
        raise expected

    with pytest.raises(RuntimeError, match="sample append failed") as captured:
        run_uvlf_streaming(config, _halo_sample_observer=fail)

    assert captured.value is expected


def test_sample_spool_is_same_directory_owned_0600_and_abort_removes_it(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()

    sink = _HDF5SampleSink(config, _provenance(config), shard_directory)
    spool_path = sink.spool_path

    assert spool_path.parent == shard_directory.resolve()
    assert spool_path.name.startswith(".auroralf-samples-")
    assert stat.S_IMODE(spool_path.stat().st_mode) == 0o600
    sink.abort()
    assert not spool_path.exists()
    sink.abort()


def test_sample_spool_appends_extensible_compressed_datasets_without_batches(
    tmp_path: Path,
) -> None:
    import h5py

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=2,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    sink = _HDF5SampleSink(config, _provenance(config), shard_directory)

    sink.append(_sample(mass_index=0, mode="canonical"))
    sink.append(_sample(mass_index=0, mode="mah_burst_mild_topheavy"))
    sink.append(_sample(mass_index=1, mode="canonical"))
    sink.append(_sample(mass_index=1, mode="mah_burst_mild_topheavy"))

    assert not hasattr(sink, "batches")
    with h5py.File(sink.spool_path, "r") as handle:
        assert handle.attrs["artifact_kind"] == "sample_spool"
        for mode in config.stellar_population.imf_modes:
            group = handle["samples"]["z=6"][mode]
            assert int(group.attrs["sample_count"]) == 4
            assert int(group.attrs["next_mass_index"]) == 2
            assert set(group) == {
                "mass_index",
                "track_index",
                "halo_mass_msun",
                "mass_weight_per_mpc3",
                "uv_luminosity_erg_per_s_hz",
                "muv",
                "sfr_msun_per_yr",
                "popiii_sfr_msun_per_yr",
            }
            for dataset in group.values():
                assert dataset.shape == (4,)
                assert dataset.maxshape == (None,)
                assert dataset.chunks is not None
                assert dataset.compression == "gzip"
    sink.abort()


def test_sample_spool_rejects_out_of_order_batch_without_partial_append(
    tmp_path: Path,
) -> None:
    import h5py

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=2,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    sink = _HDF5SampleSink(config, _provenance(config), shard_directory)
    sink.append(_sample(mass_index=0, mode="canonical"))

    with pytest.raises(RuntimeError, match="sample batch order"):
        sink.append(_sample(mass_index=1, mode="canonical"))

    with h5py.File(sink.spool_path, "r") as handle:
        group = handle["samples"]["z=6"]["canonical"]
        assert group["mass_index"].shape == (2,)
    sink.abort()


def test_run_uvlf_to_sample_shards_writes_exact_axis_order_and_keeps_root_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=2,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    expected = run_uvlf_streaming(config)
    _install_fake_science(monkeypatch, tmp_path, [])

    result, paths = run_uvlf_to_sample_shards(
        config,
        provenance,
        shard_directory,
    )

    _assert_run_science_bitwise_equal(result, expected)
    assert tuple(path.name for path in paths) == tuple(
        f"{config.run_id}.z=6.{mode}.shard.h5"
        for mode in config.stellar_population.imf_modes
    )
    assert auroralf.__all__ == ["UVLFRunConfig", "UVLFRunResult", "run_uvlf"]
    assert not hasattr(auroralf, "run_uvlf_to_sample_shards")
    for mode_index, path in enumerate(paths):
        shard = read_uvlf_shard(path, load_samples=True)
        assert shard.key == (6.0, config.stellar_population.imf_modes[mode_index])
        assert shard.sample_descriptor is not None
        assert shard.sample_descriptor.sample_count == 4
        assert shard.sample is not None
        np.testing.assert_array_equal(shard.sample.mass_index, [0, 0, 1, 1])
        np.testing.assert_array_equal(shard.sample.track_index, [0, 1, 0, 1])
        np.testing.assert_array_equal(shard.sample.sfr_msun_per_yr, np.repeat(0.2, 4))
    assert not tuple(shard_directory.glob("*.spool.h5"))


def test_run_uvlf_to_sample_shards_cleans_spool_when_observer_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    expected = RuntimeError("injected observer failure")

    def fail(self: object, sample: HaloSampleTable) -> None:
        del self, sample
        raise expected

    monkeypatch.setattr(_HDF5SampleSink, "append", fail)

    with pytest.raises(RuntimeError, match="injected observer failure") as captured:
        run_uvlf_to_sample_shards(
            config,
            _provenance(config),
            shard_directory,
        )

    assert captured.value is expected
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.shard.h5"))


def test_run_uvlf_to_sample_shards_rejects_tampered_spool_and_cleans_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    result = run_uvlf_streaming(config)

    def tamper_then_return(
        run_config: object,
        *,
        _halo_sample_observer: object,
    ) -> object:
        assert run_config == config
        observer = _halo_sample_observer
        observer(_sample(mass_index=0))  # type: ignore[operator]
        sink = observer.__self__  # type: ignore[attr-defined]
        with h5py.File(sink.spool_path, "r+") as handle:
            handle["config"]["sha256"][()] = "0" * 64
        return result

    monkeypatch.setattr(runner, "run_uvlf_streaming", tamper_then_return)

    with pytest.raises(ValueError, match="sample spool config mismatch"):
        run_uvlf_to_sample_shards(
            config,
            _provenance(config),
            shard_directory,
        )

    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.shard.h5"))


def test_run_uvlf_to_sample_shards_rejects_tampered_spool_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    result = run_uvlf_streaming(config)

    def tamper_then_return(
        run_config: object,
        *,
        _halo_sample_observer: object,
    ) -> object:
        assert run_config == config
        observer = _halo_sample_observer
        observer(_sample(mass_index=0))  # type: ignore[operator]
        sink = observer.__self__  # type: ignore[attr-defined]
        with h5py.File(sink.spool_path, "r+") as handle:
            handle.attrs["schema_name"] = "tampered"
        return result

    monkeypatch.setattr(runner, "run_uvlf_streaming", tamper_then_return)

    with pytest.raises(ValueError, match="sample spool schema_name mismatch"):
        run_uvlf_to_sample_shards(
            config,
            _provenance(config),
            shard_directory,
        )

    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.shard.h5"))


def test_overwrite_marker_failure_restores_old_shard_and_cleans_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as hdf5_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    _, (path,) = run_uvlf_to_sample_shards(
        config,
        provenance,
        shard_directory,
    )
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    _install_fake_science(monkeypatch, tmp_path, [])

    def fail_marker(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected marker failure")

    monkeypatch.setattr(hdf5_module, "_write_completion_marker_atomic", fail_marker)

    with pytest.raises(RuntimeError, match="injected marker failure"):
        run_uvlf_to_sample_shards(
            config,
            provenance,
            shard_directory,
            overwrite=True,
        )

    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.tmp"))
    assert not tuple(shard_directory.glob("*.backup"))


def test_source_change_after_sampling_fails_and_cleans_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    result = run_uvlf_streaming(config)

    def replace_source_then_return(
        run_config: object,
        *,
        _halo_sample_observer: object,
    ) -> object:
        assert run_config == config
        _halo_sample_observer(_sample(mass_index=0))  # type: ignore[operator]
        config.stellar_population.canonical_ssp_path.write_bytes(b"changed")
        return result

    monkeypatch.setattr(runner, "run_uvlf_streaming", replace_source_then_return)

    with pytest.raises(ValueError, match="source size mismatch for canonical_ssp"):
        run_uvlf_to_sample_shards(
            config,
            provenance,
            shard_directory,
        )

    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.shard.h5"))


def test_real_workers2_sample_sink_matches_disabled_result_bitwise(
    tmp_path: Path,
) -> None:
    base = _real_config(
        tmp_path,
        modes=("canonical", "mah_burst_mild_topheavy"),
        n_tracks=1,
        n_mass=2,
        mass_batch_size=1,
    )
    parallel = replace(base, sampling=replace(base.sampling, workers=2))
    expected = run_uvlf_streaming(base)
    shard_directory = tmp_path / "parallel-shards"
    shard_directory.mkdir()

    actual, paths = run_uvlf_to_sample_shards(
        parallel,
        _provenance(parallel),
        shard_directory,
    )

    _assert_run_science_bitwise_equal(actual, expected)
    assert len(paths) == 2
    for path in paths:
        shard = read_uvlf_shard(path, load_samples=True)
        assert shard.sample_descriptor is not None
        assert shard.sample_descriptor.sample_count == 2
        assert shard.sample is not None
        np.testing.assert_array_equal(shard.sample.mass_index, [0, 1])
        np.testing.assert_array_equal(shard.sample.track_index, [0, 0])
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))


def test_missing_owned_spool_fails_without_recreating_or_removing_other_files(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    unrelated = shard_directory / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    sink = _HDF5SampleSink(config, _provenance(config), shard_directory)
    sink.spool_path.unlink()

    with pytest.raises(ValueError, match="sample spool path identity changed"):
        sink.append(_sample(mass_index=0))

    sink.abort()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))


def test_overwrite_false_preserves_existing_shard_and_cleans_new_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    _, (path,) = run_uvlf_to_sample_shards(config, provenance, shard_directory)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    _install_fake_science(monkeypatch, tmp_path, [])

    with pytest.raises(FileExistsError, match="already exists"):
        run_uvlf_to_sample_shards(config, provenance, shard_directory)

    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.tmp"))
    assert not tuple(shard_directory.glob("*.backup"))


def test_concurrent_commit_lock_preserves_existing_shard_and_cleans_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as hdf5_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    _, (path,) = run_uvlf_to_sample_shards(config, provenance, shard_directory)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    lock_path = path.parent / f".{path.name}.commit.lock"
    lock = hdf5_module._acquire_commit_lock(lock_path)
    _install_fake_science(monkeypatch, tmp_path, [])
    try:
        with pytest.raises(FileExistsError, match="commit lock is held"):
            run_uvlf_to_sample_shards(
                config,
                provenance,
                shard_directory,
                overwrite=True,
            )
    finally:
        hdf5_module._close_owned_file(lock)

    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.tmp"))
    assert not tuple(shard_directory.glob("*.backup"))


def test_spool_copy_observer_reports_only_bounded_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.hdf5 as hdf5_module
    import auroralf.io.sample_sink as sample_sink_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    copied_lengths: list[int] = []
    original = hdf5_module._write_uvlf_shard_from_spool_atomic

    def write_with_spy(*args: object, **kwargs: object) -> Path:
        kwargs["copy_chunk_size"] = 1
        kwargs["_copy_observer"] = copied_lengths.append
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        sample_sink_module,
        "_write_uvlf_shard_from_spool_atomic",
        write_with_spy,
    )

    run_uvlf_to_sample_shards(
        config,
        _provenance(config),
        shard_directory,
    )

    assert copied_lengths == [1, 1]


def test_spool_create_to_init_path_replacement_never_writes_foreign_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import auroralf.io.sample_sink as sample_sink_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    victim = shard_directory / "victim.bin"
    victim_bytes = b"foreign-create-window"
    victim.write_bytes(victim_bytes)
    original_file = sample_sink_module.h5py.File
    replaced_paths: list[Path] = []

    def replace_before_init(file_object: object, *args: object, **kwargs: object):
        if not replaced_paths:
            spool_path = next(shard_directory.glob(".auroralf-samples-*.spool.h5"))
            os.replace(victim, spool_path)
            replaced_paths.append(spool_path)
        return original_file(file_object, *args, **kwargs)

    monkeypatch.setattr(sample_sink_module.h5py, "File", replace_before_init)

    with pytest.raises(ValueError, match="sample spool path identity changed"):
        _HDF5SampleSink(config, _provenance(config), shard_directory)

    assert len(replaced_paths) == 1
    assert replaced_paths[0].read_bytes() == victim_bytes


def test_spool_append_check_to_open_path_replacement_never_writes_foreign_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import auroralf.io.sample_sink as sample_sink_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    sink = _HDF5SampleSink(config, _provenance(config), shard_directory)
    descriptor = sink._descriptor
    victim = shard_directory / "victim.bin"
    victim_bytes = b"foreign-append-window"
    victim.write_bytes(victim_bytes)
    original_file = sample_sink_module.h5py.File
    attacked = False

    def replace_before_append(file_object: object, *args: object, **kwargs: object):
        nonlocal attacked
        if not attacked:
            os.replace(victim, sink.spool_path)
            attacked = True
        return original_file(file_object, *args, **kwargs)

    monkeypatch.setattr(sample_sink_module.h5py, "File", replace_before_append)

    with pytest.raises(ValueError, match="sample spool path identity changed"):
        sink.append(_sample(mass_index=0))

    assert sink.spool_path.read_bytes() == victim_bytes
    with pytest.raises(RuntimeError, match="refusing to remove a replaced sample spool"):
        sink.abort()
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert sink.spool_path.read_bytes() == victim_bytes


def test_finalize_snapshot_change_rolls_back_old_shard_and_preserves_foreign_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import shutil
    import h5py
    import auroralf.io.hdf5 as hdf5_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    _, (path,) = run_uvlf_to_sample_shards(config, provenance, shard_directory)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    original_require = hdf5_module._require_spool_snapshot
    require_count = 0
    captured_descriptors: list[int] = []
    replaced_paths: list[Path] = []

    def replace_after_copy(
        descriptor: int,
        spool_path: Path,
        owner: tuple[int, int],
        snapshot: object,
    ) -> None:
        nonlocal require_count
        require_count += 1
        captured_descriptors.append(descriptor)
        original_require(descriptor, spool_path, owner, snapshot)
        if require_count == 2:
            foreign = spool_path.with_name(spool_path.name + ".foreign")
            shutil.copyfile(spool_path, foreign)
            with h5py.File(foreign, "r+") as handle:
                dataset = handle["samples"]["z=6"]["canonical"][
                    "uv_luminosity_erg_per_s_hz"
                ]
                dataset[0:1] = np.array([9.0e29])
            os.replace(foreign, spool_path)
            replaced_paths.append(spool_path)

    monkeypatch.setattr(
        hdf5_module,
        "_require_spool_snapshot",
        replace_after_copy,
    )
    _install_fake_science(monkeypatch, tmp_path, [])

    with pytest.raises(ValueError, match="sample spool path identity changed") as captured:
        run_uvlf_to_sample_shards(
            config,
            provenance,
            shard_directory,
            overwrite=True,
        )

    assert "sample spool path identity changed" in str(captured.value)
    assert require_count >= 3
    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    assert len(replaced_paths) == 1
    with h5py.File(replaced_paths[0], "r") as handle:
        value = handle["samples"]["z=6"]["canonical"][
            "uv_luminosity_erg_per_s_hz"
        ][0:1]
    np.testing.assert_array_equal(value, [9.0e29])
    for descriptor in set(captured_descriptors):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not tuple(shard_directory.glob("*.tmp"))
    assert not tuple(shard_directory.glob("*.backup"))


def test_finalize_in_place_spool_change_rolls_back_old_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import h5py
    import auroralf.io.hdf5 as hdf5_module

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    provenance = _provenance(config)
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    _, (path,) = run_uvlf_to_sample_shards(config, provenance, shard_directory)
    marker = path.with_name(path.name + ".complete")
    old_artifact = path.read_bytes()
    old_marker = marker.read_bytes()
    original_require = hdf5_module._require_spool_snapshot
    require_count = 0
    captured_descriptors: list[int] = []

    def mutate_after_copy(
        descriptor: int,
        spool_path: Path,
        owner: tuple[int, int],
        snapshot: object,
    ) -> None:
        nonlocal require_count
        require_count += 1
        captured_descriptors.append(descriptor)
        original_require(descriptor, spool_path, owner, snapshot)
        if require_count == 2:
            with h5py.File(spool_path, "r+") as handle:
                dataset = handle["samples"]["z=6"]["canonical"][
                    "uv_luminosity_erg_per_s_hz"
                ]
                dataset[0:1] = np.array([8.0e29])

    monkeypatch.setattr(
        hdf5_module,
        "_require_spool_snapshot",
        mutate_after_copy,
    )
    _install_fake_science(monkeypatch, tmp_path, [])

    with pytest.raises(
        ValueError,
        match="sample spool (file identity|content) changed",
    ):
        run_uvlf_to_sample_shards(
            config,
            provenance,
            shard_directory,
            overwrite=True,
        )

    assert require_count >= 3
    assert path.read_bytes() == old_artifact
    assert marker.read_bytes() == old_marker
    for descriptor in set(captured_descriptors):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
    assert not tuple(shard_directory.glob("*.tmp"))
    assert not tuple(shard_directory.glob("*.backup"))


def test_successful_sample_shard_run_closes_owned_spool_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    config = _config(
        tmp_path,
        mass_batch_size=1,
        modes=("canonical",),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    _install_fake_science(monkeypatch, tmp_path, [])
    original_abort = _HDF5SampleSink.abort
    closed_descriptors: list[int] = []

    def observe_close(self: _HDF5SampleSink) -> None:
        descriptor = self._descriptor
        original_abort(self)
        with pytest.raises(OSError):
            os.fstat(descriptor)
        closed_descriptors.append(descriptor)

    monkeypatch.setattr(_HDF5SampleSink, "abort", observe_close)

    run_uvlf_to_sample_shards(
        config,
        _provenance(config),
        shard_directory,
    )

    assert len(closed_descriptors) == 1
    assert not tuple(shard_directory.glob(".auroralf-samples-*.spool.h5"))
