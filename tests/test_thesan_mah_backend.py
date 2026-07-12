from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.mah.thesan import THESAN_MAH_CACHE_SCHEMA_VERSION
from auroralf.seeding import derive_pipeline_random_seeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"
V2_RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"
BUILD_CACHE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "data" / "build_thesan_mah_cache.py"
MANUSCRIPT_PLOT_SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "plot" / "plot_manuscript_mcbride_mah_illustration.py"
)


def _load_build_cache_module():
    spec = importlib.util.spec_from_file_location("build_thesan_mah_cache", BUILD_CACHE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_thesan_mah_cache"] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_thesan_cache(
    path: Path,
    *,
    n_halos: int = 6,
    include_negative_step: bool = False,
    unresolved_first_step: bool = False,
    cache_cosmology: Cosmology = Cosmology(),
) -> None:
    z_grid = np.array([12.0, 10.0, 8.0, 6.0], dtype=float)
    t_gyr_grid = np.array([0.37, 0.48, 0.64, 0.93], dtype=float)
    mass_ratio = np.empty((n_halos, z_grid.size), dtype=float)
    logm_final = np.linspace(9.92, 10.08, n_halos, dtype=float)

    for halo_index in range(n_halos):
        early_ratio = 0.10 + 0.01 * halo_index
        mass_ratio[halo_index] = np.array([early_ratio, 0.30 + 0.01 * halo_index, 0.62, 1.0], dtype=float)
    if include_negative_step:
        mass_ratio[0] = np.array([0.20, 0.45, 0.40, 1.0], dtype=float)
    resolved_mask = np.ones_like(mass_ratio, dtype=bool)
    if unresolved_first_step:
        mass_ratio[:, 0] = 1.0e-6
        resolved_mask[:, 0] = False

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = THESAN_MAH_CACHE_SCHEMA_VERSION
        handle.attrs["source_simulation"] = "Thesan-Dark-1"
        handle.attrs["source_tree"] = "LHaloTree"
        handle.attrs["snapshot"] = 95
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["time_unit"] = "Gyr"
        handle.attrs["redshift_unit"] = "dimensionless"
        handle.attrs["mass_ratio_unit"] = "dimensionless"
        handle.attrs["selection_description"] = "Synthetic z=6 central-halo fixture"
        handle.attrs["creator_version"] = "auroralf.test_thesan_cache.v1"
        handle.attrs["z_final"] = 6.0
        handle.attrs["hubble"] = cache_cosmology.h0_km_s_mpc / 100.0
        handle.attrs["omega_m"] = cache_cosmology.omega_m
        handle.attrs["omega_b"] = cache_cosmology.omega_b
        handle.create_dataset("z_grid", data=z_grid)
        handle.create_dataset("t_gyr_grid", data=t_gyr_grid)
        handle.create_dataset("mass_ratio", data=mass_ratio)
        handle.create_dataset("resolved_mask", data=resolved_mask)
        handle.create_dataset("logM_final", data=logm_final)
        handle.create_dataset("source_subhalo_id", data=np.arange(n_halos, dtype=np.int64))
        handle.create_dataset("source_group_index", data=np.arange(100, 100 + n_halos, dtype=np.int64))
        handle.create_dataset("source_tree_file", data=np.zeros(n_halos, dtype=np.int64))
        handle.create_dataset("source_tree_num", data=np.arange(n_halos, dtype=np.int64))
        handle.create_dataset("source_tree_index", data=np.arange(n_halos, dtype=np.int64))
        handle.create_dataset("source_snapshot", data=np.full(n_halos, 95, dtype=np.int64))
        handle.create_dataset(
            "source_file_identifier",
            data=np.asarray(
                [f"thesan://Thesan-Dark-1/tree-file-{index}" for index in range(n_halos)],
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )
        handle.create_dataset(
            "source_file_sha256",
            data=np.asarray(
                [f"{index + 1:064x}" for index in range(n_halos)],
                dtype=h5py.string_dtype(encoding="ascii", length=64),
            ),
        )


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("attr", "source_simulation"),
        ("attr", "source_tree"),
        ("attr", "snapshot"),
        ("attr", "mass_unit"),
        ("attr", "time_unit"),
        ("attr", "redshift_unit"),
        ("attr", "mass_ratio_unit"),
        ("attr", "selection_description"),
        ("attr", "creator_version"),
        ("dataset", "source_subhalo_id"),
        ("dataset", "source_group_index"),
        ("dataset", "source_tree_file"),
        ("dataset", "source_tree_num"),
        ("dataset", "source_tree_index"),
        ("dataset", "source_snapshot"),
        ("dataset", "source_file_identifier"),
        ("dataset", "source_file_sha256"),
    ],
)
def test_thesan_cache_rejects_missing_required_provenance(
    tmp_path: Path,
    kind: str,
    name: str,
) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / f"missing-{name}.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        if kind == "attr":
            del handle.attrs[name]
        else:
            del handle[name]
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(KeyError, match=name):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


@pytest.mark.parametrize(
    ("kind", "name", "invalid_value"),
    [
        ("attr", "source_simulation", "unknown"),
        ("attr", "source_tree", "unknown"),
        ("attr", "selection_description", "unknown"),
        ("attr", "creator_version", "unknown"),
        ("dataset", "source_subhalo_id", -1),
        ("dataset", "source_group_index", -1),
        ("dataset", "source_tree_file", -1),
        ("dataset", "source_tree_num", -1),
        ("dataset", "source_tree_index", -1),
        ("dataset", "source_snapshot", -1),
        ("dataset", "source_file_identifier", "unknown"),
        ("dataset", "source_file_sha256", "not-a-sha256"),
    ],
)
def test_thesan_cache_rejects_placeholder_or_invalid_provenance(
    tmp_path: Path,
    kind: str,
    name: str,
    invalid_value: str | int,
) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / f"invalid-{name}.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        if kind == "attr":
            handle.attrs[name] = invalid_value
        else:
            handle[name][0] = invalid_value
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match=name):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_cache_schema_is_bumped_for_strict_provenance_contract(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    assert thesan.THESAN_MAH_CACHE_SCHEMA_VERSION == "auroralf_thesan_mah_cache_v1"
    cache = tmp_path / "old-schema.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["schema_version"] = "auroralf_thesan_mah_cache_v0"
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="schema_version"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_manuscript_plot_uses_current_thesan_cache_schema_constant() -> None:
    source = MANUSCRIPT_PLOT_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "from auroralf.mah.thesan import THESAN_MAH_CACHE_SCHEMA_VERSION" in source
    assert '"auroralf_thesan_mah_cache_v0"' not in source


def test_thesan_cache_accepts_bytes_schema_attribute(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "bytes-schema.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["schema_version"] = np.bytes_(THESAN_MAH_CACHE_SCHEMA_VERSION)
    thesan._clear_thesan_mah_cache_for_tests()

    assert thesan.preload_thesan_mah_cache(
        tmp_path,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()


def test_thesan_cache_rejects_non_bool_resolved_mask(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "integer-resolved-mask.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        values = np.asarray(handle["resolved_mask"], dtype=np.int8)
        del handle["resolved_mask"]
        handle.create_dataset("resolved_mask", data=values)
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="resolved_mask.*bool"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_cache_rejects_non_unit_final_mass_ratio(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "bad-final-ratio.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle["mass_ratio"][:, -1] = 0.9
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="mass_ratio.*final.*1"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_cache_uses_composite_source_halo_identity(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "same-subhalo-different-tree.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle["source_subhalo_id"][1] = handle["source_subhalo_id"][0]
        handle["source_tree_file"][1] = 1
    thesan._clear_thesan_mah_cache_for_tests()

    assert thesan.preload_thesan_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()


def test_thesan_cache_rejects_duplicate_composite_source_halo_identity(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "duplicate-source-id.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        for name in ("source_snapshot", "source_tree_file", "source_tree_num", "source_tree_index"):
            handle[name][1] = handle[name][0]
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="source halo identities.*unique"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_cache_rejects_conflicting_checksum_for_source_identifier(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "conflicting-checksum.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        identifiers = handle["source_file_identifier"]
        identifiers[1] = identifiers[0]
        assert handle["source_file_sha256"][1] != handle["source_file_sha256"][0]
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="source_file_identifier.*multiple SHA-256"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_cache_numeric_attribute_error_names_field(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "bad-hubble.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["hubble"] = "not-numeric"
    thesan._clear_thesan_mah_cache_for_tests()

    with pytest.raises(ValueError, match="hubble.*numeric"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_thesan_preload_reads_hdf5_datasets_once_for_repeated_mass_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)
    thesan._clear_thesan_mah_cache_for_tests()
    real_read = thesan._read_required_dataset
    reads: list[str] = []

    def read_spy(handle: h5py.File, name: str) -> np.ndarray:
        reads.append(name)
        return real_read(handle, name)

    monkeypatch.setattr(thesan, "_read_required_dataset", read_spy)
    resolved = thesan.preload_thesan_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    )
    first_read_count = len(reads)
    assert resolved == cache.resolve()
    assert first_read_count > 0

    for seed in (1, 2):
        thesan.generate_thesan_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=resolved,
            min_candidates=1,
            random_seed=seed,
        )
    assert len(reads) == first_read_count
    _, loaded = thesan._load_thesan_cache(
        resolved,
        z_final=6.0,
        cosmology=Cosmology(),
    )
    for value in loaded.values():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable is False
            with pytest.raises(ValueError, match="WRITEABLE|writeable"):
                value.setflags(write=True)


def test_thesan_preload_reloads_after_same_path_atomic_replace_with_restored_mtime(
    tmp_path: Path,
) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "thesan_mah_z6.hdf5"
    replacement = tmp_path / "replacement.hdf5"
    _write_synthetic_thesan_cache(cache)
    thesan._clear_thesan_mah_cache_for_tests()
    thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())
    _, first = thesan._load_thesan_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    )
    first_value = float(first["mass_ratio"][0, 0])
    original_mtime_ns = cache.stat().st_mtime_ns

    _write_synthetic_thesan_cache(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["mass_ratio"][0, 0] = first_value + 0.05
    assert replacement.stat().st_size == cache.stat().st_size
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, cache)
    os.utime(cache, ns=(original_mtime_ns, original_mtime_ns))

    thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())
    _, second = thesan._load_thesan_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    )

    assert second is not first
    assert float(second["mass_ratio"][0, 0]) == pytest.approx(first_value + 0.05)


def test_thesan_preload_rejects_atomic_replacement_during_open_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.thesan as thesan

    cache = tmp_path / "thesan_mah_z6.hdf5"
    replacement = tmp_path / "replacement.hdf5"
    _write_synthetic_thesan_cache(cache)
    _write_synthetic_thesan_cache(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["mass_ratio"][0, 0] = 0.25
    thesan._clear_thesan_mah_cache_for_tests()
    real_read = thesan._read_required_dataset
    replaced = False

    def replace_during_read(handle: h5py.File, name: str) -> np.ndarray:
        nonlocal replaced
        values = real_read(handle, name)
        if not replaced:
            os.replace(replacement, cache)
            replaced = True
        return values

    monkeypatch.setattr(thesan, "_read_required_dataset", replace_during_read)
    with pytest.raises(RuntimeError, match="changed during preload"):
        thesan.preload_thesan_mah_cache(cache, z_final=6.0, cosmology=Cosmology())

    assert thesan._THESAN_MAH_CACHE == {}
    assert thesan.preload_thesan_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()
    _, loaded = thesan._load_thesan_cache(cache, z_final=6.0, cosmology=Cosmology())
    assert float(loaded["mass_ratio"][0, 0]) == pytest.approx(0.25)


def test_thesan_backend_outputs_standard_halo_history_result(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    result = generate_thesan_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    assert result.metadata["mah_backend"] == "thesan"
    assert result.metadata["source_simulation"] == "Thesan-Dark-1"
    assert result.metadata["source_tree"] == "LHaloTree"
    assert result.metadata["snapshot"] == 95
    assert result.metadata["mass_unit"] == "Msun"
    assert result.metadata["time_unit"] == "Gyr"
    assert result.metadata["redshift_unit"] == "dimensionless"
    assert result.metadata["mass_ratio_unit"] == "dimensionless"
    assert result.metadata["selection_description"] == "Synthetic z=6 central-halo fixture"
    assert result.metadata["creator_version"] == "auroralf.test_thesan_cache.v1"
    assert len(result.metadata["source_file_identifier"]) == 6
    assert len(result.metadata["source_file_sha256"]) == 6
    assert result.metadata["candidate_count"] == 6
    assert result.metadata["negative_dmhdt_clip_fraction"] == pytest.approx(0.0)

    tracks = result.tracks
    mh_grid = np.asarray(tracks["Mh"], dtype=float).reshape(4, 4)
    dmhdt_grid = np.asarray(tracks["dMh_dt_sfr"], dtype=float).reshape(4, 4)
    time_grid = np.asarray(tracks["t_gyr"], dtype=float).reshape(4, 4)

    np.testing.assert_allclose(mh_grid[:, -1], 1.0e10)
    assert np.all(np.diff(time_grid, axis=1) > 0.0)
    assert np.nanmedian(dmhdt_grid[:, 1:]) > 1.0e9
    assert set(tracks) >= {
        "halo_id",
        "step",
        "z",
        "t_gyr",
        "dt_gyr",
        "Mh",
        "dMh_dt_raw",
        "dMh_dt_sfr",
        "dMh_dt_clipped",
        "active_flag",
        "termination_flag",
    }


def test_thesan_backend_requires_cosmology(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    with pytest.raises(TypeError, match="cosmology"):
        generate_thesan_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cache_path=cache,
        )


@pytest.mark.parametrize("field", ["hubble", "omega_m", "omega_b"])
def test_thesan_backend_rejects_cache_cosmology_mismatch(tmp_path: Path, field: str) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs[field] = float(handle.attrs[field]) + 0.01

    from auroralf.mah.thesan import generate_thesan_halo_histories

    with pytest.raises(ValueError, match=field):
        generate_thesan_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            min_candidates=1,
        )


def test_thesan_backend_requires_cache_cosmology_provenance(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)
    with h5py.File(cache, "r+") as handle:
        del handle.attrs["omega_m"]

    from auroralf.mah.thesan import generate_thesan_halo_histories

    with pytest.raises(KeyError, match="omega_m"):
        generate_thesan_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            min_candidates=1,
        )


def test_thesan_backend_fails_when_candidate_count_is_too_small(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache, n_halos=3)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    with pytest.raises(ValueError, match="THESAN MAH candidate count"):
        generate_thesan_halo_histories(
            n_tracks=2,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            mass_bin_width_dex=0.15,
            min_candidates=5,
            random_seed=7,
        )


def test_thesan_backend_clips_negative_accretion_and_records_fraction(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache, include_negative_step=True)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    result = generate_thesan_halo_histories(
        n_tracks=8,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=1,
    )

    raw = np.asarray(result.tracks["dMh_dt_raw"], dtype=float)
    effective = np.asarray(result.tracks["dMh_dt_sfr"], dtype=float)
    clipped = np.asarray(result.tracks["dMh_dt_clipped"], dtype=bool)
    assert np.any(raw < 0.0)
    assert np.all(effective >= 0.0)
    np.testing.assert_array_equal(effective, np.maximum(raw, 0.0))
    np.testing.assert_array_equal(clipped, raw < 0.0)
    assert result.metadata["negative_dmhdt_clip_count"] == int(np.count_nonzero(clipped))
    assert result.metadata["negative_dmhdt_clip_fraction"] == pytest.approx(
        result.metadata["negative_dmhdt_clip_count"] / result.metadata["negative_dmhdt_total_count"]
    )


def _make_thesan_builder_case(tmp_path: Path):
    module = _load_build_cache_module()
    root = tmp_path / "thesan-dark-1"
    tree_path = root / "postprocessing" / "trees" / "LHaloTree" / "trees_sf1_190.0.hdf5"
    tree_path.parent.mkdir(parents=True)
    with h5py.File(tree_path, "w") as handle:
        header = handle.create_group("Header")
        header.create_dataset("Redshifts", data=np.array([8.0, 6.0], dtype=float))
        tree = handle.create_group("Tree0")
        tree.create_dataset("SnapNum", data=np.array([1, 0], dtype=np.int64))
        tree.create_dataset("FirstProgenitor", data=np.array([1, -1], dtype=np.int64))
        tree.create_dataset("Group_M_Crit200", data=np.array([1.0, 0.4], dtype=float))
        tree.create_dataset("FirstHaloInFOFGroup", data=np.array([0, 1], dtype=np.int64))
        tree.create_dataset("SubhaloGrNr", data=np.array([5, 5], dtype=np.int64))

    selected_path = tmp_path / "selected_halos.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "snapshot",
                "redshift",
                "group_index",
                "source_subhalo_id",
                "logM_final",
                "tree_file",
                "tree_num",
                "tree_index",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "snapshot": 1,
                "redshift": 6.0,
                "group_index": 5,
                "source_subhalo_id": 42,
                "logM_final": 10.169,
                "tree_file": 0,
                "tree_num": 0,
                "tree_index": 0,
            }
        )

    output = tmp_path / "built-cache.hdf5"
    cache_cosmology = Cosmology()
    args = module.argparse.Namespace(
        root=str(root),
        selected_halos=str(selected_path),
        snapshot=1,
        tree_file="0",
        output=str(output),
        mass_field="Group_M_Crit200",
        branch_start="first_fof",
        hubble=cache_cosmology.h0_km_s_mpc / 100.0,
        omega_m=cache_cosmology.omega_m,
        omega_b=cache_cosmology.omega_b,
        min_logm=None,
        max_logm=None,
        max_halos=None,
        random_seed=42,
        unresolved_mass_ratio_fill=1.0e-6,
        force=False,
    )
    return module, args, selected_path, tree_path, output, cache_cosmology


def test_official_thesan_builder_writes_reader_accepted_provenance(tmp_path: Path) -> None:
    import auroralf.mah.thesan as thesan

    module, args, selected_path, tree_path, output, cache_cosmology = (
        _make_thesan_builder_case(tmp_path)
    )
    assert module._build_cache(args) == output.resolve()

    thesan._clear_thesan_mah_cache_for_tests()
    assert thesan.preload_thesan_mah_cache(
        output,
        z_final=6.0,
        cosmology=cache_cosmology,
    ) == output.resolve()
    with h5py.File(output, "r") as handle:
        identifiers = handle["source_file_identifier"].asstr()[...].tolist()
        checksums = handle["source_file_sha256"].asstr()[...].tolist()
        assert identifiers == [str(selected_path.resolve()), str(tree_path.resolve())]
        assert checksums == [
            hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            hashlib.sha256(tree_path.read_bytes()).hexdigest(),
        ]
        assert handle["resolved_mask"].dtype.kind == "b"
    published_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        module._build_cache(args)
    assert output.read_bytes() == published_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_thesan_builder_rejects_tree_change_after_science_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args, _selected_path, tree_path, output, _cosmology = (
        _make_thesan_builder_case(tmp_path)
    )
    original = module.verify_source_file_provenance
    changed = False

    def change_tree_then_verify(provenance):
        nonlocal changed
        if provenance.identifier == str(tree_path.resolve()) and not changed:
            tree_path.write_bytes(tree_path.read_bytes() + b"changed-after-science-read")
            changed = True
        return original(provenance)

    monkeypatch.setattr(module, "verify_source_file_provenance", change_tree_then_verify)

    with pytest.raises(ValueError, match="source file changed"):
        module._build_cache(args)

    assert changed is True
    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_thesan_builder_rejects_selected_table_change_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_build_cache_module()
    selected_path = tmp_path / "selected_halos.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "snapshot",
                "redshift",
                "group_index",
                "source_subhalo_id",
                "logM_final",
                "tree_file",
                "tree_num",
                "tree_index",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "snapshot": 1,
                "redshift": 6.0,
                "group_index": 5,
                "source_subhalo_id": 42,
                "logM_final": 10.169,
                "tree_file": 0,
                "tree_num": 0,
                "tree_index": 0,
            }
        )

    original = module._read_selected_rows

    def read_then_change(*args, **kwargs):
        rows = original(*args, **kwargs)
        selected_path.write_text(
            selected_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return rows

    monkeypatch.setattr(module, "_read_selected_rows", read_then_change)
    args = module.argparse.Namespace(
        root=str(tmp_path / "missing-tree-root"),
        selected_halos=str(selected_path),
        snapshot=1,
        tree_file="0",
        output=str(tmp_path / "must-not-exist.hdf5"),
        mass_field="Group_M_Crit200",
        branch_start="first_fof",
        hubble=0.6774,
        omega_m=0.3089,
        omega_b=0.0486,
        min_logm=None,
        max_logm=None,
        max_halos=None,
        random_seed=42,
        unresolved_mass_ratio_fill=1.0e-6,
        force=False,
    )

    with pytest.raises(ValueError, match="source file changed"):
        module._build_cache(args)


def test_thesan_backend_runs_through_uv_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=3,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=4,
        mah_backend="thesan",
        thesan_mah_cache_path=cache,
        thesan_mass_bin_width_dex=0.15,
        thesan_min_candidates=5,
        random_seeds=derive_pipeline_random_seeds(3, redshift=6.0, mass_index=0),
        workers=1,
    )

    assert result.metadata["mah_backend"] == "thesan"
    assert result.metadata["thesan_mah_cache_path"] == str(cache.resolve())
    assert result.uv_luminosities.shape == (3,)
    assert np.all(np.isfinite(result.uv_luminosities))


def test_thesan_backend_is_available_through_hmf_sampling(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf

    result = sample_uvlf_from_hmf(
        z_obs=6.0,
        cosmology=Cosmology(),
        N_mass=1,
        n_tracks=2,
        base_seed=2,
        logM_min=9.99,
        logM_max=10.01,
        z_start_max=12.0,
        n_grid=4,
        mah_backend="thesan",
        thesan_mah_cache_path=cache,
        thesan_mass_bin_width_dex=0.15,
        thesan_min_candidates=5,
        pipeline_workers=1,
    )

    assert result.metadata["mah_backend"] == "thesan"
    assert result.metadata["thesan_mah_cache_path"] == str(cache.resolve())
    assert result.samples["luminosity"].shape == (2,)


def test_v2_run_script_requires_thesan_backend_configuration_in_toml() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(V2_RUN_SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in completed.stdout
    assert "--thesan-mah-cache" not in completed.stdout
