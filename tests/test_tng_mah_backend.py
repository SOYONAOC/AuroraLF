from __future__ import annotations

import importlib.util
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from auroralf.file_version import capture_source_file_provenance
from auroralf.mah import Cosmology
from auroralf.mah.tng import TNG_MAH_CACHE_SCHEMA_VERSION
from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.ssp import DEFAULT_POPIII_UV_SSP_FILE
from auroralf.uvlf.imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"
V2_RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"
BUILD_CACHE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "data" / "build_tng_mah_cache.py"


def _load_build_cache_module():
    spec = importlib.util.spec_from_file_location("build_tng_mah_cache", BUILD_CACHE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_tng_mah_cache"] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_tng_cache(
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
        handle.attrs["schema_version"] = TNG_MAH_CACHE_SCHEMA_VERSION
        handle.attrs["source_simulation"] = "TNG100-1-Dark"
        handle.attrs["snapshot"] = 99
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["time_unit"] = "Gyr"
        handle.attrs["redshift_unit"] = "dimensionless"
        handle.attrs["mass_ratio_unit"] = "dimensionless"
        handle.attrs["selection_description"] = "Synthetic z=6 central-halo fixture"
        handle.attrs["creator_version"] = "auroralf.test_tng_cache.v1"
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
        handle.create_dataset("source_snapshot", data=np.full(n_halos, 99, dtype=np.int64))
        handle.create_dataset(
            "source_file_identifier",
            data=np.asarray(
                [f"tng://TNG100-1-Dark/snapshot-99/subhalo-{index}" for index in range(n_halos)],
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
        ("attr", "snapshot"),
        ("attr", "mass_unit"),
        ("attr", "time_unit"),
        ("attr", "redshift_unit"),
        ("attr", "mass_ratio_unit"),
        ("attr", "selection_description"),
        ("attr", "creator_version"),
        ("dataset", "source_subhalo_id"),
        ("dataset", "source_snapshot"),
        ("dataset", "source_file_identifier"),
        ("dataset", "source_file_sha256"),
    ],
)
def test_tng_cache_rejects_missing_required_provenance(
    tmp_path: Path,
    kind: str,
    name: str,
) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / f"missing-{name}.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        if kind == "attr":
            del handle.attrs[name]
        else:
            del handle[name]
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(KeyError, match=name):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


@pytest.mark.parametrize(
    ("kind", "name", "invalid_value"),
    [
        ("attr", "source_simulation", "unknown"),
        ("attr", "selection_description", "unknown"),
        ("attr", "creator_version", "unknown"),
        ("dataset", "source_subhalo_id", -1),
        ("dataset", "source_snapshot", -1),
        ("dataset", "source_file_identifier", "unknown"),
        ("dataset", "source_file_sha256", "not-a-sha256"),
    ],
)
def test_tng_cache_rejects_placeholder_or_invalid_provenance(
    tmp_path: Path,
    kind: str,
    name: str,
    invalid_value: str | int,
) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / f"invalid-{name}.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        if kind == "attr":
            handle.attrs[name] = invalid_value
        else:
            handle[name][0] = invalid_value
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match=name):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_schema_is_bumped_for_strict_provenance_contract(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    assert tng.TNG_MAH_CACHE_SCHEMA_VERSION == "auroralf_tng_mah_cache_v2"
    cache = tmp_path / "old-schema.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["schema_version"] = "auroralf_tng_mah_cache_v1"
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="schema_version"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_accepts_bytes_schema_attribute(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "bytes-schema.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["schema_version"] = np.bytes_(TNG_MAH_CACHE_SCHEMA_VERSION)
    tng._clear_tng_mah_cache_for_tests()

    assert tng.preload_tng_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()


def test_tng_cache_rejects_non_bool_resolved_mask(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "integer-resolved-mask.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        values = np.asarray(handle["resolved_mask"], dtype=np.int8)
        del handle["resolved_mask"]
        handle.create_dataset("resolved_mask", data=values)
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="resolved_mask.*bool"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_rejects_non_unit_final_mass_ratio(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "bad-final-ratio.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle["mass_ratio"][:, -1] = 0.9
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="mass_ratio.*final.*1"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_rejects_duplicate_source_halo_identity(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "duplicate-source-id.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle["source_subhalo_id"][1] = handle["source_subhalo_id"][0]
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="source halo identities.*unique"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_rejects_conflicting_checksum_for_source_identifier(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "conflicting-checksum.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        identifiers = handle["source_file_identifier"]
        identifiers[1] = identifiers[0]
        assert handle["source_file_sha256"][1] != handle["source_file_sha256"][0]
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="source_file_identifier.*multiple SHA-256"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def test_tng_cache_numeric_attribute_error_names_field(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "bad-hubble.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs["hubble"] = "not-numeric"
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(ValueError, match="hubble.*numeric"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())


def _write_raw_mpb(path: Path, *, snap: np.ndarray, mass: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("SnapNum", data=snap.astype(np.int64))
        handle.create_dataset("Group_M_Crit200", data=mass.astype(float))


def _write_tng_cache_from_provenance(
    module,
    output: Path,
    arrays: dict[str, np.ndarray],
    source_provenance: list[object],
    *,
    overwrite: bool = False,
) -> None:
    cache_cosmology = Cosmology()
    module._write_cache_file(
        output,
        arrays,
        source_simulation="TNG100-1-Dark",
        snapshot=2,
        z_final=6.0,
        mass_field="Group_M_Crit200",
        hubble=cache_cosmology.h0_km_s_mpc / 100.0,
        omega_m=cache_cosmology.omega_m,
        omega_b=cache_cosmology.omega_b,
        drop_invalid_mpb=False,
        snapshot_grid="common",
        missing_mass_ratio_floor=1.0e-6,
        selection_description="Explicit test selection of subhalos 10 and 11",
        source_provenance=source_provenance,
        overwrite=overwrite,
    )


def test_tng_preload_reads_hdf5_datasets_once_for_repeated_mass_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)
    tng._clear_tng_mah_cache_for_tests()
    real_read = tng._read_required_dataset
    reads: list[str] = []

    def read_spy(handle: h5py.File, name: str) -> np.ndarray:
        reads.append(name)
        return real_read(handle, name)

    monkeypatch.setattr(tng, "_read_required_dataset", read_spy)
    resolved = tng.preload_tng_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    )
    first_read_count = len(reads)
    assert resolved == cache.resolve()
    assert first_read_count > 0

    for seed in (1, 2):
        tng.generate_tng_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=resolved,
            min_candidates=1,
            random_seed=seed,
        )
    assert len(reads) == first_read_count
    _, loaded = tng._load_tng_cache(
        resolved,
        z_final=6.0,
        cosmology=Cosmology(),
    )
    for value in loaded.values():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable is False
            with pytest.raises(ValueError, match="WRITEABLE|writeable"):
                value.setflags(write=True)


def test_tng_failed_preload_is_not_cached(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        del handle["mass_ratio"]
    tng._clear_tng_mah_cache_for_tests()

    with pytest.raises(KeyError, match="mass_ratio"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())

    with h5py.File(cache, "r+") as handle:
        handle.create_dataset(
            "mass_ratio",
            data=np.tile(np.array([0.1, 0.3, 0.62, 1.0]), (6, 1)),
        )
    assert tng.preload_tng_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()


def test_tng_preload_reloads_after_same_path_atomic_replace_with_restored_mtime(
    tmp_path: Path,
) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "tng_mah_z6.hdf5"
    replacement = tmp_path / "replacement.hdf5"
    _write_synthetic_tng_cache(cache)
    tng._clear_tng_mah_cache_for_tests()
    tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())
    _, first = tng._load_tng_cache(cache, z_final=6.0, cosmology=Cosmology())
    first_value = float(first["mass_ratio"][0, 0])
    original_mtime_ns = cache.stat().st_mtime_ns

    _write_synthetic_tng_cache(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["mass_ratio"][0, 0] = first_value + 0.05
    assert replacement.stat().st_size == cache.stat().st_size
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, cache)
    os.utime(cache, ns=(original_mtime_ns, original_mtime_ns))

    tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())
    _, second = tng._load_tng_cache(cache, z_final=6.0, cosmology=Cosmology())

    assert second is not first
    assert float(second["mass_ratio"][0, 0]) == pytest.approx(first_value + 0.05)


def test_tng_preload_rejects_atomic_replacement_during_open_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.tng as tng

    cache = tmp_path / "tng_mah_z6.hdf5"
    replacement = tmp_path / "replacement.hdf5"
    _write_synthetic_tng_cache(cache)
    _write_synthetic_tng_cache(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["mass_ratio"][0, 0] = 0.25
    tng._clear_tng_mah_cache_for_tests()
    real_read = tng._read_required_dataset
    replaced = False

    def replace_during_read(handle: h5py.File, name: str) -> np.ndarray:
        nonlocal replaced
        values = real_read(handle, name)
        if not replaced:
            os.replace(replacement, cache)
            replaced = True
        return values

    monkeypatch.setattr(tng, "_read_required_dataset", replace_during_read)
    with pytest.raises(RuntimeError, match="changed during preload"):
        tng.preload_tng_mah_cache(cache, z_final=6.0, cosmology=Cosmology())

    assert tng._TNG_MAH_CACHE == {}
    assert tng.preload_tng_mah_cache(
        cache,
        z_final=6.0,
        cosmology=Cosmology(),
    ) == cache.resolve()
    _, loaded = tng._load_tng_cache(cache, z_final=6.0, cosmology=Cosmology())
    assert float(loaded["mass_ratio"][0, 0]) == pytest.approx(0.25)


def test_worker_context_preloads_resolved_tng_path_for_every_redshift(
    tmp_path: Path,
) -> None:
    import auroralf.mah.tng as tng
    import auroralf.uvlf.runner as runner

    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)
    tng._clear_tng_mah_cache_for_tests()
    config = UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="tng-worker-context",
        redshifts=(6.0,),
        base_seed=1,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(
            backend="tng",
            tng_cache_path=cache.resolve(),
            n_time_steps=4,
            z_start_max=12.0,
        ),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=StellarPopulationConfig(
            imf_modes=("canonical",),
            canonical_ssp_path=(PROJECT_ROOT / DEFAULT_CANONICAL_SSP_FILE).resolve(),
            topheavy_ssp_path=(
                PROJECT_ROOT / DEFAULT_MILD_TOPHEAVY_SSP_FILE
            ).resolve(),
            popiii_ssp_path=(PROJECT_ROOT / DEFAULT_POPIII_UV_SSP_FILE).resolve(),
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=False,
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=1,
            n_tracks_per_halo_mass=1,
            workers=1,
        ),
        output=OutputConfig((tmp_path / "not-written.h5").resolve()),
    )

    context = runner._build_worker_context(config)

    assert context.resolved_simulation_cache_paths == ((6.0, cache.resolve()),)
    assert context.simulation_cache_path_for(6.0) == cache.resolve()


def test_tng_backend_outputs_standard_halo_history_result(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    assert result.metadata["mah_backend"] == "tng"
    assert result.metadata["source_simulation"] == "TNG100-1-Dark"
    assert result.metadata["snapshot"] == 99
    assert result.metadata["mass_unit"] == "Msun"
    assert result.metadata["time_unit"] == "Gyr"
    assert result.metadata["redshift_unit"] == "dimensionless"
    assert result.metadata["mass_ratio_unit"] == "dimensionless"
    assert result.metadata["selection_description"] == "Synthetic z=6 central-halo fixture"
    assert result.metadata["creator_version"] == "auroralf.test_tng_cache.v1"
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


def test_tng_backend_requires_cosmology(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

    from auroralf.mah.tng import generate_tng_halo_histories

    with pytest.raises(TypeError, match="cosmology"):
        generate_tng_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cache_path=cache,
        )


@pytest.mark.parametrize("field", ["hubble", "omega_m", "omega_b"])
def test_tng_backend_rejects_cache_cosmology_mismatch(tmp_path: Path, field: str) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        handle.attrs[field] = float(handle.attrs[field]) + 0.01

    from auroralf.mah.tng import generate_tng_halo_histories

    with pytest.raises(ValueError, match=field):
        generate_tng_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            min_candidates=1,
        )


def test_tng_backend_requires_cache_cosmology_provenance(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)
    with h5py.File(cache, "r+") as handle:
        del handle.attrs["omega_b"]

    from auroralf.mah.tng import generate_tng_halo_histories

    with pytest.raises(KeyError, match="omega_b"):
        generate_tng_halo_histories(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            min_candidates=1,
        )


def test_tng_backend_can_regrid_selected_shapes_to_uniform_time_grid(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, unresolved_first_step=True)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        cache_path=cache,
        z_start_max=12.0,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
        time_grid_mode="uniform_in_t",
        target_n_grid=12,
    )

    tracks = result.tracks
    mh_grid = np.asarray(tracks["Mh"], dtype=float).reshape(4, 12)
    dmhdt_grid = np.asarray(tracks["dMh_dt_sfr"], dtype=float).reshape(4, 12)
    time_grid = np.asarray(tracks["t_gyr"], dtype=float).reshape(4, 12)
    z_grid = np.asarray(tracks["z"], dtype=float).reshape(4, 12)
    active_grid = np.asarray(tracks["active_flag"], dtype=bool).reshape(4, 12)

    assert result.metadata["time_grid_mode"] == "tng_uniform_in_t"
    assert result.metadata["grid_size"] == 12
    np.testing.assert_allclose(mh_grid[:, -1], 1.0e10)
    dt_grid = np.diff(time_grid, axis=1)
    np.testing.assert_allclose(dt_grid, np.repeat(dt_grid[:, :1], dt_grid.shape[1], axis=1))
    assert np.all(np.diff(z_grid, axis=1) < 0.0)
    assert np.all(active_grid[:, -1])
    assert np.all(~active_grid[:, 0])
    assert np.all(dmhdt_grid >= 0.0)


def test_tng_backend_fails_when_candidate_count_is_too_small(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, n_halos=3)

    from auroralf.mah.tng import generate_tng_halo_histories

    with pytest.raises(ValueError, match="TNG MAH candidate count"):
        generate_tng_halo_histories(
            n_tracks=2,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            cache_path=cache,
            mass_bin_width_dex=0.15,
            min_candidates=5,
            random_seed=7,
        )


def test_tng_backend_clips_negative_accretion_and_records_fraction(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, include_negative_step=True)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
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
    assert result.metadata["negative_dmhdt_clip_fraction"] > 0.0


def test_tng_backend_marks_unresolved_steps_and_zeros_boundary_dmhdt(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, unresolved_first_step=True)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    dmhdt_grid = np.asarray(result.tracks["dMh_dt_raw"], dtype=float).reshape(4, 4)
    active_grid = np.asarray(result.tracks["active_flag"], dtype=bool).reshape(4, 4)

    assert np.all(~active_grid[:, 0])
    assert np.all(active_grid[:, 1:])
    np.testing.assert_array_equal(dmhdt_grid[:, 0], np.zeros(4))
    np.testing.assert_array_equal(dmhdt_grid[:, 1], np.zeros(4))
    assert result.metadata["unresolved_step_fraction"] == pytest.approx(0.25)


def test_build_cache_can_explicitly_drop_invalid_mpb(tmp_path: Path) -> None:
    module = _load_build_cache_module()
    valid_path = tmp_path / "valid.hdf5"
    invalid_path = tmp_path / "invalid.hdf5"
    _write_raw_mpb(valid_path, snap=np.array([1, 2]), mass=np.array([0.5, 1.0]))
    _write_raw_mpb(invalid_path, snap=np.array([2]), mass=np.array([1.0]))

    cosmology = module.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)
    with pytest.raises(ValueError, match="fewer than two positive finite mass samples"):
        module._build_cache_arrays(
            raw_paths=[valid_path, invalid_path],
            source_subhalo_ids=np.array([10, 11], dtype=np.int64),
            snapshot=2,
            snapshot_redshift={1: 8.0, 2: 6.0},
            mass_field="Group_M_Crit200",
            hubble=1.0,
            cosmology=cosmology,
            drop_invalid_mpb=False,
        )

    arrays = module._build_cache_arrays(
        raw_paths=[valid_path, invalid_path],
        source_subhalo_ids=np.array([10, 11], dtype=np.int64),
        snapshot=2,
        snapshot_redshift={1: 8.0, 2: 6.0},
        mass_field="Group_M_Crit200",
        hubble=1.0,
        cosmology=cosmology,
        drop_invalid_mpb=True,
    )

    np.testing.assert_array_equal(arrays["source_subhalo_id"], np.array([10], dtype=np.int64))
    np.testing.assert_array_equal(arrays["dropped_source_subhalo_id"], np.array([11], dtype=np.int64))
    assert arrays["mass_ratio"].shape == (1, 2)
    assert "fewer than two positive finite mass samples" in str(arrays["dropped_reason"][0])


def test_build_cache_union_grid_fills_unresolved_prehistory(tmp_path: Path) -> None:
    module = _load_build_cache_module()
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_raw_mpb(first_path, snap=np.array([1, 2]), mass=np.array([0.5, 1.0]))
    _write_raw_mpb(second_path, snap=np.array([0, 2]), mass=np.array([0.2, 2.0]))

    arrays = module._build_cache_arrays(
        raw_paths=[first_path, second_path],
        source_subhalo_ids=np.array([10, 11], dtype=np.int64),
        snapshot=2,
        snapshot_redshift={0: 10.0, 1: 8.0, 2: 6.0},
        mass_field="Group_M_Crit200",
        hubble=1.0,
        cosmology=module.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486),
        snapshot_grid="union",
        missing_mass_ratio_floor=1.0e-6,
    )

    np.testing.assert_array_equal(arrays["snap_grid"], np.array([0, 1, 2], dtype=np.int64))
    assert arrays["mass_ratio"].shape == (2, 3)
    assert np.all(arrays["mass_ratio"] > 0.0)
    np.testing.assert_allclose(arrays["mass_ratio"][:, -1], 1.0)
    np.testing.assert_array_equal(
        arrays["resolved_mask"],
        np.array([[False, True, True], [True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(arrays["filled_snap_count"], np.array([1, 1], dtype=np.int64))


def test_official_tng_builder_writes_reader_accepted_provenance(tmp_path: Path) -> None:
    import auroralf.mah.tng as tng

    module = _load_build_cache_module()
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_raw_mpb(first_path, snap=np.array([1, 2]), mass=np.array([0.5, 1.0]))
    _write_raw_mpb(second_path, snap=np.array([1, 2]), mass=np.array([0.6, 1.2]))
    arrays = module._build_cache_arrays(
        raw_paths=[first_path, second_path],
        source_subhalo_ids=np.array([10, 11], dtype=np.int64),
        snapshot=2,
        snapshot_redshift={1: 8.0, 2: 6.0},
        mass_field="Group_M_Crit200",
        hubble=1.0,
        cosmology=module.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486),
    )
    cache_cosmology = Cosmology()
    output = tmp_path / "built-cache.hdf5"
    _write_tng_cache_from_provenance(
        module,
        output,
        arrays,
        [capture_source_file_provenance(first_path), capture_source_file_provenance(second_path)],
    )

    tng._clear_tng_mah_cache_for_tests()
    assert tng.preload_tng_mah_cache(
        output,
        z_final=6.0,
        cosmology=cache_cosmology,
    ) == output.resolve()
    with h5py.File(output, "r") as handle:
        identifiers = handle["source_file_identifier"].asstr()[...].tolist()
        checksums = handle["source_file_sha256"].asstr()[...].tolist()
        assert identifiers == [str(first_path.resolve()), str(second_path.resolve())]
        assert checksums == [
            hashlib.sha256(first_path.read_bytes()).hexdigest(),
            hashlib.sha256(second_path.read_bytes()).hexdigest(),
        ]
        assert handle["resolved_mask"].dtype.kind == "b"
    published_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        _write_tng_cache_from_provenance(
            module,
            output,
            arrays,
            [capture_source_file_provenance(first_path), capture_source_file_provenance(second_path)],
        )
    assert output.read_bytes() == published_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_tng_builder_rejects_source_change_after_science_read(tmp_path: Path) -> None:
    module = _load_build_cache_module()
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_raw_mpb(first_path, snap=np.array([1, 2]), mass=np.array([0.5, 1.0]))
    _write_raw_mpb(second_path, snap=np.array([1, 2]), mass=np.array([0.6, 1.2]))
    provenance = [
        capture_source_file_provenance(first_path),
        capture_source_file_provenance(second_path),
    ]
    arrays = module._build_cache_arrays(
        raw_paths=[first_path, second_path],
        source_subhalo_ids=np.array([10, 11], dtype=np.int64),
        snapshot=2,
        snapshot_redshift={1: 8.0, 2: 6.0},
        mass_field="Group_M_Crit200",
        hubble=1.0,
        cosmology=module.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486),
    )
    _write_raw_mpb(first_path, snap=np.array([1, 2]), mass=np.array([0.9, 1.0]))
    output = tmp_path / "must-not-exist.hdf5"

    with pytest.raises(ValueError, match="source file changed"):
        _write_tng_cache_from_provenance(module, output, arrays, provenance)

    assert not output.exists()


def test_tng_builder_rejects_one_source_path_for_multiple_halo_rows(tmp_path: Path) -> None:
    module = _load_build_cache_module()
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _write_raw_mpb(first_path, snap=np.array([1, 2]), mass=np.array([0.5, 1.0]))
    _write_raw_mpb(second_path, snap=np.array([1, 2]), mass=np.array([0.6, 1.2]))
    arrays = module._build_cache_arrays(
        raw_paths=[first_path, second_path],
        source_subhalo_ids=np.array([10, 11], dtype=np.int64),
        snapshot=2,
        snapshot_redshift={1: 8.0, 2: 6.0},
        mass_field="Group_M_Crit200",
        hubble=1.0,
        cosmology=module.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486),
    )
    duplicate = capture_source_file_provenance(first_path)

    with pytest.raises(ValueError, match="unique source file"):
        _write_tng_cache_from_provenance(
            module,
            tmp_path / "must-not-exist.hdf5",
            arrays,
            [duplicate, duplicate],
        )


def test_tng_backend_runs_through_uv_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=3,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=4,
        mah_backend="tng",
        tng_mah_cache_path=cache,
        tng_mass_bin_width_dex=0.15,
        tng_min_candidates=5,
        random_seeds=derive_pipeline_random_seeds(3, redshift=6.0, mass_index=0),
        workers=1,
    )

    assert result.metadata["mah_backend"] == "tng"
    assert result.metadata["tng_mah_cache_path"] == str(cache.resolve())
    assert result.uv_luminosities.shape == (3,)
    assert np.all(np.isfinite(result.uv_luminosities))


def test_tng_backend_uniform_time_grid_runs_through_uv_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, unresolved_first_step=True)

    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=3,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=12,
        mah_backend="tng",
        tng_mah_cache_path=cache,
        tng_mass_bin_width_dex=0.15,
        tng_min_candidates=5,
        tng_time_grid_mode="uniform_in_t",
        random_seeds=derive_pipeline_random_seeds(3, redshift=6.0, mass_index=0),
        workers=1,
    )

    assert result.metadata["mah_backend"] == "tng"
    assert result.metadata["time_grid_mode"] == "tng_uniform_in_t"
    assert result.metadata["steps_per_halo"] == 12
    assert result.uv_luminosities.shape == (3,)
    assert np.all(np.isfinite(result.uv_luminosities))


def test_mcbride_backend_remains_default_in_uv_pipeline() -> None:
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=8.0,
        n_grid=4,
        random_seeds=derive_pipeline_random_seeds(3, redshift=6.0, mass_index=0),
        workers=1,
    )

    assert result.metadata["mah_backend"] == "mcbride"
    assert result.metadata["sampler"] == "mcbride"
    assert result.histories.metadata["sampler"] == "mcbride"


def test_tng_backend_is_available_through_hmf_sampling(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

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
        mah_backend="tng",
        tng_mah_cache_path=cache,
        tng_mass_bin_width_dex=0.15,
        tng_min_candidates=5,
        pipeline_workers=1,
    )

    assert result.metadata["mah_backend"] == "tng"
    assert result.metadata["tng_mah_cache_path"] == str(cache.resolve())
    assert result.samples["luminosity"].shape == (2,)


def test_v2_run_script_requires_tng_backend_configuration_in_toml() -> None:
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
    assert "--tng-mah-cache" not in completed.stdout


def test_build_tng_cache_help_exposes_download_workers() -> None:
    build_script_path = PROJECT_ROOT / "scripts" / "data" / "build_tng_mah_cache.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(build_script_path),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--download-workers" in completed.stdout
    assert "--download-retries" in completed.stdout
    assert "--drop-invalid-mpb" in completed.stdout
    assert "--snapshot-grid" in completed.stdout
    assert "--missing-mass-ratio-floor" in completed.stdout
