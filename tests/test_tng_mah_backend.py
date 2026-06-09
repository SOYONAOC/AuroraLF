from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"
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
        handle.attrs["schema_version"] = "auroralf_tng_mah_cache_v1"
        handle.attrs["source_simulation"] = "TNG100-1-Dark"
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["z_final"] = 6.0
        handle.create_dataset("z_grid", data=z_grid)
        handle.create_dataset("t_gyr_grid", data=t_gyr_grid)
        handle.create_dataset("mass_ratio", data=mass_ratio)
        handle.create_dataset("resolved_mask", data=resolved_mask)
        handle.create_dataset("logM_final", data=logm_final)
        handle.create_dataset("source_subhalo_id", data=np.arange(n_halos, dtype=np.int64))


def _write_raw_mpb(path: Path, *, snap: np.ndarray, mass: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("SnapNum", data=snap.astype(np.int64))
        handle.create_dataset("Group_M_Crit200", data=mass.astype(float))


def test_tng_backend_outputs_standard_halo_history_result(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    assert result.metadata["mah_backend"] == "tng"
    assert result.metadata["source_simulation"] == "TNG100-1-Dark"
    assert result.metadata["candidate_count"] == 6
    assert result.metadata["negative_dmhdt_clip_fraction"] == pytest.approx(0.0)

    tracks = result.tracks
    mh_grid = np.asarray(tracks["Mh"], dtype=float).reshape(4, 4)
    dmhdt_grid = np.asarray(tracks["dMh_dt"], dtype=float).reshape(4, 4)
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
        "dMh_dt",
        "active_flag",
        "termination_flag",
    }


def test_tng_backend_can_regrid_selected_shapes_to_uniform_time_grid(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, unresolved_first_step=True)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
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
    dmhdt_grid = np.asarray(tracks["dMh_dt"], dtype=float).reshape(4, 12)
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
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=1,
    )

    dmhdt = np.asarray(result.tracks["dMh_dt"], dtype=float)
    assert np.all(dmhdt >= 0.0)
    assert result.metadata["negative_dmhdt_clip_fraction"] > 0.0


def test_tng_backend_marks_unresolved_steps_and_zeros_boundary_dmhdt(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache, unresolved_first_step=True)

    from auroralf.mah.tng import generate_tng_halo_histories

    result = generate_tng_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    dmhdt_grid = np.asarray(result.tracks["dMh_dt"], dtype=float).reshape(4, 4)
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


def test_tng_backend_runs_through_uv_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "tng_mah_z6.hdf5"
    _write_synthetic_tng_cache(cache)

    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=3,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=12.0,
        n_grid=4,
        mah_backend="tng",
        tng_mah_cache_path=cache,
        tng_mass_bin_width_dex=0.15,
        tng_min_candidates=5,
        random_seed=3,
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
        z_start_max=12.0,
        n_grid=12,
        mah_backend="tng",
        tng_mah_cache_path=cache,
        tng_mass_bin_width_dex=0.15,
        tng_min_candidates=5,
        tng_time_grid_mode="uniform_in_t",
        random_seed=3,
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
        z_start_max=8.0,
        n_grid=4,
        random_seed=3,
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
        N_mass=1,
        n_tracks=2,
        random_seed=2,
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


def test_run_script_help_exposes_tng_mah_arguments() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--mah-backend" in completed.stdout
    assert "--tng-mah-cache" in completed.stdout
    assert "--tng-mass-bin-width-dex" in completed.stdout
    assert "--tng-min-candidates" in completed.stdout
    assert "--tng-smoothing-myr" in completed.stdout
    assert "--tng-time-grid-mode" in completed.stdout


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
