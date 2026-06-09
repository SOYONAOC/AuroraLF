from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"


def _write_synthetic_thesan_cache(
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
        handle.attrs["schema_version"] = "auroralf_thesan_mah_cache_v0"
        handle.attrs["source_simulation"] = "Thesan-Dark-1"
        handle.attrs["source_tree"] = "LHaloTree"
        handle.attrs["mass_unit"] = "Msun"
        handle.attrs["z_final"] = 6.0
        handle.create_dataset("z_grid", data=z_grid)
        handle.create_dataset("t_gyr_grid", data=t_gyr_grid)
        handle.create_dataset("mass_ratio", data=mass_ratio)
        handle.create_dataset("resolved_mask", data=resolved_mask)
        handle.create_dataset("logM_final", data=logm_final)
        handle.create_dataset("source_subhalo_id", data=np.arange(n_halos, dtype=np.int64))
        handle.create_dataset("source_group_index", data=np.arange(100, 100 + n_halos, dtype=np.int64))
        handle.create_dataset("source_tree_file", data=np.zeros(n_halos, dtype=np.int64))


def test_thesan_backend_outputs_standard_halo_history_result(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    result = generate_thesan_halo_histories(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=7,
    )

    assert result.metadata["mah_backend"] == "thesan"
    assert result.metadata["source_simulation"] == "Thesan-Dark-1"
    assert result.metadata["source_tree"] == "LHaloTree"
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


def test_thesan_backend_fails_when_candidate_count_is_too_small(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache, n_halos=3)

    from auroralf.mah.thesan import generate_thesan_halo_histories

    with pytest.raises(ValueError, match="THESAN MAH candidate count"):
        generate_thesan_halo_histories(
            n_tracks=2,
            z_final=6.0,
            Mh_final=1.0e10,
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
        cache_path=cache,
        mass_bin_width_dex=0.15,
        min_candidates=5,
        random_seed=1,
    )

    assert np.all(np.asarray(result.tracks["dMh_dt"], dtype=float) >= 0.0)
    assert result.metadata["negative_dmhdt_clip_fraction"] > 0.0


def test_thesan_backend_runs_through_uv_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "thesan_mah_z6.hdf5"
    _write_synthetic_thesan_cache(cache)

    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    result = run_halo_uv_pipeline(
        n_tracks=3,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=12.0,
        n_grid=4,
        mah_backend="thesan",
        thesan_mah_cache_path=cache,
        thesan_mass_bin_width_dex=0.15,
        thesan_min_candidates=5,
        random_seed=3,
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
        N_mass=1,
        n_tracks=2,
        random_seed=2,
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


def test_run_script_help_exposes_thesan_mah_arguments() -> None:
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

    assert "--thesan-mah-cache" in completed.stdout
    assert "--thesan-mass-bin-width-dex" in completed.stdout
    assert "--thesan-min-candidates" in completed.stdout
    assert "--thesan-smoothing-myr" in completed.stdout
