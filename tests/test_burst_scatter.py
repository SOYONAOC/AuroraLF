from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

from auroralf.uvlf.imf import IMF_MODE_MAH_BURST_MILD_TOPHEAVY, IMFTransitionParameters
from auroralf.uvlf.pipeline import _apply_burst_scatter_to_sfr_grid, run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"


def _load_run_script_module():
    spec = importlib.util.spec_from_file_location("run_uvlf_compare_imf_no_delay_all_z", RUN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module from {RUN_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_burst_scatter_leaves_sfr_unchanged() -> None:
    sfr_grid = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 4.0]])
    active_grid = np.array([[True, True, True], [True, False, True]])
    t_grid = np.array([[0.0, 0.01, 0.02], [0.0, 0.01, 0.02]])

    scattered, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.0,
        correlation_timescale_myr=20.0,
        random_seed=123,
        preserve_mean=True,
    )

    np.testing.assert_allclose(scattered, sfr_grid)
    np.testing.assert_allclose(multiplier, np.ones_like(sfr_grid))


def test_burst_scatter_is_reproducible_for_fixed_seed() -> None:
    sfr_grid = np.ones((2, 6), dtype=float)
    active_grid = np.ones_like(sfr_grid, dtype=bool)
    t_grid = np.tile(np.linspace(0.0, 0.1, 6), (2, 1))

    first, first_multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.4,
        correlation_timescale_myr=25.0,
        random_seed=456,
        preserve_mean=True,
    )
    second, second_multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.4,
        correlation_timescale_myr=25.0,
        random_seed=456,
        preserve_mean=True,
    )
    third, _ = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.4,
        correlation_timescale_myr=25.0,
        random_seed=457,
        preserve_mean=True,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first_multiplier, second_multiplier)
    assert not np.allclose(first, third)


def test_burst_scatter_changes_pipeline_luminosities_with_seed() -> None:
    common = {
        "n_tracks": 4,
        "z_final": 6.0,
        "Mh_final": 1.0e10,
        "z_start_max": 10.0,
        "n_grid": 16,
        "random_seed": 101,
        "workers": 1,
        "burst_scatter_dex": 0.5,
        "burst_scatter_timescale_myr": 20.0,
    }

    first = run_halo_uv_pipeline(**common, burst_scatter_random_seed=202)
    second = run_halo_uv_pipeline(**common, burst_scatter_random_seed=202)
    third = run_halo_uv_pipeline(**common, burst_scatter_random_seed=203)

    np.testing.assert_allclose(first.uv_luminosities, second.uv_luminosities)
    assert not np.allclose(first.uv_luminosities, third.uv_luminosities)
    assert first.metadata["burst_scatter_dex"] == 0.5
    assert first.metadata["burst_scatter_timescale_myr"] == 20.0
    assert first.metadata["burst_scatter_random_seed"] == 202


def test_topheavy_pipeline_loads_default_hdf5_ssp_with_burst_scatter() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=12.5,
        Mh_final=1.0e10,
        z_start_max=15.0,
        n_grid=12,
        random_seed=303,
        workers=1,
        imf_mode=IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
        imf_transition_parameters=IMFTransitionParameters(z_topheavy_min=10.0, growth_time_threshold_myr=50.0),
        burst_scatter_dex=0.2,
        burst_scatter_random_seed=404,
    )

    assert result.uv_luminosities.shape == (2,)
    assert np.all(np.isfinite(result.uv_luminosities))
    assert result.metadata["burst_scatter_enabled"] is True
    assert result.metadata["topheavy_ssp_metallicity"] == 0.05


def test_run_script_help_exposes_burst_scatter_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUN_SCRIPT_PATH), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--burst-scatter-dex" in completed.stdout
    assert "--burst-scatter-random-seed" in completed.stdout
    assert "--enable-time-delay" in completed.stdout
    assert "--disable-time-delay" in completed.stdout


def test_run_script_defaults_to_time_delay(monkeypatch) -> None:
    module = _load_run_script_module()

    monkeypatch.setattr(sys, "argv", [str(RUN_SCRIPT_PATH)])
    args = module._parse_args()
    assert args.enable_time_delay is True

    monkeypatch.setattr(sys, "argv", [str(RUN_SCRIPT_PATH), "--disable-time-delay"])
    args = module._parse_args()
    assert args.enable_time_delay is False
