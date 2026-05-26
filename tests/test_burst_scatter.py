from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auroralf.uvlf.pipeline import _apply_burst_scatter_to_sfr_grid, run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"


def test_zero_burst_scatter_leaves_sfr_unchanged() -> None:
    sfr_grid = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.5]], dtype=float)
    active_grid = sfr_grid > 0.0
    t_grid = np.array([[0.0, 0.01, 0.02], [0.0, 0.01, 0.02]], dtype=float)

    burst_sfr, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.0,
        correlation_timescale_myr=20.0,
        random_seed=1,
        preserve_mean=True,
    )

    np.testing.assert_allclose(burst_sfr, sfr_grid)
    np.testing.assert_allclose(multiplier, np.ones_like(sfr_grid))


def test_burst_scatter_is_reproducible_for_fixed_seed() -> None:
    sfr_grid = np.ones((2, 6), dtype=float)
    active_grid = np.ones_like(sfr_grid, dtype=bool)
    t_grid = np.tile(np.arange(6, dtype=float) * 0.01, (2, 1))

    first, first_multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.4,
        correlation_timescale_myr=20.0,
        random_seed=11,
        preserve_mean=True,
    )
    second, second_multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.4,
        correlation_timescale_myr=20.0,
        random_seed=11,
        preserve_mean=True,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first_multiplier, second_multiplier)
    assert not np.allclose(first_multiplier, np.ones_like(first_multiplier))


def test_burst_scatter_preserve_mean_conserves_integrated_sfr_per_halo() -> None:
    sfr_grid = np.array(
        [
            [0.0, 1.0, 2.0, 4.0, 6.0],
            [0.0, 0.5, 1.0, 3.0, 5.0],
        ],
        dtype=float,
    )
    active_grid = sfr_grid > 0.0
    t_grid = np.array(
        [
            [0.00, 0.01, 0.03, 0.06, 0.10],
            [0.00, 0.02, 0.04, 0.07, 0.11],
        ],
        dtype=float,
    )

    burst_sfr, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.8,
        correlation_timescale_myr=20.0,
        random_seed=31,
        preserve_mean=True,
    )

    for halo_index in range(sfr_grid.shape[0]):
        source = active_grid[halo_index]
        original_mass = np.trapezoid(sfr_grid[halo_index, source], t_grid[halo_index, source])
        burst_mass = np.trapezoid(burst_sfr[halo_index, source], t_grid[halo_index, source])
        assert burst_mass == pytest.approx(original_mass)

    assert not np.allclose(multiplier[active_grid], 1.0)


def test_burst_scatter_changes_pipeline_luminosities_with_seed() -> None:
    common = dict(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=10.0,
        n_grid=12,
        random_seed=101,
        workers=1,
        burst_scatter_dex=0.5,
        burst_scatter_timescale_myr=20.0,
    )

    first = run_halo_uv_pipeline(**common, burst_scatter_random_seed=202)
    second = run_halo_uv_pipeline(**common, burst_scatter_random_seed=202)
    third = run_halo_uv_pipeline(**common, burst_scatter_random_seed=203)

    np.testing.assert_allclose(first.uv_luminosities, second.uv_luminosities)
    assert not np.allclose(first.uv_luminosities, third.uv_luminosities)
    assert first.metadata["burst_scatter_enabled"] is True
    assert first.metadata["burst_scatter_dex"] == pytest.approx(0.5)
    assert first.metadata["burst_scatter_timescale_myr"] == pytest.approx(20.0)
    assert first.metadata["burst_scatter_mass_conserving"] is True


def test_run_script_help_exposes_burst_scatter_arguments() -> None:
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

    assert "--burst-scatter-dex" in completed.stdout
    assert "--burst-scatter-timescale-myr" in completed.stdout
    assert "--burst-scatter-random-seed" in completed.stdout
    assert "--enable-time-delay" in completed.stdout
    assert "--disable-time-delay" in completed.stdout
    assert "--enable-source-redshift-topheavy-gate" in completed.stdout


def test_run_script_defaults_to_time_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("run_uvlf_compare_imf_no_delay_all_z", RUN_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", [str(RUN_SCRIPT_PATH)])
    args = module._parse_args()
    assert args.enable_time_delay is True

    monkeypatch.setattr(sys, "argv", [str(RUN_SCRIPT_PATH), "--disable-time-delay"])
    args = module._parse_args()
    assert args.enable_time_delay is False
