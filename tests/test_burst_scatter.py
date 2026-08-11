from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from auroralf.seeding import PipelineRandomSeeds, derive_pipeline_random_seeds
from auroralf.uvlf.pipeline import _apply_burst_scatter_to_sfr_grid, run_halo_uv_pipeline
from auroralf.mah import Cosmology


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"


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


@pytest.mark.parametrize(
    ("sfr_grid", "active_grid", "t_grid", "error_match"),
    [
        (
            np.array([0.0, 1.0]),
            np.array([False, True]),
            np.array([0.0, 0.01]),
            "two-dimensional",
        ),
        (
            np.array([[0.0, 1.0]]),
            np.array([[False, True]]),
            np.array([[0.0, 0.0]]),
            "finite and strictly increasing",
        ),
        (
            np.array([[0.0, -1.0]]),
            np.array([[False, True]]),
            np.array([[0.0, 0.01]]),
            "finite and non-negative",
        ),
    ],
)
def test_zero_burst_scatter_still_validates_full_inputs(
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    t_grid: np.ndarray,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _apply_burst_scatter_to_sfr_grid(
            sfr_grid=sfr_grid,
            active_grid=active_grid,
            t_grid=t_grid,
            scatter_dex=0.0,
            correlation_timescale_myr=20.0,
            random_seed=1,
            preserve_mean=True,
        )


@pytest.mark.parametrize("scatter_dex", [np.nan, np.inf, -np.inf])
def test_burst_scatter_rejects_nonfinite_scatter_dex_without_warning(scatter_dex: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="burst_scatter_dex must be finite"):
            _apply_burst_scatter_to_sfr_grid(
                sfr_grid=np.array([[0.0, 1.0]], dtype=float),
                active_grid=np.array([[False, True]]),
                t_grid=np.array([[0.0, 0.01]], dtype=float),
                scatter_dex=scatter_dex,
                correlation_timescale_myr=20.0,
                random_seed=1,
                preserve_mean=True,
            )


@pytest.mark.parametrize("timescale_myr", [np.nan, np.inf, -np.inf, 0.0])
def test_burst_scatter_rejects_invalid_timescale_without_warning(timescale_myr: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="burst_scatter_timescale_myr must be finite and positive"):
            _apply_burst_scatter_to_sfr_grid(
                sfr_grid=np.array([[0.0, 1.0]], dtype=float),
                active_grid=np.array([[False, True]]),
                t_grid=np.array([[0.0, 0.01]], dtype=float),
                scatter_dex=0.4,
                correlation_timescale_myr=timescale_myr,
                random_seed=1,
                preserve_mean=True,
            )


def test_burst_scatter_rejects_unrepresentable_correlation_segments_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="cannot be represented"):
            _apply_burst_scatter_to_sfr_grid(
                sfr_grid=np.array([[1.0, 1.0]], dtype=float),
                active_grid=np.array([[True, True]]),
                t_grid=np.array([[0.0, 0.01]], dtype=float),
                scatter_dex=0.4,
                correlation_timescale_myr=np.nextafter(0.0, 1.0),
                random_seed=1,
                preserve_mean=False,
            )


def test_burst_scatter_assigns_equal_ids_within_segment_and_distinct_ids_across(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_segment_ids: list[np.ndarray] = []

    def _record_segments(
        *,
        rng: np.random.Generator,
        segment_ids: np.ndarray,
        scatter_dex: float,
        preserve_mean: bool,
    ) -> np.ndarray:
        del rng, scatter_dex, preserve_mean
        recorded_segment_ids.append(segment_ids.copy())
        return np.ones(segment_ids.shape, dtype=float)

    monkeypatch.setattr("auroralf.uvlf.pipeline._draw_burst_multiplier_for_segments", _record_segments)
    _apply_burst_scatter_to_sfr_grid(
        sfr_grid=np.ones((1, 4), dtype=float),
        active_grid=np.ones((1, 4), dtype=bool),
        t_grid=np.array([[0.0, 0.005, 0.019, 0.025]], dtype=float),
        scatter_dex=0.4,
        correlation_timescale_myr=20.0,
        random_seed=1,
        preserve_mean=False,
    )

    assert len(recorded_segment_ids) == 1
    np.testing.assert_array_equal(recorded_segment_ids[0], np.array([0, 0, 0, 1], dtype=np.int64))


def test_mass_conserving_burst_scatter_rejects_positive_source_without_time_support() -> None:
    with pytest.raises(RuntimeError, match="positive full-grid integration support"):
        _apply_burst_scatter_to_sfr_grid(
            sfr_grid=np.array([[2.0]], dtype=float),
            active_grid=np.array([[True]]),
            t_grid=np.array([[0.0]], dtype=float),
            scatter_dex=0.4,
            correlation_timescale_myr=20.0,
            random_seed=1,
            preserve_mean=True,
        )


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
        original_mass = np.trapezoid(sfr_grid[halo_index], t_grid[halo_index])
        burst_mass = np.trapezoid(burst_sfr[halo_index], t_grid[halo_index])
        assert burst_mass == pytest.approx(original_mass)

    assert not np.allclose(multiplier[active_grid], 1.0)


def test_burst_scatter_conserves_full_grid_mass_across_internal_zero_gap() -> None:
    t_grid = np.array([[0.0, 0.01, 0.025, 0.08, 0.14, 0.20]], dtype=float)
    sfr_grid = np.array([[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]], dtype=float)
    active_grid = sfr_grid > 0.0

    # Normalizing only the compressed positive-source samples bridges the zero
    # interval and therefore does not conserve the full-grid time integral.
    source = active_grid[0]
    raw_source_multiplier = np.array([2.0, 0.5])
    compressed_scale = np.trapezoid(sfr_grid[0, source], t_grid[0, source]) / np.trapezoid(
        sfr_grid[0, source] * raw_source_multiplier,
        t_grid[0, source],
    )
    compressed_burst = sfr_grid[0].copy()
    compressed_burst[source] *= raw_source_multiplier * compressed_scale
    assert np.trapezoid(compressed_burst, t_grid[0]) != pytest.approx(
        np.trapezoid(sfr_grid[0], t_grid[0])
    )

    burst_sfr, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.8,
        correlation_timescale_myr=20.0,
        random_seed=17,
        preserve_mean=True,
    )

    assert np.trapezoid(burst_sfr[0], t_grid[0]) == pytest.approx(
        np.trapezoid(sfr_grid[0], t_grid[0])
    )
    np.testing.assert_array_equal(burst_sfr[0, ~source], 0.0)
    assert not np.allclose(multiplier[0, source], 1.0)


def test_burst_scatter_conserves_full_grid_mass_with_one_positive_source_bin() -> None:
    t_grid = np.array([[0.0, 0.01, 0.04, 0.10]], dtype=float)
    sfr_grid = np.array([[0.0, 0.0, 4.0, 0.0]], dtype=float)
    active_grid = sfr_grid > 0.0

    burst_sfr, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.8,
        correlation_timescale_myr=20.0,
        random_seed=23,
        preserve_mean=True,
    )

    assert np.trapezoid(burst_sfr[0], t_grid[0]) == pytest.approx(
        np.trapezoid(sfr_grid[0], t_grid[0])
    )
    np.testing.assert_array_equal(burst_sfr[sfr_grid == 0.0], 0.0)
    assert multiplier[0, 2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "invalid_time",
    [
        np.array([[0.0, 0.01, 0.01]], dtype=float),
        np.array([[0.0, np.nan, 0.02]], dtype=float),
    ],
)
def test_burst_scatter_rejects_invalid_full_time_grid(invalid_time: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        _apply_burst_scatter_to_sfr_grid(
            sfr_grid=np.array([[0.0, 1.0, 2.0]], dtype=float),
            active_grid=np.array([[False, True, True]]),
            t_grid=invalid_time,
            scatter_dex=0.4,
            correlation_timescale_myr=20.0,
            random_seed=7,
            preserve_mean=True,
        )


@pytest.mark.parametrize(
    "invalid_sfr",
    [
        np.array([[0.0, -1.0, 2.0]], dtype=float),
        np.array([[0.0, np.nan, 2.0]], dtype=float),
    ],
)
def test_burst_scatter_rejects_invalid_sfr_grid(invalid_sfr: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _apply_burst_scatter_to_sfr_grid(
            sfr_grid=invalid_sfr,
            active_grid=np.array([[False, True, True]]),
            t_grid=np.array([[0.0, 0.01, 0.02]], dtype=float),
            scatter_dex=0.4,
            correlation_timescale_myr=20.0,
            random_seed=7,
            preserve_mean=True,
        )


def test_burst_scatter_rejects_nonfinite_full_integral_without_sources() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeError, match="original SFR integration must be finite"):
            _apply_burst_scatter_to_sfr_grid(
                sfr_grid=np.full((1, 3), 1.0e308, dtype=float),
                active_grid=np.zeros((1, 3), dtype=bool),
                t_grid=np.array([[0.0, 1.0, 2.0]], dtype=float),
                scatter_dex=0.4,
                correlation_timescale_myr=20.0,
                random_seed=7,
                preserve_mean=True,
            )


def test_nonconserving_burst_scatter_only_changes_active_positive_sources() -> None:
    t_grid = np.array([[0.0, 0.01, 0.03, 0.07, 0.12]], dtype=float)
    sfr_grid = np.array([[1.0, 2.0, 0.0, 3.0, 4.0]], dtype=float)
    active_grid = np.array([[False, True, True, True, False]])

    burst_sfr, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=0.8,
        correlation_timescale_myr=20.0,
        random_seed=29,
        preserve_mean=False,
    )

    source = active_grid & (sfr_grid > 0.0)
    np.testing.assert_array_equal(burst_sfr[~source], sfr_grid[~source])
    np.testing.assert_array_equal(multiplier[~source], 1.0)
    assert not np.allclose(burst_sfr[source], sfr_grid[source])
    assert np.trapezoid(burst_sfr[0], t_grid[0]) != pytest.approx(
        np.trapezoid(sfr_grid[0], t_grid[0])
    )


def test_burst_scatter_changes_pipeline_luminosities_with_seed() -> None:
    common = dict(
        n_tracks=4,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=10.0,
        n_grid=12,
        workers=1,
        burst_scatter_dex=0.5,
        burst_scatter_timescale_myr=20.0,
    )

    shared = derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0)
    first_seeds = PipelineRandomSeeds(shared.mah, shared.metallicity, 202)
    third_seeds = PipelineRandomSeeds(shared.mah, shared.metallicity, 203)
    first = run_halo_uv_pipeline(**common, random_seeds=first_seeds)
    second = run_halo_uv_pipeline(**common, random_seeds=first_seeds)
    third = run_halo_uv_pipeline(**common, random_seeds=third_seeds)

    np.testing.assert_allclose(first.uv_luminosities, second.uv_luminosities)
    assert not np.allclose(first.uv_luminosities, third.uv_luminosities)
    assert first.metadata["burst_scatter_enabled"] is True
    assert first.metadata["burst_scatter_dex"] == pytest.approx(0.5)
    assert first.metadata["burst_scatter_timescale_myr"] == pytest.approx(20.0)
    assert first.metadata["burst_scatter_mass_conserving"] is True


def test_v2_cli_moves_burst_and_gate_controls_into_strict_config() -> None:
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
    assert "--burst-scatter-dex" not in completed.stdout
    assert "--enable-time-delay" not in completed.stdout


def test_production_config_defaults_to_delay_and_mass_conserving_zero_scatter() -> None:
    from auroralf import UVLFRunConfig

    config = UVLFRunConfig.from_toml(PRODUCTION_CONFIG)
    assert config.star_formation.enable_time_delay is True
    assert config.star_formation.burst_scatter_dex == pytest.approx(0.0)
    assert config.star_formation.burst_scatter_mass_conserving is True
    assert config.stellar_population.source_redshift_gate_enabled is False
