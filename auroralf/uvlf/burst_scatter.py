"""Archived correlated-lognormal SFR scatter implementation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _lognormal_unit_mean_shift_dex(scatter_dex: float) -> float:
    return -0.5 * np.log(10.0) * float(scatter_dex) ** 2


def _draw_burst_multiplier_for_segments(
    *,
    rng: np.random.Generator,
    segment_ids: np.ndarray,
    scatter_dex: float,
    preserve_mean: bool,
) -> np.ndarray:
    unique_segments, inverse = np.unique(segment_ids, return_inverse=True)
    loc = _lognormal_unit_mean_shift_dex(scatter_dex) if preserve_mean else 0.0
    delta_dex = rng.normal(loc=loc, scale=float(scatter_dex), size=unique_segments.size)
    return np.power(10.0, delta_dex[inverse])


def _apply_burst_scatter_to_sfr_grid(
    *,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    t_grid: np.ndarray,
    scatter_dex: float,
    correlation_timescale_myr: float,
    random_seed: int | None,
    preserve_mean: bool,
    draw_multiplier: Callable[..., np.ndarray] = _draw_burst_multiplier_for_segments,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the archived AuroraLF correlated-lognormal SFR diagnostic.

    This piecewise-constant scatter process, its correlation time, and the
    optional per-halo formed-mass renormalization are project conventions, not
    a time-resolved physical Pop III burst prescription or a literature SFH
    equation.  The feature remains available only for explicit archival runs.
    """
    scatter_dex = float(scatter_dex)
    correlation_timescale_myr = float(correlation_timescale_myr)
    if not np.isfinite(scatter_dex):
        raise ValueError("burst_scatter_dex must be finite")
    if scatter_dex < 0.0:
        raise ValueError("burst_scatter_dex must be non-negative")
    if not np.isfinite(correlation_timescale_myr) or correlation_timescale_myr <= 0.0:
        raise ValueError("burst_scatter_timescale_myr must be finite and positive")

    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    time = np.asarray(t_grid, dtype=float)
    if sfr.shape != active.shape or sfr.shape != time.shape:
        raise ValueError("sfr_grid, active_grid, and t_grid must have the same shape")
    if sfr.ndim != 2:
        raise ValueError("sfr_grid, active_grid, and t_grid must be two-dimensional")
    with np.errstate(over="ignore", invalid="ignore"):
        time_deltas = np.diff(time, axis=1)
    if (
        not np.all(np.isfinite(time))
        or not np.all(np.isfinite(time_deltas))
        or np.any(time_deltas <= 0.0)
    ):
        raise ValueError("each t_grid row must be finite and strictly increasing")
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise ValueError("sfr_grid values must be finite and non-negative")

    multiplier = np.ones_like(sfr, dtype=float)
    if scatter_dex == 0.0:
        return sfr.copy(), multiplier

    rng = np.random.default_rng(random_seed)
    correlation_gyr = correlation_timescale_myr / 1.0e3
    if not np.isfinite(correlation_gyr) or correlation_gyr <= 0.0:
        raise ValueError(
            "burst_scatter_timescale_myr cannot be represented as a positive Gyr interval"
        )
    burst_sfr = sfr.copy()
    source_grid = active & (sfr > 0.0)
    for halo_index in range(sfr.shape[0]):
        time_row = time[halo_index]
        sfr_row = sfr[halo_index]
        with np.errstate(over="ignore", invalid="ignore"):
            original_mass = float(np.trapezoid(sfr_row, time_row))
        if not np.isfinite(original_mass):
            raise RuntimeError("original SFR integration must be finite")

        source = source_grid[halo_index]
        if not np.any(source):
            with np.errstate(over="ignore", invalid="ignore"):
                burst_mass = float(np.trapezoid(burst_sfr[halo_index], time_row))
            if not np.isfinite(burst_mass):
                raise RuntimeError("final burst SFR integration must be finite")
            continue

        original_source_mass: float | None = None
        if preserve_mean:
            source_sfr_row = np.zeros_like(sfr_row)
            source_sfr_row[source] = sfr_row[source]
            with np.errstate(over="ignore", invalid="ignore"):
                original_source_mass = float(np.trapezoid(source_sfr_row, time_row))
            if not np.isfinite(original_source_mass):
                raise RuntimeError("burst SFR source normalization integral must be finite")
            if original_source_mass <= 0.0:
                raise RuntimeError(
                    "mass-conserving burst scatter requires positive full-grid integration support"
                )

        first_time = float(time[halo_index, np.flatnonzero(source)[0]])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            segment_ratios = (time[halo_index, source] - first_time) / correlation_gyr
            floored_segment_ratios = np.floor(segment_ratios)
        int64_upper_exclusive = float(2**63)
        if (
            not np.all(np.isfinite(floored_segment_ratios))
            or np.any(floored_segment_ratios < 0.0)
            or np.any(floored_segment_ratios >= int64_upper_exclusive)
        ):
            raise ValueError(
                "burst correlation segment ids cannot be represented as non-negative int64"
            )
        segment_ids = floored_segment_ratios.astype(np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            row_multiplier = draw_multiplier(
                rng=rng,
                segment_ids=segment_ids,
                scatter_dex=scatter_dex,
                preserve_mean=preserve_mean,
            )
        if not np.all(np.isfinite(row_multiplier)) or np.any(row_multiplier <= 0.0):
            raise RuntimeError("burst SFR multipliers must be finite and positive")

        raw_burst_row = sfr_row.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            raw_burst_row[source] = sfr_row[source] * row_multiplier
            raw_burst_mass = float(np.trapezoid(raw_burst_row, time_row))
        if not np.isfinite(raw_burst_mass):
            raise RuntimeError("burst SFR normalization integral must be finite")
        if original_mass > 0.0 and raw_burst_mass <= 0.0:
            raise RuntimeError("positive original SFR mass requires positive burst SFR mass")

        if preserve_mean:
            if original_source_mass is None:
                raise RuntimeError("missing burst SFR source normalization integral")
            raw_burst_source_row = np.zeros_like(sfr_row)
            with np.errstate(over="ignore", invalid="ignore"):
                raw_burst_source_row[source] = sfr_row[source] * row_multiplier
                raw_burst_source_mass = float(np.trapezoid(raw_burst_source_row, time_row))
            if not np.isfinite(raw_burst_source_mass):
                raise RuntimeError("burst SFR source normalization integral must be finite")
            if raw_burst_source_mass <= 0.0:
                raise RuntimeError("burst SFR source normalization integral must be positive")
            with np.errstate(over="ignore", invalid="ignore"):
                row_multiplier = row_multiplier * (
                    original_source_mass / raw_burst_source_mass
                )

        if not np.all(np.isfinite(row_multiplier)) or np.any(row_multiplier <= 0.0):
            raise RuntimeError("normalized burst SFR multipliers must be finite and positive")

        multiplier[halo_index, source] = row_multiplier
        with np.errstate(over="ignore", invalid="ignore"):
            burst_sfr[halo_index, source] = sfr[halo_index, source] * row_multiplier

        with np.errstate(over="ignore", invalid="ignore"):
            burst_mass = float(np.trapezoid(burst_sfr[halo_index], time_row))
        if not np.isfinite(burst_mass):
            raise RuntimeError("final burst SFR integration must be finite")
        if preserve_mean and not np.isclose(
            burst_mass,
            original_mass,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RuntimeError("mass-conserving burst SFR normalization failed")

    if not np.all(np.isfinite(burst_sfr)) or np.any(burst_sfr < 0.0):
        raise RuntimeError("final burst SFR must be finite and non-negative")

    return burst_sfr, multiplier
