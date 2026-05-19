from __future__ import annotations

import numpy as np


def _as_matching_grid(name: str, values: np.ndarray, shape: tuple[int, int], dtype: type) -> np.ndarray:
    grid = np.asarray(values, dtype=dtype)
    if grid.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {grid.shape}")
    return grid


def _percentiles_by_step(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps = values.shape[1]
    median = np.full(n_steps, np.nan, dtype=float)
    p16 = np.full(n_steps, np.nan, dtype=float)
    p84 = np.full(n_steps, np.nan, dtype=float)
    count = np.zeros(n_steps, dtype=np.int64)
    for step_index in range(n_steps):
        step_mask = mask[:, step_index] & np.isfinite(values[:, step_index])
        count[step_index] = int(np.count_nonzero(step_mask))
        if count[step_index] == 0:
            continue
        selected = values[step_mask, step_index]
        p16[step_index], median[step_index], p84[step_index] = np.percentile(selected, [16.0, 50.0, 84.0])
    return median, p16, p84, count


def summarize_metallicity_history(
    *,
    z_grid: np.ndarray,
    gas_metallicity_zsun_grid: np.ndarray,
    birth_metallicity_zsun_grid: np.ndarray,
    active_grid: np.ndarray,
    starforming_grid: np.ndarray,
    topheavy_source_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Summarize metallicity histories along the shared time axis.

    Gas metallicity is summarized over active halos. Birth metallicity and
    top-heavy source fractions are summarized over star-forming source-time
    cells, because birth metallicity is only physically sampled when stars form.
    """

    gas = np.asarray(gas_metallicity_zsun_grid, dtype=float)
    if gas.ndim != 2:
        raise ValueError("gas_metallicity_zsun_grid must be a 2D array")
    shape = gas.shape
    birth = _as_matching_grid("birth_metallicity_zsun_grid", birth_metallicity_zsun_grid, shape, float)
    active = _as_matching_grid("active_grid", active_grid, shape, bool)
    starforming = _as_matching_grid("starforming_grid", starforming_grid, shape, bool)

    z = np.asarray(z_grid, dtype=float)
    if z.ndim == 1:
        if z.size != shape[1]:
            raise ValueError(f"1D z_grid must have length {shape[1]}, got {z.size}")
        z_axis = z
    else:
        z = _as_matching_grid("z_grid", z, shape, float)
        if not np.allclose(z, z[0], rtol=0.0, atol=0.0, equal_nan=True):
            raise ValueError("z_grid must be shared across halos")
        z_axis = z[0]

    gas_median, gas_p16, gas_p84, active_count = _percentiles_by_step(gas, active)
    birth_median, birth_p16, birth_p84, starforming_count = _percentiles_by_step(birth, starforming)

    topheavy_fraction = np.full(shape[1], np.nan, dtype=float)
    if topheavy_source_grid is not None:
        topheavy = _as_matching_grid("topheavy_source_grid", topheavy_source_grid, shape, bool)
        for step_index in range(shape[1]):
            mask = starforming[:, step_index]
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            topheavy_fraction[step_index] = float(np.count_nonzero(topheavy[:, step_index] & mask) / count)

    return {
        "z": np.asarray(z_axis, dtype=float),
        "gas_median": gas_median,
        "gas_p16": gas_p16,
        "gas_p84": gas_p84,
        "birth_median": birth_median,
        "birth_p16": birth_p16,
        "birth_p84": birth_p84,
        "active_count": active_count,
        "starforming_count": starforming_count,
        "topheavy_source_fraction": topheavy_fraction,
    }
