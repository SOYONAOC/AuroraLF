import numpy as np


def cumulative_surviving_stellar_mass_msun(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    returned_fraction: float,
) -> np.ndarray:
    """Integrate surviving stellar mass along each active source history."""

    t_grid = np.asarray(t_grid_gyr, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    if t_grid.ndim != 2 or t_grid.shape[1] < 2:
        raise ValueError("t_grid_gyr must be 2D with at least two time steps")
    if sfr.shape != t_grid.shape or active.shape != t_grid.shape:
        raise ValueError("sfr_grid and active_grid must match t_grid_gyr shape")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid_gyr must contain finite values")

    dt_gyr = np.diff(t_grid, axis=1)
    if np.any(dt_gyr < 0.0):
        raise ValueError("t_grid_gyr must be monotonic non-decreasing along the time axis")

    returned_fraction = float(returned_fraction)
    if not np.isfinite(returned_fraction) or not 0.0 <= returned_fraction < 1.0:
        raise ValueError("returned_fraction must lie in [0, 1)")
    if np.any(active & (~np.isfinite(sfr) | (sfr < 0.0))):
        raise ValueError("sfr_grid must be finite and non-negative for active source times")

    formed_stellar_mass = np.zeros_like(t_grid)
    formed_steps = formed_stellar_mass[:, 1:]
    forming = active[:, 1:] & (dt_gyr > 0.0)
    formed_steps[forming] = (
        (1.0 - returned_fraction)
        * sfr[:, 1:][forming]
        * dt_gyr[forming]
        * 1.0e9
    )
    return np.cumsum(formed_stellar_mass, axis=1)
