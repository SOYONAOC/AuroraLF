from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_TIME_UNIT_IN_YEARS = 1.0e9
SSP_UV_LOOKBACK_MAX_MYR = 100.0


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.dtype(bool):
            return True
        if value.dtype == np.dtype(object):
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _reject_boolean_values(name: str, values: object) -> None:
    if _contains_boolean(values):
        raise ValueError(f"{name} must not contain boolean values")


def _reject_boolean_scalar(name: str, value: object) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be boolean")


def _validate_lookback_max_myr(
    name: str,
    value: object,
    *,
    t_obs: np.ndarray | float,
) -> float:
    _reject_boolean_scalar(name, value)
    lookback_myr = float(value)
    if not np.isfinite(lookback_myr) or lookback_myr <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lookback_gyr = lookback_myr / 1.0e3
    if not np.isfinite(lookback_gyr) or lookback_gyr <= 0.0:
        raise ValueError(f"{name} cannot be represented as a positive Gyr interval")
    observation_time = np.asarray(t_obs, dtype=float)
    if not np.all(np.isfinite(observation_time)):
        raise ValueError("t_obs must be finite when validating lookback")
    with np.errstate(over="ignore", invalid="ignore"):
        boundary = observation_time - lookback_gyr
    if not np.all(np.isfinite(boundary)) or not np.all(boundary < observation_time):
        raise ValueError(f"{name} lookback boundary is not representable at every observation time")
    return lookback_gyr


def _ensure_1d_float_array(name: str, values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _prepare_sorted_history(
    t_history: np.ndarray,
    mh_history: np.ndarray,
    sfr_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (t_history.size == mh_history.size == sfr_history.size):
        raise ValueError("t_history, mh_history, and sfr_history must have the same length")

    order = np.argsort(t_history, kind="stable")
    t_sorted = t_history[order]
    mh_sorted = mh_history[order]
    sfr_sorted = sfr_history[order]

    if np.any(np.diff(t_sorted) < 0.0):
        raise ValueError("t_history could not be sorted into ascending order")
    if np.any(np.diff(t_sorted) == 0.0):
        raise ValueError("t_history must be strictly increasing after sorting")

    return t_sorted, mh_sorted, sfr_sorted


def _find_mass_crossing_time(t_history: np.ndarray, mh_history: np.ndarray, m_min: float) -> float | None:
    above = mh_history >= m_min
    if not np.any(above):
        return None

    first_above = int(np.flatnonzero(above)[0])
    if first_above == 0:
        return None

    left_index = first_above - 1
    t_left = float(t_history[left_index])
    t_right = float(t_history[first_above])
    mh_left = float(mh_history[left_index])
    mh_right = float(mh_history[first_above])

    if mh_right == mh_left:
        return t_right

    weight = (m_min - mh_left) / (mh_right - mh_left)
    return t_left + weight * (t_right - t_left)


def _augment_with_boundaries(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(float(lower)) or not np.isfinite(float(upper)):
        raise ValueError("lower and upper boundaries must be finite")
    if float(lower) >= float(upper):
        raise ValueError("lower boundary must be strictly below upper boundary")
    if float(lower) < float(x[0]) or float(upper) > float(x[-1]):
        raise ValueError("lower and upper boundaries must lie within x")
    interior = x[(x > lower) & (x < upper)]
    x_used = np.concatenate(([float(lower)], interior, [float(upper)]))
    y_used = np.interp(x_used, x, y)
    return x_used, y_used


def interpolate_ssp_luminosity(
    age: float | np.ndarray,
    ssp_age_grid: np.ndarray | list[float] | tuple[float, ...],
    ssp_luv_grid: np.ndarray | list[float] | tuple[float, ...],
) -> float | np.ndarray:
    """Interpolate the SSP UV luminosity kernel on log-age, with old-age contributions truncated to zero."""

    age_grid = _ensure_1d_float_array("ssp_age_grid", ssp_age_grid)
    luv_grid = _ensure_1d_float_array("ssp_luv_grid", ssp_luv_grid)
    if age_grid.size != luv_grid.size:
        raise ValueError("ssp_age_grid and ssp_luv_grid must have the same length")
    if np.any(age_grid <= 0.0):
        raise ValueError("ssp_age_grid must contain strictly positive ages")

    order = np.argsort(age_grid, kind="stable")
    age_grid = age_grid[order]
    luv_grid = luv_grid[order]

    age_array = np.asarray(age, dtype=float)
    if np.any(age_array < 0.0):
        raise ValueError("age must be non-negative")

    result = np.zeros_like(age_array, dtype=float)

    below_mask = age_array < age_grid[0]
    in_range_mask = (age_array >= age_grid[0]) & (age_array <= age_grid[-1])
    result[below_mask] = luv_grid[0]

    if np.any(in_range_mask):
        log_age_grid = np.log10(age_grid)
        log_age = np.log10(age_array[in_range_mask])
        result[in_range_mask] = np.interp(log_age, log_age_grid, luv_grid)

    if np.ndim(age) == 0:
        return float(result)
    return result


def compute_final_ssp_observable_from_sfr_grid(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_myr: np.ndarray,
    ssp_observable_per_msun: np.ndarray,
    lookback_max_myr: float,
    time_unit_in_years: float = DEFAULT_TIME_UNIT_IN_YEARS,
) -> np.ndarray:
    """Convolve final-time SFR histories with an arbitrary SSP kernel.

    History times are in ``Gyr``, SSP ages are in ``Myr``, SFR is in
    ``Msun yr^-1``, and the SSP kernel is an observable per ``Msun``. The
    returned array therefore has the kernel's observable units.
    """

    _reject_boolean_values("t_grid_gyr", t_grid_gyr)
    _reject_boolean_values("sfr_grid", sfr_grid)
    _reject_boolean_values("ssp_age_myr", ssp_age_myr)
    _reject_boolean_values("ssp_observable_per_msun", ssp_observable_per_msun)
    _reject_boolean_scalar("time_unit_in_years", time_unit_in_years)

    t_grid = np.asarray(t_grid_gyr, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid)
    grids = (t_grid, sfr, active)
    if any(grid.ndim != 2 or grid.size == 0 for grid in grids):
        raise ValueError("t_grid_gyr, sfr_grid, and active_grid must be non-empty 2D arrays")
    if t_grid.shape != sfr.shape or t_grid.shape != active.shape:
        raise ValueError("t_grid_gyr, sfr_grid, and active_grid must have identical shapes")
    if active.dtype != np.dtype(bool):
        raise ValueError("active_grid must have boolean dtype")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid_gyr must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        time_deltas = np.diff(t_grid, axis=1)
    if not np.all(np.isfinite(time_deltas)) or np.any(time_deltas <= 0.0):
        raise ValueError("each t_grid_gyr row must be strictly increasing")
    if not np.all(np.isfinite(sfr)):
        raise ValueError("sfr_grid must contain only finite values")
    if np.any(sfr < 0.0):
        raise ValueError("sfr_grid must be non-negative")

    age_myr = np.asarray(ssp_age_myr, dtype=float)
    observable_per_msun = np.asarray(ssp_observable_per_msun, dtype=float)
    if age_myr.ndim != 1 or age_myr.size == 0:
        raise ValueError("ssp_age_myr must be a non-empty 1D array")
    if observable_per_msun.ndim != 1 or observable_per_msun.size == 0:
        raise ValueError("ssp_observable_per_msun must be a non-empty 1D array")
    if age_myr.size != observable_per_msun.size:
        raise ValueError("ssp_age_myr and ssp_observable_per_msun must have the same length")
    if not np.all(np.isfinite(age_myr)):
        raise ValueError("ssp_age_myr must contain only finite values")
    if np.any(age_myr <= 0.0):
        raise ValueError("ssp_age_myr must contain strictly positive ages")
    if np.any(np.diff(age_myr) <= 0.0):
        raise ValueError("ssp_age_myr must be strictly increasing")
    if not np.all(np.isfinite(observable_per_msun)):
        raise ValueError("ssp_observable_per_msun must contain only finite values")
    if np.any(observable_per_msun < 0.0):
        raise ValueError("ssp_observable_per_msun must be non-negative")

    time_conversion = float(time_unit_in_years)
    if not np.isfinite(time_conversion) or time_conversion <= 0.0:
        raise ValueError("time_unit_in_years must be finite and positive")
    t_obs_grid = t_grid[:, -1]
    lookback_gyr = _validate_lookback_max_myr(
        "lookback_max_myr",
        lookback_max_myr,
        t_obs=t_obs_grid,
    )

    result = np.zeros(t_grid.shape[0], dtype=float)
    for row_index in range(t_grid.shape[0]):
        time_row = t_grid[row_index]
        t_obs = float(time_row[-1])
        lower = max(float(time_row[0]), t_obs - lookback_gyr)
        if lower >= t_obs:
            continue

        source_sfr = np.where(active[row_index], sfr[row_index], 0.0)
        interior = time_row[(time_row > lower) & (time_row < t_obs)]
        time_used = np.unique(
            np.concatenate(([lower], interior, [t_obs]))
        )
        sfr_used = np.interp(time_used, time_row, source_sfr)
        age_used_myr = (t_obs - time_used) * 1.0e3
        kernel_used = np.asarray(
            interpolate_ssp_luminosity(
                age_used_myr,
                ssp_age_grid=age_myr,
                ssp_luv_grid=observable_per_msun,
            ),
            dtype=float,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            integrand = sfr_used * kernel_used
            time_used_years = time_used * time_conversion
            result[row_index] = float(
                np.trapezoid(integrand, x=time_used_years)
            )

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError("final SSP observable must be finite and non-negative")
    return result


def compute_halo_uv_luminosity(
    t_obs: float,
    t_history: np.ndarray | list[float] | tuple[float, ...],
    mh_history: np.ndarray | list[float] | tuple[float, ...],
    sfr_history: np.ndarray | list[float] | tuple[float, ...],
    ssp_age_grid: np.ndarray | list[float] | tuple[float, ...],
    ssp_luv_grid: np.ndarray | list[float] | tuple[float, ...],
    M_min: float,
    t_z50: float,
    time_unit_in_years: float = DEFAULT_TIME_UNIT_IN_YEARS,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Convolve a halo SFR history with an SSP UV kernel at ``t_obs``.

    This legacy public API takes both ``t_history`` and ``ssp_age_grid`` in
    ``Gyr``. The SSP ages are converted to ``Myr`` exactly once before calling
    :func:`compute_final_ssp_observable_from_sfr_grid`.
    """

    for name, values in (
        ("t_history", t_history),
        ("mh_history", mh_history),
        ("sfr_history", sfr_history),
        ("ssp_age_grid", ssp_age_grid),
        ("ssp_luv_grid", ssp_luv_grid),
    ):
        _reject_boolean_values(name, values)
    for name, value in (
        ("t_obs", t_obs),
        ("M_min", M_min),
        ("t_z50", t_z50),
        ("time_unit_in_years", time_unit_in_years),
        ("ssp_lookback_max_myr", ssp_lookback_max_myr),
    ):
        _reject_boolean_scalar(name, value)

    t_history_array = _ensure_1d_float_array("t_history", t_history)
    mh_history_array = _ensure_1d_float_array("mh_history", mh_history)
    sfr_history_array = _ensure_1d_float_array("sfr_history", sfr_history)
    if not np.all(np.isfinite(t_history_array)):
        raise ValueError("t_history must contain only finite values")
    if not np.all(np.isfinite(mh_history_array)):
        raise ValueError("mh_history must contain only finite values")
    if not np.all(np.isfinite(sfr_history_array)):
        raise ValueError("sfr_history must contain only finite values")
    if np.any(sfr_history_array < 0.0):
        raise ValueError("sfr_history must be non-negative")
    legacy_ssp_age_gyr = _ensure_1d_float_array("ssp_age_grid", ssp_age_grid)
    ssp_luv = _ensure_1d_float_array("ssp_luv_grid", ssp_luv_grid)
    if legacy_ssp_age_gyr.size != ssp_luv.size:
        raise ValueError("ssp_age_grid and ssp_luv_grid must have the same length")
    if not np.all(np.isfinite(legacy_ssp_age_gyr)):
        raise ValueError("ssp_age_grid must contain only finite values")
    if np.any(legacy_ssp_age_gyr <= 0.0):
        raise ValueError("ssp_age_grid must contain strictly positive ages")
    if not np.all(np.isfinite(ssp_luv)):
        raise ValueError("ssp_luv_grid must contain only finite values")
    if np.any(ssp_luv < 0.0):
        raise ValueError("ssp_luv_grid must be non-negative")
    t_obs = float(t_obs)
    t_z50 = float(t_z50)
    M_min = float(M_min)
    time_unit_in_years = float(time_unit_in_years)

    if not np.isfinite(t_obs):
        raise ValueError("t_obs must be finite")
    if not np.isfinite(t_z50):
        raise ValueError("t_z50 must be finite")
    if not np.isfinite(M_min):
        raise ValueError("M_min must be finite")
    if not np.isfinite(time_unit_in_years) or time_unit_in_years <= 0.0:
        raise ValueError("time_unit_in_years must be finite and positive")
    ssp_lookback_max_gyr = _validate_lookback_max_myr(
        "ssp_lookback_max_myr",
        ssp_lookback_max_myr,
        t_obs=t_obs,
    )

    t_sorted, mh_sorted, sfr_sorted = _prepare_sorted_history(
        t_history=t_history_array,
        mh_history=mh_history_array,
        sfr_history=sfr_history_array,
    )
    ssp_order = np.argsort(legacy_ssp_age_gyr, kind="stable")
    legacy_ssp_age_gyr = legacy_ssp_age_gyr[ssp_order]
    ssp_luv = ssp_luv[ssp_order]
    if np.any(np.diff(legacy_ssp_age_gyr) <= 0.0):
        raise ValueError("ssp_age_grid must be strictly increasing after sorting")
    if not (t_sorted[0] <= t_obs <= t_sorted[-1]):
        raise ValueError("t_obs must lie within the covered t_history range")

    t_cross = _find_mass_crossing_time(t_sorted, mh_sorted, M_min)
    ti = max(t_z50, t_cross) if t_cross is not None else t_z50
    ti = max(ti, t_sorted[0])
    ti = max(ti, t_obs - ssp_lookback_max_gyr)

    if ti >= t_obs:
        details = {
            "L_uv_halo": 0.0,
            "ti": ti,
            "mask_used": np.zeros_like(t_sorted, dtype=bool),
            "age_used": np.array([], dtype=float),
            "t_used": np.array([], dtype=float),
            "kernel_used": np.array([], dtype=float),
            "integrand_used": np.array([], dtype=float),
            "t_cross_Mmin": t_cross,
        }
        if return_details:
            return details
        return 0.0

    mask_used = (t_sorted >= ti) & (t_sorted <= t_obs)
    t_used, sfr_used = _augment_with_boundaries(t_sorted, sfr_sorted, lower=ti, upper=t_obs)
    age_used = np.maximum(t_obs - t_used, 0.0)
    kernel_used = np.asarray(
        interpolate_ssp_luminosity(
            age_used,
            ssp_age_grid=legacy_ssp_age_gyr,
            ssp_luv_grid=ssp_luv,
        ),
        dtype=float,
    )
    integrand_used = sfr_used * kernel_used
    l_uv_halo = float(
        compute_final_ssp_observable_from_sfr_grid(
            t_grid_gyr=t_used[None, :],
            sfr_grid=sfr_used[None, :],
            active_grid=np.ones((1, t_used.size), dtype=bool),
            ssp_age_myr=legacy_ssp_age_gyr * 1.0e3,
            ssp_observable_per_msun=ssp_luv,
            lookback_max_myr=ssp_lookback_max_myr,
            time_unit_in_years=time_unit_in_years,
        )[0]
    )

    details = {
        "L_uv_halo": l_uv_halo,
        "ti": ti,
        "ssp_lookback_max_myr": float(ssp_lookback_max_myr),
        "mask_used": mask_used,
        "age_used": age_used,
        "t_used": t_used,
        "kernel_used": kernel_used,
        "integrand_used": integrand_used,
        "t_cross_Mmin": t_cross,
    }
    if return_details:
        return details
    return l_uv_halo
