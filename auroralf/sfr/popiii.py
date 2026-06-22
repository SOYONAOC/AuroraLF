from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from auroralf.cooling import (
    DEFAULT_LW_BACKGROUND_J21,
    compute_atomic_cooling_mass_msun,
    compute_popiii_lw_minimum_mass_msun,
)


YEARS_PER_GYR = 1.0e9
POPIII_UPPER_MASS_MODE_ATOMIC = "atomic"
POPIII_UPPER_MASS_MODE_FIXED = "fixed"
POPIII_UPPER_MASS_MODES = (POPIII_UPPER_MASS_MODE_ATOMIC, POPIII_UPPER_MASS_MODE_FIXED)


@dataclass(frozen=True)
class PopIIISFRParameters:
    epsilon_star: float = 1.0e-3
    pivot_mass_msun: float = 1.0e7
    alpha_star: float = 0.0
    beta_star: float = 0.0
    lw_background_j21: float = DEFAULT_LW_BACKGROUND_J21
    upper_mass_mode: str = POPIII_UPPER_MASS_MODE_ATOMIC
    upper_mass_msun: float | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "epsilon_star": float(self.epsilon_star),
            "pivot_mass_msun": float(self.pivot_mass_msun),
            "alpha_star": float(self.alpha_star),
            "beta_star": float(self.beta_star),
            "lw_background_j21": float(self.lw_background_j21),
            "upper_mass_mode": str(self.upper_mass_mode).strip().lower(),
            "upper_mass_msun": None if self.upper_mass_msun is None else float(self.upper_mass_msun),
        }


@dataclass(frozen=True)
class PopIIISFRGridResult:
    sfr_grid: np.ndarray
    fstar_grid: np.ndarray
    duty_cycle_grid: np.ndarray
    lower_mass_msun_grid: np.ndarray
    upper_mass_msun_grid: np.ndarray
    starforming_grid: np.ndarray


DEFAULT_POPIII_SFR_PARAMETERS = PopIIISFRParameters()


def _resolve_popiii_sfr_parameters(parameters: PopIIISFRParameters | None) -> PopIIISFRParameters:
    params = DEFAULT_POPIII_SFR_PARAMETERS if parameters is None else parameters
    if not 0.0 <= float(params.epsilon_star) <= 1.0:
        raise ValueError("epsilon_star must lie in [0, 1]")
    if float(params.pivot_mass_msun) <= 0.0:
        raise ValueError("pivot_mass_msun must be positive")
    if not np.isfinite(float(params.alpha_star)):
        raise ValueError("alpha_star must be finite")
    if not np.isfinite(float(params.beta_star)):
        raise ValueError("beta_star must be finite")
    if float(params.lw_background_j21) < 0.0 or not np.isfinite(float(params.lw_background_j21)):
        raise ValueError("lw_background_j21 must be finite and non-negative")

    upper_mass_mode = str(params.upper_mass_mode).strip().lower()
    if upper_mass_mode not in POPIII_UPPER_MASS_MODES:
        raise ValueError(f"upper_mass_mode must be one of {POPIII_UPPER_MASS_MODES}")
    if upper_mass_mode == POPIII_UPPER_MASS_MODE_FIXED:
        if params.upper_mass_msun is None:
            raise ValueError("upper_mass_msun must be provided when upper_mass_mode='fixed'")
        if float(params.upper_mass_msun) <= 0.0 or not np.isfinite(float(params.upper_mass_msun)):
            raise ValueError("upper_mass_msun must be finite and positive")
    return params


def _as_scalar_if_scalar_input(result: np.ndarray, *inputs: object) -> np.ndarray | float:
    if all(np.ndim(value) == 0 for value in inputs):
        return float(np.asarray(result, dtype=float))
    return np.asarray(result, dtype=float)


def compute_popiii_star_formation_efficiency(
    halo_mass_msun: np.ndarray | float,
    parameters: PopIIISFRParameters | None = None,
) -> np.ndarray | float:
    """Return the Venditti/Cruz Pop III star-formation efficiency."""

    params = _resolve_popiii_sfr_parameters(parameters)
    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo_mass_msun must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo_mass_msun must be positive")

    ratio = mass / float(params.pivot_mass_msun)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = ratio ** float(params.alpha_star) + ratio ** float(params.beta_star)
    efficiency = 2.0 * float(params.epsilon_star) / denominator
    if not np.all(np.isfinite(efficiency)):
        raise RuntimeError("Pop III star-formation efficiency calculation returned non-finite values")
    if np.any(efficiency < 0.0):
        raise RuntimeError("Pop III star-formation efficiency calculation returned negative values")
    return _as_scalar_if_scalar_input(efficiency, halo_mass_msun)


def compute_popiii_upper_mass_msun(
    z_obs: np.ndarray | float,
    parameters: PopIIISFRParameters | None = None,
) -> np.ndarray | float:
    params = _resolve_popiii_sfr_parameters(parameters)
    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    upper_mass_mode = str(params.upper_mass_mode).strip().lower()
    if upper_mass_mode == POPIII_UPPER_MASS_MODE_ATOMIC:
        return compute_atomic_cooling_mass_msun(redshift)
    if upper_mass_mode == POPIII_UPPER_MASS_MODE_FIXED:
        upper = np.full(redshift.shape, float(params.upper_mass_msun), dtype=float)
        return _as_scalar_if_scalar_input(upper, z_obs)
    raise RuntimeError(f"unsupported Pop III upper_mass_mode after validation: {upper_mass_mode}")


def compute_popiii_duty_cycle(
    halo_mass_msun: np.ndarray | float,
    z_obs: np.ndarray | float,
    parameters: PopIIISFRParameters | None = None,
) -> np.ndarray | float:
    """Return ``exp(-M_mol/Mh) exp(-Mh/M_up)`` for the Pop III occupation."""

    params = _resolve_popiii_sfr_parameters(parameters)
    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo_mass_msun must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo_mass_msun must be positive")

    lower_mass = np.asarray(
        compute_popiii_lw_minimum_mass_msun(
            z_obs,
            lw_background_j21=float(params.lw_background_j21),
        ),
        dtype=float,
    )
    upper_mass = np.asarray(compute_popiii_upper_mass_msun(z_obs, params), dtype=float)
    try:
        mass, lower_mass, upper_mass = np.broadcast_arrays(mass, lower_mass, upper_mass)
    except ValueError as exc:
        raise ValueError("halo_mass_msun and z_obs must be broadcast-compatible") from exc

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        duty_cycle = np.exp(-lower_mass / mass) * np.exp(-mass / upper_mass)
    if not np.all(np.isfinite(duty_cycle)):
        raise RuntimeError("Pop III duty-cycle calculation returned non-finite values")
    if np.any((duty_cycle < 0.0) | (duty_cycle > 1.0)):
        raise RuntimeError("Pop III duty-cycle calculation returned values outside [0, 1]")
    return _as_scalar_if_scalar_input(duty_cycle, halo_mass_msun, z_obs)


def compute_popiii_sfr_from_grids(
    *,
    mh_grid: np.ndarray,
    dmhdt_grid: np.ndarray,
    z_grid: np.ndarray,
    active_grid: np.ndarray,
    baryon_fraction: float,
    parameters: PopIIISFRParameters | None = None,
) -> PopIIISFRGridResult:
    """Compute Pop III SFR in ``Msun/yr`` on halo-history grids."""

    params = _resolve_popiii_sfr_parameters(parameters)
    mh = np.asarray(mh_grid, dtype=float)
    dmhdt = np.asarray(dmhdt_grid, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    if mh.shape != dmhdt.shape or mh.shape != active.shape:
        raise ValueError("mh_grid, dmhdt_grid, and active_grid must have identical shapes")
    if z.ndim == 1:
        if mh.ndim != 2 or z.size != mh.shape[1]:
            raise ValueError("1D z_grid length must match the time axis of mh_grid")
        z = np.broadcast_to(z[None, :], mh.shape)
    elif z.shape != mh.shape:
        raise ValueError("z_grid must either be 1D over time or match mh_grid shape")
    if float(baryon_fraction) <= 0.0 or float(baryon_fraction) > 1.0:
        raise ValueError("baryon_fraction must lie in (0, 1]")

    fstar = np.zeros_like(mh, dtype=float)
    duty = np.zeros_like(mh, dtype=float)
    lower_mass = np.asarray(
        compute_popiii_lw_minimum_mass_msun(z, lw_background_j21=float(params.lw_background_j21)),
        dtype=float,
    )
    upper_mass = np.asarray(compute_popiii_upper_mass_msun(z, params), dtype=float)
    source = active & np.isfinite(mh) & (mh > 0.0) & np.isfinite(dmhdt) & (dmhdt > 0.0) & np.isfinite(z)
    if np.any(source):
        fstar[source] = np.asarray(compute_popiii_star_formation_efficiency(mh[source], params), dtype=float)
        duty[source] = np.asarray(compute_popiii_duty_cycle(mh[source], z[source], params), dtype=float)

    sfr = np.zeros_like(mh, dtype=float)
    starforming = source & np.isfinite(fstar) & np.isfinite(duty) & (fstar > 0.0) & (duty > 0.0)
    sfr[starforming] = float(baryon_fraction) * fstar[starforming] * duty[starforming] * dmhdt[starforming]
    sfr[starforming] /= YEARS_PER_GYR
    return PopIIISFRGridResult(
        sfr_grid=sfr,
        fstar_grid=fstar,
        duty_cycle_grid=duty,
        lower_mass_msun_grid=lower_mass,
        upper_mass_msun_grid=upper_mass,
        starforming_grid=starforming,
    )


__all__ = [
    "DEFAULT_POPIII_SFR_PARAMETERS",
    "POPIII_UPPER_MASS_MODE_ATOMIC",
    "POPIII_UPPER_MASS_MODE_FIXED",
    "POPIII_UPPER_MASS_MODES",
    "PopIIISFRGridResult",
    "PopIIISFRParameters",
    "compute_popiii_duty_cycle",
    "compute_popiii_sfr_from_grids",
    "compute_popiii_star_formation_efficiency",
    "compute_popiii_upper_mass_msun",
]
