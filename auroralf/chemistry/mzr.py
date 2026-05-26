from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


SOLAR_OXYGEN_ABUNDANCE = 8.69
MZR_RELATION_FIRE2_HIGHZ = "fire2_highz"
MZR_RELATION_JADES_LOWMASS = "jades_lowmass"
MZR_RELATIONS = (MZR_RELATION_FIRE2_HIGHZ, MZR_RELATION_JADES_LOWMASS)


@dataclass(frozen=True)
class MZRBirthMetallicityParameters:
    """Parameters for assigning source-time birth metallicity from an MZR prior."""

    relation: str = MZR_RELATION_FIRE2_HIGHZ
    returned_fraction: float = 0.4
    scatter_dex: float = 0.0
    stellar_mass_floor_msun: float = 1.0e6

    def as_metadata(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class MZRBirthMetallicityResult:
    stellar_mass_msun_grid: np.ndarray
    birth_metallicity_zsun_grid: np.ndarray
    active_grid: np.ndarray
    parameters: MZRBirthMetallicityParameters


def equivalent_oxygen_abundance_from_zsun(
    metallicity_zsun: np.ndarray | float,
    *,
    solar_oxygen_abundance: float = SOLAR_OXYGEN_ABUNDANCE,
) -> np.ndarray:
    metallicity = np.asarray(metallicity_zsun, dtype=float)
    if np.any(metallicity <= 0.0):
        raise ValueError("metallicity_zsun must be positive")
    return float(solar_oxygen_abundance) + np.log10(metallicity)


def fire2_highz_mzr_oh12(logmstar: np.ndarray | float) -> np.ndarray:
    logmstar_array = np.asarray(logmstar, dtype=float)
    return SOLAR_OXYGEN_ABUNDANCE + 0.37 * logmstar_array - 4.3


def jades_lowmass_mzr_oh12(logmstar: np.ndarray | float) -> np.ndarray:
    logmstar_array = np.asarray(logmstar, dtype=float)
    return 7.72 + 0.17 * (logmstar_array - 8.0)


def _validate_mzr_parameters(parameters: MZRBirthMetallicityParameters) -> None:
    if parameters.relation not in MZR_RELATIONS:
        raise ValueError(f"relation must be one of {MZR_RELATIONS}, got {parameters.relation!r}")
    if not 0.0 <= parameters.returned_fraction < 1.0:
        raise ValueError("returned_fraction must lie in [0, 1)")
    if parameters.scatter_dex < 0.0:
        raise ValueError("scatter_dex must be non-negative")
    if parameters.stellar_mass_floor_msun <= 0.0:
        raise ValueError("stellar_mass_floor_msun must be positive")


def _mzr_oxygen_abundance(
    logmstar: np.ndarray,
    z_grid: np.ndarray,
    parameters: MZRBirthMetallicityParameters,
) -> np.ndarray:
    if parameters.relation == MZR_RELATION_FIRE2_HIGHZ:
        return fire2_highz_mzr_oh12(logmstar)
    if parameters.relation == MZR_RELATION_JADES_LOWMASS:
        return jades_lowmass_mzr_oh12(logmstar)
    raise RuntimeError(f"unsupported MZR relation after validation: {parameters.relation}")


def compute_mzr_birth_metallicity(
    *,
    t_grid_gyr: np.ndarray,
    z_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    parameters: MZRBirthMetallicityParameters | None = None,
    random_seed: int | None = None,
) -> MZRBirthMetallicityResult:
    """Assign source-time birth metallicities from an empirical high-z MZR.

    The stellar mass used by the MZR is the cumulative surviving stellar mass
    through the current source-time cell, computed from the externally supplied
    SFR history. The MZR prior therefore supplies only ``Z_birth(t)`` for the
    IMF gate and does not feed back into the SFR or gas reservoir.
    """

    params = MZRBirthMetallicityParameters() if parameters is None else parameters
    _validate_mzr_parameters(params)

    t_grid = np.asarray(t_grid_gyr, dtype=float)
    if t_grid.ndim != 2:
        raise ValueError("t_grid_gyr must be a 2D array")
    if t_grid.shape[1] < 2:
        raise ValueError("t_grid_gyr must contain at least two time steps")
    shape = t_grid.shape

    z = np.asarray(z_grid, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    if z.shape != shape:
        raise ValueError(f"z_grid must have shape {shape}, got {z.shape}")
    if sfr.shape != shape:
        raise ValueError(f"sfr_grid must have shape {shape}, got {sfr.shape}")
    if active.shape != shape:
        raise ValueError(f"active_grid must have shape {shape}, got {active.shape}")
    if not np.all(np.diff(t_grid, axis=1) >= 0.0):
        raise ValueError("t_grid_gyr must be monotonic non-decreasing along the time axis")
    invalid_active_sfr = active & (~np.isfinite(sfr) | (sfr < 0.0))
    if np.any(invalid_active_sfr):
        raise ValueError("sfr_grid must be finite and non-negative for active source times")

    stellar_mass = np.zeros(shape, dtype=float)
    birth_zsun = np.zeros(shape, dtype=float)
    rng = np.random.default_rng(random_seed)
    surviving_fraction = 1.0 - float(params.returned_fraction)

    for halo_index in range(shape[0]):
        cumulative_stellar_mass = 0.0
        for step_index in range(shape[1]):
            if step_index > 0:
                dt_gyr = float(t_grid[halo_index, step_index] - t_grid[halo_index, step_index - 1])
                if dt_gyr < 0.0:
                    raise ValueError("t_grid_gyr must be monotonic non-decreasing along the time axis")
                if dt_gyr > 0.0 and bool(active[halo_index, step_index]):
                    cumulative_stellar_mass += (
                        surviving_fraction
                        * float(sfr[halo_index, step_index])
                        * dt_gyr
                        * 1.0e9
                    )

            stellar_mass[halo_index, step_index] = cumulative_stellar_mass
            if not bool(active[halo_index, step_index]):
                continue
            mzr_mass = max(cumulative_stellar_mass, float(params.stellar_mass_floor_msun))
            logmstar = np.array([np.log10(mzr_mass)], dtype=float)
            oh12 = _mzr_oxygen_abundance(
                logmstar,
                np.array([z[halo_index, step_index]], dtype=float),
                params,
            )
            metallicity_zsun = float(10.0 ** (oh12[0] - SOLAR_OXYGEN_ABUNDANCE))
            if params.scatter_dex > 0.0:
                metallicity_zsun *= float(10.0 ** rng.normal(loc=0.0, scale=float(params.scatter_dex)))
            birth_zsun[halo_index, step_index] = metallicity_zsun

    return MZRBirthMetallicityResult(
        stellar_mass_msun_grid=stellar_mass,
        birth_metallicity_zsun_grid=birth_zsun,
        active_grid=active,
        parameters=params,
    )


def max_positive_mzr_offset_dex(model_oh12: np.ndarray, reference_oh12: np.ndarray) -> float:
    model = np.asarray(model_oh12, dtype=float)
    reference = np.asarray(reference_oh12, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(f"model_oh12 and reference_oh12 must have the same shape, got {model.shape} and {reference.shape}")
    offset = model - reference
    finite = offset[np.isfinite(offset)]
    if finite.size == 0:
        raise ValueError("offset arrays contain no finite values")
    return float(max(np.max(finite), 0.0))
