from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from auroralf.constants import SOLAR_OXYGEN_ABUNDANCE
from ._stellar_mass import cumulative_surviving_stellar_mass_msun

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
    if parameters.scatter_dex < 0.0:
        raise ValueError("scatter_dex must be non-negative")
    if parameters.stellar_mass_floor_msun <= 0.0:
        raise ValueError("stellar_mass_floor_msun must be positive")


def _mzr_oxygen_abundance(
    logmstar: np.ndarray,
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
    z = np.asarray(z_grid, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid, dtype=bool)
    stellar_mass = cumulative_surviving_stellar_mass_msun(
        t_grid_gyr=t_grid,
        sfr_grid=sfr,
        active_grid=active,
        returned_fraction=float(params.returned_fraction),
    )
    shape = stellar_mass.shape
    if z.shape != shape:
        raise ValueError(f"z_grid must have shape {shape}, got {z.shape}")
    birth_zsun = np.zeros(shape, dtype=float)
    mzr_mass = np.maximum(stellar_mass[active], float(params.stellar_mass_floor_msun))
    oh12 = _mzr_oxygen_abundance(np.log10(mzr_mass), params)
    birth_zsun[active] = np.power(10.0, oh12 - SOLAR_OXYGEN_ABUNDANCE)
    if params.scatter_dex > 0.0:
        rng = np.random.default_rng(random_seed)
        birth_zsun[active] *= np.power(
            10.0,
            rng.normal(loc=0.0, scale=float(params.scatter_dex), size=np.count_nonzero(active)),
        )

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
