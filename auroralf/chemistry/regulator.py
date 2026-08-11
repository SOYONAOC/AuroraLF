from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from auroralf.constants import SOLAR_METALLICITY_MASS_FRACTION
from auroralf.mah.models import Cosmology

from ._stellar_mass import cumulative_surviving_stellar_mass_msun


@dataclass(frozen=True)
class RegulatorMetallicityParameters:
    """Parameters for the algebraic gas-regulator metallicity closure."""

    solar_metallicity_mass_fraction: float = SOLAR_METALLICITY_MASS_FRACTION
    # Historical API name; physically this is f_res = Mgas / (fb Mh),
    # not the galaxy gas fraction Mgas / (Mgas + Mstar).
    gas_fraction_norm: float = 0.02
    gas_fraction_mass_scale_msun: float = 1.0e10
    gas_fraction_mass_slope: float = 0.0
    gas_fraction_redshift_scale: float = 10.0
    gas_fraction_redshift_slope: float = 0.0
    returned_fraction: float = 0.4
    metal_yield: float = 0.01
    inflow_metallicity_zsun: float = 0.0
    metal_loading_norm: float = 20.0
    metal_loading_mass_scale_msun: float = 1.0e10
    metal_loading_mass_slope: float = -0.5
    metal_loading_redshift_scale: float = 10.0
    metal_loading_redshift_slope: float = 0.0
    stellar_mass_floor_msun: float = 0.0
    metallicity_scatter_dex: float = 0.0

    def as_metadata(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RegulatorMetallicityResult:
    stellar_mass_msun_grid: np.ndarray
    gas_mass_grid: np.ndarray
    gas_fraction_grid: np.ndarray
    metal_loading_grid: np.ndarray
    gas_metallicity_zsun_grid: np.ndarray
    birth_metallicity_zsun_grid: np.ndarray
    metal_mass_grid: np.ndarray
    active_grid: np.ndarray
    parameters: RegulatorMetallicityParameters


def _validate_parameters(parameters: RegulatorMetallicityParameters) -> None:
    if parameters.solar_metallicity_mass_fraction <= 0.0:
        raise ValueError("solar_metallicity_mass_fraction must be positive")
    if parameters.gas_fraction_norm <= 0.0:
        raise ValueError("gas_fraction_norm must be positive")
    if parameters.gas_fraction_mass_scale_msun <= 0.0:
        raise ValueError("gas_fraction_mass_scale_msun must be positive")
    if parameters.gas_fraction_redshift_scale <= 0.0:
        raise ValueError("gas_fraction_redshift_scale must be positive")
    if parameters.metal_yield < 0.0:
        raise ValueError("metal_yield must be non-negative")
    if parameters.inflow_metallicity_zsun < 0.0:
        raise ValueError("inflow_metallicity_zsun must be non-negative")
    if parameters.metal_loading_norm < 0.0:
        raise ValueError("metal_loading_norm must be non-negative")
    if parameters.metal_loading_mass_scale_msun <= 0.0:
        raise ValueError("metal_loading_mass_scale_msun must be positive")
    if parameters.metal_loading_redshift_scale <= 0.0:
        raise ValueError("metal_loading_redshift_scale must be positive")
    if parameters.stellar_mass_floor_msun < 0.0:
        raise ValueError("stellar_mass_floor_msun must be non-negative")
    if parameters.metallicity_scatter_dex < 0.0:
        raise ValueError("metallicity_scatter_dex must be non-negative")


def _as_matching_float_grid(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain finite values")
    return grid


def _as_matching_bool_grid(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(values, dtype=bool)
    if grid.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {grid.shape}")
    return grid


def _lognormal_factor_grid(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int],
    scatter_dex: float,
) -> np.ndarray:
    if scatter_dex == 0.0:
        return np.ones(shape, dtype=float)
    return np.power(10.0, rng.normal(loc=0.0, scale=float(scatter_dex), size=shape))


def _gas_fraction_grid(
    mh_grid: np.ndarray,
    z_grid: np.ndarray,
    parameters: RegulatorMetallicityParameters,
) -> np.ndarray:
    gas_fraction = float(parameters.gas_fraction_norm)
    gas_fraction *= np.power(mh_grid / float(parameters.gas_fraction_mass_scale_msun), parameters.gas_fraction_mass_slope)
    gas_fraction *= np.power(
        (1.0 + np.maximum(z_grid, 0.0)) / float(parameters.gas_fraction_redshift_scale),
        parameters.gas_fraction_redshift_slope,
    )
    if not np.all(np.isfinite(gas_fraction)):
        raise ValueError("computed gas fraction grid must contain finite values")
    if np.any(gas_fraction <= 0.0) or np.any(gas_fraction > 1.0):
        raise ValueError("computed gas fraction must lie in (0, 1]; adjust regulator gas-fraction parameters")
    return gas_fraction


def _metal_loading_grid(
    mh_grid: np.ndarray,
    z_grid: np.ndarray,
    parameters: RegulatorMetallicityParameters,
) -> np.ndarray:
    loading = float(parameters.metal_loading_norm)
    loading *= np.power(mh_grid / float(parameters.metal_loading_mass_scale_msun), parameters.metal_loading_mass_slope)
    loading *= np.power(
        (1.0 + np.maximum(z_grid, 0.0)) / float(parameters.metal_loading_redshift_scale),
        parameters.metal_loading_redshift_slope,
    )
    if not np.all(np.isfinite(loading)):
        raise ValueError("computed metal loading grid must contain finite values")
    if np.any(loading < 0.0):
        raise ValueError("computed metal loading grid must be non-negative")
    return loading


def compute_regulator_metallicity(
    *,
    t_grid_gyr: np.ndarray,
    z_grid: np.ndarray,
    mh_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    cosmology: Cosmology,
    parameters: RegulatorMetallicityParameters | None = None,
    random_seed: int | None = None,
) -> RegulatorMetallicityResult:
    """Compute gas metallicity from the algebraic gas-regulator closure.

    The supplied SFR is treated as fixed. The closure uses cumulative surviving
    stellar mass, a halo-baryon gas reservoir, and an effective metal-loading
    term to assign source-time gas metallicity without feeding back on SFR.
    """

    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    params = RegulatorMetallicityParameters() if parameters is None else parameters
    _validate_parameters(params)
    baryon_fraction = cosmology.omega_b / cosmology.omega_m

    t_grid = np.asarray(t_grid_gyr, dtype=float)
    shape = t_grid.shape

    z = _as_matching_float_grid("z_grid", z_grid, shape)
    mh = _as_matching_float_grid("mh_grid", mh_grid, shape)
    sfr = _as_matching_float_grid("sfr_grid", sfr_grid, shape)
    active = _as_matching_bool_grid("active_grid", active_grid, shape)
    if np.any(mh <= 0.0):
        raise ValueError("mh_grid must contain positive halo masses")
    stellar_mass = cumulative_surviving_stellar_mass_msun(
        t_grid_gyr=t_grid,
        sfr_grid=sfr,
        active_grid=active,
        returned_fraction=float(params.returned_fraction),
    )

    gas_fraction = _gas_fraction_grid(mh, z, params)
    gas_mass = gas_fraction * baryon_fraction * mh
    metal_loading = _metal_loading_grid(mh, z, params)

    denominator_stellar_mass = np.maximum(stellar_mass, float(params.stellar_mass_floor_msun))

    gas_to_stellar = np.full(shape, np.inf, dtype=float)
    positive_stellar = denominator_stellar_mass > 0.0
    gas_to_stellar[positive_stellar] = gas_mass[positive_stellar] / denominator_stellar_mass[positive_stellar]

    denominator = 1.0 + gas_to_stellar + metal_loading / (1.0 - float(params.returned_fraction))
    yield_term_abs = np.divide(
        float(params.metal_yield),
        denominator,
        out=np.zeros(shape, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )
    z_abs = float(params.inflow_metallicity_zsun) * float(params.solar_metallicity_mass_fraction) + yield_term_abs

    rng = np.random.default_rng(random_seed)
    zsun = z_abs / float(params.solar_metallicity_mass_fraction)
    zsun *= _lognormal_factor_grid(rng, shape=shape, scatter_dex=float(params.metallicity_scatter_dex))
    zsun = np.where(active, zsun, 0.0)
    if np.any(~np.isfinite(zsun)) or np.any(zsun < 0.0):
        raise RuntimeError("computed regulator metallicity must be finite and non-negative")

    metal_mass = zsun * float(params.solar_metallicity_mass_fraction) * gas_mass
    return RegulatorMetallicityResult(
        stellar_mass_msun_grid=stellar_mass,
        gas_mass_grid=gas_mass,
        gas_fraction_grid=gas_fraction,
        metal_loading_grid=metal_loading,
        gas_metallicity_zsun_grid=zsun,
        birth_metallicity_zsun_grid=zsun.copy(),
        metal_mass_grid=metal_mass,
        active_grid=active,
        parameters=params,
    )
