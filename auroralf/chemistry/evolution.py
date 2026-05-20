from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


SOLAR_METALLICITY_MASS_FRACTION = 0.0142
CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER = 1.28


@dataclass(frozen=True)
class MetalEnrichmentParameters:
    """Parameters for stochastic one-zone metal enrichment along fixed MAHs."""

    solar_metallicity_mass_fraction: float = SOLAR_METALLICITY_MASS_FRACTION
    gas_fraction_of_baryons: float = 0.5
    initial_metallicity_zsun: float = 0.0
    inflow_metallicity_zsun: float = 0.0
    returned_fraction: float = 0.4
    metal_yield: float = 0.02
    topheavy_yield_multiplier: float = 1.0
    mass_loading_norm: float = 5.0
    mass_loading_mass_scale_msun: float = 1.0e10
    mass_loading_mass_slope: float = -0.35
    mass_loading_redshift_scale: float = 10.0
    mass_loading_redshift_slope: float = 0.0
    yield_scatter_dex: float = 0.2
    mass_loading_scatter_dex: float = 0.3
    birth_metallicity_scatter_dex: float = 0.15

    def as_metadata(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MetallicityEvolutionResult:
    gas_mass_grid: np.ndarray
    metal_mass_grid: np.ndarray
    gas_metallicity_zsun_grid: np.ndarray
    birth_metallicity_zsun_grid: np.ndarray
    mass_loading_grid: np.ndarray
    effective_yield_grid: np.ndarray
    topheavy_source_grid: np.ndarray
    parameters: MetalEnrichmentParameters


def _validate_parameters(parameters: MetalEnrichmentParameters) -> None:
    if parameters.solar_metallicity_mass_fraction <= 0.0:
        raise ValueError("solar_metallicity_mass_fraction must be positive")
    if parameters.gas_fraction_of_baryons <= 0.0:
        raise ValueError("gas_fraction_of_baryons must be positive")
    if parameters.initial_metallicity_zsun < 0.0:
        raise ValueError("initial_metallicity_zsun must be non-negative")
    if parameters.inflow_metallicity_zsun < 0.0:
        raise ValueError("inflow_metallicity_zsun must be non-negative")
    if not 0.0 <= parameters.returned_fraction < 1.0:
        raise ValueError("returned_fraction must lie in [0, 1)")
    if parameters.metal_yield < 0.0:
        raise ValueError("metal_yield must be non-negative")
    if parameters.topheavy_yield_multiplier <= 0.0:
        raise ValueError("topheavy_yield_multiplier must be positive")
    if parameters.mass_loading_norm < 0.0:
        raise ValueError("mass_loading_norm must be non-negative")
    if parameters.mass_loading_mass_scale_msun <= 0.0:
        raise ValueError("mass_loading_mass_scale_msun must be positive")
    if parameters.mass_loading_redshift_scale <= 0.0:
        raise ValueError("mass_loading_redshift_scale must be positive")
    if parameters.yield_scatter_dex < 0.0:
        raise ValueError("yield_scatter_dex must be non-negative")
    if parameters.mass_loading_scatter_dex < 0.0:
        raise ValueError("mass_loading_scatter_dex must be non-negative")
    if parameters.birth_metallicity_scatter_dex < 0.0:
        raise ValueError("birth_metallicity_scatter_dex must be non-negative")


def _as_matching_float_grid(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {grid.shape}")
    return grid


def _as_matching_bool_grid(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(values, dtype=bool)
    if grid.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {grid.shape}")
    return grid


def _lognormal_factor(rng: np.random.Generator, scatter_dex: float) -> float:
    if scatter_dex == 0.0:
        return 1.0
    return float(10.0 ** rng.normal(loc=0.0, scale=float(scatter_dex)))


def _mass_loading_factor(
    halo_mass: float,
    redshift: float,
    parameters: MetalEnrichmentParameters,
    rng: np.random.Generator,
) -> float:
    if parameters.mass_loading_norm == 0.0:
        return 0.0
    if not np.isfinite(halo_mass) or halo_mass <= 0.0:
        return 0.0
    redshift_factor = (1.0 + max(float(redshift), 0.0)) / parameters.mass_loading_redshift_scale
    eta = parameters.mass_loading_norm
    eta *= (float(halo_mass) / parameters.mass_loading_mass_scale_msun) ** parameters.mass_loading_mass_slope
    eta *= redshift_factor**parameters.mass_loading_redshift_slope
    eta *= _lognormal_factor(rng, parameters.mass_loading_scatter_dex)
    return float(max(eta, 0.0))


def evolve_stochastic_metallicity(
    *,
    t_grid_gyr: np.ndarray,
    z_grid: np.ndarray,
    mh_grid: np.ndarray,
    dmhdt_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    baryon_fraction: float,
    parameters: MetalEnrichmentParameters | None = None,
    random_seed: int | None = None,
    topheavy_source_grid: np.ndarray | None = None,
    topheavy_birth_metallicity_max_zsun: float | None = None,
) -> MetallicityEvolutionResult:
    """Evolve stochastic gas metallicity histories on fixed halo tracks.

    ``SFR`` is treated as an externally supplied source term. The gas reservoir is
    diagnostic and scales with halo baryonic mass, so this module tracks metal
    enrichment without changing the SFR model. If ``topheavy_birth_metallicity_max_zsun``
    is provided, ``topheavy_source_grid`` is interpreted as a candidate grid and
    is filtered by the pre-step birth metallicity before applying the top-heavy
    yield multiplier.
    """

    params = MetalEnrichmentParameters() if parameters is None else parameters
    _validate_parameters(params)
    if not 0.0 < float(baryon_fraction) <= 1.0:
        raise ValueError("baryon_fraction must lie in (0, 1]")

    t_grid = np.asarray(t_grid_gyr, dtype=float)
    if t_grid.ndim != 2:
        raise ValueError("t_grid_gyr must be a 2D array")
    shape = t_grid.shape
    if shape[1] < 2:
        raise ValueError("t_grid_gyr must contain at least two time steps")

    z_grid = _as_matching_float_grid("z_grid", z_grid, shape)
    mh_grid = _as_matching_float_grid("mh_grid", mh_grid, shape)
    dmhdt_grid = _as_matching_float_grid("dmhdt_grid", dmhdt_grid, shape)
    sfr_grid = _as_matching_float_grid("sfr_grid", sfr_grid, shape)
    active_grid = _as_matching_bool_grid("active_grid", active_grid, shape)
    if topheavy_birth_metallicity_max_zsun is not None and topheavy_source_grid is None:
        raise ValueError("topheavy_source_grid must be provided when topheavy_birth_metallicity_max_zsun is set")
    if topheavy_birth_metallicity_max_zsun is not None and float(topheavy_birth_metallicity_max_zsun) <= 0.0:
        raise ValueError("topheavy_birth_metallicity_max_zsun must be positive when provided")
    if topheavy_source_grid is None:
        topheavy_candidate_grid = np.zeros(shape, dtype=bool)
    else:
        topheavy_candidate_grid = _as_matching_bool_grid("topheavy_source_grid", topheavy_source_grid, shape)

    if not np.all(np.diff(t_grid, axis=1) >= 0.0):
        raise ValueError("t_grid_gyr must be monotonic non-decreasing along the time axis")

    rng = np.random.default_rng(random_seed)
    gas_mass = np.zeros(shape, dtype=float)
    metal_mass = np.zeros(shape, dtype=float)
    gas_zsun = np.zeros(shape, dtype=float)
    birth_zsun = np.zeros(shape, dtype=float)
    mass_loading = np.zeros(shape, dtype=float)
    effective_yield = np.zeros(shape, dtype=float)
    topheavy_grid = np.zeros(shape, dtype=bool)

    z_sun = float(params.solar_metallicity_mass_fraction)
    initial_z_abs = float(params.initial_metallicity_zsun) * z_sun
    inflow_z_abs = float(params.inflow_metallicity_zsun) * z_sun
    baryon_fraction_value = float(baryon_fraction)

    for halo_index in range(shape[0]):
        initial_halo_mass = float(mh_grid[halo_index, 0])
        if not np.isfinite(initial_halo_mass) or initial_halo_mass <= 0.0:
            raise ValueError("mh_grid must contain positive finite initial halo masses")
        previous_gas_mass = (
            params.gas_fraction_of_baryons
            * baryon_fraction_value
            * initial_halo_mass
        )
        current_metal_mass = previous_gas_mass * initial_z_abs

        for step_index in range(shape[1]):
            current_halo_mass = float(mh_grid[halo_index, step_index])
            if not np.isfinite(current_halo_mass) or current_halo_mass <= 0.0:
                raise ValueError("mh_grid must contain positive finite halo masses")

            current_gas_mass = (
                params.gas_fraction_of_baryons
                * baryon_fraction_value
                * current_halo_mass
            )
            if current_gas_mass <= 0.0:
                raise ValueError("diagnostic gas mass must be positive")

            gas_mass[halo_index, step_index] = current_gas_mass
            birth_z_abs = current_metal_mass / current_gas_mass
            birth_factor = _lognormal_factor(rng, params.birth_metallicity_scatter_dex)
            birth_zsun_value = birth_z_abs / z_sun * birth_factor
            birth_zsun[halo_index, step_index] = birth_zsun_value
            candidate_topheavy = bool(topheavy_candidate_grid[halo_index, step_index])
            if candidate_topheavy and topheavy_birth_metallicity_max_zsun is None:
                topheavy_grid[halo_index, step_index] = True
            elif candidate_topheavy:
                topheavy_grid[halo_index, step_index] = birth_zsun_value <= float(topheavy_birth_metallicity_max_zsun)

            if step_index > 0:
                dt_gyr = float(t_grid[halo_index, step_index] - t_grid[halo_index, step_index - 1])
                if dt_gyr < 0.0:
                    raise ValueError("t_grid_gyr must be monotonic non-decreasing along the time axis")

                active = bool(active_grid[halo_index, step_index])
                sfr = float(sfr_grid[halo_index, step_index])
                if dt_gyr > 0.0 and active and np.isfinite(sfr) and sfr > 0.0:
                    halo_mass_growth = max(float(dmhdt_grid[halo_index, step_index]), 0.0) * dt_gyr
                    reservoir_growth = max(current_gas_mass - previous_gas_mass, 0.0)
                    inflow_mass = max(baryon_fraction_value * halo_mass_growth, reservoir_growth)
                    formed_stellar_mass = sfr * dt_gyr * 1.0e9
                    eta = _mass_loading_factor(
                        current_halo_mass,
                        float(z_grid[halo_index, step_index]),
                        params,
                        rng,
                    )
                    mass_loading[halo_index, step_index] = eta

                    yield_multiplier = (
                        params.topheavy_yield_multiplier if topheavy_grid[halo_index, step_index] else 1.0
                    )
                    yield_eff = params.metal_yield * yield_multiplier * _lognormal_factor(
                        rng,
                        params.yield_scatter_dex,
                    )
                    effective_yield[halo_index, step_index] = yield_eff

                    removed_gas_mass = (1.0 - params.returned_fraction + eta) * formed_stellar_mass
                    metal_loss_fraction = 1.0 - float(np.exp(-removed_gas_mass / current_gas_mass))
                    metal_loss = current_metal_mass * metal_loss_fraction
                    new_metals = yield_eff * formed_stellar_mass
                    inflow_metals = inflow_z_abs * inflow_mass
                    current_metal_mass = max(current_metal_mass + inflow_metals + new_metals - metal_loss, 0.0)

            metal_mass[halo_index, step_index] = current_metal_mass
            gas_zsun[halo_index, step_index] = current_metal_mass / current_gas_mass / z_sun
            previous_gas_mass = current_gas_mass

    return MetallicityEvolutionResult(
        gas_mass_grid=gas_mass,
        metal_mass_grid=metal_mass,
        gas_metallicity_zsun_grid=gas_zsun,
        birth_metallicity_zsun_grid=birth_zsun,
        mass_loading_grid=mass_loading,
        effective_yield_grid=effective_yield,
        topheavy_source_grid=topheavy_grid,
        parameters=params,
    )
