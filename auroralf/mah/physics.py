"""Analytic halo-growth and virial relations.

The MAH form is from McBride, Fakhouri & Ma (2009),
DOI: 10.1111/j.1365-2966.2009.15329.x, arXiv:0902.3659.  The virial
overdensity fit is from Bryan & Norman (1998), DOI: 10.1086/305262,
arXiv:astro-ph/9710107.
"""

from __future__ import annotations

import numpy as np

from .models import Cosmology


BRYAN_NORMAN_REFERENCE_OVERDENSITY = 18.0 * np.pi**2
ATOMIC_COOLING_TEMPERATURE_K = 1.0e4
ATOMIC_COOLING_MU = 0.61
VIRIAL_TEMPERATURE_NORMALIZATION_K = 1.98e4
VIRIAL_MASS_NORMALIZATION_MSUN = 1.0e8
VIRIAL_MU_NORMALIZATION = 0.6
VIRIAL_REDSHIFT_NORMALIZATION = 10.0


def compute_bryan_norman_virial_terms(
    redshift: float | np.ndarray,
    *,
    cosmology: Cosmology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``E(z)^2``, ``Omega_m(z)``, and the Bryan--Norman overdensity.

    Direct formula: Bryan & Norman (1998), Eq. 6 for a flat cosmology.
    """

    redshift = np.asarray(redshift, dtype=float)
    matter_term = cosmology.omega_m * (1.0 + redshift) ** 3
    expansion_squared = matter_term + cosmology.omega_lambda
    omega_m_at_redshift = matter_term / expansion_squared
    density_offset = omega_m_at_redshift - 1.0
    virial_overdensity = (
        BRYAN_NORMAN_REFERENCE_OVERDENSITY + 82.0 * density_offset - 39.0 * density_offset**2
    )
    return expansion_squared, omega_m_at_redshift, virial_overdensity


def compute_atomic_cooling_mass_msun(
    z_obs: np.ndarray | float,
    *,
    cosmology: Cosmology,
    virial_temperature_k: float = ATOMIC_COOLING_TEMPERATURE_K,
    mu: float = ATOMIC_COOLING_MU,
) -> np.ndarray | float:
    """Return the Barkana--Loeb atomic-cooling halo-mass threshold.

    This is the algebraic inverse of the Barkana & Loeb (2001) virial
    temperature relation, using the Bryan & Norman (1998) spherical-collapse
    overdensity.  The returned mass is in ``Msun``.
    """

    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")

    temperature_k = float(virial_temperature_k)
    mean_molecular_weight = float(mu)
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("virial_temperature_k must be finite and positive")
    if not np.isfinite(mean_molecular_weight) or mean_molecular_weight <= 0.0:
        raise ValueError("mu must be finite and positive")

    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    h = cosmology.h0_km_s_mpc / 100.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        _, omega_m_at_redshift, collapse_overdensity = compute_bryan_norman_virial_terms(
            redshift,
            cosmology=cosmology,
        )
        virial_correction = (
            cosmology.omega_m
            / omega_m_at_redshift
            * collapse_overdensity
            / BRYAN_NORMAN_REFERENCE_OVERDENSITY
        )
        threshold = (
            VIRIAL_MASS_NORMALIZATION_MSUN
            / h
            * (temperature_k / VIRIAL_TEMPERATURE_NORMALIZATION_K) ** 1.5
            * (mean_molecular_weight / VIRIAL_MU_NORMALIZATION) ** -1.5
            * virial_correction**-0.5
            * ((1.0 + redshift) / VIRIAL_REDSHIFT_NORMALIZATION) ** -1.5
        )
    if not np.all(np.isfinite(threshold)):
        raise RuntimeError("atomic cooling mass calculation returned non-finite masses")
    if np.any(threshold <= 0.0):
        raise RuntimeError("atomic cooling mass calculation returned non-positive masses")
    if np.ndim(z_obs) == 0:
        return float(threshold)
    return np.asarray(threshold, dtype=float)


def mass_history(
    redshift: float | np.ndarray,
    redshift_final: float,
    mass_final: float,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Evaluate the McBride et al. (2009) MAH ``M(z)`` parameterization."""
    redshift = np.asarray(redshift, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if redshift.ndim == 0:
        ratio = (1.0 + redshift) / (1.0 + redshift_final)
        return mass_final * ratio**beta * np.exp(-gamma * (redshift - redshift_final))

    ratio = (1.0 + redshift[None, :]) / (1.0 + redshift_final)
    delta_z = redshift[None, :] - redshift_final
    return mass_final * ratio**beta[:, None] * np.exp(-gamma[:, None] * delta_z)


def accretion_rate(
    redshift: np.ndarray,
    redshift_final: float,
    mass_final: float,
    beta: np.ndarray,
    gamma: np.ndarray,
    cosmology: Cosmology,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the McBride MAH analytically to obtain ``dMh/dt``."""
    mass = mass_history(redshift, redshift_final, mass_final, beta, gamma)
    one_plus_z = 1.0 + redshift[None, :]
    mdot = -cosmology.hubble(redshift)[None, :] * one_plus_z * mass * (beta[:, None] / one_plus_z - gamma[:, None])
    return mass, mdot
