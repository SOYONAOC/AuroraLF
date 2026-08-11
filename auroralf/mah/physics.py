from __future__ import annotations

import numpy as np

from .models import Cosmology


BRYAN_NORMAN_REFERENCE_OVERDENSITY = 18.0 * np.pi**2


def compute_bryan_norman_virial_terms(
    redshift: float | np.ndarray,
    *,
    cosmology: Cosmology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``E(z)^2``, ``Omega_m(z)``, and the Bryan--Norman overdensity."""

    redshift = np.asarray(redshift, dtype=float)
    matter_term = cosmology.omega_m * (1.0 + redshift) ** 3
    expansion_squared = matter_term + cosmology.omega_lambda
    omega_m_at_redshift = matter_term / expansion_squared
    density_offset = omega_m_at_redshift - 1.0
    virial_overdensity = (
        BRYAN_NORMAN_REFERENCE_OVERDENSITY + 82.0 * density_offset - 39.0 * density_offset**2
    )
    return expansion_squared, omega_m_at_redshift, virial_overdensity


def mass_history(
    redshift: float | np.ndarray,
    redshift_final: float,
    mass_final: float,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
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
    mass = mass_history(redshift, redshift_final, mass_final, beta, gamma)
    one_plus_z = 1.0 + redshift[None, :]
    mdot = -cosmology.hubble(redshift)[None, :] * one_plus_z * mass * (beta[:, None] / one_plus_z - gamma[:, None])
    return mass, mdot
