from __future__ import annotations

import numpy as np

from auroralf.mah.models import Cosmology


ATOMIC_COOLING_TEMPERATURE_K = 1.0e4
ATOMIC_COOLING_MU = 0.61
POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN = 3.3e7
POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT = -1.5
POPIII_LW_FEEDBACK_COEFFICIENT = 2.0
POPIII_LW_FEEDBACK_EXPONENT = 0.6
DEFAULT_LW_BACKGROUND_J21 = 0.0
STELLAR_CHANNEL_BELOW_POPIII_MIN = "below_popiii_min"
STELLAR_CHANNEL_POPIII = "popiii"
STELLAR_CHANNEL_POPII = "popii"
STELLAR_CHANNELS = (
    STELLAR_CHANNEL_BELOW_POPIII_MIN,
    STELLAR_CHANNEL_POPIII,
    STELLAR_CHANNEL_POPII,
)


def compute_atomic_cooling_mass_msun(
    z_obs: np.ndarray | float,
    *,
    cosmology: Cosmology,
    virial_temperature_k: float = ATOMIC_COOLING_TEMPERATURE_K,
    mu: float = ATOMIC_COOLING_MU,
) -> np.ndarray | float:
    """Return the halo mass corresponding to a virial temperature threshold."""

    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")

    if float(virial_temperature_k) <= 0.0:
        raise ValueError("virial_temperature_k must be positive")
    if float(mu) <= 0.0:
        raise ValueError("mu must be positive")

    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    try:
        import massfunc as mf
    except ImportError as exc:
        raise ImportError("atomic cooling mass requires the optional dependency massfunc") from exc

    threshold = np.asarray(
        mf.SFRD(
            h=cosmology.h0_km_s_mpc / 100.0,
            omegam=cosmology.omega_m,
        ).M_vir(float(mu), float(virial_temperature_k), redshift),
        dtype=float,
    )
    if threshold.shape != redshift.shape:
        threshold = np.broadcast_to(threshold, redshift.shape).copy()
    if not np.all(np.isfinite(threshold)):
        raise RuntimeError("massfunc.SFRD().M_vir returned non-finite atomic cooling masses")
    if np.any(threshold <= 0.0):
        raise RuntimeError("massfunc.SFRD().M_vir returned non-positive atomic cooling masses")
    if np.ndim(z_obs) == 0:
        return float(threshold)
    return threshold


def compute_popiii_lw_minimum_mass_msun(
    z_obs: np.ndarray | float,
    *,
    lw_background_j21: np.ndarray | float = DEFAULT_LW_BACKGROUND_J21,
) -> np.ndarray | float:
    """Return the Cruz/Venditti Pop III molecular-cooling mass.

    ``lw_background_j21`` is the LW intensity in units of
    ``1e-21 erg s^-1 cm^-2 Hz^-1 sr^-1``. The default zero background reduces
    to ``M0(z) = 3.3e7 * (1 + z)^(-3/2) Msun``, the Cruz/Zeus21
    zero-feedback molecular-cooling threshold for ``Tvir=1e3 K``.
    """

    redshift = np.asarray(z_obs, dtype=float)
    if not np.all(np.isfinite(redshift)):
        raise ValueError("z_obs must be finite")
    if np.any(redshift < 0.0):
        raise ValueError("z_obs must be non-negative")

    lw_background = np.asarray(lw_background_j21, dtype=float)
    if not np.all(np.isfinite(lw_background)):
        raise ValueError("lw_background_j21 must be finite")
    if np.any(lw_background < 0.0):
        raise ValueError("lw_background_j21 must be non-negative")

    try:
        redshift, lw_background = np.broadcast_arrays(redshift, lw_background)
    except ValueError as exc:
        raise ValueError("lw_background_j21 must be scalar or broadcast with z_obs") from exc

    molecular_floor = POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN * (
        1.0 + redshift
    ) ** POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT
    threshold = molecular_floor * (
        1.0 + POPIII_LW_FEEDBACK_COEFFICIENT * lw_background**POPIII_LW_FEEDBACK_EXPONENT
    )
    if not np.all(np.isfinite(threshold)):
        raise RuntimeError("Pop III LW minimum mass calculation returned non-finite masses")
    if np.any(threshold <= 0.0):
        raise RuntimeError("Pop III LW minimum mass calculation returned non-positive masses")
    if np.ndim(z_obs) == 0 and np.ndim(lw_background_j21) == 0:
        return float(threshold)
    return np.asarray(threshold, dtype=float)


def classify_halo_stellar_channels(
    halo_mass_msun: np.ndarray | float,
    *,
    cosmology: Cosmology,
    z_obs: np.ndarray | float,
    lw_background_j21: np.ndarray | float = DEFAULT_LW_BACKGROUND_J21,
    virial_temperature_k: float = ATOMIC_COOLING_TEMPERATURE_K,
    mu: float = ATOMIC_COOLING_MU,
) -> np.ndarray | str:
    """Classify halos into below-PopIII, Pop III minihalo, or Pop II channels."""

    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo masses must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo masses must be positive")

    threshold = np.asarray(
        compute_atomic_cooling_mass_msun(
            z_obs,
            cosmology=cosmology,
            virial_temperature_k=virial_temperature_k,
            mu=mu,
        ),
        dtype=float,
    )
    popiii_minimum = np.asarray(
        compute_popiii_lw_minimum_mass_msun(
            z_obs,
            lw_background_j21=lw_background_j21,
        ),
        dtype=float,
    )
    try:
        mass, popiii_minimum, threshold = np.broadcast_arrays(mass, popiii_minimum, threshold)
    except ValueError as exc:
        raise ValueError("z_obs and lw_background_j21 must be scalar or broadcast with halo_mass_msun") from exc

    channels = np.full(
        mass.shape,
        STELLAR_CHANNEL_BELOW_POPIII_MIN,
        dtype=f"<U{max(len(c) for c in STELLAR_CHANNELS)}",
    )
    channels[mass >= popiii_minimum] = STELLAR_CHANNEL_POPIII
    channels[mass >= threshold] = STELLAR_CHANNEL_POPII
    if np.ndim(halo_mass_msun) == 0 and np.ndim(z_obs) == 0 and np.ndim(lw_background_j21) == 0:
        return str(channels.item())
    return np.asarray(channels, dtype=f"<U{max(len(channel) for channel in STELLAR_CHANNELS)}")


__all__ = [
    "ATOMIC_COOLING_MU",
    "ATOMIC_COOLING_TEMPERATURE_K",
    "DEFAULT_LW_BACKGROUND_J21",
    "POPIII_LW_FEEDBACK_COEFFICIENT",
    "POPIII_LW_FEEDBACK_EXPONENT",
    "POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN",
    "POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT",
    "STELLAR_CHANNEL_BELOW_POPIII_MIN",
    "STELLAR_CHANNEL_POPII",
    "STELLAR_CHANNEL_POPIII",
    "STELLAR_CHANNELS",
    "classify_halo_stellar_channels",
    "compute_atomic_cooling_mass_msun",
    "compute_popiii_lw_minimum_mass_msun",
]
