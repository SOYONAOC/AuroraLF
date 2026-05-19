from __future__ import annotations

import numpy as np


SOLAR_OXYGEN_ABUNDANCE = 8.69


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
