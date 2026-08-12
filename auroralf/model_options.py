"""Dependency-light model option definitions shared across AuroraLF layers.

This module contains names and validation that are needed by configuration,
results, persistence, and scientific implementations.  It intentionally does
not import those layers, so they can depend on one canonical definition without
forming configuration/UVLF/I/O import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


IMF_MODE_CANONICAL = "canonical"
IMF_MODE_Z_GATED_MILD_TOPHEAVY = "z10_mild_topheavy"
IMF_MODE_MAH_BURST_MILD_TOPHEAVY = "mah_burst_mild_topheavy"
IMF_MODES = (
    IMF_MODE_CANONICAL,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
)

MASS_FUNCTION_MODEL_HMF_REED07 = "hmf_reed07"
MASS_FUNCTION_MODELS = (MASS_FUNCTION_MODEL_HMF_REED07,)
DEFAULT_MASS_FUNCTION_MODEL = MASS_FUNCTION_MODEL_HMF_REED07
DEPRECATED_MASS_FUNCTION_MODELS = frozenset({"massfunc_st", "hmf_watson13_fof"})

DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN = 0.05


@dataclass(frozen=True)
class IMFTransitionParameters:
    """Archived parameters for selecting a mild top-heavy Pop II SSP.

    These thresholds are historical AuroraLF settings rather than parameters
    from the BPASS SSP references.  Keeping the immutable value object here
    lets configuration and result validation share its schema without
    importing the archived gate implementation.
    """

    z_topheavy_min: float = 10.0
    source_redshift_gate_enabled: bool = False
    growth_time_threshold_myr: float = 50.0
    metallicity_topheavy_max_zsun: float | None = DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN


DEFAULT_IMF_TRANSITION_PARAMETERS = IMFTransitionParameters()


def validate_imf_mode(imf_mode: str) -> str:
    mode = str(imf_mode)
    if mode not in IMF_MODES:
        raise ValueError(f"imf_mode must be one of {IMF_MODES}, got {mode!r}")
    return mode


def validate_mass_function_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized in DEPRECATED_MASS_FUNCTION_MODELS:
        raise ValueError(
            f"{normalized} is no longer supported for AuroraLF production runs; "
            f"use {MASS_FUNCTION_MODEL_HMF_REED07}."
        )
    if normalized not in MASS_FUNCTION_MODELS:
        choices = ", ".join(MASS_FUNCTION_MODELS)
        raise ValueError(f"mass_function_model must be one of: {choices}")
    return normalized


__all__ = [
    "DEFAULT_IMF_TRANSITION_PARAMETERS",
    "DEFAULT_MASS_FUNCTION_MODEL",
    "DEFAULT_TOPHEAVY_METALLICITY_MAX_ZSUN",
    "DEPRECATED_MASS_FUNCTION_MODELS",
    "IMF_MODE_CANONICAL",
    "IMF_MODE_MAH_BURST_MILD_TOPHEAVY",
    "IMF_MODE_Z_GATED_MILD_TOPHEAVY",
    "IMF_MODES",
    "IMFTransitionParameters",
    "MASS_FUNCTION_MODEL_HMF_REED07",
    "MASS_FUNCTION_MODELS",
    "validate_imf_mode",
    "validate_mass_function_model",
]
