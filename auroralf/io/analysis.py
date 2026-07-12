from __future__ import annotations

from pathlib import Path

import numpy as np

from auroralf.results import IMFModeResult, UVLFRunResult
from .hdf5 import read_uvlf_artifact
from .schema import canonical_config_mapping


BURST_CONFIG_DIFFERENCES = frozenset(
    {
        "star_formation.burst_scatter_dex",
        "star_formation.burst_scatter_correlation_timescale_myr",
        "star_formation.burst_scatter_mass_conserving",
    }
)
GATE_DELAY_BURST_CONFIG_DIFFERENCES = BURST_CONFIG_DIFFERENCES | {
    "star_formation.enable_time_delay"
}
HMF_CONFIG_DIFFERENCES = frozenset({"sampling.mass_function_model"})
_RUNTIME_CONFIG_DIFFERENCES = frozenset(
    {
        "run_id",
        "output.artifact_path",
        "sampling.workers",
        "sampling.mass_batch_size",
    }
)


def load_uvlf_result(path: str | Path) -> UVLFRunResult:
    return read_uvlf_artifact(path, load_samples=False).result


def select_mode_result(
    result: UVLFRunResult,
    *,
    redshift: float,
    mode: str,
) -> IMFModeResult:
    if type(result) is not UVLFRunResult:
        raise TypeError("result must be exactly UVLFRunResult")
    try:
        redshift_result = result.for_redshift(redshift)
    except KeyError as error:
        raise ValueError(f"requested redshift is absent: {redshift:g}") from error
    try:
        return redshift_result.for_mode(mode)
    except KeyError as error:
        raise ValueError(
            f"requested mode is absent at redshift {redshift:g}: {mode}"
        ) from error


def _flatten_mapping(value: object, *, prefix: str = "") -> dict[str, object]:
    if type(value) is not dict:
        return {prefix: value}
    flattened: dict[str, object] = {}
    for key, child in value.items():
        child_prefix = str(key) if not prefix else f"{prefix}.{key}"
        flattened.update(_flatten_mapping(child, prefix=child_prefix))
    return flattened


def require_compatible_results(
    reference: UVLFRunResult,
    candidate: UVLFRunResult,
    *,
    allowed_config_differences: frozenset[str],
    context: str,
) -> None:
    if type(reference) is not UVLFRunResult or type(candidate) is not UVLFRunResult:
        raise TypeError("reference and candidate must be exactly UVLFRunResult")
    if type(allowed_config_differences) is not frozenset or not all(
        type(item) is str for item in allowed_config_differences
    ):
        raise TypeError("allowed_config_differences must be a frozenset of strings")
    if type(context) is not str or not context:
        raise TypeError("context must be a non-empty string")

    if reference.config.redshifts != candidate.config.redshifts:
        raise ValueError(f"{context}: redshift axes differ")
    reference_modes = reference.config.stellar_population.imf_modes
    candidate_modes = candidate.config.stellar_population.imf_modes
    if reference_modes != candidate_modes:
        raise ValueError(f"{context}: mode axes differ")
    reference_edges = np.asarray(reference.config.sampling.muv_bin_edges, dtype=float)
    candidate_edges = np.asarray(candidate.config.sampling.muv_bin_edges, dtype=float)
    if not np.array_equal(reference_edges, candidate_edges):
        raise ValueError(f"{context}: bin axes differ")

    reference_mapping = _flatten_mapping(canonical_config_mapping(reference.config))
    candidate_mapping = _flatten_mapping(canonical_config_mapping(candidate.config))
    if set(reference_mapping) != set(candidate_mapping):
        raise ValueError(f"{context}: config field sets differ")
    ignored = _RUNTIME_CONFIG_DIFFERENCES | allowed_config_differences
    for field_name in sorted(reference_mapping):
        if field_name in ignored:
            continue
        if reference_mapping[field_name] != candidate_mapping[field_name]:
            raise ValueError(
                f"{context}: disallowed config difference: {field_name}"
            )


__all__ = [
    "BURST_CONFIG_DIFFERENCES",
    "GATE_DELAY_BURST_CONFIG_DIFFERENCES",
    "HMF_CONFIG_DIFFERENCES",
    "load_uvlf_result",
    "require_compatible_results",
    "select_mode_result",
]
