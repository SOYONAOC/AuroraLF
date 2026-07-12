from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Stable, explicitly assigned identifiers.  These values are part of the v2
# reproducibility contract and must not be replaced with Python's salted hash.
_COMPONENT_HMF_MASS = 0x484D4601
_COMPONENT_MAH = 0x4D414801
_COMPONENT_METALLICITY = 0x4D455401
_COMPONENT_BURST = 0x42525301
_UINT64_MAX = 2**64 - 1


def _require_python_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_python_uint64(value: object, *, name: str) -> int:
    integer = _require_python_nonnegative_int(value, name=name)
    if integer > _UINT64_MAX:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer (uint64)")
    return integer


def _redshift_words(redshift: object) -> tuple[int, int]:
    if isinstance(redshift, (bool, np.bool_)):
        raise TypeError("redshift must be a real scalar, not bool")
    array = np.asarray(redshift)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        raise TypeError("redshift must be a real scalar")
    value = float(array)
    if not np.isfinite(value):
        raise ValueError("redshift must be finite")
    if value < 0.0:
        raise ValueError("redshift must be non-negative")
    if value == 0.0:
        value = 0.0
    bits = int(np.asarray(value, dtype=np.float64).view(np.uint64))
    return bits & 0xFFFFFFFF, bits >> 32


def _derive_component_seed(
    base_seed: int,
    *,
    component_id: int,
    redshift: object,
    mass_index: int,
) -> int:
    base = _require_python_uint64(base_seed, name="base_seed")
    index = _require_python_nonnegative_int(mass_index, name="mass_index")
    redshift_low, redshift_high = _redshift_words(redshift)
    sequence = np.random.SeedSequence(
        [base, component_id, redshift_low, redshift_high, index]
    )
    words = sequence.generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)


@dataclass(frozen=True)
class PipelineRandomSeeds:
    """Deterministic random seeds for the stochastic pipeline components."""

    mah: int
    metallicity: int
    burst: int

    def __post_init__(self) -> None:
        _require_python_uint64(self.mah, name="mah")
        _require_python_uint64(self.metallicity, name="metallicity")
        _require_python_uint64(self.burst, name="burst")

    def as_metadata(self) -> dict[str, int]:
        return {
            "mah": self.mah,
            "metallicity": self.metallicity,
            "burst": self.burst,
        }


def derive_pipeline_random_seeds(
    base_seed: int,
    *,
    redshift: float,
    mass_index: int,
) -> PipelineRandomSeeds:
    """Derive paired component seeds for one redshift and HMF mass sample."""

    return PipelineRandomSeeds(
        mah=_derive_component_seed(
            base_seed,
            component_id=_COMPONENT_MAH,
            redshift=redshift,
            mass_index=mass_index,
        ),
        metallicity=_derive_component_seed(
            base_seed,
            component_id=_COMPONENT_METALLICITY,
            redshift=redshift,
            mass_index=mass_index,
        ),
        burst=_derive_component_seed(
            base_seed,
            component_id=_COMPONENT_BURST,
            redshift=redshift,
            mass_index=mass_index,
        ),
    )


def derive_hmf_mass_seed(base_seed: int, redshift: float) -> int:
    """Derive the seed used only for drawing HMF integration masses."""

    return _derive_component_seed(
        base_seed,
        component_id=_COMPONENT_HMF_MASS,
        redshift=redshift,
        mass_index=0,
    )


__all__ = [
    "PipelineRandomSeeds",
    "derive_hmf_mass_seed",
    "derive_pipeline_random_seeds",
]
