from __future__ import annotations

import numpy as np
import pytest

from auroralf.uvlf.hmf_sampling import (
    MASS_FUNCTION_MODEL_HMF_REED07,
    compute_halo_mass_function_dndm,
    compute_reed07_halo_mass_function_dndm,
    validate_mass_function_model,
)


def test_validate_mass_function_model_accepts_reed07() -> None:
    assert validate_mass_function_model("HMF_REED07") == MASS_FUNCTION_MODEL_HMF_REED07


def test_validate_mass_function_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="mass_function_model"):
        validate_mass_function_model("press_schechter")


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_validate_mass_function_model_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        validate_mass_function_model(deprecated_model)


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_compute_mass_function_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        compute_halo_mass_function_dndm(1.0e10, 12.5, mass_function_model=deprecated_model)


def test_reed07_mass_function_returns_positive_dndm() -> None:
    halo_mass = np.array([1.0e9, 1.0e10, 1.0e11])

    reed07 = np.asarray(
        compute_halo_mass_function_dndm(
            halo_mass,
            12.5,
            mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
        ),
        dtype=float,
    )
    direct_reed07 = np.asarray(compute_reed07_halo_mass_function_dndm(halo_mass, 12.5), dtype=float)

    assert np.all(reed07 > 0.0)
    np.testing.assert_allclose(reed07, direct_reed07, rtol=0.0, atol=0.0)


def test_hmf_reed07_scalar_input_returns_float() -> None:
    value = compute_halo_mass_function_dndm(
        1.0e10,
        6.0,
        mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
    )

    assert isinstance(value, float)
    assert value > 0.0
