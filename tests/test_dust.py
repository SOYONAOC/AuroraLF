from __future__ import annotations

import numpy as np
import pytest

from auroralf.uvlf.dust import (
    intrinsic_muv_from_observed,
    intrinsic_muv_jacobian,
    uv_continuum_slope_beta,
)


def _centered_mapping_derivative(
    muv_obs: float,
    z: float,
    *,
    c0: float,
    c1: float = 4.85,
    step: float = 1.0e-6,
) -> float:
    upper = intrinsic_muv_from_observed(muv_obs + step, z, c0=c0, c1=c1)
    lower = intrinsic_muv_from_observed(muv_obs - step, z, c0=c0, c1=c1)
    return (upper - lower) / (2.0 * step)


def test_intrinsic_muv_jacobian_matches_mapping_finite_difference() -> None:
    analytic = intrinsic_muv_jacobian(-22.0, 6.0, c0=4.7, c1=10.0)
    numerical = _centered_mapping_derivative(-22.0, 6.0, c0=4.7, c1=10.0)

    assert analytic == pytest.approx(numerical, rel=1.0e-8)


def test_intrinsic_muv_jacobian_is_one_on_attenuation_floor() -> None:
    c0 = 2.1
    muv_obs = -15.0
    z = 6.0
    beta = uv_continuum_slope_beta(muv_obs, z)
    c1_at_boundary = -c0 * beta

    assert intrinsic_muv_jacobian(muv_obs, z, c0=c0) == 1.0
    assert intrinsic_muv_jacobian(muv_obs, z, c0=c0) == pytest.approx(
        _centered_mapping_derivative(muv_obs, z, c0=c0)
    )
    assert intrinsic_muv_jacobian(muv_obs, z, c0=c0, c1=c1_at_boundary) == 1.0


def test_intrinsic_muv_jacobian_uses_custom_c0() -> None:
    z = 6.0
    c0 = 4.7
    expected = 1.0 - c0 * (-0.007 * z - 0.09)

    assert intrinsic_muv_jacobian(-22.0, z, c0=c0, c1=10.0) == pytest.approx(expected)


def test_intrinsic_muv_jacobian_preserves_scalar_and_array_behavior() -> None:
    scalar = intrinsic_muv_jacobian(-22.0, 6.0)
    muv_obs = np.array([[-22.0, -15.0], [-21.0, -14.0]])
    array = intrinsic_muv_jacobian(muv_obs, 6.0)
    numerical = np.empty_like(muv_obs)
    for index in np.ndindex(muv_obs.shape):
        numerical[index] = _centered_mapping_derivative(muv_obs[index], 6.0, c0=2.1)

    assert isinstance(scalar, float)
    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 2)
    np.testing.assert_allclose(array, numerical, rtol=1.0e-8)
    assert array[0, 0] > 1.0
    assert array[0, 1] == 1.0


@pytest.mark.parametrize("muv_obs", [np.inf, -np.inf, np.nan])
def test_intrinsic_muv_jacobian_rejects_nonfinite_magnitude(muv_obs: float) -> None:
    assert np.isnan(intrinsic_muv_jacobian(muv_obs, 6.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"z": np.nan},
        {"z": 6.0, "c0": np.inf},
        {"z": 6.0, "c1": np.inf},
        {"z": 6.0, "m0": np.inf},
    ],
)
def test_intrinsic_muv_jacobian_rejects_nonfinite_parameters(kwargs: dict[str, float]) -> None:
    assert np.isnan(intrinsic_muv_jacobian(-22.0, **kwargs))
