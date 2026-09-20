import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "escape_rescaling",
    Path(__file__).resolve().parents[1] / "scripts/analysis/rescale_popiii_escape.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_covariance_matches_rescaled_individual_samples():
    # Correlated channels distinguish full covariance propagation from diagonal-only scaling.
    samples = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0], [2.0, 6.0, 5.0], [8.0, 9.0, 4.0]])
    arrays = {
        "redshifts": np.array([6.0]),
        "mass_msun": np.array([1e8]),
        "mean_rate": samples.mean(axis=0)[None, None],
        "se_rate": (samples.std(axis=0, ddof=1) / 2)[None, None],
        "rate_covariance_of_mean": (np.cov(samples, rowvar=False) / 4)[None, None],
        "quadrature_error": np.array([[0.4]]),
    }
    for new_fesc in [0, 0.01, 0.2, 1]:
        scaled = MODULE.scale_arrays(arrays, 0.2, new_fesc)
        raw = samples * [1, new_fesc / 0.2, new_fesc / 0.2]
        np.testing.assert_allclose(scaled["mean_rate"][0, 0], raw.mean(axis=0))
        np.testing.assert_allclose(scaled["se_rate"][0, 0], raw.std(axis=0, ddof=1) / 2)
        np.testing.assert_allclose(
            scaled["rate_covariance_of_mean"][0, 0], np.cov(raw, rowvar=False) / 4
        )
        np.testing.assert_array_equal(scaled["mean_rate"][..., 0], arrays["mean_rate"][..., 0])
    np.testing.assert_array_equal(arrays["mean_rate"][0, 0], samples.mean(axis=0))


@pytest.mark.parametrize("old,new", [(0, 0.1), (-1, 0.1), (0.2, -0.1), (0.2, 1.1), (0.2, np.nan)])
def test_unphysical_escape_rejected(old, new):
    with pytest.raises(ValueError):
        MODULE.scale_arrays({}, old, new)


@pytest.mark.parametrize("new_fesc", [0, 0.025, 0.05, 0.1, 0.2, 1])
def test_common_escape_preserves_population_mix_and_covariance(new_fesc):
    samples = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0], [2.0, 6.0, 5.0], [8.0, 9.0, 4.0]])
    arrays = {
        "redshifts": np.array([6.0]),
        "mass_msun": np.array([1e8]),
        "mean_rate": samples.mean(axis=0)[None, None],
        "se_rate": (samples.std(axis=0, ddof=1) / 2)[None, None],
        "rate_covariance_of_mean": (np.cov(samples, rowvar=False) / 4)[None, None],
        "quadrature_error": np.array([[0.4]]),
    }
    scaled = MODULE.scale_common_arrays(arrays, 0.2, new_fesc)
    raw = samples * (new_fesc / 0.2)
    np.testing.assert_allclose(scaled["mean_rate"][0, 0], raw.mean(axis=0))
    np.testing.assert_allclose(scaled["se_rate"][0, 0], raw.std(axis=0, ddof=1) / 2)
    np.testing.assert_allclose(
        scaled["rate_covariance_of_mean"][0, 0], np.cov(raw, rowvar=False) / 4
    )
    np.testing.assert_allclose(
        scaled["quadrature_error"], arrays["quadrature_error"] * new_fesc / 0.2
    )
    if new_fesc:
        np.testing.assert_allclose(
            scaled["mean_rate"][..., 1] / scaled["mean_rate"][..., 0],
            arrays["mean_rate"][..., 1] / arrays["mean_rate"][..., 0],
        )


@pytest.mark.parametrize("new_ii", [0.0, 0.05, 1.0])
def test_independent_escape_covariance_from_samples(new_ii):
    samples = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0], [2.0, 6.0, 5.0], [8.0, 9.0, 4.0]])
    arrays = {
        "redshifts": np.array([6.0]),
        "mass_msun": np.array([1e8]),
        "mean_rate": samples.mean(axis=0)[None, None],
        "se_rate": (samples.std(axis=0, ddof=1) / 2)[None, None],
        "rate_covariance_of_mean": (np.cov(samples, rowvar=False) / 4)[None, None],
        "quadrature_error": np.array([[0.4]]),
    }
    out = MODULE.scale_independent_arrays(arrays, 0.2, new_ii, 0.2, 0.5)
    raw = samples * [new_ii / 0.2, 2.5, 2.5]
    np.testing.assert_allclose(out["mean_rate"][0, 0], raw.mean(axis=0))
    np.testing.assert_allclose(out["se_rate"][0, 0], raw.std(axis=0, ddof=1) / 2)
    np.testing.assert_allclose(out["rate_covariance_of_mean"][0, 0], np.cov(raw, rowvar=False) / 4)
    np.testing.assert_allclose(out["quadrature_error"], [[1.0]])


@pytest.mark.parametrize("old,new", [(0, 0.1), (0.2, -0.1), (0.2, 1.1), (0.2, np.nan)])
def test_independent_popii_invalid_rejected(old, new):
    with pytest.raises(ValueError):
        MODULE.scale_independent_arrays({}, old, new, 0.2, 0.5)
