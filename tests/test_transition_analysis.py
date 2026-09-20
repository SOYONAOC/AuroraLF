"""Scientific limiting cases for paired UVLF sampling uncertainties."""

import numpy as np
import pytest

from scripts.analysis.analyze_popii_transition import ratio_and_se


def test_identical_paired_histories_cancel_sampling_error():
    samples = np.array([[0.0, 2.0], [1.0, 9.0], [30.0, 0.1], [4.0, 7.0]])
    ratio, error = ratio_and_se(0.5 * samples, samples)
    np.testing.assert_allclose(ratio, 0.5)
    np.testing.assert_allclose(error, 0, atol=1e-15)


def test_occupancy_fraction_has_binomial_sampling_error():
    occupied = np.r_[np.ones(30), np.zeros(70)]
    ratio, error = ratio_and_se(occupied, np.ones(100))
    np.testing.assert_allclose(ratio, 0.3)
    np.testing.assert_allclose(error, np.sqrt(0.3 * 0.7 / 99))
    scaled = ratio_and_se(occupied * 1e-23, np.ones(100) * 1e-23)
    np.testing.assert_allclose(scaled, (ratio, error))


def test_unsampled_bin_is_undefined_and_nonfinite_data_fail():
    ratio, error = ratio_and_se(np.zeros(10), np.zeros(10))
    assert np.isnan(ratio) and np.isnan(error)
    with pytest.raises(ValueError, match="Invalid paired"):
        ratio_and_se([1, np.nan], [1, 1])
