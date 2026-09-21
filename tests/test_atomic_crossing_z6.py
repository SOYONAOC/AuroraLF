"""Crossing records must retain first events and observational censoring."""

import numpy as np

from scripts.analysis.atomic_crossing_z6 import crossing
from scripts.analysis.summarize_atomic_crossing_z6 import quantiles


def record(mass):
    return crossing(
        np.arange(3),
        np.array(mass),
        np.array([50, 100, 150]),
        np.array([12.0, 10.0, 8.0]),
        np.array([1.0, 2.0, 3.0]),
        np.ones(3),
    )


def test_first_upcrossing_survives_later_mass_loss():
    result = record([0.5, 2.0, 0.3])
    assert result["status"] == "bracketed"
    assert result["z_recorded"] == 10
    assert result["z_upper"] == 12
    assert result["z_interp"] == 11


def test_first_node_above_threshold_is_not_an_exact_crossing():
    result = record([2.0, 3.0, 4.0])
    assert result["status"] == "censored"
    assert result["z_interp"] is None
    assert result["z_upper"] is None


def test_no_crossing_remains_unknown():
    result = record([0.1, 0.2, 0.3])
    assert result["status"] == "never"
    assert result["z_recorded"] is None


def test_threshold_equality_is_a_crossing():
    result = record([0.1, 1.0, 2.0])
    assert result["z_interp"] == 10


def test_population_weights_and_infinite_censoring_bound():
    assert quantiles([7.0, 12.0], [100.0, 1.0])[1] == 7.0
    assert np.isinf(quantiles([7.0, np.inf], [1.0, 100.0])[1])
