"""Paired component LF bins must follow summed luminosities and conserve counts."""

import importlib.util
from pathlib import Path

import numpy as np

from auroralf.uvlf import uv_luminosity_to_muv

path = Path(__file__).resolve().parents[1] / "scripts/analysis/build_current_uvlf.py"
spec = importlib.util.spec_from_file_location("current_uvlf", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_total_moves_to_brighter_bin_and_retains_one_object():
    p2 = np.array([[1e28, 0.0], [2e28, 1e27]])
    p3 = p2.copy()
    w = np.array([0.2, 0.05])
    edges = np.arange(-30.0, 0.1, 0.2)
    hist, counts = module.component_histograms(p2, p3, w, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for channel in range(3):
        assert counts[channel].sum() == 3
        np.testing.assert_allclose(np.sum(hist[channel] * np.diff(edges)), 0.3)
    assert np.sum(counts[2] * centers) < np.sum(counts[0] * centers)
    assert not np.allclose(hist[2], hist[0] + hist[1])
    np.testing.assert_allclose(
        uv_luminosity_to_muv(2e28) - uv_luminosity_to_muv(1e28), -2.5 * np.log10(2)
    )


def test_zero_popiii_returns_original_total():
    p2 = np.array([[1e28, 2e28]])
    hist, counts = module.component_histograms(
        p2, np.zeros_like(p2), np.array([0.1]), np.arange(-30.0, 0.0, 0.5)
    )
    np.testing.assert_array_equal(hist[0], hist[2])
    assert not hist[1].any() and not counts[1].any()
