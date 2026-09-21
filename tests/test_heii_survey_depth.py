import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from scripts.analysis.heii_survey_depth import depth_limits, summarize_arrays


def test_depth_matches_ab_flux_density():
    astro = FlatLambdaCDM(H0=67.66, Om0=0.30966)
    z = 12.5
    result = depth_limits(z, astro)
    # Independent AB definition: Fnu=(1+z)Lnu/(4*pi*DL^2).
    flux = (
        (1 + z)
        * result["uv_lnu_limit"]
        / (4 * np.pi * astro.luminosity_distance(z).to_value("cm") ** 2)
    )
    assert -2.5 * np.log10(flux) - 48.6 == pytest.approx(30.6, abs=0.002)
    assert result["muv_limit"] == pytest.approx(-17.252, abs=0.001)


def test_selection_preserves_zero_unknown_and_abundance_weights():
    # Includes an exactly-at-limit detection, a UV-faint bright line, a zero
    # emitter and an unknown early burst. Weights intentionally differ by mass.
    uv = np.array([[1.0, 1.0, 0.9], [1.0, 1.0, 1.0]])
    line = np.array([[2.0, 0.0, 100.0], [1.0, np.nan, 4.0]])
    status = np.array([[1, 0, 1], [1, 2, 1]])
    limits = {"uv_lnu_limit": 1.0, "line_luminosity_limit": 2.0}
    s = summarize_arrays(uv, line, status, np.array([1.0, 3.0]), limits)
    assert s["n_uv_selected"] == 5
    assert s["uv_selected_known"]["n"] == 4
    assert s["uv_selected_known"]["zero_weight_fraction"] == pytest.approx(1 / 8)
    assert s["unknown_weight_fraction"] == pytest.approx(3 / 11)
    assert s["uv_and_line_selected"]["n"] == 2
    assert s["line_detectable_fraction_known"] == pytest.approx(4 / 8)
    assert s["line_detectable_fraction_bounds"] == pytest.approx([4 / 11, 7 / 11])
    assert s["uv_and_line_selected"]["q16_q50_q84"][0] >= 2


def test_no_detections_have_no_invented_quantiles():
    s = summarize_arrays(
        np.ones((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.ones(2),
        {"uv_lnu_limit": 1.0, "line_luminosity_limit": 2.0},
    )
    assert s["uv_and_line_selected"]["q16_q50_q84"] is None
    assert s["line_detectable_fraction_known"] == 0
    assert s["uv_selected_known"]["q16_q50_q84"] == [0, 0, 0]


def test_unknown_history_cannot_be_silently_zeroed():
    with pytest.raises(ValueError, match="Unknown histories"):
        summarize_arrays(
            np.ones((1, 2)),
            np.zeros((1, 2)),
            np.array([[0, 2]]),
            np.ones(1),
            {"uv_lnu_limit": 1.0, "line_luminosity_limit": 2.0},
        )


def test_young_bright_selection_keeps_age_boundary_and_excludes_old_bursts():
    uv = np.array([[1.0, 1.0, 0.9, 1.0, 1.0]])
    result = summarize_arrays(
        uv,
        np.array([[4.0, 100.0, 100.0, np.nan, 0.0]]),
        np.array([[1, 1, 1, 2, 0]]),
        np.ones(1),
        {"uv_lnu_limit": 1.0, "line_luminosity_limit": 5.0},
        age_myr=np.array([[3.0, 3.01, 1.0, np.nan, np.nan]]),
        max_age_myr=3.0,
    )
    assert result["uv_selected_known"]["n"] == 1
    assert result["uv_selected_known"]["q16_q50_q84"] == [4.0, 4.0, 4.0]
    # The main magnitude/age-selected population is retained below the line limit.
    assert result["uv_and_line_selected"]["n"] == 0
