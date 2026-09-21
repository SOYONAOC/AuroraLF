"""Published correction states and weighted selection, including zero signals."""

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from scripts.analysis.heii_v24_comparison import calzetti_a1500, distribution, observations


def test_delensing_once_and_lap1_actual_limit():
    catalog = observations()
    astro = FlatLambdaCDM(**catalog["cosmology"])
    rows = {r["id"]: r for r in catalog["sources"]}
    rx = rows["RXJ2129_z8HeII_A"]
    expected = 4 * np.pi * astro.luminosity_distance(rx["z"]).to_value("cm") ** 2 * 1.2e-18
    np.testing.assert_allclose(rx["luminosity"], expected)
    paper_astro = FlatLambdaCDM(**rx["source_cosmology"])
    paper_lum = (
        4 * np.pi * paper_astro.luminosity_distance(rx["z"]).to_value("cm") ** 2 * rx["flux"]
    )
    assert abs(paper_lum / rx["published_luminosity"] - 1) < 0.01
    assert rx["muv_common_cosmology"] < rx["muv"]
    lap = rows["LAP1_arclet"]
    expected = 4 * np.pi * astro.luminosity_distance(lap["z"]).to_value("cm") ** 2 * 2.07e-19 / 120
    np.testing.assert_allclose(lap["luminosity"], expected)
    assert lap["measurement"] == "upper_limit" and lap["limit_sigma"] == 1
    assert lap["luminosity_error"] is None
    assert sum(r["primary"] for r in rows.values()) == 3
    assert all(r["muv"] is None for r in rows.values() if r["system"] == "GN-z11 halo")


def test_weighted_population_includes_known_zero_excludes_unknown():
    values = np.array([[0, 10], [20, np.nan]], dtype=float)
    result = distribution(values, np.array([0.1, 0.2]), np.isfinite(values), reference=10)
    np.testing.assert_allclose(result["density_mpc3"], 0.4)
    np.testing.assert_allclose(result["zero_fraction"], 0.25)
    np.testing.assert_allclose(result["fraction_at_or_above_reference"], 0.75)
    np.testing.assert_allclose(result["effective_mass_clusters"], 2)
    assert result["n_selected"] == 3


def test_uv_matching_corrects_in_right_direction():
    attenuation = calzetti_a1500(0.12)
    assert 0.30 < attenuation < 0.31
    assert -19.58 - attenuation < -19.58
