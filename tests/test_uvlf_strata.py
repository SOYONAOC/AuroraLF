import numpy as np
from scipy.special import ndtr

from auroralf.experiments.uvlf_components import age_strata, conditional_histograms


def test_age_weights_match_analytic_normal_first_passage_and_keep_dark_mass():
    t = np.linspace(0.1, 0.3, 201)[None, :]
    mass = 10 ** np.linspace(6.0, 8.0, 201)[None, :]
    cool = np.full_like(mass, 1e7)
    light, prob = age_strata(
        t,
        mass,
        cool,
        0.5,
        1.5,
        np.full((1, 4), 0.5),
        np.array([0.01, 100.0]),
        np.array([1e20, 1e20]),
        0.16,
        0.03,
    )
    # log10(M/Mcool)=10t-2; the last 100 Myr correspond to logq in [0,1].
    expected = ndtr((1 - 0.5) / 1.5) - ndtr((0 - 0.5) / 1.5)
    np.testing.assert_allclose(prob[0, :4].sum(), expected, rtol=1e-13)
    assert np.all(light[0, :4] > 0) and light[0, 4] == 0
    np.testing.assert_allclose(prob.sum(), 1)


def test_weighted_component_counts_keep_one_halo_and_shift_total():
    p2 = np.array([[1e28]])
    p3 = np.array([[[1e28, 0.0]]])
    prob = np.array([[[0.2, 0.8]]])
    edges = np.arange(-30.0, 0.1, 0.25)
    hist = conditional_histograms(p2, p3, prob, np.array([1.0]), edges)
    np.testing.assert_allclose((hist * np.diff(edges)).sum(axis=(1, 2)), [1.0, 0.2, 1.0])
    assert not np.allclose(hist[2], hist[0] + hist[1])


def test_old_record_cannot_retrigger_after_later_growth_below_that_record():
    t = np.linspace(0.1, 0.3, 201)[None, :]
    ratio = np.where(t <= 0.18, 10 * (t - 0.1), 0.8 - 2 * (t - 0.18))
    cool = np.full_like(t, 1e7)
    light, prob = age_strata(
        t,
        cool * 10**ratio,
        cool,
        0.5,
        1.5,
        np.full((1, 4), 0.5),
        np.array([0.01, 100.0]),
        np.array([1e20, 1e20]),
        0.16,
        0.03,
    )
    np.testing.assert_array_equal(light, 0)
    np.testing.assert_array_equal(prob, [[0, 0, 0, 0, 1]])
