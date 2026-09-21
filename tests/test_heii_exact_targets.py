"""Exact-redshift conditioning, population exclusions and efficiency selection."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.heii import HeIIKernel
from auroralf.uvlf.hmf_sampling import AB_ZEROPOINT_LNU
from scripts.analysis.heii_exact_targets import summarize_target


def inputs(status):
    status = np.asarray(status, dtype=np.int8)[:, None]
    luminosity = 10 ** ((AB_ZEROPOINT_LNU + 20) / 2.5)
    sample = dict(
        redshift=12.342,
        status=status,
        burst_halo_mass_msun=np.where(status == 1, 1e8, np.nan),
        age_myr=np.where(status == 1, 1.0, np.nan),
        popii=np.full(status.shape, luminosity),
        popiii_per_efficiency=np.zeros(status.shape),
        weight_per_track=np.ones(len(status)),
    )
    target = dict(z=12.342, population_z=12.5, muv=-20, magnification=1.3, flux=1e-19)
    values = np.ones(2)
    kernel = HeIIKernel(np.array([0.01, 1000]), values * 1e34, values, values, values, 0)
    return sample, target, kernel, FlatLambdaCDM(H0=70, Om0=0.3)


def test_exact_redshift_required():
    sample, target, kernel, astro = inputs([1])
    sample["redshift"] = 12.5
    with pytest.raises(ValueError, match="redshift"):
        summarize_target(sample, target, kernel, [0.03], astro)


def test_known_zeros_and_unknown_bursts_kept_distinct():
    sample, target, kernel, astro = inputs([0, 1, 2])
    result = summarize_target(sample, target, kernel, [0.03], astro)["0.03"]
    assert result["target"]["population_z"] == target["z"]
    assert target["population_z"] == 12.5  # source catalog stays unchanged
    for stats in result["target_windows"][0]["methods"].values():
        assert stats["n_selected"] == 3
        assert stats["n_known"] == 2
        assert stats["unknown_weight_fraction"] == pytest.approx(1 / 3)
        assert stats["zero_fraction_known"] == 0.5
        assert stats["effective_mass_clusters"] == 2


def test_efficiency_reselects_uv_population():
    sample, target, kernel, astro = inputs([1, 1])
    luminosity = sample["popii"][0, 0]
    sample["popii"] *= 0.9
    sample["popiii_per_efficiency"] = luminosity * np.array([[1.0], [10.0]])
    result = summarize_target(sample, target, kernel, [0.01, 0.1], astro)
    small = result["0.01"]["target_windows"][0]["methods"]["linear_log_age"]
    large = result["0.1"]["target_windows"][0]["methods"]["linear_log_age"]
    assert small["n_selected"] == 2
    assert large["n_selected"] == 1
    assert large["flux_q16_q50_q84"][1] == pytest.approx(10 * small["flux_q16_q50_q84"][1])
