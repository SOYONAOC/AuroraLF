import numpy as np
import pytest

from scripts.analysis.extend_ionizing_rates_lowz import join_arrays, validate_onset_match


def table(z):
    z = np.asarray(z, dtype=float)
    rates = np.broadcast_to(z[:, None, None], (len(z), 2, 3)).copy()
    return dict(
        redshifts=z,
        mass_msun=np.array([1e8, 1e9]),
        mean_rate=rates,
        se_rate=rates / 10,
        quadrature_error=rates[..., 0] / 100,
        rate_covariance_of_mean=np.einsum("zmi,zmj->zmij", rates, rates) / 100,
    )


def test_join_preserves_archived_values_and_recomputes_overlap():
    old, low = table([6, 7, 10]), table([5, 5.5, 6])
    low["computed"] = np.ones((3, 2), dtype=bool)
    merged, overlap = join_arrays(old, low)
    assert overlap == [6]
    np.testing.assert_array_equal(merged["redshifts"], [5, 5.5, 6, 7, 10])
    for key in old.keys() - {"mass_msun"}:
        np.testing.assert_array_equal(merged[key][2:], old[key])


def test_changed_overlap_cannot_be_merged():
    old, low = table([6, 7]), table([5, 6])
    low["mean_rate"][-1, 0, 0] *= 1.01
    with pytest.raises(AssertionError, match="overlap"):
        join_arrays(old, low)


def test_mass_grid_allows_only_floating_representation_roundoff():
    old, low = table([6, 7]), table([5, 6])
    low["mass_msun"] = np.nextafter(low["mass_msun"], np.inf)
    merged, _ = join_arrays(old, low)
    np.testing.assert_array_equal(merged["mass_msun"], old["mass_msun"])
    low["mass_msun"] *= 1.000001
    with pytest.raises(AssertionError):
        join_arrays(old, low)


def test_missing_overlap_or_uncomputed_cells_fail():
    with pytest.raises(ValueError, match="overlap"):
        join_arrays(table([6, 7]), table([5, 5.5]))
    low = table([5, 6])
    low["computed"] = np.array([[True, False], [True, True]])
    with pytest.raises(ValueError, match="Uncomputed"):
        join_arrays(table([6, 7]), low)


@pytest.mark.parametrize(
    "new",
    [
        {"variant": "delay0", "delay_myr": 0.0},
        {"variant": "delay30", "delay_myr": 30.0},
    ],
)
def test_legacy_independent_sources_cannot_merge_with_causal_sources(new):
    with pytest.raises(ValueError, match="Changed Pop II onset"):
        validate_onset_match({}, new)
    validate_onset_match(new, new)


def test_legacy_baseline_identity_and_corrupt_delay():
    validate_onset_match({}, {"variant": "baseline", "delay_myr": None})
    with pytest.raises(ValueError, match="Invalid Pop II onset"):
        validate_onset_match({}, {"variant": "delay30", "delay_myr": 0.0})
