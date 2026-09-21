import numpy as np
import pytest

from scripts.analysis.tng_pristine_fraction import allowed, ancestry, weighted_summary


def test_secondary_progenitor_included_but_unrelated_tree_excluded():
    tree = dict(
        SubhaloID=np.array([100, 101, 102, 103, 999]),
        FirstProgenitorID=np.array([101, 103, -1, -1, -1]),
        NextProgenitorID=np.array([-1, 102, -1, -1, -1]),
        DescendantID=np.array([-1, 100, 100, 101, -1]),
        SnapNum=np.array([5, 4, 4, 2, 1]),
    )
    all_indices, main_indices = ancestry(tree, 0)
    np.testing.assert_array_equal(all_indices, [1, 2, 3])
    np.testing.assert_array_equal(main_indices, [1, 3])
    tree["FirstProgenitorID"][2] = 555
    with pytest.raises(ValueError, match="Missing progenitor"):
        ancestry(tree, 0)


def test_pollution_delay_boundary_and_unresolved_history():
    np.testing.assert_array_equal(
        allowed([-1, 0, 29, 30, 31], 30), [True, True, True, False, False]
    )


def test_restore_catalog_abundance_instead_of_raw_sample_fraction():
    rows = [
        dict(
            weight=100.0,
            population=200,
            selected=2,
            logmass_low=9.0,
            target_cooling=True,
            age=100.0,
        ),
        dict(
            weight=100.0,
            population=200,
            selected=2,
            logmass_low=9.0,
            target_cooling=True,
            age=100.0,
        ),
        dict(weight=1.0, population=1, selected=1, logmass_low=10.0, target_cooling=True, age=-1.0),
    ]
    result = weighted_summary(rows, "age", 30)
    assert result["fraction"] == pytest.approx(1 / 201)
    assert result["sampled_unvetoed"] == 1
