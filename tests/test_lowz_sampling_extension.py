"""A further extension must preserve every previously computed sample exactly."""

import numpy as np
import pytest

from scripts.run.increase_lowz_uvlf_sampling import new_mass_tasks


def test_mass_only_extension_skips_all_old_histories():
    masses = np.arange(12) + 1.0
    tasks = new_mass_tasks(masses, (3, 128), 128)
    assert [i for i, _, _ in tasks] == list(range(3, 12))
    assert all(start == 0 for _, _, start in tasks)
    assert sum(128 - start for _, _, start in tasks) == 12 * 128 - 3 * 128


def test_extension_of_both_axes_has_no_gaps_or_duplicates():
    tasks = new_mass_tasks(np.arange(6) + 1.0, (3, 64), 128)
    retained = {(i, j) for i in range(3) for j in range(64)}
    added = [(i, j) for i, _, start in tasks for j in range(start, 128)]
    assert len(added) == len(set(added))
    assert not retained.intersection(added)
    assert retained.union(added) == {(i, j) for i in range(6) for j in range(128)}


@pytest.mark.parametrize("nm,nt", [(2, 128), (3, 64), (3, 128), (6, 127)])
def test_rejects_truncation_noop_or_partial_chunks(nm, nt):
    with pytest.raises(ValueError):
        new_mass_tasks(np.ones(nm), (3, 128), nt)
