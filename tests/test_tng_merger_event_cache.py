from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "data" / "build_tng_merger_event_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_tng_merger_event_cache", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_tng_merger_event_cache"] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_full_tree(path: Path) -> None:
    n_rows = 4
    with h5py.File(path, "w") as handle:
        handle.create_dataset("SubhaloID", data=np.array([100, 101, 102, 103], dtype=np.int64))
        handle.create_dataset("DescendantID", data=np.array([-1, 100, 100, 102], dtype=np.int64))
        handle.create_dataset("FirstProgenitorID", data=np.array([101, -1, 103, -1], dtype=np.int64))
        handle.create_dataset("NextProgenitorID", data=np.array([-1, 102, -1, -1], dtype=np.int64))
        handle.create_dataset("SnapNum", data=np.array([2, 1, 1, 0], dtype=np.int16))
        handle.create_dataset("SubfindID", data=np.array([10, 11, 12, 13], dtype=np.int32))
        handle.create_dataset("SubhaloMass", data=np.array([10.0, 6.0, 2.0, 3.0], dtype=np.float32))
        handle.create_dataset("Mass", data=np.array([10.0, 6.0, 2.0, 3.0], dtype=np.float32))
        handle.create_dataset("Group_M_Crit200", data=np.array([11.0, 7.0, 2.5, 3.5], dtype=np.float32))


def test_extract_tree_events_computes_direct_and_peak_mass_ratios(tmp_path: Path) -> None:
    module = _load_module()
    tree_path = tmp_path / "full.hdf5"
    _write_synthetic_full_tree(tree_path)

    events, summary = module._extract_tree_events(
        tree_path,
        final_snapshot=2,
        final_subhalo_id=10,
        hubble=1.0,
    )

    assert summary["event_count"] == 1
    assert summary["main_branch_length"] == 2
    assert len(events) == 1
    event = events[0]
    assert event["descendant_snap"] == 2
    assert event["primary_snap"] == 1
    assert event["secondary_snap"] == 1
    assert event["primary_subfind_id"] == 11
    assert event["secondary_subfind_id"] == 12
    assert event["primary_subhalo_mass_msun"] == pytest.approx(6.0e10)
    assert event["secondary_subhalo_mass_msun"] == pytest.approx(2.0e10)
    assert event["secondary_peak_subhalo_mass_msun"] == pytest.approx(3.0e10)
    assert event["mass_ratio_direct_ordered"] == pytest.approx(2.0 / 6.0)
    assert event["mass_ratio_peak_ordered"] == pytest.approx(3.0 / 6.0)


def test_extract_tree_events_fails_when_final_node_is_missing(tmp_path: Path) -> None:
    module = _load_module()
    tree_path = tmp_path / "full.hdf5"
    _write_synthetic_full_tree(tree_path)

    with pytest.raises(ValueError, match="exactly one final node"):
        module._extract_tree_events(
            tree_path,
            final_snapshot=2,
            final_subhalo_id=999,
            hubble=1.0,
        )
