from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "data" / "select_tng_mah_subhalos.py"


def _load_selection_module():
    spec = importlib.util.spec_from_file_location("select_tng_mah_subhalos", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["select_tng_mah_subhalos"] = module
    spec.loader.exec_module(module)
    return module


def _write_groupcat_field(path: Path, field: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Group")
        group.create_dataset(field, data=values)


def test_selection_uses_group_mcrit200_and_writes_id_files(tmp_path: Path) -> None:
    module = _load_selection_module()
    hubble = 0.6774
    first_sub = np.array([10, 11, -1, 13, 14, 15, 16], dtype=np.int64)
    logm = np.array([9.05, 9.10, 9.20, 9.35, 9.40, 9.62, 10.10], dtype=float)
    mass_code = 10.0**logm / 1.0e10 * hubble

    first_sub_path = tmp_path / "fof_subhalo_tab_013.Group.GroupFirstSub.hdf5"
    mcrit_path = tmp_path / "fof_subhalo_tab_013.Group.Group_M_Crit200.hdf5"
    _write_groupcat_field(first_sub_path, "GroupFirstSub", first_sub)
    _write_groupcat_field(mcrit_path, "Group_M_Crit200", mass_code)

    selection = module._select_snapshot_subhalos(
        target_z=6.0,
        snapshot=13,
        snapshot_z=6.0107573988449,
        group_first_sub_path=first_sub_path,
        group_mcrit200_path=mcrit_path,
        output_dir=tmp_path / "selection",
        simulation="TNG100-1-Dark",
        logm_edges=np.array([9.0, 9.25, 9.5, 9.75, 10.0], dtype=float),
        max_per_bin=2,
        min_per_bin=2,
        hubble=hubble,
        rng=np.random.default_rng(1),
    )

    np.testing.assert_array_equal(selection.selected_ids, np.array([10, 11, 13, 14, 15], dtype=np.int64))
    assert [row["available_count"] for row in selection.bin_rows] == [2, 2, 1, 0]
    assert [row["selected_count"] for row in selection.bin_rows] == [2, 2, 1, 0]
    assert [row["status"] for row in selection.bin_rows] == ["ok", "ok", "insufficient", "empty"]

    all_id_file = Path(selection.bin_rows[0]["all_id_file"])
    assert all_id_file.exists()
    assert all_id_file.read_text(encoding="utf-8").splitlines() == ["10", "11", "13", "14", "15"]
    assert Path(selection.bin_rows[-1]["id_file"]).exists()


def test_mass_bin_edges_require_exact_integer_number_of_bins() -> None:
    module = _load_selection_module()
    with pytest.raises(ValueError, match="integer multiple"):
        module._mass_bin_edges(9.0, 10.0, 0.3)


def test_selection_script_help_exposes_core_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--target-redshifts" in completed.stdout
    assert "--max-per-bin" in completed.stdout
    assert "--build-download-workers" in completed.stdout
    assert "--build-download-retries" in completed.stdout
    assert "--build-drop-invalid-mpb" in completed.stdout
    assert "--build-snapshot-grid" in completed.stdout
    assert "--build-missing-mass-ratio-floor" in completed.stdout
    assert "--groupcat-dir" in completed.stdout
    assert "--force-output" in completed.stdout
