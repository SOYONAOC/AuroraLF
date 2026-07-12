from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "data" / "prepare_thesan_dark1_download.py"


def _write_integrity_manifest(root: Path, paths: list[Path], destination: Path) -> Path:
    payload = {
        "schema_version": "auroralf.thesan_download_integrity.v1",
        "simulation": "Thesan-Dark-1",
        "files": {
            str(path.relative_to(root)): {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        },
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_thesan_dark1_download", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_thesan_dark1_download"] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_manifest_uses_nearest_snapshot_and_expected_products() -> None:
    module = _load_module()
    snapshot_map = {
        31: 12.32,
        32: 11.96,
        33: 11.73,
    }

    rows = module._build_stage_rows(
        stage="smoke",
        snapshot_redshift=snapshot_map,
        source_root="/Thesan-Dark-1",
        local_root=Path("external_data/thesan/thesan-dark-1"),
    )

    local_paths = {row["local_path"] for row in rows}
    assert "postprocessing/offsets/offsets_032.hdf5" in local_paths
    assert "output/groups_032/fof_subhalo_tab_032.0.hdf5" in local_paths
    assert "postprocessing/trees/LHaloTree/sub_desc_sf1_080" in local_paths
    assert {row["product"] for row in rows} == {"offset", "groupcat_chunk", "sub_desc"}
    assert all(row["source_path"].startswith("/Thesan-Dark-1/") for row in rows)
    assert rows[0]["stage"] == "smoke"


def test_sample_manifest_records_all_target_redshifts_and_selection_policy() -> None:
    module = _load_module()
    snapshot_map = {
        10: 12.05,
        24: 10.02,
        41: 8.01,
        62: 6.03,
    }

    rows = module._build_stage_rows(
        stage="sample",
        snapshot_redshift=snapshot_map,
        source_root="/Thesan-Dark-1",
        local_root=Path("external_data/thesan/thesan-dark-1"),
    )

    offsets = sorted(row["local_path"] for row in rows if row["product"] == "offset")
    groupcats = sorted(row["local_path"] for row in rows if row["product"] == "groupcat_all_chunks")
    tree_chunks = sorted(row["local_path"] for row in rows if row["product"] == "tree_chunk_selection")

    assert offsets == [
        "postprocessing/offsets/offsets_010.hdf5",
        "postprocessing/offsets/offsets_024.hdf5",
        "postprocessing/offsets/offsets_041.hdf5",
        "postprocessing/offsets/offsets_062.hdf5",
    ]
    assert groupcats == [
        "output/groups_010/fof_subhalo_tab_010.*.hdf5",
        "output/groups_024/fof_subhalo_tab_024.*.hdf5",
        "output/groups_041/fof_subhalo_tab_041.*.hdf5",
        "output/groups_062/fof_subhalo_tab_062.*.hdf5",
    ]
    assert tree_chunks == ["postprocessing/trees/LHaloTree/trees_sf1_190.C.hdf5"]
    assert all("logM=8.5-11.5" in row["notes"] for row in rows)
    assert all("max_per_bin=50" in row["notes"] for row in rows)


def test_all_manifest_combines_three_stages_without_snapshot_table() -> None:
    module = _load_module()

    rows = module._build_stage_rows(
        stage="all",
        snapshot_redshift=None,
        source_root="/Thesan-Dark-1",
        local_root=Path("external_data/thesan/thesan-dark-1"),
    )

    stages = {row["stage"] for row in rows}
    assert stages == {"smoke", "sample", "publication"}
    assert any(row["local_path"] == "postprocessing/offsets/offsets_NNN.hdf5" for row in rows)
    assert any(row["local_path"] == "postprocessing/offsets/offsets_000.hdf5" for row in rows)
    assert any(row["local_path"] == "postprocessing/offsets/offsets_079.hdf5" for row in rows)


def test_validate_smoke_files_checks_required_hdf5_schema(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "thesan-dark-1"
    (root / "output" / "groups_032").mkdir(parents=True)
    (root / "postprocessing" / "offsets").mkdir(parents=True)
    (root / "postprocessing" / "trees" / "LHaloTree").mkdir(parents=True)

    with h5py.File(root / "output" / "groups_032" / "fof_subhalo_tab_032.0.hdf5", "w") as handle:
        handle.create_group("Header").attrs["Redshift"] = 11.96
        handle.create_group("Group").create_dataset("Group_M_Crit200", data=[1.0])
        handle.create_group("Subhalo").create_dataset("SubhaloMass", data=[0.5])

    with h5py.File(root / "postprocessing" / "offsets" / "offsets_032.hdf5", "w") as handle:
        lhalo = handle.create_group("Subhalo").create_group("LHaloTree")
        lhalo.create_dataset("File", data=[0])
        lhalo.create_dataset("Num", data=[0])
        lhalo.create_dataset("Index", data=[0])

    desc_path = root / "postprocessing" / "trees" / "LHaloTree" / "sub_desc_sf1_080"
    desc_path.write_bytes((3).to_bytes(4, "little", signed=True) + b"\x00" * 12)
    tree_path = root / "postprocessing" / "trees" / "LHaloTree" / "trees_sf1_190.0.hdf5"
    with h5py.File(tree_path, "w") as handle:
        handle.create_group("Header").create_dataset("Redshifts", data=[12.5, 11.96])
        tree = handle.create_group("Tree0")
        tree.create_dataset("SnapNum", data=[32, 31])
        tree.create_dataset("FirstProgenitor", data=[1, -1])
        tree.create_dataset("FirstHaloInFOFGroup", data=[0, 1])
        tree.create_dataset("SubhaloGrNr", data=[0, 0])
        tree.create_dataset("Group_M_Crit200", data=[1.0, 0.5])
    integrity_manifest = _write_integrity_manifest(
        root,
        [
            root / "output" / "groups_032" / "fof_subhalo_tab_032.0.hdf5",
            root / "postprocessing" / "offsets" / "offsets_032.hdf5",
            desc_path,
            tree_path,
        ],
        tmp_path / "integrity.json",
    )

    report = module._validate_smoke_files(
        root=root,
        snapshot=32,
        tree_chunk=0,
        integrity_manifest=integrity_manifest,
    )

    assert report["groupcat_has_header"] is True
    assert report["groupcat_has_group"] is True
    assert report["groupcat_has_subhalo"] is True
    assert report["offset_mapping_dataset_count"] >= 3
    assert report["sub_desc_entry_count"] == 3
    assert report["sub_desc_dtype"] == "int32"
    assert report["tree_group_count"] == 1
    assert report["tree_mass_fields"] == ["Group_M_Crit200"]
    assert report["integrity_verified_file_count"] == 4


def test_validate_smoke_files_rejects_tree_missing_required_dataset(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "thesan-dark-1"
    groupcat = root / "output" / "groups_032" / "fof_subhalo_tab_032.0.hdf5"
    offset = root / "postprocessing" / "offsets" / "offsets_032.hdf5"
    tree_dir = root / "postprocessing" / "trees" / "LHaloTree"
    groupcat.parent.mkdir(parents=True)
    offset.parent.mkdir(parents=True)
    tree_dir.mkdir(parents=True)
    with h5py.File(groupcat, "w") as handle:
        handle.create_group("Header")
        handle.create_group("Group")
        handle.create_group("Subhalo")
    with h5py.File(offset, "w") as handle:
        lhalo = handle.create_group("Subhalo/LHaloTree")
        lhalo.create_dataset("File", data=[0])
        lhalo.create_dataset("Num", data=[0])
        lhalo.create_dataset("Index", data=[0])
    sub_desc = tree_dir / "sub_desc_sf1_080"
    sub_desc.write_bytes((1).to_bytes(4, "little", signed=True))
    tree_path = tree_dir / "trees_sf1_190.0.hdf5"
    with h5py.File(tree_path, "w") as handle:
        handle.create_group("Header").create_dataset("Redshifts", data=[12.5, 11.96])
        tree = handle.create_group("Tree0")
        tree.create_dataset("SnapNum", data=[32])
        tree.create_dataset("Group_M_Crit200", data=[1.0])
    manifest = _write_integrity_manifest(
        root,
        [groupcat, offset, sub_desc, tree_path],
        tmp_path / "integrity.json",
    )

    with pytest.raises(KeyError, match="FirstProgenitor"):
        module._validate_smoke_files(
            root=root,
            snapshot=32,
            tree_chunk=0,
            integrity_manifest=manifest,
        )


def test_validate_smoke_files_rejects_integrity_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "thesan-dark-1"
    required = {
        "output/groups_032/fof_subhalo_tab_032.0.hdf5": {"size_bytes": 123},
        "postprocessing/offsets/offsets_032.hdf5": {"size_bytes": 123},
        "postprocessing/trees/LHaloTree/sub_desc_sf1_080": {"size_bytes": 123},
        "postprocessing/trees/LHaloTree/trees_sf1_190.0.hdf5": {"size_bytes": 123},
    }
    manifest = tmp_path / "integrity.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "auroralf.thesan_download_integrity.v1",
                "simulation": "Thesan-Dark-1",
                "files": required,
            }
        ),
        encoding="utf-8",
    )
    for relative_path in required:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-the-authoritative-size")

    with pytest.raises(ValueError, match="size mismatch"):
        module._validate_smoke_files(
            root=root,
            snapshot=32,
            tree_chunk=0,
            integrity_manifest=manifest,
        )


def test_prepare_thesan_download_help_exposes_stage_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--stage" in completed.stdout
    assert "--snapshot-redshift-file" in completed.stdout
    assert "--source-root" in completed.stdout
    assert "--write-globus-batch" in completed.stdout
    assert "--validate-smoke" in completed.stdout
    assert "--smoke-integrity-manifest" in completed.stdout
