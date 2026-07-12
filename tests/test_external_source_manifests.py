from __future__ import annotations

from pathlib import Path
import re
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERAXES_MANIFEST = PROJECT_ROOT / "external_data" / "source_manifests" / "meraxes.toml"


def test_meraxes_manifest_records_exact_source_license_build_and_patch_state() -> None:
    payload = tomllib.loads(MERAXES_MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "auroralf.external_source.v1"
    assert payload["name"] == "Meraxes"
    assert payload["source_url"] == "https://github.com/meraxes-devs/meraxes.git"
    assert re.fullmatch(r"[0-9a-f]{40}", payload["commit"])
    assert payload["retrieved_at"] == "2026-06-17T10:46:40+08:00"
    assert payload["local_source_path"] == "third_party/meraxes"

    license_record = payload["license"]
    assert license_record["spdx"] == "NOASSERTION"
    assert license_record["redistribution_cleared"] is False
    assert "no repository-level license" in license_record["status"]

    bundled = payload["bundled_licenses"]
    assert bundled == [
        {
            "component": "src/mlog",
            "spdx": "MIT",
            "path": "src/mlog/LICENSE",
            "sha256": "28234bd5e43a9251f12f4c3c7d1cfeb309435bd75878906d3c6736ac785aaf7e",
            "scope_note": (
                "This license applies to the bundled mlog component only, "
                "not to the whole Meraxes repository."
            ),
        }
    ]

    build = payload["build"]
    assert build["build_type"] == "Release"
    assert build["parallel_build_jobs"] == 8
    assert build["runtime_mpi_ranks"] == 24
    assert "-DUSE_MINI_HALOS=ON" in build["cmake_flags"]
    assert "-DCALC_MAGS=OFF" in build["cmake_flags"]
    assert "-DN_HISTORY_SNAPS=120" in build["cmake_flags"]

    patches = payload["local_patches"]
    assert patches["status"] == "clean"
    assert patches["count"] == 0
    assert patches["sha256"] == []
    assert re.fullmatch(r"[0-9a-f]{64}", patches["empty_diff_sha256"])


def test_generated_source_and_run_trees_are_ignored() -> None:
    patterns = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "third_party/" in patterns
    assert "runs/" in patterns
