from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "experiments"
    / "arxiv_2608_05531"
    / "bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("arxiv_2608_05531_bootstrap", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)

DWARF_SCRIPT_PATH = SCRIPT_PATH.with_name("check_dwarf_mass_fraction.py")
DWARF_SPEC = importlib.util.spec_from_file_location(
    "arxiv_2608_05531_dwarf_fraction", DWARF_SCRIPT_PATH
)
assert DWARF_SPEC is not None and DWARF_SPEC.loader is not None
dwarf_fraction = importlib.util.module_from_spec(DWARF_SPEC)
DWARF_SPEC.loader.exec_module(dwarf_fraction)

REPRODUCTION_DIR = SCRIPT_PATH.parent
ML_NOTEBOOK_PATH = REPRODUCTION_DIR / "ML_REPRODUCTION.ipynb"
REPRODUCTION_CONFIG_PATH = REPRODUCTION_DIR / "reproduction.toml"


def test_manifest_and_stage_sizes_match_zenodo_record() -> None:
    manifest = bootstrap.load_manifest()
    expected_sizes = {
        "ml-min": 133_805_200,
        "ml-verify": 173_047_162,
        "maps": 149_100_225,
        "all": 260_933_673,
    }
    for stage, expected_size in expected_sizes.items():
        entries = bootstrap.selected_entries(manifest, stage)
        assert sum(entry["size_bytes"] for entry in entries) == expected_size


def test_manifest_file_names_are_unique() -> None:
    manifest = bootstrap.load_manifest()
    indexed = bootstrap.index_files(manifest)
    assert len(indexed) == len(manifest["files"]) == 8


def test_verify_file_accepts_exact_content_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.bin"
    content = b"auditable input"
    path.write_bytes(content)
    entry = {
        "size_bytes": len(content),
        "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
    }
    bootstrap.verify_file(path, entry)

    path.write_bytes(b"short")
    with pytest.raises(bootstrap.InputVerificationError, match="Size mismatch"):
        bootstrap.verify_file(path, entry)

    path.write_bytes(b"corrupted input")
    with pytest.raises(bootstrap.InputVerificationError, match="MD5 mismatch"):
        bootstrap.verify_file(path, entry)


def test_paper_gsmf_parameters_give_recomputed_dwarf_mass_fraction() -> None:
    config = dwarf_fraction.load_gsmf_config(dwarf_fraction.DEFAULT_CONFIG)
    calculated_percent = 100.0 * dwarf_fraction.calculate_fraction(config)
    assert calculated_percent == pytest.approx(1.545924113, abs=1.0e-9)
    assert calculated_percent != pytest.approx(
        config["paper_mass_fraction_percent"], abs=0.005
    )


def test_ml_notebook_is_clean_and_uses_author_feature_order() -> None:
    notebook = json.loads(ML_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(notebook["cells"]) == 25
    assert len(code_cells) == 15
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    for cell in code_cells:
        compile("".join(cell["source"]), ML_NOTEBOOK_PATH.name, "exec")

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert r"\[" not in markdown
    assert r"\]" not in markdown
    assert r"\(" not in markdown
    assert r"\)" not in markdown
    assert sum(line.strip() == "$$" for line in markdown.splitlines()) == 20

    config = tomllib.loads(REPRODUCTION_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["ml"]["author_feature_order"] == [
        "g-r",
        "r-z",
        "z-W1",
        "W1-W2",
        "redshift",
        "gmag",
        "logM",
    ]
    assert config["expected"]["published_table1_metrics"] == [
        "rmse_dex",
        "sigma_dex",
        "bias_dex",
        "outlier_fraction_3sigma",
    ]
