from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def test_configuration_and_result_models_do_not_import_uvlf_implementation() -> None:
    for relative in (
        "auroralf/config.py",
        "auroralf/results.py",
        "auroralf/io/schema.py",
    ):
        imports = _absolute_imports(PROJECT_ROOT / relative)
        assert not any(name == "auroralf.uvlf" or name.startswith("auroralf.uvlf.") for name in imports)


def test_mah_does_not_depend_on_top_level_cooling_router() -> None:
    imports: set[str] = set()
    for path in (PROJECT_ROOT / "auroralf/mah").glob("*.py"):
        imports.update(_absolute_imports(path))
    assert "auroralf.cooling" not in imports


def test_uvlf_runner_uses_domain_sample_model_not_io_schema() -> None:
    imports = _absolute_imports(PROJECT_ROOT / "auroralf/uvlf/runner.py")
    assert "auroralf.driver" in imports
    assert "auroralf.run_plan" in imports
    assert "auroralf.samples" in imports
    assert not any(name == "auroralf.io" or name.startswith("auroralf.io.") for name in imports)


def test_driver_is_the_only_run_plan_composition_root() -> None:
    driver_imports = _absolute_imports(PROJECT_ROOT / "auroralf/driver.py")
    plan_imports = _absolute_imports(PROJECT_ROOT / "auroralf/run_plan.py")

    assert "auroralf.uvlf.pipeline" in driver_imports
    assert "auroralf.uvlf.pipeline" not in plan_imports
    assert "auroralf.driver" not in plan_imports

    runner_source = (PROJECT_ROOT / "auroralf/uvlf/runner.py").read_text(
        encoding="utf-8"
    )
    assert runner_source.count("build_uvlf_run_plan(") == 1


def test_uvlf_public_api_excludes_archived_imf_gate() -> None:
    import auroralf.uvlf as uvlf

    archived = {
        "IMF_MODE_MAH_BURST_MILD_TOPHEAVY",
        "IMF_MODE_Z_GATED_MILD_TOPHEAVY",
        "IMFTransitionParameters",
        "compute_topheavy_source_flags",
    }
    assert archived.isdisjoint(uvlf.__all__)
    assert all(not hasattr(uvlf, name) for name in archived)
