from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from auroralf.io import read_uvlf_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"
CANONICAL_SSP = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "bpass_byrne23_imf135_300"
    / "BASEL"
    / "spectra-bin-imf135_300.BASEL.z001.a+00.dat"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_uvlf_v2_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_uvlf_v2_test"] = module
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    output = (tmp_path / "result.h5").resolve()
    config = tmp_path / "run.toml"
    config.write_text(
        f'''
schema_version = "2.0.0"
run_id = "cli-reduced"
redshifts = [10.0]
base_seed = 20260711

[cosmology]
h0_km_s_mpc = 67.4
omega_m = 0.315
omega_b = 0.04897

[mah]
backend = "mcbride"
sampler = "mcbride"
z_start_max = 50.0
n_time_steps = 32
tng_mass_bin_width_dex = 0.15
tng_min_candidates = 5
tng_smoothing_myr = 0.0
tng_time_grid_mode = "snapshot"
thesan_mass_bin_width_dex = 0.15
thesan_min_candidates = 5
thesan_smoothing_myr = 0.0
thesan_time_grid_mode = "snapshot"

[star_formation]
enable_time_delay = true
efficiency_normalization = 0.12
characteristic_halo_mass_msun = 5.011872336e11
low_mass_slope = 0.66
high_mass_slope = 0.65
burst_scatter_dex = 0.0
burst_scatter_correlation_timescale_myr = 20.0
burst_scatter_mass_conserving = true
metallicity_source = "none"

[stellar_population]
imf_modes = ["canonical"]
canonical_ssp_path = "{CANONICAL_SSP}"
topheavy_ssp_path = "{CANONICAL_SSP}"
topheavy_ssp_template_metallicity_zsun = 0.05
historical_topheavy_redshift_min = 10.0
source_redshift_gate_enabled = false
growth_time_threshold_myr = 50.0
birth_metallicity_topheavy_max_zsun = 0.05
enable_popiii = false
popiii_ssp_path = "{CANONICAL_SSP}"
popiii_efficiency = 0.001
popiii_pivot_halo_mass_msun = 10000000.0
popiii_low_mass_slope = 0.0
popiii_high_mass_slope = 0.0
lw_background_j21 = 0.0
popiii_upper_mass_mode = "atomic"

[sampling]
mass_batch_size = 1
n_halo_mass_samples = 2
n_tracks_per_halo_mass = 2
log10_halo_mass_min_msun = 9.0
log10_halo_mass_max_msun = 10.0
muv_bin_edges = [-24.0, -20.0, -16.0, -12.0]
workers = 1
mass_function_model = "hmf_reed07"
hmf_dlog10m = 0.02
apply_dust = false

[output]
artifact_path = "{output}"
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    return config, output


def _slurm_env(execution_manifest: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["SLURM_JOB_ID"] = "900001"
    env["SLURM_CPUS_PER_TASK"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if execution_manifest is not None:
        env["AURORALF_EXECUTION_MANIFEST"] = str(execution_manifest)
    else:
        env.pop("AURORALF_EXECUTION_MANIFEST", None)
    return env


def test_cli_flags_are_mutually_exclusive_and_slurm_is_required(tmp_path: Path) -> None:
    module = _load_module()
    config, _output = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        module._parse_args(["--config", str(config), "--resume", "--overwrite"])

    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_CPUS_PER_TASK", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires a SLURM allocation" in completed.stderr


def test_production_config_is_strict_and_uses_unified_modes() -> None:
    from auroralf import UVLFRunConfig

    config = UVLFRunConfig.from_toml(PRODUCTION_CONFIG)
    assert config.mah.backend == "mcbride"
    assert config.star_formation.metallicity_source == "regulator"
    assert config.stellar_population.imf_modes == (
        "canonical",
        "z10_mild_topheavy",
        "mah_burst_mild_topheavy",
    )
    assert config.stellar_population.source_redshift_gate_enabled is False
    assert config.stellar_population.birth_metallicity_topheavy_max_zsun == 0.05
    assert config.sampling.workers == 32
    assert config.output.artifact_path == PROJECT_ROOT / "data_save" / "uvlf_v2_production.h5"


def test_cli_fresh_resume_overwrite_and_strict_final_readback(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    execution_manifest = tmp_path / "execution.json"
    execution_manifest.write_text(
        json.dumps(
            {
                "job_id": "900001",
                "requested_cpus": 1,
                "requested_memory": "2G",
                "requested_time": "00:10:00",
                "command": ["run_uvlf_v2.py", "--config", str(config)],
                "stdout_path": str(tmp_path / "job.out"),
                "stderr_path": str(tmp_path / "job.err"),
                "exit_code": 0,
                "final_artifact_validated": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    env = _slurm_env(execution_manifest)
    command = [sys.executable, str(SCRIPT_PATH), "--config", str(config)]
    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert f"uvlf_v2_artifact={output}" in first.stdout
    artifact = read_uvlf_artifact(output, load_samples=False)
    assert artifact.result.config.run_id == "cli-reduced"
    assert output.with_name(output.name + ".complete").is_file()
    assert artifact.provenance.source_checksums[-1].label == "slurm_execution"
    assert artifact.provenance.source_checksums[-1].path == execution_manifest.resolve()

    duplicate = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr

    output.unlink()
    output.with_name(output.name + ".complete").unlink()
    resumed_from_shards = subprocess.run(
        [*command, "--resume"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert f"uvlf_v2_artifact={output}" in resumed_from_shards.stdout
    read_uvlf_artifact(output, load_samples=False)

    resumed = subprocess.run(
        [*command, "--resume"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert f"uvlf_v2_artifact={output}" in resumed.stdout

    overwritten = subprocess.run(
        [*command, "--overwrite"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert f"uvlf_v2_artifact={output}" in overwritten.stdout
    read_uvlf_artifact(output, load_samples=False)


def test_cli_resume_rejects_incomplete_final_artifact(tmp_path: Path) -> None:
    config, output = _write_config(tmp_path)
    output.write_bytes(b"incomplete")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config), "--resume"],
        cwd=PROJECT_ROOT,
        env=_slurm_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "incomplete artifact pair" in completed.stderr
