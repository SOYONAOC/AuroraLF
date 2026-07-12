from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from auroralf.config import OutputConfig
from auroralf.results import RedshiftResult, RunDiagnostics, UVLFRunResult
from tests.test_hdf5_artifact import _config as _artifact_config
from tests.test_hdf5_artifact import _result as _artifact_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "benchmark_uvlf_v2_streaming.py"
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "submit" / "submit_uvlf_v2_benchmark.py"


def _load_module(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_module():  # type: ignore[no-untyped-def]
    return _load_module(BENCHMARK_SCRIPT, "benchmark_uvlf_v2_streaming_test")


def _case_payload(
    module: object,
    case: str,
    digest: str,
    peak: int,
    config: dict[str, object],
) -> dict[str, object]:
    sample_shards: list[dict[str, object]] = []
    if case == "parallel_samples":
        sample_shards.append(
            {
                "filename": "uvlf-v2-benchmark-parallel_samples.z=10.canonical.shard.h5",
                "redshift": 10.0,
                "imf_mode": "canonical",
                "sample_count": 128,
                "order_validated": True,
            }
        )
    return {
        "schema_name": module.CASE_SCHEMA_NAME,
        "schema_version": module.SCHEMA_VERSION,
        "complete": True,
        "case": case,
        "config": config,
        "wall_seconds": 1.0,
        "peak_rss_bytes": peak,
        "science_digest": digest,
        "sample_shards": sample_shards,
    }


def test_benchmark_and_submit_scripts_exist_and_import_without_execution() -> None:
    assert BENCHMARK_SCRIPT.is_file()
    assert SUBMIT_SCRIPT.is_file()
    benchmark = _load_module(BENCHMARK_SCRIPT, "benchmark_uvlf_v2_streaming_import_test")
    submit = _load_module(SUBMIT_SCRIPT, "submit_uvlf_v2_benchmark_import_test")
    assert benchmark.CASES == (
        "serial_disabled",
        "parallel_disabled",
        "parallel_samples",
    )
    assert callable(benchmark.main)
    assert callable(submit.main)


def test_science_digest_excludes_timing_and_execution_only_config(
    benchmark_module: object,
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    result = _artifact_result(config)
    changed_diagnostics = RunDiagnostics(
        total_seconds=result.diagnostics.total_seconds + 100.0,
        mode_runs=tuple(
            replace(item, sampling_seconds=item.sampling_seconds + 50.0)
            for item in result.diagnostics.mode_runs
        ),
    )
    changed_config = replace(
        config,
        run_id="digest-execution-only",
        sampling=replace(
            config.sampling,
            workers=2,
            mass_batch_size=config.sampling.mass_batch_size + 1,
        ),
        output=OutputConfig((tmp_path / "other-output.h5").resolve()),
    )
    changed = UVLFRunResult(
        config=changed_config,
        redshifts=result.redshifts,
        diagnostics=changed_diagnostics,
    )

    assert benchmark_module._science_digest(result) == benchmark_module._science_digest(changed)


def test_science_digest_changes_for_result_array_or_physical_config(
    benchmark_module: object,
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    result = _artifact_result(config)
    first_redshift = result.redshifts[0]
    first_mode = first_redshift.imf_modes[0]
    changed_phi = np.array(first_mode.phi_intrinsic_per_mpc3_per_mag, copy=True)
    changed_phi[0] = np.nextafter(changed_phi[0], np.inf)
    changed_mode = replace(first_mode, phi_intrinsic_per_mpc3_per_mag=changed_phi)
    changed_redshift = RedshiftResult(
        redshift=first_redshift.redshift,
        imf_modes=(changed_mode, *first_redshift.imf_modes[1:]),
    )
    changed_result = UVLFRunResult(
        config=config,
        redshifts=(changed_redshift, *result.redshifts[1:]),
        diagnostics=result.diagnostics,
    )
    physical_config = replace(config, base_seed=config.base_seed + 1)
    physical_result = UVLFRunResult(
        config=physical_config,
        redshifts=result.redshifts,
        diagnostics=result.diagnostics,
    )

    digest = benchmark_module._science_digest(result)
    assert benchmark_module._science_digest(changed_result) != digest
    assert benchmark_module._science_digest(physical_result) != digest


def test_aggregate_rss_includes_self_and_recursive_children(
    benchmark_module: object,
) -> None:
    class FakeProcess:
        def __init__(self, rss: int, children: tuple[object, ...] = ()) -> None:
            self._rss = rss
            self._children = children

        def memory_info(self) -> object:
            return SimpleNamespace(rss=self._rss)

        def children(self, *, recursive: bool) -> list[object]:
            assert recursive is True
            return list(self._children)

    root = FakeProcess(100, (FakeProcess(20), FakeProcess(30)))
    assert benchmark_module._aggregate_rss_bytes(root) == 150


def test_peak_sampler_uses_mocked_psutil_process(
    benchmark_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def memory_info(self) -> object:
            return SimpleNamespace(rss=321)

        def children(self, *, recursive: bool) -> list[object]:
            assert recursive is True
            return []

    monkeypatch.setattr(benchmark_module.psutil, "Process", lambda pid: FakeProcess())
    sampler = benchmark_module._PeakRSSSampler(0.001)
    sampler.start()
    assert sampler.stop() == 321


def test_build_config_is_reduced_real_canonical_mcbride(
    benchmark_module: object,
    tmp_path: Path,
) -> None:
    args = benchmark_module._parse_args(
        [
            "--n-mass",
            "3",
            "--n-tracks",
            "4",
            "--n-grid",
            "5",
            "--n-bins",
            "6",
            "--mass-batch-size",
            "2",
        ]
    )
    serial = benchmark_module._build_config(
        args,
        case="serial_disabled",
        work_directory=tmp_path,
    )
    parallel = benchmark_module._build_config(
        args,
        case="parallel_samples",
        work_directory=tmp_path,
    )

    assert serial.redshifts == (10.0,)
    assert serial.mah.backend == "mcbride"
    assert serial.mah.sampler == "mcbride"
    assert serial.mah.n_time_steps == 5
    assert serial.stellar_population.imf_modes == ("canonical",)
    assert serial.stellar_population.canonical_ssp_path.is_absolute()
    assert serial.stellar_population.canonical_ssp_path.is_file()
    assert serial.sampling.n_halo_mass_samples == 3
    assert serial.sampling.n_tracks_per_halo_mass == 4
    assert len(serial.sampling.muv_bin_edges) == 7
    assert serial.sampling.workers == 1
    assert parallel.sampling.workers == 2
    assert benchmark_module._physical_config_mapping(serial) == benchmark_module._physical_config_mapping(parallel)


def test_environment_metadata_has_explicit_git_env_python_platform_sections(
    benchmark_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_module, "_git_state", lambda: ("a" * 40, True))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    metadata = benchmark_module._environment_metadata()

    assert set(metadata) == {"git", "env", "python", "platform", "packages"}
    assert metadata["git"] == {"revision": "a" * 40, "dirty": True}
    assert metadata["env"]["SLURM_JOB_ID"] == "123"
    assert Path(metadata["python"]["executable"]).is_absolute()
    assert metadata["platform"]["machine"] == platform.machine()


def test_atomic_json_write_uses_0600_and_cleans_failed_temp(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    payload = {"complete": True, "value": [1, 2, 3]}
    benchmark_module._write_json_atomic(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    failed_path = tmp_path / "failed.json"

    def fail_link(source: object, target: object, **kwargs: object) -> None:
        del kwargs
        del source, target
        raise RuntimeError("link failed")

    monkeypatch.setattr(benchmark_module.os, "link", fail_link)
    with pytest.raises(RuntimeError, match="link failed"):
        benchmark_module._write_json_atomic(payload, failed_path)
    assert not failed_path.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_json_is_hard_link_no_clobber_with_exactly_one_thread_winner(
    benchmark_module: object,
    tmp_path: Path,
) -> None:
    assert "os.replace" not in inspect.getsource(benchmark_module._write_json_atomic)
    target = tmp_path / "contended.json"
    barrier = threading.Barrier(2)

    def write(value: int) -> tuple[str, int]:
        barrier.wait()
        try:
            benchmark_module._write_json_atomic(
                {"complete": True, "writer": value},
                target,
            )
        except FileExistsError:
            return ("exists", value)
        return ("success", value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(write, (1, 2)))

    assert sorted(status for status, _ in outcomes) == ["exists", "success"]
    winner = next(value for status, value in outcomes if status == "success")
    assert json.loads(target.read_text(encoding="utf-8"))["writer"] == winner
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_json_existing_target_is_never_clobbered(
    benchmark_module: object,
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing.json"
    original = b'{"complete":true,"old":1}\n'
    target.write_bytes(original)

    with pytest.raises(FileExistsError):
        benchmark_module._write_json_atomic({"complete": True, "new": 2}, target)

    assert target.read_bytes() == original


def test_atomic_json_post_link_failure_removes_only_owned_target(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "post-link-failure.json"
    calls = 0

    def fail_first_fsync(path: Path) -> None:
        nonlocal calls
        del path
        calls += 1
        if calls == 1:
            raise RuntimeError("directory fsync failed")

    monkeypatch.setattr(benchmark_module, "_fsync_directory", fail_first_fsync)

    with pytest.raises(RuntimeError, match="directory fsync failed"):
        benchmark_module._write_json_atomic({"complete": True}, target)

    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(
    benchmark_module: object,
) -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        benchmark_module._loads_json_strict('{"case":"a","case":"b"}')
    for text in ('{"value":NaN}', '{"value":Infinity}', '{"value":1e999}'):
        with pytest.raises(ValueError, match="non-finite JSON number"):
            benchmark_module._loads_json_strict(text)


def test_canonical_array_payload_normalizes_endian_nan_payload_and_negative_zero(
    benchmark_module: object,
) -> None:
    first_bits = np.array(
        [0x7FF8000000000001, 0x8000000000000000, 0x3FF0000000000000],
        dtype=np.uint64,
    )
    second_bits = np.array(
        [0x7FF80000000000FF, 0x0000000000000000, 0x3FF0000000000000],
        dtype=np.uint64,
    )
    first = first_bits.view(np.float64)
    second = second_bits.view(np.float64).astype(">f8")

    first_payload = benchmark_module._canonical_array_payload(first)
    second_payload = benchmark_module._canonical_array_payload(second)

    assert first_payload == second_payload
    changed_position = np.array([1.0, np.nan, 0.0], dtype=np.float64)
    assert benchmark_module._canonical_array_payload(changed_position) != first_payload


def test_peak_sampler_stop_tolerates_final_access_denied(
    benchmark_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.calls = 0

        def memory_info(self) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(rss=321)
            raise benchmark_module.psutil.AccessDenied(pid=1)

        def children(self, *, recursive: bool) -> list[object]:
            assert recursive is True
            return []

    process = FakeProcess()
    monkeypatch.setattr(benchmark_module.psutil, "Process", lambda pid: process)
    sampler = benchmark_module._PeakRSSSampler(60.0)
    sampler.start()
    assert sampler.stop() == 321


def test_controller_runs_three_independent_cases_and_writes_complete_report(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "benchmark.json"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        calls.append(command)
        case = command[command.index("--child-case") + 1]
        child_output = Path(command[command.index("--child-output") + 1])
        config = benchmark_module._build_config(
            args,
            case=case,
            work_directory=child_output.parent,
        )
        benchmark_module._write_json_atomic(
            _case_payload(
                benchmark_module,
                case,
                "a" * 64,
                100 + len(calls),
                benchmark_module.canonical_config_mapping(config),
            ),
            child_output,
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = benchmark_module._parse_args(["--report", str(report)])

    benchmark_module._run_controller(args)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["digest_equal"] is True
    assert [item["case"] for item in payload["cases"]] == list(benchmark_module.CASES)
    assert [item["exit_code"] for item in payload["cases"]] == [0, 0, 0]
    assert len(calls) == 3
    assert len({tuple(command) for command in calls}) == 3
    assert payload["memory_overhead"]["parallel_samples_vs_parallel_disabled_bytes"] == 1


def test_controller_timeout_uses_fixed_child_environment_and_cleans_reports(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "benchmark.json"

    def timeout_run(command: list[str], **kwargs: object) -> object:
        assert kwargs["timeout"] == 7.5
        environment = kwargs["env"]
        assert environment["PYTHONPATH"] == str(PROJECT_ROOT)
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["MKL_NUM_THREADS"] == "1"
        assert environment["OPENBLAS_NUM_THREADS"] == "1"
        assert environment["PYTHONHASHSEED"] == "0"
        raise subprocess.TimeoutExpired(command, timeout=7.5)

    monkeypatch.setattr(benchmark_module.subprocess, "run", timeout_run)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = benchmark_module._parse_args(
        [
            "--report",
            str(report),
            "--child-timeout-seconds",
            "7.5",
        ]
    )

    with pytest.raises(RuntimeError, match="timed out"):
        benchmark_module._run_controller(args)

    assert not report.exists()
    assert not tuple(tmp_path.glob(".uvlf-v2-benchmark-controller-*"))


@pytest.mark.parametrize("attack", ("symlink", "hardlink", "replace_during_read"))
def test_controller_rejects_unstable_or_external_child_report(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    report = tmp_path / "benchmark.json"
    external = tmp_path / f"external-{attack}.json"
    active_child_path: list[Path] = []
    real_pread = benchmark_module.os.pread
    real_replace = benchmark_module.os.replace
    replaced = False
    call_count = 0

    def replace_during_pread(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal replaced
        data = real_pread(descriptor, size, offset)
        if attack == "replace_during_read" and not replaced:
            replacement = external.with_name(external.name + ".replacement")
            os.link(external, replacement)
            real_replace(replacement, active_child_path[0])
            replaced = True
        return data

    monkeypatch.setattr(benchmark_module.os, "pread", replace_during_pread)

    def fake_run(command: list[str], **kwargs: object) -> object:
        nonlocal call_count
        del kwargs
        call_count += 1
        case = command[command.index("--child-case") + 1]
        child_output = Path(command[command.index("--child-output") + 1])
        active_child_path[:] = [child_output]
        args = benchmark_module._parse_args(["--report", str(report)])
        config = benchmark_module._build_config(
            args,
            case=case,
            work_directory=child_output.parent,
        )
        payload = _case_payload(
            benchmark_module,
            case,
            "a" * 64,
            100,
            benchmark_module.canonical_config_mapping(config),
        )
        benchmark_module._write_json_atomic(payload, external)
        if attack == "symlink":
            os.symlink(external, child_output)
        elif attack == "hardlink":
            os.link(external, child_output)
        else:
            benchmark_module._write_json_atomic(payload, child_output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = benchmark_module._parse_args(["--report", str(report)])

    with pytest.raises((RuntimeError, ValueError)):
        benchmark_module._run_controller(args)

    assert not report.exists()
    assert call_count == 1
    assert external.exists()
    assert not tuple(tmp_path.glob(".uvlf-v2-benchmark-controller-*"))


@pytest.mark.parametrize("failure", ("nonzero", "digest", "incomplete"))
def test_controller_failure_never_writes_complete_report(
    benchmark_module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    report = tmp_path / "benchmark.json"
    call_count = 0

    def fake_run(command: list[str], **kwargs: object) -> object:
        nonlocal call_count
        del kwargs
        call_count += 1
        case = command[command.index("--child-case") + 1]
        child_output = Path(command[command.index("--child-output") + 1])
        if failure == "nonzero" and call_count == 2:
            return SimpleNamespace(returncode=7, stdout="", stderr="child failed")
        digest = ("b" * 64) if failure == "digest" and call_count == 3 else ("a" * 64)
        config = benchmark_module._build_config(
            args,
            case=case,
            work_directory=child_output.parent,
        )
        payload = _case_payload(
            benchmark_module,
            case,
            digest,
            100,
            benchmark_module.canonical_config_mapping(config),
        )
        if failure == "incomplete" and case == "parallel_samples":
            payload["sample_shards"] = []
        benchmark_module._write_json_atomic(payload, child_output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = benchmark_module._parse_args(["--report", str(report)])

    with pytest.raises((RuntimeError, ValueError)):
        benchmark_module._run_controller(args)

    assert not report.exists()


@pytest.mark.parametrize("mutation", ("unknown", "missing", "wrong_filename"))
def test_parallel_sample_summary_requires_exact_keys_and_filename(
    benchmark_module: object,
    tmp_path: Path,
    mutation: str,
) -> None:
    args = benchmark_module._parse_args([])
    config = benchmark_module._build_config(
        args,
        case="parallel_samples",
        work_directory=tmp_path,
    )
    payload = _case_payload(
        benchmark_module,
        "parallel_samples",
        "a" * 64,
        100,
        benchmark_module.canonical_config_mapping(config),
    )
    summary = payload["sample_shards"][0]
    if mutation == "unknown":
        summary["extra"] = True
    elif mutation == "missing":
        del summary["filename"]
    else:
        summary["filename"] = "wrong.shard.h5"

    with pytest.raises(ValueError, match="shard summary"):
        benchmark_module._validate_case_payload(
            payload,
            "parallel_samples",
            expected_sample_count=args.n_mass * args.n_tracks,
            args=args,
        )


def test_benchmark_cli_requires_slurm_but_help_is_side_effect_free(tmp_path: Path) -> None:
    help_run = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--n-mass" in help_run.stdout
    assert "--n-tracks" in help_run.stdout
    assert "--n-grid" in help_run.stdout
    assert "--n-bins" in help_run.stdout

    environment = dict(os.environ)
    environment.pop("SLURM_JOB_ID", None)
    report = tmp_path / "must-not-exist.json"
    failed = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--report", str(report)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert failed.returncode != 0
    assert "SLURM allocation" in failed.stderr
    assert not report.exists()


def test_submit_wrapper_dry_run_renders_sbatch_and_passthrough(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SUBMIT_SCRIPT),
            "--dry-run",
            "--cpus-per-task",
            "3",
            "--mem",
            "20G",
            "--time",
            "00:20:00",
            "--child-timeout-seconds",
            "300",
            "--report",
            str(report),
            "--",
            "--n-mass",
            "3",
            "--n-tracks",
            "5",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = completed.stdout.strip()
    assert rendered.startswith("sbatch ")
    assert "--cpus-per-task 3" in rendered
    assert "--mem 20G" in rendered
    assert "--time 00:20:00" in rendered
    assert "--child-timeout-seconds 300" in rendered
    assert "--output" in rendered and "%j" in rendered
    assert "--error" in rendered
    assert str(PROJECT_ROOT / ".venv" / "bin" / "python") in rendered
    assert str(report.resolve()) in rendered
    assert "--n-mass 3" in rendered
    assert "--n-tracks 5" in rendered


def test_submit_wrapper_rejects_passthrough_without_separator() -> None:
    completed = subprocess.run(
        [sys.executable, str(SUBMIT_SCRIPT), "--dry-run", "--n-mass", "3"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "after '--'" in completed.stderr
