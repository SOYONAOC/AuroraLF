#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Sequence
import uuid

import h5py
import numpy as np
import psutil

from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.io import (
    ArtifactProvenance,
    canonical_config_mapping,
    decode_canonical_config_mapping,
    read_uvlf_shard,
    run_uvlf_to_sample_shards,
    uvlf_shard_filename,
)
from auroralf.results import UVLFRunResult
from auroralf.uvlf.runner import run_uvlf_streaming


SCHEMA_VERSION = "1.0.0"
REPORT_SCHEMA_NAME = "auroralf.uvlf_v2_streaming_benchmark"
CASE_SCHEMA_NAME = "auroralf.uvlf_v2_streaming_benchmark_case"
CASES = (
    "serial_disabled",
    "parallel_disabled",
    "parallel_samples",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CHILD_REPORT_BYTES = 16 * 1024 * 1024
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT = _PROJECT_ROOT / "outputs" / "uvlf_v2_streaming_benchmark.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _grid_size(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("n-grid must be at least 2")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AuroraLF v2 serial, parallel, and parallel sample-shard "
            "streaming inside a SLURM allocation."
        )
    )
    parser.add_argument("--report", default=str(_DEFAULT_REPORT))
    parser.add_argument("--n-mass", type=_positive_int, default=8)
    parser.add_argument("--n-tracks", type=_positive_int, default=16)
    parser.add_argument("--n-grid", type=_grid_size, default=64)
    parser.add_argument("--n-bins", type=_positive_int, default=12)
    parser.add_argument("--mass-batch-size", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--rss-interval", type=_positive_float, default=0.02)
    parser.add_argument(
        "--child-timeout-seconds",
        type=_positive_float,
        default=480.0,
        help="Maximum wall time for each isolated benchmark case.",
    )
    parser.add_argument("--child-case", choices=CASES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve_output_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return (
        (Path.cwd() / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _require_slurm_allocation() -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    raise RuntimeError(
        "The UVLF v2 benchmark must run inside a SLURM allocation; use "
        "scripts/submit/submit_uvlf_v2_benchmark.py."
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_command(arguments: list[str]) -> str:
    process = subprocess.Popen(
        ["git", *arguments],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with exit {process.returncode}: {stderr.strip()}"
        )
    return stdout


def _git_state() -> tuple[str, bool]:
    revision = _git_command(["rev-parse", "HEAD"]).strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("git revision must be a lowercase 40-character hexadecimal hash")
    dirty = bool(_git_command(["status", "--porcelain"]).strip())
    return revision, dirty


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(_PROJECT_ROOT),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _environment_metadata(
    child_environment: dict[str, str] | None = None,
) -> dict[str, object]:
    revision, dirty = _git_state()
    execution_environment = (
        _child_environment()
        if child_environment is None
        else child_environment
    )
    return {
        "git": {"revision": revision, "dirty": dirty},
        "env": {
            name: execution_environment.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "PYTHONPATH",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "description": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "h5py": h5py.__version__,
        },
    }


def _write_json_atomic(payload: dict[str, object], path: Path) -> Path:
    if type(payload) is not dict:
        raise TypeError("payload must be exactly dict")
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("JSON output path must be an absolute Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    owner = os.fstat(descriptor)
    published = False
    try:
        encoded = (
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("atomic JSON write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"JSON output already exists: {path}") from error
        published = True
        _fsync_directory(path.parent)
        if not _unlink_owned_temporary(temporary, owner):
            raise RuntimeError("atomic JSON temporary file identity changed")
        _fsync_directory(path.parent)
        return path
    except BaseException as error:
        if published:
            try:
                target_stat = os.lstat(path)
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISREG(target_stat.st_mode)
                    and (target_stat.st_dev, target_stat.st_ino)
                    == (owner.st_dev, owner.st_ino)
                ):
                    path.unlink()
                    try:
                        _fsync_directory(path.parent)
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"JSON target cleanup fsync also failed: {cleanup_error}"
                        )
        try:
            if _unlink_owned_temporary(temporary, owner):
                _fsync_directory(path.parent)
        except BaseException as cleanup_error:
            error.add_note(f"JSON temporary cleanup also failed: {cleanup_error}")
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _unlink_owned_temporary(temporary, owner)


def _unlink_owned_temporary(path: Path, owner: os.stat_result) -> bool:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino) != (owner.st_dev, owner.st_ino)
    ):
        return False
    path.unlink()
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _loads_json_strict(text: str) -> object:
    if type(text) is not str:
        raise TypeError("JSON text must be exactly str")

    def reject_constant(token: str) -> object:
        raise ValueError(f"non-finite JSON number: {token}")

    def parse_float(token: str) -> float:
        value = float(token)
        if not np.isfinite(value):
            raise ValueError(f"non-finite JSON number: {token}")
        return value

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        parse_float=parse_float,
        object_pairs_hook=strict_object,
    )


def _child_report_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _require_child_report_stat(file_stat: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"child report must be a regular file: {path}")
    if file_stat.st_uid != os.getuid():
        raise ValueError(f"child report must be owned by the current uid: {path}")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise ValueError(f"child report mode must be 0600: {path}")
    if file_stat.st_nlink != 1:
        raise ValueError(f"child report must have exactly one hard link: {path}")
    if not 0 < file_stat.st_size <= _MAX_CHILD_REPORT_BYTES:
        raise ValueError(f"child report size is outside the allowed bound: {path}")


def _read_child_json_stable(path: Path) -> object:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("child report path must be an absolute Path")
    try:
        path_before = os.lstat(path)
    except FileNotFoundError as error:
        raise RuntimeError(f"benchmark child did not write a case report: {path}") from error
    _require_child_report_stat(path_before, path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        descriptor_before = os.fstat(descriptor)
        _require_child_report_stat(descriptor_before, path)
        if _child_report_identity(descriptor_before) != _child_report_identity(path_before):
            raise ValueError("child report identity changed before read")
        chunks: list[bytes] = []
        offset = 0
        while offset < descriptor_before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, descriptor_before.st_size - offset),
                offset,
            )
            if not block:
                raise ValueError("child report ended before its recorded size")
            chunks.append(block)
            offset += len(block)
        descriptor_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        _require_child_report_stat(descriptor_after, path)
        _require_child_report_stat(path_after, path)
        expected_identity = _child_report_identity(descriptor_before)
        if (
            _child_report_identity(descriptor_after) != expected_identity
            or _child_report_identity(path_after) != expected_identity
        ):
            raise ValueError("child report identity changed while being read")
    finally:
        os.close(descriptor)
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("child report is not strict UTF-8") from error
    return _loads_json_strict(text)


def _digest_part(digest: object, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _physical_config_mapping(config: UVLFRunConfig) -> dict[str, object]:
    mapping = json.loads(
        json.dumps(canonical_config_mapping(config), sort_keys=True, allow_nan=False)
    )
    del mapping["run_id"]
    del mapping["output"]
    del mapping["sampling"]["workers"]
    del mapping["sampling"]["mass_batch_size"]
    return mapping


def _canonical_array_payload(array: np.ndarray) -> tuple[bytes, bytes]:
    source = np.asarray(array)
    if source.dtype == np.dtype(object) or np.issubdtype(
        source.dtype,
        np.complexfloating,
    ):
        raise TypeError("scientific digest arrays must have real numeric dtypes")
    if np.issubdtype(source.dtype, np.floating):
        canonical_dtype = np.dtype(f"<f{source.dtype.itemsize}")
        canonical = np.array(source, dtype=canonical_dtype, order="C", copy=True)
        canonical[canonical == 0] = 0.0
        canonical[np.isnan(canonical)] = np.nan
    elif np.issubdtype(source.dtype, np.integer):
        canonical_dtype = np.dtype(
            f"<{'i' if np.issubdtype(source.dtype, np.signedinteger) else 'u'}"
            f"{source.dtype.itemsize}"
        )
        canonical = np.array(source, dtype=canonical_dtype, order="C", copy=True)
    elif np.issubdtype(source.dtype, np.bool_):
        canonical = np.array(source, dtype=np.dtype("|b1"), order="C", copy=True)
    else:
        raise TypeError("scientific digest arrays must have numeric or boolean dtypes")
    metadata = json.dumps(
        {"dtype": canonical.dtype.str, "shape": canonical.shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return metadata, canonical.tobytes(order="C")


def _science_digest(result: UVLFRunResult) -> str:
    if type(result) is not UVLFRunResult:
        raise TypeError("result must be exactly UVLFRunResult")
    digest = hashlib.sha256()
    config_json = json.dumps(
        _physical_config_mapping(result.config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _digest_part(digest, "physical_config", config_json)
    for redshift_result in result.redshifts:
        redshift_metadata, redshift_data = _canonical_array_payload(
            np.asarray([redshift_result.redshift], dtype=np.float64)
        )
        _digest_part(digest, "redshift.metadata", redshift_metadata)
        _digest_part(digest, "redshift.data", redshift_data)
        for mode_result in redshift_result.imf_modes:
            _digest_part(digest, "imf_mode", mode_result.imf_mode.encode("utf-8"))
            if mode_result.halo_tracks:
                raise ValueError("benchmark results must not retain halo_tracks")
            for field in fields(mode_result):
                value = getattr(mode_result, field.name)
                if not isinstance(value, np.ndarray):
                    continue
                metadata, data = _canonical_array_payload(value)
                _digest_part(digest, f"result.{field.name}.metadata", metadata)
                _digest_part(digest, f"result.{field.name}.data", data)
    diagnostics = []
    for diagnostic in result.diagnostics.mode_runs:
        diagnostics.append(
            {
                field.name: getattr(diagnostic, field.name)
                for field in fields(diagnostic)
                if field.name != "sampling_seconds"
            }
        )
    diagnostic_json = json.dumps(
        diagnostics,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _digest_part(digest, "non_time_diagnostics", diagnostic_json)
    return digest.hexdigest()


def _aggregate_rss_bytes(
    process: object,
    *,
    tolerate_access_denied: bool = False,
) -> int:
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    except psutil.AccessDenied:
        if not tolerate_access_denied:
            raise
        children = []
    processes = [process, *children]
    total = 0
    for candidate in processes:
        try:
            total += int(candidate.memory_info().rss)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if not tolerate_access_denied:
                raise
            continue
    if total < 0:
        raise RuntimeError("aggregate RSS must be non-negative")
    return total


class _PeakRSSSampler:
    def __init__(self, interval_seconds: float) -> None:
        if not np.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("RSS sampling interval must be finite and positive")
        self._interval_seconds = float(interval_seconds)
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0
        self._error: BaseException | None = None

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    def _sample(self, *, tolerate_access_denied: bool = False) -> None:
        self._peak_rss_bytes = max(
            self._peak_rss_bytes,
            _aggregate_rss_bytes(
                self._process,
                tolerate_access_denied=tolerate_access_denied,
            ),
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._interval_seconds):
                self._sample()
        except BaseException as error:
            self._error = error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS sampler is already started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="uvlf-v2-rss-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> int:
        if self._thread is None:
            raise RuntimeError("RSS sampler is not started")
        self._stop_event.set()
        self._thread.join()
        self._sample(tolerate_access_denied=True)
        if self._error is not None:
            raise RuntimeError("RSS sampling thread failed") from self._error
        return self._peak_rss_bytes


def _case_workers(case: str) -> int:
    if case == "serial_disabled":
        return 1
    if case in ("parallel_disabled", "parallel_samples"):
        return 2
    raise ValueError(f"unknown benchmark case: {case}")


def _build_config(
    args: argparse.Namespace,
    *,
    case: str,
    work_directory: Path,
) -> UVLFRunConfig:
    default_population = StellarPopulationConfig()
    canonical_path = default_population.canonical_ssp_path.resolve(strict=True)
    edges = tuple(np.linspace(-24.0, -12.0, args.n_bins + 1))
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id=f"uvlf-v2-benchmark-{case}",
        redshifts=(10.0,),
        base_seed=args.seed,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(backend="mcbride", sampler="mcbride", n_time_steps=args.n_grid),
        star_formation=StarFormationConfig(),
        stellar_population=StellarPopulationConfig(
            imf_modes=("canonical",),
            canonical_ssp_path=canonical_path,
            topheavy_ssp_path=default_population.topheavy_ssp_path,
            topheavy_ssp_template_metallicity_zsun=(
                default_population.topheavy_ssp_template_metallicity_zsun
            ),
            historical_topheavy_redshift_min=(
                default_population.historical_topheavy_redshift_min
            ),
            source_redshift_gate_enabled=False,
            growth_time_threshold_myr=default_population.growth_time_threshold_myr,
            birth_metallicity_topheavy_max_zsun=(
                default_population.birth_metallicity_topheavy_max_zsun
            ),
            enable_popiii=False,
            popiii_ssp_path=default_population.popiii_ssp_path,
        ),
        sampling=SamplingConfig(
            mass_batch_size=args.mass_batch_size,
            n_halo_mass_samples=args.n_mass,
            n_tracks_per_halo_mass=args.n_tracks,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=11.0,
            muv_bin_edges=edges,
            workers=_case_workers(case),
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
            apply_dust=False,
        ),
        output=OutputConfig((work_directory / f"{case}.unused.h5").resolve()),
    )


def _validate_sample_shards(
    config: UVLFRunConfig,
    paths: tuple[Path, ...],
) -> list[dict[str, object]]:
    if not paths:
        raise RuntimeError("sample shard output is empty")
    if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
        raise ValueError("sample shard paths must be absolute Paths")
    expected_paths = tuple(
        path.parent / uvlf_shard_filename(config, redshift, mode)
        for path in paths[:1]
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )
    if paths != expected_paths:
        raise RuntimeError("sample shard paths do not follow configured axis order")
    expected_count = (
        config.sampling.n_halo_mass_samples
        * config.sampling.n_tracks_per_halo_mass
    )
    summaries: list[dict[str, object]] = []
    for path, redshift, mode in zip(
        paths,
        (
            redshift
            for redshift in config.redshifts
            for _ in config.stellar_population.imf_modes
        ),
        (
            mode
            for _ in config.redshifts
            for mode in config.stellar_population.imf_modes
        ),
        strict=True,
    ):
        shard = read_uvlf_shard(path, load_samples=False)
        if shard.key != (redshift, mode) or shard.sample is not None:
            raise RuntimeError("lazy sample shard key/load state mismatch")
        if shard.sample_descriptor is None or shard.sample_descriptor.sample_count != expected_count:
            raise RuntimeError("sample shard count mismatch")
        with h5py.File(path, "r") as handle:
            group = handle["samples"][f"z={redshift:.17g}"][mode]
            mass_dataset = group["mass_index"]
            track_dataset = group["track_index"]
            chunk_size = min(
                65_536,
                mass_dataset.chunks[0],
                track_dataset.chunks[0],
            )
            for start in range(0, expected_count, chunk_size):
                stop = min(start + chunk_size, expected_count)
                indices = np.arange(start, stop, dtype=np.int64)
                expected_mass = indices // config.sampling.n_tracks_per_halo_mass
                expected_track = indices % config.sampling.n_tracks_per_halo_mass
                if not np.array_equal(mass_dataset[start:stop], expected_mass):
                    raise RuntimeError("sample shard mass_index order mismatch")
                if not np.array_equal(track_dataset[start:stop], expected_track):
                    raise RuntimeError("sample shard track_index order mismatch")
        summaries.append(
            {
                "filename": path.name,
                "redshift": redshift,
                "imf_mode": mode,
                "sample_count": expected_count,
                "order_validated": True,
            }
        )
    return summaries


def _run_child_case(args: argparse.Namespace) -> dict[str, object]:
    case = args.child_case
    if case not in CASES:
        raise ValueError("child case must be one of the configured benchmark cases")
    with tempfile.TemporaryDirectory(prefix=f"auroralf-{case}-") as raw_directory:
        work_directory = Path(raw_directory).resolve()
        config = _build_config(args, case=case, work_directory=work_directory)
        sampler = _PeakRSSSampler(args.rss_interval)
        sampler.start()
        started = time.perf_counter()
        try:
            if case == "parallel_samples":
                revision, dirty = _git_state()
                provenance = ArtifactProvenance.for_config(
                    config,
                    code_revision=revision,
                    code_dirty=dirty,
                    seed_namespace="auroralf.benchmark.v1",
                    source_paths=(("canonical_ssp", config.stellar_population.canonical_ssp_path),),
                )
                shard_directory = work_directory / "sample-shards"
                shard_directory.mkdir()
                result, paths = run_uvlf_to_sample_shards(
                    config,
                    provenance,
                    shard_directory,
                )
                sample_shards = _validate_sample_shards(config, paths)
            else:
                result = run_uvlf_streaming(config)
                sample_shards = []
        except BaseException as error:
            try:
                sampler.stop()
            except BaseException as sampler_error:
                error.add_note(f"RSS sampler cleanup also failed: {sampler_error}")
            raise
        wall_seconds = time.perf_counter() - started
        peak_rss_bytes = sampler.stop()
        if peak_rss_bytes <= 0:
            raise RuntimeError("peak RSS must be positive")
        return {
            "schema_name": CASE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "case": case,
            "config": canonical_config_mapping(config),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "science_digest": _science_digest(result),
            "sample_shards": sample_shards,
        }


def _validate_case_payload(
    payload: object,
    expected_case: str,
    *,
    expected_sample_count: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    if type(payload) is not dict:
        raise TypeError("child case payload must be exactly dict")
    expected_keys = {
        "schema_name",
        "schema_version",
        "complete",
        "case",
        "config",
        "wall_seconds",
        "peak_rss_bytes",
        "science_digest",
        "sample_shards",
    }
    if set(payload) != expected_keys:
        raise ValueError("child case payload keys are incomplete or unknown")
    if (
        payload["schema_name"] != CASE_SCHEMA_NAME
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["complete"] is not True
        or payload["case"] != expected_case
    ):
        raise ValueError("child case schema/completion identity mismatch")
    if type(payload["config"]) is not dict:
        raise TypeError("child case config must be exactly dict")
    config = decode_canonical_config_mapping(payload["config"])
    if (
        config.run_id != f"uvlf-v2-benchmark-{expected_case}"
        or config.redshifts != (10.0,)
        or config.base_seed != args.seed
        or config.mah.backend != "mcbride"
        or config.mah.sampler != "mcbride"
        or config.mah.n_time_steps != args.n_grid
        or config.stellar_population.imf_modes != ("canonical",)
        or not config.stellar_population.canonical_ssp_path.is_file()
        or config.sampling.n_halo_mass_samples != args.n_mass
        or config.sampling.n_tracks_per_halo_mass != args.n_tracks
        or len(config.sampling.muv_bin_edges) != args.n_bins + 1
        or config.sampling.mass_batch_size != args.mass_batch_size
        or config.sampling.workers != _case_workers(expected_case)
    ):
        raise ValueError("child case config does not match requested benchmark settings")
    wall = payload["wall_seconds"]
    peak = payload["peak_rss_bytes"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not np.isfinite(wall) or wall < 0:
        raise ValueError("child wall_seconds must be finite and non-negative")
    if type(peak) is not int or peak <= 0:
        raise ValueError("child peak_rss_bytes must be a positive integer")
    digest = payload["science_digest"]
    if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("child science_digest must be lowercase SHA-256")
    shards = payload["sample_shards"]
    if type(shards) is not list:
        raise TypeError("child sample_shards must be exactly list")
    if expected_case == "parallel_samples":
        if len(shards) != 1:
            raise ValueError("parallel_samples must report exactly one canonical shard")
        shard = shards[0]
        expected_shard_keys = {
            "filename",
            "redshift",
            "imf_mode",
            "sample_count",
            "order_validated",
        }
        if type(shard) is not dict or set(shard) != expected_shard_keys:
            raise ValueError("parallel_samples shard summary keys are incomplete or unknown")
        expected_filename = uvlf_shard_filename(config, 10.0, "canonical")
        if (
            shard.get("filename") != expected_filename
            or shard.get("redshift") != 10.0
            or shard.get("imf_mode") != "canonical"
            or shard.get("sample_count") != expected_sample_count
            or shard.get("order_validated") is not True
        ):
            raise ValueError("parallel_samples shard summary does not match config")
    elif shards:
        raise ValueError("disabled sample cases must not report sample shards")
    return payload


def _child_command(
    args: argparse.Namespace,
    *,
    case: str,
    child_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--n-mass",
        str(args.n_mass),
        "--n-tracks",
        str(args.n_tracks),
        "--n-grid",
        str(args.n_grid),
        "--n-bins",
        str(args.n_bins),
        "--mass-batch-size",
        str(args.mass_batch_size),
        "--seed",
        str(args.seed),
        "--rss-interval",
        str(args.rss_interval),
        "--child-timeout-seconds",
        str(args.child_timeout_seconds),
        "--child-case",
        case,
        "--child-output",
        str(child_output),
    ]


def _memory_overhead(cases: list[dict[str, object]]) -> dict[str, object]:
    peaks = {item["case"]: item["peak_rss_bytes"] for item in cases}
    serial = peaks["serial_disabled"]
    parallel = peaks["parallel_disabled"]
    samples = peaks["parallel_samples"]
    return {
        "parallel_disabled_vs_serial_bytes": parallel - serial,
        "parallel_disabled_vs_serial_ratio": parallel / serial,
        "parallel_samples_vs_parallel_disabled_bytes": samples - parallel,
        "parallel_samples_vs_parallel_disabled_ratio": samples / parallel,
    }


def _run_controller(args: argparse.Namespace) -> Path:
    _require_slurm_allocation()
    report_path = _resolve_output_path(args.report)
    if report_path.exists():
        raise FileExistsError(f"benchmark report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    child_environment = _child_environment()
    with tempfile.TemporaryDirectory(
        prefix=".uvlf-v2-benchmark-controller-",
        dir=report_path.parent,
    ) as raw_directory:
        child_directory = Path(raw_directory)
        for case in CASES:
            child_output = child_directory / f"{case}.json"
            command = _child_command(args, case=case, child_output=child_output)
            try:
                completed = subprocess.run(
                    command,
                    cwd=_PROJECT_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.child_timeout_seconds,
                    env=child_environment,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"benchmark child {case} timed out after "
                    f"{args.child_timeout_seconds:g} seconds"
                ) from error
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark child {case} failed with exit {completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )
            child_payload = _validate_case_payload(
                _read_child_json_stable(child_output),
                case,
                expected_sample_count=args.n_mass * args.n_tracks,
                args=args,
            )
            cases.append(
                {
                    **child_payload,
                    "exit_code": completed.returncode,
                    "command": command,
                }
            )
    digests = tuple(item["science_digest"] for item in cases)
    if len(set(digests)) != 1:
        raise RuntimeError(f"benchmark science digest mismatch: {digests}")
    physical_configs = tuple(
        _physical_config_mapping(decode_canonical_config_mapping(item["config"]))
        for item in cases
    )
    if any(mapping != physical_configs[0] for mapping in physical_configs[1:]):
        raise RuntimeError("benchmark child physical configs do not match")
    payload: dict[str, object] = {
        "schema_name": REPORT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_utc": _utc_now(),
        "environment": _environment_metadata(child_environment),
        "benchmark": {
            "redshift": 10.0,
            "imf_modes": ["canonical"],
            "mah_backend": "mcbride",
            "base_seed": args.seed,
            "n_halo_mass_samples": args.n_mass,
            "n_tracks_per_halo_mass": args.n_tracks,
            "n_time_steps": args.n_grid,
            "n_muv_bins": args.n_bins,
            "mass_batch_size": args.mass_batch_size,
            "rss_interval_seconds": args.rss_interval,
            "child_timeout_seconds": args.child_timeout_seconds,
        },
        "cases": cases,
        "digest_equal": True,
        "memory_overhead": _memory_overhead(cases),
    }
    return _write_json_atomic(payload, report_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _require_slurm_allocation()
    child_requested = args.child_case is not None or args.child_output is not None
    if child_requested:
        if args.child_case is None or args.child_output is None:
            raise ValueError("--child-case and --child-output must be provided together")
        child_output = _resolve_output_path(args.child_output)
        payload = _run_child_case(args)
        _write_json_atomic(payload, child_output)
        return
    _run_controller(args)


if __name__ == "__main__":
    main()
