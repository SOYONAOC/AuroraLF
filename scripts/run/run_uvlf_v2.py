#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf import UVLFRunConfig, run_uvlf
from auroralf.io import (
    ArtifactProvenance,
    SourceChecksum,
    UVLFShard,
    canonical_config_json,
    merge_uvlf_shards,
    read_uvlf_artifact,
    uvlf_shard_filename,
    validate_uvlf_resume_shards,
    write_uvlf_shard_atomic,
)
from auroralf.mah.thesan import preload_thesan_mah_cache
from auroralf.mah.tng import preload_tng_mah_cache


SEED_NAMESPACE = "auroralf.seeding.v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict, resumable AuroraLF v2 UVLF production pipeline."
    )
    parser.add_argument("--config", type=Path, required=True)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _require_slurm_allocation(config: UVLFRunConfig) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError(
            "run_uvlf_v2.py requires a SLURM allocation; use scripts/submit/submit_uvlf_v2.py"
        )
    raw_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    try:
        allocated_cpus = int(raw_cpus)
    except ValueError as exc:
        raise RuntimeError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
    if allocated_cpus <= 0:
        raise RuntimeError("SLURM_CPUS_PER_TASK must be a positive integer")
    if config.sampling.workers > allocated_cpus:
        raise ValueError(
            f"config requests {config.sampling.workers} workers but SLURM allocated "
            f"{allocated_cpus} CPUs per task"
        )


def _git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return revision, dirty


def _source_paths(config: UVLFRunConfig) -> tuple[tuple[str, Path], ...]:
    population = config.stellar_population
    sources: list[tuple[str, Path]] = [
        ("canonical_ssp", population.canonical_ssp_path),
    ]
    if len(population.imf_modes) > 1:
        sources.append(("topheavy_ssp", population.topheavy_ssp_path))
    if population.enable_popiii:
        sources.append(("popiii_ssp", population.popiii_ssp_path))
    cosmology = config.cosmology.to_model()
    if config.mah.backend == "tng":
        if config.mah.tng_cache_path is None:
            raise RuntimeError("validated TNG config has no cache path")
        for index, redshift in enumerate(config.redshifts):
            resolved = preload_tng_mah_cache(
                config.mah.tng_cache_path,
                z_final=redshift,
                cosmology=cosmology,
            )
            sources.append((f"tng_mah_cache_{index}", resolved))
    elif config.mah.backend == "thesan":
        if config.mah.thesan_cache_path is None:
            raise RuntimeError("validated THESAN config has no cache path")
        for index, redshift in enumerate(config.redshifts):
            resolved = preload_thesan_mah_cache(
                config.mah.thesan_cache_path,
                z_final=redshift,
                cosmology=cosmology,
            )
            sources.append((f"thesan_mah_cache_{index}", resolved))
    paths = tuple(path.resolve(strict=True) for _, path in sources)
    if len(set(paths)) != len(paths):
        raise ValueError("production provenance source paths must be unique")
    return tuple((label, path) for (label, _), path in zip(sources, paths, strict=True))


def _build_provenance(config: UVLFRunConfig) -> ArtifactProvenance:
    revision, dirty = _git_state()
    return ArtifactProvenance.for_config(
        config,
        code_revision=revision,
        code_dirty=dirty,
        seed_namespace=SEED_NAMESPACE,
        source_paths=_source_paths(config),
    )


def _shard_directory(config: UVLFRunConfig) -> Path:
    output = config.output.artifact_path
    return output.parent / f".{output.stem}.shards"


def _expected_shard_paths(config: UVLFRunConfig, directory: Path) -> tuple[Path, ...]:
    return tuple(
        directory / uvlf_shard_filename(config, redshift, mode)
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )


def _marker_path(path: Path) -> Path:
    return path.with_name(path.name + ".complete")


def _require_complete_pair_state(path: Path) -> bool:
    artifact_exists = path.exists()
    marker_exists = _marker_path(path).exists()
    if artifact_exists != marker_exists:
        raise ValueError(f"incomplete artifact pair: {path}")
    return artifact_exists


def _provenance_identity_matches(
    actual: ArtifactProvenance,
    expected: ArtifactProvenance,
) -> bool:
    core_matches = all(
        getattr(actual, field) == getattr(expected, field)
        for field in (
            "config_sha256",
            "code_revision",
            "code_dirty",
            "seed_namespace",
        )
    )
    expected_sources = expected.source_checksums
    return (
        core_matches
        and actual.source_checksums[: len(expected_sources)] == expected_sources
    )


def _final_provenance(scientific: ArtifactProvenance) -> ArtifactProvenance:
    raw_path = os.environ.get("AURORALF_EXECUTION_MANIFEST", "").strip()
    if not raw_path:
        return scientific
    execution_path = Path(raw_path).expanduser().resolve(strict=True)
    execution_source = SourceChecksum.from_path("slurm_execution", execution_path)
    return replace(
        scientific,
        source_checksums=(*scientific.source_checksums, execution_source),
    )


def _validate_complete_output(
    config: UVLFRunConfig,
    provenance: ArtifactProvenance,
) -> Path:
    artifact = read_uvlf_artifact(config.output.artifact_path, load_samples=False)
    if canonical_config_json(artifact.result.config) != canonical_config_json(config):
        raise ValueError("existing final artifact config does not match requested config")
    if not _provenance_identity_matches(artifact.provenance, provenance):
        raise ValueError("existing final artifact provenance does not match requested run")
    artifact.provenance.verify_sources()
    return config.output.artifact_path


def run_config(
    config_path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    if type(resume) is not bool or type(overwrite) is not bool:
        raise TypeError("resume and overwrite must be exactly boolean")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    config = UVLFRunConfig.from_toml(config_path)
    _require_slurm_allocation(config)
    provenance = _build_provenance(config)
    output = config.output.artifact_path
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = _shard_directory(config)
    expected_paths = _expected_shard_paths(config, directory)

    final_complete = _require_complete_pair_state(output)
    if resume and final_complete:
        return _validate_complete_output(config, provenance)
    if final_complete and not overwrite:
        raise FileExistsError(f"final artifact already exists: {output}")
    if not resume and not overwrite and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"shard directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    existing_paths: list[Path] = []
    if resume:
        for path in expected_paths:
            if _require_complete_pair_state(path):
                existing_paths.append(path)
        validate_uvlf_resume_shards(config, provenance, tuple(existing_paths))

    existing_set = set(existing_paths)
    paths_to_write = expected_paths if overwrite else tuple(
        path for path in expected_paths if path not in existing_set
    )
    if paths_to_write:
        result = run_uvlf(config)
        diagnostics = {
            (item.redshift, item.imf_mode): item
            for item in result.diagnostics.mode_runs
        }
        for redshift in config.redshifts:
            redshift_result = result.for_redshift(redshift)
            for mode in config.stellar_population.imf_modes:
                path = directory / uvlf_shard_filename(config, redshift, mode)
                if path not in paths_to_write:
                    continue
                shard = UVLFShard(
                    config=config,
                    provenance=provenance,
                    result=redshift_result.for_mode(mode),
                    diagnostic=diagnostics[(redshift, mode)],
                )
                write_uvlf_shard_atomic(shard, path=path, overwrite=overwrite)

    validate_uvlf_resume_shards(config, provenance, expected_paths)
    merged = merge_uvlf_shards(
        expected_paths,
        output_path=output,
        overwrite=overwrite,
        final_provenance=_final_provenance(provenance),
    )
    _validate_complete_output(config, provenance)
    return merged


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    written = run_config(
        args.config.expanduser().resolve(strict=True),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )
    print(f"uvlf_v2_artifact={written}", flush=True)


if __name__ == "__main__":
    main()
