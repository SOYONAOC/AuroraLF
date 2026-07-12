from __future__ import annotations

from pathlib import Path

from auroralf.config import UVLFRunConfig
from auroralf.results import UVLFRunResult
from auroralf.uvlf.runner import run_uvlf_streaming


def _require_input_file(label: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_input_file_or_directory(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"{label} must be an existing file or directory: {path}")


def _validate_run_paths(config: UVLFRunConfig) -> None:
    population = config.stellar_population
    _require_input_file("canonical_ssp_path", population.canonical_ssp_path)
    if len(population.imf_modes) > 1:
        _require_input_file("topheavy_ssp_path", population.topheavy_ssp_path)
    if population.enable_popiii:
        _require_input_file("popiii_ssp_path", population.popiii_ssp_path)
    if config.mah.backend == "tng":
        if config.mah.tng_cache_path is None:
            raise RuntimeError("validated tng MAH config has no tng_cache_path")
        _require_input_file_or_directory("tng_cache_path", config.mah.tng_cache_path)
    if config.mah.backend == "thesan":
        if config.mah.thesan_cache_path is None:
            raise RuntimeError("validated THESAN MAH config has no thesan_cache_path")
        _require_input_file_or_directory(
            "thesan_cache_path",
            config.mah.thesan_cache_path,
        )


def run_uvlf(config: UVLFRunConfig) -> UVLFRunResult:
    """Run the configured UVLF calculation and return an in-memory typed result."""

    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    _validate_run_paths(config)
    return run_uvlf_streaming(config)


__all__ = ["run_uvlf"]
