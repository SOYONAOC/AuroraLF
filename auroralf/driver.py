"""Composition root for AuroraLF UVLF calculations.

``UVLFRunConfig`` is the only user-facing switch source.  The driver resolves
those switches into an immutable :class:`UVLFRunPlan` before the scheduler
starts any halo task.  Scientific modules do not choose their own defaults at
execution time.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import time

from auroralf.config import UVLFRunConfig
from auroralf.mah.thesan import preload_thesan_mah_cache
from auroralf.mah.tng import preload_tng_mah_cache
from auroralf.run_plan import UVLFModuleSwitches, UVLFRunPlan
from auroralf.uvlf.pipeline import LoadedSSPKernels, load_ssp_kernels


def build_uvlf_run_plan(
    config: UVLFRunConfig,
    *,
    include_samples: bool = False,
    ssp_kernel_loader: Callable[..., LoadedSSPKernels] = load_ssp_kernels,
    tng_cache_loader: Callable[..., Path] = preload_tng_mah_cache,
    thesan_cache_loader: Callable[..., Path] = preload_thesan_mah_cache,
) -> UVLFRunPlan:
    """Resolve all configured modules and resources into one frozen run plan."""

    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if type(include_samples) is not bool:
        raise TypeError("include_samples must be exactly bool")
    for name, value in (
        ("ssp_kernel_loader", ssp_kernel_loader),
        ("tng_cache_loader", tng_cache_loader),
        ("thesan_cache_loader", thesan_cache_loader),
    ):
        if not callable(value):
            raise TypeError(f"{name} must be callable")

    switches = UVLFModuleSwitches.from_config(
        config,
        include_samples=include_samples,
    )
    cosmology = config.cosmology.to_model()
    resolved_cache_paths: list[tuple[float, Path]] = []
    if switches.mah_backend == "tng":
        if config.mah.tng_cache_path is None:
            raise RuntimeError("validated TNG config has no cache path")
        for redshift in config.redshifts:
            resolved_cache_paths.append(
                (
                    redshift,
                    tng_cache_loader(
                        config.mah.tng_cache_path,
                        z_final=redshift,
                        cosmology=cosmology,
                    ),
                )
            )
    elif switches.mah_backend == "thesan":
        if config.mah.thesan_cache_path is None:
            raise RuntimeError("validated THESAN config has no cache path")
        for redshift in config.redshifts:
            resolved_cache_paths.append(
                (
                    redshift,
                    thesan_cache_loader(
                        config.mah.thesan_cache_path,
                        z_final=redshift,
                        cosmology=cosmology,
                    ),
                )
            )

    kernels = ssp_kernel_loader(
        ssp_file=config.stellar_population.canonical_ssp_path,
        imf_modes=switches.imf_modes,
        topheavy_ssp_file=config.stellar_population.topheavy_ssp_path,
        topheavy_ssp_metallicity=(
            config.stellar_population.topheavy_ssp_template_metallicity_zsun
        ),
        enable_popiii=switches.enable_popiii,
        popiii_ssp_file=config.stellar_population.popiii_ssp_path,
    )
    return UVLFRunPlan(
        config=config,
        switches=switches,
        cosmology=cosmology,
        kernels=kernels,
        sfr_parameters=config.star_formation.to_model(),
        popiii_parameters=config.stellar_population.to_popiii_model(),
        transition_parameters=config.stellar_population.to_imf_transition_model(),
        mzr_parameters=(
            config.star_formation.mzr.to_model()
            if config.star_formation.mzr is not None
            else None
        ),
        regulator_parameters=(
            config.star_formation.regulator.to_model()
            if config.star_formation.regulator is not None
            else None
        ),
        resolved_simulation_cache_paths=tuple(resolved_cache_paths),
        context_token=f"{os.getpid()}-{time.time_ns()}-{id(config)}",
        initialization_load_count=1,
    )


__all__ = ["UVLFModuleSwitches", "UVLFRunPlan", "build_uvlf_run_plan"]
