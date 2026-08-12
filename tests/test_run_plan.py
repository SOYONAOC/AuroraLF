from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import pickle

import numpy as np
import pytest

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
from auroralf.driver import build_uvlf_run_plan
from auroralf.run_plan import UVLFModuleSwitches, UVLFRunPlan
from auroralf.uvlf.pipeline import LoadedSSPKernels


def _config(tmp_path: Path) -> UVLFRunConfig:
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="run-plan-test",
        redshifts=(6.0, 10.0),
        base_seed=123,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=8),
        star_formation=StarFormationConfig(
            enable_time_delay=True,
            enable_archived_burst_scatter=False,
            burst_scatter_dex=0.0,
            enable_archived_metallicity=False,
            metallicity_source="none",
        ),
        stellar_population=StellarPopulationConfig(
            imf_modes=("canonical",),
            enable_archived_imf_gate=False,
            canonical_ssp_path=(tmp_path / "canonical.dat").resolve(),
            topheavy_ssp_path=(tmp_path / "topheavy.hdf5").resolve(),
            enable_popiii=True,
            popiii_ssp_path=(tmp_path / "popiii.dat").resolve(),
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=2,
            n_tracks_per_halo_mass=3,
            workers=1,
            apply_dust=True,
        ),
        output=OutputConfig((tmp_path / "result.h5").resolve()),
    )


def _kernels(config: UVLFRunConfig) -> LoadedSSPKernels:
    return LoadedSSPKernels(
        canonical_age_myr=np.array([1.0, 10.0]),
        canonical_luminosity_per_msun=np.array([1.0e20, 1.0e19]),
        topheavy_age_myr=None,
        topheavy_luminosity_per_msun=None,
        popiii_age_myr=np.array([1.0, 10.0]),
        popiii_luminosity_per_msun=np.array([2.0e20, 2.0e19]),
        canonical_ssp_path=config.stellar_population.canonical_ssp_path,
        topheavy_ssp_path=config.stellar_population.topheavy_ssp_path,
        popiii_ssp_path=config.stellar_population.popiii_ssp_path,
        topheavy_ssp_template_metallicity_zsun=(
            config.stellar_population.topheavy_ssp_template_metallicity_zsun
        ),
    )


def test_module_switches_are_an_exact_readonly_view_of_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    switches = UVLFModuleSwitches.from_config(config, include_samples=True)

    assert switches.mah_backend == "mcbride"
    assert switches.parameter_sampler == "mcbride"
    assert switches.enable_time_delay is True
    assert switches.enable_popiii is True
    assert switches.imf_modes == ("canonical",)
    assert switches.enable_archived_imf_gate is False
    assert switches.enable_archived_burst_scatter is False
    assert switches.enable_archived_metallicity is False
    assert switches.metallicity_source == "none"
    assert switches.mass_function_model == "hmf_reed07"
    assert switches.apply_dust is True
    assert switches.include_samples is True
    with pytest.raises(FrozenInstanceError):
        switches.enable_popiii = False  # type: ignore[misc]


def test_builder_resolves_parameters_and_loads_ssp_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_ssp_loader(**kwargs: object) -> LoadedSSPKernels:
        calls.append(dict(kwargs))
        return _kernels(config)

    plan = build_uvlf_run_plan(
        config,
        include_samples=True,
        ssp_kernel_loader=fake_ssp_loader,
    )

    assert type(plan) is UVLFRunPlan
    assert plan.config is config
    assert plan.switches.include_samples is True
    assert plan.include_samples is True
    assert plan.resolved_simulation_cache_paths == ()
    assert plan.sfr_parameters == config.star_formation.to_model()
    assert plan.popiii_parameters == config.stellar_population.to_popiii_model()
    assert len(calls) == 1
    assert calls[0]["imf_modes"] == ("canonical",)
    assert calls[0]["enable_popiii"] is True
    restored = pickle.loads(pickle.dumps(plan))
    assert type(restored) is UVLFRunPlan
    assert restored.switches == plan.switches
    assert restored.context_token == plan.context_token


@pytest.mark.parametrize("backend", ["tng", "thesan"])
def test_builder_resolves_active_simulation_cache_for_every_redshift(
    tmp_path: Path,
    backend: str,
) -> None:
    config = _config(tmp_path)
    source = (tmp_path / backend).resolve()
    mah = (
        MAHConfig(backend="tng", tng_cache_path=source, n_time_steps=8)
        if backend == "tng"
        else MAHConfig(backend="thesan", thesan_cache_path=source, n_time_steps=8)
    )
    config = replace(config, mah=mah)
    calls: list[tuple[Path, float]] = []

    def fake_cache_loader(
        path: Path,
        *,
        z_final: float,
        cosmology: object,
    ) -> Path:
        del cosmology
        calls.append((path, z_final))
        return (tmp_path / f"{backend}-z{z_final:g}.hdf5").resolve()

    plan = build_uvlf_run_plan(
        config,
        ssp_kernel_loader=lambda **kwargs: _kernels(config),
        tng_cache_loader=fake_cache_loader,
        thesan_cache_loader=fake_cache_loader,
    )

    assert calls == [(source, 6.0), (source, 10.0)]
    assert plan.resolved_simulation_cache_paths == (
        (6.0, (tmp_path / f"{backend}-z6.hdf5").resolve()),
        (10.0, (tmp_path / f"{backend}-z10.hdf5").resolve()),
    )


def test_run_plan_rejects_switches_that_do_not_match_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = build_uvlf_run_plan(
        config,
        ssp_kernel_loader=lambda **kwargs: _kernels(config),
    )
    wrong = replace(plan.switches, apply_dust=False)
    with pytest.raises(ValueError, match="switches.*config"):
        replace(plan, switches=wrong)
