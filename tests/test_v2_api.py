from __future__ import annotations

import ast
from dataclasses import replace
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

import auroralf
import auroralf.api as api
from auroralf import UVLFRunConfig, UVLFRunResult, run_uvlf
from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
)
from auroralf.ssp import DEFAULT_POPIII_UV_SSP_FILE
from auroralf.uvlf.imf import DEFAULT_CANONICAL_SSP_FILE, DEFAULT_MILD_TOPHEAVY_SSP_FILE
from auroralf.uvlf.pipeline import LoadedSSPKernels


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _touch_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (
        tmp_path / "canonical.dat",
        tmp_path / "topheavy.hdf5",
        tmp_path / "popiii.dat",
    )
    for path in paths:
        path.write_bytes(b"test")
    return paths


def _config(
    tmp_path: Path,
    *,
    modes: tuple[str, ...] = ("canonical",),
    redshifts: tuple[float, ...] = (6.0,),
    apply_dust: bool = False,
    enable_popiii: bool = False,
    mah: MAHConfig | None = None,
    mass_batch_size: int = 1,
    n_mass: int = 2,
    n_tracks: int = 2,
    workers: int = 1,
) -> UVLFRunConfig:
    canonical, topheavy, popiii = _touch_inputs(tmp_path)
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="api-test",
        redshifts=redshifts,
        base_seed=987654321,
        cosmology=CosmologyConfig(h0_km_s_mpc=67.4, omega_m=0.315, omega_b=0.049),
        mah=MAHConfig(n_time_steps=8) if mah is None else mah,
        star_formation=StarFormationConfig(
            enable_time_delay=True,
            efficiency_normalization=0.11,
            characteristic_halo_mass_msun=2.0e11,
            low_mass_slope=0.6,
            high_mass_slope=0.7,
            enable_archived_burst_scatter=True,
            burst_scatter_dex=0.2,
            burst_scatter_correlation_timescale_myr=15.0,
            burst_scatter_mass_conserving=True,
            metallicity_source="none",
        ),
        stellar_population=StellarPopulationConfig(
            imf_modes=modes,
            enable_archived_imf_gate=any(mode != "canonical" for mode in modes),
            canonical_ssp_path=canonical.resolve(),
            topheavy_ssp_path=topheavy.resolve(),
            topheavy_ssp_template_metallicity_zsun=0.05,
            historical_topheavy_redshift_min=10.0,
            source_redshift_gate_enabled=False,
            growth_time_threshold_myr=40.0,
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=enable_popiii,
            popiii_ssp_path=popiii.resolve(),
            popiii_efficiency=2.0e-3,
            popiii_pivot_halo_mass_msun=2.0e7,
            popiii_low_mass_slope=0.1,
            popiii_high_mass_slope=-0.1,
            lw_background_j21=0.2,
            popiii_upper_mass_mode="fixed",
            popiii_upper_mass_msun=5.0e7,
        ),
        sampling=SamplingConfig(
            mass_batch_size=mass_batch_size,
            n_halo_mass_samples=n_mass,
            n_tracks_per_halo_mass=n_tracks,
            log10_halo_mass_min_msun=9.1,
            log10_halo_mass_max_msun=10.2,
            muv_bin_edges=(-24.0, -20.0, -16.0),
            workers=workers,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
            apply_dust=apply_dust,
        ),
        output=OutputConfig((tmp_path / "must-not-be-written.h5").resolve()),
    )


def test_run_uvlf_validates_paths_then_delegates_exact_config_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0, 8.0),
    )
    sentinel = object()
    calls: list[UVLFRunConfig] = []

    def fake_runner(received: UVLFRunConfig) -> UVLFRunResult:
        calls.append(received)
        return sentinel  # type: ignore[return-value]

    monkeypatch.setattr(api, "run_uvlf_streaming", fake_runner)

    result = run_uvlf(config)

    assert result is sentinel
    assert calls == [config]
    assert calls[0] is config
    assert not config.output.artifact_path.exists()


def test_run_uvlf_requires_exact_config_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="UVLFRunConfig"):
        run_uvlf(object())  # type: ignore[arg-type]


def test_run_uvlf_parallel_workers_runs_shared_spawn_path(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        workers=2,
        n_mass=2,
        n_tracks=1,
        mass_batch_size=1,
    )
    config = replace(
        config,
        mah=MAHConfig(z_start_max=8.0, n_time_steps=4),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=replace(
            config.stellar_population,
            canonical_ssp_path=(PROJECT_ROOT / DEFAULT_CANONICAL_SSP_FILE).resolve(),
            topheavy_ssp_path=(PROJECT_ROOT / DEFAULT_MILD_TOPHEAVY_SSP_FILE).resolve(),
            popiii_ssp_path=(PROJECT_ROOT / DEFAULT_POPIII_UV_SSP_FILE).resolve(),
        ),
    )

    result = run_uvlf(config)

    assert result.diagnostics.mode_runs[0].sample_count == 2
    assert not config.output.artifact_path.exists()


def test_public_run_uses_mass_chunks_and_prepares_once_per_mass_across_two_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        modes=("canonical", "mah_burst_mild_topheavy"),
        mass_batch_size=2,
        n_mass=5,
        n_tracks=2,
    )
    hmf_chunk_sizes: list[int] = []
    prepare_count = 0
    evaluate_modes: list[str] = []

    kernels = LoadedSSPKernels(
        canonical_age_myr=np.array([1.0, 10.0]),
        canonical_luminosity_per_msun=np.array([1.0e20, 1.0e19]),
        topheavy_age_myr=np.array([1.0, 10.0]),
        topheavy_luminosity_per_msun=np.array([2.0e20, 2.0e19]),
        popiii_age_myr=None,
        popiii_luminosity_per_msun=None,
        canonical_ssp_path=config.stellar_population.canonical_ssp_path,
        topheavy_ssp_path=config.stellar_population.topheavy_ssp_path,
        popiii_ssp_path=config.stellar_population.popiii_ssp_path,
        topheavy_ssp_template_metallicity_zsun=0.05,
    )
    monkeypatch.setattr(runner, "load_ssp_kernels", lambda **kwargs: kernels)

    class FakeInterpolator:
        def evaluate(self, masses: np.ndarray) -> np.ndarray:
            hmf_chunk_sizes.append(masses.size)
            return np.full(masses.shape, 1.0e-10)

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        nonlocal prepare_count
        prepare_count += 1
        n_tracks = int(kwargs["n_tracks"])
        return SimpleNamespace(
            sfr_msun_per_yr_grid=np.full((n_tracks, 2), 0.2),
            popiii_sfr_msun_per_yr_grid=np.zeros((n_tracks, 2)),
        )

    def fake_evaluate(shared: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
        mode = str(kwargs["imf_mode"])
        evaluate_modes.append(mode)
        n_tracks = shared.sfr_msun_per_yr_grid.shape[0]
        return SimpleNamespace(
            uv_luminosity_erg_per_s_hz=np.full(n_tracks, 1.0e28),
            topheavy_source_count=0 if mode == "canonical" else 1,
            starforming_source_count=2,
            popiii_source_count=0,
            active_source_count=2,
        )

    monkeypatch.setattr(
        runner,
        "prepare_reed07_hmf_interpolator",
        lambda **kwargs: FakeInterpolator(),
    )
    monkeypatch.setattr(runner, "prepare_shared_halo_batch", fake_prepare)
    monkeypatch.setattr(runner, "evaluate_shared_halo_batch", fake_evaluate)

    result = run_uvlf(config)

    assert hmf_chunk_sizes == [2, 2, 1]
    assert prepare_count == config.sampling.n_halo_mass_samples
    assert evaluate_modes == list(config.stellar_population.imf_modes) * 5
    for mode in config.stellar_population.imf_modes:
        diagnostic = next(
            item for item in result.diagnostics.mode_runs if item.imf_mode == mode
        )
        assert diagnostic.sample_count == 10
    assert not config.output.artifact_path.exists()


def test_run_uvlf_reports_exact_missing_canonical_ssp_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing = (tmp_path / "missing-canonical.dat").resolve()
    config = replace(
        config,
        stellar_population=replace(config.stellar_population, canonical_ssp_path=missing),
    )

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        run_uvlf(config)


def test_run_uvlf_reports_exact_missing_active_topheavy_ssp_path(tmp_path: Path) -> None:
    config = _config(tmp_path, modes=("canonical", "z10_mild_topheavy"))
    missing = (tmp_path / "missing-topheavy.hdf5").resolve()
    config = replace(
        config,
        stellar_population=replace(config.stellar_population, topheavy_ssp_path=missing),
    )

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        run_uvlf(config)


def test_run_uvlf_reports_exact_missing_active_popiii_ssp_path(tmp_path: Path) -> None:
    config = _config(tmp_path, enable_popiii=True)
    missing = (tmp_path / "missing-popiii.dat").resolve()
    config = replace(
        config,
        stellar_population=replace(config.stellar_population, popiii_ssp_path=missing),
    )

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        run_uvlf(config)


@pytest.mark.parametrize("backend", ["tng", "thesan"])
def test_run_uvlf_reports_exact_missing_active_backend_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    missing = (tmp_path / f"missing-{backend}.h5").resolve()
    if backend == "tng":
        mah = MAHConfig(backend="tng", tng_cache_path=missing, n_time_steps=8)
    else:
        mah = MAHConfig(backend="thesan", thesan_cache_path=missing, n_time_steps=8)
    config = _config(tmp_path, mah=mah)
    calls: list[UVLFRunConfig] = []
    monkeypatch.setattr(api, "run_uvlf_streaming", calls.append)

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        run_uvlf(config)
    assert calls == []


@pytest.mark.parametrize("backend", ["tng", "thesan"])
@pytest.mark.parametrize("cache_kind", ["file", "directory"])
def test_run_uvlf_accepts_active_backend_cache_file_or_directory_and_delegates_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cache_kind: str,
) -> None:
    cache_path = tmp_path / f"{backend}-cache"
    if cache_kind == "file":
        cache_path.write_bytes(b"cache")
    else:
        cache_path.mkdir()
    mah = (
        MAHConfig(backend="tng", tng_cache_path=cache_path.resolve())
        if backend == "tng"
        else MAHConfig(backend="thesan", thesan_cache_path=cache_path.resolve())
    )
    config = _config(tmp_path, mah=mah)
    sentinel = object()
    calls: list[UVLFRunConfig] = []

    def fake_runner(received: UVLFRunConfig) -> UVLFRunResult:
        calls.append(received)
        return sentinel  # type: ignore[return-value]

    monkeypatch.setattr(api, "run_uvlf_streaming", fake_runner)

    assert run_uvlf(config) is sentinel
    assert calls == [config]
    assert calls[0] is config


@pytest.mark.parametrize("backend", ["tng", "thesan"])
def test_run_uvlf_rejects_existing_active_backend_cache_that_is_not_file_or_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    cache_path = tmp_path / f"{backend}-cache.fifo"
    os.mkfifo(cache_path)
    mah = (
        MAHConfig(backend="tng", tng_cache_path=cache_path.resolve())
        if backend == "tng"
        else MAHConfig(backend="thesan", thesan_cache_path=cache_path.resolve())
    )
    config = _config(tmp_path, mah=mah)
    calls: list[UVLFRunConfig] = []
    monkeypatch.setattr(api, "run_uvlf_streaming", calls.append)

    with pytest.raises(
        ValueError,
        match=rf"{backend}_cache_path.*file or directory.*{re.escape(str(cache_path.resolve()))}",
    ):
        run_uvlf(config)
    assert calls == []


def test_run_uvlf_still_requires_ssp_path_to_be_a_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ssp_directory = (tmp_path / "ssp-directory").resolve()
    ssp_directory.mkdir()
    config = replace(
        config,
        stellar_population=replace(
            config.stellar_population,
            canonical_ssp_path=ssp_directory,
        ),
    )

    with pytest.raises(FileNotFoundError, match=re.escape(str(ssp_directory))):
        run_uvlf(config)


@pytest.mark.parametrize("backend", ["tng", "thesan"])
def test_run_uvlf_allows_multiredshift_cache_directory_into_worker_context_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import auroralf.uvlf.runner as runner

    cache_directory = (tmp_path / f"{backend}-cache-directory").resolve()
    cache_directory.mkdir()
    mah = (
        MAHConfig(backend="tng", tng_cache_path=cache_directory)
        if backend == "tng"
        else MAHConfig(backend="thesan", thesan_cache_path=cache_directory)
    )
    config = _config(tmp_path, mah=mah, redshifts=(6.0, 8.0))
    kernels = LoadedSSPKernels(
        canonical_age_myr=np.array([1.0, 10.0]),
        canonical_luminosity_per_msun=np.array([1.0e20, 1.0e19]),
        topheavy_age_myr=np.array([1.0, 10.0]),
        topheavy_luminosity_per_msun=np.array([2.0e20, 2.0e19]),
        popiii_age_myr=None,
        popiii_luminosity_per_msun=None,
        canonical_ssp_path=config.stellar_population.canonical_ssp_path,
        topheavy_ssp_path=config.stellar_population.topheavy_ssp_path,
        popiii_ssp_path=config.stellar_population.popiii_ssp_path,
        topheavy_ssp_template_metallicity_zsun=0.05,
    )
    preload_calls: list[tuple[Path, float]] = []

    def fake_preload(
        cache_path: Path,
        *,
        z_final: float,
        cosmology: object,
    ) -> Path:
        del cosmology
        preload_calls.append((cache_path, z_final))
        return (cache_path / f"cache-z{z_final:g}.hdf5").resolve()

    monkeypatch.setattr(runner, "load_ssp_kernels", lambda **kwargs: kernels)
    monkeypatch.setattr(
        runner,
        "preload_tng_mah_cache" if backend == "tng" else "preload_thesan_mah_cache",
        fake_preload,
    )
    sentinel = object()

    def fake_runner(received: UVLFRunConfig) -> UVLFRunResult:
        context = runner._build_worker_context(received)
        assert context.config is received
        assert context.resolved_simulation_cache_paths == (
            (6.0, (cache_directory / "cache-z6.hdf5").resolve()),
            (8.0, (cache_directory / "cache-z8.hdf5").resolve()),
        )
        return sentinel  # type: ignore[return-value]

    monkeypatch.setattr(api, "run_uvlf_streaming", fake_runner)

    assert run_uvlf(config) is sentinel
    assert preload_calls == [(cache_directory, 6.0), (cache_directory, 8.0)]


def test_package_root_exports_exact_v2_surface() -> None:
    assert auroralf.__all__ == ["UVLFRunConfig", "UVLFRunResult", "run_uvlf"]
    assert auroralf.UVLFRunConfig is UVLFRunConfig
    assert auroralf.UVLFRunResult is UVLFRunResult
    assert auroralf.run_uvlf is run_uvlf
    assert not hasattr(auroralf, "sample_uvlf_from_hmf")


def test_v2_schema_modules_do_not_use_generic_get_reads() -> None:
    for relative_path in ("auroralf/config.py", "auroralf/results.py", "auroralf/api.py"):
        path = PROJECT_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        generic_get_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ]
        assert generic_get_calls == [], f"generic .get schema reads found in {relative_path}"


def test_run_uvlf_minimal_real_canonical_no_dust_prepares_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    canonical = (PROJECT_ROOT / DEFAULT_CANONICAL_SSP_FILE).resolve()
    topheavy = (PROJECT_ROOT / DEFAULT_MILD_TOPHEAVY_SSP_FILE).resolve()
    popiii = (PROJECT_ROOT / DEFAULT_POPIII_UV_SSP_FILE).resolve()
    assert canonical.is_file(), canonical
    config = UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="real-smoke",
        redshifts=(6.0,),
        base_seed=17,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=8),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=StellarPopulationConfig(
            imf_modes=("canonical",),
            canonical_ssp_path=canonical,
            topheavy_ssp_path=topheavy,
            popiii_ssp_path=popiii,
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=False,
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=1,
            n_tracks_per_halo_mass=1,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=9.1,
            muv_bin_edges=(-40.0, 0.0),
            workers=1,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / "real-smoke.h5").resolve()),
    )

    real_prepare = runner.prepare_shared_halo_batch
    prepare_count = 0

    def prepare_spy(**kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal prepare_count
        prepare_count += 1
        return real_prepare(**kwargs)

    monkeypatch.setattr(runner, "prepare_shared_halo_batch", prepare_spy)

    result = run_uvlf(config)

    mode = result.for_redshift(6.0).for_mode("canonical")
    assert prepare_count == 1
    assert mode.raw_counts.shape == (1,)
    assert result.diagnostics.mode_runs[0].sample_count == 1
    assert not config.output.artifact_path.exists()
