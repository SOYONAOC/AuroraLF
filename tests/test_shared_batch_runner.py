from __future__ import annotations

import ast
from concurrent.futures import Future
from dataclasses import fields, replace
import gc
import inspect
from multiprocessing.reduction import ForkingPickler
import os
import pickle
from pathlib import Path
import weakref

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
from auroralf.mah import HaloHistoryResult
from auroralf.seeding import PipelineRandomSeeds
from auroralf.seeding import derive_hmf_mass_seed
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import uv_luminosity_to_muv
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf
from auroralf.uvlf.imf import IMFTransitionParameters
from auroralf.uvlf.imf import DEFAULT_CANONICAL_SSP_FILE, DEFAULT_MILD_TOPHEAVY_SSP_FILE
from auroralf.ssp import DEFAULT_POPIII_UV_SSP_FILE
from auroralf.uvlf.pipeline import (
    HaloModeEvaluation,
    LoadedSSPKernels,
    SharedHaloBatch,
    evaluate_shared_halo_batch,
)
from auroralf.uvlf.runner import run_uvlf_streaming


def _config(
    tmp_path: Path,
    *,
    mass_batch_size: int,
    modes: tuple[str, ...] = ("canonical", "mah_burst_mild_topheavy"),
    redshifts: tuple[float, ...] = (6.0, 8.0),
    n_mass: int = 5,
    n_tracks: int = 2,
    workers: int = 1,
) -> UVLFRunConfig:
    canonical = tmp_path / "canonical.dat"
    topheavy = tmp_path / "topheavy.hdf5"
    popiii = tmp_path / "popiii.dat"
    for path in (canonical, topheavy, popiii):
        path.write_bytes(b"test")
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="shared-runner-test",
        redshifts=redshifts,
        base_seed=12345,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=2),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=StellarPopulationConfig(
            imf_modes=modes,
            enable_archived_imf_gate=any(mode != "canonical" for mode in modes),
            canonical_ssp_path=canonical.resolve(),
            topheavy_ssp_path=topheavy.resolve(),
            popiii_ssp_path=popiii.resolve(),
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=False,
        ),
        sampling=SamplingConfig(
            mass_batch_size=mass_batch_size,
            n_halo_mass_samples=n_mass,
            n_tracks_per_halo_mass=n_tracks,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=10.0,
            muv_bin_edges=(-30.0, -24.0, -20.0, -16.0, -10.0),
            workers=workers,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / "not-written.h5").resolve()),
    )


def _fake_shared(
    *,
    mass_msun: float,
    redshift: float,
    n_tracks: int,
) -> SharedHaloBatch:
    time = np.tile(np.array([0.5, 0.75]), (n_tracks, 1))
    z_grid = np.tile(np.array([10.0, redshift]), (n_tracks, 1))
    mass_grid = np.tile(np.array([0.8 * mass_msun, mass_msun]), (n_tracks, 1))
    rate = np.full_like(mass_grid, 1.0e9)
    sfr = np.tile(np.array([0.1, 0.2]), (n_tracks, 1))
    active = np.ones_like(mass_grid, dtype=bool)
    birth = np.full_like(mass_grid, 0.01)
    return SharedHaloBatch(
        popiii_enabled=False,
        redshift_grid=np.array([10.0, redshift]),
        floor_mass_msun=np.array([0.8 * mass_msun, mass_msun]),
        time_gyr_grid=time,
        redshift_history_grid=z_grid,
        halo_mass_msun_grid=mass_grid,
        dmh_dt_sfr_msun_per_gyr_grid=rate,
        sfr_msun_per_yr_grid=sfr,
        active_grid=active,
        starforming_grid=active.copy(),
        popiii_sfr_msun_per_yr_grid=np.zeros_like(sfr),
        popiii_source_grid=np.zeros_like(active),
        popiii_fstar_grid=np.zeros_like(sfr),
        popiii_duty_cycle_grid=np.zeros_like(sfr),
        popiii_lower_mass_msun_grid=np.full_like(sfr, np.nan),
        popiii_upper_mass_msun_grid=np.full_like(sfr, np.nan),
        burst_sfr_multiplier_grid=np.ones_like(sfr),
        birth_metallicity_zsun_grid=birth,
        gas_metallicity_zsun_grid=None,
        metal_mass_msun_grid=None,
        gas_mass_msun_grid=None,
        metallicity_source="mzr",
        timing_mah_generation_seconds=0.1,
        timing_sfr_and_chemistry_seconds=0.2,
    )


def _fake_kernels(tmp_path: Path) -> LoadedSSPKernels:
    return LoadedSSPKernels(
        canonical_age_myr=np.array([1.0, 10.0]),
        canonical_luminosity_per_msun=np.array([1.0e20, 1.0e19]),
        topheavy_age_myr=np.array([1.0, 10.0]),
        topheavy_luminosity_per_msun=np.array([2.0e20, 2.0e19]),
        popiii_age_myr=None,
        popiii_luminosity_per_msun=None,
        canonical_ssp_path=(tmp_path / "canonical.dat").resolve(),
        topheavy_ssp_path=(tmp_path / "topheavy.hdf5").resolve(),
        popiii_ssp_path=(tmp_path / "popiii.dat").resolve(),
        topheavy_ssp_template_metallicity_zsun=0.05,
    )


def _fake_evaluation(shared: SharedHaloBatch, mode: str) -> HaloModeEvaluation:
    factors = {
        "canonical": 1.0,
        "z10_mild_topheavy": 1.5,
        "mah_burst_mild_topheavy": 2.0,
    }
    factor = factors[mode]
    final_mass = shared.halo_mass_msun_grid[:, -1]
    luminosity = factor * 1.0e19 * final_mass
    topheavy = np.zeros_like(luminosity) if mode == "canonical" else 0.25 * luminosity
    canonical = luminosity - topheavy
    topheavy_flags = np.zeros_like(shared.active_grid)
    if mode != "canonical":
        topheavy_flags[:, -1] = True
    positive = luminosity > 0.0
    top_fraction = np.zeros_like(luminosity)
    top_fraction[positive] = topheavy[positive] / luminosity[positive]
    return HaloModeEvaluation(
        imf_mode=mode,
        uv_luminosity_erg_per_s_hz=luminosity,
        canonical_uv_luminosity_erg_per_s_hz=canonical,
        topheavy_uv_luminosity_erg_per_s_hz=topheavy,
        popiii_uv_luminosity_erg_per_s_hz=np.zeros_like(luminosity),
        topheavy_source_grid=topheavy_flags,
        candidate_topheavy_source_grid=topheavy_flags.copy(),
        topheavy_light_fraction=top_fraction,
        popiii_light_fraction=np.zeros_like(luminosity),
        topheavy_source_count=int(np.count_nonzero(topheavy_flags)),
        starforming_source_count=int(np.count_nonzero(shared.starforming_grid)),
        popiii_source_count=0,
        active_source_count=int(np.count_nonzero(shared.active_grid)),
        uv_convolution_seconds=0.03,
    )


def _install_fake_science(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[tuple[object, ...]],
) -> tuple[list[weakref.ReferenceType[object]], list[int]]:
    import auroralf.uvlf.runner as runner

    references: list[weakref.ReferenceType[object]] = []
    live_counts: list[int] = []

    def fake_load(**kwargs: object) -> LoadedSSPKernels:
        events.append(("load", tuple(kwargs["imf_modes"])))
        return _fake_kernels(tmp_path)

    class FakeInterpolator:
        def __init__(self, redshift: float) -> None:
            self.redshift = redshift

        def evaluate(self, masses: np.ndarray) -> np.ndarray:
            events.append(("hmf", self.redshift, masses.size))
            return np.full(masses.shape, 1.0e-10)

    def fake_hmf_builder(**kwargs: object) -> FakeInterpolator:
        redshift = float(kwargs["z_obs"])
        events.append(("hmf_builder", redshift))
        return FakeInterpolator(redshift)

    def fake_prepare(**kwargs: object) -> SharedHaloBatch:
        gc.collect()
        live_counts.append(sum(reference() is not None for reference in references))
        shared = _fake_shared(
            mass_msun=float(kwargs["Mh_final"]),
            redshift=float(kwargs["z_final"]),
            n_tracks=int(kwargs["n_tracks"]),
        )
        references.extend(
            (
                weakref.ref(shared),
                weakref.ref(shared.sfr_msun_per_yr_grid),
                weakref.ref(shared.popiii_sfr_msun_per_yr_grid),
            )
        )
        seeds = kwargs["random_seeds"]
        assert type(seeds) is PipelineRandomSeeds
        assert seeds == derive_pipeline_random_seeds(
            12345,
            redshift=float(kwargs["z_final"]),
            mass_index=int(kwargs["mass_index"]),
        )
        events.append(("prepare", kwargs["z_final"], kwargs["mass_index"], id(shared.sfr_msun_per_yr_grid), id(shared.birth_metallicity_zsun_grid)))
        return shared

    def fake_evaluate(shared: SharedHaloBatch, **kwargs: object) -> HaloModeEvaluation:
        mode = str(kwargs["imf_mode"])
        events.append(("evaluate", mode, id(shared.sfr_msun_per_yr_grid), id(shared.birth_metallicity_zsun_grid)))
        return _fake_evaluation(shared, mode)

    monkeypatch.setattr(runner, "load_ssp_kernels", fake_load)
    monkeypatch.setattr(runner, "prepare_reed07_hmf_interpolator", fake_hmf_builder)
    monkeypatch.setattr(runner, "prepare_shared_halo_batch", fake_prepare)
    monkeypatch.setattr(runner, "evaluate_shared_halo_batch", fake_evaluate)
    return references, live_counts


def test_streaming_runner_loads_once_prepares_once_per_mass_and_respects_chunk_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, mass_batch_size=2, n_mass=5)
    events: list[tuple[object, ...]] = []
    references, live_counts = _install_fake_science(monkeypatch, tmp_path, events)

    result = run_uvlf_streaming(config)

    assert events[0] == ("load", config.stellar_population.imf_modes)
    assert [event[1] for event in events if event[0] == "hmf_builder"] == [6.0, 8.0]
    assert [event[2] for event in events if event[0] == "hmf"] == [2, 2, 1, 2, 2, 1]
    assert len([event for event in events if event[0] == "prepare"]) == 10
    assert len([event for event in events if event[0] == "evaluate"]) == 20
    for index, event in enumerate(events):
        if event[0] != "prepare":
            continue
        following = events[index + 1 : index + 3]
        assert [item[0] for item in following] == ["evaluate", "evaluate"]
        assert [item[1] for item in following] == list(config.stellar_population.imf_modes)
        assert all(item[2:] == event[3:] for item in following)
    assert max(live_counts, default=0) == 0
    gc.collect()
    assert sum(reference() is not None for reference in references) == 0
    assert len(result.redshifts) == 2
    assert not config.output.artifact_path.exists()


def test_streaming_runner_batch_partitions_and_variant_order_are_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes_a = ("canonical", "z10_mild_topheavy", "mah_burst_mild_topheavy")
    modes_b = ("canonical", "mah_burst_mild_topheavy", "z10_mild_topheavy")
    outputs = []
    for batch_size, modes in ((1, modes_a), (2, modes_a), (5, modes_a), (2, modes_b)):
        events: list[tuple[object, ...]] = []
        _install_fake_science(monkeypatch, tmp_path, events)
        outputs.append(
            run_uvlf_streaming(
                _config(
                    tmp_path,
                    mass_batch_size=batch_size,
                    modes=modes,
                    redshifts=(6.0,),
                )
            )
        )

    reference = outputs[0].for_redshift(6.0)
    for output in outputs[1:]:
        redshift = output.for_redshift(6.0)
        for mode in modes_a:
            expected = reference.for_mode(mode)
            actual = redshift.for_mode(mode)
            np.testing.assert_array_equal(actual.raw_counts, expected.raw_counts)
            np.testing.assert_allclose(
                actual.weighted_counts_per_mpc3,
                expected.weighted_counts_per_mpc3,
                rtol=2e-14,
                atol=0.0,
            )
            np.testing.assert_allclose(
                actual.weight_squared_counts_per_mpc6,
                expected.weight_squared_counts_per_mpc6,
                rtol=2e-14,
                atol=0.0,
            )
            expected_diag = next(
                item for item in outputs[0].diagnostics.mode_runs if item.imf_mode == mode
            )
            actual_diag = next(
                item for item in output.diagnostics.mode_runs if item.imf_mode == mode
            )
            assert actual_diag.sample_count == expected_diag.sample_count
            assert actual_diag.valid_sample_count == expected_diag.valid_sample_count
            assert actual_diag.topheavy_source_fraction == expected_diag.topheavy_source_fraction
            assert actual_diag.sfrd_msun_per_yr_per_mpc3 == pytest.approx(
                expected_diag.sfrd_msun_per_yr_per_mpc3,
                rel=2e-14,
            )


def test_streaming_runner_caps_workers_to_nmass_when_only_one_mass_is_requested(
    tmp_path: Path,
) -> None:
    config = _real_config(tmp_path, modes=("canonical",), n_tracks=1)
    config = replace(config, sampling=replace(config.sampling, workers=2))

    result = run_uvlf_streaming(config)

    assert result.diagnostics.mode_runs[0].sample_count == 1


def test_streaming_runner_rejects_string_hmf_values_before_float_cast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=2,
        modes=("canonical",),
        redshifts=(6.0,),
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)
    class StringInterpolator:
        def evaluate(self, masses: np.ndarray) -> np.ndarray:
            return np.full(masses.shape, "1e-10")

    monkeypatch.setattr(
        runner,
        "prepare_reed07_hmf_interpolator",
        lambda **kwargs: StringInterpolator(),
    )

    with pytest.raises(TypeError, match="HMF dndm.*real non-boolean"):
        run_uvlf_streaming(config)


def test_streaming_runner_rejects_string_dust_values_before_float_cast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=2,
        modes=("canonical",),
        redshifts=(6.0,),
    )
    config = replace(
        config,
        sampling=replace(config.sampling, apply_dust=True),
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        runner,
        "compute_dust_attenuated_uvlf",
        lambda **kwargs: {"phi_obs": np.full(kwargs["muv_obs"].shape, "1e-4")},
    )

    with pytest.raises(TypeError, match=r"dust\['phi_obs'\].*real non-boolean"):
        run_uvlf_streaming(config)


def test_streaming_runner_requires_dust_phi_obs_mapping_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=2,
        modes=("canonical",),
        redshifts=(6.0,),
    )
    config = replace(
        config,
        sampling=replace(config.sampling, apply_dust=True),
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)
    monkeypatch.setattr(runner, "compute_dust_attenuated_uvlf", lambda **kwargs: {})

    with pytest.raises(KeyError, match="phi_obs"):
        run_uvlf_streaming(config)


def test_streaming_runner_source_has_no_mapping_get_or_product_sample_allocation() -> None:
    import auroralf.uvlf.runner as runner

    tree = ast.parse(inspect.getsource(runner))
    mapping_get_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert mapping_get_calls == []
    assert not any(
        isinstance(node, ast.Name) and node.id == "classify_halo_stellar_channels"
        for node in ast.walk(tree)
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"empty", "zeros", "ones", "full"} or not node.args:
            continue
        assert not any(isinstance(child, ast.Mult) for child in ast.walk(node.args[0]))


def test_streaming_runner_fake_numeric_result_matches_direct_histogram_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        mass_batch_size=2,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=5,
        n_tracks=2,
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)

    result = run_uvlf_streaming(config)

    rng = np.random.default_rng(derive_hmf_mass_seed(config.base_seed, 6.0))
    log_mass = rng.uniform(9.0, 10.0, size=5)
    mass = np.power(10.0, log_mass)
    mass_weight = (10.0 - 9.0) * mass * np.log(10.0) * 1.0e-10 / 5.0
    edges = np.asarray(config.sampling.muv_bin_edges)
    for mode, factor in (("canonical", 1.0), ("mah_burst_mild_topheavy", 2.0)):
        luminosity = factor * 1.0e19 * mass
        muv = np.repeat(np.asarray(uv_luminosity_to_muv(luminosity)), 2)
        weights = np.repeat(mass_weight / 2.0, 2)
        raw, _ = np.histogram(muv, bins=edges)
        weighted, _ = np.histogram(muv, bins=edges, weights=weights)
        squared, _ = np.histogram(muv, bins=edges, weights=np.square(weights))
        sigma = np.sqrt(squared)
        effective = np.divide(
            np.square(weighted),
            squared,
            out=np.zeros_like(weighted),
            where=squared > 0.0,
        )
        width = np.diff(edges)
        actual = result.for_redshift(6.0).for_mode(mode)
        np.testing.assert_array_equal(actual.raw_counts, raw)
        np.testing.assert_allclose(actual.weighted_counts_per_mpc3, weighted, rtol=2e-14)
        np.testing.assert_allclose(actual.weight_squared_counts_per_mpc6, squared, rtol=2e-14)
        np.testing.assert_allclose(actual.weighted_count_sigma_per_mpc3, sigma, rtol=2e-14)
        np.testing.assert_allclose(actual.effective_counts, effective, rtol=3e-14)
        np.testing.assert_allclose(actual.phi_intrinsic_per_mpc3_per_mag, weighted / width, rtol=2e-14)
        np.testing.assert_allclose(actual.phi_intrinsic_sigma_per_mpc3_per_mag, sigma / width, rtol=2e-14)
        diagnostic = next(item for item in result.diagnostics.mode_runs if item.imf_mode == mode)
        assert diagnostic.sample_count == 10
        assert diagnostic.valid_sample_count == 10
        assert diagnostic.topheavy_source_fraction == (0.0 if mode == "canonical" else 0.5)
        assert diagnostic.popiii_source_fraction == 0.0
        assert diagnostic.sfrd_msun_per_yr_per_mpc3 == pytest.approx(
            float(np.sum(0.2 * mass_weight)),
            rel=2e-14,
        )


def test_real_shared_evaluator_does_not_mutate_mode_independent_arrays(tmp_path: Path) -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    kernels = _fake_kernels(tmp_path)
    sfr_identity = id(shared.sfr_msun_per_yr_grid)
    birth_identity = id(shared.birth_metallicity_zsun_grid)
    sfr_before = shared.sfr_msun_per_yr_grid.tobytes()
    birth_before = shared.birth_metallicity_zsun_grid.tobytes()  # type: ignore[union-attr]

    canonical = evaluate_shared_halo_batch(
        shared,
        imf_mode="canonical",
        transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        kernels=kernels,
        ssp_lookback_max_myr=100.0,
    )
    variant = evaluate_shared_halo_batch(
        shared,
        imf_mode="mah_burst_mild_topheavy",
        transition_parameters=IMFTransitionParameters(
            growth_time_threshold_myr=2_000.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
        kernels=kernels,
        ssp_lookback_max_myr=100.0,
    )

    assert canonical.uv_luminosity_erg_per_s_hz.shape == (2,)
    assert variant.uv_luminosity_erg_per_s_hz.shape == (2,)
    assert id(shared.sfr_msun_per_yr_grid) == sfr_identity
    assert id(shared.birth_metallicity_zsun_grid) == birth_identity
    assert shared.sfr_msun_per_yr_grid.tobytes() == sfr_before
    assert shared.birth_metallicity_zsun_grid.tobytes() == birth_before  # type: ignore[union-attr]


def test_shared_pipeline_dataclasses_reject_string_numeric_arrays_before_cast(
    tmp_path: Path,
) -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    kernels = _fake_kernels(tmp_path)
    evaluation = _fake_evaluation(shared, "canonical")

    with pytest.raises(TypeError, match="sfr_msun_per_yr_grid.*real non-boolean"):
        replace(
            shared,
            sfr_msun_per_yr_grid=np.full(shared.sfr_msun_per_yr_grid.shape, "0.2"),
        )
    with pytest.raises(TypeError, match="canonical_age_myr.*real non-boolean"):
        replace(kernels, canonical_age_myr=np.array(["1.0", "10.0"]))
    with pytest.raises(
        TypeError,
        match="uv_luminosity_erg_per_s_hz.*real non-boolean",
    ):
        replace(
            evaluation,
            uv_luminosity_erg_per_s_hz=np.full(
                evaluation.uv_luminosity_erg_per_s_hz.shape,
                "1e29",
            ),
        )
    with pytest.raises(TypeError, match="topheavy_source_count.*integer non-boolean"):
        replace(evaluation, topheavy_source_count=True)


def _assert_irreversibly_readonly(array: np.ndarray) -> None:
    current: object = array
    while isinstance(current, np.ndarray):
        assert current.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            current.setflags(write=True)
        current = current.base


def test_shared_batch_default_is_defensive_and_internal_view_policy_avoids_copies() -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    assert not hasattr(shared, "histories")
    assert not hasattr(shared, "sfr_tracks")
    arrays = [
        getattr(shared, field.name)
        for field in fields(shared)
        if isinstance(getattr(shared, field.name), np.ndarray)
    ]
    assert arrays
    for array in arrays:
        _assert_irreversibly_readonly(array)

    source_sfr = np.array(shared.sfr_msun_per_yr_grid, copy=True)
    source_active = np.array(shared.active_grid, copy=True)
    source_birth = np.array(shared.birth_metallicity_zsun_grid, copy=True)
    viewed_shared = replace(
        shared,
        sfr_msun_per_yr_grid=source_sfr,
        active_grid=source_active,
        birth_metallicity_zsun_grid=source_birth,
        _array_policy="view",
    )
    for source, readonly in (
        (source_sfr, viewed_shared.sfr_msun_per_yr_grid),
        (source_active, viewed_shared.active_grid),
        (source_birth, viewed_shared.birth_metallicity_zsun_grid),
    ):
        assert source.flags.writeable is True
        assert readonly.flags.writeable is False
        assert np.shares_memory(source, readonly)
        with pytest.raises(ValueError, match="read-only"):
            readonly.flat[0] = readonly.flat[0]

    defensive_source = np.array(shared.sfr_msun_per_yr_grid, copy=True)
    defensive_shared = replace(shared, sfr_msun_per_yr_grid=defensive_source)
    defensive_source[...] = 99.0
    assert not np.shares_memory(
        defensive_source,
        defensive_shared.sfr_msun_per_yr_grid,
    )
    assert not np.any(defensive_shared.sfr_msun_per_yr_grid == 99.0)

    with pytest.raises(TypeError, match="_array_policy.*str"):
        replace(shared, _array_policy=True)
    with pytest.raises(ValueError, match="_array_policy.*copy.*view.*mutable"):
        replace(shared, _array_policy="unknown")


def test_kernel_and_evaluation_arrays_remain_irreversibly_immutable_and_defensive(
    tmp_path: Path,
) -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    kernels = _fake_kernels(tmp_path)
    evaluation = _fake_evaluation(shared, "mah_burst_mild_topheavy")

    for instance in (kernels, evaluation):
        arrays = [
            getattr(instance, field.name)
            for field in fields(instance)
            if isinstance(getattr(instance, field.name), np.ndarray)
        ]
        assert arrays
        for array in arrays:
            _assert_irreversibly_readonly(array)

    source_age = np.array(kernels.canonical_age_myr, copy=True)
    defensive_kernels = replace(kernels, canonical_age_myr=source_age)
    source_age[...] = 99.0
    assert not np.any(defensive_kernels.canonical_age_myr == 99.0)

    source_total = np.array(evaluation.uv_luminosity_erg_per_s_hz, copy=True)
    defensive_evaluation = replace(
        evaluation,
        uv_luminosity_erg_per_s_hz=source_total,
    )
    source_total[...] = 99.0
    assert not np.any(defensive_evaluation.uv_luminosity_erg_per_s_hz == 99.0)


def _popiii_enabled_shared() -> SharedHaloBatch:
    base = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    popiii_sfr = np.zeros_like(base.popiii_sfr_msun_per_yr_grid)
    popiii_source = np.zeros_like(base.popiii_source_grid)
    popiii_fstar = np.zeros_like(base.popiii_fstar_grid)
    popiii_duty = np.zeros_like(base.popiii_duty_cycle_grid)
    popiii_lower = np.full_like(base.popiii_lower_mass_msun_grid, np.nan)
    popiii_upper = np.full_like(base.popiii_upper_mass_msun_grid, np.nan)
    popiii_sfr[0, -1] = 0.01
    popiii_source[0, -1] = True
    popiii_fstar[0, -1] = 0.1
    popiii_duty[0, -1] = 0.5
    popiii_lower[0, -1] = 0.5 * base.halo_mass_msun_grid[0, -1]
    popiii_upper[0, -1] = 1.5 * base.halo_mass_msun_grid[0, -1]
    return replace(
        base,
        popiii_enabled=True,
        popiii_sfr_msun_per_yr_grid=popiii_sfr,
        popiii_source_grid=popiii_source,
        popiii_fstar_grid=popiii_fstar,
        popiii_duty_cycle_grid=popiii_duty,
        popiii_lower_mass_msun_grid=popiii_lower,
        popiii_upper_mass_msun_grid=popiii_upper,
    )


def test_shared_batch_enforces_popiii_enabled_and_physical_source_invariants() -> None:
    enabled = _popiii_enabled_shared()
    source = enabled.popiii_source_grid
    np.testing.assert_array_equal(
        source,
        enabled.active_grid & (enabled.popiii_sfr_msun_per_yr_grid > 0.0),
    )

    with pytest.raises(TypeError, match="popiii_enabled.*bool"):
        replace(enabled, popiii_enabled=1)
    with pytest.raises(ValueError, match="popiii_source_grid.*exactly"):
        replace(enabled, popiii_source_grid=np.zeros_like(source))
    with pytest.raises(ValueError, match=r"duty.*\[0, 1\]"):
        replace(enabled, popiii_duty_cycle_grid=np.full(source.shape, 1.1))
    with pytest.raises(ValueError, match=r"fstar.*\[0, 1\]"):
        replace(enabled, popiii_fstar_grid=np.full(source.shape, 1.1))
    with pytest.raises(ValueError, match="source.*positive duty"):
        duty = np.array(enabled.popiii_duty_cycle_grid, copy=True)
        duty[source] = 0.0
        replace(enabled, popiii_duty_cycle_grid=duty)
    with pytest.raises(ValueError, match="source.*positive fstar"):
        fstar = np.array(enabled.popiii_fstar_grid, copy=True)
        fstar[source] = 0.0
        replace(enabled, popiii_fstar_grid=fstar)
    upper = np.array(enabled.popiii_upper_mass_msun_grid, copy=True)
    upper[source] = 0.5 * enabled.halo_mass_msun_grid[source]
    soft_suppression = replace(
        enabled,
        popiii_upper_mass_msun_grid=upper,
    )
    np.testing.assert_array_equal(
        soft_suppression.popiii_upper_mass_msun_grid[source],
        upper[source],
    )
    with pytest.raises(ValueError, match="source.*mass scales.*positive"):
        nonpositive_upper = np.array(enabled.popiii_upper_mass_msun_grid, copy=True)
        nonpositive_upper[source] = 0.0
        replace(
            enabled,
            popiii_upper_mass_msun_grid=nonpositive_upper,
        )

    disabled = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    nonzero = np.array(disabled.popiii_sfr_msun_per_yr_grid, copy=True)
    nonzero[0, -1] = 0.1
    disabled_source = disabled.active_grid & (nonzero > 0.0)
    with pytest.raises(ValueError, match="disabled.*SFR"):
        replace(
            disabled,
            popiii_sfr_msun_per_yr_grid=nonzero,
            popiii_source_grid=disabled_source,
        )
    finite_bounds = np.zeros_like(disabled.popiii_lower_mass_msun_grid)
    with pytest.raises(ValueError, match="disabled.*bounds.*NaN"):
        replace(
            disabled,
            popiii_lower_mass_msun_grid=finite_bounds,
            popiii_upper_mass_msun_grid=finite_bounds,
        )


def test_halo_mode_evaluation_enforces_component_fraction_mask_and_count_invariants() -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=2)
    canonical = _fake_evaluation(shared, "canonical")
    variant = _fake_evaluation(shared, "mah_burst_mild_topheavy")

    bad_total = np.array(variant.uv_luminosity_erg_per_s_hz, copy=True)
    bad_total[0] = np.nextafter(bad_total[0], np.inf)
    with pytest.raises(ValueError, match="sum of canonical.*top-heavy.*Pop III"):
        replace(variant, uv_luminosity_erg_per_s_hz=bad_total)

    bad_fraction = np.array(variant.topheavy_light_fraction, copy=True)
    bad_fraction[0] = np.nextafter(bad_fraction[0], np.inf)
    with pytest.raises(ValueError, match="topheavy_light_fraction.*exact"):
        replace(variant, topheavy_light_fraction=bad_fraction)

    candidate = np.zeros_like(variant.candidate_topheavy_source_grid)
    with pytest.raises(ValueError, match="subset.*candidate"):
        replace(variant, candidate_topheavy_source_grid=candidate)
    with pytest.raises(ValueError, match="topheavy_source_count.*mask"):
        replace(variant, topheavy_source_count=0)
    with pytest.raises(ValueError, match="candidate source count.*starforming"):
        replace(variant, starforming_source_count=0)
    with pytest.raises(ValueError, match="popiii_source_count.*active"):
        replace(variant, popiii_source_count=variant.active_source_count + 1)
    with pytest.raises(ValueError, match="starforming.*active"):
        replace(variant, active_source_count=variant.starforming_source_count - 1)

    top_component = np.array(canonical.topheavy_uv_luminosity_erg_per_s_hz, copy=True)
    top_component[0] = 1.0
    total = (
        canonical.canonical_uv_luminosity_erg_per_s_hz
        + top_component
        + canonical.popiii_uv_luminosity_erg_per_s_hz
    )
    fraction = top_component / total
    with pytest.raises(ValueError, match="canonical.*top-heavy"):
        replace(
            canonical,
            uv_luminosity_erg_per_s_hz=total,
            topheavy_uv_luminosity_erg_per_s_hz=top_component,
            topheavy_light_fraction=fraction,
        )
    canonical_candidate = np.array(
        canonical.candidate_topheavy_source_grid,
        copy=True,
    )
    canonical_candidate[0, -1] = True
    with pytest.raises(ValueError, match="canonical.*top-heavy"):
        replace(canonical, candidate_topheavy_source_grid=canonical_candidate)


def test_halo_mode_evaluation_counts_cannot_exceed_grid_cells_or_candidates() -> None:
    shared = _fake_shared(mass_msun=1.0e10, redshift=6.0, n_tracks=1)
    canonical = _fake_evaluation(shared, "canonical")
    variant = _fake_evaluation(shared, "mah_burst_mild_topheavy")
    assert variant.topheavy_source_grid.size == 2

    with pytest.raises(
        ValueError,
        match="active_source_count.*grid cell count.*2",
    ):
        replace(
            canonical,
            active_source_count=52,
            starforming_source_count=102,
        )

    candidate = np.ones_like(variant.candidate_topheavy_source_grid)
    with pytest.raises(
        ValueError,
        match="candidate source count.*2.*starforming_source_count.*1",
    ):
        replace(
            variant,
            candidate_topheavy_source_grid=candidate,
            starforming_source_count=1,
        )


def _real_config(
    tmp_path: Path,
    *,
    modes: tuple[str, ...],
    n_tracks: int,
    n_mass: int = 1,
    mass_batch_size: int = 1,
) -> UVLFRunConfig:
    project_root = Path(__file__).resolve().parents[1]
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="shared-runner-real",
        redshifts=(6.0,),
        base_seed=77,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(z_start_max=8.0, n_time_steps=4),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=StellarPopulationConfig(
            imf_modes=modes,
            enable_archived_imf_gate=any(mode != "canonical" for mode in modes),
            canonical_ssp_path=(project_root / DEFAULT_CANONICAL_SSP_FILE).resolve(),
            topheavy_ssp_path=(project_root / DEFAULT_MILD_TOPHEAVY_SSP_FILE).resolve(),
            popiii_ssp_path=(project_root / DEFAULT_POPIII_UV_SSP_FILE).resolve(),
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=False,
        ),
        sampling=SamplingConfig(
            mass_batch_size=mass_batch_size,
            n_halo_mass_samples=n_mass,
            n_tracks_per_halo_mass=n_tracks,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=9.1,
            muv_bin_edges=(-40.0, -20.0, 0.0),
            workers=1,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / "real-not-written.h5").resolve()),
    )


def test_compatibility_wrapper_returns_original_sources_and_independent_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.pipeline as pipeline

    config = _real_config(tmp_path, modes=("canonical",), n_tracks=1)
    captured: dict[str, object] = {}
    real_prepare = pipeline.prepare_shared_halo_batch
    real_evaluate = pipeline.evaluate_shared_halo_batch

    def prepare_spy(**kwargs: object) -> SharedHaloBatch:
        shared = real_prepare(**kwargs)
        sources = kwargs["_mutable_result_sources"]
        assert isinstance(sources, dict)
        captured.update(sources)
        captured["shared"] = shared
        return shared

    def evaluate_spy(*args: object, **kwargs: object) -> HaloModeEvaluation:
        evaluation = real_evaluate(*args, **kwargs)
        captured["evaluation"] = evaluation
        return evaluation

    monkeypatch.setattr(pipeline, "prepare_shared_halo_batch", prepare_spy)
    monkeypatch.setattr(pipeline, "evaluate_shared_halo_batch", evaluate_spy)
    result = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e9,
        cosmology=config.cosmology.to_model(),
        random_seeds=derive_pipeline_random_seeds(
            config.base_seed,
            redshift=6.0,
            mass_index=0,
        ),
        z_start_max=config.mah.z_start_max,
        n_grid=config.mah.n_time_steps,
        ssp_file=config.stellar_population.canonical_ssp_path,
        topheavy_ssp_file=config.stellar_population.topheavy_ssp_path,
        popiii_ssp_file=config.stellar_population.popiii_ssp_path,
        imf_mode="canonical",
        enable_time_delay=False,
        workers=1,
    )

    assert type(result.histories) is HaloHistoryResult
    assert result.histories is captured["histories"]
    assert result.sfr_tracks is captured["sfr_tracks"]
    assert result.metadata["negative_dmhdt_clip_count"] == (
        result.histories.metadata["negative_dmhdt_clip_count"]
    )
    shared = captured["shared"]
    evaluation = captured["evaluation"]
    assert isinstance(shared, SharedHaloBatch)
    assert isinstance(evaluation, HaloModeEvaluation)
    assert shared.sfr_msun_per_yr_grid.flags.writeable is True
    assert np.shares_memory(
        shared.sfr_msun_per_yr_grid,
        result.sfr_tracks["SFR"],
    )
    assert not np.shares_memory(result.redshift_grid, shared.redshift_grid)
    assert not np.shares_memory(result.active_grid, shared.active_grid)
    assert not np.shares_memory(
        result.uv_luminosities,
        evaluation.uv_luminosity_erg_per_s_hz,
    )


def test_real_reduced_canonical_streaming_matches_legacy_sampler(tmp_path: Path) -> None:
    config = _real_config(tmp_path, modes=("canonical",), n_tracks=1)

    streaming = run_uvlf_streaming(config)
    legacy = sample_uvlf_from_hmf(
        z_obs=6.0,
        N_mass=1,
        n_tracks=1,
        cosmology=config.cosmology.to_model(),
        base_seed=config.base_seed,
        quantity="Muv",
        bins=np.asarray(config.sampling.muv_bin_edges),
        logM_min=config.sampling.log10_halo_mass_min_msun,
        logM_max=config.sampling.log10_halo_mass_max_msun,
        z_start_max=config.mah.z_start_max,
        n_grid=config.mah.n_time_steps,
        sampler=config.mah.sampler,
        mah_backend=config.mah.backend,
        enable_time_delay=config.star_formation.enable_time_delay,
        pipeline_workers=1,
        ssp_file=str(config.stellar_population.canonical_ssp_path),
        topheavy_ssp_file=str(config.stellar_population.topheavy_ssp_path),
        topheavy_ssp_metallicity=(
            config.stellar_population.topheavy_ssp_template_metallicity_zsun
        ),
        enable_popiii=False,
        popiii_sfr_parameters=config.stellar_population.to_popiii_model(),
        popiii_ssp_file=str(config.stellar_population.popiii_ssp_path),
        imf_mode="canonical",
        imf_transition_parameters=config.stellar_population.to_imf_transition_model(),
        progress_path=None,
        print_progress=False,
        sfr_model_parameters=config.star_formation.to_model(),
        mass_function_model=config.sampling.mass_function_model,
        hmf_dlog10m=config.sampling.hmf_dlog10m,
        burst_scatter_dex=config.star_formation.burst_scatter_dex,
        burst_scatter_timescale_myr=(
            config.star_formation.burst_scatter_correlation_timescale_myr
        ),
        burst_scatter_preserve_mean=config.star_formation.burst_scatter_mass_conserving,
    )

    actual = streaming.for_redshift(6.0).for_mode("canonical")
    np.testing.assert_array_equal(actual.raw_counts, legacy.uvlf["raw_counts"])
    for actual_values, legacy_key in (
        (actual.weighted_counts_per_mpc3, "weighted_counts"),
        (actual.weight_squared_counts_per_mpc6, "weight_squared_counts"),
        (actual.weighted_count_sigma_per_mpc3, "weighted_count_sigma"),
        (actual.effective_counts, "effective_counts"),
        (actual.phi_intrinsic_per_mpc3_per_mag, "phi"),
        (actual.phi_intrinsic_sigma_per_mpc3_per_mag, "phi_sigma"),
    ):
        np.testing.assert_allclose(actual_values, legacy.uvlf[legacy_key], rtol=2e-13, atol=0.0)
    diagnostic = streaming.diagnostics.mode_runs[0]
    assert diagnostic.sample_count == legacy.samples["Muv"].size
    assert diagnostic.valid_sample_count == int(np.count_nonzero(np.isfinite(legacy.samples["Muv"])))
    assert diagnostic.topheavy_source_fraction == pytest.approx(
        legacy.metadata["topheavy_source_fraction"]
    )
    assert diagnostic.popiii_source_fraction == pytest.approx(
        legacy.metadata["popiii_source_fraction"]
    )
    assert diagnostic.sfrd_msun_per_yr_per_mpc3 == pytest.approx(
        legacy.metadata["sfrd_msun_yr_mpc3"],
        rel=2e-13,
    )


def test_real_two_mode_streaming_prepares_shared_batch_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _real_config(
        tmp_path,
        modes=("canonical", "mah_burst_mild_topheavy"),
        n_tracks=2,
    )
    real_prepare = runner.prepare_shared_halo_batch
    prepare_count = 0

    def prepare_spy(**kwargs: object) -> SharedHaloBatch:
        nonlocal prepare_count
        prepare_count += 1
        return real_prepare(**kwargs)

    monkeypatch.setattr(runner, "prepare_shared_halo_batch", prepare_spy)

    result = run_uvlf_streaming(config)

    assert prepare_count == 1
    redshift = result.for_redshift(6.0)
    assert redshift.for_mode("canonical").raw_counts.shape == (2,)
    assert redshift.for_mode("mah_burst_mild_topheavy").raw_counts.shape == (2,)


def test_real_streaming_histograms_are_bitwise_equal_for_batch_1_2_and_n(
    tmp_path: Path,
) -> None:
    results = [
        run_uvlf_streaming(
            _real_config(
                tmp_path,
                modes=("canonical",),
                n_tracks=1,
                n_mass=3,
                mass_batch_size=batch_size,
            )
        )
        for batch_size in (1, 2, 3)
    ]
    reference = results[0].for_redshift(6.0).for_mode("canonical")
    for result in results[1:]:
        actual = result.for_redshift(6.0).for_mode("canonical")
        for name in (
            "raw_counts",
            "weighted_counts_per_mpc3",
            "weight_squared_counts_per_mpc6",
            "weighted_count_sigma_per_mpc3",
            "effective_counts",
            "phi_intrinsic_per_mpc3_per_mag",
            "phi_intrinsic_sigma_per_mpc3_per_mag",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))


def _assert_run_science_bitwise_equal(actual, expected) -> None:  # type: ignore[no-untyped-def]
    assert tuple(item.redshift for item in actual.redshifts) == tuple(
        item.redshift for item in expected.redshifts
    )
    for actual_redshift, expected_redshift in zip(
        actual.redshifts,
        expected.redshifts,
        strict=True,
    ):
        assert tuple(item.imf_mode for item in actual_redshift.imf_modes) == tuple(
            item.imf_mode for item in expected_redshift.imf_modes
        )
        for actual_mode, expected_mode in zip(
            actual_redshift.imf_modes,
            expected_redshift.imf_modes,
            strict=True,
        ):
            for field in fields(actual_mode):
                actual_value = getattr(actual_mode, field.name)
                expected_value = getattr(expected_mode, field.name)
                if isinstance(actual_value, np.ndarray):
                    np.testing.assert_array_equal(actual_value, expected_value)
                else:
                    assert actual_value == expected_value
    for actual_diag, expected_diag in zip(
        actual.diagnostics.mode_runs,
        expected.diagnostics.mode_runs,
        strict=True,
    ):
        for field in fields(actual_diag):
            if field.name == "sampling_seconds":
                assert np.isfinite(getattr(actual_diag, field.name))
                assert getattr(actual_diag, field.name) >= 0.0
                continue
            assert getattr(actual_diag, field.name) == getattr(expected_diag, field.name)
    assert np.isfinite(actual.diagnostics.total_seconds)
    assert actual.diagnostics.total_seconds >= 0.0


def test_real_spawn_canonical_workers_1_2_3_are_bitwise_equal_and_context_stable(
    tmp_path: Path,
) -> None:
    base = _real_config(
        tmp_path,
        modes=("canonical",),
        n_tracks=1,
        n_mass=4,
        mass_batch_size=2,
    )
    runs = {}
    task_results = {}
    for workers in (1, 2, 3):
        config = replace(base, sampling=replace(base.sampling, workers=workers))
        observed = []
        runs[workers] = run_uvlf_streaming(
            config,
            _mass_result_observer=observed.append,
        )
        task_results[workers] = observed

    _assert_run_science_bitwise_equal(runs[2], runs[1])
    _assert_run_science_bitwise_equal(runs[3], runs[1])
    for workers in (1, 2, 3):
        observed = task_results[workers]
        assert [item.mass_index for item in observed] == [0, 1, 2, 3]
        assert all(item.worker_initialization_load_count == 1 for item in observed)
        tokens_by_pid: dict[int, set[str]] = {}
        for item in observed:
            tokens_by_pid.setdefault(item.worker_pid, set()).add(
                item.worker_context_token
            )
            for mode_result in item.mode_results:
                _assert_array_and_base_chain_are_immutable(
                    mode_result.uv_luminosity_erg_per_s_hz
                )
        assert all(len(tokens) == 1 for tokens in tokens_by_pid.values())
        assert len(tokens_by_pid) <= min(workers, 4)
        if workers > 1:
            assert len(tokens_by_pid) > 1
            assert any(
                sum(item.worker_pid == pid for item in observed) > 1
                for pid in tokens_by_pid
            )


def test_real_spawn_two_mode_workers2_matches_serial_bitwise(tmp_path: Path) -> None:
    base = _real_config(
        tmp_path,
        modes=("canonical", "mah_burst_mild_topheavy"),
        n_tracks=1,
        n_mass=2,
        mass_batch_size=1,
    )
    serial = run_uvlf_streaming(base)
    parallel = run_uvlf_streaming(
        replace(base, sampling=replace(base.sampling, workers=2))
    )

    _assert_run_science_bitwise_equal(parallel, serial)


def _fake_parallel_task_result(runner: object, mass_index: int):  # type: ignore[no-untyped-def]
    mode_result = runner._MassModeTaskResult(  # type: ignore[attr-defined]
        imf_mode="canonical",
        uv_luminosity_erg_per_s_hz=np.array([1.0e28]),
        topheavy_source_count=0,
        starforming_source_count=1,
        popiii_source_count=0,
        active_source_count=1,
        evaluation_seconds=0.01,
    )
    return runner._MassTaskResult(  # type: ignore[attr-defined]
        redshift=6.0,
        mass_index=mass_index,
        halo_mass_msun=1.0e10 + mass_index,
        mass_weight_per_mpc3=1.0e-4,
        final_sfr_mean_msun_per_yr=0.2,
        final_popiii_sfr_mean_msun_per_yr=0.0,
        mode_results=(mode_result,),
        shared_preparation_seconds=0.02,
        worker_pid=1234,
        worker_context_token="fake-context",
        worker_initialization_load_count=1,
    )


def _ipc_roundtrips(value: object):  # type: ignore[no-untyped-def]
    for protocol in (4, 5):
        yield pickle.loads(pickle.dumps(value, protocol=protocol))
        yield pickle.loads(ForkingPickler.dumps(value, protocol=protocol))


def _assert_array_and_base_chain_are_immutable(array: np.ndarray) -> None:
    current: object = array
    while isinstance(current, np.ndarray):
        assert current.flags.writeable is False
        current = current.base
    if isinstance(current, memoryview):
        assert current.readonly
    with pytest.raises(ValueError, match="WRITEABLE|writeable"):
        array.setflags(write=True)


def test_ipc_roundtrip_reconstructs_strict_immutable_mass_results() -> None:
    import auroralf.uvlf.runner as runner

    result = _fake_parallel_task_result(runner, 0)
    mode = result.mode_results[0]
    for restored_mode in _ipc_roundtrips(mode):
        assert type(restored_mode) is runner._MassModeTaskResult
        _assert_array_and_base_chain_are_immutable(
            restored_mode.uv_luminosity_erg_per_s_hz
        )
    object.__setattr__(
        mode,
        "uv_luminosity_erg_per_s_hz",
        np.array([1.0e28], dtype=float),
    )
    for restored_result in _ipc_roundtrips(result):
        assert type(restored_result) is runner._MassTaskResult
        _assert_array_and_base_chain_are_immutable(
            restored_result.mode_results[0].uv_luminosity_erg_per_s_hz
        )
    object.__setattr__(
        mode,
        "uv_luminosity_erg_per_s_hz",
        np.array([-1.0], dtype=float),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(result, mode_results=(mode,))
    spec = runner._MassTaskSpec(
        result.redshift,
        result.mass_index,
        result.halo_mass_msun,
        result.mass_weight_per_mpc3,
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        runner._validate_scheduled_result(spec, result)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pickle.loads(pickle.dumps(result, protocol=5))


def test_ipc_roundtrip_reconstructs_loaded_kernels_and_worker_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        redshifts=(6.0,),
        mass_batch_size=1,
        n_mass=1,
        n_tracks=1,
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)
    runner._clear_worker_context_for_tests()
    runner._initialize_worker(config)
    context = runner._WORKER_CONTEXT
    assert context is not None
    for restored_kernels in _ipc_roundtrips(context.kernels):
        assert type(restored_kernels) is LoadedSSPKernels
        _assert_array_and_base_chain_are_immutable(
            restored_kernels.canonical_age_myr
        )
        _assert_array_and_base_chain_are_immutable(
            restored_kernels.canonical_luminosity_per_msun
        )
    for restored_context in _ipc_roundtrips(context):
        assert type(restored_context) is runner._WorkerContext
        assert restored_context.config == config
        _assert_array_and_base_chain_are_immutable(
            restored_context.kernels.canonical_age_myr
        )
        _assert_array_and_base_chain_are_immutable(
            restored_context.kernels.canonical_luminosity_per_msun
        )
    runner._clear_worker_context_for_tests()


def test_mass_task_and_scheduled_validation_reuse_validated_nested_results() -> None:
    import auroralf.uvlf.runner as runner

    spec = runner._MassTaskSpec(6.0, 0, 1.0e10, 1.0e-4)
    result = _fake_parallel_task_result(runner, 0)
    mode_result = result.mode_results[0]
    luminosity = mode_result.uv_luminosity_erg_per_s_hz
    reconstructed = replace(result, mode_results=(mode_result,))

    validated = runner._validate_scheduled_result(spec, reconstructed)

    assert reconstructed.mode_results[0] is mode_result
    assert validated is reconstructed
    assert validated.mode_results[0] is mode_result
    assert validated.mode_results[0].uv_luminosity_erg_per_s_hz is luminosity
    _assert_array_and_base_chain_are_immutable(luminosity)
    object.__setattr__(reconstructed, "mass_index", True)
    with pytest.raises(TypeError, match="mass_index.*integer non-boolean"):
        runner._validate_scheduled_result(spec, reconstructed)
    with pytest.raises(TypeError, match="mass task must return exactly"):
        runner._validate_scheduled_result(spec, object())
    for field_name, value, message in (
        ("redshift", 7.0, "redshift does not match"),
        ("halo_mass_msun", 2.0e10, "halo mass does not match"),
        ("mass_weight_per_mpc3", 2.0e-4, "mass weight does not match"),
    ):
        mismatched = replace(result, **{field_name: value})
        with pytest.raises(RuntimeError, match=message):
            runner._validate_scheduled_result(spec, mismatched)


def test_serial_observer_receives_immutable_result_before_histogram_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        modes=("canonical",),
        redshifts=(6.0,),
        mass_batch_size=1,
        n_mass=2,
        n_tracks=1,
    )
    events: list[tuple[object, ...]] = []
    _install_fake_science(monkeypatch, tmp_path, events)
    baseline = runner.run_uvlf_streaming(config)
    observed_count = 0

    def observer(result) -> None:  # type: ignore[no-untyped-def]
        nonlocal observed_count
        observed_count += 1
        array = result.mode_results[0].uv_luminosity_erg_per_s_hz
        _assert_array_and_base_chain_are_immutable(array)
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0

    protected = runner.run_uvlf_streaming(
        config,
        _mass_result_observer=observer,
    )

    assert observed_count == config.sampling.n_halo_mass_samples
    _assert_run_science_bitwise_equal(protected, baseline)


def test_mass_task_schema_is_strict_and_task_requires_initialized_worker_context() -> None:
    import auroralf.uvlf.runner as runner

    spec = runner._MassTaskSpec(
        redshift=6.0,
        mass_index=0,
        halo_mass_msun=1.0e10,
        mass_weight_per_mpc3=1.0e-4,
    )
    runner._clear_worker_context_for_tests()

    with pytest.raises(RuntimeError, match="worker context.*not initialized"):
        runner._run_mass_task(spec)
    with pytest.raises(TypeError, match="mass_index.*integer non-boolean"):
        replace(spec, mass_index=True)
    with pytest.raises(ValueError, match="halo_mass_msun.*positive"):
        replace(spec, halo_mass_msun=0.0)


def test_worker_context_builds_once_and_mass_task_uses_only_initialized_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    config = _config(
        tmp_path,
        mass_batch_size=2,
        modes=("canonical", "mah_burst_mild_topheavy"),
        redshifts=(6.0,),
        n_mass=1,
        n_tracks=2,
    )
    events: list[tuple[object, ...]] = []
    references, _ = _install_fake_science(monkeypatch, tmp_path, events)
    runner._clear_worker_context_for_tests()

    runner._initialize_worker(config)
    context = runner._WORKER_CONTEXT
    assert context.config is config
    assert context.initialization_load_count == 1
    assert len([event for event in events if event[0] == "load"]) == 1
    result = runner._run_mass_task(
        runner._MassTaskSpec(
            redshift=6.0,
            mass_index=0,
            halo_mass_msun=1.0e10,
            mass_weight_per_mpc3=1.0e-4,
        )
    )

    assert result.worker_pid == os.getpid()
    assert result.worker_context_token == context.context_token
    assert result.worker_initialization_load_count == 1
    assert tuple(item.imf_mode for item in result.mode_results) == (
        "canonical",
        "mah_burst_mild_topheavy",
    )
    assert len([event for event in events if event[0] == "prepare"]) == 1
    assert len([event for event in events if event[0] == "evaluate"]) == 2
    gc.collect()
    assert sum(reference() is not None for reference in references) == 0
    runner._clear_worker_context_for_tests()

    tree = ast.parse(inspect.getsource(runner._run_mass_task))
    forbidden_names = {
        "load_ssp_kernels",
        "_build_worker_context",
        "preload_tng_mah_cache",
        "preload_thesan_mah_cache",
    }
    assert not any(
        isinstance(node, ast.Name) and node.id in forbidden_names
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "to_model"
        for node in ast.walk(tree)
    )


def test_bounded_parallel_scheduler_handles_slow_first_reverse_completion_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.runner as runner

    produced: list[int] = []

    def specs():  # type: ignore[no-untyped-def]
        for mass_index in range(9):
            produced.append(mass_index)
            yield runner._MassTaskSpec(
                redshift=6.0,
                mass_index=mass_index,
                halo_mass_msun=1.0e10 + mass_index,
                mass_weight_per_mpc3=1.0e-4,
            )

    class FakeExecutor:
        def __init__(self) -> None:
            self.future_by_index: dict[int, Future] = {}
            self.cancelled_indices: list[int] = []

        def submit(self, function, spec):  # type: ignore[no-untyped-def]
            del function
            future: Future = Future()
            self.future_by_index[spec.mass_index] = future
            if spec.mass_index != 0:
                future.set_result(_fake_parallel_task_result(runner, spec.mass_index))
            return future

    executor = FakeExecutor()
    wait_calls = 0

    def fake_wait(futures, *, return_when):  # type: ignore[no-untyped-def]
        nonlocal wait_calls
        assert return_when is runner.FIRST_COMPLETED
        wait_calls += 1
        future_set = set(futures)
        done = {future for future in future_set if future.done()}
        if done:
            return done, future_set - done
        first = executor.future_by_index[0]
        first.set_result(_fake_parallel_task_result(runner, 0))
        return {first}, future_set - {first}

    real_validate = runner._validate_scheduled_result
    validation_calls: list[int] = []

    def validate_once(spec, result):  # type: ignore[no-untyped-def]
        validation_calls.append(spec.mass_index)
        return real_validate(spec, result)

    monkeypatch.setattr(runner, "wait", fake_wait)
    monkeypatch.setattr(runner, "_validate_scheduled_result", validate_once)
    snapshots = []
    ordered = runner._ordered_parallel_results(
        specs(),
        executor=executor,
        max_workers=2,
        scheduling_observer=snapshots.append,
    )

    first = next(ordered)
    assert first.mass_index == 0
    assert produced == [0, 1, 2, 3]
    remaining = list(ordered)

    assert [first.mass_index, *(item.mass_index for item in remaining)] == list(range(9))
    assert max(snapshot.total_occupancy for snapshot in snapshots) <= 4
    assert max(snapshot.running_count for snapshot in snapshots) <= 4
    assert max(snapshot.completed_waiting_count for snapshot in snapshots) <= 4
    assert wait_calls >= 2
    assert sorted(validation_calls) == list(range(9))


def test_bounded_parallel_scheduler_cancels_pending_futures_on_task_exception() -> None:
    import auroralf.uvlf.runner as runner

    class FailingExecutor:
        def __init__(self) -> None:
            self.futures: list[Future] = []

        def submit(self, function, spec):  # type: ignore[no-untyped-def]
            del function
            future: Future = Future()
            self.futures.append(future)
            if spec.mass_index == 0:
                future.set_exception(RuntimeError("task boom"))
            return future

    executor = FailingExecutor()
    specs = (
        runner._MassTaskSpec(6.0, index, 1.0e10 + index, 1.0e-4)
        for index in range(4)
    )

    with pytest.raises(RuntimeError, match="task boom"):
        list(
            runner._ordered_parallel_results(
                specs,
                executor=executor,
                max_workers=2,
            )
        )

    assert len(executor.futures) == 4
    assert all(future.cancelled() for future in executor.futures[1:])


def test_bounded_parallel_scheduler_cancels_submitted_futures_on_generator_exception() -> None:
    import auroralf.uvlf.runner as runner

    class PendingExecutor:
        def __init__(self) -> None:
            self.futures: list[Future] = []

        def submit(self, function, spec):  # type: ignore[no-untyped-def]
            del function, spec
            future: Future = Future()
            self.futures.append(future)
            return future

    def failing_specs():  # type: ignore[no-untyped-def]
        yield runner._MassTaskSpec(6.0, 0, 1.0e10, 1.0e-4)
        yield runner._MassTaskSpec(6.0, 1, 1.1e10, 1.0e-4)
        raise RuntimeError("generator boom")

    executor = PendingExecutor()
    with pytest.raises(RuntimeError, match="generator boom"):
        list(
            runner._ordered_parallel_results(
                failing_specs(),
                executor=executor,
                max_workers=2,
            )
        )

    assert len(executor.futures) == 2
    assert all(future.cancelled() for future in executor.futures)


def test_bounded_parallel_scheduler_rejects_mismatched_result_index() -> None:
    import auroralf.uvlf.runner as runner

    class MismatchedExecutor:
        def submit(self, function, spec):  # type: ignore[no-untyped-def]
            del function
            future: Future = Future()
            future.set_result(
                _fake_parallel_task_result(runner, spec.mass_index + 1)
            )
            return future

    specs = (
        runner._MassTaskSpec(6.0, index, 1.0e10 + index, 1.0e-4)
        for index in range(2)
    )
    with pytest.raises(RuntimeError, match="result index.*does not match"):
        list(
            runner._ordered_parallel_results(
                specs,
                executor=MismatchedExecutor(),
                max_workers=1,
            )
        )


def test_bounded_parallel_scheduler_rejects_noncontiguous_task_indices() -> None:
    import auroralf.uvlf.runner as runner

    class PendingExecutor:
        def __init__(self) -> None:
            self.futures: list[Future] = []

        def submit(self, function, spec):  # type: ignore[no-untyped-def]
            del function, spec
            future: Future = Future()
            self.futures.append(future)
            return future

    specs = (
        runner._MassTaskSpec(6.0, index, 1.0e10 + index, 1.0e-4)
        for index in (0, 2)
    )
    executor = PendingExecutor()
    with pytest.raises(RuntimeError, match="indices must be contiguous"):
        list(
            runner._ordered_parallel_results(
                specs,
                executor=executor,
                max_workers=1,
            )
        )
    assert len(executor.futures) == 1
    assert executor.futures[0].cancelled()
