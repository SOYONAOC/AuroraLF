from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
import multiprocessing as mp
import os
import time

import numpy as np

from auroralf.config import UVLFRunConfig
from auroralf.driver import build_uvlf_run_plan
from auroralf.mah.thesan import preload_thesan_mah_cache
from auroralf.mah.tng import preload_tng_mah_cache
from auroralf.run_plan import UVLFModuleSwitches, UVLFRunPlan
from auroralf.samples import HaloSampleTable
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)
from auroralf.seeding import derive_hmf_mass_seed, derive_pipeline_random_seeds
from .accumulation import (
    _ModeAccumulatorState,
    _consume_mass_task_result,
    _observe_halo_samples,
    _observed_uvlf as _accumulate_observed_uvlf,
    _strict_nonnegative_float_1d,
)
from .dust import compute_dust_attenuated_uvlf
from .hmf_sampling import prepare_reed07_hmf_interpolator
from .pipeline import (
    evaluate_shared_halo_batch,
    load_ssp_kernels,
    prepare_shared_halo_batch,
)
from .streaming import WeightedHistogramAccumulator


def _observed_uvlf(
    *,
    centers: np.ndarray,
    intrinsic_phi: np.ndarray,
    intrinsic_sigma: np.ndarray,
    redshift: float,
    apply_dust: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility boundary that injects the configured dust transform."""

    return _accumulate_observed_uvlf(
        centers=centers,
        intrinsic_phi=intrinsic_phi,
        intrinsic_sigma=intrinsic_sigma,
        redshift=redshift,
        apply_dust=apply_dust,
        dust_transform=compute_dust_attenuated_uvlf,
    )


from .runner_models import (
    _strict_float_scalar,
    _strict_int_scalar,
    _MassTaskSpec,
    _MassModeTaskResult,
    _validate_mass_mode_task_result_integrity,
    _MassTaskResult,
    _SchedulingSnapshot,
)


_WorkerContext = UVLFRunPlan
_WORKER_CONTEXT: UVLFRunPlan | None = None


def _build_worker_context(
    config: UVLFRunConfig,
    *,
    include_samples: bool = False,
) -> UVLFRunPlan:
    return build_uvlf_run_plan(
        config,
        include_samples=include_samples,
        ssp_kernel_loader=load_ssp_kernels,
        tng_cache_loader=preload_tng_mah_cache,
        thesan_cache_loader=preload_thesan_mah_cache,
    )


def _initialize_worker(config: UVLFRunConfig, include_samples: bool = False) -> None:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        raise RuntimeError("worker context is already initialized")
    _WORKER_CONTEXT = _build_worker_context(
        config,
        include_samples=include_samples,
    )


def _clear_worker_context_for_tests() -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = None


def _run_mass_task(spec: _MassTaskSpec) -> _MassTaskResult:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("worker context is not initialized")
    if type(spec) is not _MassTaskSpec:
        raise TypeError("spec must be exactly _MassTaskSpec")
    config = context.config
    if spec.redshift not in config.redshifts:
        raise ValueError("mass task redshift is not configured in worker context")
    resolved_cache_path = context.simulation_cache_path_for(spec.redshift)
    prepare_started = time.perf_counter()
    shared = prepare_shared_halo_batch(
        n_tracks=config.sampling.n_tracks_per_halo_mass,
        z_final=spec.redshift,
        Mh_final=spec.halo_mass_msun,
        cosmology=context.cosmology,
        random_seeds=derive_pipeline_random_seeds(
            config.base_seed,
            redshift=spec.redshift,
            mass_index=spec.mass_index,
        ),
        mass_index=spec.mass_index,
        z_start_max=config.mah.z_start_max,
        n_grid=config.mah.n_time_steps,
        enable_popiii=context.switches.enable_popiii,
        popiii_sfr_parameters=context.popiii_parameters,
        sampler=context.switches.parameter_sampler,
        mah_backend=context.switches.mah_backend,
        tng_mah_cache_path=(
            resolved_cache_path
            if context.switches.mah_backend == "tng"
            else config.mah.tng_cache_path
        ),
        tng_mass_bin_width_dex=config.mah.tng_mass_bin_width_dex,
        tng_min_candidates=config.mah.tng_min_candidates,
        tng_smoothing_myr=config.mah.tng_smoothing_myr,
        tng_time_grid_mode=config.mah.tng_time_grid_mode,
        thesan_mah_cache_path=(
            resolved_cache_path
            if context.switches.mah_backend == "thesan"
            else config.mah.thesan_cache_path
        ),
        thesan_mass_bin_width_dex=config.mah.thesan_mass_bin_width_dex,
        thesan_min_candidates=config.mah.thesan_min_candidates,
        thesan_smoothing_myr=config.mah.thesan_smoothing_myr,
        thesan_time_grid_mode=config.mah.thesan_time_grid_mode,
        enable_time_delay=context.switches.enable_time_delay,
        sfr_model_parameters=context.sfr_parameters,
        mzr_metallicity_parameters=context.mzr_parameters,
        regulator_metallicity_parameters=context.regulator_parameters,
        burst_scatter_dex=config.star_formation.burst_scatter_dex,
        burst_scatter_timescale_myr=(
            config.star_formation.burst_scatter_correlation_timescale_myr
        ),
        burst_scatter_preserve_mean=(
            config.star_formation.burst_scatter_mass_conserving
        ),
    )
    shared_seconds = time.perf_counter() - prepare_started
    final_sfr_mean = float(np.mean(shared.sfr_msun_per_yr_grid[:, -1]))
    final_popiii_sfr_mean = float(
        np.mean(shared.popiii_sfr_msun_per_yr_grid[:, -1])
    )
    final_sfr_samples = (
        shared.sfr_msun_per_yr_grid[:, -1]
        if context.include_samples
        else None
    )
    final_popiii_sfr_samples = (
        shared.popiii_sfr_msun_per_yr_grid[:, -1]
        if context.include_samples
        else None
    )
    mode_results: list[_MassModeTaskResult] = []
    for mode in context.switches.imf_modes:
        evaluation_started = time.perf_counter()
        evaluation = evaluate_shared_halo_batch(
            shared,
            imf_mode=mode,
            transition_parameters=context.transition_parameters,
            kernels=context.kernels,
            ssp_lookback_max_myr=100.0,
        )
        mode_results.append(
            _MassModeTaskResult(
                imf_mode=mode,
                uv_luminosity_erg_per_s_hz=(
                    evaluation.uv_luminosity_erg_per_s_hz
                ),
                topheavy_source_count=evaluation.topheavy_source_count,
                starforming_source_count=evaluation.starforming_source_count,
                popiii_source_count=evaluation.popiii_source_count,
                active_source_count=evaluation.active_source_count,
                evaluation_seconds=time.perf_counter() - evaluation_started,
            )
        )
        del evaluation
    del shared
    return _MassTaskResult(
        redshift=spec.redshift,
        mass_index=spec.mass_index,
        halo_mass_msun=spec.halo_mass_msun,
        mass_weight_per_mpc3=spec.mass_weight_per_mpc3,
        final_sfr_mean_msun_per_yr=final_sfr_mean,
        final_popiii_sfr_mean_msun_per_yr=final_popiii_sfr_mean,
        mode_results=tuple(mode_results),
        shared_preparation_seconds=shared_seconds,
        worker_pid=os.getpid(),
        worker_context_token=context.context_token,
        worker_initialization_load_count=context.initialization_load_count,
        final_sfr_msun_per_yr=final_sfr_samples,
        final_popiii_sfr_msun_per_yr=final_popiii_sfr_samples,
    )


def _validate_scheduled_result(
    spec: _MassTaskSpec,
    result: object,
) -> _MassTaskResult:
    if type(spec) is not _MassTaskSpec:
        raise TypeError("mass task spec must be exactly _MassTaskSpec")
    if type(result) is not _MassTaskResult:
        raise TypeError("mass task must return exactly _MassTaskResult")
    for mode_result in result.mode_results:
        _validate_mass_mode_task_result_integrity(mode_result)
    result_index = _strict_int_scalar("mass task result mass_index", result.mass_index)
    result_redshift = _strict_float_scalar("mass task result redshift", result.redshift)
    result_halo_mass = _strict_float_scalar(
        "mass task result halo_mass_msun",
        result.halo_mass_msun,
    )
    result_mass_weight = _strict_float_scalar(
        "mass task result mass_weight_per_mpc3",
        result.mass_weight_per_mpc3,
    )
    if result_index != spec.mass_index:
        raise RuntimeError(
            f"mass task result index {result_index} does not match submitted "
            f"index {spec.mass_index}"
        )
    if result_redshift != spec.redshift:
        raise RuntimeError("mass task result redshift does not match submitted spec")
    if result_halo_mass != spec.halo_mass_msun:
        raise RuntimeError("mass task result halo mass does not match submitted spec")
    if result_mass_weight != spec.mass_weight_per_mpc3:
        raise RuntimeError("mass task result mass weight does not match submitted spec")
    return result


def _ordered_parallel_results(
    specs: Iterable[_MassTaskSpec],
    *,
    executor: Executor,
    max_workers: int,
    scheduling_observer: Callable[[_SchedulingSnapshot], None] | None = None,
) -> Iterator[_MassTaskResult]:
    worker_count = _strict_int_scalar("max_workers", max_workers)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    window = 2 * worker_count
    spec_iterator = iter(specs)
    running: dict[Future[_MassTaskResult], _MassTaskSpec] = {}
    completed: dict[int, _MassTaskResult] = {}
    submitted_indices: set[int] = set()
    submitted_count = 0
    consumed_count = 0
    expected_index: int | None = None
    last_submitted_index: int | None = None
    exhausted = False

    def notify() -> None:
        if scheduling_observer is None:
            return
        scheduling_observer(
            _SchedulingSnapshot(
                running_count=len(running),
                completed_waiting_count=len(completed),
                total_occupancy=len(running) + len(completed),
                submitted_count=submitted_count,
                consumed_count=consumed_count,
            )
        )

    def fill_window() -> None:
        nonlocal exhausted, expected_index, last_submitted_index, submitted_count
        while not exhausted and len(running) + len(completed) < window:
            try:
                spec = next(spec_iterator)
            except StopIteration:
                exhausted = True
                break
            if type(spec) is not _MassTaskSpec:
                raise TypeError("mass spec iterator must yield exactly _MassTaskSpec")
            if spec.mass_index in submitted_indices:
                raise RuntimeError(f"duplicate mass task index {spec.mass_index}")
            if last_submitted_index is not None and spec.mass_index != last_submitted_index + 1:
                raise RuntimeError(
                    f"mass task indices must be contiguous; got {spec.mass_index} after "
                    f"{last_submitted_index}"
                )
            if expected_index is None:
                expected_index = spec.mass_index
            future = executor.submit(_run_mass_task, spec)
            running[future] = spec
            submitted_indices.add(spec.mass_index)
            last_submitted_index = spec.mass_index
            submitted_count += 1
            notify()

    try:
        fill_window()
        while running or completed:
            while expected_index is not None and expected_index in completed:
                result = completed.pop(expected_index)
                consumed_count += 1
                notify()
                yield result
                expected_index += 1
                fill_window()
            if not running:
                if completed:
                    raise RuntimeError(
                        f"missing mass task result for expected index {expected_index}"
                    )
                break
            done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            if not done:
                raise RuntimeError("parallel mass scheduler wait returned no completed futures")
            for future in done:
                spec = running.pop(future)
                result = _validate_scheduled_result(spec, future.result())
                if result.mass_index in completed:
                    raise RuntimeError(f"duplicate mass task result index {result.mass_index}")
                completed[result.mass_index] = result
            notify()
    except BaseException:
        for future in running:
            future.cancel()
        raise


def _iter_mass_task_specs(
    config: UVLFRunConfig,
    *,
    redshift: float,
    cosmology: Cosmology,
    hmf_seconds: list[float],
) -> Iterator[_MassTaskSpec]:
    hmf_started = time.perf_counter()
    interpolator = prepare_reed07_hmf_interpolator(
        log10_halo_mass_min_msun=config.sampling.log10_halo_mass_min_msun,
        log10_halo_mass_max_msun=config.sampling.log10_halo_mass_max_msun,
        z_obs=redshift,
        cosmology=cosmology,
        hmf_dlog10m=config.sampling.hmf_dlog10m,
    )
    hmf_seconds[0] += time.perf_counter() - hmf_started
    mass_rng = np.random.default_rng(derive_hmf_mass_seed(config.base_seed, redshift))
    for chunk_start in range(
        0,
        config.sampling.n_halo_mass_samples,
        config.sampling.mass_batch_size,
    ):
        chunk_stop = min(
            chunk_start + config.sampling.mass_batch_size,
            config.sampling.n_halo_mass_samples,
        )
        chunk_size = chunk_stop - chunk_start
        hmf_started = time.perf_counter()
        log_mass = mass_rng.uniform(
            config.sampling.log10_halo_mass_min_msun,
            config.sampling.log10_halo_mass_max_msun,
            size=chunk_size,
        )
        mass_msun = np.power(10.0, log_mass)
        dndm = _strict_nonnegative_float_1d(
            "HMF dndm",
            interpolator.evaluate(mass_msun),
            expected_shape=mass_msun.shape,
        )
        dndlogm = mass_msun * np.log(10.0) * dndm
        mass_weight = (
            config.sampling.log10_halo_mass_max_msun
            - config.sampling.log10_halo_mass_min_msun
        ) * dndlogm / config.sampling.n_halo_mass_samples
        hmf_seconds[0] += time.perf_counter() - hmf_started
        for local_index in range(chunk_size):
            mass_index = chunk_start + local_index
            yield _MassTaskSpec(
                redshift=redshift,
                mass_index=mass_index,
                halo_mass_msun=float(mass_msun[local_index]),
                mass_weight_per_mpc3=float(mass_weight[local_index]),
            )
        del log_mass, mass_msun, dndm, dndlogm, mass_weight


def run_uvlf_streaming(
    config: UVLFRunConfig,
    *,
    _mass_result_observer: Callable[[_MassTaskResult], None] | None = None,
    _scheduling_observer: Callable[[_SchedulingSnapshot], None] | None = None,
    _halo_sample_observer: Callable[[HaloSampleTable], None] | None = None,
) -> UVLFRunResult:
    """Run the bounded shared-batch UVLF core without retaining halo histories."""

    if type(config) is not UVLFRunConfig:
        raise TypeError("config must be exactly UVLFRunConfig")
    if _halo_sample_observer is not None and not callable(_halo_sample_observer):
        raise TypeError("_halo_sample_observer must be callable or None")
    started = time.perf_counter()
    run_switches = UVLFModuleSwitches.from_config(
        config,
        include_samples=_halo_sample_observer is not None,
    )
    cosmology = config.cosmology.to_model()
    redshift_results: list[RedshiftResult] = []
    all_diagnostics: list[ModeRunDiagnostics] = []
    mode_count = len(run_switches.imf_modes)
    worker_count = min(
        config.sampling.workers,
        config.sampling.n_halo_mass_samples,
    )
    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, _halo_sample_observer is not None),
        )
    else:
        if _WORKER_CONTEXT is not None:
            raise RuntimeError("serial worker context was unexpectedly already initialized")
        _initialize_worker(config, _halo_sample_observer is not None)

    try:
        for redshift in config.redshifts:
            states = {
                mode: _ModeAccumulatorState(
                    histogram=WeightedHistogramAccumulator(
                        np.asarray(config.sampling.muv_bin_edges, dtype=float)
                    )
                )
                for mode in run_switches.imf_modes
            }
            hmf_seconds = [0.0]
            specs = _iter_mass_task_specs(
                config,
                redshift=redshift,
                cosmology=cosmology,
                hmf_seconds=hmf_seconds,
            )
            if executor is None:
                task_results: Iterable[_MassTaskResult] = (
                    _validate_scheduled_result(spec, _run_mass_task(spec))
                    for spec in specs
                )
            else:
                task_results = _ordered_parallel_results(
                    specs,
                    executor=executor,
                    max_workers=worker_count,
                    scheduling_observer=_scheduling_observer,
                )
            shared_seconds = 0.0
            consumed_count = 0
            for expected_index, result in enumerate(task_results):
                if result.mass_index != expected_index:
                    raise RuntimeError(
                        f"mass task results must be consumed in order; expected "
                        f"{expected_index}, got {result.mass_index}"
                    )
                if _mass_result_observer is not None:
                    _mass_result_observer(result)
                if _halo_sample_observer is not None:
                    _observe_halo_samples(
                        result,
                        config=config,
                        observer=_halo_sample_observer,
                    )
                shared_seconds += _consume_mass_task_result(
                    result,
                    config=config,
                    states=states,
                )
                consumed_count += 1
            if consumed_count != config.sampling.n_halo_mass_samples:
                raise RuntimeError(
                    f"missing mass task results: expected "
                    f"{config.sampling.n_halo_mass_samples}, got {consumed_count}"
                )
            shared_seconds += hmf_seconds[0]

            mode_results: list[IMFModeResult] = []
            for mode in run_switches.imf_modes:
                state = states[mode]
                histogram = state.histogram.finalize()
                observed_phi, observed_sigma = _observed_uvlf(
                    centers=np.array(histogram.centers, copy=True),
                    intrinsic_phi=np.array(
                        histogram.phi_per_mpc3_per_unit,
                        copy=True,
                    ),
                    intrinsic_sigma=np.array(
                        histogram.phi_sigma_per_mpc3_per_unit,
                        copy=True,
                    ),
                    redshift=redshift,
                    apply_dust=run_switches.apply_dust,
                )
                mode_results.append(
                    IMFModeResult(
                        imf_mode=mode,
                        bin_edges_muv=histogram.edges,
                        bin_centers_muv=histogram.centers,
                        bin_width_mag=histogram.width,
                        raw_counts=histogram.raw_counts,
                        weighted_counts_per_mpc3=(
                            histogram.weighted_counts_per_mpc3
                        ),
                        weight_squared_counts_per_mpc6=(
                            histogram.weight_squared_counts_per_mpc6
                        ),
                        weighted_count_sigma_per_mpc3=(
                            histogram.weighted_count_sigma_per_mpc3
                        ),
                        effective_counts=histogram.effective_counts,
                        phi_intrinsic_per_mpc3_per_mag=(
                            histogram.phi_per_mpc3_per_unit
                        ),
                        phi_intrinsic_sigma_per_mpc3_per_mag=(
                            histogram.phi_sigma_per_mpc3_per_unit
                        ),
                        phi_observed_per_mpc3_per_mag=observed_phi,
                        phi_observed_sigma_per_mpc3_per_mag=observed_sigma,
                        halo_tracks=(),
                    )
                )
                topheavy_fraction = (
                    state.topheavy_source_count / state.starforming_source_count
                    if state.starforming_source_count > 0
                    else 0.0
                )
                popiii_fraction = (
                    state.popiii_source_count / state.active_source_count
                    if state.active_source_count > 0
                    else 0.0
                )
                all_diagnostics.append(
                    ModeRunDiagnostics(
                        redshift=redshift,
                        imf_mode=mode,
                        sampling_seconds=(
                            state.evaluation_seconds + shared_seconds / mode_count
                        ),
                        sample_count=state.sample_count,
                        valid_sample_count=state.valid_sample_count,
                        topheavy_source_fraction=topheavy_fraction,
                        popiii_source_fraction=popiii_fraction,
                        sfrd_msun_per_yr_per_mpc3=(
                            state.sfrd_msun_per_yr_per_mpc3
                        ),
                        popiii_sfrd_msun_per_yr_per_mpc3=(
                            state.popiii_sfrd_msun_per_yr_per_mpc3
                        ),
                    )
                )
            redshift_results.append(
                RedshiftResult(
                    redshift=redshift,
                    imf_modes=tuple(mode_results),
                )
            )
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        raise
    finally:
        if worker_count == 1:
            _clear_worker_context_for_tests()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
    return UVLFRunResult(
        config=config,
        redshifts=tuple(redshift_results),
        diagnostics=RunDiagnostics(
            total_seconds=time.perf_counter() - started,
            mode_runs=tuple(all_diagnostics),
        ),
    )


__all__ = ["run_uvlf_streaming"]
