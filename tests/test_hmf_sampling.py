from __future__ import annotations

import inspect

import numpy as np
import pytest

from auroralf.mah import Cosmology

from auroralf.uvlf.hmf_sampling import (
    POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN,
    POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT,
    POPIII_LW_FEEDBACK_COEFFICIENT,
    POPIII_LW_FEEDBACK_EXPONENT,
    STELLAR_CHANNEL_BELOW_POPIII_MIN,
    STELLAR_CHANNEL_POPII,
    STELLAR_CHANNEL_POPIII,
    MASS_FUNCTION_MODEL_HMF_REED07,
    classify_halo_stellar_channels,
    compute_halo_mass_function_dndm,
    compute_atomic_cooling_mass_msun,
    compute_popiii_lw_minimum_mass_msun,
    compute_reed07_halo_mass_function_dndm,
    prepare_reed07_hmf_interpolator,
    validate_mass_function_model,
)


def test_validate_mass_function_model_accepts_reed07() -> None:
    assert validate_mass_function_model("HMF_REED07") == MASS_FUNCTION_MODEL_HMF_REED07


def test_validate_mass_function_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="mass_function_model"):
        validate_mass_function_model("press_schechter")


def test_hmf_sampling_rejects_legacy_independent_lw_keyword() -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling

    assert "lw_background_j21" not in inspect.signature(hmf_sampling.sample_uvlf_from_hmf).parameters
    legacy_kwargs = {"lw_background_j21": 0.2}
    with pytest.raises(TypeError, match="unexpected keyword argument 'lw_background_j21'"):
        hmf_sampling.sample_uvlf_from_hmf(
            z_obs=10.0,
            cosmology=Cosmology(),
            base_seed=42,
            N_mass=0,
            **legacy_kwargs,
        )


@pytest.mark.parametrize("invalid_lw", [-0.1, np.nan, np.inf, np.array([0.1])])
def test_hmf_sampling_validates_popiii_lw_at_entry_when_popiii_disabled(invalid_lw: object) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling
    from auroralf.sfr import PopIIISFRParameters

    params = PopIIISFRParameters(lw_background_j21=invalid_lw)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lw_background_j21 must be scalar, finite, and non-negative"):
        hmf_sampling.sample_uvlf_from_hmf(
            z_obs=10.0,
            cosmology=Cosmology(),
            base_seed=42,
            N_mass=0,
            enable_popiii=False,
            popiii_sfr_parameters=params,
        )


@pytest.mark.parametrize("pipeline_workers", [0, -1])
def test_hmf_sampling_rejects_nonpositive_pipeline_workers(
    pipeline_workers: int,
) -> None:
    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf

    with pytest.raises(ValueError, match="pipeline_workers must be positive"):
        sample_uvlf_from_hmf(
            z_obs=10.0,
            cosmology=Cosmology(),
            base_seed=42,
            N_mass=1,
            pipeline_workers=pipeline_workers,
        )


@pytest.mark.parametrize("pipeline_workers", [True, np.bool_(False), 1.5, "1"])
def test_hmf_sampling_rejects_noninteger_pipeline_workers(
    pipeline_workers: object,
) -> None:
    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf

    with pytest.raises(TypeError, match="pipeline_workers must be an integer non-boolean value"):
        sample_uvlf_from_hmf(
            z_obs=10.0,
            cosmology=Cosmology(),
            base_seed=42,
            N_mass=1,
            pipeline_workers=pipeline_workers,  # type: ignore[arg-type]
        )


def test_hmf_sampling_uses_one_popiii_lw_source_for_floor_channels_workers_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling
    from auroralf.sfr import PopIIISFRParameters

    params = PopIIISFRParameters(lw_background_j21=0.37)
    minimum_lw_values: list[float] = []
    channel_lw_values: list[float] = []
    worker_params: list[PopIIISFRParameters] = []

    def fake_minimum_mass(z_obs: float, *, lw_background_j21: float) -> float:
        del z_obs
        minimum_lw_values.append(lw_background_j21)
        return 1.0e7

    def fake_channels(
        halo_mass_msun: np.ndarray,
        *,
        z_obs: float,
        cosmology: Cosmology,
        lw_background_j21: float,
    ) -> np.ndarray:
        del z_obs, cosmology
        channel_lw_values.append(lw_background_j21)
        return np.full(np.asarray(halo_mass_msun).shape, STELLAR_CHANNEL_POPII)

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        cosmology: Cosmology,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        del z_obs, cosmology, mass_function_model, hmf_dlog10m
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_worker(args: tuple[object, ...]) -> tuple[object, ...]:
        worker_params.append(args[-3])  # type: ignore[arg-type]
        n_tracks = int(args[5])
        luminosity = np.full(n_tracks, 1.0e28, dtype=float)
        zeros = np.zeros(n_tracks, dtype=float)
        return (
            int(args[0]),
            float(args[1]),
            luminosity,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            0.0,
            0,
            0,
            0,
            n_tracks,
            np.nan,
            np.nan,
        )

    monkeypatch.setattr(hmf_sampling, "compute_popiii_lw_minimum_mass_msun", fake_minimum_mass)
    monkeypatch.setattr(hmf_sampling, "classify_halo_stellar_channels", fake_channels)
    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_worker)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=10.0,
        cosmology=Cosmology(),
        N_mass=2,
        n_tracks=1,
        base_seed=5,
        bins=np.array([-20.0, -15.0]),
        pipeline_workers=1,
        enable_popiii=False,
        popiii_sfr_parameters=params,
    )

    assert minimum_lw_values == pytest.approx([0.37])
    assert channel_lw_values == pytest.approx([0.37])
    assert len(worker_params) == 2
    assert all(worker_param is params for worker_param in worker_params)
    assert result.metadata["lw_background_j21"] == pytest.approx(0.37)
    assert result.metadata["popiii_sfr_parameters"]["lw_background_j21"] == pytest.approx(0.37)


def test_hmf_process_pool_preserves_popiii_lw_parameters_and_worker_results() -> None:
    from auroralf.sfr import PopIIISFRParameters
    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf

    common = {
        "z_obs": 10.0,
        "cosmology": Cosmology(),
        "N_mass": 2,
        "n_tracks": 1,
        "base_seed": 812,
        "bins": np.array([-100.0, 100.0]),
        "logM_min": 6.6,
        "logM_max": 6.7,
        "z_start_max": 14.0,
        "n_grid": 8,
        "enable_popiii": True,
    }
    lw_parameters = PopIIISFRParameters(lw_background_j21=0.37)

    serial = sample_uvlf_from_hmf(
        **common,
        pipeline_workers=1,
        popiii_sfr_parameters=lw_parameters,
    )
    parallel = sample_uvlf_from_hmf(
        **common,
        pipeline_workers=2,
        popiii_sfr_parameters=lw_parameters,
    )
    no_lw = sample_uvlf_from_hmf(
        **common,
        pipeline_workers=1,
        popiii_sfr_parameters=PopIIISFRParameters(lw_background_j21=0.0),
    )

    for sample_name in (
        "logMh",
        "Mh",
        "sample_weight",
        "sfr",
        "popiii_sfr",
        "popiii_luminosity",
        "popiii_light_fraction",
        "luminosity",
        "Muv",
    ):
        np.testing.assert_allclose(
            parallel.samples[sample_name],
            serial.samples[sample_name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    np.testing.assert_array_equal(
        parallel.samples["stellar_channel"],
        serial.samples["stellar_channel"],
    )
    for component in ("mah", "metallicity", "burst"):
        assert serial.metadata["pipeline_random_seeds_by_mass"][component].dtype == np.uint64
        np.testing.assert_array_equal(
            parallel.metadata["pipeline_random_seeds_by_mass"][component],
            serial.metadata["pipeline_random_seeds_by_mass"][component],
        )
    for uvlf_name in (
        "bin_edges",
        "bin_centers",
        "bin_width",
        "raw_counts",
        "weighted_counts",
        "weight_squared_counts",
        "weighted_count_sigma",
        "effective_counts",
        "phi",
        "phi_sigma",
    ):
        np.testing.assert_allclose(
            parallel.uvlf[uvlf_name],
            serial.uvlf[uvlf_name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    assert parallel.metadata["pipeline_workers"] == 2
    assert serial.metadata["pipeline_workers"] == 1
    assert parallel.metadata["lw_background_j21"] == pytest.approx(0.37)
    assert parallel.metadata["popiii_sfr_parameters"] == lw_parameters.as_metadata()
    np.testing.assert_array_equal(
        parallel.metadata["popiii_source_count_by_mass"],
        serial.metadata["popiii_source_count_by_mass"],
    )
    assert parallel.metadata["popiii_sfrd_msun_yr_mpc3"] == pytest.approx(
        serial.metadata["popiii_sfrd_msun_yr_mpc3"],
        rel=0.0,
        abs=0.0,
    )

    assert np.any(np.asarray(parallel.samples["popiii_sfr"]) > 0.0)
    assert np.any(np.asarray(parallel.samples["popiii_luminosity"]) > 0.0)
    assert np.max(
        np.abs(
            np.asarray(parallel.samples["popiii_sfr"], dtype=float)
            - np.asarray(no_lw.samples["popiii_sfr"], dtype=float)
        )
    ) > 0.0
    assert np.max(
        np.abs(
            np.asarray(parallel.samples["popiii_luminosity"], dtype=float)
            - np.asarray(no_lw.samples["popiii_luminosity"], dtype=float)
        )
    ) > 0.0


def test_hmf_public_apis_require_cosmology() -> None:
    with pytest.raises(TypeError, match="cosmology"):
        compute_reed07_halo_mass_function_dndm(1.0e10, 6.0)
    with pytest.raises(TypeError, match="cosmology"):
        compute_halo_mass_function_dndm(1.0e10, 6.0)


def test_reed07_uses_supplied_cosmology(monkeypatch: pytest.MonkeyPatch) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling

    cosmology = Cosmology(
        h0=0.7 * 100.0 * Cosmology().h0 / Cosmology().h0_km_s_mpc,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )
    captured: dict[str, object] = {}

    class FakeMassFunction:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            h = float(kwargs["cosmo_params"]["H0"]) / 100.0  # type: ignore[index]
            self.m = np.array([1.0e8, 1.0e10, 1.0e12], dtype=float) * h
            self.dndm = np.array([1.0e-8, 1.0e-10, 1.0e-12], dtype=float) / h**4

    monkeypatch.setattr(hmf_sampling, "MassFunction", FakeMassFunction)

    result = compute_reed07_halo_mass_function_dndm(
        np.array([1.0e9, 1.0e11]),
        6.0,
        cosmology=cosmology,
    )

    cosmo_params = captured["cosmo_params"]
    assert cosmo_params["H0"] == pytest.approx(cosmology.h0_km_s_mpc)
    assert cosmo_params["Om0"] == pytest.approx(cosmology.omega_m)
    assert cosmo_params["Ob0"] == pytest.approx(cosmology.omega_b)
    assert np.all(np.asarray(result) > 0.0)


def test_fixed_reed07_interpolator_is_bitwise_independent_of_chunk_partition() -> None:
    interpolator = prepare_reed07_hmf_interpolator(
        log10_halo_mass_min_msun=8.75,
        log10_halo_mass_max_msun=12.25,
        z_obs=7.0,
        cosmology=Cosmology(),
        hmf_dlog10m=0.02,
    )
    masses = np.logspace(8.75, 12.25, 17)

    whole = interpolator.evaluate(masses)
    chunked = np.concatenate(
        [
            interpolator.evaluate(masses[:1]),
            interpolator.evaluate(masses[1:3]),
            interpolator.evaluate(masses[3:8]),
            interpolator.evaluate(masses[8:]),
        ]
    )

    np.testing.assert_array_equal(chunked, whole)
    assert float(np.max(np.abs(chunked - whole))) == 0.0
    for grid in (interpolator.log_mass_grid, interpolator.log_dndm_grid):
        current: object = grid
        while isinstance(current, np.ndarray):
            assert current.flags.writeable is False
            with pytest.raises(ValueError, match="WRITEABLE|writeable"):
                current.setflags(write=True)
            current = current.base


@pytest.mark.parametrize(
    "masses",
    [
        np.array(["1e9"]),
        np.array([True]),
        np.array([1.0e9 + 0.0j]),
        np.array([1.0e8]),
        np.array([np.nan]),
    ],
)
def test_fixed_reed07_interpolator_strictly_validates_evaluation_masses(
    masses: np.ndarray,
) -> None:
    interpolator = prepare_reed07_hmf_interpolator(
        log10_halo_mass_min_msun=9.0,
        log10_halo_mass_max_msun=10.0,
        z_obs=6.0,
        cosmology=Cosmology(),
        hmf_dlog10m=0.02,
    )

    with pytest.raises((TypeError, ValueError), match="mass"):
        interpolator.evaluate(masses)


def test_hmf_sampling_preserves_cosmology_identity_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling

    cosmology = Cosmology(
        h0=0.71 * 100.0 * Cosmology().h0 / Cosmology().h0_km_s_mpc,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )
    hmf_contexts: list[Cosmology] = []
    worker_contexts: list[Cosmology] = []
    cooling_contexts: list[Cosmology] = []
    channel_contexts: list[Cosmology] = []

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        cosmology: Cosmology,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        del z_obs, mass_function_model, hmf_dlog10m
        hmf_contexts.append(cosmology)
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_worker(args: tuple[object, ...]) -> tuple[object, ...]:
        worker_contexts.append(args[-1])  # type: ignore[arg-type]
        n_tracks = int(args[5])
        luminosity = np.full(n_tracks, 1.0e28, dtype=float)
        zeros = np.zeros(n_tracks, dtype=float)
        return (
            int(args[0]),
            float(args[1]),
            luminosity,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            0.0,
            0,
            0,
            0,
            n_tracks,
            np.nan,
            np.nan,
        )

    def fake_atomic_mass(z_obs: float, *, cosmology: Cosmology, **kwargs: object) -> float:
        del z_obs, kwargs
        cooling_contexts.append(cosmology)
        return 1.0e8

    def fake_channels(
        halo_mass_msun: np.ndarray,
        *,
        z_obs: float,
        cosmology: Cosmology,
        **kwargs: object,
    ) -> np.ndarray:
        del z_obs, kwargs
        channel_contexts.append(cosmology)
        return np.full(np.asarray(halo_mass_msun).shape, STELLAR_CHANNEL_POPII)

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_worker)
    monkeypatch.setattr(hmf_sampling, "compute_atomic_cooling_mass_msun", fake_atomic_mass)
    monkeypatch.setattr(hmf_sampling, "classify_halo_stellar_channels", fake_channels)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=6.0,
        cosmology=cosmology,
        base_seed=42,
        N_mass=1,
        n_tracks=1,
        bins=np.array([-20.0, -15.0]),
        pipeline_workers=1,
    )

    assert hmf_contexts[0] is cosmology
    assert worker_contexts[0] is cosmology
    assert cooling_contexts[0] is cosmology
    assert channel_contexts[0] is cosmology
    assert result.metadata["mass_function_parameters"] == {
        "ns": hmf_sampling.MASS_FUNCTION_NS,
        "sigma8": hmf_sampling.MASS_FUNCTION_SIGMA8,
        "h": pytest.approx(cosmology.h0_km_s_mpc / 100.0),
        "h0_km_s_mpc": pytest.approx(cosmology.h0_km_s_mpc),
        "omega_m": pytest.approx(cosmology.omega_m),
        "omega_b": pytest.approx(cosmology.omega_b),
        "omega_lambda": pytest.approx(cosmology.omega_lambda),
    }


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_validate_mass_function_model_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        validate_mass_function_model(deprecated_model)


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_compute_mass_function_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        compute_halo_mass_function_dndm(
            1.0e10,
            12.5,
            cosmology=Cosmology(),
            mass_function_model=deprecated_model,
        )


def test_reed07_mass_function_returns_positive_dndm() -> None:
    halo_mass = np.array([1.0e9, 1.0e10, 1.0e11])

    reed07 = np.asarray(
        compute_halo_mass_function_dndm(
            halo_mass,
            12.5,
            cosmology=Cosmology(),
            mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
        ),
        dtype=float,
    )
    direct_reed07 = np.asarray(
        compute_reed07_halo_mass_function_dndm(
            halo_mass,
            12.5,
            cosmology=Cosmology(),
        ),
        dtype=float,
    )

    assert np.all(reed07 > 0.0)
    np.testing.assert_allclose(reed07, direct_reed07, rtol=0.0, atol=0.0)


def test_hmf_reed07_scalar_input_returns_float() -> None:
    value = compute_halo_mass_function_dndm(
        1.0e10,
        6.0,
        cosmology=Cosmology(),
        mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
    )

    assert isinstance(value, float)
    assert value > 0.0


def test_atomic_cooling_mass_matches_astropy_planck18_reference() -> None:
    z_obs = np.array([0.0, 6.0, 10.0, 12.5, 20.0, 40.0, 50.0])
    expected = np.array(
        [
            2.1548341297061906e9,
            1.5850980685330954e8,
            8.057033196980429e7,
            5.92725023366572e7,
            3.0556436621610377e7,
            1.1201604186524495e7,
            8.074256846391252e6,
        ]
    )
    cosmology = Cosmology()

    threshold = compute_atomic_cooling_mass_msun(z_obs, cosmology=cosmology)

    np.testing.assert_allclose(threshold, expected, rtol=1.0e-14, atol=0.0)


def test_atomic_cooling_mass_uses_custom_cosmology() -> None:
    custom = Cosmology(
        h0=2.0 * Cosmology().h0,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )
    expected = 3.545044267043421e7

    actual = compute_atomic_cooling_mass_msun(10.0, cosmology=custom)
    default = compute_atomic_cooling_mass_msun(10.0, cosmology=Cosmology())

    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(default)


def test_atomic_cooling_mass_preserves_scalar_and_array_shapes() -> None:
    cosmology = Cosmology()

    scalar = compute_atomic_cooling_mass_msun(10.0, cosmology=cosmology)
    array = compute_atomic_cooling_mass_msun(
        np.array([[6.0, 10.0], [20.0, 50.0]]),
        cosmology=cosmology,
    )

    assert type(scalar) is float
    assert array.shape == (2, 2)
    assert np.all(np.isfinite(array))
    assert np.all(array > 0.0)


@pytest.mark.parametrize("name", ["virial_temperature_k", "mu"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_atomic_cooling_mass_rejects_invalid_physical_parameters(
    name: str,
    value: float,
) -> None:
    kwargs = {name: value}

    with pytest.raises(ValueError, match=name):
        compute_atomic_cooling_mass_msun(
            10.0,
            cosmology=Cosmology(),
            **kwargs,
        )


def test_atomic_cooling_public_apis_require_cosmology() -> None:
    with pytest.raises(TypeError, match="cosmology"):
        compute_atomic_cooling_mass_msun(10.0)
    with pytest.raises(TypeError, match="cosmology"):
        classify_halo_stellar_channels(1.0e8, z_obs=10.0)


def test_popiii_lw_minimum_mass_defaults_to_cruz_molecular_cooling_floor() -> None:
    z_obs = np.array([10.0, 20.0, 25.0])
    expected = POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN * (
        1.0 + z_obs
    ) ** POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT

    threshold = compute_popiii_lw_minimum_mass_msun(z_obs)

    np.testing.assert_allclose(threshold, expected)


def test_popiii_lw_minimum_mass_increases_with_lw_background() -> None:
    z_obs = 20.0
    lw_background_j21 = 0.2
    base = POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN * (
        1.0 + z_obs
    ) ** POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT
    expected = base * (
        1.0 + POPIII_LW_FEEDBACK_COEFFICIENT * lw_background_j21**POPIII_LW_FEEDBACK_EXPONENT
    )

    threshold = compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=lw_background_j21)

    assert threshold == pytest.approx(expected)


def test_halo_stellar_channel_uses_popiii_and_atomic_thresholds() -> None:
    z_obs = 10.0
    popiii_min = compute_popiii_lw_minimum_mass_msun(z_obs)
    cosmology = Cosmology()
    atomic_threshold = compute_atomic_cooling_mass_msun(z_obs, cosmology=cosmology)
    halo_mass = np.array(
        [0.5 * popiii_min, 1.1 * popiii_min, atomic_threshold, 2.0 * atomic_threshold],
        dtype=float,
    )

    channels = classify_halo_stellar_channels(
        halo_mass,
        z_obs=z_obs,
        cosmology=cosmology,
    )

    np.testing.assert_array_equal(
        channels,
        np.array(
            [
                STELLAR_CHANNEL_BELOW_POPIII_MIN,
                STELLAR_CHANNEL_POPIII,
                STELLAR_CHANNEL_POPII,
                STELLAR_CHANNEL_POPII,
            ]
        ),
    )


def test_hmf_sampling_records_popiii_and_atomic_cooling_stellar_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling

    z_obs = 10.0
    popiii_min = compute_popiii_lw_minimum_mass_msun(z_obs)
    atomic_threshold = compute_atomic_cooling_mass_msun(z_obs, cosmology=Cosmology())

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        cosmology: Cosmology,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        assert isinstance(cosmology, Cosmology)
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_run_single_mass_sample(args: tuple[object, ...]) -> tuple[object, ...]:
        mass_index = int(args[0])
        log_mass = float(args[1])
        n_tracks = int(args[5])
        luminosity = np.full(n_tracks, 1.0e28, dtype=float)
        sfr = np.full(n_tracks, 1.0e-2, dtype=float)
        popiii_luminosity = np.zeros(n_tracks, dtype=float)
        return (
            mass_index,
            log_mass,
            luminosity,
            sfr,
            np.zeros(n_tracks, dtype=float),
            popiii_luminosity,
            np.zeros(n_tracks, dtype=float),
            np.zeros(n_tracks, dtype=float),
            0.0,
            0,
            n_tracks,
            0,
            n_tracks,
            np.nan,
            np.nan,
        )

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_run_single_mass_sample)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=z_obs,
        cosmology=Cosmology(),
        N_mass=32,
        n_tracks=1,
        base_seed=7,
        bins=np.array([-20.0, -15.0]),
        logM_min=np.log10(0.5 * popiii_min),
        logM_max=np.log10(2.0 * atomic_threshold),
        pipeline_workers=1,
    )

    channel_by_mass = np.asarray(result.metadata["stellar_channel_by_mass"])
    halo_mass_by_mass = np.asarray(result.metadata["halo_mass_by_mass"], dtype=float)
    expected_by_mass = np.where(
        halo_mass_by_mass < popiii_min,
        STELLAR_CHANNEL_BELOW_POPIII_MIN,
        np.where(halo_mass_by_mass < atomic_threshold, STELLAR_CHANNEL_POPIII, STELLAR_CHANNEL_POPII),
    )

    assert result.metadata["atomic_cooling_temperature_k"] == pytest.approx(1.0e4)
    assert result.metadata["atomic_cooling_mass_msun"] == pytest.approx(atomic_threshold)
    assert result.metadata["lw_background_j21"] == pytest.approx(0.0)
    assert result.metadata["popiii_minimum_mass_msun"] == pytest.approx(popiii_min)
    np.testing.assert_array_equal(channel_by_mass, expected_by_mass)
    np.testing.assert_array_equal(result.samples["stellar_channel"], expected_by_mass)
    np.testing.assert_allclose(result.samples["popiii_minimum_mass_msun"], popiii_min)
    np.testing.assert_allclose(result.samples["popiii_luminosity"], 0.0)
    np.testing.assert_allclose(result.samples["popiii_light_fraction"], 0.0)
    assert np.any(channel_by_mass == STELLAR_CHANNEL_POPIII)
    assert np.any(channel_by_mass == STELLAR_CHANNEL_POPII)
    assert np.any(channel_by_mass == STELLAR_CHANNEL_BELOW_POPIII_MIN)


def test_hmf_sampling_records_popiii_luminosity_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling
    from auroralf.sfr import PopIIISFRParameters

    z_obs = 10.0

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        cosmology: Cosmology,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        assert isinstance(cosmology, Cosmology)
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_run_single_mass_sample(args: tuple[object, ...]) -> tuple[object, ...]:
        mass_index = int(args[0])
        log_mass = float(args[1])
        n_tracks = int(args[5])
        luminosity = np.full(n_tracks, 1.0e28, dtype=float)
        sfr = np.full(n_tracks, 2.0e-2, dtype=float)
        popiii_luminosity = np.full(n_tracks, 2.5e27, dtype=float)
        popiii_sfr = np.full(n_tracks, 3.0e-4, dtype=float)
        return (
            mass_index,
            log_mass,
            luminosity,
            sfr,
            np.zeros(n_tracks, dtype=float),
            popiii_luminosity,
            popiii_luminosity / luminosity,
            popiii_sfr,
            0.0,
            0,
            n_tracks,
            n_tracks,
            n_tracks,
            np.nan,
            np.nan,
        )

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_run_single_mass_sample)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=z_obs,
        cosmology=Cosmology(),
        N_mass=4,
        n_tracks=2,
        base_seed=11,
        bins=np.array([-20.0, -15.0]),
        logM_min=7.0,
        logM_max=8.0,
        pipeline_workers=1,
        enable_popiii=True,
        popiii_sfr_parameters=PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8),
        popiii_ssp_file="popiii.dat",
    )

    np.testing.assert_allclose(result.samples["popiii_luminosity"], 2.5e27)
    np.testing.assert_allclose(result.samples["popiii_light_fraction"], 0.25)
    np.testing.assert_allclose(result.samples["sfr"], 2.0e-2)
    np.testing.assert_allclose(result.samples["popiii_sfr"], 3.0e-4)
    expected_popii_sfrd = float(np.sum(result.samples["sfr"] * result.samples["sample_weight"]))
    expected_sfrd = float(np.sum(result.samples["popiii_sfr"] * result.samples["sample_weight"]))
    assert result.metadata["sfrd_msun_yr_mpc3"] == pytest.approx(expected_popii_sfrd)
    assert result.metadata["popiii_sfrd_msun_yr_mpc3"] == pytest.approx(expected_sfrd)
    assert result.metadata["popiii_enabled"] is True
    assert result.metadata["popiii_source_count_by_mass"].shape == (4,)
    assert result.metadata["popiii_source_fraction"] == pytest.approx(1.0)
    assert result.metadata["popiii_light_fraction_median"] == pytest.approx(0.25)


def test_hmf_sampling_popiii_source_fraction_uses_active_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling
    from auroralf.sfr import PopIIISFRParameters

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        cosmology: Cosmology,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        assert isinstance(cosmology, Cosmology)
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_run_single_mass_sample(args: tuple[object, ...]) -> tuple[object, ...]:
        mass_index = int(args[0])
        log_mass = float(args[1])
        n_tracks = int(args[5])
        luminosity = np.full(n_tracks, 1.0e28, dtype=float)
        sfr = np.zeros(n_tracks, dtype=float)
        popiii_luminosity = np.full(n_tracks, 1.0e27, dtype=float)
        popiii_sfr = np.zeros(n_tracks, dtype=float)
        return (
            mass_index,
            log_mass,
            luminosity,
            sfr,
            np.zeros(n_tracks, dtype=float),
            popiii_luminosity,
            popiii_luminosity / luminosity,
            popiii_sfr,
            0.0,
            0,
            0,
            n_tracks,
            2 * n_tracks,
            np.nan,
            np.nan,
        )

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_run_single_mass_sample)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=10.0,
        cosmology=Cosmology(),
        N_mass=2,
        n_tracks=3,
        base_seed=12,
        bins=np.array([-20.0, -15.0]),
        logM_min=7.0,
        logM_max=8.0,
        pipeline_workers=1,
        enable_popiii=True,
        popiii_sfr_parameters=PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8),
        popiii_ssp_file="popiii.dat",
    )

    assert result.metadata["popiii_source_fraction"] == pytest.approx(0.5)
