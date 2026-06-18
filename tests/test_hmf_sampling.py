from __future__ import annotations

import numpy as np
import pytest

from auroralf.uvlf.hmf_sampling import (
    POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN,
    POPIII_H2_COOLING_REDSHIFT_FACTOR,
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
    validate_mass_function_model,
)


def test_validate_mass_function_model_accepts_reed07() -> None:
    assert validate_mass_function_model("HMF_REED07") == MASS_FUNCTION_MODEL_HMF_REED07


def test_validate_mass_function_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="mass_function_model"):
        validate_mass_function_model("press_schechter")


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_validate_mass_function_model_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        validate_mass_function_model(deprecated_model)


@pytest.mark.parametrize("deprecated_model", ["massfunc_st", "hmf_watson13_fof"])
def test_compute_mass_function_rejects_deprecated_models(deprecated_model: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        compute_halo_mass_function_dndm(1.0e10, 12.5, mass_function_model=deprecated_model)


def test_reed07_mass_function_returns_positive_dndm() -> None:
    halo_mass = np.array([1.0e9, 1.0e10, 1.0e11])

    reed07 = np.asarray(
        compute_halo_mass_function_dndm(
            halo_mass,
            12.5,
            mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
        ),
        dtype=float,
    )
    direct_reed07 = np.asarray(compute_reed07_halo_mass_function_dndm(halo_mass, 12.5), dtype=float)

    assert np.all(reed07 > 0.0)
    np.testing.assert_allclose(reed07, direct_reed07, rtol=0.0, atol=0.0)


def test_hmf_reed07_scalar_input_returns_float() -> None:
    value = compute_halo_mass_function_dndm(
        1.0e10,
        6.0,
        mass_function_model=MASS_FUNCTION_MODEL_HMF_REED07,
    )

    assert isinstance(value, float)
    assert value > 0.0


def test_atomic_cooling_mass_matches_massfunc_mvir() -> None:
    import massfunc as mf

    z_obs = 10.0
    expected = mf.SFRD().M_vir(0.61, 1.0e4, z_obs)

    threshold = compute_atomic_cooling_mass_msun(z_obs)

    assert threshold == pytest.approx(expected)


def test_popiii_lw_minimum_mass_defaults_to_h2_cooling_floor() -> None:
    z_obs = 25.0
    expected = POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN * (
        POPIII_H2_COOLING_REDSHIFT_FACTOR / (1.0 + z_obs)
    )

    threshold = compute_popiii_lw_minimum_mass_msun(z_obs)

    assert threshold == pytest.approx(expected)


def test_popiii_lw_minimum_mass_increases_with_lw_background() -> None:
    z_obs = 20.0
    lw_background_j21 = 0.2
    base = POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN * (
        POPIII_H2_COOLING_REDSHIFT_FACTOR / (1.0 + z_obs)
    )
    expected = base * (
        1.0 + POPIII_LW_FEEDBACK_COEFFICIENT * lw_background_j21**POPIII_LW_FEEDBACK_EXPONENT
    )

    threshold = compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=lw_background_j21)

    assert threshold == pytest.approx(expected)


def test_halo_stellar_channel_uses_popiii_and_atomic_thresholds() -> None:
    z_obs = 10.0
    popiii_min = compute_popiii_lw_minimum_mass_msun(z_obs)
    atomic_threshold = compute_atomic_cooling_mass_msun(z_obs)
    halo_mass = np.array(
        [0.5 * popiii_min, 1.1 * popiii_min, atomic_threshold, 2.0 * atomic_threshold],
        dtype=float,
    )

    channels = classify_halo_stellar_channels(halo_mass, z_obs=z_obs)

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
    atomic_threshold = compute_atomic_cooling_mass_msun(z_obs)

    def fake_dndm(
        halo_mass_msun: np.ndarray,
        z_obs: float,
        *,
        mass_function_model: str,
        hmf_dlog10m: float,
    ) -> np.ndarray:
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_run_single_mass_sample(
        args: tuple[object, ...],
    ) -> tuple[int, float, np.ndarray, np.ndarray, float, int, int, float, float]:
        mass_index = int(args[0])
        log_mass = float(args[1])
        n_tracks = int(args[5])
        return (
            mass_index,
            log_mass,
            np.full(n_tracks, 1.0e28, dtype=float),
            np.zeros(n_tracks, dtype=float),
            0.0,
            0,
            n_tracks,
            np.nan,
            np.nan,
        )

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_run_single_mass_sample)

    result = hmf_sampling.sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=32,
        n_tracks=1,
        random_seed=7,
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
    assert np.any(channel_by_mass == STELLAR_CHANNEL_POPIII)
    assert np.any(channel_by_mass == STELLAR_CHANNEL_POPII)
    assert np.any(channel_by_mass == STELLAR_CHANNEL_BELOW_POPIII_MIN)
