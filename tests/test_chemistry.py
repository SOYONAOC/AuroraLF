from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.seeding import derive_pipeline_random_seeds

from auroralf.chemistry import (
    MZRBirthMetallicityParameters,
    RegulatorMetallicityParameters,
    compute_regulator_metallicity,
    compute_mzr_birth_metallicity,
    fire2_highz_mzr_oh12,
)
from auroralf.uvlf.imf import IMF_MODE_Z_GATED_MILD_TOPHEAVY, IMFTransitionParameters
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _toy_history() -> dict[str, np.ndarray]:
    return {
        "t_grid_gyr": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float),
        "z_grid": np.array([[14.0, 12.0, 10.0, 8.0]], dtype=float),
        "mh_grid": np.array([[1.0e9, 2.0e9, 4.0e9, 8.0e9]], dtype=float),
        "dmhdt_sfr_grid": np.array([[0.0, 1.0e10, 2.0e10, 4.0e10]], dtype=float),
        "sfr_grid": np.array([[0.0, 0.02, 0.04, 0.08]], dtype=float),
        "active_grid": np.array([[False, True, True, True]], dtype=bool),
    }


def test_mzr_birth_metallicity_uses_cumulative_stellar_mass() -> None:
    history = {
        "t_grid_gyr": np.array([[0.1, 0.2, 0.3]], dtype=float),
        "z_grid": np.array([[12.0, 12.0, 12.0]], dtype=float),
        "sfr_grid": np.array([[0.0, 1.0, 1.0]], dtype=float),
        "active_grid": np.array([[False, True, True]], dtype=bool),
    }

    result = compute_mzr_birth_metallicity(
        **history,
        parameters=MZRBirthMetallicityParameters(
            relation="fire2_highz",
            returned_fraction=0.0,
            scatter_dex=0.0,
            stellar_mass_floor_msun=1.0e6,
        ),
        random_seed=1,
    )

    np.testing.assert_allclose(result.stellar_mass_msun_grid[0], [0.0, 1.0e8, 2.0e8])
    expected_oh12 = fire2_highz_mzr_oh12(np.log10(np.array([1.0e8, 2.0e8])))
    expected_zsun = 10.0 ** (expected_oh12 - 8.69)
    np.testing.assert_allclose(result.birth_metallicity_zsun_grid[0, 1:], expected_zsun)
    np.testing.assert_array_equal(result.active_grid, history["active_grid"])


def test_mzr_vectorized_history_preserves_seeded_scatter_and_active_only_validation() -> None:
    t_grid = np.array(
        [
            [0.1, 0.2, 0.2, 0.4, 0.5],
            [0.05, 0.05, 0.15, 0.3, 0.3],
        ],
        dtype=float,
    )
    active = np.array(
        [
            [False, True, True, False, True],
            [True, False, True, True, False],
        ],
        dtype=bool,
    )
    result = compute_mzr_birth_metallicity(
        t_grid_gyr=t_grid,
        z_grid=np.array(
            [
                [16.0, 14.0, 13.0, 10.0, 8.0],
                [18.0, 17.0, 14.0, 11.0, 9.0],
            ],
            dtype=float,
        ),
        sfr_grid=np.array(
            [
                [np.nan, 1.0, 2.0, -99.0, 0.5],
                [0.25, np.nan, 1.5, 2.0, -np.inf],
            ],
            dtype=float,
        ),
        active_grid=active,
        parameters=MZRBirthMetallicityParameters(
            relation="fire2_highz",
            returned_fraction=0.37,
            scatter_dex=0.23,
            stellar_mass_floor_msun=1.0e6,
        ),
        random_seed=12345,
    )

    np.testing.assert_array_equal(
        result.stellar_mass_msun_grid,
        np.array(
            [
                [0.0, 6.3e7, 6.3e7, 6.3e7, 9.45e7],
                [0.0, 0.0, 9.45e7, 2.835e8, 2.835e8],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_array_equal(
        result.birth_metallicity_zsun_grid,
        np.array(
            [
                [0.0, 0.018124926116579773, 0.07523389535089041, 0.0, 0.02822653503987153],
                [0.007250870620796875, 0.0, 0.04301112563548344, 0.04539846443042604, 0.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_array_equal(result.active_grid, active)


@pytest.mark.parametrize("invalid_sfr", [-1.0, np.nan, np.inf])
def test_mzr_rejects_invalid_sfr_at_active_source_times(invalid_sfr: float) -> None:
    with pytest.raises(ValueError, match="sfr_grid must be finite and non-negative"):
        compute_mzr_birth_metallicity(
            t_grid_gyr=np.array([[0.0, 0.1]], dtype=float),
            z_grid=np.array([[12.0, 10.0]], dtype=float),
            sfr_grid=np.array([[0.0, invalid_sfr]], dtype=float),
            active_grid=np.array([[False, True]], dtype=bool),
        )


def test_mzr_rejects_invalid_returned_fraction_in_shared_stellar_mass_integrator() -> None:
    with pytest.raises(ValueError, match="returned_fraction must lie in"):
        compute_mzr_birth_metallicity(
            t_grid_gyr=np.array([[0.0, 0.1]], dtype=float),
            z_grid=np.array([[12.0, 10.0]], dtype=float),
            sfr_grid=np.array([[0.0, 1.0]], dtype=float),
            active_grid=np.array([[False, True]], dtype=bool),
            parameters=MZRBirthMetallicityParameters(returned_fraction=1.0),
        )


def test_regulator_still_rejects_nonfinite_sfr_at_inactive_source_times() -> None:
    with pytest.raises(ValueError, match="sfr_grid must contain finite values"):
        compute_regulator_metallicity(
            t_grid_gyr=np.array([[0.0, 0.1]], dtype=float),
            z_grid=np.array([[12.0, 10.0]], dtype=float),
            mh_grid=np.array([[1.0e9, 2.0e9]], dtype=float),
            sfr_grid=np.array([[np.nan, 1.0]], dtype=float),
            active_grid=np.array([[False, True]], dtype=bool),
            cosmology=Cosmology(),
        )


def test_regulator_metallicity_uses_cumulative_stellar_and_halo_gas_mass() -> None:
    history = {
        "t_grid_gyr": np.array([[0.0, 0.1]], dtype=float),
        "z_grid": np.array([[12.0, 12.0]], dtype=float),
        "mh_grid": np.array([[1.0e9, 1.0e9]], dtype=float),
        "sfr_grid": np.array([[0.0, 1.0]], dtype=float),
        "active_grid": np.array([[False, True]], dtype=bool),
    }
    params = RegulatorMetallicityParameters(
        gas_fraction_norm=0.2,
        returned_fraction=0.4,
        metal_yield=0.02,
        inflow_metallicity_zsun=0.0,
        metal_loading_norm=3.0,
        metal_loading_mass_slope=0.0,
        metallicity_scatter_dex=0.0,
    )

    result = compute_regulator_metallicity(
        **history,
        cosmology=Cosmology(omega_m=0.4, omega_b=0.04, omega_lambda=0.6),
        parameters=params,
        random_seed=1,
    )

    expected_stellar_mass = (1.0 - 0.4) * 1.0 * 0.1 * 1.0e9
    expected_gas_mass = 0.2 * 0.1 * 1.0e9
    denominator = 1.0 + expected_gas_mass / expected_stellar_mass + 3.0 / (1.0 - 0.4)
    expected_zsun = (0.02 / denominator) / 0.0142

    np.testing.assert_allclose(result.stellar_mass_msun_grid[0, 1], expected_stellar_mass)
    np.testing.assert_allclose(result.gas_mass_grid[0, 1], expected_gas_mass)
    np.testing.assert_allclose(result.gas_metallicity_zsun_grid[0, 1], expected_zsun)
    np.testing.assert_allclose(result.birth_metallicity_zsun_grid[0, 1], expected_zsun)
    np.testing.assert_allclose(result.metal_loading_grid[0, 1], 3.0)


def test_regulator_metallicity_rejects_unphysical_gas_fraction() -> None:
    history = _toy_history()

    with pytest.raises(ValueError, match="gas fraction"):
        compute_regulator_metallicity(
            t_grid_gyr=history["t_grid_gyr"],
            z_grid=history["z_grid"],
            mh_grid=history["mh_grid"],
            sfr_grid=history["sfr_grid"],
            active_grid=history["active_grid"],
            cosmology=Cosmology(),
            parameters=RegulatorMetallicityParameters(gas_fraction_norm=1.2),
            random_seed=1,
        )


def test_pipeline_requires_metallicity_when_topheavy_metallicity_gate_is_enabled() -> None:
    with pytest.raises(ValueError, match="birth metallicity source"):
        run_halo_uv_pipeline(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            z_start_max=10.0,
            n_grid=4,
            random_seeds=derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0),
            workers=1,
            imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
            imf_transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        )


def test_canonical_pipeline_metadata_does_not_report_topheavy_metallicity_gate() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=10.0,
        n_grid=4,
        random_seeds=derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0),
        workers=1,
        imf_mode="canonical",
    )

    assert result.metadata["metallicity_topheavy_gate_applied"] is False


def test_pipeline_applies_topheavy_metallicity_gate_to_birth_metallicity() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=8,
        random_seeds=derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0),
        workers=1,
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        imf_transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
        regulator_metallicity_parameters=RegulatorMetallicityParameters(
            gas_fraction_norm=0.2,
            metal_loading_norm=0.0,
            metal_loading_mass_slope=0.0,
            metallicity_scatter_dex=0.0,
        ),
    )

    assert result.birth_metallicity_zsun_grid is not None
    assert np.any(result.imf_topheavy_source_grid)
    assert np.all(result.birth_metallicity_zsun_grid[result.imf_topheavy_source_grid] <= 0.05)
    assert result.metadata["imf_transition_parameters"]["metallicity_topheavy_max_zsun"] == 0.05


def test_pipeline_applies_mzr_birth_metallicity_gate() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=8,
        random_seeds=derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0),
        workers=1,
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        imf_transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
        mzr_metallicity_parameters=MZRBirthMetallicityParameters(
            relation="fire2_highz",
            scatter_dex=0.0,
            stellar_mass_floor_msun=1.0e6,
        ),
    )

    assert result.birth_metallicity_zsun_grid is not None
    assert result.gas_metallicity_zsun_grid is None
    assert result.metadata["metallicity_source"] == "mzr"
    assert result.metadata["mzr_metallicity_parameters"]["relation"] == "fire2_highz"
    assert np.any(result.imf_topheavy_source_grid)
    assert np.all(result.birth_metallicity_zsun_grid[result.imf_topheavy_source_grid] <= 0.05)


def test_pipeline_applies_regulator_birth_metallicity_gate() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        z_start_max=12.0,
        n_grid=8,
        random_seeds=derive_pipeline_random_seeds(101, redshift=6.0, mass_index=0),
        workers=1,
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        imf_transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=0.2,
        ),
        regulator_metallicity_parameters=RegulatorMetallicityParameters(
            gas_fraction_norm=0.2,
            metal_loading_norm=3.0,
            metal_loading_mass_slope=0.0,
            metallicity_scatter_dex=0.0,
        ),
    )

    assert result.birth_metallicity_zsun_grid is not None
    assert result.gas_metallicity_zsun_grid is not None
    assert result.metadata["metallicity_source"] == "regulator"
    assert result.metadata["regulator_metallicity_enabled"] is True
    assert result.metadata["regulator_metallicity_parameters"]["gas_fraction_norm"] == 0.2
    assert np.any(result.imf_topheavy_source_grid)
    assert np.all(result.birth_metallicity_zsun_grid[result.imf_topheavy_source_grid] <= 0.2)


def test_hmf_sampling_records_regulator_metallicity_metadata_when_enabled() -> None:
    result = sample_uvlf_from_hmf(
        z_obs=6.0,
        cosmology=Cosmology(),
        N_mass=1,
        n_tracks=2,
        base_seed=303,
        bins=np.array([-25.0, -15.0]),
        logM_min=9.0,
        logM_max=9.2,
        z_start_max=10.0,
        n_grid=8,
        pipeline_workers=1,
        regulator_metallicity_parameters=RegulatorMetallicityParameters(
            gas_fraction_norm=0.2,
            metal_loading_norm=3.0,
            metal_loading_mass_slope=0.0,
        ),
    )

    assert result.metadata["regulator_metallicity_enabled"] is True
    assert result.metadata["metallicity_source"] == "regulator"
    assert result.metadata["regulator_metallicity_parameters"]["metal_loading_norm"] == 3.0
    assert np.isfinite(result.metadata["final_gas_metallicity_zsun_median"])


def test_hmf_sampling_requires_metallicity_for_metallicity_gated_topheavy() -> None:
    with pytest.raises(ValueError, match="birth metallicity source"):
        sample_uvlf_from_hmf(
            z_obs=6.0,
            cosmology=Cosmology(),
            N_mass=1,
            n_tracks=1,
            base_seed=303,
            bins=np.array([-25.0, -15.0]),
            logM_min=9.0,
            logM_max=9.2,
            z_start_max=10.0,
            n_grid=4,
            pipeline_workers=1,
            imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
            imf_transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        )


def test_production_config_archives_metallicity_models() -> None:
    from auroralf import UVLFRunConfig

    config = UVLFRunConfig.from_toml(
        PROJECT_ROOT / "configs" / "uvlf" / "production.toml"
    )
    assert config.star_formation.enable_archived_metallicity is False
    assert config.star_formation.metallicity_source == "none"
    assert config.star_formation.mzr is None
    assert config.star_formation.regulator is None
    assert config.stellar_population.birth_metallicity_topheavy_max_zsun == pytest.approx(0.05)
