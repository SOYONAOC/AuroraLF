from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auroralf.chemistry import MetalEnrichmentParameters, evolve_stochastic_metallicity
from auroralf.uvlf.imf import IMF_MODE_Z_GATED_MILD_TOPHEAVY, IMFTransitionParameters
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf
from auroralf.uvlf.pipeline import run_halo_uv_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _toy_history() -> dict[str, np.ndarray]:
    return {
        "t_grid_gyr": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float),
        "z_grid": np.array([[14.0, 12.0, 10.0, 8.0]], dtype=float),
        "mh_grid": np.array([[1.0e9, 2.0e9, 4.0e9, 8.0e9]], dtype=float),
        "dmhdt_grid": np.array([[0.0, 1.0e10, 2.0e10, 4.0e10]], dtype=float),
        "sfr_grid": np.array([[0.0, 0.02, 0.04, 0.08]], dtype=float),
        "active_grid": np.array([[False, True, True, True]], dtype=bool),
    }


def test_stochastic_metallicity_is_reproducible_for_fixed_seed() -> None:
    history = _toy_history()
    params = MetalEnrichmentParameters(
        gas_fraction_of_baryons=0.5,
        metal_yield=0.02,
        yield_scatter_dex=0.25,
        mass_loading_scatter_dex=0.35,
        birth_metallicity_scatter_dex=0.15,
    )

    first = evolve_stochastic_metallicity(**history, baryon_fraction=0.157, parameters=params, random_seed=11)
    second = evolve_stochastic_metallicity(**history, baryon_fraction=0.157, parameters=params, random_seed=11)

    np.testing.assert_allclose(first.gas_metallicity_zsun_grid, second.gas_metallicity_zsun_grid)
    np.testing.assert_allclose(first.birth_metallicity_zsun_grid, second.birth_metallicity_zsun_grid)
    np.testing.assert_allclose(first.mass_loading_grid, second.mass_loading_grid)


def test_stochastic_metallicity_changes_with_seed() -> None:
    history = _toy_history()
    params = MetalEnrichmentParameters(
        gas_fraction_of_baryons=0.5,
        metal_yield=0.02,
        yield_scatter_dex=0.25,
        mass_loading_scatter_dex=0.35,
        birth_metallicity_scatter_dex=0.15,
    )

    first = evolve_stochastic_metallicity(**history, baryon_fraction=0.157, parameters=params, random_seed=11)
    second = evolve_stochastic_metallicity(**history, baryon_fraction=0.157, parameters=params, random_seed=12)

    assert not np.allclose(first.gas_metallicity_zsun_grid, second.gas_metallicity_zsun_grid)
    assert not np.allclose(first.birth_metallicity_zsun_grid, second.birth_metallicity_zsun_grid)


def test_metallicity_remains_finite_and_non_negative_for_active_steps() -> None:
    history = _toy_history()

    result = evolve_stochastic_metallicity(
        **history,
        baryon_fraction=0.157,
        parameters=MetalEnrichmentParameters(gas_fraction_of_baryons=0.5),
        random_seed=5,
    )

    active = history["active_grid"]
    assert np.all(np.isfinite(result.gas_metallicity_zsun_grid[active]))
    assert np.all(result.gas_metallicity_zsun_grid[active] >= 0.0)
    assert np.all(np.isfinite(result.birth_metallicity_zsun_grid[active]))
    assert np.all(result.birth_metallicity_zsun_grid[active] >= 0.0)


def test_gas_metallicity_records_post_step_enrichment_but_birth_metallicity_is_pre_step() -> None:
    history = {
        "t_grid_gyr": np.array([[0.1, 0.2]], dtype=float),
        "z_grid": np.array([[12.0, 10.0]], dtype=float),
        "mh_grid": np.array([[1.0e9, 1.0e9]], dtype=float),
        "dmhdt_grid": np.array([[0.0, 0.0]], dtype=float),
        "sfr_grid": np.array([[0.0, 0.1]], dtype=float),
        "active_grid": np.array([[False, True]], dtype=bool),
    }

    result = evolve_stochastic_metallicity(
        **history,
        baryon_fraction=0.157,
        parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            mass_loading_norm=0.0,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        random_seed=1,
    )

    assert result.birth_metallicity_zsun_grid[0, 1] == 0.0
    assert result.gas_metallicity_zsun_grid[0, 1] > 0.0


def test_topheavy_source_grid_multiplies_effective_yield_only_on_flagged_steps() -> None:
    history = _toy_history()
    topheavy_source_grid = np.array([[False, False, True, False]], dtype=bool)

    result = evolve_stochastic_metallicity(
        **history,
        baryon_fraction=0.157,
        parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            metal_yield=0.02,
            topheavy_yield_multiplier=3.0,
            mass_loading_norm=0.0,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        random_seed=1,
        topheavy_source_grid=topheavy_source_grid,
    )

    np.testing.assert_allclose(result.effective_yield_grid[0, 1:], [0.02, 0.06, 0.02])


def test_topheavy_birth_metallicity_gate_uses_pre_step_metallicity() -> None:
    history = {
        "t_grid_gyr": np.array([[0.1, 0.2, 0.3]], dtype=float),
        "z_grid": np.array([[12.0, 12.0, 12.0]], dtype=float),
        "mh_grid": np.array([[1.0e9, 1.0e9, 1.0e9]], dtype=float),
        "dmhdt_grid": np.array([[0.0, 0.0, 0.0]], dtype=float),
        "sfr_grid": np.array([[0.0, 0.1, 0.1]], dtype=float),
        "active_grid": np.array([[False, True, True]], dtype=bool),
    }
    candidate_topheavy = np.array([[False, True, True]], dtype=bool)

    result = evolve_stochastic_metallicity(
        **history,
        baryon_fraction=0.157,
        parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            metal_yield=0.02,
            topheavy_yield_multiplier=3.0,
            mass_loading_norm=0.0,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        random_seed=1,
        topheavy_source_grid=candidate_topheavy,
        topheavy_birth_metallicity_max_zsun=0.05,
    )

    np.testing.assert_array_equal(result.topheavy_source_grid, np.array([[False, True, False]], dtype=bool))
    np.testing.assert_allclose(result.effective_yield_grid[0, 1:], [0.06, 0.02])
    assert result.birth_metallicity_zsun_grid[0, 1] == 0.0
    assert result.gas_metallicity_zsun_grid[0, 1] > 0.05
    assert result.birth_metallicity_zsun_grid[0, 2] > 0.05


def test_topheavy_birth_metallicity_gate_requires_candidate_grid() -> None:
    history = _toy_history()

    with pytest.raises(ValueError, match="topheavy_source_grid"):
        evolve_stochastic_metallicity(
            **history,
            baryon_fraction=0.157,
            parameters=MetalEnrichmentParameters(gas_fraction_of_baryons=0.5),
            random_seed=1,
            topheavy_birth_metallicity_max_zsun=0.05,
        )


def test_pipeline_records_stochastic_metallicity_when_enabled() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=10.0,
        n_grid=8,
        random_seed=101,
        workers=1,
        metal_enrichment_parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        metallicity_random_seed=202,
    )

    assert result.gas_metallicity_zsun_grid is not None
    assert result.birth_metallicity_zsun_grid is not None
    assert result.gas_metallicity_zsun_grid.shape == result.active_grid.shape
    assert result.birth_metallicity_zsun_grid.shape == result.active_grid.shape
    assert result.metadata["stochastic_metallicity_enabled"] is True
    assert result.metadata["metallicity_random_seed"] == 202
    assert result.metadata["metal_enrichment_parameters"]["gas_fraction_of_baryons"] == 0.5


def test_pipeline_requires_metallicity_when_topheavy_metallicity_gate_is_enabled() -> None:
    with pytest.raises(ValueError, match="metal_enrichment_parameters"):
        run_halo_uv_pipeline(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            z_start_max=10.0,
            n_grid=4,
            random_seed=101,
            workers=1,
            imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
            imf_transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        )


def test_canonical_pipeline_metadata_does_not_report_topheavy_metallicity_gate() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=10.0,
        n_grid=4,
        random_seed=101,
        workers=1,
        imf_mode="canonical",
    )

    assert result.metadata["metallicity_topheavy_gate_applied"] is False


def test_pipeline_applies_topheavy_metallicity_gate_to_birth_metallicity() -> None:
    result = run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=12.0,
        n_grid=8,
        random_seed=101,
        workers=1,
        imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        imf_transition_parameters=IMFTransitionParameters(
            z_topheavy_min=10.0,
            metallicity_topheavy_max_zsun=0.05,
        ),
        metal_enrichment_parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            mass_loading_norm=0.0,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        metallicity_random_seed=202,
    )

    assert result.birth_metallicity_zsun_grid is not None
    assert np.any(result.imf_topheavy_source_grid)
    assert np.all(result.birth_metallicity_zsun_grid[result.imf_topheavy_source_grid] <= 0.05)
    assert result.metadata["imf_transition_parameters"]["metallicity_topheavy_max_zsun"] == 0.05


def test_hmf_sampling_records_stochastic_metallicity_metadata_when_enabled() -> None:
    result = sample_uvlf_from_hmf(
        z_obs=6.0,
        N_mass=1,
        n_tracks=2,
        random_seed=303,
        bins=np.array([-25.0, -15.0]),
        logM_min=9.0,
        logM_max=9.2,
        z_start_max=10.0,
        n_grid=8,
        pipeline_workers=1,
        metal_enrichment_parameters=MetalEnrichmentParameters(
            gas_fraction_of_baryons=0.5,
            yield_scatter_dex=0.0,
            mass_loading_scatter_dex=0.0,
            birth_metallicity_scatter_dex=0.0,
        ),
        metallicity_random_seed=404,
    )

    assert result.metadata["stochastic_metallicity_enabled"] is True
    assert result.metadata["metallicity_random_seed"] == 404
    assert result.metadata["final_gas_metallicity_zsun_median_by_mass"].shape == (1,)
    assert np.isfinite(result.metadata["final_gas_metallicity_zsun_median"])


def test_hmf_sampling_requires_metallicity_for_metallicity_gated_topheavy() -> None:
    with pytest.raises(ValueError, match="metal_enrichment_parameters"):
        sample_uvlf_from_hmf(
            z_obs=6.0,
            N_mass=1,
            n_tracks=1,
            random_seed=303,
            bins=np.array([-25.0, -15.0]),
            logM_min=9.0,
            logM_max=9.2,
            z_start_max=10.0,
            n_grid=4,
            pipeline_workers=1,
            imf_mode=IMF_MODE_Z_GATED_MILD_TOPHEAVY,
            imf_transition_parameters=IMFTransitionParameters(metallicity_topheavy_max_zsun=0.05),
        )


def test_run_script_help_exposes_topheavy_yield_multiplier() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run/run_uvlf_compare_imf_no_delay_all_z.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--metal-topheavy-yield-multiplier" in completed.stdout
    assert "--metallicity-topheavy-max-zsun" in completed.stdout
