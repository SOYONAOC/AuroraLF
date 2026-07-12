from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from auroralf.seeding import (
    PipelineRandomSeeds,
    derive_hmf_mass_seed,
    derive_pipeline_random_seeds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"
UINT64_MAX = 2**64 - 1


def test_pipeline_seed_derivation_is_stable_and_component_separated() -> None:
    first = derive_pipeline_random_seeds(42, redshift=10.0, mass_index=7)
    second = derive_pipeline_random_seeds(42, redshift=np.float64(10.0), mass_index=7)

    assert first == second
    assert isinstance(first.mah, int)
    assert isinstance(first.metallicity, int)
    assert isinstance(first.burst, int)
    assert len({first.mah, first.metallicity, first.burst}) == 3
    assert derive_pipeline_random_seeds(42, redshift=10.5, mass_index=7) != first
    assert derive_pipeline_random_seeds(42, redshift=10.0, mass_index=8) != first
    assert derive_hmf_mass_seed(42, 10.0) != first.mah


def test_signed_zero_redshifts_have_one_canonical_seed_key() -> None:
    assert derive_hmf_mass_seed(42, -0.0) == derive_hmf_mass_seed(42, 0.0)
    assert derive_pipeline_random_seeds(
        42,
        redshift=-0.0,
        mass_index=3,
    ) == derive_pipeline_random_seeds(42, redshift=0.0, mass_index=3)


def test_seed_values_are_bounded_by_uint64_serialization() -> None:
    seeds = PipelineRandomSeeds(UINT64_MAX, UINT64_MAX, UINT64_MAX)
    assert seeds.as_metadata() == {
        "mah": UINT64_MAX,
        "metallicity": UINT64_MAX,
        "burst": UINT64_MAX,
    }
    derived = derive_pipeline_random_seeds(
        UINT64_MAX,
        redshift=10.0,
        mass_index=0,
    )
    assert all(0 <= value <= UINT64_MAX for value in derived.as_metadata().values())
    assert 0 <= derive_hmf_mass_seed(UINT64_MAX, 10.0) <= UINT64_MAX

    with pytest.raises(ValueError, match="uint64"):
        PipelineRandomSeeds(UINT64_MAX + 1, 2, 3)
    with pytest.raises(ValueError, match="uint64"):
        derive_hmf_mass_seed(UINT64_MAX + 1, 10.0)


@pytest.mark.parametrize(
    ("base_seed", "redshift", "mass_index", "error"),
    [
        (True, 10.0, 0, TypeError),
        (1.5, 10.0, 0, TypeError),
        (-1, 10.0, 0, ValueError),
        (1, True, 0, TypeError),
        (1, np.inf, 0, ValueError),
        (1, -0.1, 0, ValueError),
        (1, 10.0, True, TypeError),
        (1, 10.0, 1.5, TypeError),
        (1, 10.0, -1, ValueError),
    ],
)
def test_pipeline_seed_derivation_rejects_invalid_keys(
    base_seed: object,
    redshift: object,
    mass_index: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        derive_pipeline_random_seeds(  # type: ignore[arg-type]
            base_seed,
            redshift=redshift,
            mass_index=mass_index,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mah": True, "metallicity": 2, "burst": 3},
        {"mah": 1.0, "metallicity": 2, "burst": 3},
        {"mah": -1, "metallicity": 2, "burst": 3},
    ],
)
def test_pipeline_random_seeds_validate_fields(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PipelineRandomSeeds(**kwargs)  # type: ignore[arg-type]


def test_seed_mapping_is_independent_of_mode_and_redshift_iteration_order() -> None:
    def build(redshifts: list[float], modes: list[str]) -> dict[tuple[float, int, str], int]:
        mapping: dict[tuple[float, int, str], int] = {}
        for redshift in redshifts:
            for mode in modes:
                seeds = derive_pipeline_random_seeds(1234, redshift=redshift, mass_index=4)
                mapping[(redshift, 4, "mah")] = seeds.mah
                mapping[(redshift, 4, "metallicity")] = seeds.metallicity
                mapping[(redshift, 4, "burst")] = seeds.burst
        return mapping

    assert build([6.0, 10.0, 14.5], ["canonical", "topheavy"]) == build(
        [14.5, 10.0, 6.0],
        ["topheavy", "canonical"],
    )


def test_v2_public_signatures_expose_only_seed_bundle_or_base_seed() -> None:
    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    pipeline_parameters = inspect.signature(run_halo_uv_pipeline).parameters
    assert pipeline_parameters["random_seeds"].kind is inspect.Parameter.KEYWORD_ONLY
    assert pipeline_parameters["random_seeds"].default is inspect.Parameter.empty
    assert "random_seed" not in pipeline_parameters
    assert "metallicity_random_seed" not in pipeline_parameters
    assert "burst_scatter_random_seed" not in pipeline_parameters

    hmf_parameters = inspect.signature(sample_uvlf_from_hmf).parameters
    assert hmf_parameters["base_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert hmf_parameters["base_seed"].default is inspect.Parameter.empty
    assert "random_seed" not in hmf_parameters
    assert "metallicity_random_seed" not in hmf_parameters
    assert "burst_scatter_random_seed" not in hmf_parameters


def test_hmf_rejects_bool_redshift_before_float_canonicalization() -> None:
    from auroralf.mah import Cosmology
    from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf

    with pytest.raises(TypeError, match="redshift.*bool"):
        sample_uvlf_from_hmf(
            z_obs=True,  # type: ignore[arg-type]
            cosmology=Cosmology(),
            base_seed=42,
            N_mass=1,
            n_tracks=1,
        )


def test_pipeline_routes_each_component_seed_to_its_downstream_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.pipeline as pipeline
    from auroralf.chemistry import MZRBirthMetallicityParameters
    from auroralf.mah import Cosmology

    seeds = PipelineRandomSeeds(mah=101, metallicity=202, burst=303)
    captured: dict[str, int] = {}
    real_generate = pipeline.generate_halo_histories
    real_mzr = pipeline.compute_mzr_birth_metallicity
    real_burst = pipeline._apply_burst_scatter_to_sfr_grid

    def generate_spy(**kwargs: object):
        captured["mah"] = int(kwargs["random_seed"])
        return real_generate(**kwargs)

    def mzr_spy(**kwargs: object):
        captured["metallicity"] = int(kwargs["random_seed"])
        return real_mzr(**kwargs)

    def burst_spy(**kwargs: object):
        captured["burst"] = int(kwargs["random_seed"])
        return real_burst(**kwargs)

    monkeypatch.setattr(pipeline, "generate_halo_histories", generate_spy)
    monkeypatch.setattr(pipeline, "compute_mzr_birth_metallicity", mzr_spy)
    monkeypatch.setattr(pipeline, "_apply_burst_scatter_to_sfr_grid", burst_spy)

    result = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        random_seeds=seeds,
        z_start_max=8.0,
        n_grid=4,
        workers=1,
        mzr_metallicity_parameters=MZRBirthMetallicityParameters(scatter_dex=0.1),
        burst_scatter_dex=0.1,
    )

    assert captured == seeds.as_metadata()
    assert result.metadata["random_seeds"] == seeds.as_metadata()


def test_pipeline_rejects_legacy_seed_keywords() -> None:
    from auroralf.mah import Cosmology
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    legacy_kwargs = {"random_seed": 1}
    with pytest.raises(TypeError, match="random_seed"):
        run_halo_uv_pipeline(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=Cosmology(),
            random_seeds=PipelineRandomSeeds(1, 2, 3),
            **legacy_kwargs,  # type: ignore[arg-type]
        )


def test_hmf_mass_seed_bundles_are_paired_across_imf_mode_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.hmf_sampling as hmf_sampling
    from auroralf.mah import Cosmology
    from auroralf.uvlf.imf import IMFTransitionParameters

    captured: list[PipelineRandomSeeds] = []

    def fake_dndm(halo_mass_msun: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return np.ones_like(np.asarray(halo_mass_msun, dtype=float))

    def fake_worker(args: tuple[object, ...]) -> tuple[object, ...]:
        captured.append(next(item for item in args if isinstance(item, PipelineRandomSeeds)))
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

    monkeypatch.setattr(hmf_sampling, "compute_halo_mass_function_dndm", fake_dndm)
    monkeypatch.setattr(hmf_sampling, "_run_single_mass_sample", fake_worker)

    def collect(modes: list[str]) -> dict[str, tuple[PipelineRandomSeeds, ...]]:
        mapping: dict[str, tuple[PipelineRandomSeeds, ...]] = {}
        for mode in modes:
            captured.clear()
            hmf_sampling.sample_uvlf_from_hmf(
                z_obs=10.0,
                cosmology=Cosmology(),
                base_seed=77,
                N_mass=3,
                n_tracks=1,
                bins=np.array([-100.0, 100.0]),
                pipeline_workers=1,
                imf_mode=mode,
                imf_transition_parameters=IMFTransitionParameters(
                    metallicity_topheavy_max_zsun=None
                ),
            )
            mapping[mode] = tuple(captured)
        return mapping

    forward = collect(["canonical", "z10_mild_topheavy"])
    reverse = collect(["z10_mild_topheavy", "canonical"])
    assert forward == reverse
    assert forward["canonical"] == forward["z10_mild_topheavy"]
    assert len(set(forward["canonical"])) == 3


def test_real_pipeline_draws_are_paired_across_mode_and_mode_order() -> None:
    from auroralf.chemistry import MZRBirthMetallicityParameters
    from auroralf.mah import Cosmology
    from auroralf.uvlf.imf import IMFTransitionParameters
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    cosmology = Cosmology()
    random_seeds = derive_pipeline_random_seeds(9876, redshift=6.0, mass_index=2)
    common = {
        "n_tracks": 2,
        "z_final": 6.0,
        "Mh_final": 1.0e10,
        "cosmology": cosmology,
        "random_seeds": random_seeds,
        "z_start_max": 8.0,
        "n_grid": 4,
        "workers": 1,
        "mzr_metallicity_parameters": MZRBirthMetallicityParameters(scatter_dex=0.2),
        "imf_transition_parameters": IMFTransitionParameters(
            metallicity_topheavy_max_zsun=None
        ),
    }

    def run_in_order(modes: list[str]):
        return {
            mode: run_halo_uv_pipeline(**common, imf_mode=mode)
            for mode in modes
        }

    forward = run_in_order(["canonical", "z10_mild_topheavy"])
    reverse = run_in_order(["z10_mild_topheavy", "canonical"])

    reference = forward["canonical"]
    for result in (
        forward["z10_mild_topheavy"],
        reverse["canonical"],
        reverse["z10_mild_topheavy"],
    ):
        assert result.histories.tracks.keys() == reference.histories.tracks.keys()
        for column in reference.histories.tracks:
            np.testing.assert_array_equal(
                result.histories.tracks[column],
                reference.histories.tracks[column],
            )
        assert result.birth_metallicity_zsun_grid is not None
        assert reference.birth_metallicity_zsun_grid is not None
        np.testing.assert_array_equal(
            result.birth_metallicity_zsun_grid,
            reference.birth_metallicity_zsun_grid,
            strict=True,
        )


def test_production_runner_records_the_same_explicit_base_seed_per_redshift() -> None:
    from auroralf import UVLFRunConfig

    config = UVLFRunConfig.from_toml(PRODUCTION_CONFIG)
    assert config.base_seed == 42
    assert all(
        derive_pipeline_random_seeds(config.base_seed, redshift=redshift, mass_index=0)
        == derive_pipeline_random_seeds(42, redshift=redshift, mass_index=0)
        for redshift in config.redshifts
    )
