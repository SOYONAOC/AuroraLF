from __future__ import annotations

import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.seeding import derive_pipeline_random_seeds


def test_mcbride_preserves_negative_raw_rate_and_records_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    import auroralf.mah.generator as generator

    def fixed_parameters(*, mass_ref, size, sampler, rng, pilot_samples):
        del mass_ref, sampler, rng, pilot_samples
        return np.column_stack((np.ones(size), np.zeros(size))), None

    monkeypatch.setattr(generator, "sample_parameters", fixed_parameters)
    result = generator.generate_halo_histories(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e9,
        cosmology=Cosmology(),
        z_start_max=8.0,
        M_min=1.0,
        custom_grid=np.array([8.0, 7.0, 6.0]),
        time_grid_mode="custom",
        random_seed=7,
    )

    tracks = result.tracks
    raw = np.asarray(tracks["dMh_dt_raw"], dtype=float)
    effective = np.asarray(tracks["dMh_dt_sfr"], dtype=float)
    clipped = np.asarray(tracks["dMh_dt_clipped"], dtype=bool)

    assert "dMh_dt" not in tracks
    assert np.all(raw < 0.0)
    np.testing.assert_array_equal(effective, np.zeros_like(raw))
    np.testing.assert_array_equal(clipped, raw < 0.0)
    assert result.metadata["negative_dmhdt_clip_count"] == int(np.count_nonzero(clipped))
    assert result.metadata["negative_dmhdt_total_count"] == raw.size
    assert result.metadata["negative_dmhdt_clip_fraction"] == pytest.approx(1.0)


def _manual_tracks(*, raw: np.ndarray, effective: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(raw, dtype=float)
    effective = np.asarray(effective, dtype=float)
    n_rows = raw.size
    return {
        "halo_id": np.zeros(n_rows, dtype=int),
        "step": np.arange(n_rows, dtype=int),
        "z": np.linspace(8.0, 6.0, n_rows),
        "t_gyr": np.linspace(0.64, 0.93, n_rows),
        "Mh": np.full(n_rows, 1.0e10),
        "dMh_dt_raw": raw,
        "dMh_dt_sfr": effective,
        "dMh_dt_clipped": raw < 0.0,
        "active_flag": np.ones(n_rows, dtype=bool),
    }


def test_sfr_uses_nonnegative_effective_rate_and_preserves_rate_diagnostics() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(
        raw=np.array([-1.0e9, -2.0e9, 3.0e9]),
        effective=np.array([0.0, 0.0, 3.0e9]),
    )
    result = compute_sfr_from_tracks(tracks, cosmology=Cosmology())

    assert np.all(np.asarray(result["SFR"], dtype=float) >= 0.0)
    assert "dMh_dt" not in result
    assert "dMh_dt_src" not in result
    np.testing.assert_array_equal(result["dMh_dt_raw"], tracks["dMh_dt_raw"])
    np.testing.assert_array_equal(result["dMh_dt_sfr"], tracks["dMh_dt_sfr"])
    np.testing.assert_array_equal(result["dMh_dt_clipped"], tracks["dMh_dt_clipped"])
    assert "dMh_dt_sfr_src" in result


def test_sfr_rejects_negative_effective_accretion_rate() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([-1.0e9, 1.0e9]), effective=np.array([-1.0, 1.0e9]))
    with pytest.raises(ValueError, match="dMh_dt_sfr.*non-negative"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_sfr_rejects_legacy_only_accretion_rate_column() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([1.0e9, 1.0e9]), effective=np.array([1.0e9, 1.0e9]))
    tracks["dMh_dt"] = tracks.pop("dMh_dt_sfr")
    tracks.pop("dMh_dt_raw")
    tracks.pop("dMh_dt_clipped")
    with pytest.raises(KeyError, match="dMh_dt_sfr"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dMh_dt_raw", np.array([np.nan, 1.0e9])),
        ("dMh_dt_raw", np.array([np.inf, 1.0e9])),
        ("dMh_dt_raw", np.array([-np.inf, 1.0e9])),
        ("dMh_dt_sfr", np.array([np.nan, 1.0e9])),
        ("dMh_dt_sfr", np.array([np.inf, 1.0e9])),
        ("dMh_dt_sfr", np.array([-np.inf, 1.0e9])),
    ],
)
def test_sfr_rejects_nonfinite_accretion_rate_columns(field: str, value: np.ndarray) -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([1.0e9, 1.0e9]), effective=np.array([1.0e9, 1.0e9]))
    tracks[field] = value
    with pytest.raises(ValueError, match=field):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


@pytest.mark.parametrize("field", ["dMh_dt_raw", "dMh_dt_sfr", "dMh_dt_clipped"])
def test_sfr_rejects_non_1d_accretion_rate_columns(field: str) -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([1.0e9, 1.0e9]), effective=np.array([1.0e9, 1.0e9]))
    tracks[field] = np.asarray(tracks[field]).reshape(1, 2)
    with pytest.raises(ValueError, match=f"{field}.*1D"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_sfr_rejects_non_boolean_clip_column() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([-1.0e9, 1.0e9]), effective=np.array([0.0, 1.0e9]))
    tracks["dMh_dt_clipped"] = np.array([1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="dMh_dt_clipped.*bool"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_sfr_rejects_effective_rate_inconsistent_with_raw_rate() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([-1.0e9, 1.0e9]), effective=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="dMh_dt_sfr.*maximum"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_sfr_rejects_clip_mask_inconsistent_with_raw_rate() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([-1.0e9, 1.0e9]), effective=np.array([0.0, 1.0e9]))
    tracks["dMh_dt_clipped"] = np.array([False, False], dtype=bool)
    with pytest.raises(ValueError, match="dMh_dt_clipped.*dMh_dt_raw"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_sfr_rejects_empty_tracks() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(raw=np.array([]), effective=np.array([]))
    with pytest.raises(ValueError, match="no halo history rows"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_delay_sfr_uses_effective_rate_when_raw_rate_is_negative() -> None:
    from auroralf.sfr import compute_sfr_from_tracks

    tracks = _manual_tracks(
        raw=np.array([-1.0e9, -2.0e9, 3.0e9, 4.0e9]),
        effective=np.array([0.0, 0.0, 3.0e9, 4.0e9]),
    )
    result = compute_sfr_from_tracks(
        tracks,
        cosmology=Cosmology(),
        enable_time_delay=True,
    )

    assert np.all(np.asarray(result["SFR"], dtype=float) >= 0.0)
    effective_source = np.asarray(result["dMh_dt_sfr_src"], dtype=float)
    assert np.all(effective_source[np.isfinite(effective_source)] >= 0.0)
    finite_burst = np.asarray(result["mdot_burst"], dtype=float)
    assert np.all(finite_burst[np.isfinite(finite_burst)] >= 0.0)
    np.testing.assert_array_equal(result["dMh_dt_raw"], tracks["dMh_dt_raw"])
    np.testing.assert_array_equal(result["dMh_dt_clipped"], tracks["dMh_dt_clipped"])


def test_mcbride_generator_fails_clearly_when_mass_floor_removes_all_rows() -> None:
    from auroralf.mah import generate_halo_histories

    with pytest.raises(RuntimeError, match="no halo history rows"):
        generate_halo_histories(
            n_tracks=2,
            z_final=6.0,
            Mh_final=1.0e9,
            cosmology=Cosmology(),
            z_start_max=8.0,
            M_min=1.0e30,
            custom_grid=np.array([8.0, 7.0, 6.0]),
            time_grid_mode="custom",
            store_inactive_history=False,
            random_seed=7,
        )


def test_pipeline_passes_effective_not_raw_accretion_rate_to_source_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.generator as generator
    import auroralf.uvlf.pipeline as pipeline
    from auroralf.sfr import PopIIISFRParameters

    def fixed_parameters(*, mass_ref, size, sampler, rng, pilot_samples):
        del mass_ref, sampler, rng, pilot_samples
        return np.column_stack((np.ones(size), np.zeros(size))), None

    monkeypatch.setattr(generator, "sample_parameters", fixed_parameters)
    monkeypatch.setattr(
        pipeline,
        "load_uv1600_table",
        lambda file_path, **kwargs: (np.array([1.0e-6, 1.0]), np.array([1.0, 1.0])),
    )
    monkeypatch.setattr(
        pipeline,
        "load_popiii_uv_luminosity_table",
        lambda file_path: (np.array([1.0e-6, 1.0]), np.array([1.0, 1.0])),
    )

    captured_popiii_rates: list[np.ndarray] = []
    captured_topheavy_rates: list[np.ndarray] = []
    original_popiii = pipeline.compute_popiii_sfr_from_grids
    original_topheavy = pipeline.compute_topheavy_source_flags

    def spy_popiii(**kwargs):
        captured_popiii_rates.append(np.asarray(kwargs["dmhdt_sfr_grid"], dtype=float).copy())
        return original_popiii(**kwargs)

    def spy_topheavy(**kwargs):
        captured_topheavy_rates.append(np.asarray(kwargs["dmhdt_sfr_grid"], dtype=float).copy())
        return original_topheavy(**kwargs)

    monkeypatch.setattr(pipeline, "compute_popiii_sfr_from_grids", spy_popiii)
    monkeypatch.setattr(pipeline, "compute_topheavy_source_flags", spy_topheavy)

    result = pipeline.run_halo_uv_pipeline(
        n_tracks=2,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=Cosmology(),
        random_seeds=derive_pipeline_random_seeds(7, redshift=6.0, mass_index=0),
        z_start_max=8.0,
        n_grid=4,
        workers=1,
        ssp_file="dummy.dat",
        enable_popiii=True,
        popiii_ssp_file="popiii.dat",
        popiii_sfr_parameters=PopIIISFRParameters(
            upper_mass_mode="fixed",
            upper_mass_msun=1.0e12,
        ),
    )

    raw = np.asarray(result.sfr_tracks["dMh_dt_raw"], dtype=float)
    effective = np.asarray(result.sfr_tracks["dMh_dt_sfr"], dtype=float)
    assert np.all(raw < 0.0)
    np.testing.assert_array_equal(effective, np.zeros_like(effective))
    assert captured_popiii_rates and captured_topheavy_rates
    for rates in (*captured_popiii_rates, *captured_topheavy_rates):
        np.testing.assert_array_equal(rates, np.zeros_like(rates))
