from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

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
from auroralf.results import (
    HaloTrackResult,
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)


def _track(**overrides: object) -> HaloTrackResult:
    values: dict[str, object] = {
        "halo_id": np.array([7, 7, 7], dtype=np.int64),
        "time_gyr": np.array([0.4, 0.5, 0.6]),
        "redshift": np.array([10.0, 8.0, 6.0]),
        "halo_mass_msun": np.array([1.0e9, 1.2e9, 1.4e9]),
        "dmh_dt_raw_msun_per_gyr": np.array([1.0e9, -2.0e8, 3.0e9]),
        "dmh_dt_sfr_msun_per_gyr": np.array([1.0e9, 0.0, 3.0e9]),
        "dmh_dt_clipped": np.array([False, True, False]),
        "sfr_msun_per_yr": np.array([0.1, 0.0, 0.2]),
        "active": np.array([True, True, True]),
        "birth_metallicity_zsun": np.array([0.0, 0.01, 0.02]),
        "gas_metallicity_zsun": None,
    }
    values.update(overrides)
    return HaloTrackResult(**values)  # type: ignore[arg-type]


def _mode(mode: str = "canonical", **overrides: object) -> IMFModeResult:
    values: dict[str, object] = {
        "imf_mode": mode,
        "bin_edges_muv": np.array([-24.0, -20.0, -16.0]),
        "bin_centers_muv": np.array([-22.0, -18.0]),
        "bin_width_mag": np.array([4.0, 4.0]),
        "raw_counts": np.array([2, 1], dtype=np.int64),
        "weighted_counts_per_mpc3": np.array([2.0e-5, 1.0e-4]),
        "weight_squared_counts_per_mpc6": np.array([2.0e-10, 1.0e-8]),
        "weighted_count_sigma_per_mpc3": np.array([1.4e-5, 1.0e-4]),
        "effective_counts": np.array([2.0, 1.0]),
        "phi_intrinsic_per_mpc3_per_mag": np.array([5.0e-6, 2.5e-5]),
        "phi_intrinsic_sigma_per_mpc3_per_mag": np.array([3.5e-6, 2.5e-5]),
        "phi_observed_per_mpc3_per_mag": np.array([4.0e-6, 2.0e-5]),
        "phi_observed_sigma_per_mpc3_per_mag": np.array([2.8e-6, 2.0e-5]),
        "halo_tracks": (),
    }
    values.update(overrides)
    return IMFModeResult(**values)  # type: ignore[arg-type]


def _diagnostic(redshift: float = 6.0, mode: str = "canonical") -> ModeRunDiagnostics:
    return ModeRunDiagnostics(
        redshift=redshift,
        imf_mode=mode,
        sampling_seconds=1.5,
        sample_count=6,
        valid_sample_count=5,
        topheavy_source_fraction=0.0,
        popiii_source_fraction=0.0,
        sfrd_msun_per_yr_per_mpc3=1.0e-3,
        popiii_sfrd_msun_per_yr_per_mpc3=0.0,
    )


def _config(tmp_path: Path, *, modes: tuple[str, ...] = ("canonical",)) -> UVLFRunConfig:
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="result-test",
        redshifts=(6.0,),
        base_seed=12,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=8),
        star_formation=StarFormationConfig(),
        stellar_population=StellarPopulationConfig(
            imf_modes=modes,
            canonical_ssp_path=(tmp_path / "canonical.dat").resolve(),
            topheavy_ssp_path=(tmp_path / "topheavy.hdf5").resolve(),
            popiii_ssp_path=(tmp_path / "popiii.dat").resolve(),
            birth_metallicity_topheavy_max_zsun=None,
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=2,
            n_tracks_per_halo_mass=3,
            muv_bin_edges=(-24.0, -20.0, -16.0),
            workers=1,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / "result.h5").resolve()),
    )


def test_halo_track_is_frozen_and_arrays_are_defensive_read_only_copies() -> None:
    time = np.array([0.4, 0.5, 0.6])
    result = _track(time_gyr=time)
    time[0] = 99.0

    assert result.time_gyr[0] == 0.4
    assert not result.time_gyr.flags.writeable
    assert not result.dmh_dt_clipped.flags.writeable
    assert result.gas_metallicity_zsun is None
    with pytest.raises(ValueError):
        result.time_gyr[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.time_gyr = np.array([1.0])  # type: ignore[misc]


def test_all_stored_array_categories_are_irreversibly_read_only_and_defensive() -> None:
    halo_id = np.array([7, 7, 7], dtype=np.int32)
    active = np.array([True, True, True])
    birth = np.array([0.0, 0.01, 0.02], dtype=np.float32)
    track = _track(halo_id=halo_id, active=active, birth_metallicity_zsun=birth)
    mode = _mode(halo_tracks=(track,))

    halo_id[0] = 99
    active[0] = False
    birth[0] = 99.0
    assert track.halo_id[0] == 7
    assert bool(track.active[0])
    assert track.birth_metallicity_zsun is not None
    assert track.birth_metallicity_zsun[0] == 0.0

    probes = (
        track.time_gyr,
        track.halo_id,
        track.active,
        track.birth_metallicity_zsun,
        mode.phi_intrinsic_per_mpc3_per_mag,
        mode.raw_counts,
    )
    for array in probes:
        assert array is not None
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("time_gyr", np.array([0.4, 0.4, 0.6]), "strictly increasing"),
        ("redshift", np.array([10.0, -1.0, 6.0]), "redshift"),
        ("halo_mass_msun", np.array([1.0e9, 0.0, 1.4e9]), "halo_mass_msun"),
        ("dmh_dt_sfr_msun_per_gyr", np.array([1.0, -1.0, 1.0]), "dmh_dt_sfr"),
        ("sfr_msun_per_yr", np.array([0.1, -0.1, 0.2]), "sfr_msun_per_yr"),
        ("birth_metallicity_zsun", np.array([0.0, -0.1, 0.2]), "birth_metallicity"),
        ("active", np.array([1, 1, 1]), "active.*boolean"),
        ("halo_id", np.array([7.0, 7.0, 7.0]), "halo_id.*integer"),
        ("time_gyr", np.array([0.4, 0.5]), "same length"),
    ],
)
def test_halo_track_rejects_invalid_shapes_and_physics(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _track(**{field: value})


def test_halo_track_raw_accretion_may_be_negative_but_clipped_mask_is_exact() -> None:
    result = _track()
    assert result.dmh_dt_raw_msun_per_gyr[1] < 0.0

    with pytest.raises(ValueError, match="dmh_dt_clipped.*raw"):
        _track(dmh_dt_clipped=np.array([False, False, False]))


@pytest.mark.parametrize(
    "effective",
    [
        np.array([1.0e9, 999.0, 3.0e9]),
        np.array([888.0, 0.0, 3.0e9]),
    ],
)
def test_halo_track_effective_accretion_exactly_matches_clipped_raw(
    effective: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="dmh_dt_sfr_msun_per_gyr.*maximum"):
        _track(dmh_dt_sfr_msun_per_gyr=effective)


def test_halo_track_rejects_boolean_numeric_members_before_cast() -> None:
    with pytest.raises((TypeError, ValueError), match="halo_mass_msun.*boolean"):
        _track(halo_mass_msun=[1.0e9, True, 1.4e9])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("halo_mass_msun", [1.0e9, "1.2e9", 1.4e9]),
        ("halo_mass_msun", [1.0e9, b"1200000000", 1.4e9]),
        ("halo_mass_msun", [1.0e9, 1.2e9 + 0.0j, 1.4e9]),
        ("halo_mass_msun", np.array([1.0e9, "1.2e9", 1.4e9], dtype=object)),
        ("halo_id", [7, 7.0, 7]),
        ("halo_id", [7, "7", 7]),
        ("active", [True, 1, True]),
        ("dmh_dt_clipped", np.array([False, True, False], dtype=object)),
    ],
)
def test_halo_track_rejects_non_strict_array_member_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _track(**{field: value})


def test_imf_mode_result_arrays_are_defensive_read_only_copies() -> None:
    phi = np.array([5.0e-6, 2.5e-5])
    result = _mode(phi_intrinsic_per_mpc3_per_mag=phi, halo_tracks=(_track(),))
    phi[0] = 99.0

    assert result.phi_intrinsic_per_mpc3_per_mag[0] == 5.0e-6
    assert not result.phi_intrinsic_per_mpc3_per_mag.flags.writeable
    assert type(result.halo_tracks[0]) is HaloTrackResult
    with pytest.raises(ValueError):
        result.raw_counts[0] = 0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("imf_mode", "bad", "imf_mode"),
        ("bin_edges_muv", np.array([-24.0, -20.0, -20.0]), "strictly increasing"),
        ("bin_centers_muv", np.array([-21.0, -18.0]), "bin_centers"),
        ("bin_width_mag", np.array([4.0, 3.0]), "bin_width"),
        ("raw_counts", np.array([2, -1]), "raw_counts"),
        ("raw_counts", np.array([2.0, 1.0]), "raw_counts.*integer"),
        ("weighted_counts_per_mpc3", np.array([1.0]), "bin count"),
        ("effective_counts", np.array([1.0, -1.0]), "effective_counts"),
        ("halo_tracks", [_track()], "halo_tracks.*tuple"),
        ("halo_tracks", (object(),), "HaloTrackResult"),
    ],
)
def test_imf_mode_result_rejects_invalid_bins_counts_and_tracks(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _mode(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "weighted_count_sigma_per_mpc3",
        "phi_intrinsic_sigma_per_mpc3_per_mag",
        "phi_observed_sigma_per_mpc3_per_mag",
    ],
)
def test_sigma_arrays_allow_nan_but_reject_infinity_and_negative_finite(
    field: str,
) -> None:
    accepted = _mode(**{field: np.array([np.nan, 1.0])})
    assert np.isnan(getattr(accepted, field)[0])

    with pytest.raises(ValueError, match=field):
        _mode(**{field: np.array([np.inf, 1.0])})
    with pytest.raises(ValueError, match=field):
        _mode(**{field: np.array([-1.0, 1.0])})

    with pytest.raises((TypeError, ValueError), match=field):
        _mode(**{field: np.array(["nan", "1.0"])})


@pytest.mark.parametrize(
    "value",
    [
        np.array(["1.0", "2.0"]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0, True], dtype=object),
    ],
)
def test_imf_result_rejects_non_real_upstream_numeric_arrays(value: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="weighted_counts_per_mpc3"):
        _mode(weighted_counts_per_mpc3=value)


def test_redshift_result_requires_unique_modes_and_exact_lookup() -> None:
    result = RedshiftResult(redshift=6.0, imf_modes=(_mode(), _mode("z10_mild_topheavy")))

    assert result.for_mode("canonical").imf_mode == "canonical"
    with pytest.raises(KeyError, match="missing"):
        result.for_mode("missing")
    with pytest.raises(ValueError, match="unique"):
        RedshiftResult(redshift=6.0, imf_modes=(_mode(), _mode()))
    with pytest.raises(TypeError, match="tuple"):
        RedshiftResult(redshift=6.0, imf_modes=[_mode()])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sample_count": True}, "sample_count"),
        ({"valid_sample_count": 7}, "valid_sample_count"),
        ({"sampling_seconds": -1.0}, "sampling_seconds"),
        ({"topheavy_source_fraction": 1.1}, "topheavy_source_fraction"),
        ({"popiii_source_fraction": -0.1}, "popiii_source_fraction"),
        ({"sfrd_msun_per_yr_per_mpc3": np.inf}, "sfrd"),
    ],
)
def test_mode_diagnostics_reject_invalid_counts_fractions_and_rates(
    overrides: dict[str, object],
    match: str,
) -> None:
    values = asdict(_diagnostic())
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=match):
        ModeRunDiagnostics(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_seconds", "1.5"),
        ("sampling_seconds", 1.5 + 0.0j),
        ("topheavy_source_fraction", True),
        ("sample_count", 6.0),
        ("valid_sample_count", "5"),
    ],
)
def test_mode_diagnostics_rejects_non_strict_scalar_types(
    field: str,
    value: object,
) -> None:
    values = asdict(_diagnostic())
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        ModeRunDiagnostics(**values)


def test_run_diagnostics_requires_exact_unique_mode_runs() -> None:
    diagnostic = _diagnostic()
    result = RunDiagnostics(total_seconds=2.0, mode_runs=(diagnostic,))
    assert result.mode_runs == (diagnostic,)

    with pytest.raises(ValueError, match="unique"):
        RunDiagnostics(total_seconds=2.0, mode_runs=(diagnostic, diagnostic))
    with pytest.raises(TypeError, match="ModeRunDiagnostics"):
        RunDiagnostics(total_seconds=2.0, mode_runs=(object(),))  # type: ignore[arg-type]


def test_uvlf_run_result_matches_config_axes_and_exact_lookup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    redshift = RedshiftResult(redshift=6.0, imf_modes=(_mode(),))
    diagnostic = _diagnostic()
    result = UVLFRunResult(
        config=config,
        redshifts=(redshift,),
        diagnostics=RunDiagnostics(total_seconds=2.0, mode_runs=(diagnostic,)),
    )

    assert result.for_redshift(6.0) is redshift
    with pytest.raises(KeyError, match="7.0"):
        result.for_redshift(7.0)
    with pytest.raises(TypeError, match="config"):
        UVLFRunResult(
            config=object(),  # type: ignore[arg-type]
            redshifts=(redshift,),
            diagnostics=RunDiagnostics(total_seconds=2.0, mode_runs=(diagnostic,)),
        )


def test_uvlf_run_result_rejects_mismatched_redshifts_modes_and_diagnostics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    diagnostics = RunDiagnostics(total_seconds=2.0, mode_runs=(_diagnostic(),))

    with pytest.raises(ValueError, match="redshift"):
        UVLFRunResult(
            config=config,
            redshifts=(RedshiftResult(redshift=7.0, imf_modes=(_mode(),)),),
            diagnostics=diagnostics,
        )
    with pytest.raises(ValueError, match="imf_modes"):
        UVLFRunResult(
            config=config,
            redshifts=(
                RedshiftResult(
                    redshift=6.0,
                    imf_modes=(_mode(), _mode("z10_mild_topheavy")),
                ),
            ),
            diagnostics=diagnostics,
        )
    with pytest.raises(ValueError, match="diagnostics"):
        UVLFRunResult(
            config=config,
            redshifts=(RedshiftResult(redshift=6.0, imf_modes=(_mode(),)),),
            diagnostics=RunDiagnostics(
                total_seconds=2.0,
                mode_runs=(_diagnostic(redshift=7.0),),
            ),
        )


def test_uvlf_run_result_rejects_mode_bin_edges_that_disagree_with_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    inconsistent_mode = _mode(
        bin_edges_muv=np.array([-24.0, -19.0, -16.0]),
        bin_centers_muv=np.array([-21.5, -17.5]),
        bin_width_mag=np.array([5.0, 3.0]),
    )

    with pytest.raises(ValueError, match="bin_edges_muv.*sampling.muv_bin_edges"):
        UVLFRunResult(
            config=config,
            redshifts=(RedshiftResult(redshift=6.0, imf_modes=(inconsistent_mode,)),),
            diagnostics=RunDiagnostics(total_seconds=2.0, mode_runs=(_diagnostic(),)),
        )


def test_results_module_uses_explicit_fields_not_generic_get() -> None:
    source = (Path(__file__).resolve().parents[1] / "auroralf/results.py").read_text(encoding="utf-8")
    assert ".get(" not in source
