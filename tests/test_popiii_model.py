from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from auroralf.mah import Cosmology, HaloHistoryResult
from auroralf.seeding import derive_pipeline_random_seeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"
V2_RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_v2.py"
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "uvlf" / "production.toml"


def test_popiii_flat_sfe_matches_cruz_constant_efficiency() -> None:
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_star_formation_efficiency

    params = PopIIISFRParameters(
        epsilon_star=1.0e-3,
        pivot_mass_msun=1.0e7,
        alpha_star=0.0,
        beta_star=0.0,
    )

    efficiency = compute_popiii_star_formation_efficiency(
        np.array([1.0e6, 1.0e7, 1.0e8]),
        params,
    )

    np.testing.assert_allclose(efficiency, np.full(3, 1.0e-3))


@pytest.mark.parametrize(
    "params",
    [
        {"epsilon_star": -1.0e-3},
        {"epsilon_star": 1.1},
        {"pivot_mass_msun": 0.0},
        {"lw_background_j21": -0.1},
        {"upper_mass_mode": "fixed"},
        {"upper_mass_mode": "unknown", "upper_mass_msun": 1.0e8},
    ],
)
def test_popiii_parameters_reject_invalid_values(params: dict[str, float | str]) -> None:
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_star_formation_efficiency

    with pytest.raises(ValueError):
        compute_popiii_star_formation_efficiency(
            np.array([1.0e7]),
            PopIIISFRParameters(**params),
        )


def test_popiii_duty_cycle_uses_molecular_floor_and_upper_cutoff() -> None:
    from auroralf.mah import Cosmology
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_duty_cycle
    from auroralf.uvlf.cooling import compute_popiii_lw_minimum_mass_msun

    z_obs = 20.0
    halo_mass = 1.0e6
    params = PopIIISFRParameters(
        upper_mass_mode="fixed",
        upper_mass_msun=1.0e8,
        lw_background_j21=0.0,
    )
    molecular_floor = compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=0.0)

    duty = compute_popiii_duty_cycle(
        halo_mass,
        z_obs,
        params,
        cosmology=Cosmology(),
    )

    assert duty == pytest.approx(np.exp(-molecular_floor / halo_mass) * np.exp(-halo_mass / 1.0e8))

    stronger_lw = compute_popiii_duty_cycle(
        halo_mass,
        z_obs,
        replace(params, lw_background_j21=0.2),
        cosmology=Cosmology(),
    )
    assert stronger_lw < duty


def test_popiii_sfr_grid_forms_stars_in_minihalos_below_atomic_threshold() -> None:
    from auroralf.mah import Cosmology
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_sfr_from_grids
    from auroralf.uvlf.cooling import compute_atomic_cooling_mass_msun

    z_grid = np.array([[20.0, 15.0, 10.0]])
    mh_grid = np.array([[2.0e6, 4.0e6, 8.0e6]])
    dmhdt_grid = np.full_like(mh_grid, 1.0e8)
    active_grid = np.ones_like(mh_grid, dtype=bool)
    params = PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8)
    cosmology = Cosmology(omega_m=0.4, omega_b=0.064, omega_lambda=0.6)

    result = compute_popiii_sfr_from_grids(
        mh_grid=mh_grid,
        dmhdt_sfr_grid=dmhdt_grid,
        z_grid=z_grid,
        active_grid=active_grid,
        cosmology=cosmology,
        parameters=params,
    )

    assert np.all(mh_grid < compute_atomic_cooling_mass_msun(z_grid, cosmology=cosmology))
    assert np.any(result.sfr_grid > 0.0)
    np.testing.assert_allclose(
        result.sfr_grid,
        0.16 * result.fstar_grid * result.duty_cycle_grid * dmhdt_grid / 1.0e9,
    )


def test_popiii_atomic_upper_mass_uses_supplied_cosmology() -> None:
    from auroralf.cooling import compute_atomic_cooling_mass_msun
    from auroralf.mah import Cosmology
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_upper_mass_msun

    cosmology = Cosmology(
        h0=2.0 * Cosmology().h0,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )
    result = compute_popiii_upper_mass_msun(
        10.0,
        PopIIISFRParameters(upper_mass_mode="atomic"),
        cosmology=cosmology,
    )

    assert result == pytest.approx(
        compute_atomic_cooling_mass_msun(10.0, cosmology=cosmology)
    )


def test_popiii_atomic_public_apis_require_cosmology() -> None:
    from auroralf.sfr import (
        compute_popiii_duty_cycle,
        compute_popiii_sfr_from_grids,
        compute_popiii_upper_mass_msun,
    )

    with pytest.raises(TypeError, match="cosmology"):
        compute_popiii_upper_mass_msun(10.0)
    with pytest.raises(TypeError, match="cosmology"):
        compute_popiii_duty_cycle(1.0e7, 10.0)
    with pytest.raises(TypeError, match="cosmology"):
        compute_popiii_sfr_from_grids(
            mh_grid=np.array([[1.0e7]]),
            dmhdt_sfr_grid=np.array([[1.0e8]]),
            z_grid=np.array([[10.0]]),
            active_grid=np.array([[True]]),
        )


@pytest.mark.parametrize("invalid_rate", [np.nan, np.inf, -np.inf])
def test_popiii_sfr_rejects_nonfinite_effective_accretion_rate(invalid_rate: float) -> None:
    from auroralf.sfr import compute_popiii_sfr_from_grids

    with pytest.raises(ValueError, match="dmhdt_sfr_grid.*finite"):
        compute_popiii_sfr_from_grids(
            mh_grid=np.array([[1.0e7]]),
            dmhdt_sfr_grid=np.array([[invalid_rate]]),
            z_grid=np.array([[10.0]]),
            active_grid=np.array([[True]]),
            cosmology=Cosmology(),
        )


def test_popiii_sfr_rejects_legacy_baryon_fraction_argument() -> None:
    from auroralf.sfr import compute_popiii_sfr_from_grids

    with pytest.raises(TypeError, match="baryon_fraction"):
        compute_popiii_sfr_from_grids(
            mh_grid=np.array([[1.0e7]]),
            dmhdt_sfr_grid=np.array([[1.0e8]]),
            z_grid=np.array([[10.0]]),
            active_grid=np.array([[True]]),
            baryon_fraction=0.16,
            cosmology=Cosmology(),
        )


def test_popiii_sfr_reports_overflow_without_emitting_runtime_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    import auroralf.sfr.popiii as popiii

    maximum = np.finfo(float).max
    monkeypatch.setattr(
        popiii,
        "compute_popiii_star_formation_efficiency",
        lambda halo_mass_msun, parameters=None: np.full_like(halo_mass_msun, maximum, dtype=float),
    )
    monkeypatch.setattr(
        popiii,
        "compute_popiii_duty_cycle",
        lambda halo_mass_msun, z_obs, parameters=None, *, cosmology: np.ones_like(
            halo_mass_msun,
            dtype=float,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="SFR.*non-finite or negative"):
            popiii.compute_popiii_sfr_from_grids(
                mh_grid=np.array([[1.0e7]]),
                dmhdt_sfr_grid=np.array([[maximum]]),
                z_grid=np.array([[10.0]]),
                active_grid=np.array([[True]]),
                cosmology=Cosmology(),
            )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_visbal2015_atomic_cooling_mass_matches_paper_formula() -> None:
    from auroralf.sfr import compute_visbal2015_atomic_cooling_mass_msun

    redshift = np.array([10.0, 20.0])

    mass = compute_visbal2015_atomic_cooling_mass_msun(redshift)

    expected = 5.4e7 * ((1.0 + redshift) / 11.0) ** -1.5
    np.testing.assert_allclose(mass, expected)


def test_visbal2015_minihalo_mass_matches_paper_formula() -> None:
    from auroralf.sfr import compute_visbal2015_minihalo_minimum_mass_msun

    redshift = np.array([10.0, 20.0])
    j21 = np.array([0.0, 0.1])
    expected = 2.5e5 * ((1.0 + redshift) / 26.0) ** -1.5
    expected *= 1.0 + 6.96 * (4.0 * np.pi * j21) ** 0.47

    np.testing.assert_allclose(
        compute_visbal2015_minihalo_minimum_mass_msun(
            redshift,
            lw_background_j21=j21,
        ),
        expected,
    )


def test_misattributed_visbal2015_per_halo_sfr_fails_fast() -> None:
    from auroralf.sfr import compute_popiii_sfr_visbal2015_from_grids

    with pytest.raises(RuntimeError, match="misattributed.*collapsed-fraction"):
        compute_popiii_sfr_visbal2015_from_grids(
            mh_grid=np.array([[3.0e7]]),
            z_grid=np.array([[10.0]]),
            active_grid=np.array([[True]]),
            fstar=0.1,
            eta_duty=0.1,
            cosmology=Cosmology(),
        )


def test_popiii_ssp_loader_reads_instantaneous_schaerer_l1500_table(tmp_path) -> None:
    from auroralf.ssp import load_popiii_uv_luminosity_table

    table = tmp_path / "pop3_test_is5.25"
    table.write_text(
        "\n".join(
            [
                "# Input parameters: total mass= 1.0E+00",
                "# Star-formation: instantaneous burst at age=0",
                "# log(age) log(Q_0) M_BOL L_15SB L_1500 L_2800",
                "# log [erg s^-1 A^-1]",
                " 4.000E+00  47.0  -3.0  33.10  33.00  32.00",
                " 6.000E+00  46.0  -2.0  32.10  32.00  31.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ages_myr, luminosity_per_msun = load_popiii_uv_luminosity_table(table)

    np.testing.assert_allclose(ages_myr, np.array([1.0e-2, 1.0]))
    expected = (10.0**33.0 * (u.erg / u.s / u.AA)).to(
        u.erg / u.s / u.Hz,
        equivalencies=u.spectral_density(1500.0 * u.AA),
    )
    assert luminosity_per_msun[0] == pytest.approx(expected.value)
    assert np.all(luminosity_per_msun > 0.0)


def test_popiii_uv_cache_reloads_same_path_atomic_replacement_and_hits_when_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auroralf.ssp import uv1600

    def table_text(log_luminosity: str) -> str:
        return "\n".join(
            [
                "# Star-formation: instantaneous burst at age=0",
                "# log(age) L_1500",
                f" 4.000E+00 {log_luminosity}",
                " 6.000E+00 32.00",
            ]
        ) + "\n"

    path = tmp_path / "pop3_test_is5.25"
    replacement = tmp_path / "replacement_is5.25"
    path.write_text(table_text("33.00"), encoding="utf-8")
    uv1600._load_popiii_uv_luminosity_table_cached.cache_clear()
    real_load = uv1600._load_popiii_uv_luminosity_table_from_schaerer
    reads: list[Path] = []

    def read_spy(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        reads.append(Path(str(kwargs["file_path"])))
        return real_load(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        uv1600,
        "_load_popiii_uv_luminosity_table_from_schaerer",
        read_spy,
    )
    _, first = uv1600.load_popiii_uv_luminosity_table(path)
    _, unchanged = uv1600.load_popiii_uv_luminosity_table(path)
    assert len(reads) == 1
    np.testing.assert_array_equal(unchanged, first)
    original_mtime_ns = path.stat().st_mtime_ns

    replacement.write_text(table_text("34.00"), encoding="utf-8")
    assert replacement.stat().st_size == path.stat().st_size
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, path)
    os.utime(path, ns=(original_mtime_ns, original_mtime_ns))

    _, replaced = uv1600.load_popiii_uv_luminosity_table(path)

    assert len(reads) == 2
    assert replaced[0] == pytest.approx(first[0] * 10.0)


def test_popiii_ssp_loader_rejects_constant_sfr_tables(tmp_path) -> None:
    from auroralf.ssp import load_popiii_uv_luminosity_table

    table = tmp_path / "pop3_test_cs5.25"
    table.write_text(
        "\n".join(
            [
                "# Input parameters: SFR=1 Msun/yr",
                "# Star-formation: constant SFR",
                "# log(age) L_1500",
                " 6.000E+00  40.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instantaneous"):
        load_popiii_uv_luminosity_table(table)


def test_popiii_ssp_loader_reconstructs_is5_age_grid_when_printed_log_ages_repeat(tmp_path) -> None:
    from auroralf.ssp import load_popiii_uv_luminosity_table

    table = tmp_path / "pop3_test_is5.25"
    table.write_text(
        "\n".join(
            [
                "# Input parameters: total mass= 1.0E+00",
                "# Star-formation: instantaneous burst at age=0",
                "# log(age) L_1500",
                "# log [erg s^-1 A^-1]",
                " 4.000E+00  33.00",
                " 8.667E+00  32.00",
                " 8.667E+00  31.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ages_myr, luminosity_per_msun = load_popiii_uv_luminosity_table(table)

    np.testing.assert_allclose(ages_myr, np.array([1.0e-2, 1.0, 2.0]))
    assert np.all(luminosity_per_msun > 0.0)


def test_pipeline_adds_popiii_uv_without_changing_popii_sfr(monkeypatch: pytest.MonkeyPatch) -> None:
    import auroralf.uvlf.pipeline as pipeline
    from auroralf.mah import Cosmology
    from auroralf.sfr import PopIIISFRParameters

    tracks = {
        "halo_id": np.array([0, 0, 0], dtype=int),
        "step": np.array([0, 1, 2], dtype=int),
        "z": np.array([20.0, 15.0, 10.0]),
        "t_gyr": np.array([0.18, 0.28, 0.48]),
        "dt_gyr": np.array([0.0, 0.10, 0.20]),
        "Mh": np.array([2.0e6, 4.0e6, 8.0e6]),
        "dMh_dt_raw": np.full(3, 1.0e8),
        "dMh_dt_sfr": np.full(3, 1.0e8),
        "dMh_dt_clipped": np.zeros(3, dtype=bool),
        "active_flag": np.ones(3, dtype=bool),
        "termination_flag": np.array(["active", "active", "completed"], dtype=object),
    }

    def fake_generate_halo_histories(**kwargs: object) -> HaloHistoryResult:
        return HaloHistoryResult(
            tracks={name: values.copy() for name, values in tracks.items()},
            metadata={
                "time_grid_mode": "uniform_in_t",
                "dt_gyr_median": 0.15,
                "negative_dmhdt_clip_count": 0,
                "negative_dmhdt_total_count": 3,
                "negative_dmhdt_clip_fraction": 0.0,
            },
        )

    def fake_load_uv1600_table(file_path: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0e-6, 1.0]), np.array([1.0, 1.0])

    def fake_load_popiii_table(file_path: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0e-6, 1.0]), np.array([10.0, 10.0])

    monkeypatch.setattr(pipeline, "generate_halo_histories", fake_generate_halo_histories)
    monkeypatch.setattr(pipeline, "load_uv1600_table", fake_load_uv1600_table)
    monkeypatch.setattr(pipeline, "load_popiii_uv_luminosity_table", fake_load_popiii_table)

    disabled = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=10.0,
        Mh_final=8.0e6,
        cosmology=Cosmology(),
        random_seeds=derive_pipeline_random_seeds(42, redshift=10.0, mass_index=0),
        z_start_max=20.0,
        n_grid=3,
        workers=1,
        ssp_file="dummy.dat",
        enable_popiii=False,
    )
    enabled = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=10.0,
        Mh_final=8.0e6,
        cosmology=Cosmology(),
        random_seeds=derive_pipeline_random_seeds(42, redshift=10.0, mass_index=0),
        z_start_max=20.0,
        n_grid=3,
        workers=1,
        ssp_file="dummy.dat",
        enable_popiii=True,
        popiii_sfr_parameters=PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8),
        popiii_ssp_file="popiii.dat",
    )

    np.testing.assert_allclose(disabled.sfr_tracks["SFR"], 0.0)
    np.testing.assert_allclose(enabled.sfr_tracks["SFR"], 0.0)
    assert np.all(disabled.uv_luminosities_popiii == 0.0)
    assert np.any(enabled.sfr_tracks["SFR_popiii"] > 0.0)
    assert np.any(enabled.uv_luminosities_popiii > 0.0)
    np.testing.assert_allclose(
        enabled.uv_luminosities,
        enabled.uv_luminosities_canonical
        + enabled.uv_luminosities_topheavy
        + enabled.uv_luminosities_popiii,
    )
    assert enabled.metadata["popiii_enabled"] is True
    enabled_popiii_sfr = np.asarray(enabled.sfr_tracks["SFR_popiii"]).reshape(
        enabled.active_grid.shape
    )
    expected_popiii_source = enabled.active_grid & (enabled_popiii_sfr > 0.0)
    np.testing.assert_array_equal(enabled.popiii_source_grid, expected_popiii_source)
    np.testing.assert_array_equal(
        np.asarray(enabled.sfr_tracks["popiii_source_flag"]).reshape(
            enabled.active_grid.shape
        ),
        expected_popiii_source,
    )
    assert enabled.metadata["popiii_source_count"] == int(
        np.count_nonzero(expected_popiii_source)
    )
    assert enabled.metadata["popiii_source_count"] > 0
    assert enabled.metadata["popiii_light_fraction_median"] > 0.0
    assert disabled.metadata["mah_backend"] == "mcbride"
    assert disabled.metadata["negative_dmhdt_clip_count"] == 0
    assert disabled.metadata["negative_dmhdt_total_count"] == 3
    assert disabled.metadata["negative_dmhdt_clip_fraction"] == pytest.approx(
        disabled.metadata["negative_dmhdt_clip_count"]
        / disabled.metadata["negative_dmhdt_total_count"]
    )
    assert "tng_negative_dmhdt_clip_count" not in disabled.metadata
    assert "thesan_negative_dmhdt_clip_count" not in disabled.metadata


def test_pipeline_keeps_popiii_minihalos_when_popii_atomic_gate_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.pipeline as pipeline
    from auroralf.mah import Cosmology
    from auroralf.sfr import PopIIISFRParameters
    from auroralf.uvlf.cooling import compute_atomic_cooling_mass_msun

    z_final = 14.5
    mh_final = 1.0e7

    def fake_load_uv1600_table(file_path: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0e-6, 1.0]), np.array([1.0, 1.0])

    def fake_load_popiii_table(file_path: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0e-6, 1.0]), np.array([10.0, 10.0])

    monkeypatch.setattr(pipeline, "load_uv1600_table", fake_load_uv1600_table)
    monkeypatch.setattr(pipeline, "load_popiii_uv_luminosity_table", fake_load_popiii_table)

    cosmology = Cosmology()
    result = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=z_final,
        Mh_final=mh_final,
        cosmology=cosmology,
        random_seeds=derive_pipeline_random_seeds(42, redshift=z_final, mass_index=0),
        z_start_max=16.0,
        n_grid=8,
        workers=1,
        ssp_file="dummy.dat",
        enable_popiii=True,
        popiii_sfr_parameters=PopIIISFRParameters(upper_mass_mode="atomic"),
        popiii_ssp_file="popiii.dat",
    )

    assert mh_final < compute_atomic_cooling_mass_msun(
        z_final,
        cosmology=cosmology,
    )
    np.testing.assert_allclose(result.sfr_tracks["SFR"], 0.0)
    assert np.any(result.sfr_tracks["SFR_popiii"] > 0.0)
    assert np.any(result.uv_luminosities_popiii > 0.0)


def test_v2_run_script_uses_config_only_and_legacy_entry_is_disabled() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(V2_RUN_SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in completed.stdout
    assert "--enable-popiii" not in completed.stdout
    legacy = subprocess.run(
        [sys.executable, str(RUN_SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert legacy.returncode != 0
    assert "legacy UVLF production entry point is disabled" in legacy.stderr


def test_production_config_keeps_popiii_disabled_with_explicit_parameters() -> None:
    from auroralf import UVLFRunConfig

    population = UVLFRunConfig.from_toml(PRODUCTION_CONFIG).stellar_population
    assert population.enable_popiii is False
    assert population.popiii_efficiency == pytest.approx(1.0e-3)
    assert population.popiii_pivot_halo_mass_msun == pytest.approx(1.0e7)
    assert population.popiii_low_mass_slope == pytest.approx(0.0)
    assert population.popiii_high_mass_slope == pytest.approx(0.0)
    assert population.popiii_upper_mass_mode == "atomic"
    assert population.popiii_upper_mass_msun is None
    assert population.lw_background_j21 == pytest.approx(0.0)


def test_production_uvlf_helper_has_no_independent_lw_parameter() -> None:
    from auroralf import run_uvlf

    assert tuple(inspect.signature(run_uvlf).parameters) == ("config",)
