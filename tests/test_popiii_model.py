from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from auroralf.mah import HaloHistoryResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run" / "run_uvlf_compare_imf_no_delay_all_z.py"


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

    duty = compute_popiii_duty_cycle(halo_mass, z_obs, params)

    assert duty == pytest.approx(np.exp(-molecular_floor / halo_mass) * np.exp(-halo_mass / 1.0e8))

    stronger_lw = compute_popiii_duty_cycle(
        halo_mass,
        z_obs,
        replace(params, lw_background_j21=0.2),
    )
    assert stronger_lw < duty


def test_popiii_sfr_grid_forms_stars_in_minihalos_below_atomic_threshold() -> None:
    from auroralf.sfr import PopIIISFRParameters, compute_popiii_sfr_from_grids
    from auroralf.uvlf.cooling import compute_atomic_cooling_mass_msun

    z_grid = np.array([[20.0, 15.0, 10.0]])
    mh_grid = np.array([[2.0e6, 4.0e6, 8.0e6]])
    dmhdt_grid = np.full_like(mh_grid, 1.0e8)
    active_grid = np.ones_like(mh_grid, dtype=bool)
    params = PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8)

    result = compute_popiii_sfr_from_grids(
        mh_grid=mh_grid,
        dmhdt_grid=dmhdt_grid,
        z_grid=z_grid,
        active_grid=active_grid,
        baryon_fraction=0.16,
        parameters=params,
    )

    assert np.all(mh_grid < compute_atomic_cooling_mass_msun(z_grid))
    assert np.any(result.sfr_grid > 0.0)
    np.testing.assert_allclose(
        result.sfr_grid,
        0.16 * result.fstar_grid * result.duty_cycle_grid * dmhdt_grid / 1.0e9,
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
    from auroralf.sfr import PopIIISFRParameters

    tracks = {
        "halo_id": np.array([0, 0, 0], dtype=int),
        "step": np.array([0, 1, 2], dtype=int),
        "z": np.array([20.0, 15.0, 10.0]),
        "t_gyr": np.array([0.18, 0.28, 0.48]),
        "dt_gyr": np.array([0.0, 0.10, 0.20]),
        "Mh": np.array([2.0e6, 4.0e6, 8.0e6]),
        "dMh_dt": np.full(3, 1.0e8),
        "active_flag": np.ones(3, dtype=bool),
        "termination_flag": np.array(["active", "active", "completed"], dtype=object),
    }

    def fake_generate_halo_histories(**kwargs: object) -> HaloHistoryResult:
        return HaloHistoryResult(
            tracks={name: values.copy() for name, values in tracks.items()},
            metadata={"time_grid_mode": "uniform_in_t", "dt_gyr_median": 0.15},
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
    assert enabled.metadata["popiii_source_count"] > 0
    assert enabled.metadata["popiii_light_fraction_median"] > 0.0


def test_pipeline_keeps_popiii_minihalos_when_popii_atomic_gate_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.pipeline as pipeline
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

    result = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=z_final,
        Mh_final=mh_final,
        z_start_max=16.0,
        n_grid=8,
        workers=1,
        ssp_file="dummy.dat",
        enable_popiii=True,
        popiii_sfr_parameters=PopIIISFRParameters(upper_mass_mode="atomic"),
        popiii_ssp_file="popiii.dat",
    )

    assert mh_final < compute_atomic_cooling_mass_msun(z_final)
    np.testing.assert_allclose(result.sfr_tracks["SFR"], 0.0)
    assert np.any(result.sfr_tracks["SFR_popiii"] > 0.0)
    assert np.any(result.uv_luminosities_popiii > 0.0)


def test_run_script_help_exposes_popiii_arguments() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--enable-popiii" in completed.stdout
    assert "--popiii-epsilon-star" in completed.stdout
    assert "--popiii-mp" in completed.stdout
    assert "--popiii-alpha-star" in completed.stdout
    assert "--popiii-beta-star" in completed.stdout
    assert "--popiii-upper-mass-mode" in completed.stdout
    assert "--popiii-upper-mass-msun" in completed.stdout
    assert "--popiii-ssp-file" in completed.stdout
    assert "--lw-background-j21" in completed.stdout


def test_run_script_defaults_keep_popiii_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("run_uvlf_compare_imf_no_delay_all_z", RUN_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", [str(RUN_SCRIPT_PATH)])
    args = module._parse_args()

    assert args.enable_popiii is False
    assert args.popiii_epsilon_star == pytest.approx(1.0e-3)
    assert args.popiii_mp == pytest.approx(1.0e7)
    assert args.popiii_alpha_star == pytest.approx(0.0)
    assert args.popiii_beta_star == pytest.approx(0.0)
    assert args.popiii_upper_mass_mode == "atomic"
    assert args.popiii_upper_mass_msun is None
    assert args.lw_background_j21 == pytest.approx(0.0)
