from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def test_popiii_heii1640_loader_reads_schaerer_recombination_table(tmp_path) -> None:
    from auroralf.archive.heii1640 import load_popiii_heii1640_luminosity_table

    table = tmp_path / "pop3_test_is5.22"
    table.write_text(
        "\n".join(
            [
                "# Input parameters: total mass= 1.0E+00",
                "# Star-formation: instantaneous burst at age=0",
                "# log(age) log(Q_0) log(Q_1) log(Q_2) L(H_beta) H_Lya EW I/Hb H_alpha EW I/Hb H_beta EW I/Hb HeI_4471 EW I/Hb HeII_1640 EW I/Hb",
                " 4.000E+00  47.0 46.0 45.0 1.00E+34 1.0 2.0 3.0 4.0 5.0 1.0 6.0 0.5 7.0 0.25",
                " 6.000E+00  47.0 46.0 45.0 2.00E+34 1.0 2.0 3.0 4.0 5.0 1.0 6.0 0.5 8.0 0.10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ages_myr, luminosity_per_msun, ew_rest_a = load_popiii_heii1640_luminosity_table(table)

    np.testing.assert_allclose(ages_myr, np.array([1.0e-2, 1.0]))
    np.testing.assert_allclose(luminosity_per_msun, np.array([2.5e33, 2.0e33]))
    np.testing.assert_allclose(ew_rest_a, np.array([7.0, 8.0]))


def test_popiii_heplus_loader_reads_schaerer_q2_column(tmp_path) -> None:
    from auroralf.archive.heii1640 import load_popiii_heplus_ionizing_photon_table

    table = tmp_path / "pop3_test_is5.22"
    table.write_text(
        "\n".join(
            [
                "# Input parameters: total mass= 1.0E+00",
                "# Star-formation: instantaneous burst at age=0",
                "# log(age) log(Q_0) log(Q_1) log(Q_2) L(H_beta) H_Lya EW I/Hb H_alpha EW I/Hb H_beta EW I/Hb HeI_4471 EW I/Hb HeII_1640 EW I/Hb",
                " 4.000E+00  47.0 46.0 45.0 1.00E+34 1.0 2.0 3.0 4.0 5.0 1.0 6.0 0.5 7.0 0.25",
                " 6.000E+00  47.0 46.0 44.5 2.00E+34 1.0 2.0 3.0 4.0 5.0 1.0 6.0 0.5 8.0 0.10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ages_myr, q_heplus_per_msun = load_popiii_heplus_ionizing_photon_table(table)

    np.testing.assert_allclose(ages_myr, np.array([1.0e-2, 1.0]))
    np.testing.assert_allclose(q_heplus_per_msun, np.array([1.0e45, 10.0**44.5]))


def test_popiii_heplus_loader_maps_schaerer_no_emission_sentinel_to_zero(tmp_path) -> None:
    from auroralf.archive.heii1640 import load_popiii_heplus_ionizing_photon_table

    table = tmp_path / "pop3_test_is5.22"
    table.write_text(
        "\n".join(
            [
                "# Star-formation: instantaneous burst at age=0",
                "# L(H_beta) HeII_1640",
                " 4.000E+00 47.0 46.0 45.0 1.0E+34 1 2 3 4 5 1 6 0.5 7 1.0",
                " 6.000E+00 47.0 46.0 -99.0 0.0 1 2 3 4 5 1 6 0.5 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, q_heplus_per_msun = load_popiii_heplus_ionizing_photon_table(table)

    np.testing.assert_array_equal(q_heplus_per_msun, np.array([1.0e45, 0.0]))


def test_heii_and_heplus_caches_reload_same_path_atomic_replacement_and_hit_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auroralf.archive import heii1640

    def table_text(log_q2: str, hbeta: str) -> str:
        return "\n".join(
            [
                "# Star-formation: instantaneous burst at age=0",
                "# L(H_beta) HeII_1640",
                f" 4.000E+00 47.0 46.0 {log_q2} {hbeta} 1 2 3 4 5 1 6 0.5 7 1.0",
                " 6.000E+00 47.0 46.0 45.0 1.0E+34 1 2 3 4 5 1 6 0.5 8 1.0",
            ]
        ) + "\n"

    path = tmp_path / "pop3_test_is5.22"
    replacement = tmp_path / "replacement_is5.22"
    path.write_text(table_text("45.0", "1.0E+34"), encoding="utf-8")
    heii1640._load_popiii_heii1640_luminosity_table_cached.cache_clear()
    heii1640._load_popiii_heplus_ionizing_photon_table_cached.cache_clear()
    real_loadtxt = heii1640.np.loadtxt
    reads: list[Path] = []

    def loadtxt_spy(file_path: str) -> np.ndarray:
        reads.append(Path(file_path))
        return real_loadtxt(file_path)

    monkeypatch.setattr(heii1640.np, "loadtxt", loadtxt_spy)
    _, first_line, _ = heii1640.load_popiii_heii1640_luminosity_table(path)
    _, first_q2 = heii1640.load_popiii_heplus_ionizing_photon_table(path)
    heii1640.load_popiii_heii1640_luminosity_table(path)
    heii1640.load_popiii_heplus_ionizing_photon_table(path)
    assert len(reads) == 2
    original_mtime_ns = path.stat().st_mtime_ns

    replacement.write_text(table_text("46.0", "2.0E+34"), encoding="utf-8")
    assert replacement.stat().st_size == path.stat().st_size
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, path)
    os.utime(path, ns=(original_mtime_ns, original_mtime_ns))

    _, replaced_line, _ = heii1640.load_popiii_heii1640_luminosity_table(path)
    _, replaced_q2 = heii1640.load_popiii_heplus_ionizing_photon_table(path)

    assert len(reads) == 4
    assert replaced_line[0] == pytest.approx(first_line[0] * 2.0)
    assert replaced_q2[0] == pytest.approx(first_q2[0] * 10.0)


def test_heii1640_luminosity_from_heplus_rate_applies_nebular_factors() -> None:
    from auroralf.archive.heii1640 import heii1640_luminosity_from_heplus_rate

    q_heplus = np.array([1.0e52, 2.0e52])
    luminosity = heii1640_luminosity_from_heplus_rate(
        q_heplus,
        caseb_erg_per_photon=5.7e-12,
        covering_factor=0.5,
        escape_fraction=0.2,
        photoionization_efficiency=0.25,
    )

    np.testing.assert_allclose(luminosity, q_heplus * 5.7e-12 * 0.5 * 0.8 * 0.25)


def test_heii1640_luminosity_from_heplus_rate_rejects_finite_input_overflow() -> None:
    from auroralf.archive.heii1640 import heii1640_luminosity_from_heplus_rate

    with pytest.raises(RuntimeError, match="computed HeII 1640 luminosity must be finite and non-negative"):
        heii1640_luminosity_from_heplus_rate(
            np.array([1.0e308]),
            caseb_erg_per_photon=1.0e308,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("q_heplus", [1.0e52, True]),
        ("caseb_erg_per_photon", np.bool_(True)),
        ("covering_factor", False),
        ("escape_fraction", np.bool_(False)),
        ("photoionization_efficiency", True),
    ],
)
def test_heii1640_luminosity_rejects_boolean_numeric_inputs_before_cast(
    name: str,
    value: object,
) -> None:
    from auroralf.archive.heii1640 import heii1640_luminosity_from_heplus_rate

    inputs: dict[str, object] = {
        "q_heplus": np.array([1.0e52]),
        "caseb_erg_per_photon": 5.7e-12,
        "covering_factor": 1.0,
        "escape_fraction": 0.0,
        "photoionization_efficiency": 1.0,
    }
    inputs[name] = value

    with pytest.raises(ValueError, match="boolean"):
        heii1640_luminosity_from_heplus_rate(**inputs)  # type: ignore[arg-type]


def test_final_ssp_line_convolution_uses_years_per_gyr() -> None:
    from auroralf.archive.heii1640 import compute_final_ssp_line_luminosity_from_sfr_grid

    t_grid_gyr = np.array([[0.0, 1.0e-3, 2.0e-3]])
    sfr_grid = np.array([[1.0, 1.0, 1.0]])
    active_grid = np.ones_like(sfr_grid, dtype=bool)
    ssp_age_myr = np.array([1.0e-2, 2.0])
    line_luminosity_per_msun = np.array([10.0, 10.0])

    line_luminosity = compute_final_ssp_line_luminosity_from_sfr_grid(
        t_grid_gyr=t_grid_gyr,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_myr=ssp_age_myr,
        ssp_luminosity_per_msun=line_luminosity_per_msun,
        lookback_max_myr=10.0,
    )

    np.testing.assert_allclose(line_luminosity, np.array([2.0e7]))


def test_final_ssp_line_convolution_rejects_invalid_shapes() -> None:
    from auroralf.archive.heii1640 import compute_final_ssp_line_luminosity_from_sfr_grid

    with pytest.raises(ValueError, match="identical shapes"):
        compute_final_ssp_line_luminosity_from_sfr_grid(
            t_grid_gyr=np.zeros((1, 3)),
            sfr_grid=np.zeros((1, 2)),
            active_grid=np.zeros((1, 3), dtype=bool),
            ssp_age_myr=np.array([1.0]),
            ssp_luminosity_per_msun=np.array([1.0]),
        )


def test_heii_and_heplus_convolution_wrappers_delegate_to_common_engine(monkeypatch) -> None:
    from auroralf.archive import heii1640

    assert hasattr(heii1640, "compute_final_ssp_observable_from_sfr_grid")
    calls: list[dict[str, object]] = []

    def fake_common_engine(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.array([float(len(calls))])

    monkeypatch.setattr(
        heii1640,
        "compute_final_ssp_observable_from_sfr_grid",
        fake_common_engine,
    )
    common = {
        "t_grid_gyr": np.array([[0.0, 1.0e-3]]),
        "sfr_grid": np.ones((1, 2)),
        "active_grid": np.ones((1, 2), dtype=bool),
        "ssp_age_myr": np.array([1.0e-2, 1.0]),
        "lookback_max_myr": 10.0,
    }

    line = heii1640.compute_final_ssp_line_luminosity_from_sfr_grid(
        **common,
        ssp_luminosity_per_msun=np.array([2.0, 3.0]),
    )
    q2 = heii1640.compute_final_ssp_heplus_rate_from_sfr_grid(
        **common,
        ssp_q_heplus_per_msun=np.array([4.0, 5.0]),
    )

    np.testing.assert_array_equal(line, np.array([1.0]))
    np.testing.assert_array_equal(q2, np.array([2.0]))
    np.testing.assert_array_equal(
        calls[0]["ssp_observable_per_msun"],
        np.array([2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        calls[1]["ssp_observable_per_msun"],
        np.array([4.0, 5.0]),
    )


def test_heii_and_heplus_wrappers_match_common_engine_for_single_source_bin() -> None:
    from auroralf.archive.heii1640 import (
        compute_final_ssp_heplus_rate_from_sfr_grid,
        compute_final_ssp_line_luminosity_from_sfr_grid,
        compute_final_ssp_observable_from_sfr_grid,
    )

    common = {
        "t_grid_gyr": np.array([[0.0, 1.0e-3, 2.0e-3]]),
        "sfr_grid": np.array([[5.0, 2.0, 5.0]]),
        "active_grid": np.array([[False, True, False]]),
        "ssp_age_myr": np.array([1.0e-2, 2.0]),
        "lookback_max_myr": 10.0,
    }
    line_kernel = np.array([3.0, 3.0])
    q2_kernel = np.array([4.0, 4.0])

    expected_line = compute_final_ssp_observable_from_sfr_grid(
        **common,
        ssp_observable_per_msun=line_kernel,
    )
    expected_q2 = compute_final_ssp_observable_from_sfr_grid(
        **common,
        ssp_observable_per_msun=q2_kernel,
    )
    line = compute_final_ssp_line_luminosity_from_sfr_grid(
        **common,
        ssp_luminosity_per_msun=line_kernel,
    )
    q2 = compute_final_ssp_heplus_rate_from_sfr_grid(
        **common,
        ssp_q_heplus_per_msun=q2_kernel,
    )

    np.testing.assert_allclose(line, expected_line)
    np.testing.assert_allclose(q2, expected_q2)
    np.testing.assert_allclose(line, np.array([6.0e6]))
    np.testing.assert_allclose(q2, np.array([8.0e6]))


@pytest.mark.parametrize(
    ("failure_mode", "loader_name"),
    [
        ("age", "load_popiii_heii1640_luminosity_table"),
        ("line", "load_popiii_heii1640_luminosity_table"),
        ("q2", "load_popiii_heplus_ionizing_photon_table"),
    ],
)
def test_schaerer_loaders_reject_computed_overflow(
    tmp_path,
    failure_mode: str,
    loader_name: str,
) -> None:
    import auroralf.archive.heii1640 as archived_heii

    if failure_mode == "age":
        rows = [
            " 4.000E+02 47.0 46.0 45.0 1.0 1 2 3 4 5 1 6 0.5 7 1.0",
            " 4.010E+02 47.0 46.0 45.0 1.0 1 2 3 4 5 1 6 0.5 8 1.0",
        ]
    elif failure_mode == "line":
        rows = [
            " 4.000E+00 47.0 46.0 45.0 1.0E+308 1 2 3 4 5 1 6 0.5 7 1.0E+308",
            " 6.000E+00 47.0 46.0 45.0 1.0E+308 1 2 3 4 5 1 6 0.5 8 1.0E+308",
        ]
    else:
        rows = [
            " 4.000E+00 47.0 46.0 400.0 1.0 1 2 3 4 5 1 6 0.5 7 1.0",
            " 6.000E+00 47.0 46.0 401.0 1.0 1 2 3 4 5 1 6 0.5 8 1.0",
        ]
    table = tmp_path / f"pop3_overflow_{failure_mode}_is5.22"
    table.write_text(
        "\n".join(
            [
                "# Star-formation: instantaneous burst at age=0",
                "# L(H_beta) HeII_1640",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loader = getattr(archived_heii, loader_name)
    with pytest.raises(RuntimeError, match="computed .* must be finite"):
        loader(table)


def test_hebe_comparison_defaults_match_extreme_popiii_ssp(monkeypatch) -> None:
    from scripts.experiments import archived_compare_popiii_heii_to_hebe

    monkeypatch.setattr("sys.argv", ["archived_compare_popiii_heii_to_hebe.py"])
    args = archived_compare_popiii_heii_to_hebe._parse_args()

    assert args.popiii_uv_ssp_file.name == "pop3_ge0_sal_500_050_is4.25"
    assert args.heii_ssp_file.name == "pop3_ge0_sal_500_050_is4.22"
    assert args.heii_caseb_erg_per_photon == pytest.approx(5.7e-12)
    assert args.heplus_covering_factor == pytest.approx(1.0)
    assert args.heplus_escape_fraction == pytest.approx(0.0)
    assert args.heii_photoionization_efficiency == pytest.approx(1.0)
    assert args.enable_archived_heii is False
    with pytest.raises(RuntimeError, match="He II implementation is archived"):
        archived_compare_popiii_heii_to_hebe._validate_args(args)
    args.enable_archived_heii = True
    archived_compare_popiii_heii_to_hebe._validate_args(args)


def test_heii_is_absent_from_public_ssp_api() -> None:
    import auroralf.ssp as ssp

    archived_names = (
        "DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON",
        "DEFAULT_POPIII_HEII1640_SSP_FILE",
        "compute_final_ssp_heplus_rate_from_sfr_grid",
        "compute_final_ssp_line_luminosity_from_sfr_grid",
        "heii1640_luminosity_from_heplus_rate",
        "load_popiii_heii1640_luminosity_table",
        "load_popiii_heplus_ionizing_photon_table",
    )
    assert all(not hasattr(ssp, name) for name in archived_names)


def test_hebe_observation_loader_reads_literature_constraints(tmp_path) -> None:
    from scripts.experiments.archived_compare_popiii_heii_to_hebe import _load_hebe_observation_constraints

    table = tmp_path / "hebe.csv"
    table.write_text(
        "\n".join(
            [
                "quantity,component,value,err,unit,source,notes",
                "heii1640_flux_observed,total,1.11e-19,1.7e-20,erg s^-1 cm^-2,Maiolino2026,Table 1",
                "heii1640_luminosity_intrinsic_clean,C1,5.1e40,0.9e40,erg s^-1,Maiolino2026,Table 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    constraints = _load_hebe_observation_constraints(table)

    assert constraints[("heii1640_flux_observed", "total")]["value"] == pytest.approx(1.11e-19)
    assert constraints[("heii1640_flux_observed", "total")]["err"] == pytest.approx(1.7e-20)
    assert constraints[("heii1640_luminosity_intrinsic_clean", "C1")]["unit"] == "erg s^-1"
