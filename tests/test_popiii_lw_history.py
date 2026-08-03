from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.sfr import (
    PopIIISFRParameters,
    compute_popiii_sfr_from_grids,
    load_popiii_lw_background_history,
)


def _write_history(path: Path, rows: list[str]) -> Path:
    path.write_text("redshift,j21_lw_total\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_zeus21_style_lw_history_interpolates_and_drives_popiii_sfr(tmp_path: Path) -> None:
    history = load_popiii_lw_background_history(
        _write_history(tmp_path / "zeus21.csv", ["20,1", "10,9", "15,4"])
    )
    z_grid = np.array([[10.0, 15.0, 20.0]])
    j21_grid = history.interpolate(z_grid)
    np.testing.assert_allclose(j21_grid, np.array([[9.0, 4.0, 1.0]]))

    common = dict(
        mh_grid=np.full((1, 3), 1.0e6),
        dmhdt_sfr_grid=np.full((1, 3), 1.0e8),
        z_grid=z_grid,
        active_grid=np.ones((1, 3), dtype=bool),
        cosmology=Cosmology(),
        parameters=PopIIISFRParameters(upper_mass_mode="fixed", upper_mass_msun=1.0e8),
    )
    no_feedback = compute_popiii_sfr_from_grids(**common)
    external_lw = compute_popiii_sfr_from_grids(
        **common,
        lw_background_j21_grid=j21_grid,
    )

    assert np.all(external_lw.lower_mass_msun_grid > no_feedback.lower_mass_msun_grid)
    assert np.all(external_lw.sfr_grid < no_feedback.sfr_grid)


def test_lw_history_fails_on_extrapolation_or_invalid_values(tmp_path: Path) -> None:
    history = load_popiii_lw_background_history(
        _write_history(tmp_path / "valid.csv", ["10,1", "20,2"])
    )
    with pytest.raises(ValueError, match="outside"):
        history.interpolate(21.0)

    invalid = _write_history(tmp_path / "invalid.csv", ["10,1", "20,-2"])
    with pytest.raises(ValueError, match="non-negative"):
        load_popiii_lw_background_history(invalid)


def test_popiii_sfr_rejects_bad_external_lw_shape_or_values() -> None:
    common = dict(
        mh_grid=np.full((1, 2), 1.0e6),
        dmhdt_sfr_grid=np.full((1, 2), 1.0e8),
        z_grid=np.array([[10.0, 20.0]]),
        active_grid=np.ones((1, 2), dtype=bool),
        cosmology=Cosmology(),
    )
    with pytest.raises(ValueError, match="scalar or match"):
        compute_popiii_sfr_from_grids(**common, lw_background_j21_grid=np.ones(3))
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_popiii_sfr_from_grids(**common, lw_background_j21_grid=np.array([[0.0, np.nan]]))
