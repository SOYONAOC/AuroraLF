from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from auroralf.file_version import FileVersion

from .convolution import (
    DEFAULT_TIME_UNIT_IN_YEARS,
    _reject_boolean_scalar,
    _reject_boolean_values,
    compute_final_ssp_observable_from_sfr_grid,
)


DEFAULT_POPIII_HEII1640_SSP_FILE = (
    "external_data/ssp_spectra/schaerer2010_pop3/"
    "pop3_ge0_sal_500_001_is5.22"
)
DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON = 5.7e-12
SCHAERER_HEPLUS_LOG_Q_COLUMN = 3
SCHAERER_HEII1640_EW_COLUMN = 13
SCHAERER_HEII1640_TO_HBETA_COLUMN = 14
SCHAERER_HBETA_LUMINOSITY_COLUMN = 4


def _parse_popiii_heii1640_table_header(file_path: str) -> None:
    instantaneous = False
    has_heii1640 = False
    has_hbeta = False
    with Path(file_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#"):
                continue
            text = raw_line[1:].strip()
            if "Star-formation:" in text and "instantaneous burst" in text:
                instantaneous = True
            if "HeII_1640" in text:
                has_heii1640 = True
            if "L(H_beta)" in text:
                has_hbeta = True

    if not instantaneous:
        raise ValueError("Pop III HeII 1640 SSP tables must be instantaneous burst models")
    if not has_heii1640:
        raise ValueError("Pop III HeII 1640 SSP table is missing required HeII_1640 columns")
    if not has_hbeta:
        raise ValueError("Pop III HeII 1640 SSP table is missing required L(H_beta) column")


def _reconstruct_popiii_schaerer_age_grid_from_sfh_code(file_path: str, n_rows: int) -> np.ndarray:
    stem = Path(file_path).stem
    if stem.endswith("_is5"):
        step_myr = 1.0
    elif stem.endswith("_is4"):
        step_myr = 0.1
    else:
        raise ValueError(
            "Pop III HeII SSP ages are not strictly increasing and the table name does not end with "
            "'_is4' or '_is5', so the documented Schaerer age grid cannot be reconstructed"
        )
    if int(n_rows) < 1:
        raise ValueError("Pop III HeII SSP table must contain at least one row")
    if int(n_rows) == 1:
        return np.array([1.0e-2], dtype=float)
    return np.concatenate((np.array([1.0e-2], dtype=float), np.arange(1, int(n_rows), dtype=float) * step_myr))


def _load_popiii_heii1640_luminosity_table_from_schaerer(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _parse_popiii_heii1640_table_header(file_path)
    data = np.loadtxt(file_path)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    required_column = max(
        SCHAERER_HBETA_LUMINOSITY_COLUMN,
        SCHAERER_HEII1640_EW_COLUMN,
        SCHAERER_HEII1640_TO_HBETA_COLUMN,
    )
    if data.shape[1] <= required_column:
        raise ValueError("Pop III HeII 1640 SSP table does not contain the required recombination-line columns")

    log_age_yr = data[:, 0]
    hbeta_luminosity = data[:, SCHAERER_HBETA_LUMINOSITY_COLUMN]
    ew_rest_a = data[:, SCHAERER_HEII1640_EW_COLUMN]
    heii1640_to_hbeta = data[:, SCHAERER_HEII1640_TO_HBETA_COLUMN]
    if not np.all(np.isfinite(log_age_yr)):
        raise ValueError("Pop III HeII 1640 SSP ages must be finite")
    if not np.all(np.isfinite(hbeta_luminosity)):
        raise ValueError("Pop III HeII 1640 Hbeta luminosities must be finite")
    if not np.all(np.isfinite(ew_rest_a)):
        raise ValueError("Pop III HeII 1640 equivalent widths must be finite")
    if not np.all(np.isfinite(heii1640_to_hbeta)):
        raise ValueError("Pop III HeII 1640 intensity ratios must be finite")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ages_myr = np.power(10.0, log_age_yr - 6.0)
        luminosity_per_msun = hbeta_luminosity * heii1640_to_hbeta
    if not np.all(np.isfinite(ages_myr)):
        raise RuntimeError("computed Pop III HeII 1640 SSP ages must be finite")
    if not np.all(np.isfinite(luminosity_per_msun)):
        raise RuntimeError("computed Pop III HeII 1640 SSP luminosities must be finite")
    if np.any(luminosity_per_msun < 0.0):
        raise RuntimeError("computed Pop III HeII 1640 SSP luminosities must be non-negative")
    if np.any(np.diff(ages_myr) <= 0.0):
        ages_myr = _reconstruct_popiii_schaerer_age_grid_from_sfh_code(file_path, data.shape[0])
    else:
        order = np.argsort(ages_myr, kind="stable")
        ages_myr = np.asarray(ages_myr[order], dtype=float)
        luminosity_per_msun = np.asarray(luminosity_per_msun[order], dtype=float)
        ew_rest_a = np.asarray(ew_rest_a[order], dtype=float)

    if np.any(ages_myr <= 0.0):
        raise ValueError("Pop III HeII 1640 SSP ages must be positive")
    if np.any(np.diff(ages_myr) <= 0.0):
        raise ValueError("Pop III HeII 1640 SSP ages must be strictly increasing")
    if np.any(luminosity_per_msun < 0.0):
        raise ValueError("Pop III HeII 1640 SSP luminosities must be non-negative")
    if np.any(ew_rest_a < 0.0):
        raise ValueError("Pop III HeII 1640 equivalent widths must be non-negative")
    return ages_myr, np.asarray(luminosity_per_msun, dtype=float), np.asarray(ew_rest_a, dtype=float)


def _load_popiii_heplus_ionizing_photon_table_from_schaerer(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    _parse_popiii_heii1640_table_header(file_path)
    data = np.loadtxt(file_path)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] <= SCHAERER_HEPLUS_LOG_Q_COLUMN:
        raise ValueError("Pop III HeII SSP table does not contain the required log(Q_2) column")

    log_age_yr = data[:, 0]
    log_q_heplus = data[:, SCHAERER_HEPLUS_LOG_Q_COLUMN]
    if not np.all(np.isfinite(log_age_yr)):
        raise ValueError("Pop III HeII SSP ages must be finite")
    if not np.all(np.isfinite(log_q_heplus)):
        raise ValueError("Pop III He+ ionizing photon rates must be finite")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ages_myr = np.power(10.0, log_age_yr - 6.0)
        q_heplus_per_msun = np.power(10.0, log_q_heplus)
    if not np.all(np.isfinite(ages_myr)):
        raise RuntimeError("computed Pop III He+ SSP ages must be finite")
    if not np.all(np.isfinite(q_heplus_per_msun)):
        raise RuntimeError("computed Pop III He+ ionizing photon rates must be finite")
    if np.any(q_heplus_per_msun < 0.0):
        raise RuntimeError("computed Pop III He+ ionizing photon rates must be non-negative")
    if np.any(np.diff(ages_myr) <= 0.0):
        ages_myr = _reconstruct_popiii_schaerer_age_grid_from_sfh_code(file_path, data.shape[0])
    else:
        order = np.argsort(ages_myr, kind="stable")
        ages_myr = np.asarray(ages_myr[order], dtype=float)
        q_heplus_per_msun = np.asarray(q_heplus_per_msun[order], dtype=float)

    if np.any(ages_myr <= 0.0):
        raise ValueError("Pop III He+ SSP ages must be positive")
    if np.any(np.diff(ages_myr) <= 0.0):
        raise ValueError("Pop III He+ SSP ages must be strictly increasing")
    if np.any(q_heplus_per_msun < 0.0):
        raise ValueError("Pop III He+ ionizing photon rates must be non-negative")
    return ages_myr, np.asarray(q_heplus_per_msun, dtype=float)


@lru_cache(maxsize=None)
def _load_popiii_heii1640_luminosity_table_cached(
    version: FileVersion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(version) is not FileVersion:
        raise TypeError("version must be exactly FileVersion")
    return _load_popiii_heii1640_luminosity_table_from_schaerer(version.path)


@lru_cache(maxsize=None)
def _load_popiii_heplus_ionizing_photon_table_cached(
    version: FileVersion,
) -> tuple[np.ndarray, np.ndarray]:
    if type(version) is not FileVersion:
        raise TypeError("version must be exactly FileVersion")
    return _load_popiii_heplus_ionizing_photon_table_from_schaerer(version.path)


def load_popiii_heii1640_luminosity_table(
    file_path: str | Path = DEFAULT_POPIII_HEII1640_SSP_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Pop III instantaneous-burst HeII 1640 table.

    Returns SSP age in ``Myr``, HeII 1640 luminosity in ``erg s^-1 Msun^-1``,
    and rest-frame equivalent width in Angstrom. The Schaerer/Raiter ``*.22``
    tables give ``L(H_beta)`` and ``I(HeII_1640)/I(H_beta)``; their product is
    the line luminosity per one-solar-mass burst.
    """

    version = FileVersion.from_path(file_path)
    ages_myr, luminosity_per_msun, ew_rest_a = _load_popiii_heii1640_luminosity_table_cached(version)
    return ages_myr.copy(), luminosity_per_msun.copy(), ew_rest_a.copy()


def load_popiii_heplus_ionizing_photon_table(
    file_path: str | Path = DEFAULT_POPIII_HEII1640_SSP_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Pop III instantaneous-burst He+ ionizing photon rates.

    Returns SSP age in ``Myr`` and ``Q_2`` in ``photon s^-1 Msun^-1``. In the
    Schaerer/Raiter ``*.22`` recombination tables, ``Q_2`` is the photon rate
    above the He+ ionization edge at 54.4 eV.
    """

    version = FileVersion.from_path(file_path)
    ages_myr, q_heplus_per_msun = _load_popiii_heplus_ionizing_photon_table_cached(version)
    return ages_myr.copy(), q_heplus_per_msun.copy()


def heii1640_luminosity_from_heplus_rate(
    q_heplus: float | np.ndarray,
    *,
    caseb_erg_per_photon: float = DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON,
    covering_factor: float = 1.0,
    escape_fraction: float = 0.0,
    photoionization_efficiency: float = 1.0,
) -> float | np.ndarray:
    """Convert absorbed He+ ionizing photons into HeII 1640 luminosity.

    ``q_heplus`` is in ``photon s^-1``. The returned luminosity is
    ``erg s^-1`` and equals
    ``caseb_erg_per_photon * q_heplus * covering_factor *
    (1 - escape_fraction) * photoionization_efficiency``.
    """

    _reject_boolean_values("q_heplus", q_heplus)
    for name, value in (
        ("caseb_erg_per_photon", caseb_erg_per_photon),
        ("covering_factor", covering_factor),
        ("escape_fraction", escape_fraction),
        ("photoionization_efficiency", photoionization_efficiency),
    ):
        _reject_boolean_scalar(name, value)

    q_array = np.asarray(q_heplus, dtype=float)
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q_heplus must contain only finite values")
    if np.any(q_array < 0.0):
        raise ValueError("q_heplus must be non-negative")
    if not np.isfinite(float(caseb_erg_per_photon)) or float(caseb_erg_per_photon) <= 0.0:
        raise ValueError("caseb_erg_per_photon must be positive")
    if not np.isfinite(float(covering_factor)) or not 0.0 <= float(covering_factor) <= 1.0:
        raise ValueError("covering_factor must lie in [0, 1]")
    if not np.isfinite(float(escape_fraction)) or not 0.0 <= float(escape_fraction) <= 1.0:
        raise ValueError("escape_fraction must lie in [0, 1]")
    if not np.isfinite(float(photoionization_efficiency)) or float(photoionization_efficiency) < 0.0:
        raise ValueError("photoionization_efficiency must be non-negative")

    with np.errstate(over="ignore", invalid="ignore"):
        luminosity = (
            q_array
            * float(caseb_erg_per_photon)
            * float(covering_factor)
            * (1.0 - float(escape_fraction))
            * float(photoionization_efficiency)
        )
    if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0.0):
        raise RuntimeError("computed HeII 1640 luminosity must be finite and non-negative")
    if np.ndim(q_heplus) == 0:
        return float(luminosity)
    return luminosity


def compute_final_ssp_line_luminosity_from_sfr_grid(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_myr: np.ndarray,
    ssp_luminosity_per_msun: np.ndarray,
    lookback_max_myr: float = 10.0,
    time_unit_in_years: float = DEFAULT_TIME_UNIT_IN_YEARS,
) -> np.ndarray:
    """Convolve final-time SFR histories with an SSP line-luminosity kernel.

    ``sfr_grid`` is in ``Msun yr^-1`` and ``ssp_luminosity_per_msun`` is in
    ``erg s^-1 Msun^-1``. The returned line luminosity is ``erg s^-1``.
    """

    return compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=t_grid_gyr,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_myr=ssp_age_myr,
        ssp_observable_per_msun=ssp_luminosity_per_msun,
        lookback_max_myr=lookback_max_myr,
        time_unit_in_years=time_unit_in_years,
    )


def compute_final_ssp_heplus_rate_from_sfr_grid(
    *,
    t_grid_gyr: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_myr: np.ndarray,
    ssp_q_heplus_per_msun: np.ndarray,
    lookback_max_myr: float = 10.0,
    time_unit_in_years: float = DEFAULT_TIME_UNIT_IN_YEARS,
) -> np.ndarray:
    """Convolve final-time SFR histories with an SSP He+ photon-rate kernel.

    ``sfr_grid`` is in ``Msun yr^-1`` and ``ssp_q_heplus_per_msun`` is in
    ``photon s^-1 Msun^-1``. The returned quantity is ``photon s^-1``.
    """

    return compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=t_grid_gyr,
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        ssp_age_myr=ssp_age_myr,
        ssp_observable_per_msun=ssp_q_heplus_per_msun,
        lookback_max_myr=lookback_max_myr,
        time_unit_in_years=time_unit_in_years,
    )


__all__ = [
    "DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON",
    "DEFAULT_POPIII_HEII1640_SSP_FILE",
    "compute_final_ssp_heplus_rate_from_sfr_grid",
    "compute_final_ssp_line_luminosity_from_sfr_grid",
    "heii1640_luminosity_from_heplus_rate",
    "load_popiii_heii1640_luminosity_table",
    "load_popiii_heplus_ionizing_photon_table",
]
