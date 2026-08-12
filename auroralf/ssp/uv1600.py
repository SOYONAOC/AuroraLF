"""Load UV SSP kernels used by AuroraLF.

The canonical Pop II path uses BPASS binary-population spectra: Eldridge et
al. (2017), DOI: 10.1017/pasa.2017.51, arXiv:1710.02154; Stanway & Eldridge
(2018), DOI: 10.1093/mnras/sty1353, arXiv:1805.08784; and Byrne et al. (2022),
DOI: 10.1093/mnras/stac807, arXiv:2203.13275.  The production selection is the
BPASS v2.3 binary ``imf135_300``, BASEL, ``z001`` table.

Pop III UV kernels are from Raiter, Schaerer & Fosbury (2010), DOI:
10.1051/0004-6361/201015236, arXiv:1008.2114.  File and column selection is
explicit; this module does not create a synthetic SSP fallback.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u

from auroralf.file_version import FileVersion

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised only when h5py is absent
    h5py = None


DEFAULT_WAVELENGTH_A = 1600.0
DEFAULT_POPIII_UV_WAVELENGTH_A = 1500.0
DEFAULT_POPIII_UV_COLUMN = "L_1500"
DEFAULT_POPIII_UV_SSP_FILE = (
    "external_data/ssp_spectra/schaerer2010_pop3/"
    "pop3_ge0_sal_500_001_is5.25"
)
MODEL_NORMALIZATION_MSUN = 1.0e6


def _ssp_ages_myr(n_bins: int) -> np.ndarray:
    indices = np.arange(n_bins, dtype=float)
    log_age_yr = 6.0 + 0.1 * indices
    return 10.0 ** (log_age_yr - 6.0)


def _resolve_hdf5_metallicity_index(metallicity_zsun: float, metallicities_dex: np.ndarray) -> int:
    metallicities_zsun = np.power(10.0, metallicities_dex)
    matched = np.isclose(metallicities_zsun, float(metallicity_zsun), rtol=0.0, atol=1.0e-10)
    if np.any(matched):
        return int(np.flatnonzero(matched)[0])

    formatted = ", ".join(
        f"{zsun:g} Zsun (dex={dex:.6f})" for zsun, dex in zip(metallicities_zsun, metallicities_dex, strict=True)
    )
    raise ValueError(
        "metallicity must exactly match one of the discrete HDF5 options in Z/Zsun; "
        f"requested {metallicity_zsun:g}, available: {formatted}"
    )


def _load_uv1600_table_from_dat(file_path: str, wavelength_a: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path)
    wavelength_grid = data[:, 0]
    idx = int(np.argmin(np.abs(wavelength_grid - wavelength_a)))
    l_lambda = data[idx, 1:]
    ages_myr = _ssp_ages_myr(l_lambda.size)

    lum_nu = (l_lambda * (u.L_sun / u.AA)).to(
        u.erg / u.s / u.Hz,
        equivalencies=u.spectral_density(wavelength_grid[idx] * u.AA),
    )
    luminosity_per_msun = lum_nu.value / MODEL_NORMALIZATION_MSUN
    return ages_myr, luminosity_per_msun


def _load_uv1600_table_from_npz(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(file_path)
    if "ages_myr" not in payload or "luminosity_per_msun" not in payload:
        raise ValueError("NPZ SSP files must contain 'ages_myr' and 'luminosity_per_msun' arrays")

    ages_myr = np.asarray(payload["ages_myr"], dtype=float)
    luminosity_per_msun = np.asarray(payload["luminosity_per_msun"], dtype=float)
    if ages_myr.ndim != 1 or luminosity_per_msun.ndim != 1:
        raise ValueError("NPZ SSP arrays must be 1D")
    if ages_myr.size != luminosity_per_msun.size:
        raise ValueError("ages_myr and luminosity_per_msun must have the same length")
    return ages_myr, luminosity_per_msun


def _parse_popiii_uv_table_header(file_path: str, uv_column: str) -> tuple[list[str], bool]:
    labels: list[str] | None = None
    instantaneous = False
    with Path(file_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#"):
                continue
            text = raw_line[1:].strip()
            if "Star-formation:" in text and "instantaneous burst" in text:
                instantaneous = True
            if "log(age)" in text and uv_column in text:
                labels = text.split()

    if not instantaneous:
        raise ValueError("Pop III UV SSP tables must be instantaneous burst models")
    if labels is None:
        raise ValueError(f"Pop III UV SSP table is missing required column {uv_column!r}")
    return labels, instantaneous


def _load_popiii_uv_luminosity_table_from_schaerer(
    file_path: str,
    uv_column: str,
    wavelength_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels, _ = _parse_popiii_uv_table_header(file_path=file_path, uv_column=uv_column)
    age_index = labels.index("log(age)")
    luminosity_index = labels.index(uv_column)
    data = np.loadtxt(file_path)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] <= max(age_index, luminosity_index):
        raise ValueError(f"Pop III UV SSP table does not contain column {uv_column!r}")

    log_age_yr = data[:, age_index]
    log_l_lambda = data[:, luminosity_index]
    if not np.all(np.isfinite(log_age_yr)):
        raise ValueError("Pop III UV SSP ages must be finite")
    if not np.all(np.isfinite(log_l_lambda)):
        raise ValueError("Pop III UV SSP luminosities must be finite")

    ages_myr = np.power(10.0, log_age_yr - 6.0)
    l_lambda = np.power(10.0, log_l_lambda) * (u.erg / u.s / u.AA)
    lum_nu = l_lambda.to(
        u.erg / u.s / u.Hz,
        equivalencies=u.spectral_density(float(wavelength_a) * u.AA),
    )
    luminosity_per_msun = np.asarray(lum_nu.value, dtype=float)

    if np.any(np.diff(ages_myr) <= 0.0):
        ages_myr = _reconstruct_popiii_schaerer_age_grid_from_sfh_code(file_path, data.shape[0])
    else:
        order = np.argsort(ages_myr, kind="stable")
        ages_myr = np.asarray(ages_myr[order], dtype=float)
        luminosity_per_msun = luminosity_per_msun[order]
    if np.any(ages_myr <= 0.0):
        raise ValueError("Pop III UV SSP ages must be positive")
    if np.any(np.diff(ages_myr) <= 0.0):
        raise ValueError("Pop III UV SSP ages must be strictly increasing")
    if np.any(luminosity_per_msun < 0.0):
        raise ValueError("Pop III UV SSP luminosities must be non-negative")
    return ages_myr, luminosity_per_msun


def _reconstruct_popiii_schaerer_age_grid_from_sfh_code(file_path: str, n_rows: int) -> np.ndarray:
    stem = Path(file_path).stem
    if stem.endswith("_is5"):
        step_myr = 1.0
    elif stem.endswith("_is4"):
        step_myr = 0.1
    else:
        raise ValueError(
            "Pop III UV SSP ages are not strictly increasing and the table name does not end with "
            "'_is4' or '_is5', so the documented Schaerer age grid cannot be reconstructed"
        )
    if int(n_rows) < 1:
        raise ValueError("Pop III UV SSP table must contain at least one row")
    if int(n_rows) == 1:
        return np.array([1.0e-2], dtype=float)
    return np.concatenate(
        (
            np.array([1.0e-2], dtype=float),
            np.arange(1, int(n_rows), dtype=float) * step_myr,
        )
    )


def _load_uv1600_table_from_hdf5(
    file_path: str,
    wavelength_a: float,
    metallicity: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to read HDF5 SSP files")
    if metallicity is None:
        raise ValueError("metallicity must be provided in Z/Zsun when loading an HDF5 SSP file")

    with h5py.File(file_path, "r") as handle:
        wavelength_grid = np.asarray(handle["/wavelengths"], dtype=float)
        wavelength_index = int(np.argmin(np.abs(wavelength_grid - wavelength_a)))
        metallicities_dex = np.asarray(handle["/metallicities"], dtype=float)
        metallicity_index = _resolve_hdf5_metallicity_index(metallicity, metallicities_dex)
        ages_myr = np.asarray(handle["/ages"], dtype=float) * 1.0e3
        l_nu = np.asarray(handle["/spectra"][metallicity_index, :, wavelength_index], dtype=float)

    lum_nu = (l_nu * (u.L_sun / u.Hz)).to(u.erg / u.s / u.Hz)
    # These HDF5 templates are already normalized per unit stellar mass.
    luminosity_per_msun = lum_nu.value
    return ages_myr, luminosity_per_msun


@lru_cache(maxsize=None)
def _load_uv1600_table_cached(
    version: FileVersion,
    wavelength_a: float,
    metallicity: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if type(version) is not FileVersion:
        raise TypeError("version must be exactly FileVersion")
    file_path = version.path
    suffix = file_path.suffix.lower()
    if suffix in {".hdf5", ".h5"}:
        return _load_uv1600_table_from_hdf5(file_path=file_path, wavelength_a=wavelength_a, metallicity=metallicity)
    if suffix == ".npz":
        if metallicity is not None:
            raise ValueError("metallicity is only supported for HDF5 SSP files")
        return _load_uv1600_table_from_npz(file_path=file_path)
    if metallicity is not None:
        raise ValueError("metallicity is only supported for HDF5 SSP files")
    return _load_uv1600_table_from_dat(file_path=file_path, wavelength_a=wavelength_a)


@lru_cache(maxsize=None)
def _load_popiii_uv_luminosity_table_cached(
    version: FileVersion,
    uv_column: str,
    wavelength_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    if type(version) is not FileVersion:
        raise TypeError("version must be exactly FileVersion")
    return _load_popiii_uv_luminosity_table_from_schaerer(
        file_path=version.path,
        uv_column=uv_column,
        wavelength_a=wavelength_a,
    )


def load_uv1600_table(
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
    metallicity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a 1600 A SSP luminosity table and return age and luminosity-per-Msun arrays.

    For HDF5 SSP files, ``metallicity`` must be supplied in linear ``Z/Zsun`` and must
    exactly match one of the discrete metallicity bins stored in the file.
    The module docstring records the BPASS model references; selecting a table
    is an AuroraLF configuration choice rather than a new SSP calculation.
    """

    version = FileVersion.from_path(file_path)
    ages_myr, luminosity_per_msun = _load_uv1600_table_cached(
        version,
        float(wavelength_a),
        None if metallicity is None else float(metallicity),
    )
    return ages_myr.copy(), luminosity_per_msun.copy()


def load_popiii_uv_luminosity_table(
    file_path: str | Path = DEFAULT_POPIII_UV_SSP_FILE,
    *,
    uv_column: str = DEFAULT_POPIII_UV_COLUMN,
    wavelength_a: float = DEFAULT_POPIII_UV_WAVELENGTH_A,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a Schaerer/Raiter Pop III instantaneous-burst UV table.

    The Pop III tables store ``log[erg s^-1 A^-1]`` continuum luminosities
    normalized to a one-solar-mass burst. This loader converts the selected UV
    continuum column to ``erg s^-1 Hz^-1 Msun^-1`` for use by the existing SSP
    convolution code.  The source model is Raiter, Schaerer & Fosbury (2010).
    """

    version = FileVersion.from_path(file_path)
    ages_myr, luminosity_per_msun = _load_popiii_uv_luminosity_table_cached(
        version,
        str(uv_column),
        float(wavelength_a),
    )
    return ages_myr.copy(), luminosity_per_msun.copy()


def interpolate_uv1600_luminosity_per_msun(
    time_myr: float | np.ndarray,
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
    metallicity: float | None = None,
) -> float | np.ndarray:
    """Interpolate the 1600 A luminosity per solar mass at the requested SSP age in Myr."""

    ages_myr, luminosity_per_msun = load_uv1600_table(
        file_path=file_path,
        wavelength_a=wavelength_a,
        metallicity=metallicity,
    )
    time_myr_array = np.asarray(time_myr, dtype=float)
    log_ages = np.log10(ages_myr)
    log_time = np.log10(np.clip(time_myr_array, ages_myr[0], ages_myr[-1]))
    interpolated = np.interp(log_time, log_ages, luminosity_per_msun)
    if np.ndim(time_myr) == 0:
        return float(interpolated)
    return interpolated
