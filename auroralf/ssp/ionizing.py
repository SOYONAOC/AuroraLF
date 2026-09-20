"""Hydrogen-ionizing SSP rates and age-integrated photon yields.

Rates are intrinsic photons/s/initial Msun; yields are photons/initial Msun.
Interpolation is linear in log(age), as in the UV calculation. Below the
first age we explicitly hold the first rate constant; older ages raise.
No escape fraction, duty probability, or halo abundance is included here.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.constants import c, h

from .uv1600 import _reconstruct_popiii_schaerer_age_grid_from_sfh_code

SECONDS_PER_MYR = u.Myr.to(u.s)
# BPASS manual's luminosity convention, rather than Astropy's nominal Lsun.
BPASS_LSUN_ERG_S = 3.848e33


@dataclass(frozen=True)
class IonizingKernel:
    age_myr: np.ndarray
    rate_per_msun: np.ndarray

    def __post_init__(self):
        age = np.array(self.age_myr, dtype=np.float64, copy=True)
        rate = np.array(self.rate_per_msun, dtype=np.float64, copy=True)
        if (
            age.ndim != 1
            or len(age) < 2
            or rate.shape != age.shape
            or not np.isfinite(age).all()
            or not np.isfinite(rate).all()
            or np.any(age <= 0)
            or np.any(np.diff(age) <= 0)
            or np.any(rate < 0)
        ):
            raise ValueError("invalid ionizing SSP age/rate grid")
        age.flags.writeable = rate.flags.writeable = False
        object.__setattr__(self, "age_myr", age)
        object.__setattr__(self, "rate_per_msun", rate)

    def _ages(self, age_myr):
        age = np.asarray(age_myr, dtype=np.float64)
        if (
            not np.isfinite(age).all()
            or np.any(age < 0)
            or np.any(age > self.age_myr[-1])
        ):
            raise ValueError("requested age lies outside ionizing SSP coverage")
        return age

    def rate(self, age_myr):
        age = self._ages(age_myr)
        return np.interp(
            np.log(np.maximum(age, self.age_myr[0])),
            np.log(self.age_myr),
            self.rate_per_msun,
        )

    def yield_photons(self, age_myr):
        """Exact time integral of the piecewise log-age-linear rate, from age 0."""
        age = self._ages(age_myr)
        a, q = self.age_myr, self.rate_per_msun
        slopes = np.diff(q) / np.log(a[1:] / a[:-1])
        widths = np.diff(a)
        integrals = q[:-1] * widths + slopes * (a[1:] * np.log(a[1:] / a[:-1]) - widths)
        prefix = np.r_[q[0] * a[0], q[0] * a[0] + np.cumsum(integrals)]
        index = np.clip(np.searchsorted(a, age, side="right") - 1, 0, len(a) - 2)
        upper = np.maximum(age, a[0])
        width = upper - a[index]
        partial = q[index] * width + slopes[index] * (
            upper * np.log1p(width / a[index]) - width
        )
        result = np.where(age < a[0], age * q[0], prefix[index] + partial)
        return result * SECONDS_PER_MYR


def load_popiii_ionizing_kernel(path: str | Path) -> IonizingKernel:
    """Read intrinsic Q_0 from the matching Raiter logE .20 table (unit burst)."""
    path = Path(path)
    if path.name != "pop3_ge0_logE_500_001_is5.20":
        raise ValueError("expected current Pop III logE 1--500 Msun .20 SSP")
    header = "\n".join(path.read_text().splitlines()[:15])
    for token in (
        "total mass= 1.0E+00",
        "instantaneous burst at age=0",
        "Z=0.",
        "log(Q_0)",
    ):
        if token not in header:
            raise ValueError(f"missing SSP provenance: {token}")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 25 or not np.isfinite(data).all():
        raise ValueError("invalid Pop III .20 table")
    age = _reconstruct_popiii_schaerer_age_grid_from_sfh_code(str(path), len(data))
    # Printed log ages are rounded and duplicated. Require agreement with the
    # documented reconstructed grid to within the table's rounding precision.
    if not np.allclose(data[:, 0], np.log10(age * 1e6), rtol=0, atol=0.0045):
        raise ValueError("Pop III printed ages disagree with is5 reconstruction")
    q = np.where(data[:, 2] <= -99, 0.0, 10.0 ** data[:, 2])
    return IonizingKernel(age, q)


def load_bpass_ionizing_kernel(path: str | Path) -> IonizingKernel:
    """Integrate the current BPASS stellar SED at integer wavelengths 1--911 A.

    BPASS v2.3 SED bins are 1 A, Lsun/A per 1e6 Msun instantaneous burst.
    Sum bin photon counts L_lambda * delta_lambda / (hc/lambda). The finite
    1 A sampling is retained, with 912 A (redward of the H I edge) excluded.
    The initial 0--1 Myr rate is held at the youngest BPASS SSP value.
    """
    path = Path(path)
    if path.name != "spectra-bin-imf135_300.BASEL.z001.a+00.dat":
        raise ValueError("expected current canonical Pop II BPASS BASEL z001 spectrum")
    data = np.loadtxt(path, max_rows=912)
    if (
        data.shape != (912, 52)
        or not np.array_equal(data[:, 0], np.arange(1, 913))
        or not np.isfinite(data).all()
        or np.any(data[:, 1:] < 0)
    ):
        raise ValueError("invalid BPASS wavelength grid or stellar spectrum")
    photon_energy = (h * c / (data[:-1, 0] * u.AA)).to_value(u.erg)
    q = np.sum(data[:-1, 1:] / photon_energy[:, None], axis=0) * BPASS_LSUN_ERG_S / 1e6
    return IonizingKernel(10.0 ** (0.1 * np.arange(51)), q)


def cumulative_photons_from_sfh(t_gyr, sfr_msun_yr, active, kernel: IonizingKernel):
    """Integrate birth SFR times the SSP's lifetime-to-observation photon yield.

    This exchanges the two integrals in the rate convolution, avoiding a
    quadratic history-by-history emission calculation. SFR is linear between
    nodes after inactive-node masking, matching the existing UV convention.
    Eight-point Gaussian quadrature resolves each segment; SSP coverage is
    validated even for inactive intervals. This is a main-branch integral,
    not a reconstruction of unresolved merger progenitors.
    """
    t, sfr = np.asarray(t_gyr, float), np.asarray(sfr_msun_yr, float)
    active = np.asarray(active)
    if (
        t.ndim != 2
        or t.shape[1] < 2
        or sfr.shape != t.shape
        or active.shape != t.shape
        or active.dtype != np.dtype(bool)
        or not np.isfinite(t).all()
        or not np.isfinite(sfr).all()
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(sfr < 0)
    ):
        raise ValueError("invalid star-formation history")
    kernel._ages((t[:, -1] - t[:, 0]) * 1000)
    sfr = np.where(active, sfr, 0.0)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    width = np.diff(t, axis=1)
    result = np.zeros(len(t))
    for node, weight in zip(nodes, weights, strict=True):
        fraction = (node + 1) / 2
        birth = t[:, :-1] + width * fraction
        rate = sfr[:, :-1] + np.diff(sfr, axis=1) * fraction
        photons = kernel.yield_photons((t[:, -1, None] - birth) * 1000)
        result += np.sum(rate * photons * width * (weight / 2) * 1e9, axis=1)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError("invalid cumulative photon budget")
    return result
