"""Continuous-IMF PISN yields and stellar-lifetime delay kernels.

Masses are numerical values in Msun; ages are Myr. Rates are events per
initial stellar Msun per source-frame year. These are ensemble expectations,
not realizations of a mass-conserving discrete stellar population.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .artifacts import require


@dataclass(frozen=True)
class LognormalIMF:
    """dN/dM proportional to exp[-ln(M/Mc)^2/(2 sigma^2)]/M."""

    lower: float = 1.0
    upper: float = 500.0
    characteristic: float = 60.0
    sigma: float = 1.0

    def __post_init__(self):
        require(
            np.isfinite([self.lower, self.upper, self.characteristic, self.sigma]).all()
            and 0 < self.lower < self.upper
            and self.characteristic > 0
            and self.sigma > 0,
            "invalid lognormal IMF parameters",
        )

    def moment(self, lower, upper, power=0):
        """Integral of M**power times the unnormalized number IMF."""
        lo, hi = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float))
        require(np.isfinite(lo).all() and np.isfinite(hi).all(), "nonfinite mass bounds")
        require(np.all((lo >= self.lower) & (hi <= self.upper) & (hi >= lo)), "IMF bounds")
        shift = power * self.sigma**2
        zlo = (np.log(lo / self.characteristic) - shift) / self.sigma
        zhi = (np.log(hi / self.characteristic) - shift) / self.sigma
        factor = (
            self.characteristic**power
            * np.exp(0.5 * (power * self.sigma) ** 2)
            * self.sigma
            * np.sqrt(2 * np.pi)
        )
        return factor * (ndtr(zhi) - ndtr(zlo))

    @property
    def normalization(self):
        return 1 / self.moment(self.lower, self.upper, 1)

    def number_density(self, mass):
        """Number per initial stellar Msun per progenitor Msun."""
        m = np.asarray(mass, float)
        require(np.isfinite(m).all() and np.all(m > 0), "invalid stellar mass")
        return np.where(
            (m >= self.lower) & (m <= self.upper),
            self.normalization
            * np.exp(-0.5 * (np.log(m / self.characteristic) / self.sigma) ** 2)
            / m,
            0.0,
        )

    def yield_per_msun(self, lower=140.0, upper=260.0):
        return float(self.normalization * self.moment(lower, upper))


@dataclass(frozen=True)
class LifetimeKernel:
    """Piecewise power-law lifetime interpolation; no extrapolation.

    Marigo et al. (2003) Table 1 supplies H and He burning durations up to
    central C ignition. Their sum approximates the delay to explosion; the
    short subsequent stages are not modeled. This is not a lifetime from the
    UV SSP and is not a self-consistent stellar-fate calculation.
    """

    mass: np.ndarray
    lifetime_myr: np.ndarray
    imf: LognormalIMF = LognormalIMF()
    lower: float = 140.0
    upper: float = 260.0

    def __post_init__(self):
        m, t = np.asarray(self.mass, float), np.asarray(self.lifetime_myr, float)
        require(m.ndim == 1 and len(m) >= 2 and t.shape == m.shape, "lifetime table shape")
        require(np.isfinite(m).all() and np.isfinite(t).all(), "nonfinite lifetime table")
        require(np.all(m > 0) and np.all(t > 0), "nonpositive lifetime table")
        require(np.all(np.diff(m) > 0) and np.all(np.diff(t) < 0), "nonmonotone lifetimes")
        require(m[0] <= self.lower < self.upper <= m[-1], "lifetime table coverage")
        self.imf.yield_per_msun(self.lower, self.upper)
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "lifetime_myr", t)

    def lifetime(self, mass):
        m = np.asarray(mass, float)
        require(
            np.isfinite(m).all() and np.all((m >= self.mass[0]) & (m <= self.mass[-1])),
            "lifetime mass domain",
        )
        return np.exp(np.interp(np.log(m), np.log(self.mass), np.log(self.lifetime_myr)))

    @property
    def delay_bounds_myr(self):
        return self.lifetime(np.array([self.upper, self.lower]))

    def inverse(self, age):
        return np.exp(
            np.interp(np.log(age), np.log(self.lifetime_myr[::-1]), np.log(self.mass[::-1]))
        )

    def cumulative(self, age_myr):
        """Events per initial stellar Msun by age, including pre-birth age < 0."""
        a = np.asarray(age_myr, float)
        require(np.isfinite(a).all(), "nonfinite age")
        tmin, tmax = self.delay_bounds_myr
        m = np.clip(self.inverse(np.clip(a, tmin, tmax)), self.lower, self.upper)
        return self.imf.normalization * self.imf.moment(m, self.upper)

    def rate(self, age_myr, window_myr=0.0):
        """Instantaneous or trailing-window mean delay kernel.

        A finite window diagnoses sensitivity to the saved age distribution;
        it does not regenerate histories at finer time resolution.
        """
        a = np.asarray(age_myr, float)
        require(np.isfinite(a).all(), "nonfinite age")
        require(np.isfinite(window_myr) and window_myr >= 0, "invalid rate window")
        if window_myr > 0:
            return (self.cumulative(a) - self.cumulative(a - window_myr)) / (window_myr * 1e6)
        tmin, tmax = self.delay_bounds_myr
        active = (a >= tmin) & (a <= tmax)
        safe_age = np.clip(a, tmin, tmax)
        m = self.inverse(safe_age)
        slopes = np.diff(np.log(self.lifetime_myr)) / np.diff(np.log(self.mass))
        index = np.clip(np.searchsorted(self.mass, m, side="right") - 1, 0, len(slopes) - 1)
        return np.where(
            active, self.imf.number_density(m) * m / (np.abs(slopes[index]) * safe_age * 1e6), 0.0
        )


def load_marigo_kernel(path: str | Path, *, hydrogen_only=False):
    table = np.genfromtxt(path, delimiter=",", names=True)
    require(table.dtype.names == ("mass_msun", "hydrogen_yr", "helium_yr"), "lifetime columns")
    time = table["hydrogen_yr"] + (0 if hydrogen_only else table["helium_yr"])
    return LifetimeKernel(table["mass_msun"], time / 1e6)


def observer_rate_per_deg2(source_rate, z, cosmology):
    """Events / observer year / square degree / unit redshift, before selection."""
    require(np.isfinite(z) and z >= 0, "invalid redshift")
    rate = np.asarray(source_rate, float)
    require(np.isfinite(rate).all() and np.all(rate >= 0), "invalid source rate")
    volume = cosmology.differential_comoving_volume(z).value * (np.pi / 180) ** 2
    return rate * volume / (1 + z)
