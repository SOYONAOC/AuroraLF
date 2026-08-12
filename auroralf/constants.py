"""Shared numerical constants for AuroraLF.

This module is the single source of truth for unit conversions, physical
constants, and the fiducial cosmologies used by production and simulation-cache
workflows. Runtime TOML files still record explicit values for provenance.

Citation provenance
-------------------
The production cosmology is built from Astropy's Planck18 realization of
Planck Collaboration VI (2020), DOI: 10.1051/0004-6361/201833910,
arXiv:1807.06209.  The solar abundance constants follow Asplund et al. (2009),
DOI: 10.1146/annurev.astro.46.060407.145222, arXiv:0909.0948.  Astropy supplies
the dimensional physical constants and unit conversions; AuroraLF does not
copy rounded numerical values for those quantities.
"""

from __future__ import annotations

from math import log10, pi

import astropy.units as u
from astropy.constants import G, k_B, m_p
from astropy.cosmology import FlatLambdaCDM, Planck15, Planck18


# Unit conversions and physical constants.
SECONDS_PER_GYR = float(u.Gyr.to(u.s))
YEARS_PER_GYR = float(u.Gyr.to(u.yr))
KM_PER_MPC = float(u.Mpc.to(u.km))
GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN = float(
    G.to_value(u.Mpc * u.km**2 / (u.s**2 * u.Msun))
)
PROTON_MASS_KG = float(m_p.to_value(u.kg))
BOLTZMANN_CONSTANT_J_K = float(k_B.to_value(u.J / u.K))
_AB_ZERO_FLUX_DENSITY = (0.0 * u.ABmag).to(u.erg / (u.s * u.cm**2 * u.Hz))
_AB_REFERENCE_LUMINOSITY = (
    4.0 * pi * (10.0 * u.pc) ** 2 * _AB_ZERO_FLUX_DENSITY
).to_value(u.erg / (u.s * u.Hz))
AB_ZEROPOINT_LNU = 2.5 * log10(float(_AB_REFERENCE_LUMINOSITY))
# Asplund et al. (2009): proto-solar bulk Z=0.0142 and photospheric
# 12+log10(O/H)=8.69.  These are distinct abundance conventions.
SOLAR_METALLICITY_MASS_FRACTION = 0.0142
SOLAR_OXYGEN_ABUNDANCE = 8.69

# Direct from the Appendix-A MAH parameter mixture of McBride, Fakhouri & Ma
# (2009), DOI: 10.1111/j.1365-2966.2009.15329.x, arXiv:0902.3659.
POWER_LAW_FRACTION = 0.0466

# Fiducial Astropy Planck18 cosmology used by production. AuroraLF's analytic
# equations neglect radiation, so the shared flat model explicitly uses
# Tcmb0=0 K while retaining Astropy Planck18 H0, Om0, and Ob0.
_AURORALF_PLANCK18 = FlatLambdaCDM(
    H0=Planck18.H0,
    Om0=Planck18.Om0,
    Ob0=Planck18.Ob0,
    Tcmb0=0.0 * u.K,
    name="AuroraLF Planck18 no-radiation",
)
PLANCK18_H = float(_AURORALF_PLANCK18.h)
PLANCK18_H0_KM_S_MPC = float(_AURORALF_PLANCK18.H0.to_value(u.km / (u.s * u.Mpc)))
PLANCK18_OMEGA_M = float(_AURORALF_PLANCK18.Om0)
PLANCK18_OMEGA_B = float(_AURORALF_PLANCK18.Ob0)
PLANCK18_OMEGA_LAMBDA = float(_AURORALF_PLANCK18.Ode0)
PLANCK18_SIGMA8 = float(Planck18.meta["sigma8"])
PLANCK18_NS = float(Planck18.meta["n"])
PLANCK18_H0_GYR = float(_AURORALF_PLANCK18.H0.to_value(u.Gyr**-1))

# TNG/THESAN use Astropy Planck15 H0 and Ob0, but their published simulation
# matter density is Om0=0.3089 rather than Astropy Planck15's Om0=0.3075.
_TNG_THESAN_COSMOLOGY = FlatLambdaCDM(
    H0=Planck15.H0,
    Om0=0.3089,
    Ob0=Planck15.Ob0,
    Tcmb0=0.0 * u.K,
    name="TNG/THESAN simulation cosmology",
)
PLANCK15_H = float(_TNG_THESAN_COSMOLOGY.h)
PLANCK15_H0_KM_S_MPC = float(
    _TNG_THESAN_COSMOLOGY.H0.to_value(u.km / (u.s * u.Mpc))
)
PLANCK15_OMEGA_M = 0.3089
PLANCK15_OMEGA_B = float(_TNG_THESAN_COSMOLOGY.Ob0)
PLANCK15_OMEGA_LAMBDA = float(_TNG_THESAN_COSMOLOGY.Ode0)
PLANCK15_H0_GYR = float(_TNG_THESAN_COSMOLOGY.H0.to_value(u.Gyr**-1))


__all__ = [
    "AB_ZEROPOINT_LNU",
    "BOLTZMANN_CONSTANT_J_K",
    "GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN",
    "KM_PER_MPC",
    "PLANCK15_H",
    "PLANCK15_H0_GYR",
    "PLANCK15_H0_KM_S_MPC",
    "PLANCK15_OMEGA_B",
    "PLANCK15_OMEGA_LAMBDA",
    "PLANCK15_OMEGA_M",
    "PLANCK18_H",
    "PLANCK18_H0_GYR",
    "PLANCK18_H0_KM_S_MPC",
    "PLANCK18_NS",
    "PLANCK18_OMEGA_B",
    "PLANCK18_OMEGA_LAMBDA",
    "PLANCK18_OMEGA_M",
    "PLANCK18_SIGMA8",
    "POWER_LAW_FRACTION",
    "PROTON_MASS_KG",
    "SECONDS_PER_GYR",
    "SOLAR_METALLICITY_MASS_FRACTION",
    "SOLAR_OXYGEN_ABUNDANCE",
    "YEARS_PER_GYR",
]
