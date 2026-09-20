"""Neutral-fraction calibration and Thomson depth of spatial ionization histories.

Optical depth uses the mass-weighted H II fraction, baryons tracing matter,
X_H=0.75 as in EoRCaLC, and the same flat matter+Lambda background as the
source-time integration. Low-redshift completion is an explicit continuation,
not an extrapolated simulation result. Residual recombination-era electrons
are excluded from the reionization optical depth.
"""

import numpy as np
from astropy import units as u
from astropy.constants import c, m_p, sigma_T
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import cumulative_trapezoid


def validate_history(redshifts, ionized):
    z, q = np.asarray(redshifts, dtype=float), np.asarray(ionized, dtype=float)
    if (
        z.ndim != 1
        or len(z) < 2
        or q.shape != z.shape
        or not np.isfinite(z).all()
        or not np.isfinite(q).all()
        or np.any(np.diff(z) >= 0)
        or np.any(z <= 0)
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("Require descending positive redshifts and finite fractions in [0,1]")
    if q[0] != 0:
        raise ValueError("High-z continuation requires the simulation's neutral initial state")
    return z, q


def thomson_depth(
    redshifts,
    mass_weighted_xhii,
    *,
    h,
    omega_m,
    omega_b,
    completion_redshift=5.6,
    hydrogen_mass_fraction=0.75,
    helium_double_redshift=3.5,
    helium_double_width=0.5,
    dz=0.002,
):
    """Return cumulative dimensionless tau(0,z) and its electron history.

    H and He I share the mass-weighted filling fraction. The second helium
    electron follows a tanh transition at z=3.5 (width 0.5). Between the last
    simulated snapshot and completion_redshift, Q is linear in redshift;
    below completion, Q=1. Setting completion to the last snapshot gives the
    maximal immediate-completion contribution. No model data are fabricated.
    """
    zs, qs = validate_history(redshifts, mass_weighted_xhii)
    pars = [
        h,
        omega_m,
        omega_b,
        completion_redshift,
        hydrogen_mass_fraction,
        helium_double_redshift,
        helium_double_width,
        dz,
    ]
    if not np.isfinite(pars).all() or not (
        h > 0
        and 0 < omega_b < omega_m < 1
        and 0 < completion_redshift <= zs[-1]
        and 0 < hydrogen_mass_fraction <= 1
        and 0 < helium_double_redshift < completion_redshift
        and helium_double_width > 0
        and dz > 0
    ):
        raise ValueError(
            "Invalid cosmology, abundance, integration step or completion prescription"
        )
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=omega_m, Ob0=omega_b)
    z = np.unique(np.r_[np.arange(0, zs[0], dz), zs, completion_redshift, helium_double_redshift])
    if completion_redshift == zs[-1]:
        q = np.interp(z, zs[::-1], qs[::-1], left=1)
    else:
        q = np.interp(z, np.r_[0, completion_redshift, zs[::-1]], np.r_[1, 1, qs[::-1]])
    n_h0 = (cosmo.critical_density0 * omega_b * hydrogen_mass_fraction / m_p).to(u.m**-3)
    f_he = (1 - hydrogen_mass_fraction) / (4 * hydrogen_mass_fraction)
    he3 = 0.5 * (1 + np.tanh((helium_double_redshift - z) / helium_double_width))
    electrons_per_h = q * (1 + f_he + f_he * he3)
    kernel = (c * sigma_T * n_h0 / cosmo.H(z)).to_value(u.dimensionless_unscaled)
    integrand = kernel * (1 + z) ** 2 * electrons_per_h
    tau = cumulative_trapezoid(integrand, z, initial=0)
    if completion_redshift == zs[-1]:
        # The immediate-completion limit has a jump. Integrate the low-z side
        # with Q=1 at its endpoint instead of smearing the jump over one cell.
        i = int(np.searchsorted(z, zs[-1]))
        endpoint_full = kernel[i] * (1 + z[i]) ** 2 * (1 + f_he + f_he * he3[i])
        tau[i:] += 0.5 * (z[i] - z[i - 1]) * (endpoint_full - integrand[i])
    return dict(
        z=z,
        q_mass=q,
        electrons_per_h=electrons_per_h,
        dtau_dz=integrand,
        tau=tau,
        total=float(tau[-1]),
    )


def neutral_residuals(redshifts, volume_xhii, observations):
    """Descriptive asymmetric-interval residuals; broad bins are not fitted."""
    z, q = validate_history(redshifts, volume_xhii)
    rows = []
    score = 0.0
    for obs in observations:
        zz = float(obs["z"])
        if not z[-1] <= zz <= z[0]:
            rows.append(
                dict(
                    id=obs["id"],
                    model_xhi=None,
                    used=False,
                    reason="outside simulated redshift range",
                )
            )
            continue
        x = float(np.interp(zz, z[::-1], (1 - q)[::-1]))
        center, lo, hi = (float(obs[k]) for k in ["xhi", "lower", "upper"])
        used = obs["kind"] == "interval" and float(obs["z_min"]) == float(obs["z_max"])
        residual = None
        if obs["kind"] == "interval":
            residual = (x - center) / (center - lo if x < center else hi - center)
        if used:
            score += residual**2
        rows.append(
            dict(
                id=obs["id"],
                model_xhi=x,
                used=used,
                within_interval=lo <= x <= hi,
                residual=residual,
            )
        )
    return float(score), rows
