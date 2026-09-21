"""Marginalize the existing random-q first-passage model for PISN rates.

This preserves the log-linear interpolation of mass and M/Mcool. It does not
add a pristine-gas criterion or a survey selection function.
"""

import numpy as np


def threshold_averaged_pisn_rate(time, mass, cooling, kernel, mean=0.0, sigma=1.5, order=16):
    """Expected Mhalo_at_birth * k(age), events / yr before epsilon_b*f_b.

    Integrate the Normal distribution of log10(q) only over new record levels.
    Split at both lifetime support edges and every lifetime interpolation knot;
    this resolves the narrow explosion window even on a coarse history grid.
    """
    t, m, c = (np.asarray(v, float) for v in (time, mass, cooling))
    if (
        t.ndim != 2
        or m.shape != t.shape
        or c.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, m, c))
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(m <= 0)
        or np.any(c <= 0)
        or not np.isfinite([mean, sigma]).all()
        or sigma <= 0
        or order < 2
    ):
        raise ValueError("invalid PISN first-passage input")
    if np.any((t[:, -1] - t[:, 0]) * 1000 <= kernel.delay_bounds_myr[1]):
        raise ValueError("pre-start bursts can still explode; longer histories are required")
    ratio = np.log10(m / c)
    record = np.maximum.accumulate(ratio, axis=1)
    rows, left = np.nonzero(ratio[:, 1:] > record[:, :-1])
    dr = ratio[rows, left + 1] - ratio[rows, left]
    dt = t[rows, left + 1] - t[rows, left]
    age_left = (t[rows, -1] - t[rows, left]) * 1000
    logm = np.log(m)
    delta_logm = logm[rows, left + 1] - logm[rows, left]
    knots = kernel.mass[(kernel.mass > kernel.lower) & (kernel.mass < kernel.upper)]
    ages = np.sort(kernel.lifetime(np.r_[kernel.lower, knots, kernel.upper]))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(len(t))
    for young, old in zip(ages[:-1], ages[1:], strict=True):
        f0 = np.maximum(0, (age_left - old) / (1000 * dt))
        f1 = np.minimum(1, (age_left - young) / (1000 * dt))
        lo = np.maximum(record[rows, left], ratio[rows, left] + f0 * dr)
        hi = ratio[rows, left] + f1 * dr
        valid = hi > lo
        r, col = rows[valid], left[valid]
        width = hi[valid] - lo[valid]
        values = np.zeros(len(r))
        for node, weight in zip(nodes, weights, strict=True):
            x = lo[valid] + (node + 1) * width / 2
            fraction = (x - ratio[r, col]) / dr[valid]
            birth_mass = np.exp(logm[r, col] + fraction * delta_logm[valid])
            age = age_left[valid] - fraction * dt[valid] * 1000
            pdf = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            values += weight * width / 2 * pdf * birth_mass * kernel.rate(age)
        total += np.bincount(r, weights=values, minlength=len(t))
    if not np.isfinite(total).all() or np.any(total < 0):
        raise FloatingPointError("invalid threshold-averaged PISN rate")
    return total
