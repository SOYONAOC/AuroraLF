"""Pop II birth-time gates after the same random first Pop III burst.

The gate truncates the existing delayed SFR without renormalizing it. It is
an explicit isolated-main-branch, successful-enrichment approximation, not a
metallicity evolution or gas-reservoir model. Times are Gyr; delays are Myr.
"""

import numpy as np
from scipy.special import ndtr


def validate_history(t, mass, cooling):
    t, mass, cooling = (np.asarray(x, float) for x in (t, mass, cooling))
    if (
        t.ndim != 2
        or t.shape != mass.shape
        or t.shape != cooling.shape
        or t.shape[1] < 2
        or not all(np.isfinite(x).all() for x in (t, mass, cooling))
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(mass <= 0)
        or np.any(cooling <= 0)
    ):
        raise ValueError("invalid transition history")
    return t, np.log10(mass / cooling)


def record_at(t, ratio, query):
    """Running maximum of piecewise-linear log10(M/Mcool), at supplied times."""
    query = np.asarray(query, float)
    if query.ndim != 2 or len(query) != len(t) or not np.isfinite(query).all():
        raise ValueError("invalid transition query")
    index = np.stack(
        [np.searchsorted(row, q, side="right") - 1 for row, q in zip(t, query, strict=True)]
    )
    index = np.clip(index, 0, t.shape[1] - 2)
    rows = np.arange(len(t))[:, None]
    f = np.clip((query - t[rows, index]) / (t[rows, index + 1] - t[rows, index]), 0, 1)
    return np.maximum(
        np.maximum.accumulate(ratio, axis=1)[rows, index],
        ratio[rows, index] + f * (ratio[rows, index + 1] - ratio[rows, index]),
    )


def transition_probability(t, ratio, birth, delay_myr, mean, sigma):
    """Lower/upper P(first crossing + delay <= birth), including censoring.

    Before t_start the crossing date is unknown. The lower bound counts no
    such events until t_start+delay; the upper permits the earliest possible
    cosmic birth (t=0). After t_start+delay their contribution is exact.
    """
    if not np.isfinite([delay_myr, mean, sigma]).all() or delay_myr < 0 or sigma <= 0:
        raise ValueError("invalid transition parameters")
    cutoff = birth - delay_myr / 1000
    p = ndtr((record_at(t, ratio, cutoff) - mean) / sigma)
    resolved_time = cutoff >= t[:, :1]
    lower = np.where(resolved_time, p, 0.0)
    upper = np.where(cutoff >= 0, p, 0.0)
    return lower, upper


def _validate_sfh(t, sfr, active, max_age_myr, order):
    t, sfr, active = np.asarray(t, float), np.asarray(sfr, float), np.asarray(active)
    if (
        t.ndim != 2
        or t.shape != sfr.shape
        or t.shape != active.shape
        or t.shape[1] < 2
        or active.dtype != bool
        or not np.isfinite(t).all()
        or not np.isfinite(sfr).all()
        or np.any(np.diff(t, axis=1) <= 0)
        or np.any(sfr < 0)
        or not np.isfinite(max_age_myr)
        or max_age_myr <= 0
        or order < 2
    ):
        raise ValueError("invalid transition SFH integral")
    return t, np.where(active, sfr, 0.0)


class BirthIntegral:
    """SSP convolution after arbitrary exact onset, sharing interval integrals.

    Piecewise-linear SFR [Msun/yr] times any SSP observable per initial Msun.
    A suffix sum avoids subtracting nearly equal totals for very recent onsets.
    """

    def __init__(self, t, sfr, active, kernel, max_age_myr=100.0, order=16):
        self.t, self.s = _validate_sfh(t, sfr, active, max_age_myr, order)
        self.window_start = np.maximum(self.t[:, :1], self.t[:, -1, None] - max_age_myr / 1000)
        self.kernel = kernel
        self.nodes, self.weights = np.polynomial.legendre.leggauss(order)
        self.left = np.minimum(
            np.maximum(self.t[:, :-1], self.t[:, -1, None] - max_age_myr / 1000), self.t[:, 1:]
        )
        self.right = self.t[:, 1:]
        self.slope = np.diff(self.s, axis=1) / np.diff(self.t, axis=1)
        full = self._integrate(
            self.left, self.right, np.broadcast_to(np.arange(self.t.shape[1] - 1), self.left.shape)
        )
        self.suffix = np.column_stack((np.cumsum(full[:, ::-1], axis=1)[:, ::-1], np.zeros(len(t))))

    def _integrate(self, left, right, index):
        rows = np.arange(len(self.t))[:, None]
        width = right - left
        slope = self.slope[rows, index]
        sleft = self.s[rows, index] + slope * (left - self.t[rows, index])
        result = np.zeros_like(left)
        for node, weight in zip(self.nodes, self.weights, strict=True):
            f = (node + 1) / 2
            age = np.where(
                width > 0, np.maximum(0, (self.t[:, -1, None] - left - f * width) * 1000), 0
            )
            result += (
                np.maximum(0, sleft + f * width * slope)
                * self.kernel.rate(age)
                * width
                * weight
                * 0.5e9
            )
        return result

    def after(self, onset):
        onset = np.asarray(onset, float)
        if onset.ndim != 2 or len(onset) != len(self.t) or np.isnan(onset).any():
            raise ValueError("onset must be track x proposal, without NaNs")
        q = np.clip(onset, self.window_start, self.t[:, -1, None])
        index = np.stack(
            [np.searchsorted(row, v, side="right") - 1 for row, v in zip(self.t, q, strict=True)]
        )
        index = np.clip(index, 0, self.t.shape[1] - 2)
        rows = np.arange(len(self.t))[:, None]
        result = self._integrate(q, self.t[rows, index + 1], index) + self.suffix[rows, index + 1]
        return np.where(onset >= self.t[:, -1, None], 0.0, result)


def averaged_popii(
    t, sfr, active, mass, cooling, kernel, delay_myr, mean, sigma, max_age_myr=100.0, order=16
):
    """Convolve SFR with exact Normal first-passage occupancy; return bounds."""
    t, s = _validate_sfh(t, sfr, active, max_age_myr, order)
    _, ratio = validate_history(t, mass, cooling)
    # Split at both SFR knots and shifted first-passage knots. The t_start+delay
    # discontinuity of the conservative censored contribution is explicit.
    knots = np.sort(np.concatenate((t, t + delay_myr / 1000), axis=1), axis=1)
    knots = np.clip(
        knots, np.maximum(t[:, :1], t[:, -1, None] - max_age_myr / 1000), t[:, -1, None]
    )
    left, width = knots[:, :-1], np.diff(knots, axis=1)
    result = np.zeros((len(t), 2))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for node, weight in zip(nodes, weights, strict=True):
        birth = left + (node + 1) * width / 2
        value = np.stack([np.interp(q, row, y) for row, y, q in zip(t, s, birth, strict=True)])
        age = np.where(width > 0, np.maximum(0, (t[:, -1, None] - birth) * 1000), 0)
        base = value * kernel.rate(age) * width * weight * 0.5e9
        lower, upper = transition_probability(t, ratio, birth, delay_myr, mean, sigma)
        result[:, 0] += np.sum(base * lower, axis=1)
        result[:, 1] += np.sum(base * upper, axis=1)
    return result
