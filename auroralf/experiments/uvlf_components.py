"""Age-stratified first-crossing proposals, retaining the original Normal q law."""

import numpy as np
from scipy.special import ndtr, ndtri

from auroralf.experiments.random_q import burst_light
from auroralf.uvlf import uv_luminosity_to_muv


def age_strata(t, mass, cooling, mean, sigma, uniforms, ages, kernel, fb, epsilon):
    """Condition q on burst ages 0–3–10–30–100 Myr plus a UV-dark remainder.

    Each track's proposal weights are exact Normal CDF differences. This is
    importance sampling, not a change to the probability of a recent burst.
    The UV-dark probability includes old, untriggered and left-censored events.
    """
    bounds = np.array([0.0, 3.0, 10.0, 30.0, 100.0])
    if t.shape != mass.shape or t.shape != cooling.shape or uniforms.shape != (len(t), 4):
        raise ValueError("Invalid history/stratum dimensions")
    if np.any((uniforms <= 0) | (uniforms >= 1)):
        raise ValueError("Conditional uniforms must be strictly inside (0,1)")
    if np.any((t[:, -1] - t[:, 0]) * 1000 <= bounds[-1]):
        raise ValueError("Pre-start bursts overlap the UV window")
    ratio = np.log10(mass / cooling)
    record = np.maximum.accumulate(ratio, axis=1)
    rows = np.arange(len(t))
    cdfs = []
    for age in bounds:
        cutoff = t[:, -1] - age / 1000
        lo = np.clip(np.sum(t <= cutoff[:, None], axis=1) - 1, 0, t.shape[1] - 2)
        f = (cutoff - t[rows, lo]) / (t[rows, lo + 1] - t[rows, lo])
        value = np.maximum(
            record[rows, lo], ratio[rows, lo] + f * (ratio[rows, lo + 1] - ratio[rows, lo])
        )
        cdfs.append(ndtr((value - mean) / sigma))
    cdfs = np.stack(cdfs, axis=1)
    probs = cdfs[:, :-1] - cdfs[:, 1:]
    if np.any(probs < 0):
        raise FloatingPointError("First-passage probabilities must be nonnegative")
    probability = np.column_stack((probs, 1 - probs.sum(axis=1)))
    light = np.zeros(probability.shape)
    for j in range(4):
        active = probs[:, j] > 0
        qcdf = cdfs[active, j + 1] + uniforms[active, j] * probs[active, j]
        logq = mean + sigma * ndtri(qcdf)
        event = burst_light(t[active], mass[active], cooling[active], logq, ages, kernel, fb, 100.0)
        if not np.all(event["status"] == 1):
            raise FloatingPointError("Conditional first passage was not resolved")
        np.testing.assert_array_less(bounds[j] - 1e-6, event["age_myr"])
        np.testing.assert_array_less(event["age_myr"], bounds[j + 1] + 1e-6)
        light[active, j] = epsilon * event["popiii_per_efficiency"]
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=2e-15)
    return light, probability


def conditional_histograms(p2, p3, probability, weights, edges):
    """One halo, five conditional q proposals; keep one total probability."""
    if p3.shape != probability.shape or p3.shape[:2] != p2.shape:
        raise ValueError("Inconsistent luminosity shapes")
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in [p2, p3, probability, weights]):
        raise ValueError("Invalid conditional samples")
    np.testing.assert_allclose(probability.sum(axis=-1), 1, atol=2e-15, rtol=0)
    l2 = np.broadcast_to(p2[:, :, None], p3.shape)
    result = []
    for lum in [l2, p3, l2 + p3]:
        mag = uv_luminosity_to_muv(lum)
        per_mass = np.stack(
            [
                np.histogram(m.ravel(), bins=edges, weights=p.ravel())[0]
                for m, p in zip(mag, probability, strict=True)
            ]
        )
        result.append(per_mass * weights[:, None] / np.diff(edges))
    return np.stack(result)
