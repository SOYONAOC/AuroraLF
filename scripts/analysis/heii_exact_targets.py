"""UV-conditioned He II statistics at the actual redshift of each observation."""

import numpy as np

from auroralf.experiments.heii import evaluate_kernel
from auroralf.mah import Cosmology
from auroralf.uvlf import uv_luminosity_to_muv
from scripts.analysis.compare_random_q_heii_observations import METHODS, conditional_stats


def summarize_target(sample, target, kernel, efficiencies, astro):
    z = float(sample["redshift"])
    if not np.isclose(z, target["z"], atol=1e-10, rtol=0):
        raise ValueError("Model redshift must match the observed target redshift")
    status = sample["status"]
    if not np.isin(status, [0, 1, 2]).all():
        raise ValueError("Invalid burst status")
    resolved = status == 1
    cosmo = Cosmology()
    mass_per_efficiency = sample["burst_halo_mass_msun"][resolved] * cosmo.omega_b / cosmo.omega_m
    cluster = np.broadcast_to(np.arange(len(status))[:, None], status.shape)
    weights = np.broadcast_to(sample["weight_per_track"][:, None], status.shape)
    conversion = target["magnification"] / (
        4 * np.pi * astro.luminosity_distance(z).to_value("cm") ** 2
    )
    exact_target = dict(target, population_z=z)
    by_efficiency = {}
    for epsilon in efficiencies:
        muv = uv_luminosity_to_muv(sample["popii"] + epsilon * sample["popiii_per_efficiency"])
        windows = []
        for width in (0.25, 0.5):
            selected = np.abs(muv - target["muv"]) <= width
            windows.append(
                {"muv_low": target["muv"] - width, "muv_high": target["muv"] + width, "methods": {}}
            )
            for method in METHODS:
                line = np.zeros(status.shape)
                line[status == 2] = np.nan
                line[resolved] = (
                    epsilon
                    * mass_per_efficiency
                    * evaluate_kernel(sample["age_myr"][resolved], kernel, method)
                )
                windows[-1]["methods"][method] = conditional_stats(
                    line[selected] * conversion,
                    weights[selected],
                    cluster[selected],
                    target["flux"],
                )
        by_efficiency[str(epsilon)] = {
            "target": exact_target,
            "flux_per_luminosity": conversion,
            "target_windows": windows,
        }
    return by_efficiency
