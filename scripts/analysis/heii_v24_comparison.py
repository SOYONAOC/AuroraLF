"""Unit-safe V24 source ingestion and explicitly conditional model summaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.heii import evaluate_kernel
from auroralf.mah import Cosmology
from auroralf.uvlf import uv_luminosity_to_muv

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "external_data/observations/heii/venditti2024_targets.json"
METHODS = ("linear_log_age", "log_log_age")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def observations(path=OBS):
    catalog = json.loads(Path(path).read_text())
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    rows = []
    for source in catalog["records"]:
        row = dict(source)
        if not (row["flux"] > 0 and row["mu"] > 0 and row["z"] > 0):
            raise ValueError(f"Invalid flux, lens or redshift: {row['id']}")
        correction = row["flux_corrections"]
        if correction not in ("observed", "delensed_and_dereddened"):
            raise ValueError(f"Unknown correction state: {correction}")
        factor = 4 * np.pi * astro.luminosity_distance(row["z"]).to_value("cm") ** 2
        if correction == "observed":
            factor /= row["mu"]
        if row["measurement"] not in ("detection", "upper_limit"):
            raise ValueError("Unknown measurement kind")
        if row["measurement"] == "upper_limit":
            if row["limit_sigma"] <= 0 or row["flux_error"] is not None:
                raise ValueError("An upper limit is not a zero-flux measurement")
        elif row["flux_error"] is None or not 0 < row["flux_error"] < row["flux"]:
            raise ValueError("Invalid measured flux uncertainty")
        row["luminosity"] = float(factor * row["flux"])
        row["luminosity_error"] = (
            None if row["flux_error"] is None else float(factor * row["flux_error"])
        )
        row["luminosity_error_scope"] = (
            "Quoted line-flux error scaled at fixed z and mu; no extra lens/dust systematics added"
        )
        if row.get("muv") is not None:
            source_astro = FlatLambdaCDM(**row["source_cosmology"])
            row["muv_common_cosmology"] = float(
                row["muv"]
                - 5
                * np.log10(
                    astro.luminosity_distance(row["z"]).value
                    / source_astro.luminosity_distance(row["z"]).value
                )
            )
        rows.append(row)
    return dict(
        sources=rows,
        catalog_sha256=digest(path),
        cosmology=dict(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b),
        definition="4 pi DL^2 times de-lensed flux, common project cosmology. Dust correction only where stated by source.",
    )


def calzetti_a1500(av):
    """Calzetti00 Rv=4.05, lambda=0.15 microns; explicit UV matching assumption."""
    inverse_um = 1 / 0.15
    k = 2.659 * (-2.156 + 1.509 * inverse_um - 0.198 * inverse_um**2 + 0.011 * inverse_um**3) + 4.05
    return float(k * av / 4.05)


def distribution(luminosity, weights, selection, reference=None):
    """HMF-weighted population spread; clustered weights diagnose MC sampling."""
    lum, w, mask = np.asarray(luminosity), np.asarray(weights), np.asarray(selection)
    if lum.ndim != 2 or mask.shape != lum.shape or w.shape != (len(lum),):
        raise ValueError("Inconsistent mass/history dimensions")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Invalid HMF weights")
    selected_w = np.broadcast_to(w[:, None], lum.shape)[mask]
    values = lum[mask]
    if not len(values) or selected_w.sum() <= 0:
        raise ValueError("Empty weighted model selection; increase sampling explicitly")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Nonfinite or negative selected luminosities")
    order = np.argsort(values)
    cdf = (np.cumsum(selected_w[order]) - 0.5 * selected_w[order]) / selected_w.sum()
    quantiles = np.interp([0.16, 0.5, 0.84], cdf, values[order])
    cluster_w = mask.sum(axis=1) * w
    density = float(cluster_w.sum())
    result = dict(
        n_selected=int(mask.sum()),
        n_mass_selected=int(np.count_nonzero(cluster_w)),
        effective_mass_clusters=float(density**2 / np.sum(cluster_w**2)),
        density_mpc3=density,
        density_mc_se=float(np.sqrt(len(w) * np.var(cluster_w, ddof=1))),
        luminosity_q16_q50_q84=quantiles.tolist(),
        zero_fraction=float(selected_w[values == 0].sum() / density),
    )
    if reference is not None:
        result["fraction_at_or_above_reference"] = float(
            selected_w[values >= reference].sum() / density
        )
    return result


def summarize_sample(sample, kernel, efficiencies, catalog):
    status, age = sample["status"], sample["age_myr"]
    if not np.isin(status, [0, 1, 2]).all():
        raise ValueError("Invalid crossing status")
    resolved = status == 1
    cosmo = Cosmology()
    per_efficiency_mass = sample["burst_halo_mass_msun"][resolved] * cosmo.omega_b / cosmo.omega_m
    z = float(sample["redshift"])
    rx = next(r for r in catalog["sources"] if r["id"] == "RXJ2129_z8HeII_A")
    center = rx["muv_common_cosmology"] - calzetti_a1500(rx["av"])
    result = dict(z=z, efficiencies={}, rx_intrinsic_muv_center=center)
    for eps in efficiencies:
        muv = uv_luminosity_to_muv(sample["popii"] + eps * sample["popiii_per_efficiency"])
        young_uv_bright = resolved & (age <= 3.0) & (muv <= -20)
        by_method = {}
        for method in METHODS:
            line = np.zeros(status.shape)
            line[status == 2] = np.nan  # unlocated pre-start bursts remain unknown
            line[resolved] = (
                eps * per_efficiency_mass * evaluate_kernel(age[resolved], kernel, method)
            )
            item = {
                "young_uv_bright": distribution(line, sample["weight_per_track"], young_uv_bright)
            }
            if np.isclose(z, rx["z"], atol=1e-8, rtol=0):
                selected = np.abs(muv - center) <= 0.25
                known = selected & (status != 2)
                item["rx_uv_matched"] = distribution(
                    line, sample["weight_per_track"], known, rx["luminosity"]
                )
                weights = np.broadcast_to(sample["weight_per_track"][:, None], status.shape)
                item["rx_uv_matched"]["unknown_weight_fraction"] = float(
                    weights[selected & (status == 2)].sum() / weights[selected].sum()
                )
                item["rx_uv_matched"]["muv_window_half_width"] = 0.25
            by_method[method] = item
        result["efficiencies"][str(eps)] = by_method
    return result
