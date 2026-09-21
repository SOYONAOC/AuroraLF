"""Apply Vikaeus+22 Table 1 depths to verified, saved 1500-Angstrom populations.

This is local post-processing: no MAHs, SSPs or formation parameters are resampled.
The reference survey is unlensed and dust-free, with matched rest-frame UV bands.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.constants import AB_ZEROPOINT_LNU
from auroralf.experiments.artifacts import digest
from auroralf.experiments.heii import evaluate_kernel, load_kernel
from auroralf.mah import Cosmology

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_save/threshold_zero_all_20260918"
SAMPLES = [
    (6.639, "heii_v24_targets_20260914"),
    (8.1623, "heii_v24_targets_20260914"),
    (10.6, "heii_v24_targets_20260914"),
    (12.342, "heii_uniform_band_20260917"),
    (13.86, "heii_uniform_band_20260917"),
]


def depth_limits(z, astro, apparent_limit=30.6, line_flux_limit=2.9e-19):
    """Matched-band AB conversion; magnification=1, attenuation=0."""
    if not all(np.isfinite(v) for v in (z, apparent_limit, line_flux_limit)):
        raise ValueError("Nonfinite depth parameter")
    if z <= 0 or line_flux_limit <= 0:
        raise ValueError("Redshift and line flux limit must be positive")
    absolute_limit = apparent_limit - astro.distmod(z).value + 2.5 * np.log10(1 + z)
    return {
        "muv_limit": float(absolute_limit),
        "uv_lnu_limit": float(10 ** ((AB_ZEROPOINT_LNU - absolute_limit) / 2.5)),
        "line_luminosity_limit": float(
            4 * np.pi * astro.luminosity_distance(z).to_value("cm") ** 2 * line_flux_limit
        ),
    }


def summarize_arrays(uv, line, status, mass_weights, limits, *, age_myr=None, max_age_myr=None):
    """Keep zero emitters and report unknown-history weight separately.

    Detection fractions are weighted over the known UV-selected population.
    Bounds additionally assign all UV-selected unknown histories to non-detected
    or detected. They do not bound missing UV light from unlocated early bursts.
    Empty detectable samples have null quantiles, not fabricated luminosities.
    """
    uv, line, status, mass_weights = map(np.asarray, (uv, line, status, mass_weights))
    if uv.ndim != 2 or line.shape != uv.shape or status.shape != uv.shape:
        raise ValueError("Inconsistent history arrays")
    if mass_weights.shape != (len(uv),) or not np.isfinite(mass_weights).all():
        raise ValueError("Invalid mass weights")
    if np.any(mass_weights < 0) or not np.isin(status, [0, 1, 2]).all():
        raise ValueError("Invalid weights or crossing status")
    if not np.isfinite(uv).all() or np.any(uv < 0):
        raise ValueError("Invalid stored UV luminosity")
    known = status != 2
    if not np.isfinite(line[known]).all() or np.any(line[known] < 0):
        raise ValueError("Invalid known line luminosity")
    if not np.isnan(line[~known]).all() or np.any(line[status == 0] != 0):
        raise ValueError("Unknown histories require NaN; untriggered histories require zero")
    for name in ("uv_lnu_limit", "line_luminosity_limit"):
        if not np.isfinite(limits[name]) or limits[name] <= 0:
            raise ValueError("Invalid luminosity limit")
    weights = np.broadcast_to(mass_weights[:, None], uv.shape)
    selected = (uv >= limits["uv_lnu_limit"]) & (weights > 0)
    if max_age_myr is not None:
        age_myr = np.asarray(age_myr)
        if age_myr.shape != uv.shape or not np.isfinite(max_age_myr) or max_age_myr <= 0:
            raise ValueError("Invalid burst-age selection")
        if not np.isfinite(age_myr[status == 1]).all() or np.any(age_myr[status == 1] < 0):
            raise ValueError("Invalid resolved burst ages")
        selected &= (status == 1) & (age_myr <= max_age_myr)
    selected_known = selected & known
    detected = selected_known & (line >= limits["line_luminosity_limit"])

    def stats(mask):
        count = int(mask.sum())
        cluster_weights = mask.sum(axis=1) * mass_weights
        density = float(cluster_weights.sum())
        if not count:
            return {"n": 0, "density_mpc3": 0.0, "q16_q50_q84": None, "n_eff_mass": 0.0}
        values, w = line[mask], weights[mask]
        order = np.argsort(values)
        cdf = (np.cumsum(w[order]) - 0.5 * w[order]) / density
        return {
            "n": count,
            "density_mpc3": density,
            "q16_q50_q84": np.interp([0.16, 0.5, 0.84], cdf, values[order]).tolist(),
            "zero_weight_fraction": float(w[values == 0].sum() / density),
            "n_eff_mass": float(density**2 / np.sum(cluster_weights**2)),
            "n_mass": int(np.count_nonzero(cluster_weights)),
        }

    full, observable = stats(selected_known), stats(detected)
    if full["density_mpc3"] <= 0:
        raise ValueError("No known UV-selected population")
    all_density = float(weights[selected].sum())
    unknown_density = float(weights[selected & ~known].sum())
    detected_density = observable["density_mpc3"]
    return {
        "uv_selected_known": full,
        "uv_and_line_selected": observable,
        "n_uv_selected": int(selected.sum()),
        "uv_selected_density_mpc3": all_density,
        "unknown_weight_fraction": unknown_density / all_density,
        "line_detectable_fraction_known": detected_density / full["density_mpc3"],
        "line_detectable_fraction_bounds": [
            detected_density / all_density,
            (detected_density + unknown_density) / all_density,
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--muv-limit", type=float, help="Analyst-chosen absolute UV magnitude cut")
    parser.add_argument(
        "--max-age-myr", type=float, help="Require a resolved burst no older than this"
    )
    args = parser.parse_args()
    if args.muv_limit is not None and not np.isfinite(args.muv_limit):
        raise ValueError("Nonfinite absolute magnitude limit")
    if args.max_age_myr is not None and args.muv_limit is None:
        raise ValueError("Explicit absolute-magnitude selection required with age cut")
    fixed_uv = args.muv_limit is not None
    if args.output is None:
        args.output = (
            ROOT
            / "data_save"
            / ("heii_uv_bright_20260918" if fixed_uv else "heii_survey_depth_20260918")
        )
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    parents, results = {}, []
    kernel = None
    for z, series in SAMPLES:
        folder = BASE / series / f"z{z:g}"
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        plan = manifest["plan"]
        if manifest["status"] != "complete" or len(plan["configs"]) != 1:
            raise ValueError(f"Invalid parent manifest: {folder}")
        cfg = plan["configs"][0]
        if (cfg["z"], cfg["q_log10_mean"], cfg["q_log10_sigma"], cfg["popii_wavelength_a"]) != (
            z,
            0.0,
            1.5,
            1500.0,
        ):
            raise ValueError(f"Parent model or wavelength mismatch: {folder}")
        path = folder / f"z{z:g}.npz"
        if digest(path) != manifest["products"][path.name]:
            raise ValueError(f"Corrupt population: {path}")
        required = [
            cfg["popiii_ssp"],
            plan["line_ssp"],
            "auroralf/experiments/heii.py",
            "auroralf/constants.py",
            "auroralf/mah/models.py",
        ]
        for name in required:
            if digest(ROOT / name) != plan["input_sha256"][name]:
                raise ValueError(f"Frozen reconstruction input changed: {name}")
        if kernel is None:
            kernel = load_kernel(ROOT / cfg["popiii_ssp"], ROOT / plan["line_ssp"])
        with np.load(path, allow_pickle=False) as sample:
            if float(sample["redshift"]) != z:
                raise ValueError("Population redshift mismatch")
            status = sample["status"]
            if status.shape != (cfg["n_mass"], cfg["n_tracks"]):
                raise ValueError("Population dimensions changed")
            resolved = status == 1
            uv = sample["popii"] + 0.03 * sample["popiii_per_efficiency"]
            line = np.zeros(status.shape)
            line[status == 2] = np.nan
            line[resolved] = (
                0.03
                * cosmo.omega_b
                / cosmo.omega_m
                * sample["burst_halo_mass_msun"][resolved]
                * evaluate_kernel(sample["age_myr"][resolved], kernel, "linear_log_age")
            )
            limits = depth_limits(z, astro)
            if fixed_uv:
                limits["muv_limit"] = args.muv_limit
                limits["uv_lnu_limit"] = 10 ** ((AB_ZEROPOINT_LNU - args.muv_limit) / 2.5)
            result = dict(
                z=z,
                **limits,
                **summarize_arrays(
                    uv,
                    line,
                    status,
                    sample["weight_per_track"],
                    limits,
                    age_myr=sample["age_myr"],
                    max_age_myr=args.max_age_myr,
                ),
            )
        parents[str(path.relative_to(ROOT))] = digest(path)
        parents[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "epsilon": 0.03,
        "threshold_log10_mean": 0.0,
        "threshold_log10_sigma": 1.5,
        "selection_mode": "absolute_uv" if fixed_uv else "survey_depth",
        "absolute_uv_limit": args.muv_limit,
        "max_age_myr": args.max_age_myr,
        "apparent_uv_limit": None if fixed_uv else 30.6,
        "heii_flux_limit": 2.9e-19,
        "reference": "https://arxiv.org/pdf/2107.01230v2#page=6",
        "reference_locator": "Vikaeus+22 Table 1; 5-sigma point sources; deep survey, ~28 h NIRSpec",
        "cosmology": dict(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b),
        "assumptions": [
            "Matched rest-frame 1500 A bands; no filter throughput integration",
            "No dust, no lensing; literature scalar depth is a reference scenario, not per-target ETC",
            "UV is total Pop II+III; line is Pop III Case-B; fixed linear-L/log-age SSP interpolation",
            (
                f"Analyst-chosen MUV <= {args.muv_limit}; max burst age = {args.max_age_myr} Myr; "
                "no target UV window; main distribution has no line-flux cut. "
                "Line-selected statistics are auxiliary only."
                if fixed_uv
                else "No burst-age cut, no target UV window, no global absolute-magnitude cut"
            ),
            "Quantiles are HMF-weighted population spreads, not errors on the median",
            "Unknown pre-start bursts excluded from line quantiles; their selected weight is reported",
            "No inference about missing UV contribution of unlocated pre-start bursts",
        ],
        "results": results,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest = {
        "status": "complete",
        "parents": parents,
        "script_sha256": digest(Path(__file__)),
        "products": {"summary.json": digest(summary_path)},
        "inputs": {name: digest(ROOT / name) for name in required},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
