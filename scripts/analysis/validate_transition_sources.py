"""Validate paired source tables and HMF-weight numerical/censoring diagnostics."""

import json
from pathlib import Path

import numpy as np
from hmf import MassFunction

from auroralf.experiments.artifacts import digest
from auroralf.mah import Cosmology
from auroralf.uvlf.hmf_sampling import (
    DEFAULT_HMF_DLOG10M,
    HMF_REED07_FITTING_FUNCTION,
    MASS_FUNCTION_NS,
    MASS_FUNCTION_SIGMA8,
    prepare_reed07_hmf_interpolator,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_save/popii_transition_20260920"


def main():
    for kind in ("uvlf", "rates"):
        m = json.loads((DATA / kind / "manifest.json").read_text())
        if m["status"] != "complete":
            raise ValueError("Incomplete " + kind)
        for name, sha in m["products"].items():
            if digest(DATA / kind / name) != sha:
                raise ValueError("Product changed " + name)
    names = ["baseline", "delay0", "delay30"]
    arrays = [dict(np.load(DATA / "rates" / name / "sources.npz")) for name in names]
    old = dict(np.load(ROOT / "data_save/reionization_z5_20260919/combined_sources/sources.npz"))
    baseline = arrays[0]
    comparisons = {}
    for key in baseline:
        a, b = baseline[key], old[key]
        if key in ("redshifts", "mass_msun"):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-7, atol=0)
        comparisons[key] = float(np.max(abs(a - b) / np.maximum(abs(b), 1e-100)))
    model_keys = [
        "n_tracks",
        "track_chunk",
        "n_grid",
        "seed",
        "z_start",
        "epsilon_b",
        "fesc_popii",
        "fesc_popiii",
        "q_log10_mean",
        "q_log10_sigma",
        "h",
        "omega_m",
        "omega_b",
        "max_lookback_myr",
        "popiii_max_age_myr",
    ]
    prior = json.loads(
        (ROOT / "data_save/reionization_z5_20260919/combined_sources/manifest.json").read_text()
    )
    current = json.loads((DATA / "rates/baseline/manifest.json").read_text())
    for key in model_keys:
        if prior["resolved_model"][key] != current["resolved_model"][key]:
            raise ValueError("Changed " + key)
    for key in ("popii_ssp", "popiii_ssp"):
        old_path = prior["resolved_model"][key]
        new_path = current["resolved_model"][key]
        # Extension manifest retains hashes of all declared original SSP inputs.
        original_hash = prior["input_sha256"][old_path]
        if original_hash != current["input_sha256"][new_path]:
            raise ValueError("Changed SSP " + key)
    for a in arrays[1:]:
        np.testing.assert_array_equal(a["mean_rate"][..., 1:], baseline["mean_rate"][..., 1:])
    diag = dict(np.load(DATA / "rates/diagnostics.npz"))
    z, mass = baseline["redshifts"], baseline["mass_msun"]
    density = np.empty((len(z), 3, 3))
    qerr = np.empty((len(z), 2))
    censor = np.empty((len(z), 3))
    raw_increase = []
    cosmo = Cosmology()
    h = cosmo.h0_km_s_mpc / 100
    step = DEFAULT_HMF_DLOG10M
    mf = MassFunction(
        Mmin=np.floor((4 + np.log10(h) - 2 * step) / step) * step,
        Mmax=np.ceil((15 + np.log10(h) + 2 * step) / step) * step + step,
        dlog10m=step,
        z=float(z[0]),
        hmf_model=HMF_REED07_FITTING_FUNCTION,
        sigma_8=MASS_FUNCTION_SIGMA8,
        n=MASS_FUNCTION_NS,
        cosmo_params=dict(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b),
        transfer_params={"extrapolate_with_eh": True},
    )
    for i, zz in enumerate(z):
        # Reuse the identical transfer function; only the redshift changes.
        mf.update(z=float(zz))
        gridmass, dndm = mf.m / h, mf.dndm * h**4
        if not np.isfinite(dndm).all() or np.any(dndm < 0):
            raise ValueError("Invalid native Reed07 abundance")
        index = np.searchsorted(gridmass, mass) - 1
        if np.any(index < 0) or np.any(index + 1 >= len(gridmass)):
            raise ValueError("Mass diagnostic exceeds native HMF support")
        frac = np.log(mass / gridmass[index]) / np.log(gridmass[index + 1] / gridmass[index])
        lower, upper = dndm[index], dndm[index + 1]
        # Retain native zero values where the high-z high-mass abundance
        # underflows. Interpolate the positive-to-zero boundary linearly;
        # never extrapolate the last positive log-abundance into the tail.
        weight = lower + frac * (upper - lower)
        positive = (lower > 0) & (upper > 0)
        weight[positive] = np.exp(
            (1 - frac[positive]) * np.log(lower[positive])
            + frac[positive] * np.log(upper[positive])
        )
        if i in (0, int(abs(z - 8).argmin())):
            reference = prepare_reed07_hmf_interpolator(
                log10_halo_mass_min_msun=4,
                log10_halo_mass_max_msun=15,
                z_obs=float(zz),
                cosmology=cosmo,
            ).evaluate(mass)
            np.testing.assert_allclose(weight, reference, rtol=1e-12, atol=0)
        for k, a in enumerate(arrays):
            density[i, k] = np.trapezoid(a["mean_rate"][i] * weight[:, None], mass, axis=0)
        qerr[i] = np.trapezoid(diag["popii_quadrature_error"][i] * weight[:, None], mass, axis=0)
        censor[i] = np.trapezoid(
            diag["popii_censored_upper_extra"][i] * weight[:, None], mass, axis=0
        )
        raw_increase.append(
            float(
                np.max(
                    (arrays[2]["mean_rate"][i, :, 0] - baseline["mean_rate"][i, :, 0])
                    / np.maximum(baseline["mean_rate"][i, :, 0], 1e-100)
                )
            )
        )
    np.savez_compressed(
        DATA / "source_diagnostics.npz",
        redshifts=z,
        rate_density=density,
        popii_quadrature_density=qerr,
        popii_censored_density=censor,
    )
    summary = dict(
        status="complete",
        baseline_max_relative_difference=comparisons,
        density_units="escaped photons/s/Mpc^3 at common fesc=.2",
        max_popii_quadrature_fraction=float(
            np.max(qerr / np.maximum(density[:, 0, 0, None], 1e-100))
        ),
        max_censor_fraction_total=float(
            np.max(censor[:, 2] / np.maximum(density[:, 2, :2].sum(1), 1e-100))
        ),
        max_censor_absolute=float(censor[:, 2].max()),
        censor_fraction_peak_emissivity=float(censor[:, 2].max() / density[:, 2, :2].sum(1).max()),
        max_raw_relative_increase_30_vs_baseline=max(raw_increase),
        rows=[
            dict(
                z=float(z[i]),
                popii_relative=(density[i, :, 0] / density[i, 0, 0]).tolist()
                if density[i, 0, 0] > 0
                else None,
                total_relative=(density[i, :, :2].sum(1) / density[i, 0, :2].sum()).tolist()
                if density[i, 0, :2].sum() > 0
                else None,
                censor_fraction_total=float(censor[i, 2] / max(density[i, 2, :2].sum(), 1e-100)),
            )
            for i in [
                int(abs(z - v).argmin()) for v in [5, 6, 8, 10, 12.5, 14.5, 20, 25, 30, 40, 49]
            ]
        ],
    )
    (DATA / "source_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
