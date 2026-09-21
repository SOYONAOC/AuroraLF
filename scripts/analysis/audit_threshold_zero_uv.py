"""Recompute observable bins and UV luminosity density for the completed mu=0 LF."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from auroralf.experiments.artifacts import digest
from auroralf.uvlf.dust import compute_dust_attenuated_uvlf, intrinsic_muv_from_observed
from scripts.analysis.audit_sfrd_fraction_constraints import KUV, luminosity

ROOT = Path(__file__).resolve().parents[2]
SAVE = ROOT / "data_save/threshold_zero_20260918"
OUT = ROOT / "data_save/threshold_zero_all_20260918"


def main():
    original = json.loads((SAVE / "summary.json").read_text())
    if original["status"] != "complete":
        raise ValueError("Incomplete UVLF")
    for p, h in original["sources"].items():
        if digest(p) != h:
            raise ValueError("UVLF source changed: " + p)
    bins = []
    rows = []
    sources = {}
    for z in [8, 12.5, 14.5]:
        path = SAVE / f"uvlf_z{z:g}.npz"
        sources[str(path)] = digest(path)
        with np.load(path) as d:
            x = (d["bin_edges"][1:] + d["bin_edges"][:-1]) / 2
            y = d["eps0.03"]
            if not np.all(np.isfinite(y) & (y > 0)):
                raise ValueError("Nonpositive LF; explicit support selection required")
            result = {}
            qmin = brentq(lambda m: intrinsic_muv_from_observed(m, z) - x.min(), x.min(), -17)
            for dust in [False, True]:
                left = qmin + 1e-8 if dust else x.min()
                integrals = []
                for n in [4097, 8193]:
                    q = np.linspace(left, -17, n)
                    v = (
                        compute_dust_attenuated_uvlf(x, y, z, muv_obs=q)["phi_obs"]
                        if dust
                        else 10 ** np.interp(q, x, np.log10(y))
                    )
                    integrals.append(np.trapezoid(luminosity(q) * v, q))
                np.testing.assert_allclose(*integrals, rtol=1e-4)
                mask = q <= left + 0.5
                result["eps0.03_" + ("dust" if dust else "nodust")] = dict(
                    rho_uv=integrals[-1],
                    uv_inferred_sfrd=KUV * integrals[-1],
                    bright_limit=left,
                    first_halfmag_fraction=np.trapezoid(luminosity(q[mask]) * v[mask], q[mask])
                    / integrals[-1],
                )
            if z == 8:
                rho = quad(
                    lambda m: (
                        luminosity(m)
                        * 3.30e-4
                        / (
                            10 ** (0.4 * (-2.04 + 1) * (m + 20.02))
                            + 10 ** (0.4 * (-4.26 + 1) * (m + 20.02))
                        )
                    ),
                    -40,
                    -17,
                    epsabs=1e15,
                )[0]
            else:
                rho = 10 ** {12.5: 24.64, 14.5: 23.92}[z]
            rows.append(
                dict(
                    z=z,
                    models=result,
                    observed_rho_uv=rho,
                    observed_uv_sfrd=KUV * rho,
                    ratio_total_dust=result["eps0.03_dust"]["rho_uv"] / rho,
                )
            )
            if z != 8:
                obsfile = (
                    ROOT
                    / "external_data/observations/uvlf"
                    / (
                        "redshift_12p5/donnan24.npz"
                        if z == 12.5
                        else "redshift_15/naidu26_mom_jades_spectroscopic_z14p5.npz"
                    )
                )
                sources[str(obsfile)] = digest(obsfile)
                with np.load(obsfile) as obs:
                    for mag, half, observed in zip(
                        obs["muverr"], obs["mag_err"], obs["phierr"], strict=True
                    ):
                        weights = np.maximum(
                            0,
                            np.minimum(d["bin_edges"][1:], mag + half)
                            - np.maximum(d["bin_edges"][:-1], mag - half),
                        ) / (2 * half)
                        np.testing.assert_allclose(weights.sum(), 1)
                        model = float(weights @ y)
                        bins.append(
                            dict(
                                z=z,
                                muv=float(mag),
                                model=model,
                                observed=float(observed),
                                ratio=model / observed,
                            )
                        )
    report = dict(
        q_log10_mean=0,
        q_log10_sigma=1.5,
        uv_inferred=rows,
        high_bins=bins,
        low_bins=original["lowz_observation_comparisons"],
        sources=sources,
    )
    (OUT / "uv-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(uv_inferred=rows, high_bins=bins), indent=1))


if __name__ == "__main__":
    main()
