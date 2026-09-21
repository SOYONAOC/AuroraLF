"""Separate formed-mass SFRD, UV-inferred SFRD and selected Pop III constraints."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from auroralf.uvlf.dust import compute_dust_attenuated_uvlf, intrinsic_muv_from_observed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/sfrd_fraction_audit_20260917"
RUN = ROOT / "data_save/uvlf_efficiency_scan_20260914_64x"
KUV = 1.15e-28


def luminosity(m):
    return 4 * np.pi * (10 * 3.0856775814913673e18) ** 2 * 10 ** (-0.4 * (m + 48.6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    sfrd = json.loads((ROOT / "data_save/ionizing_sources/sfrd_v1/summary.json").read_text())
    rows = []
    for z in (8, 12.5, 14.5):
        path = RUN / f"z{z}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["products"][str(path)]
        with np.load(path) as data:
            x = (data["bin_edges"][1:] + data["bin_edges"][:-1]) / 2
            result = {}
            for mode in ("eps0.03",):
                phi = data[mode]
                assert np.all(np.isfinite(phi) & (phi > 0))
                qmin = brentq(lambda m: intrinsic_muv_from_observed(m, z) - x.min(), x.min(), -17)
                for dust in (False, True):
                    left = qmin + 1e-8 if dust else x.min()
                    values = []
                    for n in (4097, 8193):
                        q = np.linspace(left, -17, n)
                        if dust:
                            p = compute_dust_attenuated_uvlf(x, phi, z, muv_obs=q)
                            assert p["Muv_intrinsic"].min() >= x.min()
                            v = p["phi_obs"]
                        else:
                            v = 10 ** np.interp(q, x, np.log10(phi))
                        values.append(np.trapezoid(luminosity(q) * v, q))
                    np.testing.assert_allclose(*values, rtol=1e-4)
                    # Fraction in brightest available 0.5 mag diagnoses omitted tail size.
                    mask = q <= left + 0.5
                    edge = np.trapezoid(luminosity(q[mask]) * v[mask], q[mask]) / values[-1]
                    result[mode + ("_dust" if dust else "_nodust")] = dict(
                        rho_uv=values[-1],
                        uv_inferred_sfrd=KUV * values[-1],
                        bright_limit=left,
                        first_halfmag_fraction=edge,
                    )
            if z == 8:
                # Donnan23 Table8 DPL. This reconstructs the central fit, not its errors.
                def dpl(m):
                    return 3.30e-4 / (
                        10 ** (0.4 * (-2.04 + 1) * (m + 20.02))
                        + 10 ** (0.4 * (-4.26 + 1) * (m + 20.02))
                    )

                rho = quad(lambda m: luminosity(m) * dpl(m), -40, -17, epsabs=1e15)[0]
                reference = "Donnan2023 Table8 DPL reconstruction; section5.3"
            else:
                rho = 10 ** ({12.5: 24.64, 14.5: 23.92}[z])
                reference = "Donnan2024 Table3; section4.3"
            rows.append(
                dict(
                    z=z,
                    reference=reference,
                    observed_rho_uv=rho,
                    observed_uv_sfrd=KUV * rho,
                    models=result,
                    ratio_total_dust=result["eps0.03_dust"]["rho_uv"] / rho,
                    ratio_total_nodust=result["eps0.03_nodust"]["rho_uv"] / rho,
                )
            )
    z6 = next(r for r in sfrd["rows"] if r["z"] == 6)
    upper = 10**-3.98
    # Ratio only: different selections and SFR estimators prohibit an exclusion test.
    result = dict(
        uv_inferred=rows,
        formed_mass_sfrd=sfrd["rows"],
        glimpsed=dict(
            arxiv="2512.11790v3",
            section="5.3",
            table=4,
            redshift=[5.6, 6.6],
            log_sfrd_interval=[-6.29, -3.98],
            upper=upper,
            model_z6_popiii=z6["popiii"],
            raw_ratio=z6["popiii"] / upper,
            warning="Selected pure young high-covering-fraction systems, conditional on AMORE6; not an upper bound on all Pop III. Model contains mixed enriched-host possibilities and uses formed mass per 10Myr.",
        ),
        limitations=[
            "UV-inferred rates are not additional independent data beyond UVLF.",
            "No likelihood or exclusion significance.",
            "High-z saved proxy mixes PopII1600A and PopIII1500A.",
            "Cosmology not homogenized.",
            "Finite bright interpolation support; no tail extrapolation.",
            "GLIMPSED PDF retrieval failed TLS locally; primary v3 HTML fully checked.",
        ],
    )
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(
            r["z"],
            "obs",
            r["observed_uv_sfrd"],
            "total dust/nodust",
            r["models"]["eps0.03_dust"]["uv_inferred_sfrd"],
            r["models"]["eps0.03_nodust"]["uv_inferred_sfrd"],
            "ratios",
            r["ratio_total_dust"],
            r["ratio_total_nodust"],
        )
    print("GLIMPSED selected upper", upper, "raw z6 ratio", z6["popiii"] / upper)


if __name__ == "__main__":
    main()
