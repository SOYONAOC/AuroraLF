"""Compare existing random-q predictions to published observable bins.

This is a fixed-model discrepancy audit, not a likelihood or rejection test.
No simulation, SSP, dust prescription or observational selection is refitted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.plot.plot_current_uvlf_dust import bin_predictions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/popiii_observation_audit_20260917"
UVLF = ROOT / "data_save/uvlf_efficiency_scan_20260914_64x"


def main():
    inputs = []

    def read(path):
        inputs.append(path)
        return json.loads(path.read_text())

    manifest = read(UVLF / "manifest.json")
    previous = read(UVLF / "summary.json")
    assert manifest["status"] == "complete"
    rows = []
    observations = read(ROOT / "external_data/observations/uvlf/current_z6_z8_z10.json")
    for z in (8, 12.5, 14.5):
        path = UVLF / f"z{z}.npz"
        inputs.append(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["products"][str(path)]
        with np.load(path) as data:
            edges = data["bin_edges"]
            centers = (edges[:-1] + edges[1:]) / 2
            if z == 8:
                datasets = [o for o in observations["datasets"] if o["z"] == z]
            else:
                name = (
                    "redshift_12p5/donnan24.npz"
                    if z == 12.5
                    else "redshift_15/naidu26_mom_jades_spectroscopic_z14p5.npz"
                )
                obs_path = ROOT / "external_data/observations/uvlf" / name
                inputs.append(obs_path)
                with np.load(obs_path) as obs:
                    datasets = [dict(obs)]
            for obs in datasets:
                label = obs["label"] if z == 8 else str(obs["label"][0])
                for i, (mag, half) in enumerate(zip(obs["muverr"], obs["mag_err"], strict=True)):
                    predictions = {}
                    for mode in ("baseline", "eps0.03"):
                        if z == 8:
                            value = bin_predictions(centers, data[mode], z, mag, half, 1025)[1]
                            coarse = bin_predictions(centers, data[mode], z, mag, half, 513)[1]
                            np.testing.assert_allclose(value, coarse, rtol=1e-3)
                            saved = next(
                                r
                                for r in previous["observational_comparison"]
                                if r["z"] == z and r["dataset"] == label and r["Muv"] == mag
                            )
                            np.testing.assert_allclose(
                                value / obs["phierr"][i],
                                saved[mode + "_dust_model_over_observed"],
                                rtol=1e-10,
                            )
                        else:
                            weights = np.maximum(
                                0,
                                np.minimum(edges[1:], mag + half)
                                - np.maximum(edges[:-1], mag - half),
                            ) / (2 * half)
                            np.testing.assert_allclose(weights.sum(), 1)
                            value = weights @ data[mode]
                        predictions[mode] = float(value)
                    observed = float(obs["phierr"][i])
                    rows.append(
                        dict(
                            z=z,
                            dataset=label,
                            Muv=float(mag),
                            half_width=float(half),
                            observed=observed,
                            error_minus=float(obs["phi_err_lo"][i]),
                            error_plus=float(obs["phi_err_up"][i]),
                            **predictions,
                            total_over_observed=predictions["eps0.03"] / observed,
                            baseline_over_observed=predictions["baseline"] / observed,
                        )
                    )
    heii = read(ROOT / "data_save/heii_exact_targets_20260916/summary.json")
    line_rows = []
    for result in heii["efficiencies"]["0.03"]["results"]:
        target = result["target"]
        window = result["target_windows"][0]
        values = window["methods"]["linear_log_age"]
        line_rows.append(
            dict(
                name=target["name"],
                z=target["z"],
                source=target["source"],
                observed_flux=target["flux"],
                measurement=target["measurement"],
                model_flux_q16_q50_q84=values["flux_q16_q50_q84"],
                below_reference_fraction=values["fraction_at_or_below_reference_flux"],
                muv_window=[window["muv_low"], window["muv_high"]],
                note="Intrinsic population fraction, not a noise-convolved nondetection probability or p value",
            )
        )
    proxy = read(ROOT / "data_save/popiii_fraction_highz_20260917/mass_proxy.json")
    row = next(r for r in proxy["rows"] if r["z"] == 12.5 and r["halo_mass_msun"] == 1e10)
    np.testing.assert_allclose(
        row["host_probability"] * row["conditional_mean_mass_fraction_proxy"],
        row["all_halos_mean_mass_fraction_proxy"],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    output = dict(
        status="complete",
        epsilon=0.03,
        uvlf_units="cMpc^-3 mag^-1",
        uvlf=rows,
        heii=line_rows,
        conditional_example=row,
        limitations=[
            "UVLF: snapshot prediction versus finite observed redshift bins; no joint likelihood/covariance",
            "z8 uses fixed production dust mapping; its uncertainty is not marginalized",
            "High-z UV proxy: Pop II 1600 A plus Pop III 1500 A, not full survey synthetic photometry",
            "HeII: limited UV windows, fixed SSP/gas assumptions; MUV measurement errors not marginalized",
            "SFRD slide integrates halo masses 1e4--1e15 Msun; UV/HeII runs sample 1e5--1e12 Msun",
            "Same burst prescription does not make differently selected quantities identical",
        ],
        input_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    for r in rows:
        if r["z"] == 8 and r["dataset"] == "Bowler+20":
            print(r)
    print(OUT / "comparison.json")


if __name__ == "__main__":
    main()
