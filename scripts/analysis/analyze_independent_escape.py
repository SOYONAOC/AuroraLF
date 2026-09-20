"""Validate independent escape trials against the existing neutral/tau calibration."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from auroralf.experiments.reionization_calibration import neutral_residuals, thomson_depth
from scripts.analysis.analyze_reionization_extension import verify_snapshot_means
from scripts.analysis.calibrate_reionization import crossing, digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reference = json.loads((Path(plan["reference_run"]) / "manifest.json").read_text())
    obs_path = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
    with obs_path.open() as stream:
        observations = list(csv.DictReader(stream))
    hashes = {
        str(p.resolve()): digest(p)
        for p in [
            args.plan,
            obs_path,
            Path(__file__),
            Path(plan["reference_run"]) / "manifest.json",
            ROOT / "auroralf/experiments/reionization_calibration.py",
            ROOT / "scripts/analysis/analyze_reionization_extension.py",
        ]
    }
    rows, arrays = [], {}
    for case in plan["cases"]:
        path = Path(case["run"])
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete spatial run: {path}")
        for key in [
            "power",
            "physics",
            "grid",
            "density_sha256",
            "redshifts",
            "runner_sha256",
            "code_sha256",
            "initialization",
            "temporal",
            "spatial",
            "recombination",
            "popiii_max_redshift",
            "popiii_min_redshift",
        ]:
            if manifest[key] != reference[key]:
                raise ValueError(f"Paired input mismatch: {key}")
        model = manifest["source_manifest"]["resolved_model"]
        for key, value in reference["source_manifest"]["resolved_model"].items():
            if key not in ["fesc_popii", "fesc_popiii"] and model[key] != value:
                raise ValueError(f"Stellar model changed: {key}")
        for key in ["fesc_popii", "fesc_popiii"]:
            if model[key] != case[key]:
                raise ValueError(f"Escape fraction mismatch: {key}")
        source = Path(case["source"])
        if digest(source / "manifest.json") != manifest["source_manifest_sha256"]:
            raise ValueError("Source manifest changed")
        if digest(source / "sources.npz") != manifest["source_manifest"]["product_sha256"]:
            raise ValueError("Source rates changed")
        history = json.loads((path / "histories.json").read_text())["popii_popiii"]
        z = np.array(history["redshifts"])
        qv = np.array(history["mean_xhii"])
        qm = np.array([r["mass_weighted_xhii"] for r in history["rows"]])
        np.testing.assert_array_equal(z, manifest["redshifts"])
        np.testing.assert_array_equal(z, [r["z"] for r in history["rows"]])
        np.testing.assert_array_equal(qv, [r["mean_xhii"] for r in history["rows"]])
        files = list((path / "reionf/popii_popiii").glob("rf_*.npy"))
        if len(files) != len(z) or z[-1] != 5.01:
            raise ValueError("Missing snapshots or incorrect final redshift")
        for zz in z:
            field = np.load(path / "reionf/popii_popiii" / f"rf_{zz:.2f}.npy", mmap_mode="r")
            if field.shape != (300, 300, 300) or field.dtype != np.float32:
                raise ValueError("Unexpected field shape or dtype")
        checks = []
        for target in [15, 10, 7, 5.01]:
            i = int(np.argmin(abs(z - target)))
            checks.append(verify_snapshot_means(path, manifest, "popii_popiii", z[i], qv[i], qm[i]))
        if qv[-1] != 1 or qm[-1] != 1:
            raise ValueError("Incomplete final ionization: cannot assume fully ionized low z")
        cosmology = {k: model[k] for k in ["h", "omega_m", "omega_b"]}
        tau = thomson_depth(z, qm, completion_redshift=float(z[-1]), **cosmology)
        fine = thomson_depth(z, qm, completion_redshift=float(z[-1]), dz=0.001, **cosmology)
        error = abs(tau["total"] - fine["total"])
        if error > 1e-7:
            raise ValueError("Optical depth integration not converged")
        score, residuals = neutral_residuals(z, qv, observations)
        tau_residual = (tau["total"] - plan["tau_reference"]["value"]) / plan["tau_reference"][
            "sigma"
        ]
        row = dict(
            key=case["key"],
            fesc_popii=case["fesc_popii"],
            fesc_popiii=case["fesc_popiii"],
            tau=tau["total"],
            tau_residual=tau_residual,
            neutral_score=score,
            combined_score=score + tau_residual**2,
            z50=crossing(z, qv, 0.5),
            z99=crossing(z, qv, 0.99),
            xhi_z7=float(np.interp(7, z[::-1], (1 - qv)[::-1])),
            xhi_z754=float(np.interp(7.54, z[::-1], (1 - qv)[::-1])),
            tau_above_z10=float(tau["total"] - np.interp(10, tau["z"], tau["tau"])),
            tau_above_z7=float(tau["total"] - np.interp(7, tau["z"], tau["tau"])),
            observations=residuals,
            integration_error=error,
            direct_field_checks=checks,
            snapshots=len(z),
            job=manifest["job"],
            elapsed_seconds=manifest["completed_unix"] - manifest["started_unix"],
        )
        rows.append(row)
        for name, arr in dict(
            z=z, q_volume=qv, q_mass=qm, tau_z=tau["z"], tau=tau["tau"], dtau_dz=tau["dtau_dz"]
        ).items():
            arrays[case["key"] + "_" + name] = arr
        for p in [
            path / "manifest.json",
            path / "histories.json",
            source / "manifest.json",
            source / "sources.npz",
        ]:
            hashes[str(p)] = digest(p)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ["observations", "direct_field_checks"]}
            ),
            flush=True,
        )
    responses = []
    ordered = sorted(rows, key=lambda r: r["fesc_popii"])
    for low, high in zip(ordered[:-1], ordered[1:], strict=True):
        responses.append(
            dict(
                lower=low["key"],
                higher=high["key"],
                min_delta_q_volume=float(
                    np.min(arrays[high["key"] + "_q_volume"] - arrays[low["key"] + "_q_volume"])
                ),
                min_delta_q_mass=float(
                    np.min(arrays[high["key"] + "_q_mass"] - arrays[low["key"] + "_q_mass"])
                ),
            )
        )
    summary = dict(
        status="complete",
        results=rows,
        assumptions=plan["assumptions"],
        tau_reference=plan["tau_reference"],
        sampled_response_checks=responses,
        input_sha256=hashes,
    )
    out = Path(plan["data_output"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "histories_tau.npz", **arrays)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
