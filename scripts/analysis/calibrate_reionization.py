"""Validate full spatial runs, rank escape fractions, and integrate optical depths."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from auroralf.experiments.reionization_calibration import neutral_residuals, thomson_depth

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crossing(z, q, target):
    indices = np.flatnonzero((q[:-1] < target) & (q[1:] >= target))
    if not len(indices):
        return None
    i = indices[0]
    w = (target - q[i]) / (q[i + 1] - q[i])
    return float(z[i] * (1 - w) + z[i + 1] * w)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    out = Path(plan["data_output"])
    out.mkdir(parents=True, exist_ok=True)
    obs_path = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
    with obs_path.open() as stream:
        obs = list(csv.DictReader(stream))
    reference = json.loads((Path(plan["baseline_run"]) / "manifest.json").read_text())
    hashes = {str(a.plan.resolve()): digest(a.plan), str(obs_path): digest(obs_path)}
    products, rows = {}, []
    for case in plan["cases"]:
        path, pop = Path(case["run"]), case["population"]
        m = json.loads((path / "manifest.json").read_text())
        if m["status"] != "complete":
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
        ]:
            if m[key] != reference[key]:
                raise ValueError(f"Paired simulation mismatch: {key} in {path}")
        model = m["source_manifest"]["resolved_model"]
        for key, value in reference["source_manifest"]["resolved_model"].items():
            if key not in ["fesc_popii", "fesc_popiii"] and model[key] != value:
                raise ValueError(f"Stellar-history mismatch: {key}")
        if model["fesc_popii"] != case["fesc"] or model["fesc_popiii"] != case["fesc"]:
            raise ValueError("Escape fractions must match the declared common value")
        hist = json.loads((path / "histories.json").read_text())[pop]
        z = np.asarray(hist["redshifts"])
        np.testing.assert_array_equal(z, m["redshifts"])
        qv = np.asarray(hist["mean_xhii"])
        qm = np.asarray([r["mass_weighted_xhii"] for r in hist["rows"]])
        np.testing.assert_array_equal(z, [r["z"] for r in hist["rows"]])
        np.testing.assert_array_equal(qv, [r["mean_xhii"] for r in hist["rows"]])
        files = sorted((path / "reionf" / pop).glob("rf_*.npy"))
        if len(files) != len(z):
            raise ValueError(f"Missing field snapshots: {path}")
        for zz in z:
            field = np.load(path / "reionf" / pop / f"rf_{zz:.2f}.npy", mmap_mode="r")
            if field.shape != (300, 300, 300) or field.dtype != np.float32:
                raise ValueError(f"Invalid field shape or type: {path}")
        cosmology = {k: model[k] for k in ["h", "omega_m", "omega_b"]}
        tau = thomson_depth(z, qm, **cosmology)
        fine = thomson_depth(z, qm, dz=0.001, **cosmology)
        late = thomson_depth(z, qm, completion_redshift=5, **cosmology)
        early = thomson_depth(z, qm, completion_redshift=float(z[-1]), **cosmology)
        volume = thomson_depth(z, qv, **cosmology)
        error = abs(fine["total"] - tau["total"])
        if error > 1e-7:
            raise ValueError(f"Optical-depth quadrature not converged: {error}")
        score, residuals = neutral_residuals(z, qv, obs)
        tau_residual = (tau["total"] - plan["tau_reference"]["value"]) / plan["tau_reference"][
            "sigma"
        ]
        key = f"{pop}_f{case['fesc']:g}"
        for name, values in dict(
            z=z,
            q_volume=qv,
            q_mass=qm,
            tau_z=tau["z"],
            tau=tau["tau"],
            dtau_dz=tau["dtau_dz"],
            tau_late=np.interp(tau["z"], late["z"], late["tau"]),
            tau_early=np.interp(tau["z"], early["z"], early["tau"]),
        ).items():
            products[f"{key}_{name}"] = values
        rows.append(
            dict(
                key=key,
                population=pop,
                fesc=case["fesc"],
                run=str(path),
                neutral_score=score,
                tau=tau["total"],
                tau_residual=tau_residual,
                combined_score=score + tau_residual**2,
                tau_lowz_completion_bounds=[late["total"], early["total"]],
                tau_using_volume_fraction=volume["total"],
                integration_error=error,
                simulated_tau_above_zmin=float(
                    tau["total"] - np.interp(z[-1], tau["z"], tau["tau"])
                ),
                tau_above_z={
                    str(zz): float(tau["total"] - np.interp(zz, tau["z"], tau["tau"]))
                    for zz in [8, 10, 15, 20, 30]
                },
                xhi_z7=float(np.interp(7, z[::-1], (1 - qv)[::-1])),
                xhi_z754=float(np.interp(7.54, z[::-1], (1 - qv)[::-1])),
                xhi_zmin=float(1 - qv[-1]),
                z50=crossing(z, qv, 0.5),
                z99=crossing(z, qv, 0.99),
                observations=residuals,
                snapshots=len(files),
                job=m["job"],
                elapsed_seconds=m["completed_unix"] - m["started_unix"],
            )
        )
        for name in ["manifest.json", "histories.json"]:
            hashes[str(path / name)] = digest(path / name)
        print(
            f"Validated {key}: {len(files)} snapshots, S_HI={score:.6f}, tau={tau['total']:.7f}",
            flush=True,
        )
    selected = {}
    for pop in ["popii", "popii_popiii"]:
        candidates = sorted([r for r in rows if r["population"] == pop], key=lambda r: r["fesc"])
        for low, high in zip(candidates[:-1], candidates[1:], strict=True):
            if np.any(
                products[low["key"] + "_q_volume"] > products[high["key"] + "_q_volume"] + 2e-6
            ):
                raise ValueError("Nonmonotonic escape-fraction response")
        selected[pop] = {
            kind: min(candidates, key=lambda r: r[kind])["key"]
            for kind in ["neutral_score", "combined_score"]
        }
    summary = dict(
        status="complete",
        results=rows,
        selected=selected,
        assumptions=plan["assumptions"],
        tau_reference=plan["tau_reference"],
        input_sha256=hashes,
    )
    for file in [Path(__file__), ROOT / "auroralf/experiments/reionization_calibration.py"]:
        hashes[str(file)] = digest(file)
    np.savez_compressed(out / "histories_tau.npz", **products)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "scores.csv").open("w") as stream:
        keys = [
            "population",
            "fesc",
            "neutral_score",
            "tau",
            "tau_residual",
            "combined_score",
            "xhi_z7",
            "xhi_z754",
            "xhi_zmin",
            "z50",
            "z99",
            "integration_error",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(selected, indent=2))
    print((out / "scores.csv").read_text())


if __name__ == "__main__":
    main()
