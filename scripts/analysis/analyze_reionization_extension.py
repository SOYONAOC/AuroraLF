"""Validate extended spatial histories and replace assumed low-z completion."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from auroralf.experiments.reionization_calibration import neutral_residuals, thomson_depth
from scripts.analysis.calibrate_reionization import crossing, digest

ROOT = Path(__file__).resolve().parents[2]


def verify_snapshot_means(path, manifest, population, redshift, qv, qm):
    """Recompute summary values from saved fields and the recorded density input."""
    field = np.load(path / "reionf" / population / f"rf_{redshift:.2f}.npy", mmap_mode="r")
    if not np.isfinite(field).all() or np.any((field < 0) | (field > 1)):
        raise ValueError("Invalid ionization field values")
    suffix = f"updated_smoothed_deltax_z{redshift:06.2f}_300_300Mpc"
    matches = [Path(p) for p in manifest["density_sha256"] if Path(p).name == suffix]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous recorded density input")
    density_path = matches[0]
    if digest(density_path) != manifest["density_sha256"][str(density_path)]:
        raise ValueError("Density input changed after simulation")
    # Same explicit float32, Fortran-order, clipped density convention as eorcalc.io.
    density = np.fromfile(density_path, dtype=np.float32).reshape(field.shape, order="F")
    rho = 1 + np.clip(density, -0.95, 6.0)
    actual_v = float(field.mean(dtype=np.float64))
    actual_m = float(np.sum(rho * field, dtype=np.float64) / np.sum(rho, dtype=np.float64))
    np.testing.assert_allclose([actual_v, actual_m], [qv, qm], atol=1e-12, rtol=0)
    return dict(
        z=redshift,
        q_volume=actual_v,
        q_mass=actual_m,
        field_sha256=digest(path / "reionf" / population / f"rf_{redshift:.2f}.npy"),
    )


def validate_pair(current, previous):
    """Check identical dynamics, source parameters, and overlapping density inputs."""
    if current["status"] != "complete" or previous["status"] != "complete":
        raise ValueError("Incomplete spatial run")
    for key in [
        "power",
        "physics",
        "grid",
        "runner_sha256",
        "code_sha256",
        "initialization",
        "temporal",
        "spatial",
        "recombination",
        "popiii_max_redshift",
        "popiii_min_redshift",
        "density_provenance_status",
    ]:
        if current[key] != previous[key]:
            raise ValueError(f"Changed spatial configuration: {key}")
    if (
        current["source_manifest"]["resolved_model"]
        != previous["source_manifest"]["resolved_model"]
    ):
        raise ValueError("Changed stellar model")
    nz = len(previous["redshifts"])
    np.testing.assert_array_equal(current["redshifts"][:nz], previous["redshifts"])
    if len(current["redshifts"]) <= nz or current["redshifts"][-1] != 5.01:
        raise ValueError("Expected extension to the real final density snapshot z=5.01")
    for key, value in previous["density_sha256"].items():
        if current["density_sha256"].get(key) != value:
            raise ValueError(f"Changed density input: {key}")
    return nz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    old_summary = json.loads(Path(plan["previous_summary"]).read_text())
    old_rows = {r["key"]: r for r in old_summary["results"]}
    obs_path = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
    with obs_path.open() as stream:
        observations = list(csv.DictReader(stream))
    hashes = {
        str(p.resolve()): digest(p)
        for p in [
            a.plan,
            obs_path,
            Path(plan["previous_summary"]),
            Path(__file__),
            ROOT / "auroralf/experiments/reionization_calibration.py",
        ]
    }
    products, rows = {}, []
    reference = None
    for case in plan["cases"]:
        path, old_path = Path(case["run"]), Path(case["previous_run"])
        pop, fesc = case["population"], case["fesc"]
        key = f"{pop}_f{fesc:g}"
        manifest = json.loads((path / "manifest.json").read_text())
        old_manifest = json.loads((old_path / "manifest.json").read_text())
        nold = validate_pair(manifest, old_manifest)
        if reference is not None:
            for item in ["redshifts", "density_sha256", "power", "physics", "grid"]:
                if manifest[item] != reference[item]:
                    raise ValueError(f"Extended runs differ in {item}")
        reference = manifest
        model = manifest["source_manifest"]["resolved_model"]
        if model["fesc_popii"] != fesc or model["fesc_popiii"] != fesc:
            raise ValueError("Mismatch in declared common escape fraction")
        hist = json.loads((path / "histories.json").read_text())[pop]
        old_hist = json.loads((old_path / "histories.json").read_text())[pop]
        z = np.array(hist["redshifts"])
        qv = np.array(hist["mean_xhii"])
        qm = np.array([r["mass_weighted_xhii"] for r in hist["rows"]])
        np.testing.assert_array_equal(z, manifest["redshifts"])
        np.testing.assert_array_equal(z, [r["z"] for r in hist["rows"]])
        np.testing.assert_array_equal(qv, [r["mean_xhii"] for r in hist["rows"]])
        old_qv = np.array(old_hist["mean_xhii"])
        old_qm = np.array([r["mass_weighted_xhii"] for r in old_hist["rows"]])
        # CUDA reductions across GPU models can differ at float32 roundoff.
        np.testing.assert_allclose(qv[:nold], old_qv, atol=2e-6, rtol=0)
        np.testing.assert_allclose(qm[:nold], old_qm, atol=2e-6, rtol=0)
        files = list((path / "reionf" / pop).glob("rf_*.npy"))
        if len(files) != len(z):
            raise ValueError("Missing spatial snapshots")
        for zz in z:
            field = np.load(path / "reionf" / pop / f"rf_{zz:.2f}.npy", mmap_mode="r")
            if field.shape != (300, 300, 300) or field.dtype != np.float32:
                raise ValueError("Invalid spatial snapshot shape/dtype")
        field_checks = []
        for zz in [6.01, 5.91, 5.59, 5.01]:
            i = int(np.flatnonzero(z == zz)[0])
            field_checks.append(verify_snapshot_means(path, manifest, pop, zz, qv[i], qm[i]))
        # This continuation is allowed only after simulated completion.
        if qv[-1] != 1 or qm[-1] != 1:
            raise ValueError(
                f"{key} is not fully ionized at z=5.01; specify low-z physics explicitly"
            )
        cosmology = {k: model[k] for k in ["h", "omega_m", "omega_b"]}
        tau = thomson_depth(z, qm, completion_redshift=float(z[-1]), **cosmology)
        fine = thomson_depth(z, qm, completion_redshift=float(z[-1]), dz=0.001, **cosmology)
        error = abs(tau["total"] - fine["total"])
        if error > 1e-7:
            raise ValueError(f"Optical depth not converged: {error}")
        old_tau = thomson_depth(z[:nold], qm[:nold], **cosmology)
        score, residuals = neutral_residuals(z, qv, observations)
        tau_residual = (tau["total"] - plan["tau_reference"]["value"]) / plan["tau_reference"][
            "sigma"
        ]
        for name, arr in dict(
            z=z,
            q_volume=qv,
            q_mass=qm,
            tau_z=tau["z"],
            tau=tau["tau"],
            dtau_dz=tau["dtau_dz"],
            old_tau=np.interp(tau["z"], old_tau["z"], old_tau["tau"]),
        ).items():
            products[f"{key}_{name}"] = arr
        rows.append(
            dict(
                key=key,
                population=pop,
                fesc=fesc,
                run=str(path),
                snapshots=len(z),
                zmin=float(z[-1]),
                z50=crossing(z, qv, 0.5),
                z99=crossing(z, qv, 0.99),
                xhi_zmin=float(1 - qv[-1]),
                xhi_z7=float(np.interp(7, z[::-1], (1 - qv)[::-1])),
                xhi_z59=float(np.interp(5.9, z[::-1], (1 - qv)[::-1])),
                xhi_z56=float(np.interp(5.6, z[::-1], (1 - qv)[::-1])),
                tau=tau["total"],
                previous_tau=old_rows[key]["tau"],
                tau_change=tau["total"] - old_rows[key]["tau"],
                integration_error=error,
                neutral_score=score,
                combined_score=score + tau_residual**2,
                tau_above_z={
                    str(zz): float(tau["total"] - np.interp(zz, tau["z"], tau["tau"]))
                    for zz in [8, 10, 15, 20, 30]
                },
                overlap_max_absolute_change=dict(
                    volume=float(np.max(abs(qv[:nold] - old_qv))),
                    mass=float(np.max(abs(qm[:nold] - old_qm))),
                ),
                observations=residuals,
                direct_field_checks=field_checks,
                job=manifest["job"],
                elapsed_seconds=manifest["completed_unix"] - manifest["started_unix"],
            )
        )
        for directory in [path, old_path]:
            for name in ["manifest.json", "histories.json"]:
                hashes[str(directory / name)] = digest(directory / name)
        print(
            f"{key}: {len(z)} snapshots, tau={tau['total']:.8f}, z99={rows[-1]['z99']}", flush=True
        )
    out = Path(plan["data_output"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "histories_tau.npz", **products)
    result = dict(
        status="complete",
        results=rows,
        assumptions=plan["assumptions"],
        tau_reference=plan["tau_reference"],
        input_sha256=hashes,
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (out / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "population",
                "fesc",
                "tau",
                "previous_tau",
                "tau_change",
                "z99",
                "xhi_z59",
                "xhi_z56",
                "xhi_z7",
                "xhi_zmin",
                "snapshots",
                "integration_error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
