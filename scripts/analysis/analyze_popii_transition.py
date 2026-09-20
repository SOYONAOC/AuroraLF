"""Paired UVLF statistics and validated spatial histories for Pop II onset gates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from auroralf.experiments.artifacts import digest
from auroralf.experiments.reionization_calibration import neutral_residuals, thomson_depth
from scripts.analysis.analyze_reionization_extension import verify_snapshot_means
from scripts.analysis.calibrate_reionization import crossing

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_save/popii_transition_20260920"
NAMES = ["baseline", "delay0", "delay30"]


def ratio_and_se(numerator, denominator):
    """Paired ratio-of-sums delta-method SE across independent mass draws."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if (
        numerator.shape != denominator.shape
        or len(numerator) < 2
        or not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
    ):
        raise ValueError("Invalid paired ratio samples")
    total = denominator.sum(axis=0)
    ratio = np.divide(
        numerator.sum(axis=0), total, out=np.full_like(total, np.nan, dtype=float), where=total > 0
    )
    residual = numerator - ratio * denominator
    error = np.divide(
        np.sqrt(len(numerator) * np.var(residual, axis=0, ddof=1)),
        total,
        out=np.full_like(total, np.nan, dtype=float),
        where=total > 0,
    )
    return ratio, error


def uvlf():
    manifest = json.loads((DATA / "uvlf/manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("UVLF incomplete")
    rows = []
    products = {}
    for name, sha in manifest["products"].items():
        path = DATA / "uvlf" / name
        if digest(path) != sha:
            raise ValueError("UVLF hash mismatch")
        with np.load(path) as d:
            z = float(d["redshift"])
            pm = d["per_mass"]
            phi = pm.sum(axis=0)
            np.testing.assert_allclose(phi, d["phi"], rtol=1e-14)
            np.testing.assert_array_equal(phi[0, 1], phi[1, 1])
            np.testing.assert_array_equal(phi[0, 1], phi[2, 1])
            ratios = []
            errors = []
            for v in range(3):
                r, e = ratio_and_se(pm[:, v], pm[:, 0])
                ratios.append(r)
                errors.append(e)
            products[f"z{z:g}_ratio"] = np.stack(ratios)
            products[f"z{z:g}_ratio_se"] = np.stack(errors)
            products[f"z{z:g}_centers"] = (d["edges"][1:] + d["edges"][:-1]) / 2
            weights = d["weight_per_halo"]
            diag = d["diagnostic"]
            popii = weights[:, None] * diag[:, :3]
            popiii = weights * diag[:, 3]
            ld = popii.sum(0)
            ldtotal = ld + popiii.sum()
            rows.append(
                dict(
                    z=z,
                    popii_luminosity_density=ld.tolist(),
                    total_luminosity_density=ldtotal.tolist(),
                    popii_relative=[
                        float(ratio_and_se(popii[:, v], popii[:, 0])[0]) for v in range(3)
                    ],
                    popii_relative_se=[
                        float(ratio_and_se(popii[:, v], popii[:, 0])[1]) for v in range(3)
                    ],
                    total_relative=[
                        float(ratio_and_se(popii[:, v] + popiii, popii[:, 0] + popiii)[0])
                        for v in range(3)
                    ],
                    total_relative_se=[
                        float(ratio_and_se(popii[:, v] + popiii, popii[:, 0] + popiii)[1])
                        for v in range(3)
                    ],
                )
            )
    rows.sort(key=lambda row: row["z"])
    np.savez_compressed(DATA / "uvlf_ratios.npz", **products)
    summary = dict(
        status="complete",
        variants=NAMES,
        rows=rows,
        units="erg/s/Hz/Mpc^3, intrinsic 1500 A",
        integral_scope="full sampled halo mass range 1e4-1e15 Msun, all luminosities; no magnitude selection",
        uncertainty="Paired mass-draw clustered delta-method SE; one batch, not total model or observational error",
        source_manifest_sha256=digest(DATA / "uvlf/manifest.json"),
    )
    (DATA / "uvlf_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


def spatial():
    planpath = ROOT / "configs/experiments/popii_transition_spatial_20260920.json"
    plan = json.loads(planpath.read_text())
    obs = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
    with obs.open() as f:
        observations = list(csv.DictReader(f))
    keys = [
        "power",
        "physics",
        "grid",
        "redshifts",
        "density_sha256",
        "code_sha256",
        "runner_sha256",
        "temporal",
        "initialization",
        "spatial",
        "source_limit",
        "recombination",
        "brightness",
        "popiii_max_redshift",
        "popiii_min_redshift",
        "density_provenance_status",
    ]
    rows = []
    products = {}
    reference = None
    for case in plan["cases"]:
        name = case["name"]
        path = Path(case["run"])
        m = json.loads((path / "manifest.json").read_text())
        if m["status"] != "complete":
            raise ValueError("Spatial run incomplete: " + name)
        if reference is None:
            reference = m
        else:
            for key in keys:
                if m[key] != reference[key]:
                    raise ValueError("Spatial mismatch: " + key)
        model = m["source_manifest"]["resolved_model"]
        if (
            model["fesc_popii"] != plan["common_fesc"]
            or model["fesc_popiii"] != plan["common_fesc"]
        ):
            raise ValueError("Changed fesc")
        hist = json.loads((path / "histories.json").read_text())["popii_popiii"]
        z = np.asarray(hist["redshifts"])
        qv = np.asarray(hist["mean_xhii"])
        qm = np.asarray([r["mass_weighted_xhii"] for r in hist["rows"]])
        np.testing.assert_array_equal(z, m["redshifts"])
        np.testing.assert_array_equal(z, [r["z"] for r in hist["rows"]])
        if len(z) != 235 or z[-1] != 5.01:
            raise ValueError("Expected all235 snapshots through z5.01")
        if len(list((path / "reionf/popii_popiii").glob("rf_*.npy"))) != len(z):
            raise ValueError("Missing spatial snapshots")
        checks = []
        for target in [15, 10, 7, 5.01]:
            i = int(abs(z - target).argmin())
            checks.append(
                verify_snapshot_means(
                    path, m, "popii_popiii", float(z[i]), float(qv[i]), float(qm[i])
                )
            )
        pars = {k: model[k] for k in ("h", "omega_m", "omega_b")}
        # Always retain the exactly simulated high-z contribution. If the
        # end is not ionized, separately label explicit low-z continuations.
        completed = bool(qm[-1] >= 1 - 1e-12 and qv[-1] >= 1 - 1e-12)
        completions = [float(z[-1])] if completed else [float(z[-1]), 4.5, 4.0]
        depths = []
        for end in completions:
            fine = thomson_depth(z, qm, **pars, completion_redshift=end, dz=0.001)
            coarse = thomson_depth(z, qm, **pars, completion_redshift=end, dz=0.002)
            if abs(fine["total"] - coarse["total"]) > 1e-7:
                raise ValueError("Tau integration not converged")
            high = fine["total"] - np.interp(z[-1], fine["z"], fine["tau"])
            depths.append(
                dict(
                    completion_redshift=end,
                    tau=fine["total"],
                    tau_simulated_interval=float(high),
                    step_difference=abs(fine["total"] - coarse["total"]),
                )
            )
        products[name + "_z"] = z
        products[name + "_qv"] = qv
        products[name + "_qm"] = qm
        products[name + "_tau_z"] = fine["z"]
        products[name + "_tau"] = fine["tau"]
        products[name + "_dtau"] = fine["dtau_dz"]
        score, residuals = neutral_residuals(z, qv, observations)
        rows.append(
            dict(
                name=name,
                run=str(path),
                qv_final=float(qv[-1]),
                qm_final=float(qm[-1]),
                fully_ionized_at_last_snapshot=completed,
                xhi_z7=float(np.interp(7, z[::-1], (1 - qv)[::-1])),
                z50=crossing(z, qv, 0.5),
                z99=crossing(z, qv, 0.99),
                tau=depths[0]["tau"] if completed else None,
                tau_continuations=depths,
                neutral_score=score,
                neutral_residuals=residuals,
                field_checks=checks,
                manifest_sha256=digest(path / "manifest.json"),
                history_sha256=digest(path / "histories.json"),
            )
        )
    for name in ["delay0", "delay30"]:
        if np.any(products[name + "_qv"] > products["baseline_qv"] + 1e-6):
            raise ValueError("Gate materially increased ionization")
    np.savez_compressed(DATA / "spatial_history.npz", **products)
    summary = dict(
        status="complete",
        common_fesc=plan["common_fesc"],
        rows=rows,
        source_plan_sha256=digest(planpath),
        observations_sha256=digest(obs),
        limitations="Fixed fesc, no recalibration; density provenance remains unverified; overlapping excursion regions do not ensure exact photon conservation; no full enrichment/gas model.",
    )
    (DATA / "spatial_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_slide_table(rows)
    print(
        json.dumps(
            {
                r["name"]: {k: r[k] for k in ["xhi_z7", "z50", "z99", "tau", "qv_final"]}
                for r in rows
            },
            indent=2,
        )
    )


def write_slide_table(rows):
    """Keep the review table derived from validated science products."""
    uv = json.loads((DATA / "uvlf_summary.json").read_text())["rows"]
    table = [
        r"{\centering\renewcommand{\arraystretch}{1.35}",
        r"\begin{tabular}{lrrr}\toprule",
        r"量 & 原模型 & 零延迟 & 30 Myr\\\midrule",
    ]
    for z in [6, 14.5]:
        values = next(row["total_relative"] for row in uv if row["z"] == z)
        table.append(
            f"总 UV 光度密度比（$z={z:g}$） & "
            + " & ".join(f"{value:.4f}" for value in values)
            + r"\\"
        )
    for key, label in [
        ("xhi_z7", r"$\langle x_{\rm HI}\rangle_V(z=7)$"),
        ("z50", r"$z_{50}$"),
        ("z99", r"$z_{99}$"),
        ("tau", r"$\tau_{\rm e}$"),
    ]:
        table.append(
            label
            + " & "
            + " & ".join(
                r"\textemdash"
                if row[key] is None
                else format(row[key], ".5f" if key == "tau" else ".3f")
                for row in rows
            )
            + r"\\"
        )
    table += [r"\bottomrule\end{tabular}\par}"]
    target = ROOT / "slides/popii_transition_20260920/results.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["uvlf", "spatial"], required=True)
    args = p.parse_args()
    {"uvlf": uvlf, "spatial": spatial}[args.stage]()
