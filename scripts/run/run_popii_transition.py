"""Mixed-population UVLF/rates; default Pop II onset follows Pop III, at zero delay."""

# ruff: noqa: E402
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from auroralf.experiments.artifacts import digest
from auroralf.experiments.ionizing_rates import RateConfig
from auroralf.experiments.transition_workers import (
    DELAYS_MYR,
    initialize,
    rate_cell,
    uv_cell,
    validate_variants,
)
from auroralf.seeding import derive_hmf_mass_seed
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "configs/uvlf/popii_popiii.json"


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--kind", choices=["uvlf", "rates"], required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    variants = validate_variants(plan["variants"])
    model = plan["model"].copy()
    for key in ("popii_ssp", "popiii_ssp"):
        model[key] = str((ROOT / model[key]).resolve(strict=True))
    uvpath = str((ROOT / plan["popiii_uv"]).resolve(strict=True))
    if args.kind == "uvlf":
        model["n_tracks"] = plan["uvlf"]["n_tracks"]
    cfg = RateConfig(**model)
    initialize(model, uvpath if args.kind == "uvlf" else None, variants)
    if args.validate_only:
        print(f"Validated {args.kind}: real SSPs and model; variants={variants}", flush=True)
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM allocation required")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    out = ROOT / plan["output"] / args.kind
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        args.plan.resolve(),
        Path(model["popii_ssp"]),
        Path(model["popiii_ssp"]),
        Path(uvpath),
        ROOT / "uv.lock",
    ]
    sources = list((ROOT / "auroralf").rglob("*.py")) + [Path(__file__)]
    manifest = dict(
        status="running",
        plan=plan,
        resolved_model=model,
        job_id=os.environ["SLURM_JOB_ID"],
        started_unix=time.time(),
        variants=list(variants),
        popii_onset={
            v: {
                "mode": "independent" if v == "baseline" else "after_popiii",
                "effective_delay_myr": DELAYS_MYR[v],
            }
            for v in variants
        },
        diagnostic_columns=["popii_" + v for v in variants] + ["popiii", "censored_probability"],
        popii_diagnostic_variants=list(variants),
        source_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(p): digest(p) for p in inputs},
        transition="Birth-time truncation of existing delayed SFR; no mass renormalization; internal successful enrichment only; same q controls Pop III burst and Pop II onset",
        left_censored_popii="Conservative onset at t_start+delay; additional uncertainty bound retained for source rates; not a fabricated physical burst date",
        products={},
    )
    write_json(out / "manifest.json", manifest)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(model, uvpath if args.kind == "uvlf" else None, variants),
    ) as pool:
        if args.kind == "rates":
            grid = plan["rates"]
            redshifts = np.asarray(grid["redshifts"])
            mass = np.logspace(grid["logmass_min"], grid["logmass_max"], grid["n_mass"])
            tasks = [
                (iz, im, float(z), float(m))
                for iz, z in enumerate(redshifts)
                for im, m in enumerate(mass)
            ]
            results = {}
            for first in range(0, len(tasks), 2 * workers):
                for iz, im, values in pool.map(rate_cell, tasks[first : first + 2 * workers]):
                    for key, value in values.items():
                        value = np.asarray(value)
                        if key not in results:
                            results[key] = np.full(
                                (len(redshifts), len(mass)) + value.shape, np.nan
                            )
                        results[key][iz, im] = value
                manifest["completed_cells"] = min(first + 2 * workers, len(tasks))
                write_json(out / "manifest.json", manifest)
                print(
                    f"rates {manifest['completed_cells']}/{len(tasks)} {time.time() - manifest['started_unix']:.1f}s",
                    flush=True,
                )
            if not all(np.isfinite(v).all() for v in results.values()):
                raise FloatingPointError("Incomplete source table")
            for i, name in enumerate(variants):
                target = out / name
                target.mkdir()
                product = target / "sources.npz"
                np.savez_compressed(
                    product,
                    redshifts=redshifts,
                    mass_msun=mass,
                    **{
                        k: results[k][:, :, i]
                        for k in ("mean_rate", "se_rate", "rate_covariance_of_mean")
                    },
                    quadrature_error=results["quadrature_error"],
                )
                record = dict(
                    manifest,
                    status="complete",
                    source_format="instantaneous-rates-v1",
                    channels=["popii", "popiii_resolved", "popiii_censored_upper_extra"],
                    config={"model": model, "grid": grid, "output": str(target)},
                    product_sha256=digest(product),
                    variant=name,
                    delay_myr=DELAYS_MYR[name],
                    popii_onset=manifest["popii_onset"][name],
                    max_stellar_age_myr={
                        "popii": cfg.max_lookback_myr,
                        "popiii": cfg.popiii_max_age_myr,
                    },
                    units="escaped photons/s per halo; no HMF weight",
                    temporal_closure="independent final-halo main branches; merged secondary branches absent",
                )
                write_json(target / "manifest.json", record)
                manifest["products"][f"{name}/sources.npz"] = digest(product)
            diag = out / "diagnostics.npz"
            np.savez_compressed(
                diag,
                redshifts=redshifts,
                mass_msun=mass,
                popii_quadrature_error=results["popii_quadrature_error"],
                popii_censored_upper_extra=results["popii_censored_upper_extra"],
            )
            manifest["products"][diag.name] = digest(diag)
        else:
            grid = plan["uvlf"]
            edges = np.arange(-28.0, 2.01, 0.5)
            for z in grid["redshifts"]:
                rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, z))
                mass = 10 ** rng.uniform(grid["logmass_min"], grid["logmass_max"], grid["n_mass"])
                hmf = prepare_reed07_hmf_interpolator(
                    log10_halo_mass_min_msun=grid["logmass_min"],
                    log10_halo_mass_max_msun=grid["logmass_max"],
                    z_obs=z,
                    cosmology=cfg.cosmology(),
                )
                weights = (
                    (grid["logmass_max"] - grid["logmass_min"])
                    * np.log(10)
                    * mass
                    * hmf.evaluate(mass)
                    / len(mass)
                )
                per_mass = np.empty((len(mass), len(variants), 3, len(edges) - 1))
                diagnostic = np.empty((len(mass), len(variants) + 2))
                for first in range(0, len(mass), 2 * workers):
                    tasks = [
                        (i, z, mass[i], edges)
                        for i in range(first, min(first + 2 * workers, len(mass)))
                    ]
                    for i, hist, diag in pool.map(uv_cell, tasks):
                        per_mass[i] = hist * weights[i] / np.diff(edges)
                        diagnostic[i] = diag
                    print(
                        f"uvlf z={z:g} {first + len(tasks)}/{len(mass)} {time.time() - manifest['started_unix']:.1f}s",
                        flush=True,
                    )
                path = out / f"z{z:g}.npz"
                difference = per_mass - per_mass[:, :1]
                np.savez_compressed(
                    path,
                    redshift=z,
                    mass_msun=mass,
                    weight_per_halo=weights,
                    edges=edges,
                    phi=per_mass.sum(0),
                    se=np.sqrt(len(mass) * per_mass.var(0, ddof=1)),
                    paired_difference=difference.sum(0),
                    paired_se=np.sqrt(len(mass) * difference.var(0, ddof=1)),
                    per_mass=per_mass,
                    diagnostic=diagnostic,
                    variants=np.asarray(variants),
                    paired_reference_variant=variants[0],
                )
                manifest["products"][path.name] = digest(path)
                write_json(out / "manifest.json", manifest)
    for section in ("source_sha256", "input_sha256"):
        for name, expected in manifest[section].items():
            if digest(Path(name)) != expected:
                raise RuntimeError(f"Input changed: {name}")
    manifest.update(status="complete", completed_unix=time.time())
    write_json(out / "manifest.json", manifest)
    print(out, flush=True)


if __name__ == "__main__":
    main()
