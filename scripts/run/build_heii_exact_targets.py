"""SLURM-only GHZ2/GS-z14-1 histories; preserves the existing high-z physics."""

# Set thread limits before numerical imports in each spawned worker.
# ruff: noqa: E402

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.artifacts import digest
from auroralf.experiments.heii import load_kernel
from auroralf.experiments.random_q import Config, initialize_worker, one_mass
from auroralf.mah import Cosmology
from auroralf.seeding import derive_hmf_mass_seed
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator
from scripts.analysis.heii_exact_targets import summarize_target

ROOT = Path(__file__).resolve().parents[2]


def verify(plan):
    for name, expected in plan["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify(plan)
    configs = [Config.load(ROOT / path) for path in plan["configs"]]
    targets = json.loads((ROOT / plan["observations"]).read_text())["targets"]
    if len(configs) != len(targets) or any(c.z != t["z"] for c, t in zip(configs, targets)):
        raise ValueError("Every configuration must use its target's exact redshift")
    if any(c.efficiencies != configs[0].efficiencies for c in configs):
        raise ValueError("Efficiency grids differ")
    kernel = load_kernel(Path(configs[0].popiii_ssp), ROOT / plan["line_ssp"])
    for cfg in configs:
        initialize_worker(cfg)  # load/validate SSP coverage, no histories
    output = ROOT / plan["output"]
    if output.exists():
        raise FileExistsError(output)
    if args.validate_only:
        print("Validated exact redshifts, frozen inputs and SSP coverage; no histories generated.")
        return
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    if not os.environ.get("SLURM_JOB_ID") or not partition or "debug" in partition.lower():
        raise RuntimeError("A non-debug SLURM allocation is required")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    if workers < 1:
        raise ValueError("Invalid worker allocation")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        job_id=os.environ["SLURM_JOB_ID"],
        workers=workers,
        partition=partition,
        node=os.environ.get("SLURM_JOB_NODELIST"),
        started_unix=time.time(),
        plan=plan,
        plan_sha256=digest(args.plan),
        products={},
    )

    def save():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    try:
        # Preserve the actual numerical implementation, not just its hashes.
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for name in plan["input_sha256"]:
                path = ROOT / name
                if path.suffix in (".py", ".toml", ".json", ".lock"):
                    archive.add(path, arcname=name, recursive=False)
            archive.add(
                args.plan,
                arcname=str(args.plan.relative_to(ROOT))
                if args.plan.is_absolute()
                else str(args.plan),
                recursive=False,
            )
        manifest["products"]["source.tar.gz"] = digest(output / "source.tar.gz")
        cosmo = Cosmology()
        astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
        summaries = {str(e): {"epsilon": e, "results": []} for e in configs[0].efficiencies}
        for original, target in zip(configs, targets):
            cfg = Config(**(asdict(original) | {"workers": workers}))
            rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, cfg.z))
            mass = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, cfg.n_mass)
            hmf = prepare_reed07_hmf_interpolator(
                log10_halo_mass_min_msun=cfg.logmass_min,
                log10_halo_mass_max_msun=cfg.logmass_max,
                z_obs=cfg.z,
                cosmology=cosmo,
            )
            weight = (
                (cfg.logmass_max - cfg.logmass_min)
                * np.log(10)
                * mass
                * hmf.evaluate(mass)
                / cfg.n_mass
                / cfg.n_tracks
            )
            names = ("popii", "popiii_per_efficiency", "age_myr", "burst_halo_mass_msun", "status")
            arrays = {
                n: np.empty((cfg.n_mass, cfg.n_tracks), dtype=np.int8 if n == "status" else float)
                for n in names
            }
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(cfg,),
            ) as pool:
                for first in range(0, cfg.n_mass, 2 * workers):
                    tasks = [
                        (i, mass[i]) for i in range(first, min(first + 2 * workers, cfg.n_mass))
                    ]
                    for i, values in pool.map(one_mass, tasks):
                        for name in names:
                            arrays[name][i] = values[name]
                    print(f"z={cfg.z:g} mass={first + len(tasks)}/{cfg.n_mass}", flush=True)
            for key in ("popii", "popiii_per_efficiency"):
                if not np.isfinite(arrays[key]).all() or np.any(arrays[key] < 0):
                    raise ValueError(f"Invalid UV sample: {key}")
            arrays.update(redshift=np.array(cfg.z), mass_msun=mass, weight_per_track=weight)
            path = output / f"z{cfg.z:g}.npz"
            np.savez_compressed(path, **arrays)
            manifest["products"][path.name] = digest(path)
            by_efficiency = summarize_target(arrays, target, kernel, cfg.efficiencies, astro)
            for eps, result in by_efficiency.items():
                summaries[eps]["results"].append(result)
            save()
        verify(plan)
        summary = dict(
            efficiencies=summaries,
            exact_redshifts=True,
            assumptions=plan["assumptions"],
            source_sha256={str(args.plan.resolve()): digest(args.plan)},
            input_sha256=plan["input_sha256"],
        )
        path = output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        manifest["products"][path.name] = digest(path)
        manifest.update(status="complete", finished_unix=time.time())
        save()
        print(f"COMPLETE: {path}", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=repr(error), finished_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
