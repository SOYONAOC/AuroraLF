"""SLURM-only exact-redshift He II populations for the V24 observation overlay."""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

from auroralf.experiments.heii import load_kernel  # noqa: E402
from auroralf.experiments.random_q import Config, initialize_worker, one_mass  # noqa: E402
from auroralf.mah import Cosmology  # noqa: E402
from auroralf.seeding import derive_hmf_mass_seed  # noqa: E402
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator  # noqa: E402
from scripts.analysis.heii_v24_comparison import (  # noqa: E402
    ROOT,
    digest,
    observations,
    summarize_sample,
)


def verify(plan):
    for name, expected in plan["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Save verified samples and summaries without generating a legacy deck",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify(plan)
    configs = [Config(**record) for record in plan["configs"]]
    kernel = load_kernel(Path(configs[0].popiii_ssp), Path(plan["line_ssp"]))
    catalog = observations()
    if args.validate_only:
        print(
            "Frozen inputs, observational conversions and He II kernel validated; no histories generated."
        )
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM allocation required")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    if workers < 1:
        raise ValueError("Invalid SLURM CPU allocation")
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        job_id=os.environ["SLURM_JOB_ID"],
        workers=workers,
        started_unix=time.time(),
        plan_sha256=digest(args.plan),
        plan=plan,
        population_selection="Main comparison: resolved bursts with age <=3 Myr and total intrinsic MUV<=-20; 16/50/84% HMF-weighted luminosity quantiles, not MC error or detectability",
        rx_selection="All known histories within +/-0.25 mag of de-lensed and Calzetti-de-reddened RXJ2129 component-A MUV; no burst-age cut; status=0 contributes zero; status=2 is unknown",
        scope="Phenomenological one-burst-per-main-branch model; no metal/pristine closure or aperture geometry. Local clumps only provide luminosity-scale context.",
        products={},
    )

    def save():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in plan["input_sha256"]:
            path = ROOT / name
            if path.suffix in (".py", ".toml", ".json", ".lock"):
                archive.add(path, arcname=name, recursive=False)
        archive.add(args.plan, arcname=str(args.plan.resolve().relative_to(ROOT)), recursive=False)
    manifest["products"]["source.tar.gz"] = digest(output / "source.tar.gz")
    results = []
    cosmo = Cosmology()
    for original in configs:
        cfg = Config(**(asdict(original) | {"workers": workers}))
        rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, cfg.z))
        mass = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, cfg.n_mass)
        hmf = prepare_reed07_hmf_interpolator(
            log10_halo_mass_min_msun=cfg.logmass_min,
            log10_halo_mass_max_msun=cfg.logmass_max,
            z_obs=cfg.z,
            cosmology=cosmo,
        )
        weights = (
            (cfg.logmass_max - cfg.logmass_min)
            * np.log(10)
            * mass
            * hmf.evaluate(mass)
            / cfg.n_mass
            / cfg.n_tracks
        )
        names = ("popii", "popiii_per_efficiency", "age_myr", "burst_halo_mass_msun", "status")
        arrays = {
            name: np.empty((cfg.n_mass, cfg.n_tracks), dtype=np.int8 if name == "status" else float)
            for name in names
        }
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(cfg,),
        ) as pool:
            for first in range(0, cfg.n_mass, 2 * workers):
                tasks = [(i, mass[i]) for i in range(first, min(first + 2 * workers, cfg.n_mass))]
                for i, values in pool.map(one_mass, tasks):
                    for name in names:
                        arrays[name][i] = values[name]
                print(f"z={cfg.z:g} mass={first + len(tasks)}/{cfg.n_mass}", flush=True)
        arrays.update(redshift=np.array(cfg.z), mass_msun=mass, weight_per_track=weights)
        path = output / f"z{cfg.z:g}.npz"
        np.savez_compressed(path, **arrays)
        manifest["products"][path.name] = digest(path)
        results.append(summarize_sample(arrays, kernel, cfg.efficiencies, catalog))
        save()
    verify(plan)
    summary = dict(
        observations=catalog,
        model=results,
        population_selection=manifest["population_selection"],
        rx_selection=manifest["rx_selection"],
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest["products"][summary_path.name] = digest(summary_path)
    if args.data_only:
        manifest.update(status="complete", finished_unix=time.time())
        save()
        print(f"COMPLETE: {summary_path}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plot/plot_heii_v24_targets.py"),
            "--summary",
            str(summary_path),
            "--base-deck",
            plan["base_deck"],
            "--deck",
            plan["final_deck"],
        ],
        check=True,
        cwd=ROOT,
    )
    pdf = ROOT / plan["final_deck"] / "popiii_heii_pisn.pdf"
    manifest["slide_pdf"] = str(pdf)
    manifest["slide_sha256"] = digest(pdf)
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    save()
    print(f"COMPLETE: {pdf}", flush=True)


if __name__ == "__main__":
    main()
