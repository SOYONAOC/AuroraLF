"""Fresh low-z conditional UVLF histories, split into disjoint mass shards on cp6.

Run from a frozen release. No old q samples are retained. The existing tested
age-strata worker is used unchanged, with the new mean supplied in the config.
"""

# ruff: noqa: E402
# Set BLAS thread limits before importing numerical modules in spawned workers.
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from auroralf.experiments.artifacts import digest
from auroralf.experiments.deployment import verify_release
from auroralf.experiments.random_q import Config, initialize_worker
from auroralf.experiments.uvlf_components import conditional_histograms
from auroralf.mah import Cosmology
from auroralf.seeding import derive_hmf_mass_seed
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator
from scripts.run.increase_lowz_uvlf_sampling import one_mass
from scripts.run.run_random_q_burst import check_compute_site, shard_indices

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    check_compute_site("cp6", os.environ)
    verify_release(ROOT)
    cfg = Config.load(args.config)
    if cfg.q_log10_mean != 0 or cfg.q_log10_sigma != 1.5 or cfg.efficiencies != [0.03]:
        raise ValueError("Unexpected threshold-zero configuration")
    if cfg.workers > int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("Insufficient allocated CPUs")
    indices = shard_indices(cfg.n_mass, args.shard_index, args.shards)
    out = ROOT / "data_save" / f"{cfg.run_id}-shard{args.shard_index}-of{args.shards}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        config=asdict(cfg),
        job=os.environ["SLURM_JOB_ID"],
        started_unix=time.time(),
        shard_index=args.shard_index,
        shards=args.shards,
        deployment_sha256=digest(ROOT / "deployment.json"),
        sampling="Fresh MAHs and exact Normal-CDF age strata; disjoint global mass indices; global N_mass normalization",
        products={},
    )

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, cfg.z))
        masses = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, cfg.n_mass)
        hmf = prepare_reed07_hmf_interpolator(
            log10_halo_mass_min_msun=cfg.logmass_min,
            log10_halo_mass_max_msun=cfg.logmass_max,
            z_obs=cfg.z,
            cosmology=Cosmology(),
        )
        weights = (
            (cfg.logmass_max - cfg.logmass_min)
            * np.log(10)
            * masses
            * hmf.evaluate(masses)
            / cfg.n_mass
            / cfg.n_tracks
        )
        edges = np.arange(-28.0, 2.01, 0.5)
        per_mass = np.empty((4, len(indices), len(edges) - 1))
        arrays = {
            "popii": np.empty((len(indices), cfg.n_tracks)),
            "popiii": np.empty((len(indices), cfg.n_tracks, 5)),
            "probability": np.empty((len(indices), cfg.n_tracks, 5)),
        }
        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(cfg,),
        ) as pool:
            for first in range(0, len(indices), 2 * cfg.workers):
                tasks = [(int(i), masses[i], 0) for i in indices[first : first + 2 * cfg.workers]]
                for i, p2, p3, probability in pool.map(one_mass, tasks):
                    j = i - indices[0]
                    arrays["popii"][j], arrays["popiii"][j], arrays["probability"][j] = (
                        p2,
                        p3,
                        probability,
                    )
                    for k, eps in enumerate((0.01, 0.03, 0.1), 1):
                        hist = conditional_histograms(
                            p2[None],
                            (p3 * (eps / 0.03))[None],
                            probability[None],
                            weights[i : i + 1],
                            edges,
                        )
                        per_mass[k, j] = hist[2, 0]
                        if k == 1:
                            per_mass[0, j] = hist[0, 0]
                manifest["mass_done"] = first + len(tasks)
                save()
                print(
                    f"z={cfg.z} shard={args.shard_index} mass_done={manifest['mass_done']}/{len(indices)} elapsed_s={time.time() - manifest['started_unix']:.1f}",
                    flush=True,
                )
        np.savez_compressed(
            out / "curves.npz",
            bin_edges=edges,
            per_mass=per_mass,
            global_mass_index=indices,
            mass_msun=masses[indices],
            weight_per_track=weights[indices],
        )
        np.savez_compressed(out / "samples.npz", **arrays)
        verify_release(ROOT)
        manifest.update(
            status="complete",
            completed_unix=time.time(),
            products={name: digest(out / name) for name in ("curves.npz", "samples.npz")},
        )
        save()
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
