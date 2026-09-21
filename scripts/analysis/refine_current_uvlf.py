"""Reduce rare Pop III burst noise using age-stratified q proposals on identical MAHs."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

from auroralf.cooling import compute_atomic_cooling_mass_msun  # noqa: E402
from auroralf.experiments import random_q  # noqa: E402
from auroralf.experiments.uvlf_components import age_strata, conditional_histograms  # noqa: E402
from auroralf.mah import generate_halo_histories  # noqa: E402
from auroralf.seeding import derive_pipeline_random_seeds  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(config):
    random_q.initialize_worker(random_q.Config(**config))


def one_mass(task):
    index, final_mass, original_p3 = task
    cfg, cosmo, dt, _, _, a3, k3 = random_q.STATE
    logq = random_q.draw_logq(cfg.seed, index, cfg.n_tracks, cfg.q_log10_mean, cfg.q_log10_sigma)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0x53545241, index, int(cfg.z)]))
    uniforms = np.clip(
        rng.random((cfg.n_tracks, 4)), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
    )
    lights, probabilities = [], []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        seed = int(
            np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(seed, redshift=cfg.z, mass_index=index)
        h = generate_halo_histories(
            n_tracks=cfg.track_chunk,
            z_final=cfg.z,
            Mh_final=final_mass,
            z_start_max=cfg.z_start,
            cosmology=cosmo,
            random_seed=seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt,
            store_inactive_history=True,
            sampler="mcbride",
        )
        t, mass, z = [h.tracks[k].reshape(cfg.track_chunk, -1) for k in ["t_gyr", "Mh", "z"]]
        cooling = compute_atomic_cooling_mass_msun(z, cosmology=cosmo)
        fb, eps = cosmo.omega_b / cosmo.omega_m, cfg.efficiencies[0]
        old = random_q.burst_light(
            t, mass, cooling, logq[start : start + cfg.track_chunk], a3, k3, fb, 100.0
        )
        np.testing.assert_allclose(
            eps * old["popiii_per_efficiency"],
            original_p3[start : start + cfg.track_chunk],
            rtol=1e-12,
            atol=0,
        )
        light, prob = age_strata(
            t,
            mass,
            cooling,
            cfg.q_log10_mean,
            cfg.q_log10_sigma,
            uniforms[start : start + cfg.track_chunk],
            a3,
            k3,
            fb,
            eps,
        )
        lights.append(light)
        probabilities.append(prob)
    return index, np.concatenate(lights), np.concatenate(probabilities)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "data_save/uvlf_current_z6_z8_z10")
    p.add_argument(
        "--output", type=Path, default=ROOT / "data_save/uvlf_current_z6_z8_z10_stratified"
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM required")
    if not 1 <= a.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("Insufficient CPUs")
    before = json.loads((a.run / "manifest.json").read_text())
    assert before["status"] == "complete"
    for name, hash_ in before["products"].items():
        assert digest(a.run / name) == hash_
    a.output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(f.resolve()): digest(f)
        for f in [
            Path(__file__),
            ROOT / "auroralf/experiments/uvlf_components.py",
            ROOT / "auroralf/experiments/random_q.py",
            a.run / "manifest.json",
        ]
    }
    manifest = {
        k: v
        for k, v in before.items()
        if k not in ["products", "started_unix", "completed_unix", "job"]
    }
    manifest.update(
        status="running",
        job=os.environ["SLURM_JOB_ID"],
        started_unix=time.time(),
        parent=str(a.run.resolve()),
        sampling="Same mass draws and MAHs; exact Normal CDF probability for burst ages 0–3–10–30–100 Myr and UV-dark remainder; conditional q uniform in CDF per stratum; no change to target q distribution",
        importance_input_sha256=hashes,
        products={},
    )

    def save_manifest():
        (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    for config in before["configs"]:
        z = config["z"]
        with np.load(a.run / f"z{z:g}.npz") as d:
            old = dict(d)
        mass, p2, w, edges = [old[k] for k in ["mass_msun", "popii", "weight_per_track", "edges"]]
        p3 = np.empty((*p2.shape, 5))
        probability = np.empty_like(p3)
        with ProcessPoolExecutor(
            max_workers=a.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(config,),
        ) as pool:
            for first in range(0, len(mass), 2 * a.workers):
                tasks = [
                    (i, mass[i], old["popiii"][i])
                    for i in range(first, min(first + 2 * a.workers, len(mass)))
                ]
                for i, light, prob in pool.map(one_mass, tasks):
                    p3[i], probability[i] = light, prob
                print(f"z={z:g} conditional mass={first + len(tasks)}/{len(mass)}", flush=True)
        per_mass = conditional_histograms(p2, p3, probability, w, edges)
        phi = per_mass.sum(axis=1)
        se = np.sqrt(len(mass) * per_mass.var(axis=1, ddof=1))
        np.testing.assert_allclose(phi[0], old["phi"][0], rtol=1e-11, atol=1e-100)
        path = a.output / f"z{z:g}.npz"
        np.savez_compressed(
            path,
            redshift=z,
            mass_msun=mass,
            weight_per_track=w,
            popii=p2,
            popiii=p3,
            probability=probability,
            edges=edges,
            phi=phi,
            se=se,
            unconditional_counts=old["counts"],
            unconditional_phi=old["phi"],
            unconditional_se=old["se"],
        )
        manifest["products"][path.name] = digest(path)
        save_manifest()
    assert all(digest(f) == h for f, h in hashes.items())
    manifest.update(status="complete", completed_unix=time.time())
    save_manifest()


if __name__ == "__main__":
    main()
