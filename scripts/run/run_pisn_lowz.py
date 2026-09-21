"""PISN-only low-z extrapolation of the current random-q model, under SLURM.

No enrichment, pristine survival or survey detectability is supplied. No UV/SSP
calculation is needed for the all-host intrinsic event rate. Config paths are
relative to the repository root. An immutable deployment must be verified.
"""

# Set BLAS limits before importing numerical libraries in parent and workers.
# ruff: noqa: E402
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.experiments.artifacts import digest
from auroralf.experiments.deployment import verify_release
from auroralf.experiments.pisn import load_marigo_kernel, observer_rate_per_deg2
from auroralf.experiments.pisn_first_passage import threshold_averaged_pisn_rate
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator


def initialize(config):
    global STATE
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    STATE = config, cosmo, astro, load_marigo_kernel(config["lifetimes"])


def cell(task):
    im, z, mf, ngrid = task
    cfg, cosmo, astro, kernel = STATE
    dt = (astro.age(z).value - astro.age(cfg["z_start"]).value) / (ngrid - 1)
    samples, differences = [], []
    for start in range(0, cfg["n_tracks"], cfg["track_chunk"]):
        block_seed = int(
            np.random.SeedSequence([cfg["seed"], 0x424C4F43, start]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds = derive_pipeline_random_seeds(block_seed, redshift=z, mass_index=im)
        h = generate_halo_histories(
            n_tracks=cfg["track_chunk"],
            z_final=z,
            Mh_final=mf,
            z_start_max=cfg["z_start"],
            cosmology=cosmo,
            random_seed=seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt,
            store_inactive_history=True,
            sampler="mcbride",
        ).tracks
        t, m, zg = [h[key].reshape(cfg["track_chunk"], -1) for key in ("t_gyr", "Mh", "z")]
        c = compute_atomic_cooling_mass_msun(zg, cosmology=cosmo)
        args = (t, m, c, kernel, cfg["q_log10_mean"], cfg["q_log10_sigma"])
        rate = threshold_averaged_pisn_rate(*args, order=16)
        check = threshold_averaged_pisn_rate(*args, order=8)
        factor = cfg["epsilon_b"] * cosmo.omega_b / cosmo.omega_m
        samples.append(rate * factor)
        differences.append(abs(rate - check) * factor)
    values = np.concatenate(samples)
    return [
        float(values.mean()),
        float(values.std(ddof=1) / np.sqrt(len(values))),
        float(np.concatenate(differences).mean()),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if os.environ.get("SLURM_JOB_PARTITION") != "cp6" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cp6 allocation required")
    verify_release(Path.cwd())
    if cfg["n_tracks"] % cfg["track_chunk"] or cfg["n_tracks"] < 2:
        raise ValueError("invalid track sampling")
    if not (0 < cfg["epsilon_b"] <= 1 and cfg["q_log10_sigma"] > 0):
        raise ValueError("invalid model parameters")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    initialize(cfg)
    _, cosmo, astro, kernel = STATE
    manifest = dict(
        status="running",
        config=cfg,
        job_id=os.environ["SLURM_JOB_ID"],
        deployment_sha256=digest("deployment.json"),
        scope="Unchanged random-q model extrapolation; no enrichment/pristine criterion; all PISNe, no survey selection",
        units="events / source year / comoving Mpc^3",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = []
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(cfg,),
        ) as pool:
            for z in cfg["redshifts"]:
                if not 0 <= z < cfg["z_start"]:
                    raise ValueError("invalid redshift")
                hmf = prepare_reed07_hmf_interpolator(
                    log10_halo_mass_min_msun=cfg["logmass_min"],
                    log10_halo_mass_max_msun=cfg["logmass_max"],
                    z_obs=z,
                    cosmology=cosmo,
                )
                for nm, ng in cfg["resolutions"]:
                    edges = np.linspace(cfg["logmass_min"], cfg["logmass_max"], nm + 1)
                    masses = 10 ** ((edges[:-1] + edges[1:]) / 2)
                    weight = np.diff(edges) * np.log(10) * masses * hmf.evaluate(masses)
                    results = np.asarray(
                        list(pool.map(cell, [(i, z, float(m), ng) for i, m in enumerate(masses)]))
                    )
                    rate = float(weight @ results[:, 0])
                    se = float(np.sqrt(np.sum((weight * results[:, 1]) ** 2)))
                    case = dict(
                        z=z,
                        n_mass=nm,
                        n_grid=ng,
                        source_rate=rate,
                        mah_mc_se=se,
                        quadrature_error=float(weight @ results[:, 2]),
                        observer_rate=float(observer_rate_per_deg2(rate, z, astro)),
                        lowest_mass_decade_fraction=float(
                            np.sum(
                                (weight * results[:, 0])[masses < 10 ** (cfg["logmass_min"] + 1)]
                            )
                            / rate
                        ),
                        highest_mass_decade_fraction=float(
                            np.sum(
                                (weight * results[:, 0])[masses > 10 ** (cfg["logmass_max"] - 1)]
                            )
                            / rate
                        ),
                    )
                    cases.append(case)
                    np.savez_compressed(
                        output / f"z{z:g}_m{nm}_t{ng}.npz",
                        mass_msun=masses,
                        weight=weight,
                        mean_rate=results[:, 0],
                        mah_mc_se=results[:, 1],
                        quadrature_error=results[:, 2],
                    )
                    (output / "summary.json").write_text(
                        json.dumps(dict(config=cfg, cases=cases), indent=2) + "\n"
                    )
                    print(json.dumps(case), flush=True)
        verify_release(Path.cwd())
        manifest.update(
            status="complete",
            products={f.name: digest(f) for f in output.iterdir() if f.name != "manifest.json"},
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
