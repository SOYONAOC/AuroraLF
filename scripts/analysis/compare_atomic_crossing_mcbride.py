"""Compare the unchanged production MAH sampler to each real THESAN endpoint.

128 independent histories per observed mass; equal weight per real target.
Only the MAH/cooling crossing is calculated, not SFR, IMF, UV or enrichment.
"""

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from auroralf.constants import PLANCK15_H0_GYR
from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.mah import Cosmology
from auroralf.mah.sampling import sample_parameters

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data_save/atomic_crossing_z6_20260917"
OUT = ROOT / "data_save/atomic_crossing_mcbride_20260917"
NTRACK = 128
SEED = 9172601
STATE = None


def solve_crossings(beta, gamma, mass, z_final, cosmology, n_scan=257):
    """Find first sampled upcrossing, then solve the exact MAH on its bracket.

    status: 0 = no crossing by endpoint; 1 = bracketed; 2 = already above at z=50.
    The analytic expression is checked against the public history generator in tests.
    """
    z = np.linspace(50.0, z_final, n_scan)
    logcool = np.log(compute_atomic_cooling_mass_msun(z, cosmology=cosmology))
    ratio = (
        np.log(mass)
        + beta[:, None] * np.log((1 + z) / (1 + z_final))
        - gamma[:, None] * (z - z_final)
        - logcool
    )
    above = ratio >= 0
    exists = above.any(axis=1)
    first = above.argmax(axis=1)
    status = np.where(~exists, 0, np.where(first == 0, 2, 1))
    recorded = np.where(exists, z[first], np.nan)
    valid = status == 1
    lo = z[first[valid]].copy()
    hi = z[first[valid] - 1].copy()
    for _ in range(32):
        mid = (lo + hi) / 2
        r = (
            np.log(mass)
            + beta[valid] * np.log((1 + mid) / (1 + z_final))
            - gamma[valid] * (mid - z_final)
            - np.log(compute_atomic_cooling_mass_msun(mid, cosmology=cosmology))
        )
        lo = np.where(r >= 0, mid, lo)
        hi = np.where(r < 0, mid, hi)
    recorded[valid] = (lo + hi) / 2
    return recorded, status


def initialize(zfinal):
    global STATE
    matched = Cosmology(h0=PLANCK15_H0_GYR, omega_m=0.3089, omega_b=0.0486, omega_lambda=0.6911)
    STATE = zfinal, matched, Cosmology()


def one_target(task):
    index, mass = task
    zfinal, matched, native = STATE
    seed = int(np.random.SeedSequence([SEED, index]).generate_state(1, dtype=np.uint64)[0])
    params, _ = sample_parameters(
        mass_ref=mass,
        size=NTRACK,
        sampler="mcbride",
        rng=np.random.default_rng(seed),
        pilot_samples=50000,
    )
    beta, gamma = params.T
    z, status = solve_crossings(beta, gamma, mass, zfinal, matched)
    zn, sn = solve_crossings(beta, gamma, mass, zfinal, native)
    if np.any(status == 0):
        raise ValueError("A matched target above Mcool has no crossing")
    return z, status, zn, sn, beta, gamma


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Full matched-sample analysis requires SLURM")
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite completed matched comparison")
    meta = json.loads((INPUT / "thesan_manifest.json").read_text())
    assert (
        hashlib.sha256((INPUT / "thesan_halos.csv").read_bytes()).hexdigest() == meta["csv_sha256"]
    )
    rows = list(csv.DictReader((INPUT / "thesan_halos.csv").open()))
    mass = np.array([float(r["mvir_msun"]) for r in rows])
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    arrays = [np.empty((len(rows), NTRACK)) for _ in range(6)]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize, initargs=(meta["redshift"],)
    ) as pool:
        for i, result in enumerate(pool.map(one_target, enumerate(mass), chunksize=16)):
            for a, r in zip(arrays, result, strict=True):
                a[i] = r
            if i % 1000 == 0:
                print(f"endpoints {i + 1}/{len(rows)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["crossing_z", "status", "native_crossing_z", "native_status", "beta", "gamma"]
    np.savez_compressed(
        OUT / "histories.npz",
        mass_msun=mass,
        subhalo_id=np.array([int(r["subhalo_id"]) for r in rows]),
        **dict(zip(keys, arrays, strict=True)),
    )
    manifest = dict(
        status="complete",
        job_id=os.environ["SLURM_JOB_ID"],
        n_endpoints=len(rows),
        histories_per_endpoint=NTRACK,
        seed=SEED,
        z_final=meta["redshift"],
        z_start=50.0,
        primary_cosmology="THESAN Planck15: h=.6774, Om=.3089, Ob=.0486",
        sensitivity_cosmology="Production Cosmology() Planck18, same beta/gamma draws",
        endpoint_weight="Equal target weight; exact individual Mvir matched, no HMF reweighting",
        definition="Same production sample_parameters(mass_ref=Mfinal,sampler=mcbride); exact mass formula; q=1 atomic threshold only",
        input_manifest_sha256=hashlib.sha256(
            (INPUT / "thesan_manifest.json").read_bytes()
        ).hexdigest(),
        input_csv_sha256=meta["csv_sha256"],
        product_sha256=hashlib.sha256((OUT / "histories.npz").read_bytes()).hexdigest(),
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [
                "auroralf/mah/sampling.py",
                "auroralf/mah/physics.py",
                "auroralf/mah/generator.py",
                "scripts/analysis/compare_atomic_crossing_mcbride.py",
            ]
        },
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
