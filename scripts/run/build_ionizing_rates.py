"""Build instantaneous escaped photon-rate tables in a SLURM allocation."""
# ruff: noqa: E402 -- pin numerical threads before importing NumPy/SciPy.

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from auroralf.experiments.ionizing_rates import RateConfig, rate_cell
from auroralf.experiments.ionizing_sources import (
    initialize_worker,
)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    raw = tomllib.loads(args.config.read_text())
    model = raw["model"].copy()
    for key in ("popii_ssp", "popiii_ssp"):
        model[key] = str((ROOT / model[key]).resolve(strict=True))
    cfg = RateConfig(**model)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    grid = raw["grid"]
    z = np.array(grid["redshifts"], dtype=float)
    mass = np.logspace(grid["logmass_min"], grid["logmass_max"], grid["n_mass"])
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("redshifts must increase strictly")
    if z[0] < 0 or z[-1] >= cfg.z_start or len(mass) < 2 or np.any(np.diff(mass) <= 0):
        raise ValueError("invalid mass/redshift support")
    if not args.merge_shards:
        initialize_worker(cfg)
    if args.validate_only:
        print("Source config and SSP inputs validated", flush=True)
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("SLURM non-debug allocation required")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    out = (ROOT / raw["output"]).resolve()
    if args.merge_shards:
        parts = [out / "shards" / str(i) for i in range(args.shard_count)]
        manifests = [json.loads((part / "manifest.json").read_text()) for part in parts]
        reference = manifests[0]
        results, covered = {}, np.zeros((len(z), len(mass)), dtype=bool)
        for part, manifest in zip(parts, manifests, strict=True):
            if (
                manifest["status"] != "complete"
                or digest(part / "sources.npz") != manifest["product_sha256"]
            ):
                raise ValueError("incomplete or corrupt shard")
            for key in ["config", "resolved_model", "source_sha256", "input_sha256", "channels"]:
                if manifest[key] != reference[key]:
                    raise ValueError(f"inconsistent shard {key}")
            with np.load(part / "sources.npz") as data:
                selected = data["computed"]
                if np.any(covered & selected):
                    raise ValueError("duplicate source cells")
                covered |= selected
                for key in data.files:
                    if key in ["computed", "redshifts", "mass_msun"]:
                        continue
                    if key not in results:
                        results[key] = np.empty_like(data[key])
                    results[key][selected] = data[key][selected]
        if not covered.all():
            raise ValueError("source cells missing")
        product = out / "sources.npz"
        if product.exists():
            raise FileExistsError(product)
        np.savez_compressed(product, redshifts=z, mass_msun=mass, **results)
        reference.update(
            status="complete",
            source_format="instantaneous-rates-v1",
            completed_unix=time.time(),
            product_sha256=digest(product),
            shard_jobs=[v["job_id"] for v in manifests],
            merge_job=os.environ["SLURM_JOB_ID"],
        )
        reference.pop("shard_index", None)
        (out / "manifest.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
        print(product, flush=True)
        return
    if args.shard_count > 1:
        out = out / "shards" / str(args.shard_index)
    out.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "auroralf").rglob("*.py")) + [
        Path(__file__),
        args.config,
        ROOT / "uv.lock",
    ]
    manifest = dict(
        config=raw,
        resolved_model=asdict(cfg),
        job_id=os.environ["SLURM_JOB_ID"],
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        source_sha256={str(p.resolve()): digest(p) for p in sources},
        input_sha256={p: digest(p) for p in (cfg.popii_ssp, cfg.popiii_ssp)},
        source_format="instantaneous-rates-shard-v1"
        if args.shard_count > 1
        else "instantaneous-rates-v1",
        threshold_sampling="Normal(log10 q) integrated on record-crossing intervals; 16-point Gauss",
        channels=["popii", "popiii_resolved", "popiii_censored_upper_extra"],
        units="escaped photons/s per halo; float64, no HMF weight, no extra duty factor",
        temporal_closure="independent final-halo main branches; merged secondary branches absent",
        max_stellar_age_myr={"popii": cfg.max_lookback_myr, "popiii": cfg.popiii_max_age_myr},
        age_window="PopII birth-time integral and PopIII first-passage contribution use their separate max_stellar_age_myr limits; complete MAH and SFR history retained",
        left_censored="resolved contribution is a lower bound; upper extra uses q*Mcool(z_start) and maximum rate over allowed pre-start ages",
        popii_boundary="integrated only over supplied history; initially active PopII counts recorded",
        started_unix=time.time(),
        status="running",
    )
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results = {}
    tasks = [
        (iz, im, float(redshift), float(m))
        for iz, redshift in enumerate(z)
        for im, m in enumerate(mass)
    ]
    tasks = tasks[args.shard_index :: args.shard_count]
    computed = np.zeros((len(z), len(mass)), dtype=bool)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(cfg,),
    ) as pool:
        for first in range(0, len(tasks), workers * 2):
            for iz, im, values in pool.map(rate_cell, tasks[first : first + workers * 2]):
                for key, value in values.items():
                    value = np.asarray(value)
                    if key not in results:
                        results[key] = np.full(
                            (len(z), len(mass)) + value.shape, np.nan, dtype=value.dtype
                        )
                    results[key][iz, im] = value
                computed[iz, im] = True
                completed += 1
            print(
                f"{completed}/{len(tasks)} source cells; {time.time() - manifest['started_unix']:.1f} s",
                flush=True,
            )
    product = out / "sources.npz"
    np.savez_compressed(product, redshifts=z, mass_msun=mass, computed=computed, **results)
    manifest.update(status="complete", completed_unix=time.time(), product_sha256=digest(product))
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(product, flush=True)


if __name__ == "__main__":
    main()
