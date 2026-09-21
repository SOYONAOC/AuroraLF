"""Intrinsic 1500-A component UVLF for the current random-first-burst model."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402

from auroralf.experiments.random_q import Config, initialize_worker, one_mass  # noqa: E402
from auroralf.mah import Cosmology  # noqa: E402
from auroralf.seeding import derive_hmf_mass_seed  # noqa: E402
from auroralf.uvlf import uv_luminosity_to_muv  # noqa: E402
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def component_histograms(p2, p3, weights, edges):
    """Per-mass contributions keep paired light and clustered sampling errors."""
    if p2.shape != p3.shape or p2.ndim != 2 or weights.shape != (len(p2),):
        raise ValueError("Inconsistent UV sample dimensions")
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in (p2, p3, weights)):
        raise ValueError("Invalid luminosities or HMF weights")
    contributions, counts = [], []
    for lum in (p2, p3, p2 + p3):
        muv = uv_luminosity_to_muv(lum)
        hist = np.array([np.histogram(row, bins=edges)[0] for row in muv])
        contributions.append(hist * weights[:, None] / np.diff(edges))
        counts.append(hist.sum(0))
    return np.stack(contributions), np.stack(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "data_save/uvlf_current_z6_z8_z10")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-mass", type=int, default=720)
    p.add_argument("--n-tracks", type=int, default=512)
    a = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM allocation required")
    if (
        not 1 <= a.workers <= int(os.environ["SLURM_CPUS_PER_TASK"])
        or a.n_mass < 2
        or a.n_tracks < 2
        or a.n_tracks % 64
    ):
        raise ValueError("Invalid allocation or sampling")
    source = ROOT / "data_save/ionizing_sources/instantaneous_100myr_v1/manifest.json"
    original = json.loads(source.read_text())
    assert original["status"] == "complete"
    model = original["resolved_model"]
    cosmo = Cosmology()
    np.testing.assert_allclose(
        [cosmo.h0_km_s_mpc, cosmo.omega_m, cosmo.omega_b],
        [100 * model["h"], model["omega_m"], model["omega_b"]],
        rtol=1e-12,
    )
    ionizing_path = Path(model["popiii_ssp"])
    popiii_uv = ionizing_path.with_name(ionizing_path.name.removesuffix(".20") + ".25")
    # .20 contains ionizing diagnostics; .25 is the matching IMF's UV table.
    assert str(model["popiii_ssp"]).endswith("is5.20") and popiii_uv.is_file()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = [
        source,
        Path(model["popii_ssp"]),
        popiii_uv,
        Path(__file__),
        ROOT / "auroralf/experiments/random_q.py",
        ROOT / "uv.lock",
    ]
    files += (
        list((ROOT / "auroralf/mah").glob("*.py"))
        + list((ROOT / "auroralf/sfr").glob("*.py"))
        + list((ROOT / "auroralf/ssp").glob("*.py"))
    )
    hashes = {str(f.resolve()): digest(f) for f in files}
    manifest = dict(
        status="running",
        job=os.environ["SLURM_JOB_ID"],
        started_unix=time.time(),
        source_model=model,
        input_sha256=hashes,
        redshifts=[6.0, 8.0, 10.0],
        wavelength_a=1500.0,
        populations=["popii", "popiii", "total"],
        definition="Halo component luminosity functions; total histogram uses LII+LIII for each same halo, not a sum of LFs",
        source_prescription="Pop II stellar BPASS; Pop III matching .25 L_1500 includes the tabulated nebular continuum (Te=30000 K, absorbed LyC fraction=1); no UV dust correction or ionizing-fesc multiplier",
        scope="Original random-first-burst model, epsilon_b=0.03, no enrichment or redshift emission gate; UV age window 100 Myr for both populations, independent of the 6 Myr ionizing-rate cutoff",
        sampling="Independent uniform log halo mass; one independent random q per MAH; clustered Monte Carlo SE over independent mass draws, not observational Poisson error",
    )

    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    edges = np.arange(-28.0, 2.01, 0.5)
    products = {}
    for iz, z in enumerate(manifest["redshifts"]):
        cfg = Config(
            run_id=f"AUR-EX-0006-R{33 + iz:03d}",
            z=z,
            n_mass=a.n_mass,
            n_tracks=a.n_tracks,
            track_chunk=64,
            n_grid=model["n_grid"],
            workers=a.workers,
            seed=model["seed"],
            z_start=model["z_start"],
            logmass_min=4.0,
            logmass_max=15.0,
            q_log10_mean=model["q_log10_mean"],
            q_log10_sigma=model["q_log10_sigma"],
            efficiencies=[model["epsilon_b"]],
            popii_ssp=model["popii_ssp"],
            popiii_ssp=str(popiii_uv),
            lookback_myr=100.0,
            sfr_convolution="direct",
            popii_wavelength_a=1500.0,
        )
        rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, z))
        mass = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, cfg.n_mass)
        hmf = prepare_reed07_hmf_interpolator(
            log10_halo_mass_min_msun=4.0, log10_halo_mass_max_msun=15.0, z_obs=z, cosmology=cosmo
        )
        weights = 11 * np.log(10) * mass * hmf.evaluate(mass) / cfg.n_mass / cfg.n_tracks
        p2, p3 = np.empty((cfg.n_mass, cfg.n_tracks)), np.empty((cfg.n_mass, cfg.n_tracks))
        ages = np.empty_like(p2)
        status = np.empty(p2.shape, dtype=np.int8)
        with ProcessPoolExecutor(
            max_workers=a.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(cfg,),
        ) as pool:
            for first in range(0, cfg.n_mass, 2 * a.workers):
                tasks = [(i, mass[i]) for i in range(first, min(first + 2 * a.workers, cfg.n_mass))]
                for i, values in pool.map(one_mass, tasks):
                    p2[i] = values["popii"]
                    p3[i] = cfg.efficiencies[0] * values["popiii_per_efficiency"]
                    ages[i], status[i] = values["age_myr"], values["status"]
                print(f"z={z:g} mass={first + len(tasks)}/{cfg.n_mass}", flush=True)
        per_mass, counts = component_histograms(p2, p3, weights, edges)
        phi = per_mass.sum(axis=1)
        se = np.sqrt(cfg.n_mass * per_mass.var(axis=1, ddof=1))
        path = out / f"z{z:g}.npz"
        np.savez_compressed(
            path,
            redshift=z,
            mass_msun=mass,
            weight_per_track=weights,
            popii=p2,
            popiii=p3,
            popiii_age_myr=ages,
            status=status,
            edges=edges,
            phi=phi,
            se=se,
            counts=counts,
        )
        products[path.name] = digest(path)
        manifest.setdefault("configs", []).append(asdict(cfg))
        manifest["products"] = products
        save_manifest()
    assert all(digest(f) == h for f, h in hashes.items()), "Inputs changed during run"
    manifest.update(status="complete", completed_unix=time.time())
    save_manifest()


if __name__ == "__main__":
    main()
