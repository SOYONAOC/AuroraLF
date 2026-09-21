"""Increase low-z MAH and mass sampling, then export the existing efficiency deck."""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np  # noqa: E402

from auroralf.cooling import compute_atomic_cooling_mass_msun  # noqa: E402
from auroralf.experiments import random_q  # noqa: E402
from auroralf.experiments.uvlf_components import age_strata, conditional_histograms  # noqa: E402
from auroralf.mah import Cosmology, generate_halo_histories  # noqa: E402
from auroralf.seeding import derive_hmf_mass_seed, derive_pipeline_random_seeds  # noqa: E402
from auroralf.sfr import compute_sfr_from_tracks  # noqa: E402
from auroralf.ssp import compute_final_ssp_observable_from_sfr_grid  # noqa: E402
from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator  # noqa: E402
from scripts.analysis.build_uvlf_efficiency_scan import ROOT, digest, write_json  # noqa: E402


def one_mass(task):
    index, final_mass, first_track = task
    cfg, cosmo, dt, a2, k2, a3, k3 = random_q.STATE
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0x53545241, index, int(cfg.z)]))
    uniforms = np.clip(
        rng.random((cfg.n_tracks, 4)), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
    )
    p2, p3, probabilities = [], [], []
    if first_track < 0 or first_track >= cfg.n_tracks or first_track % cfg.track_chunk:
        raise ValueError("Invalid first new history index")
    for start in range(first_track, cfg.n_tracks, cfg.track_chunk):
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
        tracks = compute_sfr_from_tracks(
            h.tracks,
            cosmology=cosmo,
            enable_time_delay=True,
            regular_convolution_backend=cfg.sfr_convolution,
        )
        shape = (cfg.track_chunk, -1)
        t, mass, z = [tracks[k].reshape(shape) for k in ("t_gyr", "Mh", "z")]
        cooling = compute_atomic_cooling_mass_msun(z, cosmology=cosmo)
        p2.append(
            compute_final_ssp_observable_from_sfr_grid(
                t_grid_gyr=t,
                sfr_grid=tracks["SFR"].reshape(shape),
                active_grid=tracks["active_flag"].reshape(shape),
                ssp_age_myr=a2,
                ssp_observable_per_msun=k2,
                lookback_max_myr=cfg.lookback_myr,
            )
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
            cosmo.omega_b / cosmo.omega_m,
            cfg.efficiencies[0],
        )
        p3.append(light)
        probabilities.append(prob)
    return index, np.concatenate(p2), np.concatenate(p3), np.concatenate(probabilities)


def estimate(p2, p3, probability, weights, edges):
    contributions = conditional_histograms(p2, p3, probability, weights, edges)
    return contributions.sum(axis=1), np.sqrt(len(p2) * contributions.var(axis=1, ddof=1))


def new_mass_tasks(masses, parent_shape, n_tracks):
    """Extend absolute history indices without recomputing any retained sample."""
    pm, pt = parent_shape
    if len(masses) < pm or n_tracks < pt or pt % 64 or n_tracks % 64:
        raise ValueError("Expanded samples must contain the full parent sample")
    if len(masses) == pm and n_tracks == pt:
        raise ValueError("No new samples requested")
    return [
        (i, mass, pt if i < pm else 0) for i, mass in enumerate(masses) if i >= pm or n_tracks > pt
    ]


def diagnostics(p2, p3, probability, weights, edges, old, path):
    """Nested-size comparison plus independent half-sample differences, per efficiency."""
    nm, nt = p2.shape
    pm, pt = old["popii"].shape
    saved = {
        "bin_edges": edges,
        "original_size_shape": np.array([pm, pt]),
        "half_size_shape": np.array([nm // 2, nt]),
        "full_shape": np.array([nm, nt]),
    }
    for eps in (0.01, 0.03, 0.1):
        for tag, m, n in (
            ("original_size", pm, pt),
            ("half_size", nm // 2, nt),
            ("full", nm, nt),
        ):
            phi, se = estimate(
                p2[:m, :n],
                p3[:m, :n] * (eps / 0.03),
                probability[:m, :n],
                weights[:m] * nm * nt / (m * n),
                edges,
            )
            saved[f"eps{eps:g}_{tag}_phi"] = phi
            saved[f"eps{eps:g}_{tag}_se"] = se
        halves = []
        for selection in (slice(0, nm // 2), slice(nm // 2, nm)):
            halves.append(
                estimate(
                    p2[selection],
                    p3[selection] * (eps / 0.03),
                    probability[selection],
                    weights[selection] * 2,
                    edges,
                )
            )
        delta = halves[0][0] - halves[1][0]
        error = np.hypot(halves[0][1], halves[1][1])
        saved[f"eps{eps:g}_half_difference_sigma"] = np.divide(
            delta, error, out=np.full_like(delta, np.nan), where=error > 0
        )
    np.testing.assert_allclose(
        saved["eps0.03_original_size_phi"], old["phi"], rtol=1e-10, atol=1e-100
    )
    np.testing.assert_allclose(
        saved["eps0.03_original_size_se"], old["se"], rtol=1e-10, atol=1e-100
    )
    np.savez_compressed(path, **saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan["input_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    parent = Path(plan["parent"])
    before = json.loads((parent / "manifest.json").read_text())
    if before["status"] != "complete":
        raise ValueError("Incomplete parent run")
    nm, nt = plan["n_mass"], plan["n_tracks"]
    if nm < 1440 or nm % 2 or nt < 512 or nt % 64:
        raise ValueError("Invalid expanded sample dimensions")
    for z in (6, 8):
        if digest(parent / f"z{z}.npz") != before["products"][f"z{z}.npz"]:
            raise ValueError(f"Corrupt parent z={z}")
        cfg = next(c for c in before["configs"] if c["z"] == z)
        if (
            cfg["n_mass"] > nm
            or cfg["n_tracks"] > nt
            or cfg["n_tracks"] % cfg["track_chunk"]
            or cfg["efficiencies"] != [0.03]
            or cfg["lookback_myr"] != 100
            or cfg["n_mass"] * cfg["n_tracks"] >= nm * nt
        ):
            raise ValueError("Unexpected parent configuration")
    if args.validate_only:
        print(f"Validated {len(plan['input_sha256'])} frozen inputs; no scientific computation")
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM allocation required")
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "job": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"),
        "started_unix": time.time(),
        "plan": str(args.plan.resolve()),
        "input_sha256": plan["input_sha256"],
        "redshifts": [6, 8],
        "configs": [],
        "products": {},
        "parent": str(parent),
        "sampling": "Extend independent uniform log-mass draws and MAHs, retaining the exact full parent sample; same nested random streams and conditional burst-age strata",
    }
    write_json(out / "manifest.json", manifest)
    for z in (6, 8):
        config = dict(next(c for c in before["configs"] if c["z"] == z))
        config.update(n_mass=nm, n_tracks=nt, workers=workers)
        cfg = random_q.Config(**config)
        manifest["configs"].append(asdict(cfg))
        with np.load(parent / f"z{z}.npz") as data:
            old = dict(data)
        pm, pt = old["popii"].shape
        parent_config = next(c for c in before["configs"] if c["z"] == z)
        if (pm, pt) != (parent_config["n_mass"], parent_config["n_tracks"]):
            raise ValueError("Parent sample shape differs from its manifest")
        edges = old["edges"]
        rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, z))
        masses = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, nm)
        # exp10 can differ by a few ULPs across CPU vector-math implementations.
        # Verify the same draws, then reuse the exact archived masses for nesting.
        np.testing.assert_allclose(masses[:pm], old["mass_msun"], rtol=1e-14, atol=0)
        masses[:pm] = old["mass_msun"]
        hmf = prepare_reed07_hmf_interpolator(
            log10_halo_mass_min_msun=cfg.logmass_min,
            log10_halo_mass_max_msun=cfg.logmass_max,
            z_obs=z,
            cosmology=Cosmology(),
        )
        weights = (
            (cfg.logmass_max - cfg.logmass_min)
            * np.log(10)
            * masses
            * hmf.evaluate(masses)
            / nm
            / nt
        )
        # Original samples stay unchanged; only their normalization changes.
        parent_weights = old["weight_per_track"] * (pm * pt) / (nm * nt)
        # HMF transfer-function integration is not bitwise reproducible across
        # CPUs. Verify 8 significant relative digits, well below MC precision,
        # while preserving the archived weights exactly for the retained draws.
        np.testing.assert_allclose(weights[:pm], parent_weights, rtol=1e-8, atol=0)
        weights[:pm] = parent_weights
        arrays = {}
        for key, shape in (
            ("popii", (nm, nt)),
            ("popiii", (nm, nt, 5)),
            ("probability", (nm, nt, 5)),
        ):
            arrays[key] = np.lib.format.open_memmap(
                out / f"z{z}_{key}.npy", mode="w+", dtype="float64", shape=shape
            )
            arrays[key][:pm, :pt] = old[key]
        tasks = new_mass_tasks(masses, (pm, pt), nt)
        copied_masses = pm if nt == pt else 0
        print(
            f"z={z}: retained {pm} x {pt} histories; computing {nm * nt - pm * pt} new histories",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=random_q.initialize_worker,
            initargs=(cfg,),
        ) as pool:
            for first in range(0, len(tasks), 2 * workers):
                batch = tasks[first : first + 2 * workers]
                for i, p2, p3, prob in pool.map(one_mass, batch):
                    start = pt if i < pm else 0
                    for key, value in (("popii", p2), ("popiii", p3), ("probability", prob)):
                        if value.shape != arrays[key][i, start:].shape:
                            raise ValueError("Missing or duplicate new history samples")
                        arrays[key][i, start:] = value
                for value in arrays.values():
                    value.flush()
                print(
                    f"z={z} mass={copied_masses + first + len(batch)}/{nm}, tracks/mass={nt}",
                    flush=True,
                )
        for key in arrays:
            np.testing.assert_array_equal(arrays[key][:pm, :pt], old[key])
        p2, p3, prob = [arrays[k] for k in ("popii", "popiii", "probability")]
        phi, se = estimate(p2, p3, prob, weights, edges)
        product = out / f"z{z}.npz"
        np.savez_compressed(
            product,
            redshift=z,
            mass_msun=masses,
            weight_per_track=weights,
            popii=p2,
            popiii=p3,
            probability=prob,
            edges=edges,
            phi=phi,
            se=se,
        )
        convergence = out / f"z{z}_convergence.npz"
        diagnostics(p2, p3, prob, weights, edges, old, convergence)
        for path in (product, convergence):
            manifest["products"][path.name] = digest(path)
        write_json(out / "manifest.json", manifest)
        del p2, p3, prob, arrays
    for name, expected in plan["input_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Input changed during computation: {name}")
    manifest.update(status="complete", completed_unix=time.time())
    write_json(out / "manifest.json", manifest)
    export = plan["export_plan"]
    export["input_sha256"] = {
        **plan["input_sha256"],
        **{str(out / k): v for k, v in manifest["products"].items()},
        str(out / "manifest.json"): digest(out / "manifest.json"),
    }
    export_path = out / "export_plan.json"
    write_json(export_path, export)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analysis/build_uvlf_efficiency_scan.py"),
            "--plan",
            str(export_path),
        ],
        cwd=ROOT,
        check=True,
    )
    plot_convergence(out, Path(plan["convergence_output"]))
    print(f"Sampling and deck export complete: {export['deck_output']}", flush=True)


def plot_convergence(run, output):
    """Keep convergence evidence separate from presentation curves; never smooth."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    plt.style.use("apj")
    report = {
        "definition": "Intrinsic mass-cluster MC errors; does not test time resolution or model systematics",
        "bins": [],
    }
    for z in (6, 8):
        with np.load(run / f"z{z}_convergence.npz") as data:
            x = (data["bin_edges"][1:] + data["bin_edges"][:-1]) / 2
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
            for ax, (eps, component, title) in zip(
                axes.flat,
                [
                    (0.03, 0, "Pop II"),
                    (0.01, 2, "Pop II + III, epsilon=0.01"),
                    (0.03, 2, "Pop II + III, epsilon=0.03"),
                    (0.1, 2, "Pop II + III, epsilon=0.1"),
                ],
                strict=True,
            ):
                for tag, color in [
                    ("original_size", "#b86632"),
                    ("half_size", "#008060"),
                    ("full", "#286491"),
                ]:
                    label = " x ".join(str(n) for n in data[f"{tag}_shape"])
                    phi, se = [data[f"eps{eps:g}_{tag}_{k}"][component] for k in ("phi", "se")]
                    ax.plot(x, np.where(phi > 0, phi, np.nan), label=label, color=color)
                    ax.fill_between(
                        x,
                        np.where(phi > se, phi - se, np.nan),
                        np.where(phi > 0, phi + se, np.nan),
                        color=color,
                        alpha=0.15,
                    )
                    for i in np.flatnonzero((x >= -24) & (x <= -16) & (phi > 0)):
                        report["bins"].append(
                            {
                                "z": z,
                                "component": title,
                                "sampling": tag,
                                "Muv": float(x[i]),
                                "phi": float(phi[i]),
                                "relative_mc_se": float(se[i] / phi[i]),
                            }
                        )
                ax.set(
                    title=title,
                    yscale="log",
                    xlim=(-24, -16),
                    ylim=(1e-8, 0.05),
                    xlabel=r"$M_{\rm UV}$",
                    ylabel=r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$",
                )
                ax.legend(fontsize=9)
            fig.suptitle(f"z={z}: intrinsic UVLF sampling comparison (bands: MC SE)")
            fig.tight_layout()
            fig.savefig(output / f"z{z}_convergence.pdf")
            fig.savefig(output / f"z{z}_convergence.png", dpi=150)
            plt.close(fig)
    write_json(output / "mc_error_comparison.json", report)


if __name__ == "__main__":
    main()
