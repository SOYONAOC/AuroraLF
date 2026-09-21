"""Measure main-branch atomic-cooling crossings in the real THESAN-HR-large trees.

Full-file processing requires SLURM. No extrapolation before the first tree node.
"""

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.constants import PLANCK15_H0_GYR
from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.mah import Cosmology

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "external_data/thesan/thesan-hr-large/postprocessing/trees/LHaloTree/trees_sf1_091.0.hdf5"
)
OUT = ROOT / "data_save/atomic_crossing_z6_20260917"
H = 0.6774


def crossing(snap, mass, ndm, z, time, cooling):
    """Chronological branch; record the first above node and actual bracket."""
    q = mass / cooling[snap]
    hit = np.flatnonzero(q >= 1)
    if not len(hit):
        return dict(
            status="never",
            z_recorded=None,
            z_upper=None,
            z_interp=None,
            ndm_cross=None,
            dt_bracket_myr=None,
        )
    j = int(hit[0])
    row = dict(
        status="censored" if j == 0 else "bracketed",
        z_recorded=float(z[snap[j]]),
        z_upper=None,
        z_interp=None,
        ndm_cross=int(ndm[j]),
        dt_bracket_myr=None,
    )
    if j:
        fraction = -np.log(q[j - 1]) / (np.log(q[j]) - np.log(q[j - 1]))
        tc = time[snap[j - 1]] + fraction * (time[snap[j]] - time[snap[j - 1]])
        row.update(
            z_upper=float(z[snap[j - 1]]),
            z_interp=float(np.interp(tc, time, z)),
            dt_bracket_myr=float(time[snap[j]] - time[snap[j - 1]]),
        )
    return row


def process_chunk(args):
    start, stop, target, z, time, cooling = args
    rows = []
    count = 0
    with h5py.File(SOURCE, "r") as f:
        for k in range(start, stop):
            g = f[f"Tree{k}"]
            snap = g["SnapNum"][:]
            targets = np.flatnonzero(snap == target)
            count += len(targets)
            if not len(targets):
                continue
            central = g["FirstHaloInFOFGroup"][:]
            targets = targets[central[targets] == targets]
            if not len(targets):
                continue
            mvir = g["Group_M_TopHat200"][:].astype(float) * 1e10 / H
            m200 = g["Group_M_Crit200"][:].astype(float) * 1e10 / H
            targets = targets[mvir[targets] >= cooling[target]]
            if not len(targets):
                continue
            first = g["FirstProgenitor"][:]
            desc = g["Descendant"][:]
            ids = g["SubhaloNumber"][:]
            number = g["SubhaloLen"][:]
            for root in targets:
                branch = [int(root)]
                while first[branch[-1]] != -1:
                    parent = branch[-1]
                    i = int(first[parent])
                    if not 0 <= i < len(snap) or snap[i] >= snap[parent] or desc[i] != parent:
                        raise ValueError(f"Invalid main branch Tree{k}, node {i}")
                    branch.append(i)
                branch = np.array(branch[::-1])
                valid = (central[branch] == branch) & np.isfinite(mvir[branch]) & (mvir[branch] > 0)
                row = dict(
                    tree=k,
                    node=int(root),
                    subhalo_id=int(ids[root]),
                    z_target=float(z[target]),
                    mvir_msun=float(mvir[root]),
                    m200c_msun=float(m200[root]),
                    branch_nodes=len(branch),
                    satellite_branch_nodes=int(np.sum(central[branch] != branch)),
                    invalid_mass_nodes=int(
                        np.sum(~np.isfinite(mvir[branch]) | (mvir[branch] <= 0))
                    ),
                )
                # Use native spherical-overdensity masses only at central nodes.
                # Invalid/ satellite nodes are missing observations, never zeros or substitutes.
                good = branch[valid]
                result = crossing(snap[good], mvir[good], number[good], z, time, cooling)
                result["particles_total_cross"] = result.pop("ndm_cross")
                if result["status"] != "never":
                    hit = np.flatnonzero(valid & (mvir[branch] >= cooling[snap[branch]]))[0]
                    if np.any(~valid[: hit + 1]):
                        result.update(
                            status="incomplete_early_history",
                            z_upper=None,
                            z_interp=None,
                            dt_bracket_myr=None,
                        )
                row.update(result)
                rows.append(row)
    return rows, count


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Full tree analysis requires a SLURM allocation")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "thesan_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite completed analysis")
    cosmo = Cosmology(h0=PLANCK15_H0_GYR, omega_m=0.3089, omega_b=0.0486, omega_lambda=0.6911)
    astro = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)
    with h5py.File(SOURCE) as f:
        z = f["Header/Redshifts"][:].astype(float)
        ntree = int(f["Header"].attrs["NtreesPerFile"])
        assert int(f["Header"].attrs["NumberOfOutputFiles"]) == 1
        target = int(np.argmin(abs(z - 6)))
        expected = int(f["Header/TotNsubhalos"][target])
        mdm = float(f["Header"].attrs["ParticleMass"]) * 1e10 / H
    time = astro.age(z).to_value("Myr")
    assert np.all(np.diff(time) > 0)
    cooling = compute_atomic_cooling_mass_msun(z, cosmology=cosmo)
    chunks = [(i, min(i + 1000, ntree), target, z, time, cooling) for i in range(0, ntree, 1000)]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    rows, count = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j, (batch, n) in enumerate(pool.map(process_chunk, chunks)):
            rows.extend(batch)
            count += n
            if j % 10 == 0:
                print(
                    f"chunks {j + 1}/{len(chunks)}; cooling central targets {len(rows)}", flush=True
                )
    if count > expected:
        raise ValueError(f"Tree nodes exceed catalog count: {count} > {expected}")
    if count < expected:
        print(
            f"INCOMPLETE CATALOG COVERAGE: {count}/{expected}; output describes tree-covered centrals only",
            flush=True,
        )
    if len({r["subhalo_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate targets")
    with (OUT / "thesan_halos.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_bytes=SOURCE.stat().st_size,
        source_mtime_ns=SOURCE.stat().st_mtime_ns,
        trees=ntree,
        snapshot=target,
        redshift=float(z[target]),
        snapshot_subhalos=count,
        catalog_subhalos=expected,
        tree_catalog_coverage=count / expected,
        completeness="Tree-covered central sample only. Missing catalog subhalos have unknown masses; no completeness correction is applied.",
        selected_centrals=len(rows),
        dm_particle_mass_msun=mdm,
        cooling_mass_msun=float(cooling[target]),
        mu=0.61,
        temperature_k=1e4,
        h=H,
        omega_m=0.3089,
        omega_b=0.0486,
        z_grid=z.tolist(),
        mass_definition="Target and main central progenitor Mvir=Group_M_TopHat200. Satellite nodes and invalid masses are missing observations; any such node preceding first known crossing prevents a bracketed-first-crossing claim. SubhaloMassType and SubhaloLenType are unusable: Tree0 node21 has scalar SubhaloLen=134621 but sum(SubhaloLenType)=249.",
        crossing="Earliest chronological above-threshold node. Bracket interpolation in log(M/Mcool) versus cosmic time; censored first nodes are not extrapolated.",
        job_id=os.environ["SLURM_JOB_ID"],
        workers=workers,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256((OUT / "thesan_halos.csv").read_bytes()).hexdigest(),
    )
    (OUT / "thesan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "z_grid"}, indent=2))


if __name__ == "__main__":
    main()
