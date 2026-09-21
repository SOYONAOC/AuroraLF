"""Resolved-ancestor veto for Pop III candidates in existing TNG-Dark trees.

This is an efficient-cooling, full-mixing diagnostic, NOT a metallicity
measurement or the random-threshold burst model. Unvetoed trees remain unknown.
Full catalog processing requires SLURM; unit tests use small constructed trees.
"""

import argparse
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
SELECTION = (
    ROOT
    / "data_save/tng_mah_cache/selection/TNG100-1-Dark_logM9p00_13p00_dlogM0p25_n1000_seed42/selected_subhalos_manifest.csv"
)
OUTPUT = ROOT / "data_save/tng_pristine_20260914"
RAW = ROOT / "external_data/tng/TNG100-1-Dark/raw_sublink_full"
FIELDS = (
    "SubhaloID",
    "SubfindID",
    "SnapNum",
    "FirstProgenitorID",
    "NextProgenitorID",
    "DescendantID",
    "SubhaloMass",
    "SubhaloLen",
    "Group_M_Crit200",
)
DELAYS = (10.0, 30.0, 100.0)
PARTICLES = (20, 100)
STATE = None


def ancestry(tree, root):
    """Traverse all progenitors of root, excluding unrelated sibling trees."""
    ids = tree["SubhaloID"]
    lookup = {int(s): i for i, s in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError("Duplicate SubhaloIDs")
    first, sibling = tree["FirstProgenitorID"], tree["NextProgenitorID"]
    seen = {root}
    pending = [root]
    while pending:
        parent = pending.pop()
        child = int(first[parent])
        sibling_seen = set()
        while child != -1:
            if child not in lookup:
                raise ValueError(f"Missing progenitor {child}")
            if child in sibling_seen:
                raise ValueError("Cyclic sibling links")
            sibling_seen.add(child)
            i = lookup[child]
            if i in seen:
                raise ValueError("Repeated or cyclic progenitor")
            if tree["DescendantID"][i] != ids[parent]:
                raise ValueError("Inconsistent descendant link")
            if tree["SnapNum"][i] >= tree["SnapNum"][parent]:
                raise ValueError("Noncausal snapshot ordering")
            seen.add(i)
            pending.append(i)
            child = int(sibling[i])
    main = []
    current = int(first[root])
    while current != -1:
        i = lookup[current]
        main.append(i)
        current = int(first[i])
    return np.array(sorted(seen - {root}), dtype=int), np.array(main, dtype=int)


def allowed(age_since_eligible_myr, delay_myr):
    """Negative age means no resolved eligible ancestor, NOT proven pristine."""
    return np.asarray(age_since_eligible_myr) < delay_myr


def initialize(z_by_snap):
    global STATE
    cosmo = Cosmology(h0=PLANCK15_H0_GYR, omega_m=0.3089, omega_b=0.0486, omega_lambda=0.6911)
    astro = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)
    n = max(z_by_snap) + 1
    z = np.full(n, np.nan)
    for s, value in z_by_snap.items():
        z[s] = value
    if not np.isfinite(z).all():
        raise ValueError("Incomplete snapshot redshift mapping")
    STATE = (z, astro.age(z).to_value("Myr"), compute_atomic_cooling_mass_msun(z, cosmology=cosmo))


def process_target(target):
    snap, sid, loglo, loghi, population, selected = target
    z, time, cooling = STATE
    path = RAW / f"snap_{snap:03d}/subhalo_{sid}_sublink_full.hdf5"
    with h5py.File(path, "r") as f:
        tree = {name: f[name][:] for name in FIELDS}
    matches = np.flatnonzero((tree["SnapNum"] == snap) & (tree["SubfindID"] == sid))
    if len(matches) != 1:
        raise ValueError(f"Ambiguous target in {path}")
    root = int(matches[0])
    ancestors, main = ancestry(tree, root)
    snapshots = tree["SnapNum"]
    if np.any(snapshots < 0) or np.any(snapshots >= len(time)):
        raise ValueError(f"Unknown snapshot in {path}")
    mass = tree["SubhaloMass"].astype(float) * 1e10 / 0.6774
    particles = tree["SubhaloLen"]
    if not np.isfinite(mass).all() or np.any(mass <= 0) or np.any(particles <= 0):
        raise ValueError(f"Invalid mass/particle count in {path}")
    target_mass = float(tree["Group_M_Crit200"][root] * 1e10 / 0.6774)
    if not loglo - 1e-5 <= np.log10(target_mass) <= loghi + 1e-5:
        raise ValueError(f"Target mass outside selection stratum: {path}")
    qualified = mass >= cooling[snapshots]
    row = dict(
        snapshot=snap,
        z=float(z[snap]),
        subhalo_id=sid,
        logmass_low=loglo,
        logmass_high=loghi,
        population=population,
        selected=selected,
        weight=population / selected,
        target_m200_msun=target_mass,
        target_subhalo_mass_msun=float(mass[root]),
        target_particles=int(particles[root]),
        target_cooling=bool(qualified[root]),
        ancestor_count=len(ancestors),
        main_count=len(main),
    )
    for nmin in PARTICLES:
        for label, indices in (("all", ancestors), ("main", main)):
            eligible = indices[qualified[indices] & (particles[indices] >= nmin)]
            earliest = float(time[snapshots[eligible]].min()) if len(eligible) else None
            row[f"lookback_{label}_n{nmin}_myr"] = (
                float(time[snap] - earliest) if earliest is not None else -1.0
            )
        leaves = ancestors[tree["FirstProgenitorID"][ancestors] == -1]
        row[f"censored_leaf_n{nmin}"] = bool(
            np.any(qualified[leaves] & (particles[leaves] >= nmin))
        )
    hash_input = hashlib.sha256()
    for name in FIELDS:
        array = np.ascontiguousarray(tree[name])
        hash_input.update(name.encode())
        hash_input.update(str(array.dtype).encode())
        hash_input.update(str(array.shape).encode())
        hash_input.update(array.tobytes())
    row["input_fields_sha256"] = hash_input.hexdigest()
    row["input_path"] = str(path.relative_to(ROOT))
    return row


def weighted_summary(rows, column, delay):
    """Restore parent catalog counts; compute stratified ratio-estimator MC SE."""
    weights = np.array([r["weight"] for r in rows])
    eligible = np.array([r["target_cooling"] for r in rows])
    survives = eligible & allowed([r[column] for r in rows], delay)
    denom = float(weights @ eligible)
    if denom <= 0:
        raise ValueError("No cooling-eligible targets")
    fraction = float(weights @ survives / denom)
    variance = 0.0
    for lo in sorted({r["logmass_low"] for r in rows}):
        mask = np.array([r["logmass_low"] == lo for r in rows])
        subset = [r for r, flag in zip(rows, mask, strict=True) if flag]
        population, selected = subset[0]["population"], subset[0]["selected"]
        if len(subset) != selected:
            raise ValueError("Incomplete selection stratum")
        if selected < population:
            if selected < 2:
                raise ValueError("Cannot estimate stratum sampling uncertainty")
            residual = survives[mask].astype(float) - fraction * eligible[mask]
            variance += (
                population**2 * (1 - selected / population) * np.var(residual, ddof=1) / selected
            )
    return dict(
        fraction=fraction,
        mc_se=float(np.sqrt(variance) / denom),
        sampled_total=len(rows),
        sampled_eligible=int(eligible.sum()),
        sampled_unvetoed=int(survives.sum()),
        catalog_eligible_estimate=denom,
    )


def make_summary(rows):
    records = []
    ranges = ((9.0, 13.0), (9.0, 10.0), (10.0, 11.0), (11.0, 13.0))
    for snap in sorted({r["snapshot"] for r in rows}):
        for lo, hi in ranges:
            subset = [
                r
                for r in rows
                if r["snapshot"] == snap and r["logmass_low"] >= lo and r["logmass_high"] <= hi
            ]
            if not subset:
                continue
            for nmin in PARTICLES:
                for branch in ("all", "main"):
                    for delay in DELAYS:
                        item = weighted_summary(subset, f"lookback_{branch}_n{nmin}_myr", delay)
                        records.append(
                            dict(
                                snapshot=snap,
                                z=subset[0]["z"],
                                logmass_low=lo,
                                logmass_high=hi,
                                particles=nmin,
                                branch=branch,
                                delay_myr=delay,
                                **item,
                            )
                        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full tree analysis requires a SLURM compute allocation")
    if not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        raise ValueError("Worker count exceeds allocation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Preserve the existing run; choose a new experiment directory")
    targets = []
    id_hashes = {}
    for r in csv.DictReader(SELECTION.open()):
        selected = int(r["selected_count"])
        if selected == 0:
            if int(r["available_count"]) != 0:
                raise ValueError("Unsampled nonempty stratum")
            continue
        p = Path(r["id_file"])
        ids = np.loadtxt(p, dtype=np.int64, ndmin=1)
        if len(ids) != selected or len(np.unique(ids)) != selected:
            raise ValueError(f"Invalid selection IDs: {p}")
        id_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        targets.extend(
            (
                int(r["snapshot"]),
                int(sid),
                float(r["logM_low"]),
                float(r["logM_high"]),
                int(r["available_count"]),
                selected,
            )
            for sid in ids
        )
    if len({(t[0], t[1]) for t in targets}) != len(targets):
        raise ValueError("Repeated target within a snapshot")
    cache = ROOT / "data_save/tng_mah_cache/TNG100-1-Dark_sublink_mpb_z6p011_n8956.hdf5"
    with h5py.File(cache, "r") as f:
        z_by_snap = dict(zip(f["snap_grid"][:].tolist(), f["z_grid"][:].tolist(), strict=True))
    initialize(z_by_snap)
    z, time, cooling = STATE
    manifest = dict(
        status="running",
        job_id=os.environ["SLURM_JOB_ID"],
        node=os.environ.get("SLURMD_NODENAME"),
        workers=args.workers,
        target_count=len(targets),
        selection=str(SELECTION.relative_to(ROOT)),
        selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        id_hashes=id_hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshot_z=z.tolist(),
        snapshot_time_myr=time.tolist(),
        atomic_cooling_mass_msun=np.asarray(cooling).tolist(),
        particle_thresholds=PARTICLES,
        delays_myr=DELAYS,
        model="Every resolved ancestor above the atomic-cooling mass starts a clock; after the total delay it pollutes all descendant star-forming gas. Bound SubhaloMass is used as the ancestor mass proxy. No H2 cooling, external pollution, metal yields or random-threshold burst formation is modeled.",
        interpretation="Fraction not vetoed by resolved ancestors; optimistic envelope ONLY under this full-mixing efficient-cooling prescription. Survivors are unknown, not confirmed pristine; target snapshots are not burst-rate events.",
        denominator="Cooling-eligible selected central halos with M200c in the stated range; inverse inclusion weights restore TNG group catalog counts.",
        uncertainty="MC SE is finite-population stratified sampling only, zero at sampled boundaries; no cosmic variance or physical systematic uncertainty.",
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(z_by_snap,)
    ) as pool:
        for i, row in enumerate(pool.map(process_target, targets, chunksize=16), 1):
            rows.append(row)
            if i % 1000 == 0:
                print(f"processed {i}/{len(targets)}", flush=True)
    path = OUTPUT / "halo_diagnostics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(make_summary(rows), indent=2) + "\n")
    manifest.update(
        status="complete",
        processed=len(rows),
        diagnostics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
