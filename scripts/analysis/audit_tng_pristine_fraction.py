"""Check resolved-ancestor results using independent forward descendant links."""

import csv
import hashlib
import json

import h5py
import numpy as np

from scripts.analysis.tng_pristine_fraction import OUTPUT, ROOT


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    table = OUTPUT / "halo_diagnostics.csv"
    if hashlib.sha256(table.read_bytes()).hexdigest() != manifest["diagnostics_sha256"]:
        raise ValueError("Diagnostics hash mismatch")
    rows = list(csv.DictReader(table.open()))
    if len(rows) != manifest["target_count"]:
        raise ValueError("Missing targets")
    for r in rows:
        for n in (20, 100):
            if float(r[f"lookback_all_n{n}_myr"]) < float(r[f"lookback_main_n{n}_myr"]):
                raise ValueError("Main branch older than full ancestry")
        for branch in ("all", "main"):
            if float(r[f"lookback_{branch}_n20_myr"]) < float(r[f"lookback_{branch}_n100_myr"]):
                raise ValueError("Particle threshold violates nesting")
    times = np.array(manifest["snapshot_time_myr"])
    cooling = np.array(manifest["atomic_cooling_mass_msun"])
    checked, particle_mass = [], []
    for snap in sorted({r["snapshot"] for r in rows}):
        subset = [r for r in rows if r["snapshot"] == snap]
        chosen = subset[:3] + [max(subset, key=lambda r: int(r["ancestor_count"]))]
        for r in chosen:
            with h5py.File(ROOT / r["input_path"], "r") as f:
                ids, descendants, snaps = f["SubhaloID"][:], f["DescendantID"][:], f["SnapNum"][:]
                subfind = f["SubfindID"][:]
                mass = f["SubhaloMass"][:].astype(float) * 1e10 / 0.6774
                number = f["SubhaloLen"][:]
            lookup = {int(s): i for i, s in enumerate(ids)}
            root = int(np.flatnonzero((snaps == int(snap)) & (subfind == int(r["subhalo_id"])))[0])
            ancestor = []
            for i in range(len(ids)):
                if i == root:
                    continue
                descendant = int(descendants[i])
                visited = set()
                while descendant != -1 and descendant in lookup:
                    if descendant in visited:
                        raise ValueError("Cyclic descendants")
                    visited.add(descendant)
                    if descendant == ids[root]:
                        ancestor.append(i)
                        break
                    descendant = int(descendants[lookup[descendant]])
            if len(ancestor) != int(r["ancestor_count"]):
                raise ValueError("Forward/backward ancestry mismatch")
            ancestor = np.array(ancestor, dtype=int)
            for nmin in (20, 100):
                qualifying = ancestor[
                    (mass[ancestor] >= cooling[snaps[ancestor]]) & (number[ancestor] >= nmin)
                ]
                lookback = (
                    float(times[int(snap)] - times[snaps[qualifying]].min())
                    if len(qualifying)
                    else -1
                )
                if not np.isclose(lookback, float(r[f"lookback_all_n{nmin}_myr"]), atol=1e-8):
                    raise ValueError("Independent clock mismatch")
            particle_mass.append(float(np.median(mass / number)))
            checked.append(r["input_path"])
    summary = json.loads((OUTPUT / "summary.json").read_text())
    for s in summary:
        selected = [
            r
            for r in rows
            if int(r["snapshot"]) == s["snapshot"]
            and float(r["logmass_low"]) >= s["logmass_low"]
            and float(r["logmass_high"]) <= s["logmass_high"]
        ]
        denominator = numerator = 0.0
        for r in selected:
            if r["target_cooling"] == "True":
                weight = int(r["population"]) / int(r["selected"])
                denominator += weight
                key = f"lookback_{s['branch']}_n{s['particles']}_myr"
                if float(r[key]) < s["delay_myr"]:
                    numerator += weight
        if not np.isclose(numerator / denominator, s["fraction"], atol=1e-12):
            raise ValueError("Independent abundance weighting mismatch")
    result = dict(
        status="passed",
        target_count=len(rows),
        checked_raw_trees=checked,
        independent_method="Follow DescendantID forward, rather than First/NextProgenitorID backward",
        summary_records_checked=len(summary),
        dm_particle_mass_msun=float(np.median(particle_mass)),
        checks=[
            "diagnostics hash",
            "all/main and particle-cut nesting",
            "raw ancestor membership and clocks",
            "all weighted summary fractions",
        ],
        visual_review="pending",
    )
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_raw_trees"}, indent=2))


if __name__ == "__main__":
    main()
