"""Apply the production Reed07 weights to new-stellar-mass rates."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT.parent / "SmallScale21cm"
RUN = ROOT / "data_save/ionizing_sources/sfrd_v1"


def main():
    global RUN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-only", action="store_true")
    p.add_argument("--run", type=Path, default=RUN)
    args = p.parse_args()
    RUN = args.run.resolve()
    if args.weights_only:
        sys.path.insert(0, str(SMALL))
        from eorcalc import MassFunctionConfig, PowerSpectrumConfig
        from eorcalc.powerspec import create_mass_functions
        from smallscale21cm.sources import table_hmf_weights

        manifest = json.loads(
            (SMALL / "runs/aurora_maps/instantaneous_100myr_v1/manifest.json").read_text()
        )
        with np.load(RUN / "cells.npz") as d:
            zs, mass = d["redshifts"], d["mass_msun"]
        cosmo = create_mass_functions(
            PowerSpectrumConfig(**manifest["power"]),
            MassFunctionConfig(hmf_model="Reed07"),
            cache_dir=RUN / "hmf_cache",
        )
        weights = np.array([table_hmf_weights(mass, float(z), cosmo) for z in zs])
        np.savez(RUN / "weights.npz", redshifts=zs, mass_msun=mass, weights_mpc3=weights)
        return
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (
        hashlib.sha256((RUN / "cells.npz").read_bytes()).hexdigest() == manifest["product_sha256"]
    )
    subprocess.run(
        [
            str(SMALL / "packages/EoRCaLC/.venv/bin/python"),
            str(Path(__file__).resolve()),
            "--weights-only",
            "--run",
            str(RUN),
        ],
        check=True,
    )
    with np.load(RUN / "cells.npz") as d:
        zs, mass, windows, means, cov, error = [
            d[k]
            for k in [
                "redshifts",
                "mass_msun",
                "windows_myr",
                "mean_sfr",
                "covariance_of_mean",
                "popiii_quadrature_error",
            ]
        ]
    with np.load(RUN / "weights.npz") as d:
        np.testing.assert_array_equal(zs, d["redshifts"])
        np.testing.assert_array_equal(mass, d["mass_msun"])
        w = d["weights_mpc3"]
    total = np.einsum("zm,zmwp->zwp", w, means)
    variance = np.einsum("zm,zmij->zij", w * w, cov)
    se = np.sqrt(np.diagonal(variance, axis1=1, axis2=2)).reshape(total.shape)
    rows = []
    for iz, z in enumerate(zs):
        for iw, window in enumerate(windows):
            n2, n3 = map(float, total[iz, iw])
            rows.append(
                dict(
                    z=float(z),
                    window_myr=float(window),
                    popii=n2,
                    popiii=n3,
                    total=n2 + n3,
                    popiii_fraction=n3 / (n2 + n3),
                    popii_se=float(se[iz, iw, 0]),
                    popiii_se=float(se[iz, iw, 1]),
                )
            )
    with (RUN / "sfrd.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    relative_to10 = total / total[:, 2:3, :] - 1
    result = dict(
        status="complete",
        units="Msun/yr/cMpc^3",
        job_id=manifest["job_id"],
        reference_window_myr=10,
        rows=[r for r in rows if r["window_myr"] == 10],
        mass_range_msun=[float(mass[0]), float(mass[-1])],
        max_window_relative_to10={
            str(window): np.max(abs(relative_to10[:, i, :]), axis=0).tolist()
            for i, window in enumerate(windows)
        },
        low_z_window_relative_to10={
            str(window): relative_to10[0, i].tolist() for i, window in enumerate(windows)
        },
        max_popiii_quadrature_relative=float(np.max(np.sum(w * error, axis=1) / total[:, 2, 1])),
        max_relative_sampling_se=np.max(se[:, 2] / total[:, 2], axis=0).tolist(),
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [RUN / "cells.npz", RUN / "weights.npz"]
        },
    )
    (RUN / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
