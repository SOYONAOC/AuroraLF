"""Weight paired Pop III rates with the production map HMF and report errors."""

import argparse
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
sys.path.insert(0, str(ROOT))
SMALL = ROOT.parent / "SmallScale21cm"


def weights(a):
    sys.path.insert(0, str(SMALL))
    from eorcalc import MassFunctionConfig, PowerSpectrumConfig
    from eorcalc.powerspec import create_mass_functions
    from smallscale21cm.sources import table_hmf_weights

    manifest = json.loads((a.map_run / "manifest.json").read_text())
    with np.load(a.run / "cells.npz") as data:
        zs, mass = data["redshifts"], data["mass_msun"]
    cosmo = create_mass_functions(
        PowerSpectrumConfig(**manifest["power"]),
        MassFunctionConfig(hmf_model="Reed07"),
        cache_dir=a.run / "hmf_cache",
    )
    w = np.array([table_hmf_weights(mass, float(z), cosmo) for z in zs])
    np.savez(a.run / "weights.npz", weight_mpc3=w, redshifts=zs, mass_msun=mass)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=ROOT / "data_save/ionizing_sources/popiii_age_window_v1",
    )
    p.add_argument(
        "--map-run",
        type=Path,
        default=SMALL / "runs/aurora_maps/instantaneous_100myr_v1",
    )
    p.add_argument("--weights-only", action="store_true")
    a = p.parse_args()
    if a.weights_only:
        weights(a)
        return
    from scipy.optimize import brentq

    from auroralf.ssp.ionizing import load_popiii_ionizing_kernel

    manifest = json.loads((a.run / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or hashlib.sha256((a.run / "cells.npz").read_bytes()).hexdigest()
        != manifest["product_sha256"]
    ):
        raise ValueError("incomplete or changed rate comparison")
    subprocess.run(
        [
            str(SMALL / "packages/EoRCaLC/.venv/bin/python"),
            str(Path(__file__).resolve()),
            "--run",
            str(a.run),
            "--map-run",
            str(a.map_run),
            "--weights-only",
        ],
        check=True,
    )
    with np.load(a.run / "cells.npz") as data:
        zs, mass, windows = data["redshifts"], data["mass_msun"], data["windows_myr"]
        rates, covariance, upper, checks = [
            data[k] for k in ["mean", "covariance_of_mean", "censored_upper", "checks"]
        ]
    with np.load(a.run / "weights.npz") as data:
        w = data["weight_mpc3"]
    total = np.einsum("zm,zmw->zw", w, rates)
    cov = np.einsum("zm,zmij->zij", w**2, covariance)
    retained = total / total[:, -1, None]
    loss = 1 - retained
    variance = (
        np.diagonal(cov, axis1=1, axis2=2)
        + retained**2 * cov[:, -1, -1, None]
        - 2 * retained * cov[:, :, -1]
    ) / total[:, -1, None] ** 2
    # Roundoff can make the variance of the identically-zero reference tiny negative.
    if np.min(variance) < -1e-12:
        raise ValueError("invalid paired ratio variance")
    se = np.sqrt(np.maximum(variance, 0))
    ratio_cell = np.divide(
        rates,
        rates[:, :, -1, None],
        out=np.ones_like(rates),
        where=rates[:, :, -1, None] > 0,
    )
    loss_cell = 1 - ratio_cell
    budget_fraction = w * rates[:, :, -1] / total[:, -1, None]
    best = np.flatnonzero(np.max(loss + 2 * se, axis=0) < 0.01)
    k = load_popiii_ionizing_kernel(manifest["model"]["popiii_ssp"])
    yield100 = k.yield_photons(100)
    yield99 = brentq(lambda t: k.yield_photons(t) / yield100 - 0.99, 0.01, 100)
    rows = []
    for j, window in enumerate(windows):
        rows.append(
            dict(
                window_myr=float(window),
                ssp_yield_loss=float(1 - k.yield_photons(window) / yield100),
                max_global_loss=float(np.max(loss[:, j])),
                max_global_loss_plus_2se=float(np.max(loss[:, j] + 2 * se[:, j])),
                worst_global_z=float(zs[np.argmax(loss[:, j])]),
                worst_nonzero_cell_loss=float(np.max(loss_cell[:, :, j])),
                max_budget_fraction_in_cells_above_1pct=float(
                    np.max(np.sum(budget_fraction * (loss_cell[:, :, j] > 0.01), axis=1))
                ),
            )
        )
    qcheck = np.einsum("zm,zmc->zc", w, checks)
    result = dict(
        status="complete",
        criterion="Pop III global instantaneous photon rate: max over tested redshifts of relative loss + 2 paired Monte Carlo SE < 1%; not a spatial-map convergence claim",
        tested_redshifts=zs.tolist(),
        mass_range_msun=[float(mass[0]), float(mass[-1])],
        tracks_per_cell=manifest["model"]["n_tracks"],
        ssp_yield_99pct_age_myr=float(yield99),
        shortest_tested_window_myr=float(windows[best[0]]) if len(best) else None,
        windows=rows,
        per_redshift=[
            dict(z=float(z), loss=loss[i].tolist(), paired_se=se[i].tolist())
            for i, z in enumerate(zs)
        ],
        max_quadrature_relative_100myr=float(
            np.max(abs(qcheck[:, 1] - total[:, -1]) / total[:, -1])
        ),
        max_quadrature_relative_6myr=float(np.max(abs(qcheck[:, 2] - total[:, 5]) / total[:, -1])),
        max_censored_upper_over_resolved=float(
            np.max(np.einsum("zm,zmw->zw", w, upper)[:, -1] / total[:, -1])
        ),
        source_manifest_sha256=hashlib.sha256((a.run / "manifest.json").read_bytes()).hexdigest(),
        map_manifest_sha256=hashlib.sha256((a.map_run / "manifest.json").read_bytes()).hexdigest(),
    )
    (a.run / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        a.run / "comparison.npz",
        redshifts=zs,
        mass_msun=mass,
        windows_myr=windows,
        total_rate=total,
        relative_loss=loss,
        paired_se=se,
        cell_relative_loss=loss_cell,
        cell_reference_budget_fraction=budget_fraction,
    )
    print(
        json.dumps(
            {key: val for key, val in result.items() if key not in ["per_redshift"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
