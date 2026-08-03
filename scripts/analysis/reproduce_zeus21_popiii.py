#!/usr/bin/env python3
"""Reproduce the Zeus21 fiducial Pop III global history.

This script follows the public Pop II+III fiducial tutorial accompanying
Cruz et al. (2025, arXiv:2407.18294).  It deliberately imports Zeus21 from a
pinned source checkout rather than treating it as an undeclared AuroraLF
dependency.  The resulting CSV is an explicit bridge product: its redshift-
dependent LW background can be supplied to AuroraLF's molecular-cooling
threshold without changing either model's equations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_popiii_lw_minimum_mass_msun


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZEUS21_ROOT = PROJECT_ROOT / "third_party" / "zeus21"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data_save" / "zeus21_popiii_fiducial.csv"
DEFAULT_OUTPUT_FIGURE = PROJECT_ROOT / "outputs" / "zeus21_popiii_fiducial.png"
EXPECTED_ZEUS21_COMMIT = "9f2d2105e99e74096092e2061082a79c3f85eaca"
PAPER_ARXIV_ID = "2407.18294"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeus21-root", type=Path, default=DEFAULT_ZEUS21_ROOT)
    parser.add_argument("--expected-commit", default=EXPECTED_ZEUS21_COMMIT)
    parser.add_argument("--z-min", type=float, default=10.0)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_OUTPUT_FIGURE)
    return parser.parse_args()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_checkout(root: Path, expected_commit: str) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"Zeus21 source checkout is missing: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            f"Zeus21 checkout is at {commit}, expected pinned commit {expected_commit}"
        )
    dirty_paths = _git_output(root, "status", "--short", "--untracked-files=no")
    if dirty_paths:
        raise RuntimeError(f"Zeus21 tracked source has local changes:\n{dirty_paths}")
    return commit


def _build_zeus21_fiducial(zeus21: object, z_min: float) -> tuple[object, object, object]:
    if not np.isfinite(z_min) or z_min < 5.0 or z_min >= 50.0:
        raise ValueError("z_min must be finite and lie in [5, 50)")

    user = zeus21.User_Parameters(precisionboost=1.2)
    cosmo_input = zeus21.Cosmo_Parameters_Input(
        omegac=0.11933,
        omegab=0.02242,
        h_fid=0.6766,
        As=np.exp(3.047) * 1.0e-10,
        ns=0.9665,
        tau_fid=0.0544,
        USE_RELATIVE_VELOCITIES=True,
        Flag_emulate_21cmfast=False,
    )
    classy_cosmo = zeus21.runclass(cosmo_input)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_input, classy_cosmo)
    # Building the correlation structure also registers the zero-lag
    # correlation table used internally by get_T21_coefficients.
    zeus21.Correlations(user, cosmo, classy_cosmo)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy_cosmo)
    astro = zeus21.Astro_Parameters(
        user,
        cosmo,
        astromodel=0,
        accretion_model=0,
        alphastar=0.5,
        betastar=-0.5,
        epsstar=0.1,
        Mc=3.0e11,
        dlog10epsstardz=0.0,
        fesc10=0.1,
        alphaesc=0.0,
        L40_xray=10.0**0.5,
        E0_xray=500.0,
        alpha_xray=-1.0,
        Emax_xray_norm=2000.0,
        Nalpha_lyA_II=9690,
        Nalpha_lyA_III=17900,
        Mturn_fixed=None,
        FLAG_MTURN_SHARP=False,
        C0dust=4.43,
        C1dust=1.99,
        sigmaUV=0.5,
        USE_POPIII=True,
        USE_LW_FEEDBACK=True,
        alphastar_III=0.0,
        betastar_III=0.0,
        fstar_III=1.0e-3,
        Mc_III=1.0e7,
        dlog10epsstardz_III=0.0,
        fesc7_III=10.0**-1.35,
        alphaesc_III=-0.3,
        L40_xray_III=10.0**0.5,
        alpha_xray_III=-1.0,
        A_LW=2.0,
        beta_LW=0.6,
        A_vcb=1.0,
        beta_vcb=1.8,
    )
    coefficients = zeus21.get_T21_coefficients(
        user,
        cosmo,
        classy_cosmo,
        astro,
        hmf,
        zmin=z_min,
    )
    return coefficients, cosmo, astro


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    lengths = {np.asarray(values).size for values in columns.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"output columns have inconsistent lengths: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for values in zip(*(np.asarray(value, dtype=float) for value in columns.values()), strict=True):
            writer.writerow({name: f"{value:.12e}" for name, value in zip(columns, values, strict=True)})


def _plot_history(path: Path, columns: dict[str, np.ndarray]) -> None:
    z = columns["redshift"]
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True)

    axes[0].semilogy(z, columns["sfrd_popii_msun_yr_mpc3"], label="Pop II", color="#4C78A8")
    axes[0].semilogy(z, columns["sfrd_popiii_msun_yr_mpc3"], label="Pop III", color="#E45756")
    axes[0].set_ylabel(r"SFRD [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].semilogy(z, columns["j21_lw_popii"], label="Pop II", color="#4C78A8")
    axes[1].semilogy(z, columns["j21_lw_popiii"], label="Pop III", color="#E45756")
    axes[1].semilogy(z, columns["j21_lw_total"], label="total", color="#222222", linewidth=1.8)
    axes[1].set_ylabel(r"$J_{21}^{\rm LW}$")
    axes[1].legend(frameon=False, ncol=3)

    axes[2].semilogy(
        z,
        columns["mmol_no_feedback_msun"],
        label="no feedback",
        color="#72B7B2",
    )
    axes[2].semilogy(
        z,
        columns["mmol_lw_only_msun"],
        label="self-consistent LW",
        color="#F58518",
    )
    axes[2].semilogy(
        z,
        columns["mmol_lw_vcb_mean_msun"],
        label=r"LW + mean $v_{cb}$",
        color="#B279A2",
    )
    axes[2].set_xlabel("redshift z")
    axes[2].set_ylabel(r"$M_{\rm mol}$ [$M_\odot$]")
    axes[2].legend(frameon=False, ncol=3, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(direction="in", which="both", top=True, right=True)
    axes[-1].set_xlim(float(np.nanmax(z)), float(np.nanmin(z)))
    figure.suptitle("Zeus21 fiducial Pop III reproduction (Cruz et al. 2025)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = _parse_args()
    zeus21_root = args.zeus21_root.resolve()
    commit = _validate_source_checkout(zeus21_root, args.expected_commit)
    sys.path.insert(0, str(zeus21_root))

    import zeus21  # type: ignore[import-not-found]  # noqa: PLC0415
    from zeus21.sfrd import Mmol, Mmol_LW  # type: ignore[import-not-found]  # noqa: PLC0415

    started = time.perf_counter()
    coefficients, cosmo, astro = _build_zeus21_fiducial(zeus21, args.z_min)
    runtime_seconds = time.perf_counter() - started

    redshift = np.asarray(coefficients.zintegral, dtype=float)
    j21_popii = np.asarray(coefficients.J_21_LW_II, dtype=float)
    j21_popiii = np.asarray(coefficients.J_21_LW_III, dtype=float)
    j21_total = j21_popii + j21_popiii
    j21_interpolator = lambda z: np.interp(z, redshift, j21_total)  # noqa: E731

    mmol_no_feedback = np.asarray(
        compute_popiii_lw_minimum_mass_msun(redshift, lw_background_j21=0.0),
        dtype=float,
    )
    mmol_lw_auroralf = np.asarray(
        compute_popiii_lw_minimum_mass_msun(redshift, lw_background_j21=j21_total),
        dtype=float,
    )
    mmol_lw_zeus21 = np.asarray(Mmol_LW(astro, j21_interpolator, redshift), dtype=float)
    np.testing.assert_allclose(mmol_lw_auroralf, mmol_lw_zeus21, rtol=2.0e-12, atol=0.0)
    mmol_full = np.asarray(
        Mmol(astro, cosmo, j21_interpolator, redshift, cosmo.vcb_avg),
        dtype=float,
    )

    columns = {
        "redshift": redshift,
        "sfrd_popii_msun_yr_mpc3": np.asarray(coefficients.SFRD_II_avg, dtype=float),
        "sfrd_popiii_msun_yr_mpc3": np.asarray(coefficients.SFRD_III_avg, dtype=float),
        "j21_lw_popii": j21_popii,
        "j21_lw_popiii": j21_popiii,
        "j21_lw_total": j21_total,
        "mmol_no_feedback_msun": mmol_no_feedback,
        "mmol_lw_only_msun": mmol_lw_auroralf,
        "mmol_lw_vcb_mean_msun": mmol_full,
    }
    if not all(np.all(np.isfinite(values)) for values in columns.values()):
        raise RuntimeError("Zeus21 reproduction produced non-finite output")
    if np.any(columns["sfrd_popiii_msun_yr_mpc3"] < 0.0):
        raise RuntimeError("Zeus21 reproduction produced negative Pop III SFRD")

    _write_csv(args.output_csv.resolve(), columns)
    _plot_history(args.output_figure.resolve(), columns)
    metadata_path = args.output_csv.resolve().with_suffix(".metadata.json")
    metadata = {
        "schema_version": "auroralf.zeus21_popiii_fiducial.v1",
        "paper_arxiv_id": PAPER_ARXIV_ID,
        "source_url": "https://github.com/ZeusCosmo/Zeus21",
        "source_commit": commit,
        "runtime_seconds": runtime_seconds,
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("zeus21", "classy", "numpy", "scipy", "mcfit", "powerbox", "pyfftw")
        },
        "parameters": {
            "z_min": float(args.z_min),
            "precisionboost": 1.2,
            "fstar_popiii": 1.0e-3,
            "A_LW": 2.0,
            "beta_LW": 0.6,
            "A_vcb": 1.0,
            "beta_vcb": 1.8,
            "relative_velocities": True,
        },
        "outputs": {
            "csv": _portable_path(args.output_csv),
            "figure": _portable_path(args.output_figure),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
