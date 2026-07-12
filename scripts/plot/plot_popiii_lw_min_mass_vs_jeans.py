from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import massfunc as mf
from auroralf.mah import Cosmology
from auroralf.uvlf import (
    DEFAULT_LW_BACKGROUND_J21,
    compute_atomic_cooling_mass_msun,
    compute_popiii_lw_minimum_mass_msun,
)


DEFAULT_OUTPUT_PREFIX = "outputs/popiii_lw_min_mass_vs_jeans_jlw0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Pop III LW minimum mass with massfunc's Jeans mass."
    )
    parser.add_argument("--z-min", type=float, default=6.0, help="Minimum redshift to plot.")
    parser.add_argument("--z-max", type=float, default=50.0, help="Maximum redshift to plot.")
    parser.add_argument("--n-z", type=int, default=400, help="Number of redshift samples.")
    parser.add_argument(
        "--lw-background-j21",
        type=float,
        default=DEFAULT_LW_BACKGROUND_J21,
        help="Homogeneous LW background in units of 1e-21 erg s^-1 cm^-2 Hz^-1 sr^-1.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PREFIX),
        help="Output prefix for PNG/PDF/CSV/TXT products.",
    )
    return parser.parse_args()


def _require_positive_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains non-finite values")
    if np.any(values <= 0.0):
        raise RuntimeError(f"{name} contains non-positive values")


def main() -> None:
    args = parse_args()
    cosmology = Cosmology()
    if args.z_min < 0.0 or args.z_max <= args.z_min:
        raise ValueError("--z-min must be non-negative and --z-max must be larger than --z-min")
    if args.n_z < 2:
        raise ValueError("--n-z must be at least 2")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")

    z_grid = np.linspace(args.z_min, args.z_max, args.n_z)
    sfrd = mf.SFRD(
        h=cosmology.h0_km_s_mpc / 100.0,
        omegam=cosmology.omega_m,
    )
    jeans_mass = np.asarray(sfrd.M_Jeans(z_grid), dtype=float)
    popiii_minimum_mass = np.asarray(
        compute_popiii_lw_minimum_mass_msun(
            z_grid,
            lw_background_j21=args.lw_background_j21,
        ),
        dtype=float,
    )
    atomic_cooling_mass = np.asarray(
        compute_atomic_cooling_mass_msun(z_grid, cosmology=cosmology),
        dtype=float,
    )

    _require_positive_finite("massfunc.SFRD().M_Jeans", jeans_mass)
    _require_positive_finite("Pop III LW minimum mass", popiii_minimum_mass)
    _require_positive_finite("atomic cooling mass", atomic_cooling_mass)

    output_prefix = args.output_prefix.expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_suffix(".txt")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "z",
                "lw_background_j21",
                "popiii_minimum_mass_msun",
                "massfunc_jeans_mass_msun",
                "atomic_cooling_mass_msun",
                "popiii_minimum_over_jeans",
            ],
        )
        writer.writeheader()
        for z_obs, popiii_mass, jeans, atomic in zip(
            z_grid,
            popiii_minimum_mass,
            jeans_mass,
            atomic_cooling_mass,
            strict=True,
        ):
            writer.writerow(
                {
                    "z": f"{z_obs:.8g}",
                    "lw_background_j21": f"{args.lw_background_j21:.8g}",
                    "popiii_minimum_mass_msun": f"{popiii_mass:.8e}",
                    "massfunc_jeans_mass_msun": f"{jeans:.8e}",
                    "atomic_cooling_mass_msun": f"{atomic:.8e}",
                    "popiii_minimum_over_jeans": f"{popiii_mass / jeans:.8e}",
                }
            )

    ratio = popiii_minimum_mass / jeans_mass
    summary_lines = [
        f"lw_background_j21={args.lw_background_j21:.8g}",
        f"z_range={args.z_min:.8g} {args.z_max:.8g}",
        f"n_z={args.n_z}",
        f"popiii_minimum_over_jeans_min={np.min(ratio):.8e}",
        f"popiii_minimum_over_jeans_max={np.max(ratio):.8e}",
    ]
    for z_ref in (10.0, 20.0, 30.0):
        popiii_ref = float(
            compute_popiii_lw_minimum_mass_msun(z_ref, lw_background_j21=args.lw_background_j21)
        )
        jeans_ref = float(sfrd.M_Jeans(z_ref))
        atomic_ref = float(
            compute_atomic_cooling_mass_msun(z_ref, cosmology=cosmology)
        )
        summary_lines.extend(
            [
                f"z={z_ref:g}",
                f"  popiii_minimum_mass_msun={popiii_ref:.8e}",
                f"  massfunc_jeans_mass_msun={jeans_ref:.8e}",
                f"  atomic_cooling_mass_msun={atomic_ref:.8e}",
                f"  popiii_minimum_over_jeans={popiii_ref / jeans_ref:.8e}",
            ]
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plt.style.use("apj")
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, (ax_mass, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(5.8, 5.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    lw_label = rf"Pop III $M_{{\rm min}}$ ($J_{{\rm LW,21}}={args.lw_background_j21:g}$)"
    ax_mass.plot(z_grid, popiii_minimum_mass, color="black", lw=2.2, label=lw_label)
    ax_mass.plot(z_grid, jeans_mass, color="#1f77b4", lw=2.0, ls="-.", label=r"massfunc $M_{\rm Jeans}$")
    ax_mass.plot(
        z_grid,
        atomic_cooling_mass,
        color="0.45",
        lw=1.4,
        ls=":",
        label=r"Atomic cooling $T_{\rm vir}=10^4$ K",
    )
    ax_mass.set_yscale("log")
    ax_mass.set_ylabel(r"Halo mass [$M_\odot$]")
    ax_mass.grid(alpha=0.22)
    ax_mass.legend(frameon=False, loc="best")

    ax_ratio.plot(z_grid, ratio, color="black", lw=2.0)
    ax_ratio.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel(r"Redshift $z$")
    ax_ratio.set_ylabel(r"$M_{\rm min}^{\rm PopIII}/M_{\rm Jeans}$")
    ax_ratio.grid(alpha=0.22)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    print(f"saved_png={png_path}", flush=True)
    print(f"saved_pdf={pdf_path}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
