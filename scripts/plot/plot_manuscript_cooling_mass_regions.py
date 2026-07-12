from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_atomic_cooling_mass_msun, compute_popiii_lw_minimum_mass_msun
from auroralf.mah import Cosmology


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "manuscript"
    / "auroralf_project_summary"
    / "assets"
    / "cooling_mass_regions.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Pop III/Pop II cooling-based halo-mass routing regions for the manuscript."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=40.0)
    parser.add_argument("--n-z", type=int, default=400)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.lw_background_j21) or args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be finite and non-negative")
    if not np.isfinite(args.z_min) or args.z_min < 0.0:
        raise ValueError("--z-min must be finite and non-negative")
    if not np.isfinite(args.z_max) or args.z_max <= args.z_min:
        raise ValueError("--z-max must be finite and larger than --z-min")
    if args.n_z < 2:
        raise ValueError("--n-z must be at least 2")


def main() -> None:
    args = parse_args()
    cosmology = Cosmology()
    validate_args(args)

    z = np.linspace(float(args.z_min), float(args.z_max), int(args.n_z))
    m_mol = np.asarray(
        compute_popiii_lw_minimum_mass_msun(z, lw_background_j21=float(args.lw_background_j21)),
        dtype=float,
    )
    m_atom = np.asarray(
        compute_atomic_cooling_mass_msun(z, cosmology=cosmology),
        dtype=float,
    )

    if not np.all(np.isfinite(m_mol)) or np.any(m_mol <= 0.0):
        raise RuntimeError("computed molecular-cooling floor must be finite and positive")
    if not np.all(np.isfinite(m_atom)) or np.any(m_atom <= 0.0):
        raise RuntimeError("computed atomic-cooling threshold must be finite and positive")
    if np.any(m_atom <= m_mol):
        raise RuntimeError("atomic-cooling threshold must remain above the molecular-cooling floor")

    plt.style.use("apj")
    plt.rcParams.update(
        {
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )

    y_min = 2.0e4
    y_max = 5.0e8
    fig, ax = plt.subplots(figsize=(6.05, 3.85), constrained_layout=True)
    ax.fill_between(z, y_min, m_mol, color="#f2f3f6", lw=0.0)
    ax.fill_between(z, m_mol, m_atom, color="#d9cdea", lw=0.0)
    ax.fill_between(z, m_atom, y_max, color="#c7dceb", lw=0.0)
    ax.plot(z, m_mol, color="white", lw=1.2)
    ax.plot(z, m_atom, color="white", lw=1.2)

    ax.text(26.0, 1.55e8, "Pop II + Pop III", color="#1f2a3d", fontsize=10, ha="center")
    ax.text(28.0, 1.9e6, "Pop III minihalos", color="#4f3e78", fontsize=10, ha="center")
    ax.text(28.0, 4.5e4, "No star formation", color="#4b5563", fontsize=9, ha="center")

    z_arrow = 12.5
    m_arrow = float(np.interp(z_arrow, z, m_mol))
    ax.annotate(
        "LW-regulated\nlower bound\n" + r"$M_{\rm min,III}(z,J_{\rm LW})$",
        xy=(z_arrow, m_arrow),
        xytext=(8.0, 4.3e6),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#4f3e78", "lw": 0.9},
        color="#4f3e78",
        fontsize=7.4,
        ha="left",
        va="center",
    )

    ax.set_yscale("log")
    ax.set_xlim(float(args.z_min), float(args.z_max))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Halo mass [$M_\odot$]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"wrote_pdf={args.output}", flush=True)
    print(f"lw_background_j21={float(args.lw_background_j21):.8g}", flush=True)
    print(f"z_range={float(args.z_min):.8g},{float(args.z_max):.8g}", flush=True)


if __name__ == "__main__":
    main()
