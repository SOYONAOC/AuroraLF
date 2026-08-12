#!/usr/bin/env python3
"""Archived Hebe burst-mass inversion; not a production AuroraLF model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.archive.heii1640 import (
    DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON,
    load_popiii_heii1640_luminosity_table,
    load_popiii_heplus_ionizing_photon_table,
)
from auroralf.ssp.convolution import interpolate_ssp_luminosity
from auroralf.archive.heii1640 import SCHAERER_HBETA_LUMINOSITY_COLUMN
from scripts.experiments.archived_compare_popiii_heii_to_hebe import (
    DEFAULT_HEBE_OBSERVATION_FILE,
    _load_hebe_observation_constraints,
    _require_constraint,
)


DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "archived_heii" / "hebe_burst_mass_inversion"
DEFAULT_SLIDE_OUTPUT = PROJECT_ROOT / "outputs" / "archived_heii" / "hebe_burst_mass_inversion_slide.pdf"
DEFAULT_SSP_MODELS = (
    (
        "Salpeter 1-100",
        PROJECT_ROOT / "external_data" / "ssp_spectra" / "schaerer2010_pop3" / "pop3_ge0_sal_100_001_is5.22",
    ),
    (
        "Salpeter 1-500",
        PROJECT_ROOT / "external_data" / "ssp_spectra" / "schaerer2010_pop3" / "pop3_ge0_sal_500_001_is5.22",
    ),
    (
        "Salpeter 50-500",
        PROJECT_ROOT / "external_data" / "ssp_spectra" / "schaerer2010_pop3" / "pop3_ge0_sal_500_050_is4.22",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Hebe Pop III burst-mass inversion and a Case-B HeII/Hgamma diagnostic."
    )
    parser.add_argument(
        "--enable-archived-heii",
        action="store_true",
        help="Explicitly run the archived, non-production He II diagnostic.",
    )
    parser.add_argument("--observation-file", type=Path, default=DEFAULT_HEBE_OBSERVATION_FILE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--slide-output", type=Path, default=DEFAULT_SLIDE_OUTPUT)
    parser.add_argument("--age-min-myr", type=float, default=0.01)
    parser.add_argument("--age-max-myr", type=float, default=3.0)
    parser.add_argument("--age-samples", type=int, default=500)
    parser.add_argument("--heii-caseb-erg-per-photon", type=float, default=DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON)
    parser.add_argument("--hgamma-to-hbeta", type=float, default=0.47)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.enable_archived_heii:
        raise RuntimeError(
            "This He II implementation is archived and excluded from production. "
            "Pass --enable-archived-heii only for historical reproduction."
        )
    if args.age_min_myr <= 0.0:
        raise ValueError("--age-min-myr must be positive")
    if args.age_max_myr <= args.age_min_myr:
        raise ValueError("--age-max-myr must be larger than --age-min-myr")
    if args.age_samples < 3:
        raise ValueError("--age-samples must be at least 3")
    if args.heii_caseb_erg_per_photon <= 0.0:
        raise ValueError("--heii-caseb-erg-per-photon must be positive")
    if args.hgamma_to_hbeta <= 0.0:
        raise ValueError("--hgamma-to-hbeta must be positive")


def required_burst_mass_msun(
    *,
    heii_luminosity_erg_s: float | np.ndarray,
    q_heplus_per_msun: float | np.ndarray,
    caseb_erg_per_photon: float = DEFAULT_CASEB_HEII1640_ERG_PER_PHOTON,
) -> float | np.ndarray:
    luminosity = np.asarray(heii_luminosity_erg_s, dtype=float)
    q_per_mass = np.asarray(q_heplus_per_msun, dtype=float)
    if not np.all(np.isfinite(luminosity)):
        raise ValueError("heii_luminosity_erg_s must contain only finite values")
    if np.any(luminosity < 0.0):
        raise ValueError("heii_luminosity_erg_s must be non-negative")
    if not np.all(np.isfinite(q_per_mass)):
        raise ValueError("q_heplus_per_msun must contain only finite values")
    if np.any(q_per_mass <= 0.0):
        raise ValueError("q_heplus_per_msun must be positive")
    if not np.isfinite(float(caseb_erg_per_photon)) or float(caseb_erg_per_photon) <= 0.0:
        raise ValueError("caseb_erg_per_photon must be positive")

    mass = luminosity / (float(caseb_erg_per_photon) * q_per_mass)
    if np.ndim(heii_luminosity_erg_s) == 0 and np.ndim(q_heplus_per_msun) == 0:
        return float(mass)
    return mass


def caseb_heii_to_hgamma_ratio(
    *,
    heii1640_luminosity_per_msun: float | np.ndarray,
    hbeta_luminosity_per_msun: float | np.ndarray,
    hgamma_to_hbeta: float = 0.47,
) -> float | np.ndarray:
    heii = np.asarray(heii1640_luminosity_per_msun, dtype=float)
    hbeta = np.asarray(hbeta_luminosity_per_msun, dtype=float)
    if not np.all(np.isfinite(heii)) or not np.all(np.isfinite(hbeta)):
        raise ValueError("line luminosities must contain only finite values")
    if np.any(heii < 0.0):
        raise ValueError("heii1640_luminosity_per_msun must be non-negative")
    if np.any(hbeta <= 0.0):
        raise ValueError("hbeta_luminosity_per_msun must be positive")
    if not np.isfinite(float(hgamma_to_hbeta)) or float(hgamma_to_hbeta) <= 0.0:
        raise ValueError("hgamma_to_hbeta must be positive")

    ratio = heii / (hbeta * float(hgamma_to_hbeta))
    if np.ndim(heii1640_luminosity_per_msun) == 0 and np.ndim(hbeta_luminosity_per_msun) == 0:
        return float(ratio)
    return ratio


def _schaerer_age_grid_from_table(file_path: Path, data: np.ndarray) -> np.ndarray:
    log_age_yr = data[:, 0]
    ages_myr = np.power(10.0, log_age_yr - 6.0)
    if np.all(np.diff(ages_myr) > 0.0):
        return ages_myr
    stem = file_path.stem
    if stem.endswith("_is4"):
        step_myr = 0.1
    elif stem.endswith("_is5"):
        step_myr = 1.0
    else:
        raise ValueError(f"Cannot reconstruct non-monotonic Schaerer age grid for {file_path}")
    return np.concatenate((np.array([1.0e-2], dtype=float), np.arange(1, data.shape[0], dtype=float) * step_myr))


def _load_hbeta_luminosity_table(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] <= SCHAERER_HBETA_LUMINOSITY_COLUMN:
        raise ValueError(f"Schaerer table lacks L(H_beta) column: {file_path}")
    ages_myr = _schaerer_age_grid_from_table(file_path, data)
    hbeta = np.asarray(data[:, SCHAERER_HBETA_LUMINOSITY_COLUMN], dtype=float)
    order = np.argsort(ages_myr, kind="stable")
    ages_myr = ages_myr[order]
    hbeta = hbeta[order]
    if np.any(ages_myr <= 0.0) or np.any(np.diff(ages_myr) <= 0.0):
        raise ValueError(f"Schaerer Hbeta ages must be strictly increasing for {file_path}")
    if np.any(hbeta <= 0.0):
        raise ValueError(f"Schaerer Hbeta luminosities must be positive for {file_path}")
    return ages_myr, hbeta


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ssp_label",
        "age_myr",
        "q_heplus_per_msun_s",
        "heii1640_per_msun_erg_s",
        "hbeta_per_msun_erg_s",
        "heii_to_hgamma_caseb",
        "mass_required_c1_msun",
        "mass_required_total_msun",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    output_prefix: Path,
    slide_output: Path,
    rows: list[dict[str, float | str]],
    observations: dict[str, float],
    age_grid_myr: np.ndarray,
) -> None:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    labels = list(dict.fromkeys(str(row["ssp_label"]) for row in rows))
    colors = {
        "Salpeter 1-100": "#2563eb",
        "Salpeter 1-500": "#059669",
        "Salpeter 50-500": "#dc2626",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), constrained_layout=True)
    ax_mass, ax_ratio = axes

    rusta_min = observations["rusta_mass_min"]
    rusta_max = observations["rusta_mass_max"]
    jeon_min = observations["jeon_mass_min"]
    jeon_max = observations["jeon_mass_max"]
    ax_mass.axhspan(rusta_min, rusta_max, color="#f59e0b", alpha=0.16, lw=0, label="Rusta C1 mass range")
    ax_mass.axhspan(jeon_min, jeon_max, color="#7c3aed", alpha=0.12, lw=0, label="Jeon LW-trigger range")

    for label in labels:
        subset = [row for row in rows if row["ssp_label"] == label]
        age = np.asarray([float(row["age_myr"]) for row in subset], dtype=float)
        mass_c1 = np.asarray([float(row["mass_required_c1_msun"]) for row in subset], dtype=float)
        mass_total = np.asarray([float(row["mass_required_total_msun"]) for row in subset], dtype=float)
        ratio = np.asarray([float(row["heii_to_hgamma_caseb"]) for row in subset], dtype=float)
        color = colors.get(label, "0.2")
        ax_mass.plot(age, mass_c1, color=color, lw=2.0, label=f"{label}, C1")
        ax_mass.plot(age, mass_total, color=color, lw=1.4, ls="--", alpha=0.75, label=f"{label}, total")
        ax_ratio.plot(age, ratio, color=color, lw=2.0, label=label)

    ax_mass.set_yscale("log")
    ax_mass.set_xlim(float(np.min(age_grid_myr)), float(np.max(age_grid_myr)))
    ax_mass.set_ylim(1.0e4, 2.0e8)
    ax_mass.set_xlabel("burst age [Myr]")
    ax_mass.set_ylabel(r"required $M_{\star,\rm III}$ [$M_\odot$]")
    ax_mass.set_title(r"HeII luminosity $\rightarrow$ burst mass")
    ax_mass.grid(True, which="major", alpha=0.22)
    ax_mass.grid(True, which="minor", alpha=0.08)
    ax_mass.legend(loc="upper left", frameon=False, ncols=1, fontsize=6.8)

    heii_hgamma_total = observations["heii_hgamma_total"]
    heii_hgamma_c2 = observations["heii_hgamma_c2"]
    ax_ratio.axhline(heii_hgamma_total, color="black", lw=1.5, ls="-", label="Hebe total/Hgamma")
    ax_ratio.axhspan(
        observations["heii_hgamma_total_lo"],
        observations["heii_hgamma_total_hi"],
        color="black",
        alpha=0.10,
        lw=0,
    )
    ax_ratio.axhline(heii_hgamma_c2, color="0.35", lw=1.3, ls=":", label="Hebe C2/Hgamma")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlim(float(np.min(age_grid_myr)), float(np.max(age_grid_myr)))
    ax_ratio.set_ylim(1.0e-3, 1.0e1)
    ax_ratio.set_xlabel("burst age [Myr]")
    ax_ratio.set_ylabel(r"Case-B HeII 1640 / H$\gamma$")
    ax_ratio.set_title(r"hardness diagnostic, Case-B only")
    ax_ratio.grid(True, which="major", alpha=0.22)
    ax_ratio.grid(True, which="minor", alpha=0.08)
    ax_ratio.legend(loc="lower left", frameon=False, fontsize=7.0)
    ax_ratio.text(
        0.04,
        0.95,
        rf"[NeIII]3870/H$\gamma < {observations['neiii_hgamma_upper']:.2f}$" + "\n"
        + rf"$EW_0({{H}}\gamma)>{observations['hgamma_ew_lower']:.0f}$ Å",
        transform=ax_ratio.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )

    fig.text(
        0.5,
        -0.03,
        "Mass inversion uses Schaerer instantaneous-burst Q(He+) kernels. "
        "Right panel is a Case-B sanity check, not a CLOUDY replacement for the Rusta diagnostic.",
        ha="center",
        va="top",
        fontsize=7.4,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500, bbox_inches="tight")
    slide_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(slide_output, dpi=500, bbox_inches="tight")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    observation_file = _resolve_project_path(args.observation_file)
    output_prefix = _resolve_project_path(args.output_prefix)
    slide_output = _resolve_project_path(args.slide_output)
    if not observation_file.is_file():
        raise FileNotFoundError(f"Hebe observation table not found: {observation_file}")

    constraints = _load_hebe_observation_constraints(observation_file)
    heii_lum_c1 = float(_require_constraint(constraints, "heii1640_luminosity_intrinsic_clean", "C1")["value"])
    heii_lum_total = float(_require_constraint(constraints, "heii1640_luminosity_intrinsic_clean", "total")["value"])
    heii_flux_total = float(_require_constraint(constraints, "heii1640_flux_observed", "total")["value"])
    heii_flux_total_err = float(_require_constraint(constraints, "heii1640_flux_observed", "total")["err"])
    heii_flux_c2 = float(_require_constraint(constraints, "heii1640_flux_observed", "C2")["value"])
    hgamma_flux = float(_require_constraint(constraints, "hgamma_flux_corrected", "total")["value"])
    hgamma_flux_err = float(_require_constraint(constraints, "hgamma_flux_corrected", "total")["err"])
    neiii_upper = float(_require_constraint(constraints, "neiii3870_flux_upper_limit", "total")["value"])
    hgamma_ew_lower = float(_require_constraint(constraints, "hgamma_ew0_lower_limit", "total")["value"])
    rusta_mass_min = float(_require_constraint(constraints, "popiii_stellar_mass_required_min", "C1")["value"])
    rusta_mass_max = float(_require_constraint(constraints, "popiii_stellar_mass_required_max", "C1")["value"])
    jeon_mass_min = float(_require_constraint(constraints, "popiii_lw_trigger_mass_min", "total")["value"])
    jeon_mass_max = float(_require_constraint(constraints, "popiii_lw_trigger_mass_max", "total")["value"])

    age_grid = np.linspace(float(args.age_min_myr), float(args.age_max_myr), int(args.age_samples))
    rows: list[dict[str, float | str]] = []
    for label, ssp_file in DEFAULT_SSP_MODELS:
        if not ssp_file.is_file():
            raise FileNotFoundError(f"Required Schaerer SSP table not found: {ssp_file}")
        q_age, q_heplus = load_popiii_heplus_ionizing_photon_table(ssp_file)
        heii_age, heii_luminosity, _ = load_popiii_heii1640_luminosity_table(ssp_file)
        hbeta_age, hbeta_luminosity = _load_hbeta_luminosity_table(ssp_file)
        q_interp = np.asarray(
            interpolate_ssp_luminosity(age_grid, ssp_age_grid=q_age, ssp_luv_grid=q_heplus),
            dtype=float,
        )
        heii_interp = np.asarray(
            interpolate_ssp_luminosity(age_grid, ssp_age_grid=heii_age, ssp_luv_grid=heii_luminosity),
            dtype=float,
        )
        hbeta_interp = np.asarray(
            interpolate_ssp_luminosity(age_grid, ssp_age_grid=hbeta_age, ssp_luv_grid=hbeta_luminosity),
            dtype=float,
        )
        ratio = np.asarray(
            caseb_heii_to_hgamma_ratio(
                heii1640_luminosity_per_msun=heii_interp,
                hbeta_luminosity_per_msun=hbeta_interp,
                hgamma_to_hbeta=float(args.hgamma_to_hbeta),
            ),
            dtype=float,
        )
        mass_c1 = np.asarray(
            required_burst_mass_msun(
                heii_luminosity_erg_s=heii_lum_c1,
                q_heplus_per_msun=q_interp,
                caseb_erg_per_photon=float(args.heii_caseb_erg_per_photon),
            ),
            dtype=float,
        )
        mass_total = np.asarray(
            required_burst_mass_msun(
                heii_luminosity_erg_s=heii_lum_total,
                q_heplus_per_msun=q_interp,
                caseb_erg_per_photon=float(args.heii_caseb_erg_per_photon),
            ),
            dtype=float,
        )
        for age, q_value, heii_value, hbeta_value, ratio_value, mass_c1_value, mass_total_value in zip(
            age_grid,
            q_interp,
            heii_interp,
            hbeta_interp,
            ratio,
            mass_c1,
            mass_total,
            strict=True,
        ):
            rows.append(
                {
                    "ssp_label": label,
                    "age_myr": float(age),
                    "q_heplus_per_msun_s": float(q_value),
                    "heii1640_per_msun_erg_s": float(heii_value),
                    "hbeta_per_msun_erg_s": float(hbeta_value),
                    "heii_to_hgamma_caseb": float(ratio_value),
                    "mass_required_c1_msun": float(mass_c1_value),
                    "mass_required_total_msun": float(mass_total_value),
                }
            )

    heii_hgamma_total = heii_flux_total / hgamma_flux
    heii_hgamma_total_err = heii_hgamma_total * np.sqrt(
        (heii_flux_total_err / heii_flux_total) ** 2 + (hgamma_flux_err / hgamma_flux) ** 2
    )
    observations = {
        "rusta_mass_min": rusta_mass_min,
        "rusta_mass_max": rusta_mass_max,
        "jeon_mass_min": jeon_mass_min,
        "jeon_mass_max": jeon_mass_max,
        "heii_hgamma_total": heii_hgamma_total,
        "heii_hgamma_total_lo": max(heii_hgamma_total - heii_hgamma_total_err, 1.0e-6),
        "heii_hgamma_total_hi": heii_hgamma_total + heii_hgamma_total_err,
        "heii_hgamma_c2": heii_flux_c2 / hgamma_flux,
        "neiii_hgamma_upper": neiii_upper / hgamma_flux,
        "hgamma_ew_lower": hgamma_ew_lower,
    }

    _write_rows(output_prefix.with_suffix(".csv"), rows)
    _plot(
        output_prefix=output_prefix,
        slide_output=slide_output,
        rows=rows,
        observations=observations,
        age_grid_myr=age_grid,
    )
    print(f"wrote {output_prefix.with_suffix('.csv')}")
    print(f"wrote {output_prefix.with_suffix('.pdf')}")
    print(f"wrote {output_prefix.with_suffix('.png')}")
    print(f"wrote {slide_output}")
    print(f"Hebe total HeII/Hgamma={heii_hgamma_total:.3f} +/- {heii_hgamma_total_err:.3f}")
    print(f"Hebe C2 HeII/Hgamma={observations['heii_hgamma_c2']:.3f}")


if __name__ == "__main__":
    main()
