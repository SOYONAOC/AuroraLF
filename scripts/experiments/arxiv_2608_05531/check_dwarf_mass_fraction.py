#!/usr/bin/env python3
"""Recompute the mass-weighted dwarf-galaxy fraction in paper Eqs. (4)-(5)."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from scipy.special import gamma, gammainc, gammaincc


DEFAULT_CONFIG = Path(__file__).resolve().with_name("reproduction.toml")


def load_gsmf_config(path: Path) -> dict[str, float]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    return config["gsmf"]


def finite_mass_integral_scaled(
    lower_msun: float,
    upper_msun: float,
    mstar_msun: float,
    components: tuple[tuple[float, float], ...],
) -> float:
    """Return the mass integral with the common factor Mstar omitted.

    For one Schechter component, the mass-weighted integral is proportional to
    phi * integral x**(beta + 1) exp(-x) dx, with x = M / Mstar.
    """
    lower_x = lower_msun / mstar_msun
    upper_x = upper_msun / mstar_msun
    total = 0.0
    for phi_mpc3, beta in components:
        shape = beta + 2.0
        total += phi_mpc3 * gamma(shape) * (
            gammainc(shape, upper_x) - gammainc(shape, lower_x)
        )
    return float(total)


def upper_mass_integral_scaled(
    lower_msun: float,
    mstar_msun: float,
    components: tuple[tuple[float, float], ...],
) -> float:
    """Return the mass integral from lower_msun to infinity, omitting Mstar."""
    lower_x = lower_msun / mstar_msun
    total = 0.0
    for phi_mpc3, beta in components:
        shape = beta + 2.0
        total += phi_mpc3 * gamma(shape) * gammaincc(shape, lower_x)
    return float(total)


def calculate_fraction(config: dict[str, float]) -> float:
    mstar_msun = 10.0 ** config["log10_mstar_msun"]
    components = (
        (config["phi1_mpc3"], config["beta1"]),
        (config["phi2_mpc3"], config["beta2"]),
    )
    numerator = finite_mass_integral_scaled(
        config["dwarf_mass_min_msun"],
        config["dwarf_mass_max_msun"],
        mstar_msun,
        components,
    )
    denominator = upper_mass_integral_scaled(
        config["dwarf_mass_min_msun"],
        mstar_msun,
        components,
    )
    return numerator / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_gsmf_config(args.config)
    calculated_percent = 100.0 * calculate_fraction(config)
    paper_percent = config["paper_mass_fraction_percent"]
    print(f"calculated_mass_fraction_percent={calculated_percent:.9f}")
    print(f"paper_mass_fraction_percent={paper_percent:.9f}")
    print(f"difference_percentage_points={calculated_percent - paper_percent:.9f}")
    print(
        "relative_difference="
        f"{calculated_percent / paper_percent - 1.0:.9%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
