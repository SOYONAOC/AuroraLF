#!/usr/bin/env python3
"""Audit the current interpreter against the reconstructed paper dependencies."""

from __future__ import annotations

import argparse
from importlib import metadata
import sys


REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "ml": (
        ("numpy", "author listed; exact version unavailable"),
        ("pandas", "author listed; exact version unavailable"),
        ("scipy", "author README misspells this as 'spicy'"),
        ("scikit-learn", "author listed; exact version unavailable"),
        ("catboost", "author notebook states 1.2.x"),
        ("matplotlib", "author listed; exact version unavailable"),
        ("jupyter", "author listed; exact version unavailable"),
    ),
    "maps": (
        ("numpy", "author listed; exact version unavailable"),
        ("pandas", "author listed; exact version unavailable"),
        ("healpy", "author notebook states >=1.16"),
        ("matplotlib", "author listed; exact version unavailable"),
        ("jupyter", "author listed; exact version unavailable"),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ml", "maps", "all"), default="ml")
    return parser.parse_args(argv)


def selected_requirements(stage: str) -> tuple[tuple[str, str], ...]:
    if stage != "all":
        return REQUIREMENTS[stage]
    combined: dict[str, str] = {}
    for requirements in REQUIREMENTS.values():
        combined.update(requirements)
    return tuple(combined.items())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    missing: list[str] = []
    for distribution, provenance in selected_requirements(args.stage):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = "MISSING"
            missing.append(distribution)
        print(f"{distribution:<16} {version:<16} {provenance}")

    if missing:
        print(
            "missing required distributions: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    print("all reconstructed stage dependencies are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
