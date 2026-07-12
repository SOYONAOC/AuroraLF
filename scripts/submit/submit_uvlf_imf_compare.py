#!/usr/bin/env python3
from __future__ import annotations

import argparse


MIGRATION_MESSAGE = (
    "This legacy UVLF submit entry point is disabled. Use "
    "scripts/submit/submit_uvlf_v2.py --config configs/uvlf/production.toml."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MIGRATION_MESSAGE)
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _parse_args(argv)
    raise SystemExit(MIGRATION_MESSAGE)


if __name__ == "__main__":
    main()
