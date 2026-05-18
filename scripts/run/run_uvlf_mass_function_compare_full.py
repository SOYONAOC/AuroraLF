#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    raise RuntimeError(
        "Mass-function comparison production runs are disabled. "
        "AuroraLF now uses hmf Reed07 for UVLF sampling; run "
        f"{PROJECT_ROOT / 'scripts/run/run_uvlf_compare_imf_no_delay_all_z.py'} directly."
    )


if __name__ == "__main__":
    main()
