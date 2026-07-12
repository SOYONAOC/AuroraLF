from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.io.legacy import convert_legacy_uvlf_npz


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one legacy AuroraLF UVLF NPZ into strict v2 HDF5.",
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing complete output transactionally.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = convert_legacy_uvlf_npz(
        args.npz,
        args.config,
        args.manifest,
        args.output,
        overwrite=bool(args.overwrite),
    )
    print(f"converted_hdf5={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
