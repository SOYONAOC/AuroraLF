from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_id_file(path: Path, *, limit: int | None) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"subhalo id file not found: {path}")
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        ids.append(int(stripped.split()[0]))
        if limit is not None and len(ids) >= int(limit):
            break
    if len(ids) == 0:
        raise ValueError(f"subhalo id file contains no IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"subhalo id file contains duplicate IDs: {path}")
    return ids


def _pairs_from_manifest(path: Path) -> list[tuple[int, Path]]:
    if not path.exists():
        raise FileNotFoundError(f"selection manifest not found: {path}")
    pairs: dict[tuple[int, Path], None] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot", "all_id_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"selection manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            all_id_file = str(row["all_id_file"]).strip()
            if all_id_file == "":
                continue
            pairs[(int(row["snapshot"]), _resolve_path(all_id_file))] = None
    if len(pairs) == 0:
        raise ValueError(f"selection manifest contains no all_id_file rows: {path}")
    return list(pairs)
