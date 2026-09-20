"""Prepare a reproducible, separately calibrated Pop II / mixed spatial run."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

from scripts.analysis.rescale_popiii_escape import derive

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT.parent / "SmallScale21cm"
TAG = "reionization_calibration_20260919"


def default_source(population):
    """Mixed runs use the explicitly adopted, validated causal-onset source table."""
    if population == "popii":
        return ROOT / "data_save/threshold_zero_all_20260918/ionizing"
    plan = json.loads((ROOT / "configs/uvlf/popii_popiii.json").read_text())
    if plan["variants"] != ["delay0"]:
        raise ValueError("Selected model differs from adopted sources; provide --source-base")
    adoption = plan["adoption"]
    path = ROOT / adoption["validated_sources"]
    manifest = json.loads((path / "manifest.json").read_text())
    expected = adoption["source_manifest_sha256"]
    if hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() != expected:
        raise ValueError("Adopted source manifest changed")
    if manifest["status"] != "complete" or manifest["variant"] != "delay0":
        raise ValueError("Default mixed source must be the validated zero-delay model")
    if plan["rates"] != manifest["config"]["grid"]:
        raise ValueError("Source grid changed; provide --source-base")
    for key, value in plan["model"].items():
        recorded = manifest["resolved_model"][key]
        if key in ("popii_ssp", "popiii_ssp"):
            actual = hashlib.sha256((ROOT / value).read_bytes()).hexdigest()
            if actual != manifest["input_sha256"][recorded]:
                raise ValueError("SSP changed; provide --source-base")
        elif value != recorded:
            raise ValueError("Source model changed: " + key + "; provide --source-base")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", choices=["popii", "popii_popiii"], required=True)
    p.add_argument("--fesc", type=float, required=True)
    p.add_argument(
        "--fesc-popiii", type=float, help="Independent Pop III escape; --fesc sets Pop II"
    )
    p.add_argument("--tag", help="Output campaign directory name")
    p.add_argument(
        "--source-base",
        type=Path,
        help="Explicit source table; mixed default is the adopted Pop III -> II model",
    )
    a = p.parse_args()
    if a.tag is None:
        a.tag = "popii_onset" if a.population == "popii_popiii" else TAG
    if a.source_base is None:
        a.source_base = default_source(a.population)
    if not a.tag or Path(a.tag).name != a.tag or a.tag in [".", ".."]:
        raise ValueError("tag must be one directory name")
    name = f"{a.population}_f{a.fesc:g}"
    if a.fesc_popiii is not None:
        if a.population != "popii_popiii":
            raise ValueError("Independent escape requires the mixed population")
        name = f"popii_f{a.fesc:g}_popiii_f{a.fesc_popiii:g}"
    source = ROOT / "data_save" / a.tag / "sources" / name
    run = SMALL / "runs/aurora_maps" / a.tag / name
    out = ROOT / "outputs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if a.fesc_popiii is None:
        derive(a.source_base, source, a.fesc, common=True)
    else:
        derive(a.source_base, source, a.fesc_popiii, fesc_popii=a.fesc)
    command = [
        str(SMALL / "packages/EoRCaLC/.venv/bin/python"),
        str(SMALL / "scripts/run/run_aurora_rates.py"),
        "--source-table",
        str(source),
        "--density-root",
        str(SMALL / "runs/eor_sweep_fesc005/kinf"),
        "--redshift-list",
        str(SMALL / "runs/eor_sweep_fesc005/redshift_list.npy"),
        "--output",
        str(run),
        "--cases",
        a.population,
        "--density-provenance-status",
        "legacy-cosmology-unverified",
    ]
    script = out / f"{name}.sh"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(SMALL))}\n"
        "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
        f"exec {shlex.join(command)}\n"
    )
    paths = [source / "sources.npz", source / "manifest.json", Path(command[1]), script]
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    record = dict(
        population=a.population,
        fesc=a.fesc,
        source=str(source),
        run=str(run),
        script=str(script),
        input_sha256=hashes,
    )
    if a.fesc_popiii is not None:
        record.update(fesc_popii=a.fesc, fesc_popiii=a.fesc_popiii)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
