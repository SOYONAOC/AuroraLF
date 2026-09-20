"""Derive escaped-rate tables at an independent, constant Pop III escape fraction."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scale_arrays(arrays, old_fesc, new_fesc):
    """Scale escaped rates and their full covariance, keeping Pop II unchanged."""
    if not np.isfinite([old_fesc, new_fesc]).all() or not 0 < old_fesc <= 1:
        raise ValueError("Original Pop III escape fraction must be in (0,1]")
    if not 0 <= new_fesc <= 1:
        raise ValueError("New Pop III escape fraction must be in [0,1]")
    required = {
        "redshifts",
        "mass_msun",
        "mean_rate",
        "se_rate",
        "rate_covariance_of_mean",
        "quadrature_error",
    }
    if set(arrays) != required:
        raise ValueError("Unexpected rate-table fields; review scaling semantics")
    out = {key: np.array(value, copy=True) for key, value in arrays.items()}
    shape = (len(out["redshifts"]), len(out["mass_msun"]))
    for key in ["mean_rate", "se_rate"]:
        if out[key].shape != (*shape, 3):
            raise ValueError(f"Invalid {key} shape")
    if out["rate_covariance_of_mean"].shape != (*shape, 3, 3):
        raise ValueError("Invalid covariance shape")
    if out["quadrature_error"].shape != shape:
        raise ValueError("Invalid quadrature error shape")
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Nonfinite rate-table input")
    ratio = new_fesc / old_fesc
    factors = np.array([1.0, ratio, ratio])
    for key in ["mean_rate", "se_rate"]:
        out[key] *= factors
    out["rate_covariance_of_mean"] *= factors[:, None] * factors[None, :]
    out["quadrature_error"] *= ratio
    return out


def scale_common_arrays(arrays, old_fesc, new_fesc):
    """Scale every population equally, preserving source ratios and covariance."""
    out = scale_arrays(arrays, old_fesc, new_fesc)
    ratio = new_fesc / old_fesc
    for key in ["mean_rate", "se_rate"]:
        out[key][..., 0] *= ratio
    factors = np.array([ratio, 1.0, 1.0])
    out["rate_covariance_of_mean"] *= factors[:, None] * factors[None, :]
    return out


def scale_independent_arrays(arrays, old_ii, new_ii, old_iii, new_iii):
    """Rescale both escape fractions, including zero Pop II and cross covariance."""
    if not np.isfinite([old_ii, new_ii]).all() or not 0 < old_ii <= 1:
        raise ValueError("Original Pop II escape fraction must be in (0,1]")
    if not 0 <= new_ii <= 1:
        raise ValueError("New Pop II escape fraction must be in [0,1]")
    out = scale_arrays(arrays, old_iii, new_iii)
    factors = np.array([new_ii / old_ii, 1.0, 1.0])
    for key in ["mean_rate", "se_rate"]:
        out[key] *= factors
    out["rate_covariance_of_mean"] *= factors[:, None] * factors[None, :]
    return out


def derive(source, output, fesc, *, common=False, fesc_popii=None):
    if common and fesc_popii is not None:
        raise ValueError("Select common or independent escape fractions")
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "manifest.json").read_text())
    if original["status"] != "complete" or original["source_format"] != "instantaneous-rates-v1":
        raise ValueError("A completed instantaneous-rate table is required")
    if original["channels"] != ["popii", "popiii_resolved", "popiii_censored_upper_extra"]:
        raise ValueError("Unexpected channels")
    if digest(source / "sources.npz") != original["product_sha256"]:
        raise ValueError("Original product hash mismatch")
    old_fesc = original["resolved_model"]["fesc_popiii"]
    if common and original["resolved_model"]["fesc_popii"] != old_fesc:
        raise ValueError("Common scaling requires equal original Pop II and Pop III escape")
    with np.load(source / "sources.npz", allow_pickle=False) as data:
        if fesc_popii is None:
            arrays = (scale_common_arrays if common else scale_arrays)(dict(data), old_fesc, fesc)
        else:
            arrays = scale_independent_arrays(
                dict(data), original["resolved_model"]["fesc_popii"], fesc_popii, old_fesc, fesc
            )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "sources.npz", **arrays)
    manifest = deepcopy(original)
    manifest["resolved_model"]["fesc_popiii"] = fesc
    manifest["config"]["model"]["fesc_popiii"] = fesc
    if common:
        manifest["resolved_model"]["fesc_popii"] = fesc
        manifest["config"]["model"]["fesc_popii"] = fesc
    manifest["config"]["output"] = str(output)
    # Original simulation identities are retained inside the parent manifest.
    for key in ["job_id", "shard_jobs", "merge_job", "started_unix", "completed_unix"]:
        manifest.pop(key, None)
    manifest["derivation"] = {
        "kind": "exact linear escaped-source rescaling at fixed stellar histories",
        "parent_directory": str(source),
        "parent_manifest_sha256": digest(source / "manifest.json"),
        "parent_product_sha256": original["product_sha256"],
        "script_sha256": digest(__file__),
        "old_fesc_popiii": old_fesc,
        "new_fesc_popiii": fesc,
        "fesc_popii": original["resolved_model"]["fesc_popii"],
        "created_unix": time.time(),
        "scope": "H-ionizing escape to IGM only; HeII escape and nebular SSP unchanged",
        "escape_mode": "common_popii_popiii" if common else "independent_popiii",
    }
    if common:
        manifest["derivation"]["fesc_popii"] = fesc
        manifest["derivation"]["old_fesc_popii"] = old_fesc
    if fesc_popii is not None:
        manifest["resolved_model"]["fesc_popii"] = fesc_popii
        manifest["config"]["model"]["fesc_popii"] = fesc_popii
        manifest["derivation"].update(
            fesc_popii=fesc_popii,
            old_fesc_popii=original["resolved_model"]["fesc_popii"],
            escape_mode="independent_popii_popiii",
        )
    manifest["product_sha256"] = digest(output / "sources.npz")
    (output / "parent_manifest.json").write_text(json.dumps(original, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fesc-popiii", type=float)
    group.add_argument("--fesc-common", type=float)
    parser.add_argument("--fesc-popii", type=float, help="Optional independent Pop II escape")
    args = parser.parse_args()
    common = args.fesc_common is not None
    fesc = args.fesc_common if common else args.fesc_popiii
    result = derive(args.source, args.output, fesc, common=common, fesc_popii=args.fesc_popii)
    print(json.dumps(result["derivation"], indent=2))


if __name__ == "__main__":
    main()
