"""Join a newly computed low-z source table to unchanged archived high-z cells."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def join_arrays(base, low):
    """Require an independently recomputed overlap; retain original high-z values."""
    required = {
        "redshifts",
        "mass_msun",
        "mean_rate",
        "se_rate",
        "rate_covariance_of_mean",
        "quadrature_error",
    }
    base, low = dict(base), dict(low)
    for arrays in [base, low]:
        if "computed" in arrays:
            mask = arrays.pop("computed")
            if mask.shape != arrays["mean_rate"].shape[:2] or not np.all(mask):
                raise ValueError("Uncomputed source cells")
        if set(arrays) != required or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Invalid source fields")
        if np.any(np.diff(arrays["redshifts"]) <= 0):
            raise ValueError("Redshifts must increase")
    # np.logspace differs by one float64 ULP across the archived and local
    # numerical builds. Allow only representation-level differences; retain
    # the archived mass nodes in the combined product.
    np.testing.assert_array_max_ulp(base["mass_msun"], low["mass_msun"], maxulp=2)
    common, old_i, new_i = np.intersect1d(base["redshifts"], low["redshifts"], return_indices=True)
    if len(common) < 1 or common[0] != base["redshifts"][0]:
        raise ValueError("Require a recomputed overlap at the original low-z boundary")
    for key in required - {"redshifts", "mass_msun"}:
        np.testing.assert_allclose(
            base[key][old_i],
            low[key][new_i],
            rtol=2e-8,
            atol=0,
            err_msg=f"Changed overlap physics or sampling: {key}",
        )
    new = low["redshifts"] < base["redshifts"][0]
    if not new.any() or np.any(low["redshifts"] > base["redshifts"][0]):
        raise ValueError("Low-z extension must terminate at the original boundary")
    result = {"mass_msun": base["mass_msun"].copy()}
    for key in required - {"mass_msun"}:
        result[key] = np.concatenate([low[key][new], base[key]])
    return result, common.tolist()


def validate_onset_match(old, new):
    """Pre-transition v1 manifests without a variant describe independent onset."""
    identities = []
    for manifest in (old, new):
        variant = manifest.get("variant", "baseline")
        delay = manifest.get("delay_myr")
        expected = {"baseline": None, "delay0": 0.0, "delay30": 30.0}
        if variant not in expected or delay != expected[variant]:
            raise ValueError("Invalid Pop II onset identity")
        identities.append((variant, delay))
    if identities[0] != identities[1]:
        raise ValueError("Changed Pop II onset; cannot merge source tables")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--low", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifests, arrays = [], []
    for path in [a.base, a.low]:
        m = json.loads((path / "manifest.json").read_text())
        if m["status"] != "complete" or m["source_format"] != "instantaneous-rates-v1":
            raise ValueError(f"Source table incomplete: {path}")
        if digest(path / "sources.npz") != m["product_sha256"]:
            raise ValueError(f"Source product hash mismatch: {path}")
        manifests.append(m)
        with np.load(path / "sources.npz") as data:
            arrays.append(dict(data))
    old, new = manifests
    validate_onset_match(old, new)
    for key, value in old["resolved_model"].items():
        if key not in ["popii_ssp", "popiii_ssp"] and new["resolved_model"][key] != value:
            raise ValueError(f"Changed source parameter: {key}")
    for key in ["popii_ssp", "popiii_ssp"]:
        if (
            old["input_sha256"][old["resolved_model"][key]]
            != new["input_sha256"][new["resolved_model"][key]]
        ):
            raise ValueError(f"Changed SSP input: {key}")
    if old["channels"] != new["channels"]:
        raise ValueError("Changed source channels")
    combined, overlap = join_arrays(*arrays)
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / "sources.npz", **combined)
    merged = deepcopy(old)
    merged["config"]["output"] = str(a.output.resolve())
    merged["config"]["grid"]["redshifts"] = combined["redshifts"].tolist()
    merged["product_sha256"] = digest(a.output / "sources.npz")
    for key in [
        "job_id",
        "started_unix",
        "completed_unix",
        "shard_jobs",
        "merge_job",
        "shard_count",
    ]:
        merged.pop(key, None)
    merged["extension"] = dict(
        kind="New low-z cells; original high-z table retained exactly",
        verified_overlap_redshifts=overlap,
        overlap_relative_tolerance=2e-8,
        mass_grid_maximum_allowed_ulps=2,
        mass_grid_maximum_relative_difference=float(
            np.max(abs(arrays[0]["mass_msun"] - arrays[1]["mass_msun"]) / arrays[0]["mass_msun"])
        ),
        parents=[
            dict(
                directory=str(p.resolve()),
                manifest_sha256=digest(p / "manifest.json"),
                product_sha256=digest(p / "sources.npz"),
            )
            for p in [a.base, a.low]
        ],
        script_sha256=digest(__file__),
    )
    (a.output / "manifest.json").write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
    print(json.dumps(merged["extension"], indent=2))


if __name__ == "__main__":
    main()
