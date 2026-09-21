"""Verify the completed low-z extension without repeating formation histories."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/uvlf_lowz_sampling_64x_20260914"
SCAN = ROOT / "data_save/uvlf_efficiency_scan_20260914_64x"
OUT = ROOT / "outputs/uvlf_lowz_review_20260915"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    scan = json.loads((SCAN / "manifest.json").read_text())
    if manifest["status"] != "complete" or scan["status"] != "complete":
        raise ValueError("Incomplete run/export")
    checked = {}
    for name, expected in {
        **manifest["input_sha256"],
        **{str(RUN / k): v for k, v in manifest["products"].items()},
        **scan["products"],
    }.items():
        actual = digest(name)
        if actual != expected:
            raise ValueError(f"Provenance mismatch: {name}")
        checked[name] = actual
    print(f"Verified {len(checked)} frozen input/product hashes", flush=True)
    report = dict(
        status="verified",
        job="157889",
        input_product_sha256=checked,
        scope="Mass/MAH Monte Carlo sampling only; no dust, time-grid or physical-model convergence claim",
        rows=[],
        visual_review="pending",
    )
    for z in (6, 8):
        with (
            np.load(RUN / f"z{z}.npz") as current,
            np.load(Path(manifest["parent"]) / f"z{z}.npz") as parent,
            np.load(RUN / f"z{z}_convergence.npz") as convergence,
            np.load(SCAN / f"z{z}.npz") as export,
        ):
            # Mmap the uncompressed arrays; verify in bounded blocks.
            arrays = {
                k: np.load(RUN / f"z{z}_{k}.npy", mmap_mode="r")
                for k in ("popii", "popiii", "probability")
            }
            if arrays["popii"].shape != (23040, 1024):
                raise ValueError("Unexpected final history count")
            for k in ("popiii", "probability"):
                if arrays[k].shape != (23040, 1024, 5):
                    raise ValueError("Invalid conditional-age strata")
            for start in range(0, 23040, 128):
                for k, array in arrays.items():
                    block = array[start : start + 128]
                    if not np.isfinite(block).all() or np.any(block < 0):
                        raise ValueError(f"Invalid {k} z={z} block={start}")
                np.testing.assert_allclose(
                    arrays["probability"][start : start + 128].sum(axis=-1), 1, atol=2e-15, rtol=0
                )
            for k, array in arrays.items():
                old = parent[k]
                np.testing.assert_array_equal(array[: old.shape[0], : old.shape[1]], old)
                del old
            np.testing.assert_array_equal(convergence["full_shape"], [23040, 1024])
            np.testing.assert_array_equal(current["mass_msun"][:5760], parent["mass_msun"])
            np.testing.assert_allclose(
                current["weight_per_track"][:5760] * 4,
                parent["weight_per_track"],
                rtol=1e-14,
                atol=0,
            )
            np.testing.assert_allclose(
                convergence["eps0.03_original_size_phi"], parent["phi"], rtol=1e-10
            )
            np.testing.assert_allclose(
                convergence["eps0.03_original_size_se"], parent["se"], rtol=1e-10
            )
            np.testing.assert_array_equal(current["phi"], convergence["eps0.03_full_phi"])
            np.testing.assert_array_equal(current["se"], convergence["eps0.03_full_se"])
            x = (convergence["bin_edges"][1:] + convergence["bin_edges"][:-1]) / 2
            for eps in (0.01, 0.03, 0.1):
                key = f"eps{eps:g}"
                phi, se = [convergence[key + "_full_" + a][2] for a in ("phi", "se")]
                old, ose = [convergence[key + "_original_size_" + a][2] for a in ("phi", "se")]
                np.testing.assert_allclose(export[key], phi, rtol=1e-11, atol=1e-100)
                np.testing.assert_allclose(export[key + "_se"], se, rtol=1e-11, atol=1e-100)
                mask = (x >= -24) & (x <= -16) & (phi > 0) & (old > 0)
                d = convergence[key + "_half_difference_sigma"][2][mask]
                report["rows"].append(
                    dict(
                        z=z,
                        efficiency=eps,
                        magnitude_range=[float(x[mask].min()), float(x[mask].max())],
                        mc_fraction_min=float(np.min(se[mask] / phi[mask])),
                        mc_fraction_max=float(np.max(se[mask] / phi[mask])),
                        mc_fraction_median=float(np.median(se[mask] / phi[mask])),
                        parent_mc_fraction_median=float(np.median(ose[mask] / old[mask])),
                        independent_halves_max_abs_sigma=float(np.nanmax(np.abs(d))),
                        parent_max_fraction_shift=float(np.max(np.abs(phi[mask] / old[mask] - 1))),
                    )
                )
            print(f"z={z}: samples, nesting, strata normalization and exports verified", flush=True)
    obs = json.loads((SCAN / "summary.json").read_text())["observational_comparison"]
    report["observations_near_minus21"] = [r for r in obs if abs(r["Muv"] + 21) < 0.3]
    (OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
