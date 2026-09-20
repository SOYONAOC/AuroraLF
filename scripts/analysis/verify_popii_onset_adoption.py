"""Compare the adopted single-model workers with the frozen, validated experiment."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from auroralf.experiments import transition_workers as current
from auroralf.experiments.artifacts import digest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs/deployments/popii-transition-20260920-01"
OUT = ROOT / "outputs/popii_onset_adoption_20260921"


def main():
    # Require the real archived implementation, never regenerate a reference
    # with current code. Numerical kernels must still match that release.
    for name in ["popii_transition", "random_q", "ionizing_rates", "ionizing_audit"]:
        relative = f"auroralf/experiments/{name}.py"
        if digest(ROOT / relative) != digest(FROZEN / relative):
            raise ValueError("Reference dependency changed: " + relative)
    reference_path = FROZEN / "auroralf/experiments/transition_workers.py"
    spec = importlib.util.spec_from_file_location("frozen_transition_workers", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    plan = json.loads((ROOT / "configs/uvlf/popii_popiii.json").read_text())
    model = dict(plan["model"], n_tracks=4, track_chunk=4)
    for name in ["popii_ssp", "popiii_ssp"]:
        model[name] = str((ROOT / model[name]).resolve(strict=True))
    uv = str((ROOT / plan["popiii_uv"]).resolve(strict=True))
    edges = np.arange(-28.0, 2.01, 0.5)
    reference.initialize(model, uv)
    old_rate = reference.rate_cell((0, 0, 8.0, 1e9))[2]
    _, old_uv, old_diag = reference.uv_cell((0, 8.0, 1e9, edges))
    reports = {}
    for i, name in enumerate(["baseline", "delay0", "delay30"]):
        current.initialize(model, uv, (name,))
        new_rate = current.rate_cell((0, 0, 8.0, 1e9))[2]
        _, new_uv, new_diag = current.uv_cell((0, 8.0, 1e9, edges))
        for key in [
            "mean_rate",
            "se_rate",
            "rate_covariance_of_mean",
            "popii_censored_upper_extra",
        ]:
            np.testing.assert_array_equal(new_rate[key][0], old_rate[key][i])
        if i:
            np.testing.assert_array_equal(
                new_rate["popii_quadrature_error"][0], old_rate["popii_quadrature_error"][i - 1]
            )
        np.testing.assert_array_equal(new_uv[0], old_uv[i])
        np.testing.assert_array_equal(new_diag, old_diag[[i, 3, 4]])
        reports[name] = dict(exact_match=True, mean_rate=new_rate["mean_rate"][0].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(
        status="passed",
        tracks=4,
        redshift=8,
        halo_mass_msun=1e9,
        frozen_worker_sha256=digest(reference_path),
        current_worker_sha256=digest(Path(current.__file__)),
        variants=reports,
    )
    (OUT / "frozen_regression.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
