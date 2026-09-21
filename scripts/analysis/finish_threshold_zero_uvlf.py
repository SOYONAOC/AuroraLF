"""Validate fresh cp6 mu=0 products and generate the two replacement UVLF figures.

Transfer high-z complete run directories and low-z manifest/curves products to
data_save/threshold_zero_20260918 first. Raw low-z samples remain in the frozen
remote release; their hashes stay in each manifest.
"""

import json
from pathlib import Path

import numpy as np

from auroralf.experiments.artifacts import digest
from scripts.analysis.analyze_random_q import summarize_run
from scripts.analysis.build_uvlf_efficiency_scan import CASES, make_figures

ROOT = Path(__file__).resolve().parents[2]
SAVE = ROOT / "data_save/threshold_zero_20260918"
OUT = ROOT / "outputs/threshold_zero_campaign_20260918"
DECK = ROOT / "slides/popiii_heii_pisn_complete_20260916"


def main():
    curves, sources, configs = {}, {}, {}
    for z, rid in [(12.5, 40), (14.5, 41)]:
        path = SAVE / f"AUR-EX-0006-R{rid:03}"
        edges = np.arange(-26.0, -7.99, 0.5)
        cfg, values, _, diagnostics = summarize_run(path, edges)
        if (cfg["z"], cfg["q_log10_mean"], cfg["q_log10_sigma"]) != (z, 0, 1.5):
            raise ValueError("Unexpected high-z config")
        result = {"bin_edges": edges}
        for key, *_ in CASES:
            result[key], result[key + "_se"], result[key + "_counts"] = values[key]
        curves[z], configs[str(z)] = result, dict(config=cfg, diagnostics=diagnostics)
        sources[str(path / "manifest.json")] = digest(path / "manifest.json")
    for z, rid in [(6, 42), (8, 43)]:
        parts, indices = [], []
        reference = None
        for shard in range(4):
            path = SAVE / f"AUR-EX-0006-R{rid:03}-shard{shard}-of4"
            manifest = json.loads((path / "manifest.json").read_text())
            cfg = manifest["config"]
            if (
                manifest["status"] != "complete"
                or manifest["shard_index"] != shard
                or manifest["shards"] != 4
            ):
                raise ValueError("Incomplete or wrong low-z shard")
            if (cfg["z"], cfg["q_log10_mean"], cfg["q_log10_sigma"]) != (z, 0, 1.5):
                raise ValueError("Unexpected low-z config")
            if reference is not None and cfg != reference:
                raise ValueError("Incompatible shard configs")
            reference = cfg
            if digest(path / "curves.npz") != manifest["products"]["curves.npz"]:
                raise ValueError("Corrupt shard curves")
            with np.load(path / "curves.npz") as data:
                edges = data["bin_edges"]
                np.testing.assert_array_equal(edges, np.arange(-28.0, 2.01, 0.5))
                parts.append(data["per_mass"])
                indices.append(data["global_mass_index"])
            sources[str(path / "manifest.json")] = digest(path / "manifest.json")
            sources[str(path / "curves.npz")] = digest(path / "curves.npz")
        np.testing.assert_array_equal(np.concatenate(indices), np.arange(reference["n_mass"]))
        per_mass = np.concatenate(parts, axis=1)
        if not np.isfinite(per_mass).all() or np.any(per_mass < 0):
            raise ValueError("Invalid per-mass UVLF contributions")
        phi = per_mass.sum(axis=1)
        se = np.sqrt(reference["n_mass"] * per_mass.var(axis=1, ddof=1))
        result = {"bin_edges": edges}
        for k, (key, *_) in enumerate(CASES):
            result[key], result[key + "_se"] = phi[k], se[k]
        curves[z], configs[str(z)] = result, dict(config=reference)
    for z, data in curves.items():
        np.savez_compressed(SAVE / f"uvlf_z{z:g}.npz", **data)
    comparison = make_figures(curves, {}, DECK, OUT)
    report = dict(
        status="complete",
        source="Fresh cp6 calculations, no reweighting",
        configs=configs,
        sources=sources,
        lowz_observation_comparisons=comparison,
        code_sha256=digest(__file__),
    )
    (SAVE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Validated four redshifts; replaced uvlf_high_z.pdf and uvlf_low_z_dust.pdf")


if __name__ == "__main__":
    main()
