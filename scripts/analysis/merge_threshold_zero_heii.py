"""Verify and combine the disjoint-redshift cp6 He II products without resampling."""

import json
from pathlib import Path

from auroralf.experiments.artifacts import digest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_save/threshold_zero_all_20260918"


def main():
    for original, target in [
        ("heii_v24_targets_20260914", "heii_low"),
        ("heii_uniform_band_20260917", "heii_uniform"),
    ]:
        old = json.loads((ROOT / f"configs/experiments/{original}.json").read_text())
        rows = []
        configs = []
        parents = {}
        reference = None
        for cfg in old["configs"]:
            folder = BASE / original / f"z{cfg['z']:g}"
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["status"] != "complete":
                raise ValueError("Incomplete " + str(folder))
            for name, sha in manifest["products"].items():
                if digest(folder / name) != sha:
                    raise ValueError("Corrupt " + str(folder / name))
            current = manifest["plan"]["configs"][0]
            if (current["z"], current["q_log10_mean"], current["q_log10_sigma"]) != (
                cfg["z"],
                0,
                1.5,
            ):
                raise ValueError("Wrong model")
            summary = json.loads((folder / "summary.json").read_text())
            if reference and summary["observations"] != reference["observations"]:
                raise ValueError("Observation conversion changed")
            reference = summary
            rows.extend(summary["model"])
            configs.append(current)
            parents[str(folder / "manifest.json")] = digest(folder / "manifest.json")
        out = BASE / target
        out.mkdir(exist_ok=True)
        reference["model"] = rows
        (out / "summary.json").write_text(json.dumps(reference, indent=2) + "\n")
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    plan=dict(configs=configs),
                    parents=parents,
                    products={"summary.json": digest(out / "summary.json")},
                ),
                indent=2,
            )
            + "\n"
        )
        print(out)


if __name__ == "__main__":
    main()
