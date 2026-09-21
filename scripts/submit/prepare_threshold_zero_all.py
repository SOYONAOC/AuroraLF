"""Freeze all threshold-dependent He II and SFRD reruns for cp6."""

import json
from pathlib import Path

from auroralf.experiments.artifacts import digest
from auroralf.experiments.deployment import freeze_release

ROOT = Path(__file__).resolve().parents[2]
REL = "threshold-zero-all-20260918-02"
REMOTE = "/fs2/home/xuelei/zhuhr/AuroraLF/releases/" + REL
OUT = ROOT / "outputs/threshold_zero_all_20260918"
CONFIG = ROOT / "configs/experiments/threshold_zero_all_20260918"
CONFIG.mkdir(exist_ok=True)
commands = []
files = [
    str(p.relative_to(ROOT))
    for base in ["auroralf", "scripts"]
    for p in (ROOT / base).rglob("*.py")
    if ".venv" not in p.parts
]
files += ["pyproject.toml", "uv.lock"]
# Use frozen original plans to preserve sampling/selection, changing only mu.
for source in ["heii_v24_targets_20260914", "heii_uniform_band_20260917"]:
    original = json.loads((ROOT / f"configs/experiments/{source}.json").read_text())
    for i, cfg in enumerate(original["configs"]):
        cfg = dict(cfg)
        cfg["q_log10_mean"] = 0.0
        cfg["run_id"] = cfg["run_id"].replace("R05", "R07").replace("R060", "R080")
        for key in ["popii_ssp", "popiii_ssp"]:
            cfg[key] = str(Path(cfg[key]).relative_to(ROOT))
            files.append(cfg[key])
        plan = {
            k: v
            for k, v in original.items()
            if k not in ["configs", "input_sha256", "base_deck", "final_deck", "reuse"]
        }
        plan.update(
            configs=[cfg],
            line_ssp="external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_logE_500_001_is5.22",
            output=f"data_save/threshold_zero_all_20260918/{source}/z{cfg['z']:g}",
        )
        plan["input_sha256"] = {
            f: digest(ROOT / f)
            for f in original["input_sha256"]
            if (ROOT / f).is_file() and f not in [f"configs/experiments/{source}.json"]
        }
        name = f"configs/experiments/threshold_zero_all_20260918/{source}-{i}.json"
        (ROOT / name).write_text(json.dumps(plan, indent=2) + "\n")
        files.append(name)
        commands.append(
            f".venv/bin/python scripts/run/build_heii_v24_targets.py --data-only --plan {name}"
        )
original = json.loads((ROOT / "configs/experiments/heii_exact_targets_20260916.json").read_text())
newconfigs = []
for i, name in enumerate(original["configs"]):
    text = (
        (ROOT / name)
        .read_text()
        .replace("q_log10_mean = 0.5", "q_log10_mean = 0.0")
        .replace(str(ROOT) + "/", "../../../")
        .replace("AUR-EX-0006-R05", "AUR-EX-0006-R07")
    )
    new = f"configs/experiments/threshold_zero_all_20260918/exact-{i}.toml"
    (ROOT / new).write_text(text)
    newconfigs.append(new)
    files.append(new)
original["configs"] = newconfigs
original["output"] = "data_save/threshold_zero_all_20260918/exact"
original["assumptions"] = [v.replace("mean 0.5", "mean 0") for v in original["assumptions"]]
original["input_sha256"] = {
    f: digest(ROOT / f) for f in original["input_sha256"] if (ROOT / f).is_file()
}
original["input_sha256"].update({f: digest(ROOT / f) for f in newconfigs})
name = "configs/experiments/threshold_zero_all_20260918/exact.json"
(ROOT / name).write_text(json.dumps(original, indent=2) + "\n")
files.append(name)
commands.append(f".venv/bin/python scripts/run/build_heii_exact_targets.py --plan {name}")
commands.append(
    ".venv/bin/python scripts/analysis/build_current_sfrd.py --output data_save/threshold_zero_all_20260918/sfrd --q-log10-mean 0"
)
files += [
    "data_save/ionizing_sources/instantaneous_100myr_v1/" + f
    for f in ["manifest.json", "sources.npz"]
]
files += [
    str(p.relative_to(ROOT)) for p in (ROOT / "external_data/observations/heii").glob("*.json")
]
files += [
    "external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_logE_500_001_is5." + v
    for v in ["20", "22", "25"]
]
# All hashed dependencies must be shipped, including original configuration records.
for name in list(files):
    if name.endswith(".json") and name.startswith("configs/experiments/threshold_zero_all"):
        files += list(json.loads((ROOT / name).read_text()).get("input_sha256", {}))
# Add the complete 100/100 Myr source rerun, preserving the slide's stated experiment.
old = json.loads(
    (ROOT / "data_save/ionizing_sources/instantaneous_100myr_v1/manifest.json").read_text()
)["config"]
old["model"]["q_log10_mean"] = 0.0
old["model"]["popiii_max_age_myr"] = 100.0
old["output"] = "data_save/threshold_zero_all_20260918/ionizing"
s = "output = " + json.dumps(old["output"]) + "\n"
for table in ["grid", "model"]:
    s += "\n[" + table + "]\n"
    for k, v in old[table].items():
        s += k + " = " + json.dumps(v) + "\n"
name = "configs/experiments/threshold_zero_all_20260918/ionizing.toml"
(ROOT / name).write_text(s)
files.append(name)
for i in range(8):
    commands.append(
        f".venv/bin/python scripts/run/build_ionizing_rates.py --config {name} --shard-index {i} --shard-count 8"
    )
freeze_release(ROOT, ROOT / "outputs/deployments" / REL, sorted(set(files)))
(OUT / "tasks.json").write_text(
    json.dumps(dict(release=REL, remote=REMOTE, commands=commands), indent=2) + "\n"
)
print(len(commands), "tasks frozen at", REL)
