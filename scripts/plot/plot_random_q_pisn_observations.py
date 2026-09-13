"""Show the redshift scope of the HSC bound alongside the current PISN rates.

The HSC limit is template dependent and order of magnitude, not a 95% bound.
This plot does not perform a fit or extrapolate it to the model redshifts.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.experiments.artifacts import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("data_save/pisn_random_q_20260911/summary.json")
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("external_data/observations/pisn_rate_constraints.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("slides/pisn_random_q_20260911/assets/observation_context.pdf"),
    )
    args = parser.parse_args()
    model = json.loads(args.summary.read_text())
    for name, sha in model["source_hashes"].items():
        require(digest(name) == sha, f"parent input/code changed: {name}")
    observations = json.loads(args.observations.read_text())
    obs = observations["constraints"][0]
    require(
        obs["kind"] == "order_of_magnitude_upper_bound" and obs["confidence_level"] is None,
        "bound semantics changed",
    )
    require(obs["time_frame"] == "source" and obs["volume"] == "comoving", "rate units differ")
    lo, hi = obs["typical_sensitivity_redshift"]
    value = obs["value_gpc3_yr"]
    plt.style.use("apj")
    plt.rcParams["text.latex.preamble"] += r"\usepackage{amssymb}"
    fig, ax = plt.subplots(figsize=(9.5, 3.8), layout="constrained")
    ax.axvspan(lo, hi, color="0.93", zorder=0)
    ax.hlines(value, lo, hi, color="0.3", linestyle="--", linewidth=1.8)
    for z in (1.4, 2.6):
        ax.annotate(
            "",
            xy=(z, value / 2.3),
            xytext=(z, value),
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1.3},
        )
    ax.text(
        0.45,
        220,
        "Subaru/HSC: luminous templates\nOrder-of-magnitude bound\nMoriya et al. (2021)",
        fontsize=10,
        va="bottom",
    )
    ax.text(0.65, 125, r"$\lesssim 100$", fontsize=12)
    points = []
    for case in model["cases"]:
        source = case["estimates"]["source_rate"]
        # 1 Gpc^3 = 10^9 Mpc^3; rate per Gpc^3 is therefore larger by 10^9.
        rate = source["value"] * 1e9
        error = source["cluster_mc_se"] * 1e9
        require(np.isfinite([rate, error]).all() and rate > 0 and error >= 0, "invalid model rate")
        ax.errorbar(case["z"], rate, yerr=error, fmt="o", capsize=4, color="C0")
        points.append({"z": case["z"], "rate_gpc3_yr": rate, "cluster_mc_se_gpc3_yr": error})
    ax.text(
        10.0,
        1850,
        r"Current $\epsilon_b=0.03$ model" + "\nAll 140--260 " + r"$M_\odot$" + " PISNe",
        fontsize=11,
        color="C0",
    )
    ax.text(
        7.0,
        90,
        "Different redshift coverage\nand progenitor selection",
        fontsize=11,
        color="0.4",
        ha="center",
    )
    ax.set(
        xlim=(0, 16),
        ylim=(30, 3500),
        yscale="log",
        xlabel="Redshift",
        ylabel=r"PISN rate [source yr$^{-1}$ cGpc$^{-3}$]",
    )
    ax.set_xticks([0, 1, 3, 6, 9, 12.5, 14.5, 16])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    result = {
        "model_summary_sha256": digest(args.summary),
        "observation_source_sha256": digest(args.observations),
        "plot_source_sha256": digest(__file__),
        "output_sha256": digest(args.output),
        "model_points": points,
        "observation": obs,
        "interpretation": "Redshift-coverage context only. HSC is not a constraint on these high-z points; no statistical consistency or exclusion inferred.",
    }
    output = args.summary.parent / "observation_context.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
