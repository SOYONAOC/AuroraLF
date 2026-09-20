"""Compare completed 100 Myr and full-history instantaneous-source map runs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from plot_instantaneous_21cm import CASES, crossing, save, style


def read_run(path, *, rates_only=False):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["source_manifest"]["status"] != "complete":
        raise ValueError(f"Incomplete source table: {path}")
    if manifest["status"] != "complete" and not (rates_only and manifest["status"] == "running"):
        raise ValueError(f"Incomplete run: {path}")
    return manifest, None if rates_only else json.loads((path / "histories.json").read_text())


def percent_tex(value):
    if value == 0:
        return r"$0\%$"
    if abs(value) >= 0.001:
        return f"${value:.4f}\\%$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}\\%$"


def plot_rates(a):
    budgets = [json.loads((p / "rate_budget.json").read_text()) for p in [a.reference, a.run]]
    z = np.array([r["z"] for r in budgets[0]])
    np.testing.assert_array_equal(z, [r["z"] for r in budgets[1]])
    rates = [np.array([r["rate_s_mpc3"] for r in b]) for b in budgets]
    good = (z >= 6) & (z <= 20)
    if not all(np.isfinite(r).all() and np.all(r >= 0) for r in rates):
        raise ValueError("Invalid escaped rates")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    for j, (_, label, color) in enumerate(CASES[:2]):
        population = "Pop II" if j == 0 else "Pop III"
        for data, ls, window in zip(rates, ["--", "-"], ["full", "100 Myr"], strict=True):
            axes[0].plot(
                z[good], data[good, j], ls=ls, color=color, label=f"{population}, {window}"
            )
        axes[1].plot(
            z[good],
            100 * (1 - rates[1][good, j] / rates[0][good, j]),
            color=color,
            label=population,
        )
    axes[0].set(yscale="log", ylabel=r"Escaped rate [s$^{-1}$ cMpc$^{-3}$]")
    axes[1].set(ylabel="Reduction of current photon rate [%]")
    for ax in axes:
        ax.set(xlim=(20, 6), xlabel="Redshift")
        ax.legend(fontsize=9)
    save(fig, a, "lookback_rate_comparison")
    rows = []
    for target in [6, 8, 10, 12.5, 15, 20]:
        i = int(np.flatnonzero(z == target)[0])
        rows.append(
            dict(
                z=target,
                full_rates=rates[0][i].tolist(),
                window_rates=rates[1][i].tolist(),
                reduction_fraction=(1 - rates[1][i, :2] / rates[0][i, :2]).tolist(),
                popiii_share=float(rates[1][i, 1] / rates[1][i, :2].sum()),
            )
        )
    table = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"红移 & Pop II 率减少 & Pop III 率减少 & 当前 Pop III 占比\\",
        r"\midrule",
    ]
    for row in reversed(rows):
        two, three = 100 * np.asarray(row["reduction_fraction"])
        table.append(
            f"{row['z']:g} & {percent_tex(two)} & {percent_tex(three)} & {100 * row['popiii_share']:.2f}\\%\\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (a.assets / "lookback_rate_table.tex").write_text("\n".join(table) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--rates-only",
        action="store_true",
        help="Plot completed source budgets while maps are still running",
    )
    parser.add_argument(
        "--assets", type=Path, default=Path("slides/21cm_map/assets/lookback_100myr")
    )
    parser.add_argument("--previews", type=Path, default=Path("outputs/21cm_map/lookback_100myr"))
    a = parser.parse_args()
    manifests, histories = zip(
        *(read_run(p, rates_only=a.rates_only) for p in [a.reference, a.run]), strict=True
    )
    for key in ["grid", "physics", "power", "temporal", "redshifts", "density_sha256"]:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Comparison changes {key}")
    models = [m["source_manifest"]["resolved_model"].copy() for m in manifests]
    if models[1].pop("max_lookback_myr") != 100:
        raise ValueError("Expected a 100 Myr source window")
    if models[0].get("max_lookback_myr") is not None or models[0] != models[1]:
        raise ValueError("Reference must be the same model without an age cutoff")
    cosmology = FlatLambdaCDM(
        H0=100 * models[1]["h"], Om0=models[1]["omega_m"], Ob0=models[1]["omega_b"]
    )
    a.assets.mkdir(parents=True, exist_ok=True)
    a.previews.mkdir(parents=True, exist_ok=True)
    style()
    if a.rates_only:
        print(json.dumps(plot_rates(a), indent=2, allow_nan=False))
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    metrics = {}
    for ax, (key, label, color) in zip(axes, CASES[:2], strict=True):
        previous, current = (h[key] for h in histories)
        z = np.asarray(current["redshifts"])
        np.testing.assert_array_equal(z, previous["redshifts"])
        for h, ls, line_label in [
            (previous, "--", "Full stellar history"),
            (current, "-", "Maximum stellar age: 100 Myr"),
        ]:
            q = np.asarray(h["mean_xhii"])
            if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                raise ValueError("Invalid ionization history")
            ax.plot(z, q, color=color, ls=ls, label=line_label)
        ax.set(title=label, xlim=(20, 6), ylim=(0, 1.02), xlabel="Redshift")
        ax.axhline(0.5, color=".75", ls=":", lw=1)
        ax.legend(fontsize=9)
        metrics[key] = {
            "milestones": {
                str(target): {
                    "full": crossing(z, previous["mean_xhii"], target),
                    "100myr": crossing(z, current["mean_xhii"], target),
                }
                for target in [0.1, 0.5, 0.9]
            },
            "max_mean_xhii_change": float(
                np.max(abs(np.asarray(current["mean_xhii"]) - previous["mean_xhii"]))
            ),
        }
        stages = metrics[key]["milestones"]["0.5"]
        metrics[key]["half_ionization_delay_myr"] = float(
            (cosmology.age(stages["100myr"]["z"]) - cosmology.age(stages["full"]["z"])).to_value(
                "Myr"
            )
        )
    axes[0].set_ylabel(r"Volume-mean $x_{\rm HII}$")
    save(fig, a, "lookback_history_comparison")

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for key, label, color in CASES[:2]:
        previous, current = (h[key] for h in histories)
        difference = np.asarray(current["mean_xhii"]) - previous["mean_xhii"]
        ax.plot(current["redshifts"], 100 * difference, color=color, label=label)
    ax.axhline(0, color=".7", ls=":", lw=1)
    ax.set(
        xlim=(20, 6),
        xlabel="Redshift",
        ylabel="Ionized-fraction change [percentage points]",
        title="100 Myr stellar window minus full stellar history",
    )
    ax.legend()
    save(fig, a, "lookback_history_difference")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for key, label, color in CASES[:2]:
        h = histories[1][key]
        z50 = metrics[key]["milestones"]["0.5"]["100myr"]["z"]
        ax.plot(h["redshifts"], h["mean_xhii"], color=color, label=f"{label}: $z_{{50}}={z50:.3f}$")
        ax.scatter([z50], [0.5], color=color, s=24)
    ax.set(
        xlim=(20, 6),
        ylim=(0, 1.02),
        xlabel="Redshift (time increases to the right)",
        ylabel=r"Volume-mean $x_{\rm HII}$",
        title="Instantaneous sources: maximum stellar age 100 Myr",
    )
    ax.axhline(0.5, color=".75", ls=":", lw=1)
    ax.legend()
    save(fig, a, "lookback_current_history")

    rows = plot_rates(a)
    result = dict(
        status="complete",
        window_myr=100,
        metrics=metrics,
        rates=rows,
        inputs={
            str(p / f): hashlib.sha256((p / f).read_bytes()).hexdigest()
            for p in [a.reference, a.run]
            for f in ["manifest.json", "histories.json", "rate_budget.json"]
        },
    )
    (a.previews / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    values = {}
    for key, word in [("popii", "Two"), ("popii_popiii", "Three")]:
        milestones = metrics[key]["milestones"]["0.5"]
        values[f"Z{word}Full"] = milestones["full"]["z"]
        values[f"Z{word}Window"] = milestones["100myr"]["z"]
        values[f"MaxDiff{word}"] = 100 * metrics[key]["max_mean_xhii_change"]
        values[f"Delay{word}Myr"] = metrics[key]["half_ionization_delay_myr"]
    (a.assets / "lookback_values.tex").write_text(
        "\n".join(f"\\newcommand{{\\{key}}}{{{value:.6f}}}" for key, value in values.items()) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
