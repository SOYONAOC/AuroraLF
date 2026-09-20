"""Compare full spatial escape-fraction reruns with neutral-fraction constraints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_run(path):
    path = Path(path)
    m = json.loads((path / "manifest.json").read_text())
    if m["status"] != "complete":
        raise ValueError(f"Incomplete spatial run: {path}")
    return m, json.loads((path / "histories.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    common = plan.get("escape_mode") == "common_popii_popiii"
    baseline = Path(plan["baseline_run"])
    bm, bh = read_run(baseline)
    z = np.asarray(bh["popii"]["redshifts"])
    q0, q2 = [np.asarray(bh[c]["mean_xhii"]) for c in ["popii", "popii_popiii"]]
    curves = [(0.0, q0, str(baseline)), (0.2, q2, str(baseline))]
    hashes = {str(args.plan.resolve()): digest(args.plan)}
    for case in plan["cases"]:
        path = Path(case["run"])
        m, h = read_run(path)
        for key in ["power", "physics", "grid", "density_sha256", "redshifts"]:
            if m[key] != bm[key]:
                raise ValueError(f"Paired run mismatch: {key}")
        model = m["source_manifest"]["resolved_model"]
        expected_ii = case["fesc_popiii"] if common else 0.2
        if model["fesc_popii"] != expected_ii or model["fesc_popiii"] != case["fesc_popiii"]:
            raise ValueError("Wrong escape fractions")
        for key, value in bm["source_manifest"]["resolved_model"].items():
            varying = ["fesc_popii", "fesc_popiii"] if common else ["fesc_popiii"]
            if key not in varying and model[key] != value:
                raise ValueError(f"Stellar input changed: {key}")
        row = h["popii_popiii"]
        np.testing.assert_array_equal(row["redshifts"], z)
        q = np.asarray(row["mean_xhii"])
        lower_bound = 0.0 if common else q0 - 2e-6
        if not np.isfinite(q).all() or np.any(q < lower_bound) or np.any(q > q2 + 2e-6):
            raise ValueError("Invalid or non-monotonic escape-fraction history")
        curves.append((case["fesc_popiii"], q, str(path)))
    curves.sort(key=lambda x: x[0])
    ordered = curves[1:] if common else curves
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        if np.any(lower[1] > upper[1] + 2e-6):
            raise ValueError("Ionization is not monotonic in Pop III escape fraction")
    for _, _, path in curves:
        for name in ["manifest.json", "histories.json"]:
            file = Path(path) / name
            hashes[str(file)] = digest(file)
    obs_path = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
    hashes[str(obs_path)] = digest(obs_path)
    with obs_path.open() as stream:
        observations = list(csv.DictReader(stream))
    results = []
    for fesc, q, path in curves:
        values, score = [], 0.0
        for obs in observations:
            zz = float(obs["z"])
            if not z.min() <= zz <= z.max():
                values.append(
                    {
                        "id": obs["id"],
                        "model_xhi": None,
                        "reason": "outside simulated redshift range",
                    }
                )
                continue
            value = float(np.interp(zz, z[::-1], (1 - q)[::-1]))
            center, lo, hi = (float(obs[k]) for k in ["xhi", "lower", "upper"])
            narrow = float(obs["z_min"]) == float(obs["z_max"]) and obs["kind"] == "interval"
            residual = (value - center) / (center - lo if value < center else hi - center)
            if narrow:
                score += residual**2
            values.append(
                {
                    "id": obs["id"],
                    "model_xhi": value,
                    "within_reported_interval": lo <= value <= hi,
                    "broad_redshift_bin": float(obs["z_min"]) != float(obs["z_max"]),
                    "used_in_diagnostic_score": narrow,
                    "signed_interval_scaled_residual": residual,
                }
            )
        crossings = np.where((q[:-1] < 0.99) & (q[1:] >= 0.99))[0]
        z99 = None
        if len(crossings):
            j = crossings[0]
            w = (0.99 - q[j]) / (q[j + 1] - q[j])
            z99 = float(z[j] * (1 - w) + z[j + 1] * w)
        results.append(
            {
                "fesc_popiii": fesc,
                "fesc_popii": fesc if common and fesc else 0.2,
                "population": "popii" if fesc == 0 else "popii_popiii",
                "run": path,
                "diagnostic_score": score,
                "z99": z99,
                "observations": values,
            }
        )
    data_out, plot_out = Path(plan["data_output"]), Path(plan["plot_output"])
    data_out.mkdir(parents=True, exist_ok=True)
    plot_out.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    colors = ["#777777", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#B65D27"]
    if len(curves) > len(colors):
        colors = ["#777777"] + [
            plt.get_cmap("viridis")(v) for v in np.linspace(0.1, 0.9, len(curves) - 1)
        ]
    model_lines = []
    for (fesc, q, _), color in zip(curves, colors[: len(curves)], strict=True):
        label = rf"$f_{{\rm esc,III}}={fesc:g}$"
        if common:
            label = (
                r"Pop II only, $f_{\rm esc}=0.2$"
                if fesc == 0
                else rf"Pop II+III, $f_{{\rm esc}}={fesc:g}$"
            )
        (line,) = ax.plot(
            z,
            1 - q,
            color=color,
            lw=1.8,
            ls="--" if fesc in (0.0, 0.2) else "-",
            label=label,
        )
        model_lines.append((fesc, line))
    for obs in observations:
        zz = float(obs["z"])
        if obs["kind"] != "interval":
            continue
        center, lo, hi = (float(obs[k]) for k in ["xhi", "lower", "upper"])
        broad = float(obs["z_min"]) != float(obs["z_max"])
        label = {
            "mason18_7": "Mason+18",
            "davies18_709": "Davies+18",
            "davies18_754": None,
            "mason26_65": "Mason+26 (broad z bins)",
            "mason26_93": None,
        }[obs["id"]]
        ax.errorbar(
            zz,
            center,
            yerr=[[center - lo], [hi - center]],
            xerr=[[zz - float(obs["z_min"])], [float(obs["z_max"]) - zz]],
            fmt="s" if broad else "o",
            color="#79549D" if broad else "black",
            mfc="white",
            ms=5,
            capsize=3,
            lw=1,
            label=label,
            zorder=5,
        )
    ax.set(
        xlim=(5.4, 10.8),
        ylim=(-0.025, 1.025),
        xlabel="Redshift z",
        ylabel=r"Volume-averaged neutral fraction $\langle x_{\rm HI}\rangle_V$",
    )
    title = (
        r"Shared $f_{\rm esc,II}=f_{\rm esc,III}$ in mixed model, $\mu=0$"
        if common
        else r"Fixed $f_{\rm esc,II}=0.2$, threshold $\mu=0$"
    )
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_out / "escape_scan.pdf")
    fig.savefig(plot_out / "escape_scan.png", dpi=160)
    if common:
        best = min(results[1:], key=lambda r: r["diagnostic_score"])["fesc_popiii"]
        selected = {0.0, 0.05, best, 0.1, 0.2}
        for value, line in model_lines:
            if value not in selected:
                line.remove()
            elif value == best:
                line.set_linewidth(2.6)
        fig.set_size_inches(10.4, 4.6)
        ax.legend(fontsize=10, ncol=3, loc="lower right")
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        fig.tight_layout()
        fig.savefig(plot_out / "escape_selected.pdf")
        fig.savefig(plot_out / "escape_selected.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "complete",
        "escape_mode": "common_popii_popiii" if common else "independent_popiii",
        "results": results,
        "lowest_score_sampled_fesc": min(
            results[1:] if common else results, key=lambda r: r["diagnostic_score"]
        )["fesc_popiii"],
        "score_definition": "Descriptive sum of squared residuals scaled by the appropriate side of published68%intervals at three narrow-z points. Not a fitted likelihood, confidence region or independent validation. Broad-z bins plotted at centers only and excluded from score; no redshift-selection weights supplied.",
        "limitations": "Paired100/100Myr source windows; unchanged recombination; legacy density cosmology unverified. Constant H-ionizing escape only; no HeII or nebular-continuum coupling added.",
        "input_sha256": hashes,
    }
    (data_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "lowest_score_sampled_fesc": summary["lowest_score_sampled_fesc"],
                "scores": [
                    {k: r[k] for k in ["fesc_popiii", "diagnostic_score", "z99"]} for r in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
