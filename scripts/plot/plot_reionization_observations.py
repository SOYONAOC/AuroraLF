"""Compare completed spatial histories with sourced neutral-fraction constraints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from plot_instantaneous_21cm import CASES, crossing

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "external_data/observations/reionization/neutral_fraction.csv"
OUT = ROOT / "outputs/21cm_map/observations"
ASSETS = ROOT / "slides/assets/reionization"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=ROOT.parent / "SmallScale21cm/runs/aurora_maps/instantaneous_100myr_v1",
    )
    p.add_argument("--gated-run", type=Path, help="Completed Pop III emission-gate counterfactual")
    p.add_argument("--output-dir", type=Path, help="Override diagnostic output directory")
    p.add_argument("--asset-dir", type=Path, default=ASSETS, help="Slide asset directory")
    a = p.parse_args()
    out = OUT if a.gated_run is None else OUT.parent / "popiii_z10"
    figure_name = "neutral_history" if a.gated_run is None else "popiii_z10"
    manifest_path = a.run / "manifest.json"
    history_path = a.run / "histories.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("A completed spatial run is required")
    histories = json.loads(history_path.read_text())
    cases = list(CASES[:2])
    input_paths = [manifest_path, history_path, OBS]
    gate_validation = None
    if a.gated_run is not None:
        gate_manifest_path = a.gated_run / "manifest.json"
        gate_history_path = a.gated_run / "histories.json"
        gm = json.loads(gate_manifest_path.read_text())
        gh = json.loads(gate_history_path.read_text())["popii_popiii"]
        assert gm["status"] == "complete"
        high_only = gm.get("popiii_min_redshift") == 10.0
        assert (high_only and gm.get("popiii_max_redshift") is None) or (
            gm.get("popiii_max_redshift") == 10.0 and gm.get("popiii_min_redshift") is None
        )
        gate_key = "popiii_zge10" if high_only else "popiii_zle10"
        gate_tex = r"\geq" if high_only else r"\leq"
        if high_only:
            figure_name = "popiii_zge10"
            out = OUT.parent / figure_name
        for key in [
            "source_manifest_sha256",
            "power",
            "physics",
            "grid",
            "redshifts",
            "density_sha256",
        ]:
            assert gm[key] == manifest[key], f"Paired input mismatch: {key}"
        histories[gate_key] = gh
        cases.append((gate_key, rf"Pop II + Pop III only at $z{gate_tex}10$", "#178479"))
        input_paths += [gate_manifest_path, gate_history_path]
        z_gate = np.asarray(gh["redshifts"])
        q_gate = np.asarray(gh["mean_xhii"])
        q_ii = np.asarray(histories["popii"]["mean_xhii"])
        q_all = np.asarray(histories["popii_popiii"]["mean_xhii"])
        high = z_gate > 10
        high_reference = q_all if high_only else q_ii
        np.testing.assert_allclose(q_gate[high], high_reference[high], atol=1e-7, rtol=1e-6)
        assert np.all(q_gate >= q_ii - 1e-6) and np.all(q_gate <= q_all + 1e-6)
        splits = [r for r in gh["rows"] if len(r["step"].get("substeps", [])) == 2]
        assert len(splits) == 1
        first, second = splits[0]["step"]["substeps"]
        assert first["z_after"] == second["z_before"] == 10
        assert first["popiii_enabled"] == high_only and second["popiii_enabled"] != high_only
        gate_validation = {
            "high_z_reference": "popii_popiii" if high_only else "popii",
            "max_high_z_difference": float(np.max(abs(q_gate[high] - high_reference[high]))),
            "history_between_popii_and_ungated": True,
            "switch_row": splits[0],
            "gate": gm["emission_gate"],
            "popiii_max_redshift": gm["popiii_max_redshift"],
            "popiii_min_redshift": gm.get("popiii_min_redshift"),
        }
        budget_paths = [a.run / "rate_budget.json", a.gated_run / "rate_budget.json"]
        old_budget, gated_budget = [json.loads(path.read_text()) for path in budget_paths]
        assert len(old_budget) == len(gated_budget)
        for before, after in zip(old_budget, gated_budget, strict=True):
            assert before["z"] == after["z"]
            expected = np.array(before["rate_s_mpc3"])
            if (high_only and after["z"] < 10) or (not high_only and after["z"] > 10):
                expected[1:] = 0
            np.testing.assert_array_equal(expected, after["rate_s_mpc3"])
        gate_validation["popiii_photon_shares"] = {
            str(r["z"]): r["rate_s_mpc3"][1] / sum(r["rate_s_mpc3"][:2])
            for r in gated_budget
            if r["z"] in [6.0, 8.0, 10.0]
        }
        if high_only:
            post_z, post_q = z_gate[~high], q_gate[~high]
            drawdown = np.maximum.accumulate(post_q) - post_q
            j = int(np.argmax(drawdown))
            i = int(np.argmax(post_q[: j + 1]))
            gate_validation["post_switch_decline"] = {
                "z_before": float(post_z[i]),
                "z_after": float(post_z[j]),
                "xhii_before": float(post_q[i]),
                "xhii_after": float(post_q[j]),
                "drop_in_ionized_fraction": float(drawdown[j]),
                "definition": "largest decline below a previous post-switch snapshot's volume mean; no reset of ionized state",
            }
        input_paths += budget_paths
    with OBS.open() as f:
        obs = list(csv.DictReader(f))
    for row in obs:
        for k in ["z", "z_min", "z_max", "xhi", "lower", "upper"]:
            row[k] = float(row[k])
        assert 0 <= row["lower"] <= row["xhi"] <= row["upper"] <= 1
        assert row["z_min"] <= row["z"] <= row["z_max"]
    plt.style.use("apj")
    for file in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / file)
        )
    plt.rcParams.update(
        {"font.family": "Arial", "text.usetex": False, "font.size": 12, "mathtext.fontset": "stix"}
    )
    if a.output_dir is not None:
        out = a.output_dir
    assets = a.asset_dir
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 5.8), layout="constrained")
    result = {}
    for key, label, color in cases:
        hist = histories[key]
        z, q = np.asarray(hist["redshifts"]), np.asarray(hist["mean_xhii"])
        np.testing.assert_array_equal(z, manifest["redshifts"])
        assert np.all(np.diff(z) < 0) and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
        x = 1 - q
        ax.plot(
            z,
            x,
            color=color,
            lw=2.8,
            ls="--" if key.startswith("popiii_z") else "-",
            label=label,
            zorder=4 if key.startswith("popiii_z") else 3,
        )
        result[key] = {
            "ionized_fraction_crossings": {
                str(t): crossing(z, q, t) for t in [0.1, 0.5, 0.9, 0.99]
            },
            "at_observation_redshifts": {
                r["id"]: float(np.interp(r["z"], z[::-1], x[::-1]))
                if z.min() <= r["z"] <= z.max()
                else None
                for r in obs
            },
            "neutral_fraction_at_z": {
                str(zz): float(np.interp(zz, z[::-1], x[::-1]))
                for zz in [6.5, 7, 7.09, 7.54, 8, 9, 9.3, 10, 12]
            },
        }
    styles = {
        "dark_pixels": ("v", "#707070", "McGreer+2015: dark-pixel upper limits"),
        "Lya_EW": ("D", "#237b35", r"Mason+2018: galaxy Ly$\alpha$ EW"),
        "QSO_damping_wing": ("o", "#202020", "Davies+2018: quasar damping wings"),
        "JWST_damping_wing": ("s", "#8b4d9a", "Mason+2026: JWST damping wings"),
    }
    for method, (marker, color, label) in styles.items():
        group = [r for r in obs if r["method"] == method]
        for i, row in enumerate(group):
            if row["kind"] == "upper_limit":
                ax.errorbar(
                    row["z"],
                    row["xhi"],
                    yerr=0.065,
                    uplims=True,
                    fmt="none",
                    color=color,
                    capsize=4,
                    label=label if i == 0 else None,
                    zorder=5,
                )
            else:
                ax.errorbar(
                    row["z"],
                    row["xhi"],
                    yerr=[[row["xhi"] - row["lower"]], [row["upper"] - row["xhi"]]],
                    xerr=[[row["z"] - row["z_min"]], [row["z_max"] - row["z"]]],
                    fmt=marker,
                    color=color,
                    mfc="white",
                    ms=7,
                    capsize=3,
                    lw=1.4,
                    label=label if i == 0 else None,
                    zorder=5,
                )
    ax.set(
        xlim=(5.3, 15),
        ylim=(-0.015, 1.03),
        xlabel="Redshift z",
        ylabel=r"Volume-averaged neutral fraction $\langle x_{\rm HI}\rangle_V$",
    )
    ax.set_xticks([6, 7, 8, 9, 10, 12, 14])
    ax.grid(axis="y", alpha=0.15)
    if a.gated_run is not None:
        ax.axvline(10, ls=":", lw=1.2, color="#178479", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.96)
    ax.set_title("Reionization history compared with observational constraints")
    fig.savefig(out / f"{figure_name}.png", dpi=160)
    fig.savefig(out / f"{figure_name}.pdf")
    fig.set_size_inches(10.5, 4.3)
    fig.savefig(assets / f"{figure_name}.pdf")
    plt.close(fig)
    result["provenance"] = {
        "run": str(a.run.resolve()),
        "quantity": "volume-mean neutral fraction = 1 - mean_xhii",
        "source_model": manifest["source_manifest"]["resolved_model"],
        "gate_validation": gate_validation,
        "density_provenance_status": manifest["density_provenance_status"],
        "history_redshift_range": [float(z.min()), float(z.max())],
        "observations": obs,
        "hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
        },
        "limits": "No model extrapolation beyond saved snapshots. Error bars are reported intervals, arrows are upper bounds; JWST horizontal bars are wide sample bins, not z errors. No joint likelihood or significance inferred. This is the completed 100/100 Myr run, not an uncomputed Pop III 6 Myr run.",
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if a.gated_run is not None:
        keys = ["popii", "popii_popiii", gate_key]
        z50 = [result[k]["ionized_fraction_crossings"]["0.5"]["z"] for k in keys]
        x754 = [result[k]["neutral_fraction_at_z"]["7.54"] for k in keys]
        removed = "低" if high_only else "高"
        (assets / f"{figure_name}_summary.tex").write_text(
            f"排除{removed}红移 Pop III 后，半电离红移由 {z50[1]:.2f} 变为 {z50[2]:.2f}；"
            f"纯 Pop II 为 {z50[0]:.2f}。\\par\n"
            f"在 $z=7.54$，新实验的中性比例为 {x754[2]:.4f}，"
            "类星体观测推断为 $0.60^{+0.20}_{-0.23}$。\n"
        )
        lines = [
            r"\begin{center}\renewcommand{\arraystretch}{1.22}",
            r"\begin{tabular}{l c c c}\toprule",
            rf"指标 & Pop II & Pop II+III & III 仅在 $z{gate_tex}10$\\\midrule",
        ]
        lower_check_z = "7" if high_only else "9.3"
        for title, values in [
            ("半电离红移", z50),
            (
                r"99\% 电离红移",
                [result[k]["ionized_fraction_crossings"]["0.99"]["z"] for k in keys],
            ),
            (r"$\langle x_{\rm HI}\rangle_V$，$z=7.54$", x754),
            (
                rf"$\langle x_{{\rm HI}}\rangle_V$，$z={lower_check_z}$",
                [result[k]["neutral_fraction_at_z"][lower_check_z] for k in keys],
            ),
        ]:
            precision = 4 if "7.54" in title else 3
            lines.append(title + " & " + " & ".join(f"{v:.{precision}f}" for v in values) + r"\\")
        lines.append(r"\bottomrule\end{tabular}\end{center}")
        (assets / f"{figure_name}_table.tex").write_text("\n".join(lines) + "\n")
        if high_only:
            decline = gate_validation["post_switch_decline"]
            source_text = (
                f"关闭后，电离比例从 $z={decline['z_before']:.2f}$ 的 "
                f"{100 * decline['xhii_before']:.1f}\\% 回落至 "
                f"$z={decline['z_after']:.2f}$ 的 {100 * decline['xhii_after']:.1f}\\%; "
                "随后由 Pop II 供光继续推进。\\par\n"
            )
        else:
            share8 = gate_validation["popiii_photon_shares"]["8.0"]
            source_text = f"在 $z=8$，Pop III 仍提供 {100 * share8:.1f}\\% 的瞬时逃逸电离光子，低红移源并未被削弱。\\par\n"
        (assets / f"{figure_name}_source.tex").write_text(source_text)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
