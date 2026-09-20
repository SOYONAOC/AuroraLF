"""Compare calibrated models at matched Q_V using real slices and full 3D powers."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLE = [("#2863A5", "-"), ("#C65C28", "-"), ("#C65C28", "--")]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_case(case, reference, targets):
    run, data = Path(case["run"]), Path(case["matched_output"])
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    if manifest["status"] != "complete" or summary["status"] != "complete":
        raise ValueError("Incomplete spatial run or matched-stage analysis")
    if summary["targets"] != targets or summary["run"] != str(run.resolve()):
        raise ValueError("Wrong stage targets or source run")
    for key in [
        "grid",
        "physics",
        "power",
        "density_sha256",
        "redshifts",
        "code_sha256",
        "runner_sha256",
        "initialization",
    ]:
        if manifest[key] != reference[key]:
            raise ValueError(f"Unpaired simulation inputs: {key}")
    model = manifest["source_manifest"]["resolved_model"]
    for key, value in reference["source_manifest"]["resolved_model"].items():
        if key not in ["fesc_popii", "fesc_popiii"] and model[key] != value:
            raise ValueError(f"Changed source physics: {key}")
    if model["fesc_popii"] != case["fesc"] or model["fesc_popiii"] != case["fesc"]:
        raise ValueError("Wrong common escape fraction")
    for name, expected in summary["product_sha256"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Changed analysis product: {name}")
    for name in [run / "manifest.json", run / "histories.json"]:
        if digest(name) != summary["input_sha256"][str(name.resolve())]:
            raise ValueError("Spatial run changed since FFT postprocessing")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--slide-assets", action="store_true")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    out, data = Path(plan["plot_output"]), Path(plan["data_output"])
    cases, targets = plan["cases"], plan["targets"]
    if len(cases) != 3 or targets != [0.25, 0.5, 0.75, 0.9]:
        raise ValueError("Expected three calibrated models and four displayed stages")
    reference = json.loads((Path(cases[0]["run"]) / "manifest.json").read_text())
    summaries = [validate_case(c, reference, targets) for c in cases]
    labels = [
        rf"{'Pop II' if c['population'] == 'popii' else 'Pop II+III'}, $f_{{esc}}={c['fesc']:g}$"
        for c in cases
    ]
    stages = [s["cases"][c["population"]]["stages"] for c, s in zip(cases, summaries, strict=True)]
    spectra, slices, rows, ratios = {}, {}, [], {}
    for i, case in enumerate(cases):
        for j, target in enumerate(targets):
            st = stages[i][j]
            with np.load(Path(case["matched_output"]) / st["power_file"]) as f:
                spectra[i, j] = dict(f)
            with np.load(Path(case["matched_output"]) / st["nearest_snapshot"]["file"]) as f:
                slices[i, j] = dict(f)
            for quantity in ["delta2_xhi", "delta2_21_mk2", "delta2_brightness_over_t0"]:
                arr = spectra[i, j][quantity]
                if np.any(arr <= 0) or not np.isfinite(arr).all():
                    raise ValueError("Invalid power in a partially ionized field")
            lo, hi = st["bracket_snapshots"]
            w = st["upper_weight"]
            nearest = st["nearest_snapshot"]
            mean_t = (1 - w) * lo["mean_brightness_mk"] + w * hi["mean_brightness_mk"]
            full = (1 - w) * lo["fully_ionized_volume_fraction"] + w * hi[
                "fully_ionized_volume_fraction"
            ]
            rows.append(
                dict(
                    key=case["key"],
                    population=case["population"],
                    fesc=case["fesc"],
                    target=target,
                    z_estimated=st["estimated_redshift"],
                    z_nearest=st["nearest_redshift"],
                    q_nearest=st["nearest_mean_xhii"],
                    q_residual=st["residual"],
                    mean_brightness_mk=mean_t,
                    fully_ionized_volume_fraction=full,
                    nearest_mean_brightness_mk=nearest["mean_brightness_mk"],
                    nearest_fully_ionized_volume_fraction=nearest["fully_ionized_volume_fraction"],
                )
            )
            if i:
                np.testing.assert_array_equal(
                    spectra[0, j]["k_mpc_inverse"], spectra[i, j]["k_mpc_inverse"]
                )
                ratio = {"k": spectra[i, j]["k_mpc_inverse"]}
                for name in ["delta2_xhi", "delta2_21_mk2", "delta2_brightness_over_t0"]:
                    base, new = spectra[0, j], spectra[i, j]
                    ratio[name] = new[name] / base[name]
                    ratio[name + "_min"] = new[name + "_bracket_min"] / base[name + "_bracket_max"]
                    ratio[name + "_max"] = new[name + "_bracket_max"] / base[name + "_bracket_min"]
                ratios[i, j] = ratio
    plt.style.use("apj")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10.5})
    out.mkdir(parents=True, exist_ok=True)
    names = []

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=150)
        names.append(name)
        plt.close(fig)

    for j, target in enumerate(targets):
        fig, axes = plt.subplots(
            2, 3, figsize=(11.8, 7.3), layout="constrained", sharex=True, sharey=True
        )
        vmax = max(float(slices[i, j]["brightness_slice_mk"].max()) for i in range(3))
        for row, (quantity, cmap, maximum, title) in enumerate(
            [
                ("xhii_slice", "magma", 1.0, r"Ionized fraction $x_{HII}$"),
                ("brightness_slice_mk", "viridis", vmax, r"21 cm brightness $\delta T_b$ [mK]"),
            ]
        ):
            for i in range(3):
                ax = axes[row, i]
                st = stages[i][j]
                im = ax.imshow(
                    slices[i, j][quantity].T,
                    origin="lower",
                    extent=(0, 300, 0, 300),
                    vmin=0,
                    vmax=maximum,
                    cmap=cmap,
                    interpolation="nearest",
                )
                if row == 0:
                    ax.set_title(
                        labels[i]
                        + "\n"
                        + rf"$z={st['nearest_redshift']:.2f},\ Q_V={st['nearest_mean_xhii']:.4f}$",
                        fontsize=12,
                    )
                ax.set(xticks=[0, 100, 200, 300], yticks=[0, 100, 200, 300])
                if row == 1:
                    ax.set_xlabel("y [cMpc]")
                if i == 0:
                    ax.set_ylabel("z coordinate [cMpc]")
            fig.colorbar(im, ax=axes[row, :], shrink=0.9, pad=0.02, label=title)
        save(fig, f"fields_q{target:.2f}")

    for quantity, label, name in [
        ("delta2_xhi", r"$\Delta^2_{x_{HII}}$ [dimensionless]", "ionization_power"),
        ("delta2_21_mk2", r"$\Delta^2_{21}$ [mK$^2$]", "brightness_power"),
    ]:
        fig, axes = plt.subplots(
            2, 2, figsize=(11.5, 6.5), layout="constrained", sharex=True, sharey=True
        )
        for j, (ax, target) in enumerate(zip(axes.flat, targets, strict=True)):
            for i, (color, ls) in enumerate(STYLE):
                v = spectra[i, j]
                k = v["k_mpc_inverse"]
                ax.loglog(k, v[quantity], ls, color=color, lw=2, label=labels[i])
                ax.fill_between(
                    k,
                    v[quantity + "_bracket_min"],
                    v[quantity + "_bracket_max"],
                    color=color,
                    alpha=0.15,
                )
            ax.set(title=rf"$Q_V={target:.2f}$", ylabel=label)
            ax.grid(alpha=0.15)
        for ax in axes[1]:
            ax.set_xlabel(r"$k$ [cMpc$^{-1}$]")
        axes[0, 0].legend(fontsize=10)
        save(fig, name)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.5), layout="constrained", sharex=True, sharey=True
    )
    colors = {1: "#C65C28", 2: "#328575"}
    for j, (ax, target) in enumerate(zip(axes.flat, targets, strict=True)):
        ax.axhline(1, color="0.4", lw=1, ls=":")
        for i in [1, 2]:
            v = ratios[i, j]
            ax.semilogx(
                v["k"],
                v["delta2_21_mk2"],
                color=colors[i],
                lw=2,
                label=rf"$f_{{esc}}={cases[i]['fesc']:g}$: 21 cm",
            )
            ax.semilogx(
                v["k"],
                v["delta2_brightness_over_t0"],
                "--",
                color=colors[i],
                lw=1.6,
                label=rf"$f_{{esc}}={cases[i]['fesc']:g}$: divide by $T_0^2$",
            )
            ax.fill_between(
                v["k"], v["delta2_21_mk2_min"], v["delta2_21_mk2_max"], color=colors[i], alpha=0.12
            )
        ax.set(title=rf"$Q_V={target:.2f}$", ylabel="Mixed / Pop II power")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=10.5)
    axes[0, 0].set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel(r"$k$ [cMpc$^{-1}$]")
    save(fig, "brightness_ratio")

    # Report a real k bin near an explicitly chosen display scale, without rebinning.
    metrics = []
    for j, target in enumerate(targets):
        k = spectra[0, j]["k_mpc_inverse"]
        ix = int(np.argmin(abs(np.log(k / 0.1))))
        for i in [1, 2]:
            v = ratios[i, j]
            metrics.append(
                dict(
                    key=cases[i]["key"],
                    target=target,
                    k=float(k[ix]),
                    ionization_ratio=float(v["delta2_xhi"][ix]),
                    brightness_ratio=float(v["delta2_21_mk2"][ix]),
                    brightness_ratio_bracket=[
                        float(v["delta2_21_mk2_min"][ix]),
                        float(v["delta2_21_mk2_max"][ix]),
                    ],
                    brightness_over_t0_ratio=float(v["delta2_brightness_over_t0"][ix]),
                    brightness_ratio_range=[
                        float(v["delta2_21_mk2"].min()),
                        float(v["delta2_21_mk2"].max()),
                    ],
                )
            )
    result = dict(
        status="complete",
        targets=targets,
        stage_records=rows,
        power_comparisons=metrics,
        assumptions={
            "matching": "Nearest real 300^3 snapshots for maps; spectra and scalar summaries linearly interpolated in Q_V between bracketing snapshots. No interpolated/rescaled spatial fields. Targets are display choices.",
            "slices": "Same comoving x=150.5 Mpc plane, 300 cMpc box, shared color range within each stage and field quantity.",
            "power": "3D mean-subtracted absolute fluctuations, not fractional fluctuations. xHI and xHII power are identical for k>0. P binning retains conjugate-mode counts; Delta2=k^3 P/(2pi^2).",
            "brackets": "Envelope of adjacent snapshots, not uncertainty/confidence/interpolation-error bounds.",
            "brightness": "Ts>>Tcmb; no RSD, light cone, instrument, foregrounds or thermal noise. Matter density clipped to [-.95,6] traces baryons.",
            "normalization": "Divide each endpoint power by its own T0(z)^2 before interpolation; this removes only the explicit brightness prefactor. Density and morphology still correspond to different redshifts.",
            "limits": "No bubble-size or connectivity measurement. No spatial-resolution convergence or observational detectability claim. Legacy density cosmology remains unverified; overlapping regions not globally photon conserving.",
        },
        input_sha256={
            str(p): digest(p)
            for p in [a.plan, Path(__file__)]
            + [Path(c["matched_output"]) / "summary.json" for c in cases]
        },
        plots=names,
    )
    (data / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (data / "stages.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if a.slide_assets:
        deck = ROOT / "slides/popiii_heii_pisn_complete_20260916"
        assets = deck / "assets"
        for name in names:
            shutil.copy2(out / f"{name}.pdf", assets / f"calibrated_matched_{name}.pdf")
        lines = [
            r"\begin{center}\setlength{\tabcolsep}{5mm}\begin{tabular}{lrrrr}\toprule",
            r"目标 $Q_V$ & 纯 II 的 $z$ & 混合 0.064 的 $z$ & 混合 0.0725 的 $z$ & 最大 $|\Delta Q_V|$ \\",
            r"\midrule",
        ]
        for j, t in enumerate(targets):
            zz = [stages[i][j]["nearest_redshift"] for i in range(3)]
            res = max(abs(stages[i][j]["residual"]) for i in range(3))
            lines.append(f"{t:.2f} & {zz[0]:.2f} & {zz[1]:.2f} & {zz[2]:.2f} & {res:.4f}" + r" \\")
        lines += [r"\bottomrule\end{tabular}\end{center}"]
        (deck / "matched_stage_table.tex").write_text("\n".join(lines) + "\n")
        lines = [
            r"\begin{center}\setlength{\tabcolsep}{5mm}\begin{tabular}{llrrrr}\toprule",
            r"$Q_V$ & 混合 $f_{\rm esc}$ & $R_{x_{\rm HII}}$ & $R_{21}$ & 去 $T_0^2$ 后 & 相邻快照范围 \\",
            r"\midrule",
        ]
        for m in metrics:
            c = next(c for c in cases if c["key"] == m["key"])
            lo, hi = m["brightness_ratio_bracket"]
            lines.append(
                f"{m['target']:.2f} & {c['fesc']:g} & {m['ionization_ratio']:.3f} & {m['brightness_ratio']:.3f} & {m['brightness_over_t0_ratio']:.3f} & {lo:.3f}--{hi:.3f}"
                + r" \\"
            )
        lines += [
            r"\bottomrule\end{tabular}\end{center}",
            f"此表取 $k={metrics[0]['k']:.3f}\\,\\mathrm{{cMpc}}^{{-1}}$ 的真实谱箱；只作为展示尺度。",
        ]
        (deck / "matched_power_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
