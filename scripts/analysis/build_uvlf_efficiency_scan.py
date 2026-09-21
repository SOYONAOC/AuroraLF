"""Background UVLF efficiency scan using saved halo light, followed by slide export.

Run from the repository root with --plan <frozen-input-plan.json>.
--validate-only checks input hashes and dependencies without running the scan.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from auroralf.experiments.uvlf_components import conditional_histograms  # noqa: E402
from scripts.analysis.analyze_random_q import observation_specs  # noqa: E402
from scripts.plot.plot_current_uvlf_dust import bin_predictions, mapped  # noqa: E402
from scripts.plot.plot_heii_pisn_uvlf import observations  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EFFICIENCIES = (0.01, 0.03, 0.1)
CASES = [
    ("baseline", "#286491", "--", "Pop II"),
    ("eps0.01", "#008060", "-", r"Pop II + III, $\epsilon_b=0.01$"),
    ("eps0.03", "#b86632", "-", r"Pop II + III, $\epsilon_b=0.03$"),
    ("eps0.1", "#7a5195", "-", r"Pop II + III, $\epsilon_b=0.1$"),
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def low_redshift(task):
    z, run, config = task
    with np.load(Path(run) / f"z{z}.npz") as data:
        edges, p2, p3, probability, weights = [
            data[k] for k in ("edges", "popii", "popiii", "probability", "weight_per_track")
        ]
        reference_phi, reference_se = data["phi"], data["se"]
    if config["efficiencies"] != [0.03] or config["lookback_myr"] != 100.0:
        raise ValueError("Unexpected source efficiency or UV window")
    result = {"bin_edges": edges, "baseline": reference_phi[0], "baseline_se": reference_se[0]}
    for epsilon in EFFICIENCIES:
        # Scale each halo's Pop III luminosity, then rebuild the paired total LF.
        # Multiplying an already-binned LF by epsilon would be incorrect.
        scaled_p3 = p3 if epsilon == 0.03 else p3 * (epsilon / 0.03)
        per_mass = conditional_histograms(p2, scaled_p3, probability, weights, edges)
        phi = per_mass.sum(axis=1)
        se = np.sqrt(len(p2) * per_mass.var(axis=1, ddof=1))
        np.testing.assert_allclose(phi[0], reference_phi[0], rtol=1e-11, atol=1e-100)
        if epsilon == 0.03:
            np.testing.assert_allclose(phi, reference_phi, rtol=1e-11, atol=1e-100)
            np.testing.assert_allclose(se, reference_se, rtol=1e-11, atol=1e-100)
        result[f"eps{epsilon:g}"] = phi[2]
        result[f"eps{epsilon:g}_se"] = se[2]
    print(f"z={z}: all efficiencies computed; 0.03 reproduces saved phi and MC SE", flush=True)
    return z, result


def make_figures(curves, plan, deck, output):
    plt.style.use("apj")
    fonts = Path("/home/zhuhourui/.local/share/fonts/microsoft-academic")
    for name in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF"):
        font_manager.fontManager.addfont(str(fonts / name))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    obs_root = ROOT / "external_data/observations/uvlf"
    low_obs = json.loads((obs_root / "current_z6_z8_z10.json").read_text())
    comparison = []
    for high, redshifts, filename in [
        (True, [12.5, 14.5], "uvlf_high_z"),
        (False, [6, 8], "uvlf_low_z_dust"),
    ]:
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.15), sharey=True)
        ymax = 2e-3 if high else 0.05
        for ax, z in zip(axs, redshifts, strict=True):
            data = curves[z]
            x = (data["bin_edges"][1:] + data["bin_edges"][:-1]) / 2
            lines = []
            for key, color, style, label in CASES:
                phi, se = data[key], data[key + "_se"]
                if high:
                    valid = (phi > 0) & (data[key + "_counts"] >= 8)
                    xx, yy = x, np.where(valid, phi, np.nan)
                    ax.fill_between(
                        x,
                        np.where(valid & (phi > se), phi - se, np.nan),
                        np.where(valid, phi + se, np.nan),
                        color=color,
                        alpha=0.12,
                    )
                else:
                    xx = np.linspace(-24, -16, 321)
                    yy = mapped(x, phi, z, xx)["phi_obs"]
                lines += ax.plot(xx, yy, color=color, ls=style, lw=2, label=label)
                in_view = (xx >= (-23 if high else -24)) & (xx <= (-17.5 if high else -16))
                ymax = max(ymax, float(np.nanmax(yy[in_view])) * 1.4)
            handles = []
            if high:
                for path, label, marker, color, flag in observation_specs(z):
                    with np.load(obs_root / path) as obs:
                        if flag and "is_upper_limit" not in obs:
                            raise ValueError(f"Missing upper-limit metadata: {path}")
                        handles.append(observations(ax, obs, label, marker, color))
            else:
                for i, obs in enumerate(o for o in low_obs["datasets"] if o["z"] == z):
                    label = obs["label"].replace(" (1600 A)", "")
                    handles.append(
                        observations(ax, obs, label, ["o", "s"][i], ["#222222", "#777777"][i])
                    )
                    for magnitude, dx, observed in zip(
                        obs["muverr"], obs["mag_err"], obs["phierr"], strict=True
                    ):
                        row = {"z": z, "dataset": label, "Muv": magnitude, "phi_observed": observed}
                        for key, *_ in CASES:
                            fine = bin_predictions(x, data[key], z, magnitude, dx, 1025)[1]
                            coarse = bin_predictions(x, data[key], z, magnitude, dx, 513)[1]
                            np.testing.assert_allclose(fine, coarse, rtol=1e-3)
                            row[key + "_dust_model_over_observed"] = float(fine / observed)
                        comparison.append(row)
            ax.legend(
                handles=handles,
                loc="upper left" if high else "lower right",
                fontsize=9.2,
                frameon=False,
            )
            ax.set(
                title=rf"$z={z}$",
                yscale="log",
                xlim=(-23, -17.5) if high else (-24, -16),
                xlabel=r"$M_{\rm UV}$" if high else r"$M_{\rm UV}^{\rm obs}$",
            )
            ax.grid(axis="y", alpha=0.12)
        axs[0].set_ylim(1e-8, ymax)
        axs[0].set_ylabel(r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$")
        fig.legend(
            lines,
            [line.get_label() for line in lines],
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.53, 1.01),
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.82, wspace=0.13)
        fig.savefig(deck / "assets" / f"{filename}.pdf")
        fig.savefig(output / f"{filename}.png", dpi=160)
        plt.close(fig)
    return comparison


def deck_source(plan, deck):
    source = Path(plan["source_deck"])
    tex = (source / "popiii_heii_pisn.tex").read_text()
    replacements = {
        str(source.relative_to(ROOT)) + "/assets/": str(deck.relative_to(ROOT)) + "/assets/",
        "$\\epsilon_b=0.03$ 明显提高亮端丰度，接近部分观测估计；$z\\simeq14$ 的不同样本仍有较大差异。": "固定成星临界质量分布，比较 $\\epsilon_b=0.01,\\ 0.03,\\ 0.1$；不同高红移样本仍有差异。",
        "橙线为逐晕合并 Pop II+III 光度后的 UVLF；阴影仅为 MC 标准误，本页未加尘埃。": "彩色实线为逐晕合并 Pop II+III 光度后的 UVLF；阴影仅为 MC 标准误，本页未加尘埃。",
        "两条模型曲线均已加入尘埃。在 $M_{\\rm UV}\\simeq-21$ 的观测分箱内，": "各模型曲线均已加入尘埃；彩色实线比较三档爆发效率，虚线为 Pop II 基线。",
        "总 UVLF / 观测为 $1.10$（$z=6$）、$4.05$（$z=8$）；$z=8$ 的 Pop II 基线也偏高。": "在 $M_{\\rm UV}\\simeq-21$ 处，$\\epsilon_b=0.03$ 的总 UVLF / 观测为 $1.10$、$4.05$。",
    }
    for old, new in replacements.items():
        if tex.count(old) != 1:
            raise ValueError(f"Expected exactly one slide-text target: {old}")
        tex = tex.replace(old, new)
    for old, new in plan.get("slide_text_overrides", {}).items():
        if tex.count(old) != 1:
            raise ValueError(f"Expected exactly one slide-text override: {old}")
        tex = tex.replace(old, new)
    return tex


def export_deck(plan, deck, output, summary_path):
    source = Path(plan["source_deck"])
    tex = deck_source(plan, deck)
    tex_path = deck / "popiii_heii_pisn.tex"
    tex_path.write_text(tex)
    for _ in range(2):
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={deck}",
                str(tex_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        (output / "build.log").write_text(result.stdout + result.stderr)
    if re.search(
        r"Overfull|Underfull|Missing character|undefined|Warning", result.stdout + result.stderr
    ):
        raise RuntimeError("Slide compilation has warnings; inspect build.log")
    pdf = deck / "popiii_heii_pisn.pdf"
    for page in (3, 4):
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-scale-to",
                "1500",
                "-png",
                str(pdf),
                str(output / f"slide-{page}"),
            ],
            check=True,
        )
    provenance = json.loads((source / "provenance.json").read_text())
    for name in ("uvlf_high_z.pdf", "uvlf_low_z_dust.pdf"):
        provenance["assets"][name] = {
            "source": str(Path(__file__).relative_to(ROOT)),
            "sha256": digest(deck / "assets" / name),
        }
    provenance.update(
        final_pdf_sha256=digest(pdf),
        source_tex_sha256=digest(tex_path),
        efficiency_scan_summary=str(summary_path),
        scope="Efficiency scan 0.01/0.03/0.1 on saved histories; separate deck export.",
    )
    write_json(deck / "provenance.json", provenance)
    write_json(
        deck / "qa.json",
        {
            "status": "compiled_visual_review_pending",
            "pages": 16,
            "latex_warnings": False,
            "visual_review": "Pending; previews exported",
            "pdf_sha256": digest(pdf),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    for command in ("xelatex", "pdftoppm"):
        if shutil.which(command) is None:
            raise FileNotFoundError(command)
    if args.validate_only:
        deck_source(plan, Path(plan["deck_output"]))
        for directory in plan["high_inputs"].values():
            with np.load(Path(directory) / "uvlf.npz") as data:
                for key, *_ in CASES:
                    for suffix in ("", "_cluster_se", "_counts"):
                        if key + suffix not in data:
                            raise ValueError(f"Missing cached efficiency: {key + suffix}")
        print(f"Validated {len(plan['input_sha256'])} frozen files; no science run performed")
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or "debug" in os.environ.get("SLURM_JOB_PARTITION", "").lower()
    ):
        raise RuntimeError("Non-debug SLURM allocation required")
    data_out, output, deck = [
        Path(plan[k]) for k in ("data_output", "diagnostic_output", "deck_output")
    ]
    data_out.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    deck.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(plan["source_deck"]) / "assets", deck / "assets")
    manifest = {
        "status": "running",
        "job_id": os.environ["SLURM_JOB_ID"],
        "started_unix": time.time(),
        "efficiencies": list(EFFICIENCIES),
        "plan": str(args.plan.resolve()),
        "scope": "Reuse saved histories; rescale per-halo Pop III luminosity before binning",
        "high_z_errors": "Saved mass-cluster MC SE",
        "low_z_errors": "Intrinsic MC SE saved; dust central curves only",
        "dust": "Existing mapping at z6,z8; no dust-free curves plotted",
    }
    manifest_path = data_out / "manifest.json"
    write_json(manifest_path, manifest)
    curves = {}
    for z in (12.5, 14.5):
        with np.load(Path(plan["high_inputs"][str(z)]) / "uvlf.npz") as data:
            result = {"bin_edges": data["bin_edges"]}
            for key, *_ in CASES:
                for suffix, source_suffix in [
                    ("", ""),
                    ("_se", "_cluster_se"),
                    ("_counts", "_counts"),
                ]:
                    result[key + suffix] = data[key + source_suffix]
            curves[z] = result
    low_run = Path(plan["low_input"])
    low_manifest = json.loads((low_run / "manifest.json").read_text())
    tasks = [
        (z, str(low_run), next(c for c in low_manifest["configs"] if c["z"] == z)) for z in (6, 8)
    ]
    with ProcessPoolExecutor(max_workers=min(2, int(os.environ["SLURM_CPUS_PER_TASK"]))) as pool:
        curves.update(pool.map(low_redshift, tasks))
    for z, data in curves.items():
        np.savez_compressed(data_out / f"z{z:g}.npz", **data)
    comparison = make_figures(curves, plan, deck, output)
    summary = {
        **manifest,
        "status": "analyzed",
        "observational_comparison": comparison,
        "input_sha256": plan["input_sha256"],
        "comparison_limits": "High-z retains historical Pop II1600/Pop III1500 proxy; low-z1500; original observation cosmology; no likelihood fit",
    }
    summary_path = data_out / "summary.json"
    write_json(summary_path, summary)
    export_deck(plan, deck, output, summary_path)
    manifest.update(
        status="complete",
        completed_unix=time.time(),
        visual_review="pending",
        products={str(p): digest(p) for p in data_out.glob("*.npz")},
        pdf=str(deck / "popiii_heii_pisn.pdf"),
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
