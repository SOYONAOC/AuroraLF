"""Replot saved UVLFs for the He II/PISN deck; no new formation histories."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.analysis.analyze_random_q import observation_specs
from scripts.plot.plot_current_uvlf_dust import bin_predictions, mapped

ROOT = Path(__file__).resolve().parents[2]
HIGH_ROOT = ROOT.parent / "AuroraLF-visbal-duty"
HIGH = {
    12.5: HIGH_ROOT / "data_save/random_q_z12p5_20260909",
    14.5: HIGH_ROOT / "data_save/random_q_combined_20260909_rev02",
}
LOW = ROOT / "data_save/uvlf_current_z6_z8_z10_stratified"
OBS = ROOT / "external_data/observations/uvlf"
ASSETS = ROOT / "slides/popiii_heii_pisn_20260913/assets"
OUT = ROOT / "outputs/popiii_heii_pisn_20260913"
SAVE = ROOT / "data_save/popiii_heii_pisn_uvlf_20260913"
BLUE, ORANGE = "#286491", "#b86632"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, sources, expected=None):
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    sources[str(path)] = value


def observations(ax, data, label, marker, color):
    x, y, dx, lo, hi = [
        np.asarray(data[k], float)
        for k in ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up")
    ]
    if not (
        x.shape == y.shape == dx.shape == lo.shape == hi.shape
        and np.isfinite(np.stack([x, y, dx, lo, hi])).all()
        and np.all(y > 0)
        and np.all(dx > 0)
        and np.all(lo >= 0)
        and np.all(hi >= 0)
        and np.all(lo <= y)
    ):
        raise ValueError(f"Invalid observations: {label}")
    if "is_upper_limit" in data and np.any(data["is_upper_limit"]):
        raise ValueError(f"New upper limits need explicit plotting: {label}")
    return ax.errorbar(
        x,
        y,
        xerr=dx,
        yerr=np.vstack([lo, hi]),
        fmt=marker,
        color=color,
        mfc="white",
        ms=4.5,
        capsize=2,
        elinewidth=1,
        zorder=5,
        label=label,
    )


def finish(fig, axs, name, handles, labels, high):
    for ax in axs:
        ax.set(
            yscale="log",
            xlim=(-23, -17.5) if high else (-24, -16),
            ylim=(1e-8, 2e-3) if high else (1e-8, 0.05),
            xlabel=r"$M_{\rm UV}$" if high else r"$M_{\rm UV}^{\rm obs}$",
        )
        ax.grid(axis="y", alpha=0.12)
    axs[0].set_ylabel(r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.53, 1.01),
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.82, wspace=0.13)
    fig.savefig(ASSETS / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def main():
    for p in (OUT, ASSETS, SAVE):
        p.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    for name in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF"):
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / name)
        )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    report = {"sources": {}, "high": {}, "low": {}, "bin_comparisons": []}
    sources = report["sources"]
    arrays = {}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.15), sharey=True)
    for ax, (z, run) in zip(axs, HIGH.items(), strict=True):
        summary = json.loads((run / "summary.json").read_text())
        record(run / "summary.json", sources)
        record(run / "uvlf.npz", sources)
        for path, sha in summary["manifests_sha256"].items():
            record(HIGH_ROOT / path, sources, sha)
        for path, sha in summary["observations_sha256"].items():
            record(Path(path), sources, sha)
        with np.load(run / "uvlf.npz") as data:
            x = (data["bin_edges"][1:] + data["bin_edges"][:-1]) / 2
            arrays[f"z{z}_muv"] = x
            lines = []
            for key, color, style, label in [
                ("baseline", BLUE, "--", "Pop II"),
                ("eps0.03", ORANGE, "-", r"Pop II + III, $\epsilon_b=0.03$"),
            ]:
                y, se, counts = [data[key + suffix] for suffix in ("", "_cluster_se", "_counts")]
                if not (np.isfinite(y).all() and np.isfinite(se).all() and np.all(se >= 0)):
                    raise ValueError(f"Invalid model LF: {run}/{key}")
                valid = (y > 0) & (counts >= 8)
                plotted = np.where(valid, y, np.nan)
                lines += ax.plot(x, plotted, color=color, ls=style, lw=2, label=label)
                ax.fill_between(
                    x,
                    np.where(valid & (y > se), y - se, np.nan),
                    np.where(valid, y + se, np.nan),
                    color=color,
                    alpha=0.16,
                )
                arrays[f"z{z}_{key}"] = y
                arrays[f"z{z}_{key}_se"] = se
                arrays[f"z{z}_{key}_valid"] = valid
        obs_handles = []
        for path, label, marker, color, needs_flag in observation_specs(z):
            with np.load(OBS / path) as data:
                if needs_flag and "is_upper_limit" not in data:
                    raise ValueError(f"Missing detection/limit metadata: {path}")
                obs_handles.append(observations(ax, data, label, marker, color))
        ax.legend(handles=obs_handles, loc="upper left", fontsize=9.2, frameon=False)
        ax.set_title(rf"$z={z}$", fontsize=15)
        report["high"][str(z)] = {
            "config": summary["config"],
            "runs": summary["runs"],
            "uncertainty": "Saved mass-cluster MC standard error; counts >= 8; no dust",
        }
    finish(fig, axs, "uvlf_high_z", lines, [line.get_label() for line in lines], True)

    manifest = json.loads((LOW / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Low-z run is incomplete")
    record(LOW / "manifest.json", sources)
    record(OBS / "current_z6_z8_z10.json", sources)
    low_obs = json.loads((OBS / "current_z6_z8_z10.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.15), sharey=True)
    for ax, z in zip(axs, [6, 8], strict=True):
        record(LOW / f"z{z}.npz", sources, manifest["products"][f"z{z}.npz"])
        with np.load(LOW / f"z{z}.npz") as data:
            x = (data["edges"][1:] + data["edges"][:-1]) / 2
            phi, se = data["phi"], data["se"]
        query = np.linspace(-24, -16, 321)
        lines = []
        for index, label, color in [(0, "Pop II", BLUE), (2, "Pop II + III", ORANGE)]:
            result = mapped(x, phi[index], z, query)
            lines += ax.plot(query, result["phi_obs"], color=color, lw=2, label=label + ", dust")
            arrays[f"z{z}_{index}_intrinsic_phi"] = phi[index]
            arrays[f"z{z}_{index}_intrinsic_se"] = se[index]
            arrays[f"z{z}_{index}_dust_phi"] = result["phi_obs"]
        arrays[f"z{z}_intrinsic_muv"], arrays[f"z{z}_observed_muv"] = x, query
        obs_handles = []
        for i, obs in enumerate(o for o in low_obs["datasets"] if o["z"] == z):
            label = obs["label"].replace(" (1600 A)", "")
            obs_handles.append(
                observations(ax, obs, label, ["o", "s"][i], ["#222222", "#777777"][i])
            )
            for m, dx, observed in zip(obs["muverr"], obs["mag_err"], obs["phierr"], strict=True):
                values = np.array([bin_predictions(x, phi[k], z, m, dx, 1025) for k in [0, 2]])
                coarse = np.array([bin_predictions(x, phi[k], z, m, dx, 513) for k in [0, 2]])
                np.testing.assert_allclose(values, coarse, rtol=1e-3)
                report["bin_comparisons"].append(
                    {
                        "z": z,
                        "dataset": label,
                        "muv": m,
                        "half_width": dx,
                        "observed_phi": observed,
                        "ratio_popii_total_by_nodust_dust": (values / observed).tolist(),
                    }
                )
        ax.legend(handles=obs_handles, loc="lower right", fontsize=10, frameon=False)
        ax.set_title(rf"$z={z}$", fontsize=15)
        report["low"][str(z)] = {
            "config": next(c for c in manifest["configs"] if c["z"] == z),
            "scope": manifest["scope"],
            "sampling": manifest["sampling"],
            "uncertainty": "Central dust curves only; no propagated MC or dust-law uncertainty",
        }
    finish(fig, axs, "uvlf_low_z_dust", lines, [line.get_label() for line in lines], False)
    report["dust"] = {
        "z": [6, 8],
        "law": "A1600=max(4.85+2.10*beta,0), A1500 approximated by A1600",
        "beta": "-0.09*z-1.49+(-0.007*z-0.09)*(Mobs+19.5)",
        "method": "Existing LF mapping with Jacobian; phi_obs=min(raw,nodust); no extrapolation in magnitude",
        "scope": "Empirical z=3-5 attenuation law extrapolated; separately mapped Pop II and total LFs, not a common halo dust screen",
        "references": ["https://arxiv.org/abs/1802.05272", "https://arxiv.org/abs/1801.00791"],
    }
    report["comparison_limits"] = [
        "High-z historical proxy: Pop II stellar 1600 A + Pop III total 1500 A; low-z both 1500 A",
        "Published observed LFs retain native cosmology, wavelength and redshift selection; visual comparison, no fitted likelihood",
        "No metallicity or z<10 Pop III UV emission gate introduced",
        "Total intrinsic LF uses halo-by-halo LII+LIII, not the sum of component LFs",
    ]
    for path in [
        Path(__file__),
        ROOT / "auroralf/uvlf/dust.py",
        ROOT / "scripts/plot/plot_current_uvlf_dust.py",
    ]:
        record(path, sources)
    np.savez_compressed(SAVE / "plotted_curves.npz", **arrays)
    report["products"] = {
        str(p): digest(p)
        for p in [
            SAVE / "plotted_curves.npz",
            ASSETS / "uvlf_high_z.pdf",
            ASSETS / "uvlf_low_z_dust.pdf",
        ]
    }
    (SAVE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for z in [6, 8]:
        row = min(
            (r for r in report["bin_comparisons"] if r["z"] == z), key=lambda r: abs(r["muv"] + 21)
        )
        print(json.dumps(row))


if __name__ == "__main__":
    main()
