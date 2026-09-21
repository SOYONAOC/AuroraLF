"""Plot the saved Pop II baseline for the opening of slide deck AUR-S01."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.analysis.analyze_random_q import observation_specs
from scripts.plot.plot_current_uvlf_dust import mapped
from scripts.plot.plot_heii_pisn_uvlf import observations

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/uvlf_efficiency_scan_20260914_64x"
DECK = ROOT / "slides/popiii_heii_pisn_complete_20260916"
OUT = ROOT / "outputs/popii_motivation_20260916"
OBS = ROOT / "external_data/observations/uvlf"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("UVLF source is incomplete")
    provenance = {"sources": {}, "model": "saved baseline only; no parameter refit"}

    def record(path, expected=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        provenance["sources"][str(path.relative_to(ROOT))] = digest

    record(RUN / "manifest.json")
    record(OBS / "current_z6_z8_z10.json")
    low_obs = json.loads((OBS / "current_z6_z8_z10.json").read_text())["datasets"]
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
    for high, zs, name in [(False, [6, 8], "popii_low_z"), (True, [12.5, 14.5], "popii_high_z")]:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.15), sharey=True)
        for ax, z in zip(axes, zs, strict=True):
            path = RUN / f"z{z}.npz"
            record(path, manifest["products"][str(path)])
            with np.load(path) as data:
                x = (data["bin_edges"][:-1] + data["bin_edges"][1:]) / 2
                phi, se = data["baseline"], data["baseline_se"]
                if not (
                    np.isfinite(phi).all()
                    and np.isfinite(se).all()
                    and np.all(phi >= 0)
                    and np.all(se >= 0)
                ):
                    raise ValueError(f"Invalid baseline: {path}")
                if high:
                    valid = (phi > 0) & (data["baseline_counts"] >= 8)
                    xx, yy = x, np.where(valid, phi, np.nan)
                    ax.fill_between(
                        x,
                        np.where(valid & (phi > se), phi - se, np.nan),
                        np.where(valid, phi + se, np.nan),
                        color="#286491",
                        alpha=0.15,
                    )
                else:
                    xx = np.linspace(-24, -16, 321)
                    yy = mapped(x, phi, z, xx)["phi_obs"]
            (line,) = ax.plot(xx, yy, color="#286491", lw=2.3)
            handles = []
            if high:
                for source, label, marker, color, needs_flag in observation_specs(z):
                    record(OBS / source)
                    with np.load(OBS / source) as obs:
                        if needs_flag and "is_upper_limit" not in obs:
                            raise ValueError(f"Missing upper-limit flags: {source}")
                        handles.append(observations(ax, obs, label, marker, color))
            else:
                for i, obs in enumerate(o for o in low_obs if o["z"] == z):
                    handles.append(
                        observations(
                            ax,
                            obs,
                            obs["label"].replace(" (1600 A)", ""),
                            ["o", "s"][i],
                            ["#222222", "#777777"][i],
                        )
                    )
            ax.legend(
                handles=handles,
                loc="upper left" if high else "lower right",
                fontsize=9.2,
                frameon=False,
            )
            ax.set(
                title=rf"$z={z}$",
                yscale="log",
                ylim=(1e-8, 2e-3 if high else 0.05),
                xlim=(-23, -17.5) if high else (-24, -16),
                xlabel=r"$M_{\rm UV}$" if high else r"$M_{\rm UV}^{\rm obs}$",
            )
            ax.grid(axis="y", alpha=0.12)
        axes[0].set_ylabel(r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$")
        fig.legend(
            [line],
            ["Pop II only" if high else "Pop II only, dust"],
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.53, 1.01),
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.82, wspace=0.13)
        fig.savefig(DECK / "assets" / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=160)
        plt.close(fig)
    provenance["scope"] = [
        "Low-z: existing dust mapping, central curves only",
        "High-z: no dust; saved mass-cluster MC SE, counts>=8",
        "Original observational selections retained; no likelihood fit",
    ]
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
