"""Plot same-halo Pop II, Pop III and total intrinsic 1500-A luminosity functions."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from auroralf.uvlf import uv_luminosity_to_muv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/21cm_map/uvlf"
ASSETS = ROOT / "slides/assets/current_uvlf"
OBS = ROOT / "external_data/observations/uvlf/current_z6_z8_z10.json"
CASES = [("Pop II", "#286491", "-"), ("Pop III", "#7a5195", "--"), ("Total", "#b86632", "-")]


def tex_number(value):
    coefficient, exponent = f"{value:.2e}".split("e")
    return "$" + coefficient + r"\times10^{" + str(int(exponent)) + "}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "data_save/uvlf_current_z6_z8_z10_stratified"
    )
    a = parser.parse_args()
    observations = json.loads(OBS.read_text())
    manifest = json.loads((a.run / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["redshifts"] == [6.0, 8.0, 10.0]
    for name, value in manifest["products"].items():
        assert hashlib.sha256((a.run / name).read_bytes()).hexdigest() == value
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    for name in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / name)
        )
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 14, "text.usetex": False, "mathtext.fontset": "stix"}
    )
    fig, axs = plt.subplots(
        1, 3, figsize=(12.6, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    summary = {}
    for ax, z in zip(axs, manifest["redshifts"], strict=True):
        with np.load(a.run / f"z{z:g}.npz") as d:
            data = dict(d)
        conditional = "probability" in data
        count_key = "unconditional_counts" if conditional else "counts"
        edges, phi, se, counts = [data[k] for k in ["edges", "phi", "se", count_key]]
        assert phi.shape == se.shape == counts.shape == (3, len(edges) - 1)
        assert np.isfinite(phi).all() and np.isfinite(se).all() and np.all(phi >= 0)
        centers = (edges[:-1] + edges[1:]) / 2
        for k, (label, color, ls) in enumerate(CASES):
            positive = phi[k] > 0
            ax.plot(
                centers,
                np.where(positive, phi[k], np.nan),
                color=color,
                ls=ls,
                lw=2.2,
                label=label,
                zorder=4 if k == 2 else 3,
            )
            ax.fill_between(
                centers,
                np.maximum(phi[k] - se[k], 1e-12),
                phi[k] + se[k],
                where=positive,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            # Open markers show bins where the mass-clustered MC error exceeds 30%.
            sparse = positive & (se[k] > 0.3 * phi[k]) & (phi[k] >= 1e-8)
            ax.plot(centers[sparse], phi[k, sparse], "o", ms=4, mfc="white", mec=color, zorder=5)
        for index, obs in enumerate(o for o in observations["datasets"] if o["z"] == z):
            x, y, dx, lo, hi = [
                np.asarray(obs[key], dtype=float)
                for key in ["muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up"]
            ]
            if not (
                x.shape == y.shape == dx.shape == lo.shape == hi.shape
                and np.isfinite(np.stack([x, y, dx, lo, hi])).all()
                and np.all(y > 0)
                and np.all(dx >= 0)
                and np.all(lo >= 0)
                and np.all(hi >= 0)
                and np.all(lo <= y)
            ):
                raise ValueError(f"Invalid observational table: {obs['label']}")
            # A lower error reaching zero stays zero; it is not an upper limit.
            ax.errorbar(
                x,
                y,
                xerr=dx,
                yerr=np.vstack([lo, hi]),
                fmt=["o", "s"][index],
                color=["#252525", "#666666"][index],
                mfc="white",
                ms=5,
                elinewidth=1.1,
                capsize=2,
                zorder=6,
                label=obs["label"].replace(" (1600 A)", ""),
            )
        handles, labels = ax.get_legend_handles_labels()
        start = 0 if z == 6 else 3
        ax.legend(handles[start:], labels[start:], loc="upper left", fontsize=10, frameon=False)
        ax.set(
            xlim=(-24, -16),
            ylim=(1e-8, 0.08),
            yscale="log",
            title=rf"$z={z:g}$",
            xlabel=r"$M_{\rm UV}$",
        )
        ax.set_xticks([-24, -22, -20, -18, -16])
        ax.grid(axis="y", alpha=0.12)
        p2, p3, w = data["popii"], data["popiii"], data["weight_per_track"]
        if conditional:
            p2 = np.broadcast_to(p2[:, :, None], p3.shape)
            probability = data["probability"]
            np.testing.assert_allclose(probability.sum(axis=-1), 1.0, atol=2e-15)
        else:
            probability = np.ones_like(p3)
        weighted = probability * w.reshape((-1,) + (1,) * (p3.ndim - 1))
        rho2, rho3 = [float(np.sum(v * weighted)) for v in [p2, p3]]
        brightness = {}
        for cut in [-22.0, -20.0, -18.0, -16.0]:
            densities = []
            for lum in [p2, p3, p2 + p3]:
                densities.append(float(np.sum((uv_luminosity_to_muv(lum) <= cut) * weighted)))
            assert densities[2] >= max(densities[:2]) * (1 - 1e-12)
            brightness[str(cut)] = dict(popii=densities[0], popiii=densities[1], total=densities[2])
        summary[str(z)] = dict(
            uv_luminosity_density_erg_s_hz_mpc3=[rho2, rho3, rho2 + rho3],
            popiii_uv_luminosity_fraction=rho3 / (rho2 + rho3),
            cumulative_number_density=brightness,
            unconditional_counts_in_display_range=counts[
                :, (centers >= -24) & (centers <= -16)
            ].tolist(),
            conditional_age_strata=conditional,
        )
    axs[0].set_ylabel(r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$")
    fig.suptitle("Model: intrinsic 1500 Å; observations: 1500 Å (Bouwens+21: 1600 Å)", fontsize=12)
    fig.savefig(OUT / "uvlf_components.png", dpi=170)
    fig.savefig(OUT / "uvlf_components.pdf")
    fig.set_size_inches(12.6, 3.75)
    fig.savefig(ASSETS / "uvlf_components.pdf")
    plt.close(fig)
    summary["provenance"] = dict(
        run=str(a.run.resolve()),
        manifest_sha256=hashlib.sha256((a.run / "manifest.json").read_bytes()).hexdigest(),
        plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quantity=manifest["definition"],
        source_prescription=manifest["source_prescription"],
        sampling=manifest["sampling"],
        observations=observations,
        observations_sha256=hashlib.sha256(OBS.read_bytes()).hexdigest(),
        uncertainty="Shaded +/- one clustered Monte Carlo standard error; open markers indicate fractional SE>30%; no smoothing, no observational confidence intervals",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [
        r"\begin{center}\renewcommand{\arraystretch}{1.2}\begin{tabular}{r r r r}\toprule",
        r"$z$ & Pop III 的 UV 光度密度占比 & $n_{\rm II}(M_{\rm UV}<-20)$ & $n_{\rm total}(M_{\rm UV}<-20)$\\\midrule",
    ]
    for z in manifest["redshifts"]:
        s = summary[str(z)]
        b = s["cumulative_number_density"]["-20.0"]
        rows.append(
            f"{z:g} & {100 * s['popiii_uv_luminosity_fraction']:.1f}"
            + r"\%"
            + f" & {tex_number(b['popii'])} & {tex_number(b['total'])}"
            + r"\\"
        )
    rows.append(r"\bottomrule\end{tabular}\end{center}")
    (ASSETS / "summary.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
