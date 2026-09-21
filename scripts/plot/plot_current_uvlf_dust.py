"""Apply the existing empirical dust-LF mapping to the z=6,8,10 diagnostic."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from auroralf.uvlf import compute_dust_attenuated_uvlf, uv_dust_attenuation
from auroralf.uvlf import dust as dust_module

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/uvlf_current_z6_z8_z10_stratified"
OUT = ROOT / "outputs/21cm_map/uvlf"
ASSETS = ROOT / "slides/assets/current_uvlf"
OBS = ROOT / "external_data/observations/uvlf/current_z6_z8_z10.json"
CASES = [("Pop II", "#286491", "-"), ("Pop III", "#7a5195", "--"), ("Total", "#b86632", "-")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapped(centers, phi, z, query):
    """Use the production formula and cap, requiring supported interpolation."""
    if not (np.isfinite(phi).all() and np.all(phi > 0)):
        raise ValueError("This diagnostic requires positive, finite intrinsic LF bins")
    result = compute_dust_attenuated_uvlf(centers, phi, z, muv_obs=query)
    required = np.r_[query, result["Muv_intrinsic"]]
    if required.min() < centers.min() or required.max() > centers.max():
        raise ValueError("Dust comparison would extrapolate beyond the computed intrinsic LF")
    if not np.all(result["phi_obs"] <= result["phi_nodust_obs"] * (1 + 1e-12)):
        raise AssertionError("Production dust cap was not preserved")
    return result


def bin_predictions(centers, phi, z, x, halfwidth, n):
    """Average differential LFs over the published magnitude bins."""
    query = np.linspace(x - halfwidth, x + halfwidth, n)
    result = mapped(centers, phi, z, query)
    return np.array(
        [
            np.trapezoid(result[key], query) / (2 * halfwidth)
            for key in ["phi_nodust_obs", "phi_obs"]
        ]
    )


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("UVLF source run is incomplete")
    for name, sha in manifest["products"].items():
        if digest(RUN / name) != sha:
            raise ValueError(f"Source product hash changed: {name}")
    observations = json.loads(OBS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    for name in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            str(Path("/home/zhuhourui/.local/share/fonts/microsoft-academic") / name)
        )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.8), sharey=True, layout="constrained")
    summary = {"redshifts": {}, "observational_comparison": []}
    for ax, z in zip(axs, [6, 8, 10], strict=True):
        with np.load(RUN / f"z{z}.npz") as data:
            edges, phi = data["edges"], data["phi"]
        centers = (edges[:-1] + edges[1:]) / 2
        query = np.linspace(-24, -16, 321)
        results = [mapped(centers, values, z, query) for values in phi]
        ax.plot(
            query,
            results[2]["phi_nodust_obs"],
            color="#999999",
            ls=":",
            lw=2,
            label="Total, no dust",
            zorder=2,
        )
        for result, (label, color, ls) in zip(results, CASES, strict=True):
            ax.plot(query, result["phi_obs"], color=color, ls=ls, lw=2.2, label=label, zorder=3)
        for index, obs in enumerate(o for o in observations["datasets"] if o["z"] == z):
            x, y, dx, lo, hi = [
                np.asarray(obs[key], float)
                for key in ["muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up"]
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
                raise ValueError(f"Invalid observations: {obs['label']}")
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
                zorder=5,
                label=obs["label"].replace(" (1600 A)", ""),
            )
            for magnitude, halfwidth, value in zip(x, dx, y, strict=True):
                estimates = np.array(
                    [bin_predictions(centers, p, z, magnitude, halfwidth, 513) for p in phi]
                )
                fine = np.array(
                    [bin_predictions(centers, p, z, magnitude, halfwidth, 1025) for p in phi]
                )
                np.testing.assert_allclose(estimates, fine, rtol=1e-3)
                summary["observational_comparison"].append(
                    {
                        "z": z,
                        "dataset": obs["label"],
                        "Muv": float(magnitude),
                        "bin_half_width": float(halfwidth),
                        "phi_observed": float(value),
                        "A_uv_at_bin_center": uv_dust_attenuation(magnitude, z),
                        "phi_bin_average_II_III_total_before_after": fine.tolist(),
                        "ratio_II_III_total_before_after": (fine / value).tolist(),
                    }
                )
        handles, labels = ax.get_legend_handles_labels()
        start = 0 if z == 6 else 4
        ax.legend(handles[start:], labels[start:], loc="lower right", fontsize=10, frameon=False)
        ax.set(
            xlim=(-24, -16),
            ylim=(1e-8, 0.08),
            yscale="log",
            title=rf"$z={z}$",
            xlabel=r"$M_{\rm UV}^{\rm obs}$",
        )
        ax.set_xticks([-24, -22, -20, -18, -16])
        ax.grid(axis="y", alpha=0.12)
        summary["redshifts"][str(z)] = {
            "Muv_obs": query.tolist(),
            "A_uv": results[2]["A_uv"].tolist(),
            "phi_dust_II_III_total": [r["phi_obs"].tolist() for r in results],
            "phi_nodust_II_III_total": [r["phi_nodust_obs"].tolist() for r in results],
            "cap_active_grid_points_II_III_total": [
                int(np.count_nonzero(r["phi_obs_eval"] > r["phi_nodust_obs"])) for r in results
            ],
        }
    axs[0].set_ylabel(r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$")
    fig.suptitle("Existing empirical dust mapping on; grey dotted: Total without dust", fontsize=12)
    fig.savefig(OUT / "uvlf_components_dust.png", dpi=170)
    fig.savefig(OUT / "uvlf_components_dust.pdf")
    fig.set_size_inches(12.6, 3.75)
    fig.savefig(ASSETS / "uvlf_components_dust.pdf")
    plt.close(fig)
    summary["provenance"] = {
        "run": str(RUN),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "plot_sha256": digest(Path(__file__)),
        "dust_sha256": digest(Path(dust_module.__file__)),
        "observations_sha256": digest(OBS),
        "apply_dust": True,
        "parameters": {"c0": 2.10, "c1": 4.85, "m0": -19.5},
        "scope": "Existing LF-level mapping separately on intrinsic II, III, total LFs; total intrinsic luminosities were summed within each halo before histogramming. Component mappings are diagnostics, not a common halo dust screen or a calibrated Pop III attenuation law.",
        "cap": "phi_obs=min(phi_obs_raw,phi_nodust_obs), preserved from production",
        "wavelength": "Use A1600 as an approximate A1500, as in the existing UVLF dust prescription; no new extinction curve fitted.",
        "uncertainty": "Central model curves only; original MC errors remain in source NPZ. Published observational errors shown. No propagated dust-law uncertainty or likelihood fit.",
        "calibration": "Williams2018 beta(z,Mobs); Koprowski2018 A1600(beta) calibrated at z=3-5, extrapolated here to z=6-10.",
        "source_history": "No new MAH/SSP calculation and no change to SFR, metallicity gates or ionizing emissivity.",
        "bin_quadrature_check": "All 29 observed bins: 513 vs 1025 grid points agree to relative 1e-3.",
    }
    (OUT / "dust_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    representatives = [
        min(
            (row for row in summary["observational_comparison"] if row["z"] == z),
            key=lambda row: abs(row["Muv"] + 21),
        )
        for z in [6, 8, 10]
    ]
    rows = [
        r"\begin{center}\renewcommand{\arraystretch}{1.15}\begin{tabular}{r r r r r}\toprule",
        r"$z$ & $M_{\rm UV}^{\rm obs}$ & $A_{\rm UV}$ [mag] & Total/观测：关闭 $\to$ 开启 & Pop II/观测：开启\\\midrule",
    ]
    for row in representatives:
        ratio = row["ratio_II_III_total_before_after"]
        rows.append(
            f"{row['z']} & {row['Muv']:.2f} & {row['A_uv_at_bin_center']:.2f} & "
            f"${ratio[2][0]:.2f} \\to {ratio[2][1]:.2f}$ & {ratio[0][1]:.2f}" + r"\\"
        )
    rows.append(r"\bottomrule\end{tabular}\end{center}")
    (ASSETS / "dust_summary.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps(representatives, indent=2))


if __name__ == "__main__":
    main()
