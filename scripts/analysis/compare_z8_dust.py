"""Fixed-intrinsic-UVLF dust sensitivity; literature recipes, no production edits.

TNG Model A: Vogelsberger+2020 eqs. 3,6,7 and Table 3 (z=8).
REBELS: Bowler+2024 (arXiv:2309.17386v2), section 4.4.
All but TNG keep the production Williams beta(Mobs,z) to isolate A(beta).
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.optimize import brentq

from auroralf.uvlf.dust import compute_dust_attenuated_uvlf

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/uvlf_efficiency_scan_20260914_64x"
OUT = ROOT / "outputs/dust_z8_20260917"
ASSETS = ROOT / "slides/atomic_crossing_z6/assets"
# intercept of beta at Mobs=-19.5, slope, A intercept, A slope, sigma_beta
MODELS = {
    "current": (-2.21, -0.146, 4.85, 2.10, 0.0),
    "meurer": (-2.21, -0.146, 4.43, 1.99, 0.0),
    "smc": (-2.21, -0.146, 2.45, 1.10, 0.0),
    "rebels23": (-2.21, -0.146, 2.11 * 2.3, 2.11, 0.0),
    "rebels25": (-2.21, -0.146, 1.38 * 2.5, 1.38, 0.0),
    "tng_a": (-2.66, -0.34, 4.43, 1.99, 0.34),
}
LABELS = {
    "current": "Current: Williams + Koprowski",
    "meurer": "Meurer 1999",
    "smc": "SMC-like",
    "rebels23": r"REBELS: $\beta_0=-2.3$",
    "rebels25": r"REBELS: $\beta_0=-2.5$",
    "tng_a": "TNG Model A (UVLF-calibrated)",
}
COLORS = {
    "current": "#18599c",
    "meurer": "#9b672e",
    "smc": "#999999",
    "rebels23": "#29965d",
    "rebels25": "#a054a1",
    "tng_a": "#d44d35",
}


def attenuation(q, name, delta=0.0):
    b0, slope, c0, c1, sigma = MODELS[name]
    raw = c0 + c1 * (b0 + slope * (q + 19.5)) + 0.2 * np.log(10) * c1**2 * sigma**2
    return np.maximum(raw, 0.0) + delta, np.where(raw > 0, 1 - c1 * slope, 1.0)


def mapped(q, x, phi, name, delta=0.0):
    if not np.all(np.isfinite(phi) & (phi > 0)):
        raise ValueError("Invalid intrinsic LF")
    a, jac = attenuation(q, name, delta)
    mint = q - a
    if min(q.min(), mint.min()) < x.min() or max(q.max(), mint.max()) > x.max():
        raise ValueError("Mapping outside saved LF support")
    bare = 10 ** np.interp(q, x, np.log10(phi))
    raw = 10 ** np.interp(mint, x, np.log10(phi)) * jac
    return np.minimum(raw, bare)


def average(mag, half, x, phi, name, delta=0.0, n=2049):
    q = np.linspace(mag - half, mag + half, n)
    return np.trapezoid(mapped(q, x, phi, name, delta), q) / (2 * half)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = RUN / "z8.npz"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert sha == manifest["products"][str(source)]
    obs_path = ROOT / "external_data/observations/uvlf/current_z6_z8_z10.json"
    obs = [d for d in json.loads(obs_path.read_text())["datasets"] if d["z"] == 8]
    with np.load(source) as data:
        x = (data["bin_edges"][1:] + data["bin_edges"][:-1]) / 2
        curves = {k: data[k] for k in ("baseline", "eps0.03")}
    q = np.linspace(-23.4, -18.0, 1081)
    # Independent agreement with the live production mapping.
    for phi in curves.values():
        prod = compute_dust_attenuated_uvlf(x, phi, 8, muv_obs=q)["phi_obs"]
        np.testing.assert_allclose(mapped(q, x, phi, "current"), prod, rtol=2e-13)
    # Verify the change-of-variable Jacobian conserves counts before the cap.
    from scipy.integrate import quad

    for name in MODELS:
        lo, hi = -23.4, -21.4
        a, jac = attenuation(np.array([lo, hi]), name)
        assert (a > 0).all() and (jac > 0).all()
        phi = curves["eps0.03"]
        mint = np.array([lo, hi]) - a

        def f(m):
            return 10 ** np.interp(m, x, np.log10(phi))

        lhs = quad(
            lambda m: f(m - attenuation(np.array(m), name)[0]) * jac[0],
            lo,
            hi,
            points=[v for v in q if lo < v < hi],
            epsabs=1e-14,
            limit=1000,
        )[0]
        rhs = quad(f, *mint, points=x[(x > mint[0]) & (x < mint[1])], epsabs=1e-14)[0]
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)
    rows = []
    for d in obs:
        for i, (mag, half, value) in enumerate(
            zip(d["muverr"], d["mag_err"], d["phierr"], strict=True)
        ):
            r = dict(
                dataset=d["label"],
                Muv=mag,
                half_width=half,
                observed=value,
                error_minus=d["phi_err_lo"][i],
                error_plus=d["phi_err_up"][i],
                models={},
            )
            for name in MODELS:
                predictions = {}
                for mode, phi in curves.items():
                    fine = average(mag, half, x, phi, name)
                    coarse = average(mag, half, x, phi, name, n=1025)
                    np.testing.assert_allclose(fine, coarse, rtol=2e-3)
                    predictions[mode] = dict(phi=fine, ratio=fine / value)
                r["models"][name] = dict(
                    A_center=float(attenuation(np.array(mag), name)[0]), **predictions
                )
            # Per-bin added screen: diagnostic inversion, not an independent model.
            if "Bowler" in d["label"]:
                r["required_extra_A"] = {}
                for mode, phi in curves.items():
                    low = mag - half
                    maxdelta = (
                        low - float(attenuation(np.array(low), "current")[0]) - x.min() - 1e-6
                    )
                    delta = brentq(
                        lambda a: average(mag, half, x, phi, "current", a) - value, 0, maxdelta
                    )
                    r["required_extra_A"][mode] = dict(
                        delta=delta, A_center=delta + r["models"]["current"]["A_center"]
                    )
            rows.append(r)
    sources = {
        "current": "https://doi.org/10.1093/mnras/sty1527",
        "tng_a": "https://doi.org/10.1093/mnras/staa137",
        "rebels": "https://arxiv.org/html/2309.17386v2#S4.SS4",
        "meurer_smc": "Vogelsberger+2020 section 3.2.1, citing Meurer99 and Bouwens16",
    }
    summary = dict(
        z=8,
        epsilon=0.03,
        source=str(source),
        sha256=sha,
        observations_sha256=hashlib.sha256(obs_path.read_bytes()).hexdigest(),
        recipes=MODELS,
        sources=sources,
        rows=rows,
        scope="Fixed intrinsic LF; deterministic mappings with production cap. TNG effective mean correction includes sigma_beta=0.34, not a scatter convolution. Same Williams beta relation for all other variants. A1600 approximated as A1500. Required delta fitted independently to each Bowler central value; no likelihood significance.",
    )
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.style.use("apj")
    for name in ["Arial.TTF", "Arialbd.TTF", "Ariali.TTF"]:
        font_manager.fontManager.addfont(
            "/home/zhuhourui/.local/share/fonts/microsoft-academic/" + name
        )
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 13, "text.usetex": False, "mathtext.fontset": "stix"}
    )
    fig, (ax, aa) = plt.subplots(
        1, 2, figsize=(13.6, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained"
    )
    for name in MODELS:
        style = "--" if name in ["meurer", "smc", "rebels25"] else "-"
        ax.plot(
            q,
            mapped(q, x, curves["eps0.03"], name),
            color=COLORS[name],
            ls=style,
            lw=2,
            label=LABELS[name],
        )
        aa.plot(q, attenuation(q, name)[0], color=COLORS[name], ls=style, lw=2)
    for j, d in enumerate(obs):
        ax.errorbar(
            d["muverr"],
            d["phierr"],
            xerr=d["mag_err"],
            yerr=[d["phi_err_lo"], d["phi_err_up"]],
            fmt=["o", "s"][j],
            ms=5,
            color="black",
            mfc="white",
            label=d["label"].replace(" (1600 A)", ""),
        )
    bowler = [r for r in rows if "required_extra_A" in r]
    aa.plot(
        [r["Muv"] for r in bowler],
        [r["required_extra_A"]["eps0.03"]["A_center"] for r in bowler],
        ls="none",
        marker="*",
        ms=12,
        color="black",
        label="Required per bin (fitted diagnostic)",
    )
    ax.set(
        xlim=(-23.5, -18),
        ylim=(3e-8, 1e-2),
        yscale="log",
        xlabel=r"$M_{\rm UV}^{\rm obs}$",
        ylabel=r"$\phi\ [{\rm cMpc}^{-3}\,{\rm mag}^{-1}]$",
        title=r"$z=8$, fixed intrinsic Pop II + Pop III",
    )
    aa.set(
        xlim=(-23.5, -18),
        ylim=(0, 3.2),
        xlabel=r"$M_{\rm UV}^{\rm obs}$",
        ylabel=r"$A_{1600}\ [{\rm mag}]$",
        title="Dust attenuation at the same observed magnitude",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    aa.legend(loc="upper right", fontsize=10, frameon=False)
    fig.savefig(ASSETS / "z8_dust_alternatives.pdf")
    fig.savefig(OUT / "comparison.png", dpi=150)
    for r in bowler:
        print(
            r["Muv"],
            {n: round(v["eps0.03"]["ratio"], 3) for n, v in r["models"].items()},
            r["required_extra_A"],
        )
    print("Production agreement, bin convergence, Jacobian conservation and source hash passed.")


if __name__ == "__main__":
    main()
