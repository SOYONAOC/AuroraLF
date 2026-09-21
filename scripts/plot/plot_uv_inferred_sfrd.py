"""Plot the audited UV-inferred SFRD with published high-z uncertainties."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/sfrd_fraction_audit_20260917/comparison.json"
PAPER = ROOT / "external_data/literature_sources/papers/Donnan2024UVLF/paper.pdf"
OUT = ROOT / "outputs/uv_inferred_sfrd_20260917"
ASSET = ROOT / "slides/atomic_crossing_z6/assets/uv_inferred_sfrd.pdf"


def main():
    rows = json.loads(INPUT.read_text())["uv_inferred"]
    z = np.array([r["z"] for r in rows])
    obs = np.array([r["observed_uv_sfrd"] for r in rows])
    model = np.array([r["models"]["eps0.03_dust"]["uv_inferred_sfrd"] for r in rows])
    np.testing.assert_array_equal(z, [8, 12.5, 14.5])
    assert np.all(np.isfinite(obs) & (obs > 0))
    assert np.all(np.isfinite(model) & (model > 0))
    # Donnan et al. 2024, Table 3, PDF page 10: log10(rho_UV) errors.
    log_rho = np.array([24.64, 23.92])
    lower_dex = np.array([0.32, 0.81])
    upper_dex = np.array([0.18, 0.27])
    np.testing.assert_allclose(obs[1:], 1.15e-28 * 10**log_rho)
    lower = obs[1:] * 10**-lower_dex
    upper = obs[1:] * 10**upper_dex
    yerr = np.array([obs[1:] - lower, upper - obs[1:]])
    np.testing.assert_allclose(model / obs, [r["ratio_total_dust"] for r in rows])

    plt.style.use("apj")
    font_manager.fontManager.addfont(
        "/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
    )
    plt.rcParams.update({"font.family": "Arial", "font.size": 14, "text.usetex": False})
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.96)
    ax.set_yscale("log")
    ax.scatter(
        z,
        model,
        color="#d34935",
        marker="s",
        s=72,
        label="AuroraLF: Pop II + III, with dust",
        zorder=4,
    )
    ax.scatter(
        z[:1],
        obs[:1],
        color="#2460a7",
        marker="D",
        s=68,
        label="Donnan+23: DPL central fit",
        zorder=4,
    )
    ax.errorbar(
        z[1:],
        obs[1:],
        yerr=yerr,
        fmt="o",
        color="#2460a7",
        markersize=8,
        capsize=5,
        elinewidth=1.7,
        label="Donnan+24: Table 3",
        zorder=3,
    )
    # Preserve the original Fig. 8 distinction for the tentative z=14.5 estimate.
    ax.plot(z[-1], obs[-1], "o", mfc="white", mec="#2460a7", ms=8, zorder=4)
    for zz, mm, oo in zip(z, model, obs, strict=True):
        ax.plot([zz, zz], [oo, mm], color="0.6", lw=1.1, ls=":", zorder=1)
        ax.annotate(
            f"{mm / oo:.2f}" + r"$\times$",
            (zz, mm),
            xytext=(12, 4),
            textcoords="offset points",
            color="#a63727",
            fontsize=14,
        )
    ax.set(
        xlim=(7.3, 15.4),
        ylim=(1e-5, 1.4e-2),
        xlabel="Redshift z",
        ylabel=r"UV-inferred SFRD [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{cMpc}^{-3}$]",
    )
    ax.set_xticks([8, 9, 10, 11, 12.5, 14.5])
    ax.grid(axis="y", which="major", color="0.90", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.text(
        0.03,
        0.08,
        r"$M_{\rm UV}\leq -17$; same UV-to-SFR conversion",
        transform=ax.transAxes,
        fontsize=13,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET)
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "input": str(INPUT.relative_to(ROOT)),
                "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                "paper": str(PAPER.relative_to(ROOT)),
                "paper_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest(),
                "errors_source": "Donnan2024 Table3, PDF page10, arXiv:2403.03171v3",
                "redshifts": z.tolist(),
                "observation": obs.tolist(),
                "model": model.tolist(),
                "highz_error_lower_dex": lower_dex.tolist(),
                "highz_error_upper_dex": upper_dex.tolist(),
                "z8_error": "Not reconstructed; diamond is central DPL fit, not zero uncertainty",
                "model_error": "Not propagated; model symbols show central values only",
                "connections": "Vertical dotted segments compare model and data at identical redshift",
                "checks": "Central values and ratios equal audited table; exact dex-to-linear errors",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
