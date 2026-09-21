"""Apply verified Pandeia noise bases to current He II populations; make diagnostic plots."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from scripts.analysis.heii_instrument_populations import OUT, ROOT


def quantiles(x, weights):
    order = np.argsort(x)
    x, weights = x[order], weights[order]
    return np.interp([0.16, 0.5, 0.84], (np.cumsum(weights) - 0.5 * weights) / weights.sum(), x)


def snr_and_limit(basis, flux, continuum, repeats):
    s = basis["signal_per_fref"]
    a = basis["variance_line_per_fref"]
    b = basis["variance_background"]
    c = basis["variance_continuum_per_cref"]
    ff, cc, fc = basis["variance_ff"], basis["variance_cc"], basis["variance_fc"]
    x, y = flux / basis["fref"], continuum / basis["cref_mjy"]
    variance = b + a * x + c * y + ff * x * x + cc * y * y + fc * x * y
    if np.any(variance <= 0):
        raise ValueError("Invalid extrapolated detector variance")
    snr = s * x * np.sqrt(repeats) / np.sqrt(variance)
    aa = s * s * repeats - 25 * ff
    bb = 25 * (a + fc * y)
    dd = 25 * (b + c * y + cc * y * y)
    if aa <= 0 or np.any(dd <= 0):
        raise ValueError("No finite positive 5-sigma limit")
    limit = (bb + np.sqrt(bb * bb + 4 * aa * dd)) / (2 * aa)
    return snr, limit * basis["fref"]


def main():
    destination = ROOT / "outputs/heii_instrument_20260918"
    destination.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "populations.json").read_text())
    records = []
    for pop in meta["records"]:
        z = pop["z"]
        with np.load(OUT / f"population_z{z:g}.npz") as p:
            data = {k: p[k] for k in p.files}
        w, lum = data["weight"], data["luminosity"]
        paths = sorted((OUT / "response").glob(f"z{z:g}_*_basis.json"))
        if len(paths) != 12:
            raise ValueError(f"Expected 12 completed responses at z={z}; got {len(paths)}")
        for path in paths:
            basis = json.loads(path.read_text())
            for hours in [10, 50]:
                # Repeat the identical readout ramp; report actual discrete exposure.
                repeats = max(1, round(hours * 3600 / basis["exposure_seconds"]))
                snr, limits = snr_and_limit(basis, data["flux"], data["fnu_mjy"], repeats)
                # Conservative sum of global pixel maxima. Above this bound the
                # unsaturated polynomial is not trusted; report the weight as a range.
                saturation_bound = (
                    basis["saturation_bound_zero"]
                    + data["flux"] / basis["fref"] * basis["saturation_bound_line"]
                    + data["fnu_mjy"] / basis["cref_mjy"] * basis["saturation_bound_cont"]
                )
                valid = saturation_bound < 1
                ambiguous = float(w[~valid].sum() / w.sum())
                hard = (snr >= 5) & valid
                fraction = float(w[hard].sum() / w.sum())
                probability = ndtr(snr - 5) * valid
                groups = np.bincount(data["cluster"], weights=w)
                residual = np.bincount(data["cluster"], weights=w * (hard - fraction))
                n = len(groups)
                se = float(np.sqrt(n / (n - 1) * np.sum(residual**2)) / w.sum())
                record = dict(
                    z=z,
                    mode=basis["mode"],
                    size_fwhm_arcsec=basis["size_fwhm_arcsec"],
                    line_fwhm_kms=basis["line_fwhm_kms"],
                    requested_hours=hours,
                    actual_hours=repeats * basis["exposure_seconds"] / 3600,
                    nexp=repeats,
                    fraction_expected_snr_ge5=fraction,
                    fraction_bright_unvalidated=ambiguous,
                    fraction_expected_snr_ge5_upper=fraction + ambiguous,
                    fraction_all_selected_lower=(1 - pop["unknown_weight_fraction"]) * fraction,
                    fraction_all_selected_upper=(1 - pop["unknown_weight_fraction"])
                    * (fraction + ambiguous)
                    + pop["unknown_weight_fraction"],
                    fraction_mc_se=se,
                    gaussian_detection_probability=float(w @ probability / w.sum()),
                    effective_mass_clusters=float(w.sum() ** 2 / np.sum(groups**2)),
                    luminosity_q16_q50_q84=quantiles(lum, w).tolist(),
                    selected_luminosity_q16_q50_q84=None
                    if not hard.any()
                    else quantiles(lum[hard], w[hard]).tolist(),
                    snr_q16_q50_q84=quantiles(snr, w).tolist(),
                    f5_q16_q50_q84=quantiles(limits, w).tolist(),
                    unknown_weight_fraction=pop["unknown_weight_fraction"],
                    selection=pop["selection"],
                )
                records.append(record)
    (OUT / "summary.json").write_text(
        json.dumps(
            dict(
                records=records,
                interpretation="S/N>=5 is an expected-SNR selection; Gaussian probability assumes known redshift and isolated line, not survey completeness",
                population_parameters="current plotted mu=0.5, epsilon=0.03; lens=1; no dust",
                continuum="flat fnu, per-object UV amplitude; local continuum fit noise propagated",
                integration="central 95% line-count pixels; continuum sidebands 3--7 core half-widths",
                instrument="Pandeia 2026.7 NIRSpec MOS, centered 1x3 slitlet q3_183_86; full-shutter extraction with local background subtraction",
                background="Pandeia minzodi benchmark",
                noise="Pandeia defaults; diagonal extracted spectral noise per STScI integrated-SNR method",
                limitations=[
                    "No OIII] blend or uncertain redshift",
                    "Same centered Gaussian morphology for continuum and line",
                    "No target-specific lens/shear/spatial offsets",
                    "Low-z young-bright sample is not LAP1 UV-matched",
                ],
                populations=meta,
            ),
            indent=2,
        )
    )
    flat = [
        {
            k: v
            for k, v in r.items()
            if not isinstance(v, (list, dict)) and not k.endswith("q16_q50_q84")
        }
        | {
            "median_snr": r["snr_q16_q50_q84"][1],
            "median_f5": r["f5_q16_q50_q84"][1],
            "intrinsic_median_luminosity": r["luminosity_q16_q50_q84"][1],
            "selected_median_luminosity": None
            if r["selected_luminosity_q16_q50_q84"] is None
            else r["selected_luminosity_q16_q50_q84"][1],
        }
        for r in records
    ]
    with (OUT / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 12,
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    colors = {"prism": "#236b9a", "medium": "#c46a28"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    base = [r for r in records if r["size_fwhm_arcsec"] == 0 and r["line_fwhm_kms"] == 50]
    for mode in colors:
        for hours in [10, 50]:
            rows = [r for r in base if r["mode"] == mode and r["requested_hours"] == hours]
            z = np.array([r["z"] for r in rows])
            color = colors[mode]
            marker = "o" if hours == 10 else "s"
            label = f"{'PRISM' if mode == 'prism' else 'R~1000'}, {hours} h"
            axs[0, 0].plot(
                0.164 * (1 + z),
                [r["f5_q16_q50_q84"][1] for r in rows],
                marker=marker,
                color=color,
                ls="--" if hours == 10 else "-",
                label=label,
            )
            axs[0, 1].errorbar(
                z + (0.05 if mode == "prism" else -0.05),
                [r["fraction_expected_snr_ge5"] for r in rows],
                yerr=[r["fraction_mc_se"] for r in rows],
                fmt=marker,
                color=color,
                mfc="white" if hours == 10 else color,
                label=label,
                capsize=2,
            )
        rows = [r for r in base if r["mode"] == mode and r["requested_hours"] == 50]
        for r in rows:
            q = r["selected_luminosity_q16_q50_q84"]
            if q is not None:
                axs[1, 0].errorbar(
                    r["z"] + (0.1 if mode == "prism" else -0.1),
                    q[1],
                    yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                    fmt="s",
                    color=colors[mode],
                    capsize=3,
                )
        rows = [
            r
            for r in records
            if r["mode"] == mode and r["z"] == 12.342 and r["requested_hours"] == 50
        ]
        for width in [50, 500]:
            subset = sorted(
                [r for r in rows if r["line_fwhm_kms"] == width],
                key=lambda r: r["size_fwhm_arcsec"],
            )
            axs[1, 1].plot(
                [r["size_fwhm_arcsec"] for r in subset],
                [r["fraction_expected_snr_ge5"] for r in subset],
                color=colors[mode],
                ls="-" if width == 50 else "--",
                marker="o",
                label=f"{mode}, {width} km/s",
            )
    for r in [r for r in base if r["mode"] == "prism" and r["requested_hours"] == 50]:
        axs[1, 0].plot(r["z"], r["luminosity_q16_q50_q84"][1], "kx", ms=8)
    axs[0, 0].set(
        xlabel="Observed He II wavelength [micron]",
        ylabel=r"5-sigma line flux [erg s$^{-1}$ cm$^{-2}$]",
        yscale="log",
        title="Median flux limit at each population's continuum",
    )
    axs[0, 1].set(
        xlabel="Redshift",
        ylabel="Fraction above expected S/N = 5",
        ylim=(-0.04, 1.04),
        title="Point source, 50 km/s; bars = sampling SE",
    )
    axs[1, 0].set(
        xlabel="Redshift",
        ylabel=r"Intrinsic He II luminosity [erg s$^{-1}$]",
        yscale="log",
        title="50 h: selected 16-84%; crosses = parent medians",
    )
    axs[1, 1].set(
        xlabel="Source FWHM [arcsec]",
        ylabel="Fraction above expected S/N = 5",
        ylim=(-0.04, 1.04),
        title="z = 12.342, UV-matched sample, 50 h",
    )
    axs[1, 1].set_xticks([0, 0.1, 0.2])
    for ax in [axs[0, 0], axs[1, 1]]:
        ax.legend(fontsize=9)
    fig.suptitle(
        "He II instrument experiment | Pandeia 2026.7 | current mu=0.5, epsilon=0.03\n"
        "No lensing or dust; low-z young-bright / high-z UV-matched populations\n"
        "Fractions conservatively exclude possible saturation; CSV records the bounds",
        fontsize=12,
    )
    fig.savefig(destination / "instrument_comparison.pdf")
    fig.savefig(destination / "instrument_comparison.png", dpi=150)
    print(
        json.dumps(
            [r for r in flat if r["size_fwhm_arcsec"] == 0 and r["line_fwhm_kms"] == 50], indent=2
        )
    )


if __name__ == "__main__":
    main()
