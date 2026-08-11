"""Plot a schematic broadband-photometry forward-modeling workflow.

This is an explanatory figure, not the response curve or SED of a particular
survey.  It distinguishes the filter response across wavelength from the
measurement uncertainty of the final scalar photometric datum.
"""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "reproductions" / "arxiv_2608_05531"
STYLE_PATH = SCRIPT_DIR / "styles" / "apj.mplstyle"
MPL_CACHE_DIR = Path("/tmp/auroralf_filter_photometry_mplconfig")

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#9C5AA6"
RED = "#D55E00"
GRAY = "#5B6573"


def build_schematic_sed(wavelength_nm: np.ndarray) -> np.ndarray:
    """Return a positive, normalized toy galaxy SED in arbitrary units."""
    continuum = 0.78 + 0.23 * (wavelength_nm / 550.0) ** 0.65
    spectral_break = 0.12 * np.tanh((wavelength_nm - 445.0) / 16.0)
    emission_1 = 0.22 * np.exp(-0.5 * ((wavelength_nm - 520.0) / 9.0) ** 2)
    emission_2 = 0.30 * np.exp(-0.5 * ((wavelength_nm - 656.0) / 7.0) ** 2)
    absorption = -0.10 * np.exp(-0.5 * ((wavelength_nm - 590.0) / 15.0) ** 2)
    flux_density = continuum + spectral_break + emission_1 + emission_2 + absorption
    if np.any(flux_density <= 0.0) or not np.all(np.isfinite(flux_density)):
        raise ValueError("The schematic SED must be finite and strictly positive.")
    return flux_density


def build_schematic_response(wavelength_nm: np.ndarray) -> np.ndarray:
    """Return a smooth, asymmetric toy filter throughput."""
    gaussian_core = np.exp(-0.5 * ((wavelength_nm - 575.0) / 72.0) ** 2)
    blue_edge = 1.0 / (1.0 + np.exp(-(wavelength_nm - 415.0) / 10.0))
    red_edge = 1.0 / (1.0 + np.exp((wavelength_nm - 735.0) / 13.0))
    asymmetry = np.clip(1.0 + 0.10 * (wavelength_nm - 575.0) / 72.0, 0.0, None)
    response = gaussian_core * blue_edge * red_edge * asymmetry
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("The filter response must be finite and non-negative.")
    peak = float(np.max(response))
    if peak <= 0.0:
        raise ValueError("The filter response has zero total throughput.")
    return response / peak


def weighted_quantile(
    values: np.ndarray,
    normalized_kernel: np.ndarray,
    quantile: float,
) -> float:
    """Evaluate a wavelength quantile for a normalized integration kernel."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must lie in [0, 1], received {quantile}.")
    cumulative = np.zeros_like(values)
    cumulative[1:] = np.cumsum(
        0.5
        * (normalized_kernel[1:] + normalized_kernel[:-1])
        * np.diff(values)
    )
    if cumulative[-1] <= 0.0:
        raise ValueError("The normalized kernel has zero integrated weight.")
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, values))


def calculate_band_measurement(
    wavelength_nm: np.ndarray,
    flux_density: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Compute a photon-weighted band flux and wavelength summary.

    For a photon-counting detector the common factor 1/(hc) cancels between
    numerator and denominator, leaving the weight lambda*T(lambda).
    """
    if not (
        wavelength_nm.shape == flux_density.shape == response.shape
        and wavelength_nm.ndim == 1
    ):
        raise ValueError("Wavelength, SED, and response must be matching 1-D arrays.")
    if np.any(np.diff(wavelength_nm) <= 0.0):
        raise ValueError("Wavelength samples must be strictly increasing.")

    photon_weight = wavelength_nm * response
    denominator = float(np.trapezoid(photon_weight, wavelength_nm))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("The photon-weighted filter integral must be positive.")
    kernel = photon_weight / denominator
    normalization = float(np.trapezoid(kernel, wavelength_nm))
    if not np.isclose(normalization, 1.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(f"The integration kernel is not normalized: {normalization}.")

    band_flux = float(np.trapezoid(flux_density * kernel, wavelength_nm))
    effective_wavelength = float(np.trapezoid(wavelength_nm * kernel, wavelength_nm))
    lower_wavelength = weighted_quantile(wavelength_nm, kernel, 0.16)
    upper_wavelength = weighted_quantile(wavelength_nm, kernel, 0.84)
    return kernel, band_flux, effective_wavelength, lower_wavelength, upper_wavelength


def make_figure() -> tuple[plt.Figure, dict[str, float]]:
    """Create the three-stage explanatory figure and return its key numbers."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Required ApJ style is missing: {STYLE_PATH}")
    plt.style.use(STYLE_PATH)

    wavelength_nm = np.linspace(350.0, 850.0, 2501)
    flux_density = build_schematic_sed(wavelength_nm)
    response = build_schematic_response(wavelength_nm)
    kernel, model_band_flux, effective_wavelength, lambda_16, lambda_84 = (
        calculate_band_measurement(wavelength_nm, flux_density, response)
    )

    observed_sigma = 0.055
    observed_band_flux = model_band_flux + 1.35 * observed_sigma
    standardized_residual = (observed_band_flux - model_band_flux) / observed_sigma

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)
    ax_sed, ax_integral, ax_compare = axes

    # 1. The continuous model and the wavelength-dependent filter throughput.
    sed_line = ax_sed.plot(
        wavelength_nm,
        flux_density,
        color=BLUE,
        label=r"Continuous model $F_\lambda$",
        zorder=3,
    )[0]
    ax_sed.set(
        xlim=(350.0, 850.0),
        ylim=(0.54, 1.48),
        xlabel=r"Observed wavelength $\lambda$ [nm]",
        ylabel=r"Normalized flux density $F_\lambda$",
        title=r"1. SED passes through a filter",
    )
    ax_sed.grid(True, alpha=0.20)
    ax_sed_response = ax_sed.twinx()
    response_line = ax_sed_response.plot(
        wavelength_nm,
        response,
        color=ORANGE,
        linestyle="--",
        label=r"Filter response $T_b(\lambda)$",
        zorder=2,
    )[0]
    ax_sed_response.fill_between(
        wavelength_nm,
        0.0,
        response,
        color=ORANGE,
        alpha=0.13,
        zorder=1,
    )
    ax_sed_response.set_ylim(0.0, 1.08)
    ax_sed_response.set_ylabel(r"Throughput $T_b(\lambda)$", color=ORANGE)
    ax_sed_response.tick_params(axis="y", colors=ORANGE)
    ax_sed.legend(
        [sed_line, response_line],
        [sed_line.get_label(), response_line.get_label()],
        loc="upper left",
        fontsize=10,
    )
    ax_sed.text(
        0.04,
        0.06,
        r"The band samples a wavelength range,\\not a single wavelength.",
        transform=ax_sed.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.90},
    )

    # 2. Photon weighting and integration into one scalar model datum.
    kernel_display = kernel / np.max(kernel)
    contribution = flux_density * kernel
    contribution_display = contribution / np.max(contribution)
    ax_integral.plot(
        wavelength_nm,
        kernel_display,
        color=ORANGE,
        linestyle="--",
        label=r"Photon weight $\lambda T_b$",
    )
    ax_integral.plot(
        wavelength_nm,
        contribution_display,
        color=PURPLE,
        label=r"Detected contribution $F_\lambda\lambda T_b$",
    )
    ax_integral.fill_between(
        wavelength_nm,
        0.0,
        contribution_display,
        color=PURPLE,
        alpha=0.22,
    )
    ax_integral.axvline(
        effective_wavelength,
        color=GRAY,
        linestyle=":",
        linewidth=1.2,
    )
    ax_integral.set(
        xlim=(350.0, 850.0),
        ylim=(0.0, 1.16),
        xlabel=r"Observed wavelength $\lambda$ [nm]",
        ylabel="Relative weight / contribution",
        title=r"2. Weight and integrate",
    )
    ax_integral.grid(True, alpha=0.20)
    ax_integral.legend(loc="upper left", fontsize=9.5)
    ax_integral.text(
        0.50,
        0.08,
        (
            r"$F_b^{\rm model}="
            r"\frac{\int F_\lambda T_b(\lambda)\lambda\,d\lambda}"
            r"{\int T_b(\lambda)\lambda\,d\lambda}$"
        ),
        transform=ax_integral.transAxes,
        ha="center",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.93},
    )
    ax_integral.annotate(
        r"$\lambda_{\rm eff}$",
        xy=(effective_wavelength, 0.76),
        xytext=(effective_wavelength + 52.0, 0.92),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color=GRAY,
        fontsize=11,
    )

    # 3. Compare the scalar model prediction with the measured band flux.
    ax_compare.plot(
        wavelength_nm,
        flux_density,
        color=GRAY,
        alpha=0.34,
        label=r"Continuous model $F_\lambda$",
        zorder=1,
    )
    asymmetric_band_width = np.array(
        [
            [effective_wavelength - lambda_16],
            [lambda_84 - effective_wavelength],
        ]
    )
    ax_compare.errorbar(
        effective_wavelength,
        model_band_flux,
        xerr=asymmetric_band_width,
        fmt="o",
        color=BLUE,
        ecolor=ORANGE,
        elinewidth=4.0,
        capsize=0.0,
        markersize=7.0,
        label=r"Synthetic point; bar = filter width",
        zorder=4,
    )
    ax_compare.errorbar(
        effective_wavelength,
        observed_band_flux,
        yerr=observed_sigma,
        fmt="s",
        color=RED,
        ecolor=RED,
        elinewidth=1.6,
        capsize=4.0,
        markersize=6.5,
        label=r"Observed point; bar = $1\sigma_b$",
        zorder=5,
    )
    ax_compare.annotate(
        "",
        xy=(effective_wavelength + 19.0, observed_band_flux),
        xytext=(effective_wavelength + 19.0, model_band_flux),
        arrowprops={"arrowstyle": "<->", "color": RED, "linewidth": 1.2},
    )
    ax_compare.text(
        0.96,
        0.94,
        (
            r"$r_b="
            r"\frac{F_b^{\rm obs}-F_b^{\rm model}}{\sigma_b}="
            f"{standardized_residual:.2f}$"
        ),
        transform=ax_compare.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.94},
    )
    ax_compare.set(
        xlim=(395.0, 755.0),
        ylim=(0.65, 1.43),
        xlabel=r"Wavelength assigned to the band [nm]",
        ylabel=r"Normalized band flux $F_b$",
        title=r"3. Compare model and observation",
    )
    ax_compare.grid(True, alpha=0.20)
    ax_compare.legend(loc="lower right", fontsize=9.0)

    # Inset: the observational uncertainty is a distribution in measured flux.
    likelihood_ax = ax_compare.inset_axes([0.69, 0.40, 0.27, 0.30])
    flux_grid = np.linspace(
        observed_band_flux - 3.2 * observed_sigma,
        observed_band_flux + 3.2 * observed_sigma,
        301,
    )
    likelihood = np.exp(-0.5 * ((flux_grid - observed_band_flux) / observed_sigma) ** 2)
    likelihood_ax.plot(likelihood, flux_grid, color=RED, linewidth=1.3)
    likelihood_ax.fill_betweenx(
        flux_grid,
        0.0,
        likelihood,
        color=RED,
        alpha=0.18,
    )
    likelihood_ax.axhline(observed_band_flux, color=RED, linestyle=":", linewidth=1.0)
    likelihood_ax.set(
        xlim=(0.0, 1.05),
        xlabel=r"$p(F_b)$",
        ylabel=r"$F_b$",
        title=r"Flux uncertainty",
    )
    likelihood_ax.set_xticks([])
    likelihood_ax.tick_params(labelsize=7)
    likelihood_ax.xaxis.label.set_size(8)
    likelihood_ax.yaxis.label.set_size(8)
    likelihood_ax.title.set_size(9)

    fig.suptitle(
        "Broadband photometry: continuous SED to one comparable datum "
        "(schematic response)",
        fontsize=16,
    )

    metrics = {
        "effective_wavelength_nm": effective_wavelength,
        "lambda_16_nm": lambda_16,
        "lambda_84_nm": lambda_84,
        "model_band_flux": model_band_flux,
        "observed_band_flux": observed_band_flux,
        "observed_sigma": observed_sigma,
        "standardized_residual": standardized_residual,
    }
    return fig, metrics


def main() -> None:
    """Render PDF and high-resolution PNG outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, metrics = make_figure()
    pdf_path = OUTPUT_DIR / "filter_photometry_workflow.pdf"
    png_path = OUTPUT_DIR / "filter_photometry_workflow.png"
    figure.savefig(pdf_path, dpi=500)
    figure.savefig(png_path, dpi=500)
    plt.close(figure)

    print(f"effective_wavelength_nm={metrics['effective_wavelength_nm']:.3f}")
    print(
        "central_68_percent_band_nm="
        f"[{metrics['lambda_16_nm']:.3f}, {metrics['lambda_84_nm']:.3f}]"
    )
    print(f"model_band_flux={metrics['model_band_flux']:.6f}")
    print(f"observed_band_flux={metrics['observed_band_flux']:.6f}")
    print(f"observed_sigma={metrics['observed_sigma']:.6f}")
    print(f"standardized_residual={metrics['standardized_residual']:.3f}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_png={png_path}")


if __name__ == "__main__":
    main()
