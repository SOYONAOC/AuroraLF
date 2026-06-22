#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss

from auroralf.cooling import compute_atomic_cooling_mass_msun, compute_popiii_lw_minimum_mass_msun
from auroralf.uvlf import sample_uvlf_from_hmf, uv_luminosity_to_muv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_PATH = (
    PROJECT_ROOT
    / "slides"
    / "group_meeting_popiii_20260622"
    / "assets"
    / "uvlf_z14p5_popii_popiii_components_slide.pdf"
)
DEFAULT_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popii_popiii_components_slide.csv"
DEFAULT_NPZ_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popii_popiii_components_slide.npz"
DEFAULT_OBSERVATION_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_observations_slide.csv"
DEFAULT_EXTREME_POPIII_SSP_FILE = (
    PROJECT_ROOT
    / "external_data"
    / "ssp_spectra"
    / "schaerer2010_pop3"
    / "pop3_ge0_sal_500_050_is4.25"
)
DEFAULT_POPIII_SSP_LABEL = r"extreme Pop III IMF ($50$--$500\,M_\odot$)"
DEFAULT_OBSERVATION_PATHS = (
    PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_14" / "whitler25_jades_z14p3.npz",
    PROJECT_ROOT / "external_data" / "observations" / "uvlf" / "redshift_15" / "donnan24_primer_z14p5.npz",
)
OBSERVATION_LABELS = {
    "JADES": r"JADES ($z_{\rm med}=14.3$)",
    "PRIMER": r"PRIMER ($13.5<z<15.5$)",
}
OBSERVATION_STYLES = {
    "JADES": {"marker": "D", "color": "#7A4EAB"},
    "PRIMER": {"marker": "o", "color": "#2A9D8F"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the z=14.5 Pop II/Pop III component UVLF slide figure."
    )
    parser.add_argument("--z", type=float, default=14.5)
    parser.add_argument("--N-mass", type=int, default=720)
    parser.add_argument("--n-tracks", type=int, default=24)
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=14501)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--muv-min", type=float, default=-26.0)
    parser.add_argument("--muv-max", type=float, default=1.5)
    parser.add_argument("--muv-bin-width", type=float, default=0.5)
    parser.add_argument("--smooth-sigma-mag", type=float, default=0.60)
    parser.add_argument("--plot-min-raw-counts", type=int, default=10)
    parser.add_argument("--popiii-burst-sigma-mag", type=float, default=2.0)
    parser.add_argument("--popiii-burst-quadrature-order", type=int, default=31)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--popiii-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_SSP_FILE)
    parser.add_argument("--popiii-ssp-label", type=str, default=DEFAULT_POPIII_SSP_LABEL)
    parser.add_argument("--observation-table-path", type=Path, default=DEFAULT_OBSERVATION_TABLE_PATH)
    parser.add_argument("--observation-paths", nargs="+", type=Path, default=list(DEFAULT_OBSERVATION_PATHS))
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def _load_observation_table(path: Path) -> dict[str, np.ndarray | str]:
    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    required = ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up", "is_upper_limit", "label", "source")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise KeyError(f"Observation file {path} is missing required fields: {missing}")

    label = str(np.asarray(payload["label"])[0])
    source = str(np.asarray(payload["source"])[0])
    z_note = str(np.asarray(payload["z_note"])[0]) if "z_note" in payload.files else ""
    table = {
        "label": label,
        "source": source,
        "z_note": z_note,
        "muv": np.asarray(payload["muverr"], dtype=float),
        "muv_err": np.asarray(payload["mag_err"], dtype=float),
        "phi": np.asarray(payload["phierr"], dtype=float),
        "phi_lo": np.asarray(payload["phi_err_lo"], dtype=float),
        "phi_up": np.asarray(payload["phi_err_up"], dtype=float),
        "upper_limit": np.asarray(payload["is_upper_limit"], dtype=bool),
    }
    sizes = {np.asarray(value).shape for key, value in table.items() if key not in {"label", "source", "z_note"}}
    if len(sizes) != 1:
        raise ValueError(f"Observation file {path} has inconsistent array shapes: {sizes}")
    finite = np.isfinite(table["muv"]) & np.isfinite(table["muv_err"]) & np.isfinite(table["phi"])
    finite &= np.isfinite(table["phi_lo"]) & np.isfinite(table["phi_up"])
    if not np.all(finite):
        raise ValueError(f"Observation file {path} contains non-finite values")
    if np.any(table["phi"] <= 0.0):
        raise ValueError(f"Observation file {path} contains non-positive phi values")
    return table


def _write_observation_csv(path: Path, observations: list[dict[str, np.ndarray | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "source",
                "z_note",
                "Muv",
                "Muv_err",
                "phi_Mpc^-3_mag^-1",
                "phi_err_lo",
                "phi_err_up",
                "is_upper_limit",
            ]
        )
        for obs in observations:
            label = str(obs["label"])
            source = str(obs["source"])
            z_note = str(obs["z_note"])
            for index in range(np.asarray(obs["muv"]).size):
                writer.writerow(
                    [
                        label,
                        source,
                        z_note,
                        f"{float(np.asarray(obs['muv'])[index]):.8e}",
                        f"{float(np.asarray(obs['muv_err'])[index]):.8e}",
                        f"{float(np.asarray(obs['phi'])[index]):.8e}",
                        f"{float(np.asarray(obs['phi_lo'])[index]):.8e}",
                        f"{float(np.asarray(obs['phi_up'])[index]):.8e}",
                        int(bool(np.asarray(obs["upper_limit"])[index])),
                    ]
                )


def _weighted_uvlf_from_luminosity(
    luminosity: np.ndarray,
    sample_weight: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    muv = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    valid = np.isfinite(muv) & np.isfinite(weights) & (weights > 0.0)
    weighted_counts, used_edges = np.histogram(muv[valid], bins=bin_edges, weights=weights[valid])
    raw_counts, raw_edges = np.histogram(muv[valid], bins=bin_edges)
    weight_squared_counts, squared_edges = np.histogram(muv[valid], bins=bin_edges, weights=np.square(weights[valid]))
    if not np.array_equal(used_edges, raw_edges) or not np.array_equal(used_edges, squared_edges):
        raise RuntimeError("Component UVLF histogram bin edges differ")
    bin_width = np.diff(used_edges)
    if np.any(bin_width <= 0.0):
        raise RuntimeError("Component UVLF bin edges are not strictly increasing")
    phi = weighted_counts / bin_width
    phi_sigma = np.sqrt(weight_squared_counts) / bin_width
    return {
        "muv": muv,
        "phi": phi,
        "phi_sigma": phi_sigma,
        "raw_counts": raw_counts.astype(np.int64),
    }


def _gaussian_magnitude_scattered_luminosities(
    *,
    luminosity: np.ndarray,
    sample_weight: np.ndarray,
    sigma_mag: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    luminosity = np.asarray(luminosity, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if luminosity.shape != sample_weight.shape:
        raise ValueError("luminosity and sample_weight must have identical shapes")
    if float(sigma_mag) < 0.0 or not np.isfinite(float(sigma_mag)):
        raise ValueError("sigma_mag must be finite and non-negative")
    if int(quadrature_order) < 3:
        raise ValueError("quadrature_order must be at least 3")
    if float(sigma_mag) == 0.0:
        return luminosity.copy(), sample_weight.copy()

    nodes, weights = hermgauss(int(quadrature_order))
    probability_weights = weights / np.sqrt(np.pi)
    if not np.all(np.isfinite(probability_weights)) or np.any(probability_weights <= 0.0):
        raise RuntimeError("Gaussian quadrature returned invalid weights")
    if not np.isclose(np.sum(probability_weights), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Gaussian quadrature weights do not sum to unity")

    delta_mag = np.sqrt(2.0) * float(sigma_mag) * nodes
    luminosity_scale = np.power(10.0, -0.4 * delta_mag)
    positive_luminosity = np.where(np.isfinite(luminosity) & (luminosity > 0.0), luminosity, 0.0)
    positive_weight = np.where(np.isfinite(sample_weight) & (sample_weight > 0.0), sample_weight, 0.0)
    scattered_luminosity = (positive_luminosity[np.newaxis, :] * luminosity_scale[:, np.newaxis]).reshape(-1)
    scattered_weight = (positive_weight[np.newaxis, :] * probability_weights[:, np.newaxis]).reshape(-1)
    return scattered_luminosity, scattered_weight


def _compute_plot_curve(
    centers: np.ndarray,
    phi: np.ndarray,
    raw_counts: np.ndarray,
    *,
    component_name: str,
    min_raw_counts: int,
    smooth_sigma_mag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plot_mask = np.isfinite(centers) & np.isfinite(phi) & (phi > 0.0)
    plot_mask &= np.asarray(raw_counts, dtype=int) >= int(min_raw_counts)
    if np.count_nonzero(plot_mask) < 3:
        raise RuntimeError(
            f"Fewer than three positive {component_name} bins pass --plot-min-raw-counts; "
            "lower the threshold or increase the sampling."
        )

    x_plot = np.asarray(centers[plot_mask], dtype=float)
    y_raw = np.asarray(phi[plot_mask], dtype=float)
    order = np.argsort(x_plot)
    x_plot = x_plot[order]
    y_raw = y_raw[order]

    if smooth_sigma_mag == 0.0:
        return x_plot, y_raw, plot_mask

    distances = x_plot[:, np.newaxis] - x_plot[np.newaxis, :]
    kernel = np.exp(-0.5 * (distances / smooth_sigma_mag) ** 2)
    kernel_sum = np.sum(kernel, axis=1)
    if np.any(kernel_sum <= 0.0) or not np.all(np.isfinite(kernel_sum)):
        raise RuntimeError(f"Invalid smoothing kernel normalization for {component_name}")
    log_phi_smooth = kernel @ np.log10(y_raw) / kernel_sum
    y_smooth = np.power(10.0, log_phi_smooth)
    if not np.all(np.isfinite(y_smooth)) or np.any(y_smooth <= 0.0):
        raise RuntimeError(f"Smoothed {component_name} UVLF contains invalid values")
    return x_plot, y_smooth, plot_mask


def _plot_column(phi: np.ndarray, centers: np.ndarray, plot_mask: np.ndarray, plot_phi: np.ndarray) -> np.ndarray:
    output = np.full_like(phi, np.nan, dtype=float)
    output[np.where(plot_mask)[0][np.argsort(centers[plot_mask])]] = plot_phi
    return output


def main() -> None:
    args = _parse_args()
    if args.N_mass <= 0:
        raise ValueError("--N-mass must be positive")
    if args.n_tracks <= 0:
        raise ValueError("--n-tracks must be positive")
    if args.n_grid <= 1:
        raise ValueError("--n-grid must be greater than 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.logM_max <= 0.0:
        raise ValueError("--logM-max must be positive")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")
    if args.muv_max <= args.muv_min:
        raise ValueError("--muv-max must be larger than --muv-min")
    if args.muv_bin_width <= 0.0:
        raise ValueError("--muv-bin-width must be positive")
    if args.smooth_sigma_mag < 0.0:
        raise ValueError("--smooth-sigma-mag must be non-negative")
    if args.plot_min_raw_counts < 1:
        raise ValueError("--plot-min-raw-counts must be at least 1")
    if args.popiii_burst_sigma_mag < 0.0:
        raise ValueError("--popiii-burst-sigma-mag must be non-negative")
    if args.popiii_burst_quadrature_order < 3:
        raise ValueError("--popiii-burst-quadrature-order must be at least 3")
    if len(args.observation_paths) == 0:
        raise ValueError("--observation-paths must contain at least one file")
    popiii_ssp_file = _resolve_path(args.popiii_ssp_file)
    if not popiii_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III SSP file not found: {popiii_ssp_file}")

    z_obs = float(args.z)
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=float(args.lw_background_j21))
    )
    atomic_mass_msun = float(compute_atomic_cooling_mass_msun(z_obs))
    logm_min = float(np.log10(popiii_minimum_mass_msun))
    if args.logM_max <= logm_min:
        raise ValueError("--logM-max must be larger than log10(M_popIII_min)")

    bin_edges = np.arange(args.muv_min, args.muv_max + args.muv_bin_width, args.muv_bin_width)
    if bin_edges.size < 2:
        raise RuntimeError("MUV bin construction produced fewer than two bin edges")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    result = sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        random_seed=int(args.random_seed),
        quantity="Muv",
        bins=bin_edges,
        logM_min=logm_min,
        logM_max=float(args.logM_max),
        z_start_max=50.0,
        n_grid=int(args.n_grid),
        sampler="mcbride",
        enable_time_delay=True,
        pipeline_workers=int(args.workers),
        enable_popiii=True,
        popiii_ssp_file=str(popiii_ssp_file),
        imf_mode="canonical",
        lw_background_j21=float(args.lw_background_j21),
    )

    total_luminosity = np.asarray(result.samples["luminosity"], dtype=float)
    popiii_luminosity = np.asarray(result.samples["popiii_luminosity"], dtype=float)
    sample_weight = np.asarray(result.samples["sample_weight"], dtype=float)
    popii_luminosity = total_luminosity - popiii_luminosity
    popii_luminosity[~np.isfinite(popii_luminosity) | (popii_luminosity <= 0.0)] = 0.0
    scattered_popiii_luminosity, scattered_sample_weight = _gaussian_magnitude_scattered_luminosities(
        luminosity=popiii_luminosity,
        sample_weight=sample_weight,
        sigma_mag=float(args.popiii_burst_sigma_mag),
        quadrature_order=int(args.popiii_burst_quadrature_order),
    )
    scattered_total_luminosity = (
        np.tile(popii_luminosity, int(args.popiii_burst_quadrature_order)) + scattered_popiii_luminosity
    )

    components = {
        "popii": _weighted_uvlf_from_luminosity(popii_luminosity, sample_weight, bin_edges),
        "popiii": _weighted_uvlf_from_luminosity(popiii_luminosity, sample_weight, bin_edges),
        "total": _weighted_uvlf_from_luminosity(total_luminosity, sample_weight, bin_edges),
        "popiii_burst": _weighted_uvlf_from_luminosity(
            scattered_popiii_luminosity,
            scattered_sample_weight,
            bin_edges,
        ),
        "total_burst": _weighted_uvlf_from_luminosity(
            scattered_total_luminosity,
            scattered_sample_weight,
            bin_edges,
        ),
    }
    for name, payload in components.items():
        if not np.any(np.isfinite(payload["phi"]) & (payload["phi"] > 0.0)):
            raise RuntimeError(f"{name} UVLF has no positive bins")

    plot_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plot_columns: dict[str, np.ndarray] = {}
    for name, payload in components.items():
        plot_x, plot_y, plot_mask = _compute_plot_curve(
            centers,
            np.asarray(payload["phi"], dtype=float),
            np.asarray(payload["raw_counts"], dtype=int),
            component_name=name,
            min_raw_counts=int(args.plot_min_raw_counts),
            smooth_sigma_mag=float(args.smooth_sigma_mag),
        )
        plot_data[name] = (plot_x, plot_y)
        plot_columns[name] = _plot_column(np.asarray(payload["phi"], dtype=float), centers, plot_mask, plot_y)

    table_path = _resolve_path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        [
            centers,
            components["popii"]["phi"],
            components["popii"]["phi_sigma"],
            components["popii"]["raw_counts"],
            plot_columns["popii"],
            components["popiii"]["phi"],
            components["popiii"]["phi_sigma"],
            components["popiii"]["raw_counts"],
            plot_columns["popiii"],
            components["total"]["phi"],
            components["total"]["phi_sigma"],
            components["total"]["raw_counts"],
            plot_columns["total"],
            components["popiii_burst"]["phi"],
            components["popiii_burst"]["phi_sigma"],
            components["popiii_burst"]["raw_counts"],
            plot_columns["popiii_burst"],
            components["total_burst"]["phi"],
            components["total_burst"]["phi_sigma"],
            components["total_burst"]["raw_counts"],
            plot_columns["total_burst"],
        ]
    )
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header=(
            "Muv_center,phi_popii,phi_sigma_popii,raw_counts_popii,phi_plot_popii,"
            "phi_popiii,phi_sigma_popiii,raw_counts_popiii,phi_plot_popiii,"
            "phi_total,phi_sigma_total,raw_counts_total,phi_plot_total,"
            "phi_popiii_burst,phi_sigma_popiii_burst,raw_counts_popiii_burst,phi_plot_popiii_burst,"
            "phi_total_burst,phi_sigma_total_burst,raw_counts_total_burst,phi_plot_total_burst\n"
            f"z={z_obs},N_mass={args.N_mass},n_tracks={args.n_tracks},n_grid={args.n_grid},"
            f"logM_min=log10(M_popIII_min)={logm_min:.8f},logM_max={args.logM_max},"
            "enable_popiii=True,imf_mode=canonical,enable_time_delay=True,"
            f"popiii_ssp_file={popiii_ssp_file},"
            f"lw_background_j21={args.lw_background_j21},smooth_sigma_mag={args.smooth_sigma_mag},"
            f"plot_min_raw_counts={args.plot_min_raw_counts},"
            f"popiii_burst_sigma_mag={args.popiii_burst_sigma_mag},"
            f"popiii_burst_quadrature_order={args.popiii_burst_quadrature_order}"
        ),
        comments="# ",
    )

    npz_path = _resolve_path(args.npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        bin_edges=bin_edges,
        bin_centers=centers,
        phi_popii=components["popii"]["phi"],
        phi_popiii=components["popiii"]["phi"],
        phi_total=components["total"]["phi"],
        phi_popiii_burst=components["popiii_burst"]["phi"],
        phi_total_burst=components["total_burst"]["phi"],
        phi_plot_popii=plot_columns["popii"],
        phi_plot_popiii=plot_columns["popiii"],
        phi_plot_total=plot_columns["total"],
        phi_plot_popiii_burst=plot_columns["popiii_burst"],
        phi_plot_total_burst=plot_columns["total_burst"],
        sigma_popii=components["popii"]["phi_sigma"],
        sigma_popiii=components["popiii"]["phi_sigma"],
        sigma_total=components["total"]["phi_sigma"],
        sigma_popiii_burst=components["popiii_burst"]["phi_sigma"],
        sigma_total_burst=components["total_burst"]["phi_sigma"],
        count_popii=components["popii"]["raw_counts"],
        count_popiii=components["popiii"]["raw_counts"],
        count_total=components["total"]["raw_counts"],
        count_popiii_burst=components["popiii_burst"]["raw_counts"],
        count_total_burst=components["total_burst"]["raw_counts"],
        total_luminosity=total_luminosity,
        popii_luminosity=popii_luminosity,
        popiii_luminosity=popiii_luminosity,
        scattered_popiii_luminosity=scattered_popiii_luminosity,
        scattered_total_luminosity=scattered_total_luminosity,
        scattered_sample_weight=scattered_sample_weight,
        sample_weight=sample_weight,
        sample_mh=np.asarray(result.samples["Mh"], dtype=float),
        sample_stellar_channel=np.asarray(result.samples["stellar_channel"]),
        z=np.asarray([z_obs], dtype=float),
        M_popiii_min_msun=np.asarray([popiii_minimum_mass_msun], dtype=float),
        M_atomic_msun=np.asarray([atomic_mass_msun], dtype=float),
        logM_min=np.asarray([logm_min], dtype=float),
        logM_max=np.asarray([args.logM_max], dtype=float),
        N_mass=np.asarray([args.N_mass], dtype=int),
        n_tracks=np.asarray([args.n_tracks], dtype=int),
        n_grid=np.asarray([args.n_grid], dtype=int),
        random_seed=np.asarray([args.random_seed], dtype=int),
        smooth_sigma_mag=np.asarray([args.smooth_sigma_mag], dtype=float),
        popiii_burst_sigma_mag=np.asarray([args.popiii_burst_sigma_mag], dtype=float),
        popiii_burst_quadrature_order=np.asarray([args.popiii_burst_quadrature_order], dtype=int),
        plot_min_raw_counts=np.asarray([args.plot_min_raw_counts], dtype=int),
        popiii_ssp_file=np.asarray([str(popiii_ssp_file)]),
        popiii_ssp_label=np.asarray([str(args.popiii_ssp_label)]),
    )

    observation_paths = [_resolve_path(path) for path in args.observation_paths]
    observations = [_load_observation_table(path) for path in observation_paths]
    observation_table_path = _resolve_path(args.observation_table_path)
    _write_observation_csv(observation_table_path, observations)

    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(*plot_data["total_burst"], color="#202020", linewidth=2.9, label=r"Pop II + burst Pop III", zorder=5)
    ax.plot(*plot_data["popii"], color="#1F5C8B", linewidth=2.7, label="Pop II only", zorder=6)
    popiii_label = f"Pop III only ({args.popiii_ssp_label})"
    popiii_burst_label = rf"Pop III burst tail ($\sigma_{{\rm UV}}={args.popiii_burst_sigma_mag:g}$)"
    ax.plot(
        *plot_data["popiii"],
        color="#2A9D8F",
        linewidth=2.7,
        linestyle="--",
        label=popiii_label,
        zorder=7,
    )
    ax.plot(
        *plot_data["popiii_burst"],
        color="#8C5FBF",
        linewidth=3.0,
        linestyle="-.",
        label=popiii_burst_label,
        zorder=8,
    )

    for obs in observations:
        label = str(obs["label"])
        style = OBSERVATION_STYLES.get(label)
        if style is None:
            raise KeyError(f"No plotting style configured for observation label: {label}")
        legend_label = OBSERVATION_LABELS.get(label, label)
        upper_limit = np.asarray(obs["upper_limit"], dtype=bool)
        detection = ~upper_limit
        if np.any(detection):
            ax.errorbar(
                np.asarray(obs["muv"])[detection],
                np.asarray(obs["phi"])[detection],
                xerr=np.asarray(obs["muv_err"])[detection],
                yerr=[
                    np.asarray(obs["phi_lo"])[detection],
                    np.asarray(obs["phi_up"])[detection],
                ],
                fmt=str(style["marker"]),
                ms=8.2,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.7,
                capsize=3.4,
                elinewidth=1.2,
                linestyle="none",
                label=legend_label,
                zorder=12,
            )
        if np.any(upper_limit):
            ax.errorbar(
                np.asarray(obs["muv"])[upper_limit],
                np.asarray(obs["phi"])[upper_limit],
                xerr=np.asarray(obs["muv_err"])[upper_limit],
                yerr=0.35 * np.asarray(obs["phi"])[upper_limit],
                uplims=True,
                fmt=str(style["marker"]),
                ms=8.2,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.7,
                capsize=3.4,
                elinewidth=1.2,
                linestyle="none",
                label=legend_label if not np.any(detection) else None,
                zorder=12,
            )

    all_positive = np.concatenate(
        [
            plot_data["popii"][1],
            plot_data["popiii"][1],
            plot_data["popiii_burst"][1],
            plot_data["total_burst"][1],
            *[np.asarray(obs["phi"], dtype=float) for obs in observations],
        ]
    )
    all_positive = all_positive[np.isfinite(all_positive) & (all_positive > 0.0)]
    if all_positive.size == 0:
        raise RuntimeError("No positive values available for plot limits")
    ax.set_yscale("log")
    ax.set_xlim(args.muv_min, -4.5)
    ax.set_ylim(max(float(np.min(all_positive)) * 0.35, 1.0e-16), float(np.max(all_positive)) * 3.0)
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\Phi\ [{\rm Mpc}^{-3}\ {\rm mag}^{-1}]$")
    ax.grid(True, which="major", color="#C8D2DF", linewidth=0.75, alpha=0.85)
    ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.45, alpha=0.70)
    handles, labels = ax.get_legend_handles_labels()
    label_order = [
        "Pop II only",
        popiii_label,
        popiii_burst_label,
        "Pop II + burst Pop III",
        OBSERVATION_LABELS["JADES"],
        OBSERVATION_LABELS["PRIMER"],
    ]
    ordered_handles = []
    ordered_labels = []
    for target_label in label_order:
        for handle, label in zip(handles, labels):
            if label == target_label:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break
    ax.legend(ordered_handles, ordered_labels, loc="lower right", frameon=True, fontsize=13.5)
    ax.text(
        0.035,
        0.93,
        rf"$z={z_obs:.1f}$, Pop III burst diagnostic with $\sigma_{{\rm UV}}={args.popiii_burst_sigma_mag:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="#1F3A5F",
    )
    fig.tight_layout()

    figure_path = _resolve_path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=500)
    plt.close(fig)

    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {observation_table_path}")


if __name__ == "__main__":
    main()
