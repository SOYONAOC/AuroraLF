#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_atomic_cooling_mass_msun, compute_popiii_lw_minimum_mass_msun
from auroralf.mah import Cosmology
from auroralf.sfr import (
    POPIII_UPPER_MASS_MODE_ATOMIC,
    POPIII_UPPER_MASS_MODE_FIXED,
    PopIIISFRParameters,
)
from auroralf.uvlf import compute_dust_attenuated_uvlf, sample_uvlf_from_hmf
from scripts.plot.plot_group_meeting_popiii_components_uvlf import (
    DEFAULT_EXTREME_POPIII_SSP_FILE,
    DEFAULT_OBSERVATION_PATHS,
    DEFAULT_POPIII_SSP_LABEL,
    OBSERVATION_LABELS as BASE_OBSERVATION_LABELS,
    OBSERVATION_STYLES as BASE_OBSERVATION_STYLES,
    PROJECT_ROOT,
    _compute_plot_curve,
    _gaussian_magnitude_scattered_luminosities,
    _plot_column,
    _resolve_path,
    _weighted_uvlf_from_luminosity,
    _write_observation_csv,
)


DEFAULT_FIGURE_PATH = (
    PROJECT_ROOT
    / "slides"
    / "group_meeting_popiii_20260622"
    / "assets"
    / "uvlf_z14p5_popiii_mup_comparison_slide.pdf"
)
DEFAULT_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popiii_mup_comparison_slide.csv"
DEFAULT_NPZ_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popiii_mup_comparison_slide.npz"
DEFAULT_OBSERVATION_TABLE_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popiii_mup_comparison_observations.csv"
DEFAULT_CURRENT_NPZ_PATH = PROJECT_ROOT / "outputs" / "uvlf_z14p5_popii_popiii_components_slide.npz"
OBSERVATION_LABELS = {
    **BASE_OBSERVATION_LABELS,
    "Bowler+15": "Bowler+15",
    "Bouwens+21": "Bouwens+21",
    "Finkelstein+15": "Finkelstein+15",
    "Bowler+20": "Bowler+20",
    "Donnan+23": "Donnan+23",
    "McLure+13": "McLure+13",
    "Donnan+24": "Donnan+24",
    "Bouwens+23, $z\\sim12-13$": r"Bouwens+23, $z\sim12-13$",
    "Donnan+24, $z\\sim12.5$": r"Donnan+24, $z\sim12.5$",
    "Harikane+23, $z\\sim12$": r"Harikane+23, $z\sim12$",
}
OBSERVATION_STYLES = {
    **BASE_OBSERVATION_STYLES,
    "Bowler+15": {"marker": "s", "color": "#D55E00"},
    "Bouwens+21": {"marker": "^", "color": "#0072B2"},
    "Finkelstein+15": {"marker": "P", "color": "#009E73"},
    "Bowler+20": {"marker": "s", "color": "#D55E00"},
    "Donnan+23": {"marker": "D", "color": "#CC79A7"},
    "McLure+13": {"marker": "^", "color": "#0072B2"},
    "Donnan+24": {"marker": "D", "color": "#CC79A7"},
    "Bouwens+23, $z\\sim12-13$": {"marker": "^", "color": "#0072B2"},
    "Donnan+24, $z\\sim12.5$": {"marker": "D", "color": "#CC79A7"},
    "Harikane+23, $z\\sim12$": {"marker": "P", "color": "#009E73"},
}


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    upper_mass_mode: str
    upper_mass_msun: float | None


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    components: dict[str, dict[str, np.ndarray]]
    plot_data: dict[str, tuple[np.ndarray, np.ndarray]]
    plot_columns: dict[str, np.ndarray]
    total_luminosity: np.ndarray
    popii_luminosity: np.ndarray
    popiii_luminosity: np.ndarray
    scattered_popiii_luminosity: np.ndarray
    scattered_total_luminosity: np.ndarray
    scattered_sample_weight: np.ndarray
    sample_weight: np.ndarray
    sample_mh: np.ndarray
    sample_stellar_channel: np.ndarray
    popiii_upper_mass_msun: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the z=14.5 Pop III component UVLF for atomic and fixed M_up."
    )
    parser.add_argument("--z", type=float, default=14.5)
    parser.add_argument("--N-mass", type=int, default=720)
    parser.add_argument("--n-tracks", type=int, default=24)
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=14501)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--fixed-upper-mass-msun", type=float, default=1.0e10)
    parser.add_argument("--lw-background-j21", type=float, default=0.0)
    parser.add_argument("--muv-min", type=float, default=-26.0)
    parser.add_argument("--muv-max", type=float, default=1.5)
    parser.add_argument("--muv-bin-width", type=float, default=0.5)
    parser.add_argument(
        "--phi-min",
        type=float,
        default=None,
        help="Optional fixed lower y-axis bound. By default the lower bound is data-driven.",
    )
    parser.add_argument("--smooth-sigma-mag", type=float, default=0.60)
    parser.add_argument("--plot-min-raw-counts", type=int, default=10)
    parser.add_argument("--popiii-burst-sigma-mag", type=float, default=2.0)
    parser.add_argument("--popiii-burst-quadrature-order", type=int, default=31)
    parser.add_argument("--apply-dust", action="store_true")
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument(
        "--current-npz-path",
        type=Path,
        default=None,
        help=f"Optional validated current-scenario NPZ to reuse, for example {DEFAULT_CURRENT_NPZ_PATH}.",
    )
    parser.add_argument("--popiii-ssp-file", type=Path, default=DEFAULT_EXTREME_POPIII_SSP_FILE)
    parser.add_argument("--popiii-ssp-label", type=str, default=DEFAULT_POPIII_SSP_LABEL)
    parser.add_argument("--no-observations", action="store_true")
    parser.add_argument("--observation-table-path", type=Path, default=DEFAULT_OBSERVATION_TABLE_PATH)
    parser.add_argument("--observation-paths", nargs="+", type=Path, default=list(DEFAULT_OBSERVATION_PATHS))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
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
    if args.fixed_upper_mass_msun <= 0.0 or not np.isfinite(args.fixed_upper_mass_msun):
        raise ValueError("--fixed-upper-mass-msun must be finite and positive")
    if args.lw_background_j21 < 0.0:
        raise ValueError("--lw-background-j21 must be non-negative")
    if args.muv_max <= args.muv_min:
        raise ValueError("--muv-max must be larger than --muv-min")
    if args.muv_bin_width <= 0.0:
        raise ValueError("--muv-bin-width must be positive")
    if args.phi_min is not None and (args.phi_min <= 0.0 or not np.isfinite(args.phi_min)):
        raise ValueError("--phi-min must be finite and positive")
    if args.smooth_sigma_mag < 0.0:
        raise ValueError("--smooth-sigma-mag must be non-negative")
    if args.plot_min_raw_counts < 1:
        raise ValueError("--plot-min-raw-counts must be at least 1")
    if args.popiii_burst_sigma_mag < 0.0:
        raise ValueError("--popiii-burst-sigma-mag must be non-negative")
    if args.popiii_burst_quadrature_order < 3:
        raise ValueError("--popiii-burst-quadrature-order must be at least 3")
    if not bool(args.no_observations) and len(args.observation_paths) == 0:
        raise ValueError("--observation-paths must contain at least one file")


def _scenario_parameters(args: argparse.Namespace, scenario: Scenario) -> PopIIISFRParameters:
    return PopIIISFRParameters(
        lw_background_j21=float(args.lw_background_j21),
        upper_mass_mode=scenario.upper_mass_mode,
        upper_mass_msun=scenario.upper_mass_msun,
    )


def _load_observation_table_for_comparison(path: Path) -> dict[str, np.ndarray | str]:
    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    required = ("muverr", "phierr", "mag_err", "phi_err_lo", "phi_err_up", "label")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise KeyError(f"Observation file {path} is missing required fields: {missing}")

    label = str(np.asarray(payload["label"]).reshape(-1)[0])
    source = str(np.asarray(payload["source"]).reshape(-1)[0]) if "source" in payload.files else path.name
    z_note = str(np.asarray(payload["z_note"]).reshape(-1)[0]) if "z_note" in payload.files else ""
    phi = np.asarray(payload["phierr"], dtype=float)
    upper_limit = (
        np.asarray(payload["is_upper_limit"], dtype=bool)
        if "is_upper_limit" in payload.files
        else np.zeros_like(phi, dtype=bool)
    )
    table = {
        "label": label,
        "source": source,
        "z_note": z_note,
        "muv": np.asarray(payload["muverr"], dtype=float),
        "muv_err": np.asarray(payload["mag_err"], dtype=float),
        "phi": phi,
        "phi_lo": np.asarray(payload["phi_err_lo"], dtype=float),
        "phi_up": np.asarray(payload["phi_err_up"], dtype=float),
        "upper_limit": upper_limit,
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


def _apply_optional_dust_to_component(
    centers: np.ndarray,
    phi: np.ndarray,
    *,
    z_obs: float,
    apply_dust: bool,
) -> np.ndarray:
    phi_array = np.asarray(phi, dtype=float)
    if not apply_dust:
        return phi_array.copy()
    dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=np.asarray(centers, dtype=float),
        intrinsic_phi=phi_array,
        z=float(z_obs),
        muv_obs=np.asarray(centers, dtype=float),
    )
    return np.asarray(dust["phi_obs"], dtype=float)


def _scale_sigma_to_final_phi(
    *,
    intrinsic_phi: np.ndarray,
    intrinsic_phi_sigma: np.ndarray,
    final_phi: np.ndarray,
) -> np.ndarray:
    intrinsic_phi = np.asarray(intrinsic_phi, dtype=float)
    intrinsic_phi_sigma = np.asarray(intrinsic_phi_sigma, dtype=float)
    final_phi = np.asarray(final_phi, dtype=float)
    fractional_sigma = np.divide(
        intrinsic_phi_sigma,
        intrinsic_phi,
        out=np.zeros_like(intrinsic_phi_sigma, dtype=float),
        where=np.isfinite(intrinsic_phi) & (intrinsic_phi > 0.0),
    )
    return final_phi * fractional_sigma


def _prepare_components_for_plot(
    *,
    components: dict[str, dict[str, np.ndarray]],
    centers: np.ndarray,
    args: argparse.Namespace,
    scenario_key: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    for name, payload in components.items():
        intrinsic_phi = np.asarray(payload["phi"], dtype=float)
        final_phi = _apply_optional_dust_to_component(
            centers,
            intrinsic_phi,
            z_obs=float(args.z),
            apply_dust=bool(args.apply_dust),
        )
        payload["phi"] = final_phi
        if bool(args.apply_dust):
            payload["phi_sigma"] = _scale_sigma_to_final_phi(
                intrinsic_phi=intrinsic_phi,
                intrinsic_phi_sigma=np.asarray(payload["phi_sigma"], dtype=float),
                final_phi=final_phi,
            )

    plot_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plot_columns: dict[str, np.ndarray] = {}
    for name, payload in components.items():
        plot_x, plot_y, plot_mask = _compute_plot_curve(
            centers,
            np.asarray(payload["phi"], dtype=float),
            np.asarray(payload["raw_counts"], dtype=int),
            component_name=f"{scenario_key}:{name}",
            min_raw_counts=int(args.plot_min_raw_counts),
            smooth_sigma_mag=float(args.smooth_sigma_mag),
        )
        plot_data[name] = (plot_x, plot_y)
        plot_columns[name] = _plot_column(np.asarray(payload["phi"], dtype=float), centers, plot_mask, plot_y)
    return plot_data, plot_columns


def _finite_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _resolve_log_y_limits(
    *,
    all_positive: np.ndarray,
    comparison_positive: np.ndarray,
    observation_positive: np.ndarray | None = None,
    fixed_phi_min: float | None,
) -> tuple[float, float]:
    all_values = _finite_positive(all_positive)
    comparison_values = _finite_positive(comparison_positive)
    observation_values = (
        np.empty(0, dtype=float) if observation_positive is None else _finite_positive(observation_positive)
    )
    if all_values.size == 0:
        raise RuntimeError("No positive values available for plot limits")
    if fixed_phi_min is not None:
        y_min = float(fixed_phi_min)
    else:
        limit_values = comparison_values if comparison_values.size else all_values
        y_min = float(np.min(limit_values) * 0.4)
        if observation_values.size:
            y_min = max(y_min, float(np.min(observation_values) * 1.0e-2))
    y_max = float(np.max(all_values) * 3.0)
    if y_min <= 0.0 or not np.isfinite(y_min):
        raise RuntimeError(f"Invalid y-axis lower limit: {y_min}")
    if y_max <= y_min or not np.isfinite(y_max):
        raise RuntimeError(f"Invalid y-axis limits: ymin={y_min}, ymax={y_max}")
    return y_min, y_max


def _require_npz_fields(payload: np.lib.npyio.NpzFile, path: Path, fields: tuple[str, ...]) -> None:
    missing = [name for name in fields if name not in payload.files]
    if missing:
        raise KeyError(f"NPZ file {path} is missing required fields: {missing}")


def _scalar_from_npz(payload: np.lib.npyio.NpzFile, name: str) -> float:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f"NPZ field {name!r} must contain a scalar, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def _int_scalar_from_npz(payload: np.lib.npyio.NpzFile, name: str) -> int:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f"NPZ field {name!r} must contain a scalar, got shape {value.shape}")
    return int(value.reshape(-1)[0])


def _assert_close(name: str, actual: float, expected: float, *, rtol: float = 1.0e-10, atol: float = 1.0e-12) -> None:
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"loaded current NPZ has {name}={actual}, expected {expected}")


def _validate_loaded_current_npz(
    *,
    payload: np.lib.npyio.NpzFile,
    path: Path,
    args: argparse.Namespace,
    centers: np.ndarray,
    popiii_minimum_mass_msun: float,
    atomic_mass_msun: float,
    logm_min: float,
    popiii_ssp_file: Path,
) -> None:
    _require_npz_fields(
        payload,
        path,
        (
            "bin_centers",
            "z",
            "M_popiii_min_msun",
            "M_atomic_msun",
            "logM_min",
            "logM_max",
            "N_mass",
            "n_tracks",
            "n_grid",
            "base_seed",
            "smooth_sigma_mag",
            "popiii_burst_sigma_mag",
            "popiii_burst_quadrature_order",
            "plot_min_raw_counts",
            "popiii_ssp_file",
        ),
    )
    loaded_centers = np.asarray(payload["bin_centers"], dtype=float)
    if loaded_centers.shape != centers.shape or not np.allclose(loaded_centers, centers, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"loaded current NPZ bin centers do not match requested MUV bins: {path}")
    _assert_close("z", _scalar_from_npz(payload, "z"), float(args.z))
    _assert_close("M_popiii_min_msun", _scalar_from_npz(payload, "M_popiii_min_msun"), popiii_minimum_mass_msun)
    _assert_close("M_atomic_msun", _scalar_from_npz(payload, "M_atomic_msun"), atomic_mass_msun)
    _assert_close("logM_min", _scalar_from_npz(payload, "logM_min"), logm_min)
    _assert_close("logM_max", _scalar_from_npz(payload, "logM_max"), float(args.logM_max))
    if _int_scalar_from_npz(payload, "N_mass") != int(args.N_mass):
        raise ValueError("loaded current NPZ N_mass does not match requested value")
    if _int_scalar_from_npz(payload, "n_tracks") != int(args.n_tracks):
        raise ValueError("loaded current NPZ n_tracks does not match requested value")
    if _int_scalar_from_npz(payload, "n_grid") != int(args.n_grid):
        raise ValueError("loaded current NPZ n_grid does not match requested value")
    if _int_scalar_from_npz(payload, "base_seed") != int(args.random_seed):
        raise ValueError("loaded current NPZ base_seed does not match requested value")
    if _int_scalar_from_npz(payload, "popiii_burst_quadrature_order") != int(args.popiii_burst_quadrature_order):
        raise ValueError("loaded current NPZ popiii_burst_quadrature_order does not match requested value")
    if _int_scalar_from_npz(payload, "plot_min_raw_counts") != int(args.plot_min_raw_counts):
        raise ValueError("loaded current NPZ plot_min_raw_counts does not match requested value")
    _assert_close("smooth_sigma_mag", _scalar_from_npz(payload, "smooth_sigma_mag"), float(args.smooth_sigma_mag))
    _assert_close(
        "popiii_burst_sigma_mag",
        _scalar_from_npz(payload, "popiii_burst_sigma_mag"),
        float(args.popiii_burst_sigma_mag),
    )
    loaded_ssp = Path(str(np.asarray(payload["popiii_ssp_file"]).reshape(-1)[0])).expanduser().resolve()
    if loaded_ssp != popiii_ssp_file:
        raise ValueError(f"loaded current NPZ popiii_ssp_file={loaded_ssp}, expected {popiii_ssp_file}")


def _load_current_scenario_from_npz(
    *,
    path: Path,
    args: argparse.Namespace,
    scenario: Scenario,
    centers: np.ndarray,
    popiii_minimum_mass_msun: float,
    atomic_mass_msun: float,
    logm_min: float,
    popiii_ssp_file: Path,
) -> ScenarioResult:
    with np.load(path, allow_pickle=True) as payload:
        _validate_loaded_current_npz(
            payload=payload,
            path=path,
            args=args,
            centers=centers,
            popiii_minimum_mass_msun=popiii_minimum_mass_msun,
            atomic_mass_msun=atomic_mass_msun,
            logm_min=logm_min,
            popiii_ssp_file=popiii_ssp_file,
        )
        required = [
            "total_luminosity",
            "popii_luminosity",
            "popiii_luminosity",
            "scattered_popiii_luminosity",
            "scattered_total_luminosity",
            "scattered_sample_weight",
            "sample_weight",
            "sample_mh",
            "sample_stellar_channel",
        ]
        for name in ("popii", "popiii", "total", "popiii_burst", "total_burst"):
            required.extend([f"phi_{name}", f"sigma_{name}", f"count_{name}", f"phi_plot_{name}"])
        _require_npz_fields(payload, path, tuple(required))

        components: dict[str, dict[str, np.ndarray]] = {}
        for name in ("popii", "popiii", "total", "popiii_burst", "total_burst"):
            components[name] = {
                "phi": np.asarray(payload[f"phi_{name}"], dtype=float),
                "phi_sigma": np.asarray(payload[f"sigma_{name}"], dtype=float),
                "raw_counts": np.asarray(payload[f"count_{name}"], dtype=np.int64),
            }

        plot_data, plot_columns = _prepare_components_for_plot(
            components=components,
            centers=centers,
            args=args,
            scenario_key=scenario.key,
        )

        return ScenarioResult(
            scenario=scenario,
            components=components,
            plot_data=plot_data,
            plot_columns=plot_columns,
            total_luminosity=np.asarray(payload["total_luminosity"], dtype=float),
            popii_luminosity=np.asarray(payload["popii_luminosity"], dtype=float),
            popiii_luminosity=np.asarray(payload["popiii_luminosity"], dtype=float),
            scattered_popiii_luminosity=np.asarray(payload["scattered_popiii_luminosity"], dtype=float),
            scattered_total_luminosity=np.asarray(payload["scattered_total_luminosity"], dtype=float),
            scattered_sample_weight=np.asarray(payload["scattered_sample_weight"], dtype=float),
            sample_weight=np.asarray(payload["sample_weight"], dtype=float),
            sample_mh=np.asarray(payload["sample_mh"], dtype=float),
            sample_stellar_channel=np.asarray(payload["sample_stellar_channel"]),
            popiii_upper_mass_msun=atomic_mass_msun,
        )


def _run_scenario(
    *,
    cosmology: Cosmology,
    args: argparse.Namespace,
    scenario: Scenario,
    bin_edges: np.ndarray,
    centers: np.ndarray,
    logm_min: float,
    popiii_ssp_file: Path,
) -> ScenarioResult:
    params = _scenario_parameters(args, scenario)
    result = sample_uvlf_from_hmf(
        z_obs=float(args.z),
        cosmology=cosmology,
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        base_seed=int(args.random_seed),
        quantity="Muv",
        bins=bin_edges,
        logM_min=float(logm_min),
        logM_max=float(args.logM_max),
        z_start_max=50.0,
        n_grid=int(args.n_grid),
        sampler="mcbride",
        enable_time_delay=True,
        pipeline_workers=int(args.workers),
        enable_popiii=True,
        popiii_sfr_parameters=params,
        popiii_ssp_file=str(popiii_ssp_file),
        imf_mode="canonical",
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
            raise RuntimeError(f"{scenario.key}:{name} UVLF has no positive bins")

    plot_data, plot_columns = _prepare_components_for_plot(
        components=components,
        centers=centers,
        args=args,
        scenario_key=scenario.key,
    )

    popiii_upper_mass = (
        float(compute_atomic_cooling_mass_msun(float(args.z), cosmology=cosmology))
        if scenario.upper_mass_mode == POPIII_UPPER_MASS_MODE_ATOMIC
        else float(scenario.upper_mass_msun)
    )
    return ScenarioResult(
        scenario=scenario,
        components=components,
        plot_data=plot_data,
        plot_columns=plot_columns,
        total_luminosity=total_luminosity,
        popii_luminosity=popii_luminosity,
        popiii_luminosity=popiii_luminosity,
        scattered_popiii_luminosity=scattered_popiii_luminosity,
        scattered_total_luminosity=scattered_total_luminosity,
        scattered_sample_weight=scattered_sample_weight,
        sample_weight=sample_weight,
        sample_mh=np.asarray(result.samples["Mh"], dtype=float),
        sample_stellar_channel=np.asarray(result.samples["stellar_channel"]),
        popiii_upper_mass_msun=popiii_upper_mass,
    )


def _write_comparison_csv(path: Path, centers: np.ndarray, results: list[ScenarioResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "Muv_center",
                "phi_popii",
                "phi_sigma_popii",
                "raw_counts_popii",
                "phi_plot_popii",
                "phi_popiii",
                "phi_sigma_popiii",
                "raw_counts_popiii",
                "phi_plot_popiii",
                "phi_total",
                "phi_sigma_total",
                "raw_counts_total",
                "phi_plot_total",
                "phi_popiii_burst",
                "phi_sigma_popiii_burst",
                "raw_counts_popiii_burst",
                "phi_plot_popiii_burst",
                "phi_total_burst",
                "phi_sigma_total_burst",
                "raw_counts_total_burst",
                "phi_plot_total_burst",
            ]
        )
        for result in results:
            for index, center in enumerate(centers):
                row = [result.scenario.key, f"{float(center):.8e}"]
                for name in ("popii", "popiii", "total", "popiii_burst", "total_burst"):
                    payload = result.components[name]
                    row.extend(
                        [
                            f"{float(payload['phi'][index]):.8e}",
                            f"{float(payload['phi_sigma'][index]):.8e}",
                            int(payload["raw_counts"][index]),
                            f"{float(result.plot_columns[name][index]):.8e}"
                            if np.isfinite(result.plot_columns[name][index])
                            else "nan",
                        ]
                    )
                writer.writerow(row)


def _save_npz(
    path: Path,
    *,
    args: argparse.Namespace,
    bin_edges: np.ndarray,
    centers: np.ndarray,
    popiii_minimum_mass_msun: float,
    atomic_mass_msun: float,
    logm_min: float,
    popiii_ssp_file: Path,
    results: list[ScenarioResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "bin_edges": bin_edges,
        "bin_centers": centers,
        "z": np.asarray([float(args.z)], dtype=float),
        "M_popiii_min_msun": np.asarray([popiii_minimum_mass_msun], dtype=float),
        "M_atomic_msun": np.asarray([atomic_mass_msun], dtype=float),
        "logM_min": np.asarray([logm_min], dtype=float),
        "logM_max": np.asarray([float(args.logM_max)], dtype=float),
        "N_mass": np.asarray([int(args.N_mass)], dtype=int),
        "n_tracks": np.asarray([int(args.n_tracks)], dtype=int),
        "n_grid": np.asarray([int(args.n_grid)], dtype=int),
        "base_seed": np.asarray([int(args.random_seed)], dtype=np.uint64),
        "smooth_sigma_mag": np.asarray([float(args.smooth_sigma_mag)], dtype=float),
        "phi_min": np.asarray(
            [float(args.phi_min) if args.phi_min is not None else np.nan],
            dtype=float,
        ),
        "popiii_burst_sigma_mag": np.asarray([float(args.popiii_burst_sigma_mag)], dtype=float),
        "popiii_burst_quadrature_order": np.asarray([int(args.popiii_burst_quadrature_order)], dtype=int),
        "plot_min_raw_counts": np.asarray([int(args.plot_min_raw_counts)], dtype=int),
        "lw_background_j21": np.asarray([float(args.lw_background_j21)], dtype=float),
        "apply_dust": np.asarray([bool(args.apply_dust)], dtype=bool),
        "popiii_ssp_file": np.asarray([str(popiii_ssp_file)]),
        "popiii_ssp_label": np.asarray([str(args.popiii_ssp_label)]),
    }
    for result in results:
        prefix = result.scenario.key
        payload[f"{prefix}_upper_mass_mode"] = np.asarray([result.scenario.upper_mass_mode])
        payload[f"{prefix}_upper_mass_msun"] = np.asarray([result.popiii_upper_mass_msun], dtype=float)
        for name in ("popii", "popiii", "total", "popiii_burst", "total_burst"):
            payload[f"{prefix}_phi_{name}"] = result.components[name]["phi"]
            payload[f"{prefix}_sigma_{name}"] = result.components[name]["phi_sigma"]
            payload[f"{prefix}_count_{name}"] = result.components[name]["raw_counts"]
            payload[f"{prefix}_phi_plot_{name}"] = result.plot_columns[name]
        payload[f"{prefix}_total_luminosity"] = result.total_luminosity
        payload[f"{prefix}_popii_luminosity"] = result.popii_luminosity
        payload[f"{prefix}_popiii_luminosity"] = result.popiii_luminosity
        payload[f"{prefix}_scattered_popiii_luminosity"] = result.scattered_popiii_luminosity
        payload[f"{prefix}_scattered_total_luminosity"] = result.scattered_total_luminosity
        payload[f"{prefix}_scattered_sample_weight"] = result.scattered_sample_weight
        payload[f"{prefix}_sample_weight"] = result.sample_weight
        payload[f"{prefix}_sample_mh"] = result.sample_mh
        payload[f"{prefix}_sample_stellar_channel"] = result.sample_stellar_channel
    np.savez(path, **payload)


def _plot_observations(ax: plt.Axes, observations: list[dict[str, np.ndarray | str]]) -> None:
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
                ms=6.8,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.6,
                capsize=2.8,
                elinewidth=1.0,
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
                ms=6.8,
                color=str(style["color"]),
                markeredgecolor="#1C1C1C",
                markeredgewidth=0.6,
                capsize=2.8,
                elinewidth=1.0,
                linestyle="none",
                label=legend_label if not np.any(detection) else None,
                zorder=12,
            )


def _draw_figure(
    path: Path,
    *,
    args: argparse.Namespace,
    results: list[ScenarioResult],
    observations: list[dict[str, np.ndarray | str]],
) -> None:
    all_positive = []
    comparison_positive = []
    obs_muv_values = []
    obs_muv_err_values = []
    plotted_component_names = ("popii", "popiii_burst", "total_burst")
    comparison_component_names = ("popii", "total_burst")
    for result in results:
        for name in plotted_component_names:
            all_positive.append(result.plot_data[name][1])
    for obs in observations:
        all_positive.append(np.asarray(obs["phi"], dtype=float))
        comparison_positive.append(np.asarray(obs["phi"], dtype=float))
        obs_muv_values.append(np.asarray(obs["muv"], dtype=float))
        obs_muv_err_values.append(np.asarray(obs["muv_err"], dtype=float))

    if obs_muv_values:
        obs_muv = np.concatenate(obs_muv_values)
        obs_muv_err = np.concatenate(obs_muv_err_values)
        comparison_x_min = float(np.nanmin(obs_muv - obs_muv_err) - 0.5)
        comparison_x_max = float(np.nanmax(obs_muv + obs_muv_err) + 0.5)
        for result in results:
            for name in comparison_component_names:
                x_values, y_values = result.plot_data[name]
                in_window = (x_values >= comparison_x_min) & (x_values <= comparison_x_max)
                if np.any(in_window):
                    comparison_positive.append(y_values[in_window])
    else:
        comparison_positive = all_positive

    y_min, y_max = _resolve_log_y_limits(
        all_positive=np.concatenate(all_positive),
        comparison_positive=np.concatenate(comparison_positive),
        observation_positive=np.concatenate([np.asarray(obs["phi"], dtype=float) for obs in observations])
        if observations
        else None,
        fixed_phi_min=args.phi_min,
    )

    plt.style.use("apj")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.35), sharex=True, sharey=True)
    style_by_name = {
        "popii": {"color": "#1F5C8B", "linewidth": 2.35, "linestyle": "-", "label": "Pop II only", "zorder": 6},
        "popiii_burst": {
            "color": "#8C5FBF",
            "linewidth": 2.55,
            "linestyle": "-.",
            "label": rf"Pop III burst tail ($\sigma_{{\rm UV}}={args.popiii_burst_sigma_mag:g}$)",
            "zorder": 8,
        },
        "total_burst": {
            "color": "#202020",
            "linewidth": 2.55,
            "linestyle": "-",
            "label": "Pop II + burst Pop III",
            "zorder": 5,
        },
    }

    for ax, result in zip(axes, results, strict=True):
        for name in ("total_burst", "popii", "popiii_burst"):
            style = style_by_name[name]
            ax.plot(
                *result.plot_data[name],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                label=style["label"],
                zorder=style["zorder"],
            )
        if observations:
            _plot_observations(ax, observations)
        ax.set_title(result.scenario.title, fontsize=15, color="#1F3A5F")
        ax.set_yscale("log")
        ax.set_xlim(args.muv_min, -4.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r"$M_{\rm UV}$")
        ax.grid(True, which="major", color="#C8D2DF", linewidth=0.70, alpha=0.85)
        ax.grid(True, which="minor", color="#E4E9F0", linewidth=0.40, alpha=0.70)
    ylabel = (
        r"$\Phi_{\rm obs}\ [{\rm Mpc}^{-3}\ {\rm mag}^{-1}]$"
        if bool(args.apply_dust)
        else r"$\Phi\ [{\rm Mpc}^{-3}\ {\rm mag}^{-1}]$"
    )
    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    label_order = [
        "Pop II only",
        style_by_name["popiii_burst"]["label"],
        "Pop II + burst Pop III",
    ]
    if observations:
        label_order.extend(
            OBSERVATION_LABELS.get(str(obs["label"]), str(obs["label"]))
            for obs in observations
        )
    ordered_handles = []
    ordered_labels = []
    for target_label in label_order:
        for handle, label in zip(handles, labels):
            if label == target_label:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=True,
        fontsize=11.0,
    )
    uvlf_text = "dust attenuated" if bool(args.apply_dust) else "intrinsic"
    fig.suptitle(
        rf"$z={float(args.z):.1f}$, {uvlf_text} Pop III burst diagnostic with "
        rf"$\sigma_{{\rm UV}}={float(args.popiii_burst_sigma_mag):g}$",
        y=0.985,
        fontsize=17,
        color="#1F3A5F",
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=500)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cosmology = Cosmology()
    _validate_args(args)

    popiii_ssp_file = _resolve_path(args.popiii_ssp_file)
    if not popiii_ssp_file.is_file():
        raise FileNotFoundError(f"Pop III SSP file not found: {popiii_ssp_file}")

    z_obs = float(args.z)
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=float(args.lw_background_j21))
    )
    atomic_mass_msun = float(
        compute_atomic_cooling_mass_msun(z_obs, cosmology=cosmology)
    )
    logm_min = float(np.log10(popiii_minimum_mass_msun))
    if args.logM_max <= logm_min:
        raise ValueError("--logM-max must be larger than log10(M_popIII_min)")

    bin_edges = np.arange(args.muv_min, args.muv_max + args.muv_bin_width, args.muv_bin_width)
    if bin_edges.size < 2:
        raise RuntimeError("MUV bin construction produced fewer than two bin edges")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    scenarios = [
        Scenario(
            key="current",
            title=rf"current: $M_{{\rm up}}=M_{{\rm vir}}(10^4\,{{\rm K}})={atomic_mass_msun:.2e}\,M_\odot$",
            upper_mass_mode=POPIII_UPPER_MASS_MODE_ATOMIC,
            upper_mass_msun=None,
        ),
        Scenario(
            key="fixed_mup1e10",
            title=rf"fixed: $M_{{\rm up}}=10^{{10}}\,M_\odot$",
            upper_mass_mode=POPIII_UPPER_MASS_MODE_FIXED,
            upper_mass_msun=float(args.fixed_upper_mass_msun),
        ),
    ]

    if args.current_npz_path is None:
        current_result = _run_scenario(
            cosmology=cosmology,
            args=args,
            scenario=scenarios[0],
            bin_edges=bin_edges,
            centers=centers,
            logm_min=logm_min,
            popiii_ssp_file=popiii_ssp_file,
        )
    else:
        current_npz_path = _resolve_path(args.current_npz_path)
        if not current_npz_path.is_file():
            raise FileNotFoundError(f"current NPZ file not found: {current_npz_path}")
        current_result = _load_current_scenario_from_npz(
            path=current_npz_path,
            args=args,
            scenario=scenarios[0],
            centers=centers,
            popiii_minimum_mass_msun=popiii_minimum_mass_msun,
            atomic_mass_msun=atomic_mass_msun,
            logm_min=logm_min,
            popiii_ssp_file=popiii_ssp_file,
        )
    fixed_result = _run_scenario(
        cosmology=cosmology,
        args=args,
        scenario=scenarios[1],
        bin_edges=bin_edges,
        centers=centers,
        logm_min=logm_min,
        popiii_ssp_file=popiii_ssp_file,
    )
    results = [current_result, fixed_result]

    if args.no_observations:
        observations = []
        observation_table_path = None
    else:
        observation_paths = [_resolve_path(path) for path in args.observation_paths]
        observations = [_load_observation_table_for_comparison(path) for path in observation_paths]
        observation_table_path = _resolve_path(args.observation_table_path)
        _write_observation_csv(observation_table_path, observations)

    table_path = _resolve_path(args.table_path)
    _write_comparison_csv(table_path, centers, results)

    npz_path = _resolve_path(args.npz_path)
    _save_npz(
        npz_path,
        args=args,
        bin_edges=bin_edges,
        centers=centers,
        popiii_minimum_mass_msun=popiii_minimum_mass_msun,
        atomic_mass_msun=atomic_mass_msun,
        logm_min=logm_min,
        popiii_ssp_file=popiii_ssp_file,
        results=results,
    )

    figure_path = _resolve_path(args.figure_path)
    _draw_figure(figure_path, args=args, results=results, observations=observations)

    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {npz_path}")
    if observation_table_path is not None:
        print(f"Wrote {observation_table_path}")
    print(f"apply_dust={bool(args.apply_dust)}")
    for result in results:
        positive = result.plot_data["total_burst"][1]
        print(
            f"{result.scenario.key}: M_up={result.popiii_upper_mass_msun:.6e} Msun, "
            f"max Phi_total_burst={np.nanmax(positive):.6e}"
        )


if __name__ == "__main__":
    main()
