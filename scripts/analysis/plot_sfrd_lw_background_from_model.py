#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auroralf.cooling import compute_popiii_lw_minimum_mass_msun
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.seeding import derive_hmf_mass_seed, derive_pipeline_random_seeds
from auroralf.sfr import DEFAULT_SFR_MODEL_PARAMETERS, PopIIISFRParameters, compute_popiii_sfr_from_grids
from auroralf.sfr.calculator import compute_sfr_from_tracks
from auroralf.uvlf.hmf_sampling import (
    DEFAULT_HMF_DLOG10M,
    DEFAULT_MASS_FUNCTION_MODEL,
    compute_halo_mass_function_dndm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "outputs" / "sfrd_lw_background_from_model"
DEFAULT_SLIDE_OUTPUT = (
    PROJECT_ROOT
    / "slides"
    / "group_meeting_popiii_20260622"
    / "assets"
    / "sfrd_lw_background_from_model_slide.pdf"
)
YEARS_PER_GYR = 1.0e9
SFRD_LW_ARTIFACT_SCHEMA_VERSION = "auroralf_sfrd_lw_proxy_v1"


@dataclass(frozen=True)
class SFRDPoint:
    z: float
    scenario: str
    lw_background_j21: float
    rho_sfr_popii: float
    rho_sfr_popiii: float
    popiii_minimum_mass_msun: float


@dataclass(frozen=True)
class SFRDRunProvenance:
    schema_version: str
    base_seed: int
    N_mass: int
    n_tracks: int
    n_grid: int
    logM_max: float
    z_start_max: float
    lw_horizon_fraction: float
    lw_proxy_dense_size: int
    lw_support_dz: float
    lw_support_max_points: int
    cosmology_h0_gyr_inv: float
    cosmology_h0_km_s_mpc: float
    cosmology_omega_m: float
    cosmology_omega_b: float
    cosmology_omega_lambda: float
    mass_function_model: str
    hmf_dlog10m: float
    fixed_lw_j21_values: tuple[float, ...]
    enable_time_delay: bool


def _parse_fixed_lw_j21_values(text: str) -> np.ndarray:
    values = np.asarray([float(item.strip()) for item in str(text).split(",") if item.strip()], dtype=float)
    if values.size == 0:
        raise ValueError("--fixed-lw-j21-values must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError("--fixed-lw-j21-values must be finite")
    if np.any(values < 0.0):
        raise ValueError("--fixed-lw-j21-values must be non-negative")
    return values


def _parse_z_values(args: argparse.Namespace) -> np.ndarray:
    if args.z_values is not None:
        values = np.asarray(
            [float(item.strip()) for item in str(args.z_values).split(",") if item.strip()],
            dtype=float,
        )
    else:
        values = np.linspace(float(args.z_min), float(args.z_max), int(args.n_z))
    if values.size < 2:
        raise ValueError("at least two redshift values are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("redshift values must be finite")
    if np.any(values < 0.0):
        raise ValueError("redshift values must be non-negative")
    values = np.unique(values)
    if values.size < 2:
        raise ValueError("at least two unique redshift values are required")
    return np.sort(values)


def _build_lw_support_grid(
    requested_z: np.ndarray,
    *,
    horizon_fraction: float,
    z_start_max: float,
    support_dz: float | None = None,
    max_support_points: int = 512,
) -> np.ndarray:
    requested = np.asarray(requested_z, dtype=float)
    if requested.ndim != 1 or requested.size < 2:
        raise ValueError("requested_z must be a 1D array with at least two values")
    if not np.all(np.isfinite(requested)) or np.any(requested < 0.0):
        raise ValueError("requested_z must be finite and non-negative")
    fraction = float(horizon_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("horizon_fraction must be finite and positive")
    if not np.isfinite(float(z_start_max)):
        raise ValueError("z_start_max must be finite")

    ordered = np.sort(requested)
    minimum_requested_spacing = float(np.min(np.diff(ordered)))
    if minimum_requested_spacing <= 0.0:
        raise ValueError("requested_z values must be unique")
    if support_dz is None:
        spacing = minimum_requested_spacing
    else:
        spacing = float(support_dz)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("support_dz must be finite and positive")
    if isinstance(max_support_points, (bool, np.bool_)) or not isinstance(
        max_support_points,
        (int, np.integer),
    ):
        raise ValueError("max_support_points must be a positive integer")
    maximum_points = int(max_support_points)
    if maximum_points <= 0:
        raise ValueError("max_support_points must be a positive integer")
    required_zmax = float(np.max(ordered + fraction * (1.0 + ordered)))
    if required_zmax <= float(ordered[-1]):
        raise ValueError(
            "LW horizon endpoint must be strictly above the maximum requested redshift"
        )
    if required_zmax >= float(z_start_max):
        raise ValueError(
            f"LW support requires zmax={required_zmax:.8g} "
            f"below z_start_max={float(z_start_max):.8g}"
        )
    extension_span = required_zmax - float(ordered[-1])
    spacing_ratio = extension_span / spacing
    if not np.isfinite(spacing_ratio):
        raise ValueError(
            f"LW support grid requires more than {maximum_points} points; "
            "set --lw-support-dz explicitly"
        )
    interior_count = max(0, int(np.ceil(spacing_ratio)) - 1)
    estimated_points = int(ordered.size) + interior_count + 1
    if estimated_points > maximum_points:
        raise ValueError(
            f"LW support grid requires {estimated_points} points, exceeding "
            f"max_support_points={maximum_points}; set --lw-support-dz explicitly"
        )
    extension = float(ordered[-1]) + spacing * np.arange(1, interior_count + 1, dtype=float)
    extension = extension[extension < required_zmax]
    return np.concatenate((ordered, extension, np.array([required_zmax], dtype=float)))


def _compute_lw_proxy(
    support_z: np.ndarray,
    rho_sfrd: np.ndarray,
    *,
    evaluation_z: np.ndarray,
    cosmology: Cosmology,
    horizon_fraction: float,
    dense_size: int = 4096,
) -> np.ndarray:
    redshift = np.asarray(support_z, dtype=float)
    rho = np.asarray(rho_sfrd, dtype=float)
    evaluation = np.asarray(evaluation_z, dtype=float)
    if redshift.shape != rho.shape:
        raise ValueError("support_z and rho_sfrd must have identical shapes")
    if redshift.ndim != 1:
        raise ValueError("support_z and rho_sfrd must be 1D")
    if redshift.size < 2:
        raise ValueError("at least two support redshift points are required")
    if evaluation.ndim != 1 or evaluation.size == 0:
        raise ValueError("evaluation_z must be a non-empty 1D array")
    if float(horizon_fraction) <= 0.0 or not np.isfinite(float(horizon_fraction)):
        raise ValueError("horizon_fraction must be finite and positive")
    if int(dense_size) < 128:
        raise ValueError("dense_size must be at least 128")
    if not np.all(np.isfinite(redshift)) or np.any(redshift < 0.0):
        raise ValueError("support_z must be finite and non-negative")
    if not np.all(np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("rho_sfrd must be finite and non-negative")
    if not np.all(np.isfinite(evaluation)) or np.any(evaluation < 0.0):
        raise ValueError("evaluation_z must be finite and non-negative")
    if np.any(np.diff(redshift) <= 0.0):
        raise ValueError("support_z must be strictly increasing")
    if np.any(np.diff(evaluation) <= 0.0):
        raise ValueError("evaluation_z must be strictly increasing")
    if float(evaluation[0]) < float(redshift[0]):
        raise ValueError(
            "LW support starts above an evaluation redshift: "
            f"required zmin={float(evaluation[0]):.8g}, provided zmin={float(redshift[0]):.8g}"
        )

    horizon_redshift = evaluation + float(horizon_fraction) * (1.0 + evaluation)
    required_zmax = float(np.max(horizon_redshift))
    provided_zmax = float(redshift[-1])
    if provided_zmax < required_zmax:
        raise ValueError(
            "LW support is insufficient: "
            f"required zmax={required_zmax:.8g}, provided zmax={provided_zmax:.8g}"
        )

    dense_support_z = np.linspace(float(redshift[0]), provided_zmax, int(dense_size))
    proxy = np.empty_like(evaluation, dtype=float)
    for index, (z_now, z_horizon) in enumerate(zip(evaluation, horizon_redshift, strict=True)):
        interior = dense_support_z[(dense_support_z > z_now) & (dense_support_z < z_horizon)]
        support_knots = redshift[(redshift >= z_now) & (redshift <= z_horizon)]
        integration_z = np.unique(
            np.concatenate(
                ([float(z_now)], interior, support_knots, [float(z_horizon)])
            )
        )
        if np.any(np.diff(integration_z) <= 0.0):
            raise RuntimeError("LW integration grid must be strictly increasing")
        integration_rho = np.interp(integration_z, redshift, rho)
        hubble_gyr = np.asarray(cosmology.hubble(integration_z), dtype=float)
        if not np.all(np.isfinite(hubble_gyr)) or np.any(hubble_gyr <= 0.0):
            raise RuntimeError("cosmology.hubble returned non-finite or non-positive values")
        dt_dz_gyr = 1.0 / ((1.0 + integration_z) * hubble_gyr)
        proxy[index] = YEARS_PER_GYR * float(
            np.trapezoid(integration_rho * dt_dz_gyr, x=integration_z)
        )

    if not np.all(np.isfinite(proxy)) or np.any(proxy < 0.0):
        raise RuntimeError("LW proxy calculation returned non-finite or negative values")
    return proxy


def _scenario_key(lw_background_j21: float) -> str:
    value = float(lw_background_j21)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("lw_background_j21 must be finite and non-negative")
    if value == 0.0:
        return "popiii_no_external_lw"
    return f"popiii_fixed_lw_j21_{format(value, '.17g')}"


def _scenario_label(scenario: str) -> str:
    if scenario == "popiii_no_external_lw":
        return "Pop III, no external LW"
    prefix = "popiii_fixed_lw_j21_"
    if scenario.startswith(prefix):
        value = float(scenario.removeprefix(prefix))
        return rf"Pop III, fixed $J_{{\rm LW,21}}={value:g}$"
    return scenario


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.N_mass) <= 0:
        raise ValueError("--N-mass must be positive")
    if int(args.n_tracks) <= 0:
        raise ValueError("--n-tracks must be positive")
    if int(args.n_grid) < 2:
        raise ValueError("--n-grid must be at least 2")
    if float(args.logM_max) <= 0.0:
        raise ValueError("--logM-max must be positive")
    if float(args.lw_horizon_fraction) <= 0.0:
        raise ValueError("--lw-horizon-fraction must be positive")
    if int(args.lw_proxy_dense_size) < 128:
        raise ValueError("--lw-proxy-dense-size must be at least 128")
    if args.lw_support_dz is not None:
        support_dz = float(args.lw_support_dz)
        if not np.isfinite(support_dz) or support_dz <= 0.0:
            raise ValueError("--lw-support-dz must be finite and positive")
    if int(args.lw_support_max_points) <= 0:
        raise ValueError("--lw-support-max-points must be positive")


def _final_grid_column(values: np.ndarray, *, n_tracks: int, n_steps: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected = int(n_tracks) * int(n_steps)
    if array.size != expected:
        raise RuntimeError(f"{name} has size {array.size}, expected {expected}")
    return array.reshape(int(n_tracks), int(n_steps))[:, -1]


def _integrate_sfrd_at_z(
    *,
    cosmology: Cosmology,
    z: float,
    popiii_sfr_parameters: PopIIISFRParameters,
    N_mass: int,
    n_tracks: int,
    n_grid: int,
    logM_max: float,
    z_start_max: float,
    base_seed: int,
    enable_time_delay: bool,
    mass_function_model: str,
    hmf_dlog10m: float,
) -> SFRDPoint:
    lw_background_j21 = float(popiii_sfr_parameters.lw_background_j21)
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(
            z,
            lw_background_j21=lw_background_j21,
        )
    )
    logM_min = float(np.log10(popiii_minimum_mass_msun))
    if float(logM_max) <= logM_min:
        raise ValueError(f"--logM-max must exceed log10(M_PopIII,min)={logM_min:.4f} at z={z:g}")

    hmf_mass_seed = derive_hmf_mass_seed(int(base_seed), float(z))
    rng = np.random.default_rng(hmf_mass_seed)
    log_mass = rng.uniform(logM_min, float(logM_max), size=int(N_mass))
    halo_mass = np.power(10.0, log_mass)
    dndm = np.asarray(
        compute_halo_mass_function_dndm(
            halo_mass,
            float(z),
            cosmology=cosmology,
            mass_function_model=mass_function_model,
            hmf_dlog10m=float(hmf_dlog10m),
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(dndm)) or np.any(dndm < 0.0):
        raise RuntimeError("halo mass function returned non-finite or negative values")
    dndlogm = halo_mass * np.log(10.0) * dndm
    mass_weight = (float(logM_max) - logM_min) * dndlogm / float(N_mass)

    t_start_gyr = float(cosmo_age_gyr(float(z_start_max), cosmology))
    t_end_gyr = float(cosmo_age_gyr(float(z), cosmology))
    dt_gyr = (t_end_gyr - t_start_gyr) / float(int(n_grid) - 1)
    if not np.isfinite(dt_gyr) or dt_gyr <= 0.0:
        raise RuntimeError("computed non-positive MAH time step")

    rho_sfr_popii = 0.0
    rho_sfr_popiii = 0.0
    for mass_index, (mass, weight) in enumerate(zip(halo_mass, mass_weight, strict=True)):
        pipeline_seeds = derive_pipeline_random_seeds(
            int(base_seed),
            redshift=float(z),
            mass_index=mass_index,
        )
        histories = generate_halo_histories(
            n_tracks=int(n_tracks),
            z_final=float(z),
            Mh_final=float(mass),
            z_start_max=float(z_start_max),
            M_min=lambda redshift: compute_popiii_lw_minimum_mass_msun(
                redshift,
                lw_background_j21=float(lw_background_j21),
            ),
            cosmology=cosmology,
            random_seed=pipeline_seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt_gyr,
            store_inactive_history=True,
            sampler="mcbride",
        )
        n_steps = int(histories.metadata["grid_size"])
        sfr_tracks = compute_sfr_from_tracks(
            histories.tracks,
            cosmology=cosmology,
            enable_time_delay=bool(enable_time_delay),
            model_parameters=DEFAULT_SFR_MODEL_PARAMETERS,
        )
        mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(int(n_tracks), n_steps)
        dmhdt_sfr_grid = np.asarray(sfr_tracks["dMh_dt_sfr"], dtype=float).reshape(int(n_tracks), n_steps)
        z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(int(n_tracks), n_steps)
        active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(int(n_tracks), n_steps)
        sfr_final = _final_grid_column(sfr_tracks["SFR"], n_tracks=int(n_tracks), n_steps=n_steps, name="SFR")
        popiii_result = compute_popiii_sfr_from_grids(
            mh_grid=mh_grid,
            dmhdt_sfr_grid=dmhdt_sfr_grid,
            z_grid=z_grid,
            active_grid=active_grid,
            cosmology=cosmology,
            parameters=popiii_sfr_parameters,
        )
        popiii_sfr_final = np.asarray(popiii_result.sfr_grid, dtype=float)[:, -1]
        if sfr_final.size != int(n_tracks) or popiii_sfr_final.size != int(n_tracks):
            raise RuntimeError("final SFR arrays do not match n_tracks")
        rho_sfr_popii += float(weight) * float(np.mean(sfr_final))
        rho_sfr_popiii += float(weight) * float(np.mean(popiii_sfr_final))

    scenario = _scenario_key(lw_background_j21)
    return SFRDPoint(
        z=float(z),
        scenario=scenario,
        lw_background_j21=float(lw_background_j21),
        rho_sfr_popii=float(rho_sfr_popii),
        rho_sfr_popiii=float(rho_sfr_popiii),
        popiii_minimum_mass_msun=popiii_minimum_mass_msun,
    )


def cosmo_age_gyr(redshift: float, cosmology: Cosmology) -> float:
    z_grid = np.linspace(float(redshift), 2000.0, 20000)
    hubble = np.asarray(cosmology.hubble(z_grid), dtype=float)
    integrand = 1.0 / ((1.0 + z_grid) * hubble)
    return float(np.trapezoid(integrand, x=z_grid))


def _build_npz_payload(
    *,
    requested_z: np.ndarray,
    support_z: np.ndarray,
    popii_requested_sfrd: np.ndarray,
    popii_support_sfrd: np.ndarray,
    popiii_requested_sfrd_by_scenario: dict[str, np.ndarray],
    popiii_support_sfrd_by_scenario: dict[str, np.ndarray],
    lw_proxy_by_series: dict[str, np.ndarray],
    provenance: SFRDRunProvenance,
) -> dict[str, np.ndarray]:
    if not isinstance(provenance, SFRDRunProvenance):
        raise TypeError("provenance must be an SFRDRunProvenance")
    requested = np.asarray(requested_z, dtype=float)
    support = np.asarray(support_z, dtype=float)
    if requested.ndim != 1 or requested.size < 2 or np.any(np.diff(requested) <= 0.0):
        raise ValueError("requested_z must be a strictly increasing 1D array")
    if support.ndim != 1 or support.size < 2 or np.any(np.diff(support) <= 0.0):
        raise ValueError("support_z must be a strictly increasing 1D array")
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(support)):
        raise ValueError("requested_z and support_z must be finite")
    if np.any(requested < 0.0) or np.any(support < 0.0):
        raise ValueError("requested_z and support_z must be non-negative")
    if not np.all(np.isin(requested, support)):
        raise ValueError("requested_z must be an exact float64 subset of support_z")
    horizon_fraction = float(provenance.lw_horizon_fraction)
    if not np.isfinite(horizon_fraction) or horizon_fraction <= 0.0:
        raise ValueError("provenance.lw_horizon_fraction must be finite and positive")
    required_zmax = float(
        np.max(requested + horizon_fraction * (1.0 + requested))
    )
    provided_zmax = float(support[-1])
    if provided_zmax < required_zmax:
        raise ValueError(
            "LW support is insufficient for artifact provenance: "
            f"required zmax={required_zmax:.8g}, provided zmax={provided_zmax:.8g}"
        )

    def require_series(name: str, values: np.ndarray, expected_size: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size != expected_size:
            raise ValueError(f"{name} must be 1D with size {expected_size}")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        return array

    requested_scenarios = set(popiii_requested_sfrd_by_scenario)
    support_scenarios = set(popiii_support_sfrd_by_scenario)
    if requested_scenarios != support_scenarios:
        raise ValueError("requested and support Pop III scenario sets must match")
    expected_proxy_series = {"popii", *requested_scenarios}
    if set(lw_proxy_by_series) != expected_proxy_series:
        raise ValueError("LW proxy series must contain popii and every Pop III scenario exactly")

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(provenance.schema_version),
        "requested_z": requested,
        "lw_support_z": support,
        "popii_requested_sfrd_msun_yr_mpc3": require_series(
            "popii_requested_sfrd",
            popii_requested_sfrd,
            requested.size,
        ),
        "popii_support_sfrd_msun_yr_mpc3": require_series(
            "popii_support_sfrd",
            popii_support_sfrd,
            support.size,
        ),
        "popii_cumulative_formed_mass_proxy_msun_mpc3": require_series(
            "popii_lw_proxy",
            lw_proxy_by_series["popii"],
            requested.size,
        ),
        "provenance_base_seed": np.asarray(provenance.base_seed, dtype=np.uint64),
        "provenance_N_mass": np.asarray(provenance.N_mass, dtype=np.int64),
        "provenance_n_tracks": np.asarray(provenance.n_tracks, dtype=np.int64),
        "provenance_n_grid": np.asarray(provenance.n_grid, dtype=np.int64),
        "provenance_logM_max": np.asarray(provenance.logM_max, dtype=float),
        "provenance_z_start_max": np.asarray(provenance.z_start_max, dtype=float),
        "provenance_lw_horizon_fraction": np.asarray(provenance.lw_horizon_fraction, dtype=float),
        "provenance_lw_proxy_dense_size": np.asarray(provenance.lw_proxy_dense_size, dtype=np.int64),
        "provenance_lw_support_dz": np.asarray(provenance.lw_support_dz, dtype=float),
        "provenance_lw_support_max_points": np.asarray(provenance.lw_support_max_points, dtype=np.int64),
        "provenance_cosmology_h0_gyr_inv": np.asarray(provenance.cosmology_h0_gyr_inv, dtype=float),
        "provenance_cosmology_h0_km_s_mpc": np.asarray(
            provenance.cosmology_h0_km_s_mpc,
            dtype=float,
        ),
        "provenance_cosmology_omega_m": np.asarray(provenance.cosmology_omega_m, dtype=float),
        "provenance_cosmology_omega_b": np.asarray(provenance.cosmology_omega_b, dtype=float),
        "provenance_cosmology_omega_lambda": np.asarray(
            provenance.cosmology_omega_lambda,
            dtype=float,
        ),
        "provenance_mass_function_model": np.asarray(provenance.mass_function_model),
        "provenance_hmf_dlog10m": np.asarray(provenance.hmf_dlog10m, dtype=float),
        "provenance_fixed_lw_background_j21": np.asarray(
            provenance.fixed_lw_j21_values,
            dtype=float,
        ),
        "provenance_enable_time_delay": np.asarray(provenance.enable_time_delay, dtype=bool),
    }
    for scenario in sorted(requested_scenarios):
        payload[f"{scenario}_requested_sfrd_msun_yr_mpc3"] = require_series(
            f"{scenario}_requested_sfrd",
            popiii_requested_sfrd_by_scenario[scenario],
            requested.size,
        )
        payload[f"{scenario}_support_sfrd_msun_yr_mpc3"] = require_series(
            f"{scenario}_support_sfrd",
            popiii_support_sfrd_by_scenario[scenario],
            support.size,
        )
        payload[f"{scenario}_cumulative_formed_mass_proxy_msun_mpc3"] = require_series(
            f"{scenario}_lw_proxy",
            lw_proxy_by_series[scenario],
            requested.size,
        )
    if any(array.dtype == object for array in payload.values()):
        raise RuntimeError("NPZ payload must not contain object arrays")
    return payload


def _write_csv(
    path: Path,
    points: list[SFRDPoint],
    lw_proxy_by_series: dict[str, np.ndarray],
    *,
    provenance: SFRDRunProvenance,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points_by_scenario: dict[str, list[SFRDPoint]] = {}
    for point in points:
        points_by_scenario.setdefault(point.scenario, []).append(point)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "component",
                "scenario",
                "z",
                "lw_background_j21",
                "rho_sfrd_msun_yr^-1_Mpc^-3",
                "cumulative_formed_mass_proxy_msun_Mpc^-3",
                "popiii_minimum_mass_msun",
                "schema_version",
                "base_seed",
                "N_mass",
                "n_tracks",
                "n_grid",
                "logM_max",
                "z_start_max",
                "lw_horizon_fraction",
                "lw_proxy_dense_size",
                "lw_support_dz",
                "lw_support_max_points",
                "cosmology_h0_gyr_inv",
                "cosmology_h0_km_s_mpc",
                "cosmology_omega_m",
                "cosmology_omega_b",
                "cosmology_omega_lambda",
                "mass_function_model",
                "hmf_dlog10m",
                "fixed_lw_background_j21",
                "enable_time_delay",
            ]
        )
        provenance_values = [
            provenance.schema_version,
            str(provenance.base_seed),
            str(provenance.N_mass),
            str(provenance.n_tracks),
            str(provenance.n_grid),
            format(provenance.logM_max, ".17g"),
            format(provenance.z_start_max, ".17g"),
            format(provenance.lw_horizon_fraction, ".17g"),
            str(provenance.lw_proxy_dense_size),
            format(provenance.lw_support_dz, ".17g"),
            str(provenance.lw_support_max_points),
            format(provenance.cosmology_h0_gyr_inv, ".17g"),
            format(provenance.cosmology_h0_km_s_mpc, ".17g"),
            format(provenance.cosmology_omega_m, ".17g"),
            format(provenance.cosmology_omega_b, ".17g"),
            format(provenance.cosmology_omega_lambda, ".17g"),
            provenance.mass_function_model,
            format(provenance.hmf_dlog10m, ".17g"),
            ";".join(format(value, ".17g") for value in provenance.fixed_lw_j21_values),
            "true" if provenance.enable_time_delay else "false",
        ]
        baseline = sorted(points_by_scenario["popiii_no_external_lw"], key=lambda item: item.z)
        for index, point in enumerate(baseline):
            writer.writerow(
                [
                    "Pop II",
                    "baseline",
                    f"{point.z:.8e}",
                    "0.00000000e+00",
                    f"{point.rho_sfr_popii:.8e}",
                    f"{lw_proxy_by_series['popii'][index]:.8e}",
                    "",
                    *provenance_values,
                ]
            )
        for scenario, scenario_points in sorted(points_by_scenario.items()):
            sorted_points = sorted(scenario_points, key=lambda item: item.z)
            for index, point in enumerate(sorted_points):
                writer.writerow(
                    [
                        "Pop III",
                        scenario,
                        f"{point.z:.8e}",
                        f"{point.lw_background_j21:.8e}",
                        f"{point.rho_sfr_popiii:.8e}",
                        f"{lw_proxy_by_series[scenario][index]:.8e}",
                        f"{point.popiii_minimum_mass_msun:.8e}",
                        *provenance_values,
                    ]
                )


def _positive_ylim(
    series: list[np.ndarray],
    *,
    floor_factor: float = 0.6,
    ceiling_factor: float = 1.8,
) -> tuple[float, float]:
    positive_parts = [np.asarray(values, dtype=float)[np.asarray(values, dtype=float) > 0.0] for values in series]
    positive = np.concatenate([part for part in positive_parts if part.size > 0])
    if positive.size == 0:
        raise RuntimeError("cannot choose log limits without positive values")
    ymin = 10.0 ** np.floor(np.log10(float(np.min(positive)) * floor_factor))
    ymax = 10.0 ** np.ceil(np.log10(float(np.max(positive)) * ceiling_factor))
    return ymin, ymax


def _resolve_log_ylim(explicit_ylim: tuple[float, float] | None, series: list[np.ndarray]) -> tuple[float, float]:
    if explicit_ylim is None:
        return _positive_ylim(series)
    ymin, ymax = explicit_ylim
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin <= 0.0 or ymax <= ymin:
        raise ValueError("explicit log y-limits must satisfy 0 < ymin < ymax")
    return float(ymin), float(ymax)


def _plot(
    *,
    z: np.ndarray,
    popii_sfrd: np.ndarray,
    popiii_sfrd_by_scenario: dict[str, np.ndarray],
    lw_proxy_by_series: dict[str, np.ndarray],
    output_prefix: Path,
    slide_output: Path | None,
    lw_horizon_fraction: float,
    plot_z_min: float | None,
    plot_z_max: float | None,
    sfrd_ylim: tuple[float, float] | None,
    lw_ylim: tuple[float, float] | None,
) -> None:
    plt.style.use("apj")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    ax_sfrd, ax_lw = axes

    ax_sfrd.plot(z, popii_sfrd, color="black", lw=2.1, label="Pop II")
    ax_lw.plot(z, lw_proxy_by_series["popii"], color="black", lw=2.1, label="Pop II")

    colors = ["#008EBC", "#E26D3D", "#C63737", "#6A994E"]
    linestyles = ["-.", ":", "--", "-"]
    for index, (scenario, values) in enumerate(sorted(popiii_sfrd_by_scenario.items())):
        color = colors[index % len(colors)]
        linestyle = linestyles[index % len(linestyles)]
        label = _scenario_label(scenario)
        ax_sfrd.plot(z, values, color=color, ls=linestyle, lw=2.0, label=label)
        ax_lw.plot(z, lw_proxy_by_series[scenario], color=color, ls=linestyle, lw=2.0, label=label)

    ax_sfrd.set_yscale("log")
    ax_lw.set_yscale("log")
    x_min = float(np.min(z)) if plot_z_min is None else float(plot_z_min)
    x_max = float(np.max(z)) if plot_z_max is None else float(plot_z_max)
    if x_max <= x_min:
        raise ValueError("plot_z_max must exceed plot_z_min")
    plot_mask = (z >= x_min) & (z <= x_max)
    if np.count_nonzero(plot_mask) < 2:
        raise ValueError("plot redshift range must contain at least two computed redshift points")
    ax_sfrd.set_xlim(x_min, x_max)
    ax_lw.set_xlim(x_min, x_max)
    ax_sfrd.set_ylim(
        *_resolve_log_ylim(
            sfrd_ylim,
            [popii_sfrd[plot_mask], *[values[plot_mask] for values in popiii_sfrd_by_scenario.values()]],
        )
    )
    ax_lw.set_ylim(*_resolve_log_ylim(lw_ylim, [values[plot_mask] for values in lw_proxy_by_series.values()]))
    ax_sfrd.set_xlabel(r"$z$")
    ax_lw.set_xlabel(r"$z$")
    ax_sfrd.set_ylabel(r"$\rho_\star(z)$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
    ax_lw.set_ylabel(
        r"$J_{\rm LW}^{\rm proxy}\propto\int\rho_{\rm SFR}\,dt$ "
        r"[M$_\odot$ Mpc$^{-3}$ per unit LW yield]"
    )
    ax_sfrd.legend(loc="best", frameon=False, fontsize=9.6)
    ax_lw.legend(loc="best", frameon=False, fontsize=9.6)
    ax_sfrd.grid(alpha=0.18)
    ax_lw.grid(alpha=0.18)
    ax_lw.text(
        0.03,
        0.04,
        rf"finite LW horizon: $\Delta z={lw_horizon_fraction:g}(1+z)$",
        transform=ax_lw.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=500)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=500)
    if slide_output is not None:
        slide_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(slide_output, dpi=500)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AuroraLF SFRD and finite-horizon LW proxy versus redshift.")
    parser.add_argument(
        "--z-values",
        default=None,
        help="Comma-separated redshift values. Overrides --z-min/--z-max/--n-z.",
    )
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=35.0)
    parser.add_argument("--n-z", type=int, default=13)
    parser.add_argument("--N-mass", type=int, default=48)
    parser.add_argument("--n-tracks", type=int, default=4)
    parser.add_argument("--n-grid", type=int, default=80)
    parser.add_argument("--logM-max", type=float, default=12.0)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--plot-z-min", type=float, default=None)
    parser.add_argument("--plot-z-max", type=float, default=None)
    parser.add_argument("--sfrd-ymin", type=float, default=None)
    parser.add_argument("--sfrd-ymax", type=float, default=None)
    parser.add_argument("--lw-ymin", type=float, default=None)
    parser.add_argument("--lw-ymax", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=260625)
    parser.add_argument("--disable-time-delay", action="store_true")
    parser.add_argument("--fixed-lw-j21-values", default="0.1")
    parser.add_argument("--lw-horizon-fraction", type=float, default=0.2)
    parser.add_argument("--lw-proxy-dense-size", type=int, default=4096)
    parser.add_argument("--lw-support-dz", type=float, default=None)
    parser.add_argument("--lw-support-max-points", type=int, default=512)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--slide-output", type=Path, default=DEFAULT_SLIDE_OUTPUT)
    parser.add_argument("--no-slide-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cosmology = Cosmology()
    _validate_args(args)
    z_values = _parse_z_values(args)
    lw_support_z = _build_lw_support_grid(
        z_values,
        horizon_fraction=float(args.lw_horizon_fraction),
        z_start_max=float(args.z_start_max),
        support_dz=None if args.lw_support_dz is None else float(args.lw_support_dz),
        max_support_points=int(args.lw_support_max_points),
    )
    effective_lw_support_dz = (
        float(np.min(np.diff(z_values)))
        if args.lw_support_dz is None
        else float(args.lw_support_dz)
    )
    fixed_lw_values = _parse_fixed_lw_j21_values(args.fixed_lw_j21_values)
    scenario_lw_values = np.unique(np.concatenate([np.array([0.0], dtype=float), fixed_lw_values]))
    output_prefix = _resolve_path(args.output_prefix)
    slide_output = None if args.no_slide_output else _resolve_path(args.slide_output)
    sfrd_ylim = None
    if args.sfrd_ymin is not None or args.sfrd_ymax is not None:
        if args.sfrd_ymin is None or args.sfrd_ymax is None:
            raise ValueError("--sfrd-ymin and --sfrd-ymax must be provided together")
        sfrd_ylim = (float(args.sfrd_ymin), float(args.sfrd_ymax))
    lw_ylim = None
    if args.lw_ymin is not None or args.lw_ymax is not None:
        if args.lw_ymin is None or args.lw_ymax is None:
            raise ValueError("--lw-ymin and --lw-ymax must be provided together")
        lw_ylim = (float(args.lw_ymin), float(args.lw_ymax))

    provenance = SFRDRunProvenance(
        schema_version=SFRD_LW_ARTIFACT_SCHEMA_VERSION,
        base_seed=int(args.random_seed),
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        n_grid=int(args.n_grid),
        logM_max=float(args.logM_max),
        z_start_max=float(args.z_start_max),
        lw_horizon_fraction=float(args.lw_horizon_fraction),
        lw_proxy_dense_size=int(args.lw_proxy_dense_size),
        lw_support_dz=effective_lw_support_dz,
        lw_support_max_points=int(args.lw_support_max_points),
        cosmology_h0_gyr_inv=float(cosmology.h0),
        cosmology_h0_km_s_mpc=float(cosmology.h0_km_s_mpc),
        cosmology_omega_m=float(cosmology.omega_m),
        cosmology_omega_b=float(cosmology.omega_b),
        cosmology_omega_lambda=float(cosmology.omega_lambda),
        mass_function_model=DEFAULT_MASS_FUNCTION_MODEL,
        hmf_dlog10m=float(DEFAULT_HMF_DLOG10M),
        fixed_lw_j21_values=tuple(float(value) for value in fixed_lw_values),
        enable_time_delay=not bool(args.disable_time_delay),
    )

    support_points: list[SFRDPoint] = []
    for lw_background_j21 in scenario_lw_values:
        popiii_sfr_parameters = PopIIISFRParameters(
            lw_background_j21=float(lw_background_j21),
        )
        for z in lw_support_z:
            print(
                f"computing z={z:g}, J_LW,21={lw_background_j21:g}, "
                f"N_mass={args.N_mass}, n_tracks={args.n_tracks}",
                flush=True,
            )
            support_points.append(
                _integrate_sfrd_at_z(
                    cosmology=cosmology,
                    z=float(z),
                    popiii_sfr_parameters=popiii_sfr_parameters,
                    N_mass=int(args.N_mass),
                    n_tracks=int(args.n_tracks),
                    n_grid=int(args.n_grid),
                    logM_max=float(args.logM_max),
                    z_start_max=float(args.z_start_max),
                    base_seed=int(args.random_seed),
                    enable_time_delay=not bool(args.disable_time_delay),
                    mass_function_model=DEFAULT_MASS_FUNCTION_MODEL,
                    hmf_dlog10m=DEFAULT_HMF_DLOG10M,
                )
            )

    points_by_scenario: dict[str, list[SFRDPoint]] = {}
    for point in support_points:
        points_by_scenario.setdefault(point.scenario, []).append(point)
    baseline_support = sorted(points_by_scenario["popiii_no_external_lw"], key=lambda item: item.z)
    baseline_by_z = {point.z: point for point in baseline_support}
    popii_support_sfrd = np.asarray([point.rho_sfr_popii for point in baseline_support], dtype=float)
    popii_sfrd = np.asarray([baseline_by_z[float(z)].rho_sfr_popii for z in z_values], dtype=float)
    popiii_support_sfrd_by_scenario = {
        scenario: np.asarray([point.rho_sfr_popiii for point in sorted(scenario_points, key=lambda item: item.z)])
        for scenario, scenario_points in sorted(points_by_scenario.items())
    }
    popiii_sfrd_by_scenario = {
        scenario: np.asarray(
            [
                {point.z: point for point in scenario_points}[float(z)].rho_sfr_popiii
                for z in z_values
            ],
            dtype=float,
        )
        for scenario, scenario_points in sorted(points_by_scenario.items())
    }
    requested_redshifts = {float(value) for value in z_values}
    points = [point for point in support_points if point.z in requested_redshifts]
    lw_proxy_by_series = {
        "popii": _compute_lw_proxy(
            lw_support_z,
            popii_support_sfrd,
            evaluation_z=z_values,
            cosmology=cosmology,
            horizon_fraction=float(args.lw_horizon_fraction),
            dense_size=int(args.lw_proxy_dense_size),
        )
    }
    for scenario, values in popiii_support_sfrd_by_scenario.items():
        lw_proxy_by_series[scenario] = _compute_lw_proxy(
            lw_support_z,
            values,
            evaluation_z=z_values,
            cosmology=cosmology,
            horizon_fraction=float(args.lw_horizon_fraction),
            dense_size=int(args.lw_proxy_dense_size),
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_prefix.with_suffix(".csv"),
        points,
        lw_proxy_by_series,
        provenance=provenance,
    )
    npz_payload = _build_npz_payload(
        requested_z=z_values,
        support_z=lw_support_z,
        popii_requested_sfrd=popii_sfrd,
        popii_support_sfrd=popii_support_sfrd,
        popiii_requested_sfrd_by_scenario=popiii_sfrd_by_scenario,
        popiii_support_sfrd_by_scenario=popiii_support_sfrd_by_scenario,
        lw_proxy_by_series=lw_proxy_by_series,
        provenance=provenance,
    )
    np.savez(output_prefix.with_suffix(".npz"), **npz_payload)
    _plot(
        z=z_values,
        popii_sfrd=popii_sfrd,
        popiii_sfrd_by_scenario=popiii_sfrd_by_scenario,
        lw_proxy_by_series=lw_proxy_by_series,
        output_prefix=output_prefix,
        slide_output=slide_output,
        lw_horizon_fraction=float(args.lw_horizon_fraction),
        plot_z_min=None if args.plot_z_min is None else float(args.plot_z_min),
        plot_z_max=None if args.plot_z_max is None else float(args.plot_z_max),
        sfrd_ylim=sfrd_ylim,
        lw_ylim=lw_ylim,
    )
    print(f"wrote {output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"wrote {output_prefix.with_suffix('.png')}", flush=True)
    print(f"wrote {output_prefix.with_suffix('.csv')}", flush=True)
    if slide_output is not None:
        print(f"wrote {slide_output}", flush=True)


if __name__ == "__main__":
    main()
