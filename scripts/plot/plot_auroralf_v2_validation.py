#!/usr/bin/env python3
"""Generate the numerical evidence and vector assets for the AuroraLF v2 review deck."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.io import (
    ArtifactProvenance,
    UVLFArtifact,
    read_uvlf_artifact,
    write_uvlf_artifact_atomic,
)
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import (
    compute_popiii_sfr_visbal2015_from_grids,
    compute_visbal2015_atomic_cooling_mass_msun,
)
from auroralf.ssp import DEFAULT_POPIII_UV_SSP_FILE
from auroralf.uvlf.dust import intrinsic_muv_from_observed, intrinsic_muv_jacobian
from auroralf.uvlf.hmf_sampling import sample_uvlf_from_hmf
from auroralf.uvlf.imf import DEFAULT_CANONICAL_SSP_FILE, DEFAULT_MILD_TOPHEAVY_SSP_FILE
from auroralf.uvlf.pipeline import _apply_burst_scatter_to_sfr_grid
from auroralf.uvlf.runner import run_uvlf_streaming


DEFAULT_DECK_DIR = PROJECT_ROOT / "slides" / "auroralf_v2_validation_20260710"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "auroralf_v2_validation_metrics.json"
DEFAULT_ARTIFACT = PROJECT_ROOT / "outputs" / "auroralf_v2_validation_artifact.h5"
DEFAULT_BENCHMARK = PROJECT_ROOT / "outputs" / "uvlf_v2_streaming_benchmark_20260711.json"
SEED_NAMESPACE = "auroralf.seeding.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--overwrite-artifact", action="store_true")
    return parser.parse_args()


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=500)
    plt.close(fig)


def _load_lw_module():
    path = PROJECT_ROOT / "scripts" / "analysis" / "plot_sfrd_lw_background_from_model.py"
    spec = importlib.util.spec_from_file_location("auroralf_lw_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load LW validation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validation_config(artifact_path: Path) -> UVLFRunConfig:
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id="auroralf-v2-validation",
        redshifts=(6.0,),
        base_seed=20260711,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(z_start_max=10.0, n_time_steps=12),
        star_formation=StarFormationConfig(enable_time_delay=False),
        stellar_population=StellarPopulationConfig(
            imf_modes=("canonical",),
            canonical_ssp_path=(PROJECT_ROOT / DEFAULT_CANONICAL_SSP_FILE).resolve(strict=True),
            topheavy_ssp_path=(PROJECT_ROOT / DEFAULT_MILD_TOPHEAVY_SSP_FILE).resolve(strict=True),
            popiii_ssp_path=(PROJECT_ROOT / DEFAULT_POPIII_UV_SSP_FILE).resolve(strict=True),
            birth_metallicity_topheavy_max_zsun=None,
            enable_popiii=False,
        ),
        sampling=SamplingConfig(
            mass_batch_size=2,
            n_halo_mass_samples=8,
            n_tracks_per_halo_mass=8,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=11.0,
            muv_bin_edges=tuple(np.linspace(-30.0, -10.0, 21)),
            workers=1,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
            apply_dust=False,
        ),
        output=OutputConfig(artifact_path.resolve()),
    )


def _git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return revision, dirty


def _dust_validation(asset_dir: Path) -> dict[str, float]:
    observed = np.linspace(-24.0, -14.0, 801)
    redshifts = (6.0, 10.0, 14.0)
    step = 1.0e-5
    maximum_absolute_error = 0.0

    fig, (axis, residual_axis) = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    colors = ("#1f77b4", "#d95f02", "#2ca02c")
    for redshift, color in zip(redshifts, colors, strict=True):
        analytic = np.asarray(intrinsic_muv_jacobian(observed, redshift), dtype=float)
        plus = np.asarray(intrinsic_muv_from_observed(observed + step, redshift), dtype=float)
        minus = np.asarray(intrinsic_muv_from_observed(observed - step, redshift), dtype=float)
        finite_difference = (plus - minus) / (2.0 * step)
        absolute_error = np.abs(finite_difference - analytic)
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(absolute_error)))
        axis.plot(observed, analytic, color=color, label=rf"analytic, $z={redshift:g}$")
        axis.plot(observed[::32], finite_difference[::32], "o", color=color, ms=2.5, mfc="none")
        residual_axis.semilogy(observed, np.maximum(absolute_error, 1.0e-15), color=color)

    axis.set_ylabel(r"$\mathrm{d}M_{\rm UV}/\mathrm{d}M_{\rm UV}^{\rm obs}$")
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    axis.text(0.02, 0.08, "lines: analytic; circles: centered finite difference", transform=axis.transAxes, fontsize=8)
    residual_axis.set_xlabel(r"Observed $M_{\rm UV}$ [mag]")
    residual_axis.set_ylabel("absolute error")
    residual_axis.set_ylim(1.0e-12, 1.0e-8)
    _save_figure(fig, asset_dir / "dust_jacobian_validation.pdf")
    return {"finite_difference_step_mag": step, "max_absolute_error": maximum_absolute_error}


def _physical_gates_validation(asset_dir: Path) -> dict[str, object]:
    cosmology = Cosmology()
    ratio = np.linspace(0.5, 2.5, 401)
    redshift_grid = np.full((1, ratio.size), 10.0)
    cooling_mass = compute_visbal2015_atomic_cooling_mass_msun(redshift_grid)
    halo_mass = ratio[None, :] * cooling_mass
    visbal = compute_popiii_sfr_visbal2015_from_grids(
        mh_grid=halo_mass,
        z_grid=redshift_grid,
        active_grid=np.ones_like(halo_mass, dtype=bool),
        fstar=0.1,
        eta_duty=1.0,
        cosmology=cosmology,
    )
    normalized_sfr = np.divide(
        visbal.sfr_grid[0],
        visbal.raw_sfr_scaling_grid[0],
        out=np.zeros_like(visbal.sfr_grid[0]),
        where=visbal.raw_sfr_scaling_grid[0] > 0.0,
    )

    histories = generate_halo_histories(
        n_tracks=128,
        z_final=10.0,
        Mh_final=1.0e10,
        cosmology=cosmology,
        z_start_max=20.0,
        custom_grid=np.linspace(20.0, 10.0, 64),
        time_grid_mode="custom",
        random_seed=20260711,
    )
    production_fraction = float(histories.metadata["negative_dmhdt_clip_fraction"])
    fixture_rates = {
        "McBride reduced run": np.asarray(histories.tracks["dMh_dt_raw"], dtype=float),
        "mixed-sign fixture": np.array([-2.0, -1.0, 3.0, 4.0, -5.0]),
        "all-negative fixture": np.array([-3.0, -2.0, -1.0]),
    }
    clipping_fractions = {
        label: float(np.count_nonzero(values < 0.0) / values.size)
        for label, values in fixture_rates.items()
    }

    fig, (gate_axis, clipping_axis) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    gate_axis.plot(ratio, normalized_sfr, color="#4c78a8", lw=2.2)
    gate_axis.axvline(1.0, color="0.35", ls="--", lw=1.0)
    gate_axis.axvline(2.0, color="0.35", ls="--", lw=1.0)
    gate_axis.fill_between(ratio, 0.0, 1.0, where=(ratio >= 1.0) & (ratio <= 2.0), color="#4c78a8", alpha=0.12)
    gate_axis.set(xlabel=r"$M_h/M_{\rm cool}$", ylabel="gated / raw Pop III SFR", xlim=(0.5, 2.5), ylim=(-0.05, 1.12))
    gate_axis.set_title(r"Visbal window: $1\leq M_h/M_{\rm cool}\leq2$")

    labels = tuple(clipping_fractions)
    values = np.asarray(tuple(clipping_fractions.values())) * 100.0
    bars = clipping_axis.bar(np.arange(len(labels)), values, color=("#59a14f", "#f28e2b", "#e15759"))
    clipping_axis.set_xticks(np.arange(len(labels)), ("McBride\nreduced", "mixed-sign\nfixture", "all-negative\nfixture"))
    clipping_axis.set_ylabel("clipped rate samples [%]")
    clipping_axis.set_ylim(0.0, 112.0)
    clipping_axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3)
    clipping_axis.set_title(r"Only $\dot M_h<0$ is clipped for SFR")
    _save_figure(fig, asset_dir / "physical_gates_and_clipping.pdf")

    return {
        "visbal_active_ratio_min": float(ratio[normalized_sfr > 0.0][0]),
        "visbal_active_ratio_max": float(ratio[normalized_sfr > 0.0][-1]),
        "mcbride_negative_clip_fraction": production_fraction,
        "clipping_fractions": clipping_fractions,
    }


def _burst_and_seed_validation() -> dict[str, object]:
    sfr = np.array(
        [[0.0, 1.0, 2.0, 4.0, 6.0], [0.0, 0.5, 1.0, 3.0, 5.0]],
        dtype=float,
    )
    time = np.array(
        [[0.00, 0.01, 0.03, 0.06, 0.10], [0.00, 0.02, 0.04, 0.07, 0.11]],
        dtype=float,
    )
    burst, multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr,
        active_grid=sfr > 0.0,
        t_grid=time,
        scatter_dex=0.8,
        correlation_timescale_myr=20.0,
        random_seed=31,
        preserve_mean=True,
    )
    original_mass = np.trapezoid(sfr, time, axis=1)
    burst_mass = np.trapezoid(burst, time, axis=1)
    relative_error = np.abs(burst_mass - original_mass) / original_mass

    coordinates = tuple((redshift, mass_index) for redshift in (6.0, 10.0, 14.0) for mass_index in range(8))
    first = tuple(derive_pipeline_random_seeds(20260711, redshift=z, mass_index=i) for z, i in coordinates)
    second = tuple(derive_pipeline_random_seeds(20260711, redshift=z, mass_index=i) for z, i in coordinates)
    different = tuple(derive_pipeline_random_seeds(20260712, redshift=z, mass_index=i) for z, i in coordinates)
    paired_equal = all(a == b for a, b in zip(first, second, strict=True))
    changed_seed_distinct = all(a != b for a, b in zip(first, different, strict=True))
    return {
        "burst_max_relative_mass_error": float(np.max(relative_error)),
        "burst_nontrivial_multiplier": bool(not np.allclose(multiplier[sfr > 0.0], 1.0)),
        "paired_coordinate_count": len(coordinates),
        "paired_seed_exact_equality": paired_equal,
        "changed_base_seed_all_distinct": changed_seed_distinct,
        "example_component_seeds": first[0].as_metadata(),
    }


def _cosmic_age_gyr(redshift: np.ndarray, cosmology: Cosmology) -> np.ndarray:
    z = np.asarray(redshift, dtype=float)
    if cosmology.omega_lambda <= 0.0:
        raise ValueError("analytic LW closure requires positive omega_lambda")
    coefficient = 2.0 / (3.0 * cosmology.h0 * np.sqrt(cosmology.omega_lambda))
    argument = np.sqrt(cosmology.omega_lambda / cosmology.omega_m) / np.power(1.0 + z, 1.5)
    return coefficient * np.arcsinh(argument)


def _lw_validation(asset_dir: Path) -> dict[str, float]:
    module = _load_lw_module()
    cosmology = Cosmology()
    evaluation = np.linspace(6.0, 20.0, 29)
    horizon_fraction = 0.04
    support = module._build_lw_support_grid(
        evaluation,
        horizon_fraction=horizon_fraction,
        z_start_max=30.0,
        support_dz=0.05,
        max_support_points=1024,
    )
    rho = np.ones_like(support)
    numerical = module._compute_lw_proxy(
        support,
        rho,
        evaluation_z=evaluation,
        cosmology=cosmology,
        horizon_fraction=horizon_fraction,
        dense_size=8192,
    )
    horizon = evaluation + horizon_fraction * (1.0 + evaluation)
    analytic = 1.0e9 * (_cosmic_age_gyr(evaluation, cosmology) - _cosmic_age_gyr(horizon, cosmology))
    relative_error = np.abs(numerical - analytic) / analytic

    fig, (main_axis, residual_axis) = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    main_axis.plot(evaluation, analytic / 1.0e6, color="#4c78a8", lw=2, label="analytic constant-SFRD closure")
    main_axis.plot(evaluation, numerical / 1.0e6, "o", color="#e45756", mfc="none", label="numerical horizon integral")
    main_axis.set_ylabel(r"LW lookback support [Myr]$\times\rho_{\rm SFRD}$")
    main_axis.legend(frameon=False)
    residual_axis.semilogy(evaluation, relative_error, color="#59a14f", marker="o", ms=3)
    residual_axis.set(xlabel="Evaluation redshift", ylabel="relative error")
    residual_axis.set_ylim(1.0e-10, 1.0e-5)
    _save_figure(fig, asset_dir / "lw_horizon_closure.pdf")
    return {
        "horizon_fraction": horizon_fraction,
        "support_point_count": int(support.size),
        "max_relative_error": float(np.max(relative_error)),
    }


def _v1_v2_and_artifact_validation(asset_dir: Path, artifact_path: Path, *, overwrite: bool) -> dict[str, object]:
    config = _validation_config(artifact_path)
    streaming = run_uvlf_streaming(config)
    legacy = sample_uvlf_from_hmf(
        z_obs=6.0,
        N_mass=config.sampling.n_halo_mass_samples,
        n_tracks=config.sampling.n_tracks_per_halo_mass,
        cosmology=config.cosmology.to_model(),
        base_seed=config.base_seed,
        quantity="Muv",
        bins=np.asarray(config.sampling.muv_bin_edges),
        logM_min=config.sampling.log10_halo_mass_min_msun,
        logM_max=config.sampling.log10_halo_mass_max_msun,
        z_start_max=config.mah.z_start_max,
        n_grid=config.mah.n_time_steps,
        sampler=config.mah.sampler,
        mah_backend=config.mah.backend,
        enable_time_delay=config.star_formation.enable_time_delay,
        pipeline_workers=1,
        ssp_file=str(config.stellar_population.canonical_ssp_path),
        topheavy_ssp_file=str(config.stellar_population.topheavy_ssp_path),
        topheavy_ssp_metallicity=config.stellar_population.topheavy_ssp_template_metallicity_zsun,
        enable_popiii=False,
        popiii_sfr_parameters=config.stellar_population.to_popiii_model(),
        popiii_ssp_file=str(config.stellar_population.popiii_ssp_path),
        imf_mode="canonical",
        imf_transition_parameters=config.stellar_population.to_imf_transition_model(),
        progress_path=None,
        print_progress=False,
        sfr_model_parameters=config.star_formation.to_model(),
        mass_function_model=config.sampling.mass_function_model,
        hmf_dlog10m=config.sampling.hmf_dlog10m,
        burst_scatter_dex=config.star_formation.burst_scatter_dex,
        burst_scatter_timescale_myr=config.star_formation.burst_scatter_correlation_timescale_myr,
        burst_scatter_preserve_mean=config.star_formation.burst_scatter_mass_conserving,
    )
    actual = streaming.for_redshift(6.0).for_mode("canonical")
    legacy_phi = np.asarray(legacy.uvlf["phi"], dtype=float)
    v2_phi = np.asarray(actual.phi_intrinsic_per_mpc3_per_mag, dtype=float)
    exact_counts = bool(np.array_equal(actual.raw_counts, legacy.uvlf["raw_counts"]))
    maximum_absolute_difference = float(np.max(np.abs(v2_phi - legacy_phi)))
    positive = (v2_phi > 0.0) & (legacy_phi > 0.0)
    maximum_relative_difference = float(np.max(np.abs(v2_phi[positive] - legacy_phi[positive]) / legacy_phi[positive]))

    centers = np.asarray(actual.bin_centers_muv)
    fig, (main_axis, residual_axis) = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    plot_mask = legacy_phi > 0.0
    main_axis.semilogy(centers[plot_mask], legacy_phi[plot_mask], color="#4c78a8", lw=2, label="v1 sampler")
    main_axis.semilogy(centers[plot_mask], v2_phi[plot_mask], "o", color="#e45756", mfc="none", label="v2 streaming")
    main_axis.set_ylabel(r"$\phi_{\rm UV}$ [Mpc$^{-3}$ mag$^{-1}$]")
    main_axis.legend(frameon=False)
    residual = np.full_like(v2_phi, np.nan)
    residual[positive] = (v2_phi[positive] - legacy_phi[positive]) / legacy_phi[positive]
    residual_axis.plot(centers[positive], residual[positive], "o", color="#59a14f", ms=4)
    residual_axis.axhline(0.0, color="0.3", lw=1)
    residual_axis.set(xlabel=r"$M_{\rm UV}$ [mag]", ylabel="(v2-v1) / v1")
    residual_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _save_figure(fig, asset_dir / "v1_v2_uvlf_equivalence.pdf")

    revision, dirty = _git_state()
    provenance = ArtifactProvenance.for_config(
        config,
        code_revision=revision,
        code_dirty=dirty,
        seed_namespace=SEED_NAMESPACE,
        source_paths=(("canonical_ssp", config.stellar_population.canonical_ssp_path),),
    )
    marker = artifact_path.with_name(artifact_path.name + ".complete")
    if artifact_path.exists() != marker.exists():
        raise RuntimeError(f"incomplete pre-existing validation artifact pair: {artifact_path}")
    if artifact_path.exists() and not overwrite:
        raise FileExistsError(f"validation artifact already exists: {artifact_path}; use --overwrite-artifact")
    write_uvlf_artifact_atomic(UVLFArtifact(result=streaming, provenance=provenance), overwrite=overwrite)
    readback = read_uvlf_artifact(artifact_path, load_samples=False)
    readback.provenance.verify_sources()
    if readback.result.config != config:
        raise RuntimeError("validation artifact config changed during strict readback")
    return {
        "raw_counts_exact": exact_counts,
        "positive_bin_count": int(np.count_nonzero(positive)),
        "max_absolute_phi_difference": maximum_absolute_difference,
        "max_relative_phi_difference": maximum_relative_difference,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "artifact_marker_path": str(marker.resolve()),
        "artifact_source_count": len(readback.provenance.source_checksums),
        "artifact_config_sha256": readback.provenance.config_sha256,
        "artifact_code_revision": readback.provenance.code_revision,
        "artifact_code_dirty": readback.provenance.code_dirty,
        "artifact_seed_namespace": readback.provenance.seed_namespace,
    }


def _memory_validation(asset_dir: Path, benchmark_path: Path) -> dict[str, object]:
    payload = json.loads(benchmark_path.resolve(strict=True).read_text(encoding="utf-8"))
    if payload.get("schema_name") != "auroralf.uvlf_v2_streaming_benchmark" or payload.get("complete") is not True:
        raise ValueError(f"invalid or incomplete streaming benchmark: {benchmark_path}")
    cases = payload["cases"]
    labels = ("serial\nno samples", "2 workers\nno samples", "2 workers\nwith samples")
    memory_mib = np.asarray([case["peak_rss_bytes"] for case in cases], dtype=float) / 2.0**20
    seconds = np.asarray([case["wall_seconds"] for case in cases], dtype=float)
    digests = {case["science_digest"] for case in cases}
    if len(digests) != 1 or payload.get("digest_equal") is not True:
        raise ValueError("benchmark science digests are not identical")

    fig, (memory_axis, time_axis) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    x = np.arange(len(labels))
    colors = ("#4c78a8", "#f2cf5b", "#e45756")
    memory_bars = memory_axis.bar(x, memory_mib, color=colors)
    memory_axis.set_xticks(x, labels)
    memory_axis.set_ylabel("peak RSS [MiB]")
    memory_axis.bar_label(memory_bars, labels=[f"{value:.0f}" for value in memory_mib], padding=3)
    memory_axis.set_ylim(0.0, float(np.max(memory_mib)) * 1.2)
    memory_axis.set_title("Bounded streaming memory")
    time_bars = time_axis.bar(x, seconds, color=colors)
    time_axis.set_xticks(x, labels)
    time_axis.set_ylabel("wall time [s]")
    time_axis.bar_label(time_bars, labels=[f"{value:.2f}" for value in seconds], padding=3)
    time_axis.set_ylim(0.0, float(np.max(seconds)) * 1.2)
    time_axis.set_title("Same science digest in all cases")
    _save_figure(fig, asset_dir / "streaming_memory_benchmark.pdf")
    return {
        "benchmark_path": str(benchmark_path.resolve()),
        "peak_rss_mib": memory_mib.tolist(),
        "wall_seconds": seconds.tolist(),
        "science_digest": next(iter(digests)),
        "digest_equal": True,
        "sample_storage_memory_overhead_fraction": float(payload["memory_overhead"]["parallel_samples_vs_parallel_disabled_ratio"] - 1.0),
        "slurm_job_id": payload["environment"]["env"]["SLURM_JOB_ID"],
    }


def main() -> None:
    args = _parse_args()
    if "apj" not in plt.style.available:
        raise RuntimeError("required Matplotlib style 'apj' is not installed")
    plt.style.use("apj")
    deck_dir = args.deck_dir.resolve()
    asset_dir = deck_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    args.metrics.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.artifact.resolve().parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "schema_name": "auroralf.v2_validation_metrics",
        "schema_version": "1.0.0",
        "dust": _dust_validation(asset_dir),
        "physical_gates": _physical_gates_validation(asset_dir),
        "burst_and_seeding": _burst_and_seed_validation(),
        "lw_horizon": _lw_validation(asset_dir),
        "v1_v2_and_artifact": _v1_v2_and_artifact_validation(
            asset_dir,
            args.artifact.resolve(),
            overwrite=args.overwrite_artifact,
        ),
        "memory": _memory_validation(asset_dir, args.benchmark),
    }
    args.metrics.resolve().write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
