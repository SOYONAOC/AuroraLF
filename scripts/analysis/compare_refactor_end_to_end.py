#!/usr/bin/env python3
"""Run and compare paired pre/post-refactor MAH -> SFR -> UVLF simulations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEFORE = PROJECT_ROOT / "outputs" / "refactor_e2e_before_main.npz"
DEFAULT_AFTER = PROJECT_ROOT / "outputs" / "refactor_e2e_after_current.npz"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "refactor_e2e_comparison_metrics.json"
DEFAULT_PNG = PROJECT_ROOT / "outputs" / "refactor_e2e_comparison.png"
DEFAULT_PDF = (
    PROJECT_ROOT
    / "slides"
    / "auroralf_v2_validation_20260710"
    / "assets"
    / "end_to_end_pre_post_comparison.pdf"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one implementation and write an NPZ result")
    run.add_argument("--implementation", choices=("before", "after"), required=True)
    run.add_argument("--source-label", required=True)
    run.add_argument("--expected-auroralf-root", type=Path, required=True)
    run.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--redshift", type=float, default=10.0)
    run.add_argument("--z-start-max", type=float, default=30.0)
    run.add_argument("--n-mass", type=int, default=32)
    run.add_argument("--n-tracks", type=int, default=48)
    run.add_argument("--n-grid", type=int, default=80)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--base-seed", type=int, default=20260711)
    run.add_argument("--logm-min", type=float, default=9.0)
    run.add_argument("--logm-max", type=float, default=12.0)
    run.add_argument("--muv-min", type=float, default=-28.0)
    run.add_argument("--muv-max", type=float, default=-10.0)
    run.add_argument("--n-muv-bins", type=int, default=24)
    run.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare", help="compare two NPZ results and plot them")
    compare.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    compare.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    compare.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    compare.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    compare.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser


def _source_tree_sha256(expected_root: Path) -> str:
    root = expected_root.expanduser().resolve(strict=True)
    files = sorted((root / "auroralf").rglob("*.py"))
    if not files:
        raise RuntimeError(f"no Python source files found under {root / 'auroralf'}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _assert_import_root(expected_root: Path) -> tuple[str, str]:
    import auroralf

    actual_file = Path(auroralf.__file__).resolve(strict=True)
    expected = expected_root.expanduser().resolve(strict=True)
    if not actual_file.is_relative_to(expected):
        raise RuntimeError(
            f"auroralf import root mismatch: expected under {expected}, imported {actual_file}"
        )
    return str(actual_file), _source_tree_sha256(expected)


def _reshape_regular(name: str, values: object, *, n_tracks: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size % n_tracks != 0:
        raise RuntimeError(
            f"{name} must be flat with a length divisible by n_tracks={n_tracks}; "
            f"received shape={array.shape}"
        )
    return np.asarray(array.reshape(n_tracks, -1))


def _run_pipeline(
    *,
    implementation: str,
    n_tracks: int,
    z_final: float,
    halo_mass_msun: float,
    z_start_max: float,
    n_grid: int,
    seed: int,
    canonical_ssp: Path,
    topheavy_ssp: Path,
    popiii_ssp: Path,
):
    from auroralf.mah import Cosmology
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    cosmology = Cosmology()
    common = dict(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=halo_mass_msun,
        z_start_max=z_start_max,
        n_grid=n_grid,
        ssp_file=str(canonical_ssp),
        topheavy_ssp_file=str(topheavy_ssp),
        popiii_ssp_file=str(popiii_ssp),
        imf_mode="canonical",
        sampler="mcbride",
        mah_backend="mcbride",
        enable_time_delay=True,
        enable_popiii=False,
        workers=1,
        burst_scatter_dex=0.0,
        burst_scatter_preserve_mean=True,
    )
    if implementation == "before":
        return run_halo_uv_pipeline(
            **common,
            cosmology=cosmology,
            random_seed=seed,
            metallicity_random_seed=seed,
            burst_scatter_random_seed=seed,
        )
    if implementation == "after":
        from auroralf.seeding import PipelineRandomSeeds

        return run_halo_uv_pipeline(
            **common,
            cosmology=cosmology,
            random_seeds=PipelineRandomSeeds(
                mah=seed,
                metallicity=seed,
                burst=seed,
            ),
        )
    raise RuntimeError(f"unexpected implementation after argparse validation: {implementation}")


def _hmf_dndm(
    halo_mass_msun: np.ndarray,
    *,
    redshift: float,
    implementation: str,
) -> np.ndarray:
    from auroralf.mah import Cosmology
    from auroralf.uvlf.hmf_sampling import compute_halo_mass_function_dndm

    if implementation == "before":
        legacy_compute_halo_mass_function_dndm = compute_halo_mass_function_dndm
        values = legacy_compute_halo_mass_function_dndm(
            halo_mass_msun,
            redshift,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
        )
    elif implementation == "after":
        values = compute_halo_mass_function_dndm(
            halo_mass_msun,
            redshift,
            cosmology=Cosmology(),
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.02,
        )
    else:
        raise RuntimeError(f"unexpected implementation after argparse validation: {implementation}")
    result = np.asarray(values, dtype=float)
    if result.shape != halo_mass_msun.shape or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("Reed07 HMF returned an invalid array")
    return result


def _representative_history(result: object, *, n_tracks: int) -> dict[str, np.ndarray]:
    tracks = result.sfr_tracks
    mass = _reshape_regular("Mh", tracks["Mh"], n_tracks=n_tracks).astype(float)
    sfr = _reshape_regular("SFR", tracks["SFR"], n_tracks=n_tracks).astype(float)
    redshift = _reshape_regular("z", tracks["z"], n_tracks=n_tracks).astype(float)
    active = np.asarray(result.active_grid, dtype=bool)
    if mass.shape != sfr.shape or mass.shape != redshift.shape or mass.shape != active.shape:
        raise RuntimeError("representative MAH/SFR grids do not have identical shapes")
    if not np.allclose(redshift, redshift[0], rtol=0.0, atol=0.0):
        raise RuntimeError("representative redshift grids differ between halo tracks")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise RuntimeError("representative MAH contains invalid halo masses")
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise RuntimeError("representative SFR contains invalid values")
    return {
        "history_redshift": redshift[0],
        "history_mass_grid_msun": mass,
        "history_sfr_grid_msun_per_yr": sfr,
        "history_active_grid": active,
        "history_mass_p16_msun": np.percentile(mass, 16.0, axis=0),
        "history_mass_p50_msun": np.percentile(mass, 50.0, axis=0),
        "history_mass_p84_msun": np.percentile(mass, 84.0, axis=0),
        "history_sfr_p16_msun_per_yr": np.percentile(sfr, 16.0, axis=0),
        "history_sfr_p50_msun_per_yr": np.percentile(sfr, 50.0, axis=0),
        "history_sfr_p84_msun_per_yr": np.percentile(sfr, 84.0, axis=0),
    }


def _run_mass_task(
    task: tuple[
        int,
        str,
        int,
        float,
        float,
        float,
        int,
        int,
        str,
        str,
        str,
        str,
    ],
) -> tuple[int, np.ndarray, np.ndarray, int, int, int, int]:
    (
        mass_index,
        implementation,
        n_tracks,
        redshift,
        halo_mass_msun,
        z_start_max,
        n_grid,
        seed,
        canonical_ssp,
        topheavy_ssp,
        popiii_ssp,
        expected_auroralf_root,
    ) = task
    expected_root = str(Path(expected_auroralf_root).resolve(strict=True))
    if expected_root not in sys.path:
        sys.path.insert(0, expected_root)
    result = _run_pipeline(
        implementation=implementation,
        n_tracks=n_tracks,
        z_final=redshift,
        halo_mass_msun=halo_mass_msun,
        z_start_max=z_start_max,
        n_grid=n_grid,
        seed=seed,
        canonical_ssp=Path(canonical_ssp),
        topheavy_ssp=Path(topheavy_ssp),
        popiii_ssp=Path(popiii_ssp),
    )
    luminosity = np.asarray(result.uv_luminosities, dtype=float)
    sfr_grid = _reshape_regular("SFR", result.sfr_tracks["SFR"], n_tracks=n_tracks)
    if luminosity.shape != (n_tracks,):
        raise RuntimeError("pipeline returned an unexpected luminosity shape")
    if not np.all(np.isfinite(luminosity)):
        raise RuntimeError(
            "pipeline returned non-finite UV luminosities: "
            f"mass_index={mass_index}, halo_mass_msun={halo_mass_msun:.17g}, "
            f"finite={int(np.count_nonzero(np.isfinite(luminosity)))}/{luminosity.size}, "
            f"nan={int(np.count_nonzero(np.isnan(luminosity)))}, "
            f"positive_infinity={int(np.count_nonzero(np.isposinf(luminosity)))}, "
            f"negative_infinity={int(np.count_nonzero(np.isneginf(luminosity)))}"
        )
    negative_uv_count = int(np.count_nonzero(luminosity < 0.0))
    if implementation == "after" and negative_uv_count > 0:
        raise RuntimeError(
            "refactored pipeline returned negative UV luminosities: "
            f"mass_index={mass_index}, halo_mass_msun={halo_mass_msun:.17g}, "
            f"negative_finite={negative_uv_count}"
        )
    final_sfr = np.asarray(sfr_grid[:, -1], dtype=float)
    if not np.all(np.isfinite(final_sfr)):
        raise RuntimeError(
            "pipeline returned non-finite final SFR values: "
            f"mass_index={mass_index}, halo_mass_msun={halo_mass_msun:.17g}, "
            f"finite={int(np.count_nonzero(np.isfinite(final_sfr)))}/{final_sfr.size}"
        )
    negative_final_sfr_count = int(np.count_nonzero(final_sfr < 0.0))
    if implementation == "after" and negative_final_sfr_count > 0:
        raise RuntimeError(
            "refactored pipeline returned negative final SFR values: "
            f"mass_index={mass_index}, halo_mass_msun={halo_mass_msun:.17g}, "
            f"negative_finite={negative_final_sfr_count}"
        )
    if implementation == "after":
        negative_clip_count = int(result.metadata["negative_dmhdt_clip_count"])
        negative_total_count = int(result.metadata["negative_dmhdt_total_count"])
    else:
        negative_clip_count = 0
        negative_total_count = 0
    return (
        mass_index,
        luminosity,
        final_sfr,
        negative_clip_count,
        negative_total_count,
        negative_uv_count,
        negative_final_sfr_count,
    )


def _run(args: argparse.Namespace) -> None:
    if args.n_mass <= 1 or args.n_tracks <= 1 or args.n_grid < 3:
        raise ValueError("n_mass and n_tracks must exceed 1 and n_grid must be at least 3")
    if not 0.0 <= args.redshift < args.z_start_max:
        raise ValueError("redshift must be non-negative and below z_start_max")
    if args.logm_max <= args.logm_min:
        raise ValueError("logm_max must exceed logm_min")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.muv_max <= args.muv_min or args.n_muv_bins < 2:
        raise ValueError("muv_max must exceed muv_min and n_muv_bins must be at least 2")

    expected_root = args.expected_auroralf_root.expanduser().resolve(strict=True)
    if str(expected_root) not in sys.path:
        sys.path.insert(0, str(expected_root))
    imported_file, source_tree_sha256 = _assert_import_root(expected_root)
    from auroralf.uvlf.dust import compute_dust_attenuated_uvlf
    from auroralf.uvlf.hmf_sampling import uv_luminosity_to_muv
    from auroralf.uvlf.imf import DEFAULT_CANONICAL_SSP_FILE, DEFAULT_MILD_TOPHEAVY_SSP_FILE
    from auroralf.ssp import DEFAULT_POPIII_UV_SSP_FILE

    data_root = args.data_root.expanduser().resolve(strict=True)
    canonical_ssp = (data_root / DEFAULT_CANONICAL_SSP_FILE).resolve(strict=True)
    topheavy_ssp = (data_root / DEFAULT_MILD_TOPHEAVY_SSP_FILE).resolve(strict=True)
    popiii_ssp = (data_root / DEFAULT_POPIII_UV_SSP_FILE).resolve(strict=True)
    bin_edges = np.linspace(args.muv_min, args.muv_max, args.n_muv_bins + 1)

    started = time.perf_counter()
    representative = _run_pipeline(
        implementation=args.implementation,
        n_tracks=args.n_tracks,
        z_final=args.redshift,
        halo_mass_msun=1.0e10,
        z_start_max=args.z_start_max,
        n_grid=args.n_grid,
        seed=args.base_seed,
        canonical_ssp=canonical_ssp,
        topheavy_ssp=topheavy_ssp,
        popiii_ssp=popiii_ssp,
    )
    history = _representative_history(representative, n_tracks=args.n_tracks)

    rng = np.random.default_rng(args.base_seed)
    sampled_logm = rng.uniform(args.logm_min, args.logm_max, size=args.n_mass)
    sampled_mass = np.power(10.0, sampled_logm)
    dndm = _hmf_dndm(
        sampled_mass,
        redshift=args.redshift,
        implementation=args.implementation,
    )
    dndlogm = sampled_mass * np.log(10.0) * dndm
    mass_weight = (args.logm_max - args.logm_min) * dndlogm / args.n_mass

    sample_count = args.n_mass * args.n_tracks
    luminosity = np.empty(sample_count, dtype=float)
    final_sfr = np.empty(sample_count, dtype=float)
    sample_weight = np.empty(sample_count, dtype=float)
    negative_clip_count = 0
    negative_total_count = 0
    negative_uv_luminosity_count = 0
    negative_final_sfr_count = 0
    tasks = tuple(
        (
            mass_index,
            args.implementation,
            args.n_tracks,
            args.redshift,
            float(halo_mass),
            args.z_start_max,
            args.n_grid,
            args.base_seed + mass_index,
            str(canonical_ssp),
            str(topheavy_ssp),
            str(popiii_ssp),
            str(expected_root),
        )
        for mass_index, halo_mass in enumerate(sampled_mass)
    )
    if args.workers == 1:
        results = map(_run_mass_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(args.workers, args.n_mass),
            mp_context=mp.get_context("spawn"),
        )
        results = executor.map(_run_mass_task, tasks, chunksize=1)
    progress_stride = max(1, args.n_mass // 100)
    try:
        for completed, (
            mass_index,
            mass_luminosity,
            mass_final_sfr,
            mass_clip_count,
            mass_total_count,
            mass_negative_uv_count,
            mass_negative_final_sfr_count,
        ) in enumerate(results, start=1):
            start = mass_index * args.n_tracks
            stop = start + args.n_tracks
            luminosity[start:stop] = mass_luminosity
            final_sfr[start:stop] = mass_final_sfr
            sample_weight[start:stop] = mass_weight[mass_index] / args.n_tracks
            negative_clip_count += mass_clip_count
            negative_total_count += mass_total_count
            negative_uv_luminosity_count += mass_negative_uv_count
            negative_final_sfr_count += mass_negative_final_sfr_count
            if completed == args.n_mass or completed % progress_stride == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"progress={completed}/{args.n_mass} "
                    f"elapsed_seconds={elapsed:.3f}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    muv = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
    finite = np.isfinite(muv)
    raw_counts, used_edges = np.histogram(muv[finite], bins=bin_edges)
    weighted_counts, weighted_edges = np.histogram(
        muv[finite], bins=bin_edges, weights=sample_weight[finite]
    )
    if not np.array_equal(used_edges, bin_edges) or not np.array_equal(weighted_edges, bin_edges):
        raise RuntimeError("UVLF histogram changed the configured bin edges")
    bin_width = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    phi = weighted_counts / bin_width
    dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=bin_centers,
        intrinsic_phi=phi,
        z=args.redshift,
        muv_obs=bin_centers,
        clip_to_bounds=False,
    )
    phi_observed = np.asarray(dust["phi_obs"], dtype=float)
    if not np.all(np.isfinite(phi_observed)) or np.any(phi_observed < 0.0):
        raise RuntimeError("dust transform returned an invalid observed UVLF")

    metadata = {
        "schema_name": "auroralf.refactor_end_to_end_run",
        "schema_version": "1.0.0",
        "implementation": args.implementation,
        "source_label": args.source_label,
        "imported_auroralf_file": imported_file,
        "source_tree_sha256": source_tree_sha256,
        "redshift": args.redshift,
        "z_start_max": args.z_start_max,
        "n_mass": args.n_mass,
        "n_tracks": args.n_tracks,
        "n_grid": args.n_grid,
        "workers": args.workers,
        "base_seed": args.base_seed,
        "logm_min": args.logm_min,
        "logm_max": args.logm_max,
        "muv_min": args.muv_min,
        "muv_max": args.muv_max,
        "n_muv_bins": args.n_muv_bins,
        "imf_mode": "canonical",
        "mah_backend": "mcbride",
        "mass_function_model": "hmf_reed07",
        "enable_time_delay": True,
        "apply_dust": True,
        "sample_count": int(luminosity.size),
        "negative_dmhdt_clip_count": negative_clip_count if args.implementation == "after" else None,
        "negative_dmhdt_total_count": negative_total_count if args.implementation == "after" else None,
        "negative_uv_luminosity_count": negative_uv_luminosity_count,
        "negative_final_sfr_count": negative_final_sfr_count,
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("output must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; use --overwrite explicitly")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"owned temporary output unexpectedly exists: {temporary}")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        sampled_log10_halo_mass=sampled_logm,
        sampled_halo_mass_msun=sampled_mass,
        hmf_dndm_per_mpc3_per_msun=dndm,
        mass_weight_per_mpc3=mass_weight,
        sample_uv_luminosity_erg_per_s_hz=luminosity,
        sample_final_sfr_msun_per_yr=final_sfr,
        sample_weight_per_mpc3=sample_weight,
        sample_muv=muv,
        bin_edges_muv=bin_edges,
        bin_centers_muv=bin_centers,
        raw_counts=raw_counts,
        phi_intrinsic_per_mpc3_per_mag=phi,
        phi_observed_per_mpc3_per_mag=phi_observed,
        **history,
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if output.exists() and not args.overwrite:
        temporary.unlink()
        raise FileExistsError(f"output appeared during run: {output}")
    os.replace(temporary, output)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    resolved = path.expanduser().resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as handle:
        if "metadata_json" not in handle.files:
            raise ValueError(f"missing metadata_json in {resolved}")
        metadata = json.loads(str(handle["metadata_json"].item()))
        arrays = {
            name: np.array(handle[name], copy=True)
            for name in handle.files
            if name != "metadata_json"
        }
    if metadata.get("schema_name") != "auroralf.refactor_end_to_end_run":
        raise ValueError(f"unexpected result schema in {resolved}")
    return metadata, arrays


def _symmetric_relative(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    first = np.asarray(before, dtype=float)
    second = np.asarray(after, dtype=float)
    denominator = np.abs(first) + np.abs(second)
    result = np.zeros_like(first)
    valid = denominator > 0.0
    result[valid] = 2.0 * np.abs(second[valid] - first[valid]) / denominator[valid]
    return result


def _maximum_relative(before: np.ndarray, after: np.ndarray) -> float:
    first = np.asarray(before, dtype=float)
    second = np.asarray(after, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second) & (first != 0.0)
    if not np.any(valid):
        raise RuntimeError("relative comparison has no finite nonzero baseline values")
    return float(np.max(np.abs(second[valid] - first[valid]) / np.abs(first[valid])))


def _assert_same_configuration(before: dict[str, object], after: dict[str, object]) -> None:
    ignored = {
        "implementation",
        "source_label",
        "imported_auroralf_file",
        "source_tree_sha256",
        "negative_dmhdt_clip_count",
        "negative_dmhdt_total_count",
        "negative_uv_luminosity_count",
        "negative_final_sfr_count",
        "wall_seconds",
    }
    before_common = {key: value for key, value in before.items() if key not in ignored}
    after_common = {key: value for key, value in after.items() if key not in ignored}
    if before_common != after_common:
        raise ValueError("pre/post result configurations are not identical")


def _compare(args: argparse.Namespace) -> None:
    before_meta, before = _load_npz(args.before)
    after_meta, after = _load_npz(args.after)
    _assert_same_configuration(before_meta, after_meta)
    if before_meta["implementation"] != "before" or after_meta["implementation"] != "after":
        raise ValueError("comparison inputs are not ordered as before/after")
    required_equal = (
        "sampled_log10_halo_mass",
        "sampled_halo_mass_msun",
        "bin_edges_muv",
        "bin_centers_muv",
        "history_redshift",
    )
    for name in required_equal:
        if not np.array_equal(before[name], after[name]):
            raise ValueError(f"paired comparison axis differs: {name}")

    from auroralf.uvlf.dust import compute_dust_attenuated_uvlf

    before_with_after_dust = np.asarray(
        compute_dust_attenuated_uvlf(
            intrinsic_muv=before["bin_centers_muv"],
            intrinsic_phi=before["phi_intrinsic_per_mpc3_per_mag"],
            z=float(before_meta["redshift"]),
            muv_obs=before["bin_centers_muv"],
            clip_to_bounds=False,
        )["phi_obs"],
        dtype=float,
    )
    supported = np.asarray(before["raw_counts"] > 0, dtype=bool)
    if np.count_nonzero(supported) < 2:
        raise RuntimeError("UVLF comparison requires at least two sample-supported bins")
    hmf_relative = np.abs(
        after["hmf_dndm_per_mpc3_per_msun"]
        / before["hmf_dndm_per_mpc3_per_msun"]
        - 1.0
    )
    before_sfr = before["sample_final_sfr_msun_per_yr"]
    after_sfr = after["sample_final_sfr_msun_per_yr"]
    sfr_changed = ~np.isclose(
        before_sfr,
        after_sfr,
        rtol=1.0e-12,
        atol=0.0,
        equal_nan=True,
    )
    before_uv = before["sample_uv_luminosity_erg_per_s_hz"]
    after_uv = after["sample_uv_luminosity_erg_per_s_hz"]
    uv_changed = ~np.isclose(
        before_uv,
        after_uv,
        rtol=1.0e-12,
        atol=0.0,
        equal_nan=True,
    )
    positive_uv_pair = (before_uv > 0.0) & (after_uv > 0.0)
    positive_uv_relative = np.abs(
        after_uv[positive_uv_pair] - before_uv[positive_uv_pair]
    ) / before_uv[positive_uv_pair]
    before_muv = before["sample_muv"]
    after_muv = after["sample_muv"]
    muv_min = float(before_meta["muv_min"])
    muv_max = float(before_meta["muv_max"])
    in_bin_pair = (
        positive_uv_pair
        & np.isfinite(before_muv)
        & np.isfinite(after_muv)
        & (before_muv >= muv_min)
        & (before_muv <= muv_max)
        & (after_muv >= muv_min)
        & (after_muv <= muv_max)
    )
    in_bin_uv_relative = np.abs(after_uv[in_bin_pair] - before_uv[in_bin_pair]) / before_uv[
        in_bin_pair
    ]
    if in_bin_uv_relative.size == 0:
        raise RuntimeError("full comparison has no paired UV samples inside configured bins")

    metrics = {
        "schema_name": "auroralf.refactor_end_to_end_comparison",
        "schema_version": "1.0.0",
        "configuration": {
            key: before_meta[key]
            for key in (
                "redshift",
                "z_start_max",
                "n_mass",
                "n_tracks",
                "n_grid",
                "sample_count",
                "base_seed",
                "logm_min",
                "logm_max",
                "imf_mode",
                "mah_backend",
                "mass_function_model",
                "enable_time_delay",
                "apply_dust",
            )
        },
        "before_wall_seconds": before_meta["wall_seconds"],
        "after_wall_seconds": after_meta["wall_seconds"],
        "mah_mass_grid_max_relative_difference": _maximum_relative(
            before["history_mass_grid_msun"], after["history_mass_grid_msun"]
        ),
        "sfr_grid_max_symmetric_relative_difference": float(
            np.max(
                _symmetric_relative(
                    before["history_sfr_grid_msun_per_yr"],
                    after["history_sfr_grid_msun_per_yr"],
                )
            )
        ),
        "sfr_zero_pattern_exact": bool(
            np.array_equal(
                before["history_sfr_grid_msun_per_yr"] == 0.0,
                after["history_sfr_grid_msun_per_yr"] == 0.0,
            )
        ),
        "full_sample_final_sfr_changed_count_rtol_1e_12": int(
            np.count_nonzero(sfr_changed)
        ),
        "full_sample_final_sfr_changed_fraction_rtol_1e_12": float(
            np.mean(sfr_changed)
        ),
        "hmf_dndm_max_relative_difference": _maximum_relative(
            before["hmf_dndm_per_mpc3_per_msun"],
            after["hmf_dndm_per_mpc3_per_msun"],
        ),
        "mass_weight_max_relative_difference": _maximum_relative(
            before["mass_weight_per_mpc3"],
            after["mass_weight_per_mpc3"],
        ),
        "hmf_dndm_median_absolute_relative_difference": float(np.median(hmf_relative)),
        "full_sample_uv_luminosity_changed_count_rtol_1e_12": int(
            np.count_nonzero(uv_changed)
        ),
        "full_sample_uv_luminosity_changed_fraction_rtol_1e_12": float(
            np.mean(uv_changed)
        ),
        "uv_luminosity_positive_pair_max_relative_difference": float(
            np.max(positive_uv_relative)
        ),
        "uv_luminosity_positive_pair_p99_9_relative_difference": float(
            np.percentile(positive_uv_relative, 99.9)
        ),
        "in_bin_positive_uv_pair_count": int(in_bin_uv_relative.size),
        "in_bin_uv_luminosity_changed_count_rtol_1e_12": int(
            np.count_nonzero(in_bin_uv_relative > 1.0e-12)
        ),
        "in_bin_uv_luminosity_max_relative_difference": float(
            np.max(in_bin_uv_relative)
        ),
        "raw_uvlf_counts_exact": bool(np.array_equal(before["raw_counts"], after["raw_counts"])),
        "intrinsic_uvlf_max_relative_difference": _maximum_relative(
            before["phi_intrinsic_per_mpc3_per_mag"],
            after["phi_intrinsic_per_mpc3_per_mag"],
        ),
        "intrinsic_uvlf_supported_max_relative_difference": _maximum_relative(
            before["phi_intrinsic_per_mpc3_per_mag"][supported],
            after["phi_intrinsic_per_mpc3_per_mag"][supported],
        ),
        "observed_uvlf_max_relative_difference": _maximum_relative(
            before["phi_observed_per_mpc3_per_mag"],
            after["phi_observed_per_mpc3_per_mag"],
        ),
        "observed_uvlf_supported_max_relative_difference": _maximum_relative(
            before["phi_observed_per_mpc3_per_mag"][supported],
            after["phi_observed_per_mpc3_per_mag"][supported],
        ),
        "dust_algorithm_only_supported_max_relative_difference": _maximum_relative(
            before["phi_observed_per_mpc3_per_mag"][supported],
            before_with_after_dust[supported],
        ),
        "hmf_effect_with_common_after_dust_supported_max_relative_difference": _maximum_relative(
            before_with_after_dust[supported],
            after["phi_observed_per_mpc3_per_mag"][supported],
        ),
        "wall_time_speedup_before_over_after": float(
            float(before_meta["wall_seconds"]) / float(after_meta["wall_seconds"])
        ),
        "after_negative_dmhdt_clip_count": after_meta["negative_dmhdt_clip_count"],
        "after_negative_dmhdt_total_count": after_meta["negative_dmhdt_total_count"],
        "after_negative_dmhdt_clip_fraction": float(
            int(after_meta["negative_dmhdt_clip_count"])
            / int(after_meta["negative_dmhdt_total_count"])
        ),
        "before_negative_uv_luminosity_count": before_meta.get(
            "negative_uv_luminosity_count"
        ),
        "after_negative_uv_luminosity_count": after_meta.get(
            "negative_uv_luminosity_count"
        ),
        "before_negative_final_sfr_count": before_meta.get(
            "negative_final_sfr_count"
        ),
        "after_negative_final_sfr_count": after_meta.get(
            "negative_final_sfr_count"
        ),
    }
    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    if "apj" not in plt.style.available:
        raise RuntimeError("required Matplotlib style 'apj' is unavailable")
    plt.style.use("apj")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 5.4),
        gridspec_kw={"height_ratios": (3.0, 1.25)},
        sharex="col",
    )
    colors = {"before": "#4c78a8", "after": "#e45756"}
    z = before["history_redshift"]
    for label, values, color, marker in (
        ("before: main", before, colors["before"], None),
        ("after: refactor", after, colors["after"], "o"),
    ):
        axes[0, 0].semilogy(
            z,
            values["history_mass_p50_msun"],
            color=color,
            marker=marker,
            markevery=7,
            ms=3,
            mfc="none",
            label=label,
        )
        positive_sfr = values["history_sfr_p50_msun_per_yr"] > 0.0
        axes[0, 1].semilogy(
            z[positive_sfr],
            values["history_sfr_p50_msun_per_yr"][positive_sfr],
            color=color,
            marker=marker,
            markevery=7,
            ms=3,
            mfc="none",
            label=label,
        )
        positive_phi = values["phi_intrinsic_per_mpc3_per_mag"] > 0.0
        axes[0, 2].semilogy(
            values["bin_centers_muv"][positive_phi],
            values["phi_intrinsic_per_mpc3_per_mag"][positive_phi],
            color=color,
            marker=marker,
            markevery=2,
            ms=3,
            mfc="none",
            label=label + " intrinsic",
        )
        positive_observed = values["phi_observed_per_mpc3_per_mag"] > 0.0
        axes[0, 2].semilogy(
            values["bin_centers_muv"][positive_observed],
            values["phi_observed_per_mpc3_per_mag"][positive_observed],
            color=color,
            ls="--",
            alpha=0.85,
            label=label + " dust",
        )

    axes[0, 0].set_ylabel(r"median $M_h$ [$M_\odot$]")
    axes[0, 0].set_title(r"MAH: $M_h(z=10)=10^{10}\,M_\odot$")
    axes[0, 1].set_ylabel(r"median SFR [$M_\odot\,{\rm yr}^{-1}$]")
    axes[0, 1].set_title("Delayed SFR on the same tracks")
    axes[0, 2].set_ylabel(r"$\phi_{\rm UV}$ [Mpc$^{-3}$ mag$^{-1}$]")
    axes[0, 2].set_title("Reed07-weighted UVLF at z=10")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=7, ncol=2)

    residuals = (
        _symmetric_relative(before["history_mass_p50_msun"], after["history_mass_p50_msun"]),
        _symmetric_relative(
            before["history_sfr_p50_msun_per_yr"], after["history_sfr_p50_msun_per_yr"]
        ),
        _symmetric_relative(
            before["phi_intrinsic_per_mpc3_per_mag"],
            after["phi_intrinsic_per_mpc3_per_mag"],
        ),
    )
    for axis, x, residual in zip(axes[1, :2], (z, z), residuals[:2], strict=True):
        finite = np.isfinite(residual)
        axis.semilogy(x[finite], np.maximum(residual[finite], 1.0e-16), color="#59a14f")
        axis.set_ylabel("sym. rel. diff.")
        axis.set_ylim(1.0e-16, 2.0)
    uv_intrinsic_residual = residuals[2]
    uv_observed_residual = _symmetric_relative(
        before["phi_observed_per_mpc3_per_mag"],
        after["phi_observed_per_mpc3_per_mag"],
    )
    for residual, color, label in (
        (uv_intrinsic_residual, "#59a14f", "intrinsic"),
        (uv_observed_residual, "#f28e2b", "dust"),
    ):
        finite = np.isfinite(residual)
        axes[1, 2].semilogy(
            before["bin_centers_muv"][finite],
            np.maximum(residual[finite], 1.0e-16),
            color=color,
            label=label,
        )
    axes[1, 2].set_ylabel("sym. rel. diff.")
    axes[1, 2].set_ylim(1.0e-4, 3.0e-1)
    axes[1, 2].legend(frameon=False, fontsize=7)
    axes[1, 0].set_xlabel("redshift")
    axes[1, 1].set_xlabel("redshift")
    axes[1, 2].set_xlabel(r"$M_{\rm UV}$ [mag]")
    axes[0, 0].invert_xaxis()
    axes[0, 1].invert_xaxis()
    axes[1, 0].invert_xaxis()
    axes[1, 1].invert_xaxis()

    pdf_path = args.pdf.expanduser().resolve()
    png_path = args.png.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", dpi=500)
    figure.savefig(png_path, bbox_inches="tight", dpi=500)
    plt.close(figure)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        _run(args)
    elif args.command == "compare":
        _compare(args)
    else:
        raise RuntimeError(f"unexpected command after argparse validation: {args.command}")


if __name__ == "__main__":
    main()
