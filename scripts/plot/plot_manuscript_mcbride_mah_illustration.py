from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from auroralf.mah.physics import mass_history
from auroralf.mah.sampling import sample_mcbride_appendix_a
from auroralf.mah.thesan import THESAN_MAH_CACHE_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THESAN_CACHE = (
    PROJECT_ROOT
    / "data_save"
    / "thesan_mah_cache"
    / "thesan-dark-1_LHaloTree_z11highmass_z10p980_n171_smoke.hdf5"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "manuscript"
    / "auroralf_project_summary"
    / "assets"
    / "mcbride_mah_sampling_illustration.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the manuscript figure illustrating McB09 MAH sampling and a THESAN high-redshift check."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thesan-cache", type=Path, default=DEFAULT_THESAN_CACHE)
    parser.add_argument("--final-log-mass", type=float, default=10.25)
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.15)
    parser.add_argument("--min-thesan-candidates", type=int, default=20)
    parser.add_argument("--mcbride-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.final_log_mass):
        raise ValueError("--final-log-mass must be finite")
    if args.mass_bin_width_dex <= 0.0 or not np.isfinite(args.mass_bin_width_dex):
        raise ValueError("--mass-bin-width-dex must be finite and positive")
    if args.min_thesan_candidates <= 0:
        raise ValueError("--min-thesan-candidates must be positive")
    if args.mcbride_sample_size <= 0:
        raise ValueError("--mcbride-sample-size must be positive")


def percentile_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.percentile(values, [16.0, 50.0, 84.0], axis=0)
    return low, median, high


def nan_percentile_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.nanpercentile(values, [16.0, 50.0, 84.0], axis=0)
    return low, median, high


def resolve_thesan_cache(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        resolved = expanded.resolve()
    else:
        resolved = (PROJECT_ROOT / expanded).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"THESAN MAH cache not found: {resolved}")
    return resolved


def load_thesan_comparison(args: argparse.Namespace) -> dict[str, object]:
    thesan_cache = resolve_thesan_cache(args.thesan_cache)
    with h5py.File(thesan_cache, "r") as handle:
        required = ("z_grid", "t_gyr_grid", "mass_ratio", "resolved_mask", "logM_final")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"THESAN MAH cache is missing required datasets: {missing}")
        schema_version = str(handle.attrs.get("schema_version", ""))
        mass_unit = str(handle.attrs.get("mass_unit", ""))
        source_simulation = str(handle.attrs.get("source_simulation", "unknown"))
        source_tree = str(handle.attrs.get("source_tree", "unknown"))
        z_grid = np.asarray(handle["z_grid"], dtype=float)
        z_final = float(handle.attrs.get("z_final", z_grid[-1]))
        t_gyr_grid = np.asarray(handle["t_gyr_grid"], dtype=float)
        mass_ratio = np.asarray(handle["mass_ratio"], dtype=float)
        resolved_mask = np.asarray(handle["resolved_mask"], dtype=bool)
        logm_final = np.asarray(handle["logM_final"], dtype=float)

    if schema_version != THESAN_MAH_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unexpected THESAN MAH cache schema_version: {schema_version!r}")
    if mass_unit != "Msun":
        raise ValueError(f"THESAN MAH cache mass_unit must be 'Msun'; got {mass_unit!r}")
    if z_grid.ndim != 1 or z_grid.size < 2:
        raise ValueError("THESAN MAH cache z_grid must be a 1D array with at least two entries")
    if t_gyr_grid.shape != z_grid.shape:
        raise ValueError("THESAN MAH cache t_gyr_grid must match z_grid")
    if mass_ratio.ndim != 2 or mass_ratio.shape[1] != z_grid.size:
        raise ValueError("THESAN MAH cache mass_ratio must have shape (n_halos, n_steps)")
    if resolved_mask.shape != mass_ratio.shape:
        raise ValueError("THESAN MAH cache resolved_mask must match mass_ratio")
    if logm_final.ndim != 1 or logm_final.size != mass_ratio.shape[0]:
        raise ValueError("THESAN MAH cache logM_final must match mass_ratio rows")
    if np.any(np.diff(z_grid) >= 0.0):
        raise ValueError("THESAN MAH cache z_grid must be strictly decreasing")
    if np.any(np.diff(t_gyr_grid) <= 0.0):
        raise ValueError("THESAN MAH cache t_gyr_grid must be strictly increasing")
    if not np.isclose(z_final, float(z_grid[-1]), rtol=0.0, atol=1.0e-3):
        raise ValueError("THESAN MAH cache z_final attribute does not match z_grid[-1]")
    resolved_values = mass_ratio[resolved_mask]
    if resolved_values.size == 0:
        raise ValueError("THESAN MAH cache has no resolved mass-ratio values")
    if not np.all(np.isfinite(resolved_values)) or np.any(resolved_values <= 0.0):
        raise ValueError("resolved THESAN MAH cache mass_ratio values must be finite and positive")
    if not np.all(np.isfinite(logm_final)):
        raise ValueError("THESAN MAH cache logM_final values must be finite")

    resolved_count = np.count_nonzero(resolved_mask, axis=1)
    selected = (np.abs(logm_final - float(args.final_log_mass)) <= float(args.mass_bin_width_dex)) & (
        resolved_count >= 2
    ) & resolved_mask[:, -1]
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size < int(args.min_thesan_candidates):
        raise ValueError(
            "THESAN candidate count "
            f"{selected_indices.size} is below min_thesan_candidates={int(args.min_thesan_candidates)} "
            f"for log10(Mh_final)={float(args.final_log_mass):.3f} within "
            f"{float(args.mass_bin_width_dex):.3f} dex"
        )

    selected_ratio = mass_ratio[selected_indices].copy()
    selected_resolved = resolved_mask[selected_indices]
    final_ratio = selected_ratio[:, -1]
    if np.any(final_ratio <= 0.0) or not np.all(np.isfinite(final_ratio)):
        raise ValueError("selected THESAN final mass ratios must be finite and positive")
    selected_ratio /= final_ratio[:, None]
    selected_ratio[:, -1] = 1.0
    selected_ratio = np.where(selected_resolved, selected_ratio, np.nan)

    resolved_per_snapshot = np.count_nonzero(np.isfinite(selected_ratio), axis=0)
    if np.any(resolved_per_snapshot < int(args.min_thesan_candidates)):
        raise ValueError(
            "selected THESAN sample has too few resolved tracks at one or more snapshots: "
            + ", ".join(str(int(value)) for value in resolved_per_snapshot)
        )

    return {
        "cache_path": thesan_cache,
        "source_simulation": source_simulation,
        "source_tree": source_tree,
        "z_grid": z_grid,
        "t_gyr_grid": t_gyr_grid,
        "mass_ratio": selected_ratio,
        "resolved_per_snapshot": resolved_per_snapshot.astype(int),
        "candidate_count": int(selected_indices.size),
        "logm_min": float(np.min(logm_final[selected_indices])),
        "logm_max": float(np.max(logm_final[selected_indices])),
    }


def plot_mcbride_illustration(args: argparse.Namespace) -> None:
    validate_args(args)
    thesan = load_thesan_comparison(args)

    thesan_z = np.asarray(thesan["z_grid"], dtype=float)
    z_final = float(thesan_z[-1])
    z_start = float(thesan_z[0])
    final_mass_msun = float(10.0 ** float(args.final_log_mass))
    n_thesan_tracks = int(thesan["candidate_count"])
    n_mcbride_samples = int(args.mcbride_sample_size)

    rng = np.random.default_rng(int(args.seed))
    samples = sample_mcbride_appendix_a(
        mass_ref=final_mass_msun,
        size=n_mcbride_samples,
        rng=rng,
    )
    beta = samples[:, 0]
    gamma = samples[:, 1]

    redshift = np.linspace(z_start, z_final, 240)
    mcbride_ratio = mass_history(
        redshift=redshift,
        redshift_final=z_final,
        mass_final=final_mass_msun,
        beta=beta,
        gamma=gamma,
    ) / final_mass_msun
    mcbride_at_thesan = mass_history(
        redshift=thesan_z,
        redshift_final=z_final,
        mass_final=final_mass_msun,
        beta=beta,
        gamma=gamma,
    ) / final_mass_msun

    if not np.all(np.isfinite(mcbride_ratio)) or np.any(mcbride_ratio <= 0.0):
        raise RuntimeError("McBride MAH sampling produced non-positive or non-finite mass ratios")

    mc_lo, mc_med, mc_hi = percentile_band(mcbride_ratio)
    mc_thesan_lo, mc_thesan_med, mc_thesan_hi = percentile_band(mcbride_at_thesan)
    thesan_ratio = np.asarray(thesan["mass_ratio"], dtype=float)
    thesan_lo, thesan_med, thesan_hi = nan_percentile_band(thesan_ratio)
    compare_mask = thesan_z > z_final + 1.0e-3
    median_offsets = np.abs(np.log10(mc_thesan_med[compare_mask] / thesan_med[compare_mask]))
    max_median_offset = float(np.max(median_offsets))
    lookback_myr = (float(np.asarray(thesan["t_gyr_grid"], dtype=float)[-1]) - np.asarray(thesan["t_gyr_grid"], dtype=float)) * 1.0e3
    recent_mask = compare_mask & (lookback_myr <= 100.0)
    if not np.any(recent_mask):
        raise RuntimeError("THESAN comparison grid has no snapshots within the last 100 Myr before z_final")
    recent_offsets = np.abs(np.log10(mc_thesan_med[recent_mask] / thesan_med[recent_mask]))
    max_recent_median_offset = float(np.max(recent_offsets))
    thesan_median_inside_mcbride_band = bool(
        np.all(
            (thesan_med[compare_mask] >= mc_thesan_lo[compare_mask])
            & (thesan_med[compare_mask] <= mc_thesan_hi[compare_mask])
        )
    )
    power_law = gamma == 0.0

    plt.style.use("apj")
    plt.rcParams.update(
        {
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.82, 1.18]},
    )

    ax = axes[0]
    ax.scatter(beta[~power_law], gamma[~power_law], s=10, color="#1f77b4", alpha=0.42, linewidths=0)
    if np.any(power_law):
        ax.scatter(beta[power_law], gamma[power_law], s=13, color="#d62728", alpha=0.75, linewidths=0)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("sampled McB09 parameters")
    ax.axhline(0.0, color="0.35", lw=0.7, ls=":")
    ax.text(0.03, 0.94, rf"$N_{{\rm McB09}}={n_mcbride_samples}$", transform=ax.transAxes, va="top", fontsize=7)

    history_indices = np.linspace(0, n_mcbride_samples - 1, min(70, n_mcbride_samples), dtype=int)

    ax = axes[1]
    for idx in history_indices:
        ax.plot(redshift, mcbride_ratio[idx], color="#4c78a8", alpha=0.07, lw=0.65, solid_capstyle="round")
    for row in range(thesan_ratio.shape[0]):
        valid = np.isfinite(thesan_ratio[row])
        if np.count_nonzero(valid) >= 2:
            ax.plot(thesan_z[valid], thesan_ratio[row, valid], color="#009e73", alpha=0.085, lw=0.65)
    ax.fill_between(redshift, mc_lo, mc_hi, color="#4c78a8", alpha=0.20, lw=0, label="McB09 16--84%")
    ax.plot(redshift, mc_med, color="#1f4e79", lw=1.6, label="McB09 median")
    ax.fill_between(thesan_z, thesan_lo, thesan_hi, color="#009e73", alpha=0.18, lw=0, label="THESAN 16--84%")
    ax.plot(thesan_z, thesan_med, color="#00785a", lw=1.35, marker="o", ms=4.0, label="THESAN median")
    ax.scatter(
        np.broadcast_to(thesan_z[None, :], thesan_ratio.shape)[np.isfinite(thesan_ratio)],
        thesan_ratio[np.isfinite(thesan_ratio)],
        s=8.0,
        color="#009e73",
        alpha=0.18,
        linewidths=0.0,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlim(z_start + 0.4, z_final - 0.3)
    ax.set_ylim(4.0e-3, 1.35)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$M_{\rm h}(z)/M_{\rm h}(z_f)$")
    ax.set_title(rf"MAH shape at $z_f={z_final:.2f}$")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.text(
        0.04,
        0.94,
        rf"$N_{{\rm THESAN}}={n_thesan_tracks},\ \log M_f={args.final_log_mass:.2f}\pm{args.mass_bin_width_dex:.2f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=6.4,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote_pdf={args.output}", flush=True)
    print(f"thesan_cache={thesan['cache_path']}", flush=True)
    print(f"thesan_candidate_count={n_thesan_tracks}", flush=True)
    print(f"mcbride_sample_size={n_mcbride_samples}", flush=True)
    print(f"thesan_z_final={z_final:.9f}", flush=True)
    print(f"max_median_offset_dex={max_median_offset:.6f}", flush=True)
    print(f"max_recent_100myr_median_offset_dex={max_recent_median_offset:.6f}", flush=True)
    print(f"thesan_median_inside_mcbride_band={thesan_median_inside_mcbride_band}", flush=True)


def main() -> None:
    args = parse_args()
    plot_mcbride_illustration(args)


if __name__ == "__main__":
    main()
