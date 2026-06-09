#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data_save"
    / "tng_merger_event_cache"
    / "TNG100-1-Dark_sublink_full_merger_events_logM10_pilot.hdf5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tng_merger_events"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diagnostics from a compact TNG merger event cache.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input merger event cache HDF5.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"TNG merger event cache not found: {path}")
    required = {
        "events/final_snapshot",
        "events/mass_ratio_peak_ordered",
        "events/event_z",
        "halos/final_snapshot",
        "halos/event_count",
        "halos/major_1to4_peak_count",
        "halos/major_1to10_peak_count",
    }
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")
        for name in sorted(required):
            out[name.replace("/", "_")] = np.asarray(handle[name])
        out["schema_version"] = np.asarray(handle.attrs.get("schema_version", "unknown"), dtype="S")
        out["source_simulation"] = np.asarray(handle.attrs.get("source_simulation", "unknown"), dtype="S")
    return out


def _final_snapshot_labels(final_snapshots: np.ndarray, event_z: np.ndarray, event_final_snapshots: np.ndarray) -> dict[int, str]:
    labels: dict[int, str] = {}
    for snap in sorted({int(value) for value in final_snapshots}, reverse=True):
        event_mask = event_final_snapshots == snap
        if np.any(event_mask):
            z_final = float(np.nanmin(event_z[event_mask]))
            labels[snap] = rf"$z_f\simeq{z_final:.1f}$"
        else:
            labels[snap] = f"snap {snap}"
    return labels


def _write_summary_csv(path: Path, data: dict[str, np.ndarray]) -> Path:
    csv_path = path / "tng_merger_event_cache_summary.csv"
    halo_snap = np.asarray(data["halos_final_snapshot"], dtype=int)
    event_snap = np.asarray(data["events_final_snapshot"], dtype=int)
    event_ratio = np.asarray(data["events_mass_ratio_peak_ordered"], dtype=float)
    event_count = np.asarray(data["halos_event_count"], dtype=float)
    major_1to4 = np.asarray(data["halos_major_1to4_peak_count"], dtype=float)
    major_1to10 = np.asarray(data["halos_major_1to10_peak_count"], dtype=float)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "final_snapshot",
                "n_halos",
                "n_events",
                "events_per_halo_mean",
                "events_per_halo_median",
                "halo_fraction_mu_peak_ge_0p25",
                "halo_fraction_mu_peak_ge_0p10",
                "event_mu_peak_q50",
                "event_mu_peak_q90",
                "event_mu_peak_q99",
            ]
        )
        for snap in sorted({int(value) for value in halo_snap}, reverse=True):
            hmask = halo_snap == snap
            emask = event_snap == snap
            ratios = event_ratio[emask]
            finite = ratios[np.isfinite(ratios)]
            if finite.size == 0:
                quantiles = [np.nan, np.nan, np.nan]
            else:
                quantiles = np.nanquantile(finite, [0.5, 0.9, 0.99])
            writer.writerow(
                [
                    snap,
                    int(np.count_nonzero(hmask)),
                    int(np.count_nonzero(emask)),
                    float(np.mean(event_count[hmask])),
                    float(np.median(event_count[hmask])),
                    float(np.mean(major_1to4[hmask] > 0.0)),
                    float(np.mean(major_1to10[hmask] > 0.0)),
                    float(quantiles[0]),
                    float(quantiles[1]),
                    float(quantiles[2]),
                ]
            )
    return csv_path


def _plot(data: dict[str, np.ndarray], output_dir: Path) -> tuple[Path, Path]:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    halo_snap = np.asarray(data["halos_final_snapshot"], dtype=int)
    event_snap = np.asarray(data["events_final_snapshot"], dtype=int)
    event_z = np.asarray(data["events_event_z"], dtype=float)
    event_ratio = np.asarray(data["events_mass_ratio_peak_ordered"], dtype=float)
    event_count = np.asarray(data["halos_event_count"], dtype=float)

    labels = _final_snapshot_labels(halo_snap, event_z, event_snap)
    snaps = list(labels.keys())
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    positions = np.arange(len(snaps), dtype=float)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.8), constrained_layout=True)
    ax_count, ax_ratio = axes

    count_values = [event_count[halo_snap == snap] for snap in snaps]
    parts = ax_count.violinplot(
        count_values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for color, body in zip(colors, parts["bodies"], strict=False):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.3)
    ax_count.set_xticks(positions, [labels[snap] for snap in snaps])
    ax_count.set_ylabel("events per halo")
    ax_count.set_title("Event counts")

    bins = np.geomspace(1.0e-2, 1.0, 32)
    for color, snap in zip(colors, snaps, strict=False):
        values = event_ratio[(event_snap == snap) & np.isfinite(event_ratio) & (event_ratio > 0.0)]
        if values.size == 0:
            raise ValueError(f"no finite positive peak mass ratios for final snapshot {snap}")
        ax_ratio.hist(
            values,
            bins=bins,
            histtype="step",
            lw=1.4,
            density=True,
            color=color,
            label=labels[snap],
        )
    ax_ratio.axvline(0.25, color="black", lw=1.0, ls=":", label=r"$\mu_{\rm peak}=1/4$")
    ax_ratio.axvline(0.10, color="0.35", lw=1.0, ls="--", label=r"$\mu_{\rm peak}=1/10$")
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel(r"peak ordered mass ratio $\mu_{\rm peak}$")
    ax_ratio.set_ylabel("density")
    ax_ratio.set_title("Merger mass ratios")
    ax_ratio.legend(frameon=False, fontsize=8)

    png_path = output_dir / "tng_merger_event_cache_logM10_pilot.png"
    pdf_path = output_dir / "tng_merger_event_cache_logM10_pilot.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = _parse_args()
    input_path = _resolve_path(args.input)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_cache(input_path)
    csv_path = _write_summary_csv(output_dir, data)
    png_path, pdf_path = _plot(data, output_dir)
    print(f"saved_summary={csv_path}", flush=True)
    print(f"saved_plot_png={png_path}", flush=True)
    print(f"saved_plot_pdf={pdf_path}", flush=True)


if __name__ == "__main__":
    main()
