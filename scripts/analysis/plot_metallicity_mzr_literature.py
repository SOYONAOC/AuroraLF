#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroralf.chemistry import (  # noqa: E402
    SOLAR_OXYGEN_ABUNDANCE,
    equivalent_oxygen_abundance_from_zsun,
    fire2_highz_mzr_oh12,
    jades_lowmass_mzr_oh12,
)
from auroralf.uvlf import (  # noqa: E402
    IMF_MODE_CANONICAL,
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    IMF_MODE_Z_GATED_MILD_TOPHEAVY,
)
from auroralf.uvlf.pipeline import run_halo_uv_pipeline  # noqa: E402


MODE_LABELS = {
    IMF_MODE_CANONICAL: "canonical",
    IMF_MODE_Z_GATED_MILD_TOPHEAVY: r"$z_{\rm src}\geq10$ top-heavy",
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY: "MAH-burst top-heavy",
}
MODE_COLORS = {
    IMF_MODE_CANONICAL: "black",
    IMF_MODE_Z_GATED_MILD_TOPHEAVY: "#c44e52",
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY: "#1f77b4",
}
MODE_MARKERS = {
    IMF_MODE_CANONICAL: "o",
    IMF_MODE_Z_GATED_MILD_TOPHEAVY: "s",
    IMF_MODE_MAH_BURST_MILD_TOPHEAVY: "^",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AuroraLF metallicity histories with high-redshift mass-metallicity literature relations."
    )
    parser.add_argument("--history-npz", required=True, help="Path to metallicity history NPZ output.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix for PNG/PDF/TXT under outputs/ unless an absolute path is supplied.",
    )
    parser.add_argument(
        "--table-path",
        default=None,
        help="CSV output path. Defaults to data_save/<output-stem>.csv.",
    )
    return parser.parse_args()


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_output_prefix(output_prefix: str | None, history_path: Path) -> Path:
    if output_prefix is None:
        return PROJECT_ROOT / "outputs" / f"{history_path.stem}_literature_mzr"
    path = Path(output_prefix).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().with_suffix("") if path.suffix else path.resolve()


def _mass_tag(log_mass: float) -> str:
    return f"logM{str(float(log_mass)).replace('.', 'p')}"


def _mode_names(payload: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    modes = tuple(str(item) for item in np.asarray(payload["mode_names"]))
    expected = (
        IMF_MODE_CANONICAL,
        IMF_MODE_Z_GATED_MILD_TOPHEAVY,
        IMF_MODE_MAH_BURST_MILD_TOPHEAVY,
    )
    missing = [mode for mode in expected if mode not in modes]
    if missing:
        raise ValueError(f"history file is missing required modes: {missing}")
    return expected


def _final_finite_value(values: np.ndarray, key: str) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if finite.size == 0:
        raise ValueError(f"{key} contains no finite values")
    return float(array[finite[-1]])


def _estimate_stellar_mass_medians(payload: np.lib.npyio.NpzFile, log_masses: np.ndarray) -> dict[float, dict[str, float]]:
    z_final = float(np.asarray(payload["z_final"], dtype=float)[0])
    z_start_max = float(np.asarray(payload["z_start_max"], dtype=float)[0])
    n_tracks = int(np.asarray(payload["n_tracks"], dtype=int)[0])
    n_grid = int(np.asarray(payload["n_grid"], dtype=int)[0])
    random_seed = int(np.asarray(payload["random_seed"], dtype=int)[0])
    returned_fraction = float(np.asarray(payload["metal_returned_fraction"], dtype=float)[0])

    if not 0.0 <= returned_fraction < 1.0:
        raise ValueError("metal_returned_fraction must lie in [0, 1)")

    summaries: dict[float, dict[str, float]] = {}
    for mass_index, log_mass in enumerate(log_masses):
        seed = int(random_seed + 1000 * mass_index)
        result = run_halo_uv_pipeline(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=float(10.0**float(log_mass)),
            z_start_max=z_start_max,
            n_grid=n_grid,
            random_seed=seed,
            workers=1,
            imf_mode=IMF_MODE_CANONICAL,
        )
        n_halos = int(result.metadata["n_tracks"])
        n_steps = int(result.metadata["steps_per_halo"])
        t_grid = np.asarray(result.sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, n_steps)
        sfr_grid = np.asarray(result.sfr_tracks["SFR"], dtype=float).reshape(n_halos, n_steps)
        active_grid = np.asarray(result.sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, n_steps)

        surviving_mass = np.zeros(n_halos, dtype=float)
        for halo_index in range(n_halos):
            mask = active_grid[halo_index] & np.isfinite(sfr_grid[halo_index]) & (sfr_grid[halo_index] > 0.0)
            if np.count_nonzero(mask) < 2:
                continue
            formed_mass = float(np.trapezoid(sfr_grid[halo_index, mask], x=t_grid[halo_index, mask] * 1.0e9))
            surviving_mass[halo_index] = max(formed_mass, 0.0) * (1.0 - returned_fraction)

        positive = surviving_mass[np.isfinite(surviving_mass) & (surviving_mass > 0.0)]
        if positive.size == 0:
            raise RuntimeError(f"no positive surviving stellar masses for logMh={float(log_mass):g}")
        summaries[float(log_mass)] = {
            "logmstar_median": float(np.log10(np.median(positive))),
            "logmstar_p16": float(np.log10(np.percentile(positive, 16.0))),
            "logmstar_p84": float(np.log10(np.percentile(positive, 84.0))),
        }
    return summaries


def _plot_comparison(
    *,
    output_prefix: Path,
    rows: list[dict[str, float | str]],
    log_masses: np.ndarray,
    mode_names: tuple[str, ...],
) -> tuple[Path, Path]:
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 7.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    x_min = min(float(row["logmstar_median"]) for row in rows) - 0.35
    x_max = max(float(row["logmstar_median"]) for row in rows) + 0.35
    x_relation = np.linspace(x_min, x_max, 256)
    ax.plot(x_relation, fire2_highz_mzr_oh12(x_relation), color="#029e73", lw=2.2, label="FIRE-2 z=5-12 MZR")
    ax.plot(
        x_relation,
        jades_lowmass_mzr_oh12(x_relation),
        color="#de8f05",
        lw=2.0,
        ls="--",
        label="JADES z=3-10 low-mass MZR",
    )

    ghz2_logmstar = 8.8
    ghz2_oh12 = float(equivalent_oxygen_abundance_from_zsun(0.05))
    ghz2_lower = ghz2_oh12 - float(equivalent_oxygen_abundance_from_zsun(0.02))
    ghz2_upper = float(equivalent_oxygen_abundance_from_zsun(0.17)) - ghz2_oh12
    ax.errorbar(
        [ghz2_logmstar],
        [ghz2_oh12],
        xerr=[[0.2], [0.2]],
        yerr=[[ghz2_lower], [ghz2_upper]],
        fmt="*",
        ms=13,
        color="#cc78bc",
        mec="black",
        mew=0.5,
        capsize=3,
        label="GHZ2/GLASS-z12",
        zorder=4,
    )

    for mode in mode_names:
        selected = [row for row in rows if row["mode"] == mode]
        x = np.asarray([float(row["logmstar_median"]) for row in selected], dtype=float)
        y = np.asarray([float(row["oh12"]) for row in selected], dtype=float)
        xerr_low = x - np.asarray([float(row["logmstar_p16"]) for row in selected], dtype=float)
        xerr_high = np.asarray([float(row["logmstar_p84"]) for row in selected], dtype=float) - x
        ax.errorbar(
            x,
            y,
            xerr=[xerr_low, xerr_high],
            yerr=None,
            fmt=MODE_MARKERS[mode],
            ms=6.3,
            lw=1.0,
            capsize=2.5,
            color=MODE_COLORS[mode],
            mec="white",
            mew=0.5,
            label=MODE_LABELS[mode],
            zorder=5,
        )

    canonical_rows = [row for row in rows if row["mode"] == IMF_MODE_CANONICAL]
    for log_mass, row in zip(log_masses, canonical_rows, strict=True):
        ax.text(
            float(row["logmstar_median"]) + 0.035,
            float(row["oh12"]) - 0.085,
            rf"$\log M_h={float(log_mass):g}$",
            fontsize=7.2,
            color="0.25",
        )

    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$12+\log({\rm O/H})$")
    ax.set_xlim(x_min, x_max)
    y_values = np.asarray([float(row["oh12"]) for row in rows], dtype=float)
    y_min = min(float(np.nanmin(y_values)), ghz2_oh12 - ghz2_lower, float(np.nanmin(fire2_highz_mzr_oh12(x_relation)))) - 0.18
    y_max = max(
        float(np.nanmax(y_values)),
        ghz2_oh12 + ghz2_upper,
        float(np.nanmax(jades_lowmass_mzr_oh12(x_relation))),
    ) + 0.18
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.text(
        0.98,
        0.03,
        r"Model gas $Z/Z_\odot$ converted with solar $12+\log({\rm O/H})=8.69$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.35",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = _parse_args()
    history_path = _resolve_path(args.history_npz)
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    output_prefix = _resolve_output_prefix(args.output_prefix, history_path)
    table_path = _resolve_path(args.table_path) if args.table_path else PROJECT_ROOT / "data_save" / f"{output_prefix.name}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(history_path, allow_pickle=False) as payload:
        log_masses = np.asarray(payload["log_masses"], dtype=float)
        mode_names = _mode_names(payload)
        stellar_mass_summary = _estimate_stellar_mass_medians(payload, log_masses)

        rows: list[dict[str, float | str]] = []
        for log_mass in log_masses:
            mass_tag = _mass_tag(float(log_mass))
            stellar = stellar_mass_summary[float(log_mass)]
            for mode in mode_names:
                gas_key = f"{mass_tag}_{mode}_gas_median"
                zgas = _final_finite_value(np.asarray(payload[gas_key], dtype=float), gas_key)
                oh12 = float(equivalent_oxygen_abundance_from_zsun(zgas)) if zgas > 0.0 else np.nan
                fire = float(fire2_highz_mzr_oh12(np.asarray([stellar["logmstar_median"]], dtype=float))[0])
                jades = float(jades_lowmass_mzr_oh12(np.asarray([stellar["logmstar_median"]], dtype=float))[0])
                rows.append(
                    {
                        "logmh": float(log_mass),
                        "mode": mode,
                        "logmstar_median": stellar["logmstar_median"],
                        "logmstar_p16": stellar["logmstar_p16"],
                        "logmstar_p84": stellar["logmstar_p84"],
                        "zgas_zsun": zgas,
                        "oh12": oh12,
                        "delta_fire2_dex": oh12 - fire,
                        "delta_jades_dex": oh12 - jades,
                    }
                )

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "logmh",
            "mode",
            "logmstar_median",
            "logmstar_p16",
            "logmstar_p84",
            "zgas_zsun",
            "oh12",
            "delta_fire2_dex",
            "delta_jades_dex",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    png_path, pdf_path = _plot_comparison(
        output_prefix=output_prefix,
        rows=rows,
        log_masses=log_masses,
        mode_names=mode_names,
    )
    summary_path = output_prefix.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"history_npz: {history_path}\n")
        handle.write(f"table_csv: {table_path}\n")
        handle.write(f"saved_png: {png_path}\n")
        handle.write(f"saved_pdf: {pdf_path}\n")
        handle.write("solar_oxygen_abundance: 8.69\n")
        handle.write("fire2_relation: log(Z/Zsun)=0.37 log(Mstar/Msun)-4.3\n")
        handle.write("jades_relation: 12+log(O/H)=7.72+0.17 log(Mstar/1e8 Msun)\n")
        handle.write("ghz2_point: logMstar=8.8+/-0.2, Z=0.05(-0.03,+0.12) Zsun\n\n")
        for row in rows:
            handle.write(
                f"logMh={float(row['logmh']):g} {row['mode']}: "
                f"logMstar={float(row['logmstar_median']):.3f}, "
                f"Zgas={float(row['zgas_zsun']):.4g}, "
                f"OH12={float(row['oh12']):.3f}, "
                f"delta_FIRE2={float(row['delta_fire2_dex']):+.3f}, "
                f"delta_JADES={float(row['delta_jades_dex']):+.3f}\n"
            )
    print(f"saved_table={table_path}")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
