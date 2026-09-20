"""Compare real ionization slices at approximately matched volume means.

Targets are display choices, not physical thresholds. Never interpolate fields.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fesc", nargs="+", type=float, required=True)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    common = plan.get("escape_mode") == "common_popii_popiii"
    baseline = Path(plan["baseline_run"])
    base = json.loads((baseline / "manifest.json").read_text())
    if base["status"] != "complete":
        raise ValueError("Baseline is incomplete")
    cases = [(0.0, baseline, "popii")]
    for value in args.fesc:
        if value == 0.2:
            cases.append((value, baseline, "popii_popiii"))
        else:
            case = next(c for c in plan["cases"] if c["fesc_popiii"] == value)
            cases.append((value, Path(case["run"]), "popii_popiii"))
    cases.sort()
    dim = base["grid"]["dim"]
    length = base["grid"]["box_length_mpc"]
    index = dim // 2
    targets = args.targets
    if 0.5 not in targets or any(not 0 < t < 1 for t in targets):
        raise ValueError("Display targets must include 0.5 and lie inside (0,1)")
    if common and len(args.fesc) != 3:
        raise ValueError("Common-escape comparison requires exactly three mixed-model curves")
    records, arrays, hashes = [], {}, {str(args.plan): digest(args.plan)}
    for fesc, run, population in cases:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {run}")
        for key in ["grid", "physics", "power", "density_sha256", "code_sha256"]:
            if manifest[key] != base[key]:
                raise ValueError(f"Unpaired input {key}: {run}")
        model = manifest["source_manifest"]["resolved_model"]
        for key, value in base["source_manifest"]["resolved_model"].items():
            varying = ["fesc_popii", "fesc_popiii"] if common and fesc else ["fesc_popiii"]
            if key not in varying and model[key] != value:
                raise ValueError(f"Changed stellar parameter {key}")
        if fesc and model["fesc_popiii"] != fesc:
            raise ValueError("Wrong escape fraction")
        if common and fesc and model["fesc_popii"] != fesc:
            raise ValueError("Mixed model populations must share escape fraction")
        for name in ["manifest.json", "histories.json"]:
            hashes[str(run / name)] = digest(run / name)
        history = json.loads((run / "histories.json").read_text())[population]
        z = np.asarray(history["redshifts"])
        q = np.asarray(history["mean_xhii"])
        if not np.isfinite(q).all() or np.any(np.diff(z) >= 0):
            raise ValueError("Invalid history")
        for target in targets:
            above = np.flatnonzero(q >= target)
            if not len(above) or above[0] == 0:
                raise ValueError(f"Target {target} not bracketed")
            hi = int(above[0])
            lo = hi - 1
            nearest = min([lo, hi], key=lambda i: abs(q[i] - target))
            row = dict(
                fesc=fesc,
                target=target,
                redshift=float(z[nearest]),
                mean_xhii=float(q[nearest]),
                residual=float(q[nearest] - target),
                bracket_z=z[[lo, hi]].tolist(),
                bracket_q=q[[lo, hi]].tolist(),
                population=population,
                run=str(run),
            )
            for suffix, i in [("nearest", nearest), ("other", hi if nearest == lo else lo)]:
                path = run / "reionf" / population / f"rf_{z[i]:.2f}.npy"
                field = np.load(path, mmap_mode="r", allow_pickle=False)
                if field.shape != (dim,) * 3 or not np.isfinite(field).all():
                    raise ValueError(f"Invalid map: {path}")
                if np.min(field) < 0 or np.max(field) > 1:
                    raise ValueError(f"Unbounded map: {path}")
                np.testing.assert_allclose(field.mean(dtype=np.float64), q[i], atol=1e-7, rtol=0)
                if suffix == "nearest":
                    full_fraction = float(np.count_nonzero(field == 1) / field.size)
                    row["fully_ionized_volume_fraction"] = full_fraction
                    row["mean_xhii_outside_fully_ionized"] = float(
                        (q[i] - full_fraction) / (1 - full_fraction)
                    )
                key = f"f{fesc:g}_q{target:g}_{suffix}"
                arrays[key] = np.array(field[index], copy=True)
                row[suffix + "_key"] = key
                hashes[str(path)] = digest(path)
                del field
            records.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.style.use("apj")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )

    def escape_label(fesc):
        if fesc == 0:
            return r"Pop II only, $f_{\rm esc}=0.2$" if common else "Pop II only"
        return (
            rf"Pop II+III, $f_{{\rm esc}}={fesc:g}$" if common else rf"$f_{{\rm esc,III}}={fesc:g}$"
        )

    def draw(ax, row):
        field = arrays[row["nearest_key"]]
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=(0, length, 0, length),
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        label = escape_label(row["fesc"])
        ax.set_title(label + "\n" + rf"$z={row['redshift']:.2f},\ Q_V={row['mean_xhii']:.3f}$")
        ax.set(
            xlabel="y [comoving Mpc]",
            ylabel="z [comoving Mpc]",
            xticks=[0, 100, 200, 300],
            yticks=[0, 100, 200, 300],
        )
        return im

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=150)
        fig.savefig(args.output / f"{name}.pdf")
        plt.close(fig)

    core = [
        r for r in records if r["target"] == 0.5 and (common or r["fesc"] in [0, 0.01, 0.03, 0.2])
    ]
    if len(core) != 4:
        raise ValueError("Main comparison requires pure II and exactly three mixed-model curves")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.1), layout="constrained")
    for ax, row in zip(axes.flat, core, strict=True):
        im = draw(ax, row)
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"Local ionized fraction $x_{\rm HII}$")
    fig.suptitle(
        "Approximately 50% ionized: nearest real snapshots\n"
        + ("Shared escape in mixed model; " if common else r"Fixed $f_{\rm esc,II}=0.2$; ")
        + rf"same slice at $x={(index + 0.5) * length / dim:g}$ comoving Mpc",
        fontsize=14,
    )
    save(fig, "matched_q50")

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.6), layout="constrained")
    for ax, row in zip(axes, core, strict=True):
        im = draw(ax, row)
        if ax != axes[0]:
            ax.set_ylabel("")
    fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        shrink=0.5,
        pad=0.08,
        label=r"Local ionized fraction $x_{\rm HII}$",
    )
    save(fig, "matched_q50_wide")

    fig, axes = plt.subplots(
        len(targets),
        len(cases),
        figsize=(3.1 * len(cases), 3.3 * len(targets)),
        layout="constrained",
        squeeze=False,
    )
    for j, target in enumerate(targets):
        for i, (fesc, _, _) in enumerate(cases):
            row = next(r for r in records if r["target"] == target and r["fesc"] == fesc)
            im = draw(axes[j, i], row)
            if i:
                axes[j, i].set_ylabel("")
            else:
                axes[j, i].set_ylabel(f"Target Q = {target:.2f}\nz [comoving Mpc]")
    fig.colorbar(im, ax=axes, shrink=0.75, label=r"Local ionized fraction $x_{\rm HII}$")
    fig.suptitle("Matched-stage overview: real snapshots, no field interpolation", fontsize=15)
    save(fig, "matched_stages")

    ref = arrays[core[0]["nearest_key"]].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    diffs = [arrays[row["nearest_key"]].astype(float) - ref for row in core[1:]]
    limit = max(float(np.max(np.abs(d))) for d in diffs)
    if limit <= 0:
        raise ValueError("All comparison fields are identical")
    for ax, row, delta in zip(axes, core[1:], diffs, strict=True):
        im = ax.imshow(
            delta.T,
            origin="lower",
            extent=(0, length, 0, length),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(escape_label(row["fesc"]) + " minus pure II")
        ax.set(xlabel="y [comoving Mpc]", ylabel="z [comoving Mpc]")
    fig.colorbar(im, ax=axes, label=r"$\Delta x_{\rm HII}$", shrink=0.85)
    fig.suptitle("Near Q = 0.5; residual stage offsets remain (see map labels)")
    save(fig, "matched_q50_difference")

    # Both bracketing snapshots show the remaining temporal sampling uncertainty.
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4), layout="constrained")
    for ax, row in zip(axes.flat, core, strict=True):
        im = ax.imshow(
            arrays[row["other_key"]].T,
            origin="lower",
            extent=(0, length, 0, length),
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        other_q = next(v for v in row["bracket_q"] if v != row["mean_xhii"])
        other_z = next(v for v in row["bracket_z"] if v != row["redshift"])
        ax.set_title(escape_label(row["fesc"]) + "\n" + rf"$z={other_z:.2f},\ Q_V={other_q:.3f}$")
        ax.set(xlabel="y [comoving Mpc]", ylabel="z [comoving Mpc]")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"Local ionized fraction $x_{\rm HII}$")
    fig.suptitle("Other bracketing snapshot around Q = 0.5")
    save(fig, "matched_q50_other_bracket")
    for row in records:
        same = next(r for r in records if r["target"] == row["target"] and r["fesc"] == 0)
        field = arrays[row["nearest_key"]].astype(float)
        reference = arrays[same["nearest_key"]].astype(float)
        row["slice_rms_vs_popii"] = float(np.sqrt(np.mean((field - reference) ** 2)))
        row["slice_correlation_vs_popii"] = float(
            np.corrcoef(field.ravel(), reference.ravel())[0, 1]
        )
        row["slice_rms_between_brackets"] = float(
            np.sqrt(np.mean((field - arrays[row["other_key"]]) ** 2))
        )
    np.savez_compressed(args.output / "slices.npz", **arrays)
    summary = dict(
        status="complete",
        escape_mode="common_popii_popiii" if common else "independent_popiii",
        targets=targets,
        slice_axis=0,
        slice_index=index,
        box_length_mpc=length,
        records=records,
        input_sha256=hashes,
        method="First upward crossing, nearest real snapshot. No field interpolation, rescaling, thresholding or smoothing.",
        limitations="Approximate matched volume means, different redshifts; slice metrics are descriptive, not 3D topology or bubble-size measurements. Same legacy density realization with unverified cosmology; excursion regions not globally photon-conserving.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "fesc",
                        "target",
                        "redshift",
                        "mean_xhii",
                        "slice_rms_vs_popii",
                        "slice_correlation_vs_popii",
                        "slice_rms_between_brackets",
                    ]
                }
                for r in records
                if r["target"] == 0.5
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
