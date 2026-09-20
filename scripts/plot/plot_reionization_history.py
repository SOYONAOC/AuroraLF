"""Compare completed instantaneous Pop II and Pop II + III histories."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_instantaneous_21cm import CASES, crossing, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/21cm_map/instantaneous"))
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("A completed map run is required")
    path = args.run / "histories.json"
    histories = json.loads(path.read_text())
    style()
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    result = {}
    for key, label, color in CASES[:2]:
        history = histories[key]
        z, q = np.asarray(history["redshifts"]), np.asarray(history["mean_xhii"])
        np.testing.assert_array_equal(z, manifest["redshifts"])
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError(f"Invalid ionization history: {key}")
        result[key] = {str(t): crossing(z, q, t) for t in [0.1, 0.5, 0.9]}
        z50 = result[key]["0.5"]["z"]
        ax.plot(z, q, color=color, lw=2.3, label=f"{label} ($z_{{50}}={z50:.2f}$)")
        ax.plot([z50, z50], [0, 0.5], color=color, lw=1, ls=":")
        ax.scatter([z50], [0.5], color=color, s=32, zorder=4)
    ax.axhline(0.5, color=".7", lw=1, ls=":", zorder=0)
    ax.set(
        xlim=(20, 6),
        ylim=(0, 1.02),
        xlabel="Redshift (time increases to the right)",
        ylabel=r"Volume-mean ionized fraction $\langle x_{\mathrm{HII}}\rangle_V$",
        title="Reionization histories: instantaneous photon sources",
    )
    ax.legend(loc="upper left", fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / "popii_vs_popii_popiii_history"
    fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    result["history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
