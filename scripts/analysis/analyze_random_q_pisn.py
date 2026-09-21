"""PISN ensemble rates from verified, existing random-q burst samples."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.artifacts import (
    digest,
    read_completed_manifest,
    read_toml,
    record_experiment_sources,
    require,
    resolve_path,
)
from auroralf.experiments.heii import cluster_sum_and_se
from auroralf.experiments.pisn import load_marigo_kernel, observer_rate_per_deg2
from auroralf.mah import Cosmology
from auroralf.uvlf import uv_luminosity_to_muv


def analyze_case(case, config, base, kernel, hydrogen_kernel, hashes):
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    eps, z = config["efficiency"], case["z"]
    # Marigo et al. section 2.3 explicitly maps He-core 64--133 to ZAMS
    # 127--252 Msun. Keep this sourced fate uncertainty separate from baseline.
    marigo_fate = replace(kernel, lower=127.0, upper=252.0)
    require(np.isfinite(eps) and 0 < eps <= 1, "invalid burst efficiency")
    statistics, seeds, runs = {}, set(), []
    mass_edges = np.arange(1.0, 10.01, 0.25)
    progenitor_edges = np.arange(140.0, 260.01, 10.0)
    age_edges = np.arange(0.0, 5.001, 0.05)

    def add(name, rows):
        statistics.setdefault(name, []).append(rows)

    for value in case["runs"]:
        path = resolve_path(base, value)
        manifest = read_completed_manifest(path, required_products=("samples.npz", "source.tar.gz"))
        c = manifest["config"]
        require(c["z"] == z and c["seed"] not in seeds, "redshift mismatch / duplicate seed")
        require(
            c["q_log10_mean"] == config.get("q_log10_mean", 0.5)
            and c["q_log10_sigma"] == config.get("q_log10_sigma", 1.5),
            "different q model",
        )
        require(c["logmass_min"] == 5.0 and c["logmass_max"] == 12.0, "different halo mass range")
        require(eps in c["efficiencies"], "efficiency absent from parent run")
        uv = resolve_path(base, config["executed_uv_ssp"])
        require(manifest["input_sha256"][c["popiii_ssp"]] == digest(uv), "different Pop III SSP")
        for name in ("auroralf/constants.py", "auroralf/mah/models.py"):
            require(digest(name) == manifest["code_sha256"][name], f"changed cosmology: {name}")
        elapsed = (astro.age(z).value - astro.age(c["z_start"]).value) * 1000
        # Status 2 crossed before the saved history starts. Its youngest possible
        # age is older than every PISN delay and every requested trailing window.
        require(
            elapsed > kernel.delay_bounds_myr[1] + max(config["windows_myr"]),
            "left-censored events could still explode",
        )
        seeds.add(c["seed"])
        hashes[str(path / "manifest.json")] = digest(path / "manifest.json")
        with np.load(path / "samples.npz") as data:
            status, age, halo, w, p2, p3 = (
                data[k]
                for k in (
                    "status",
                    "age_myr",
                    "burst_halo_mass_msun",
                    "weight_per_track",
                    "popii",
                    "popiii_per_efficiency",
                )
            )
        shape = (c["n_mass"], c["n_tracks"])
        require(all(a.shape == shape for a in (status, age, halo, p2, p3)), "sample shapes")
        require(
            w.shape == (shape[0],) and np.isfinite(w).all() and np.all(w > 0), "invalid weights"
        )
        require(np.isin(status, [0, 1, 2]).all(), "unknown burst state")
        known = status == 1
        require(np.isfinite(age[known]).all() and np.all(age[known] >= 0), "invalid burst age")
        require(np.isfinite(halo[known]).all() and np.all(halo[known] > 0), "invalid burst mass")
        require(
            np.isnan(age[~known]).all() and np.isnan(halo[~known]).all(), "non-event coordinates"
        )
        require(all(np.isfinite(a).all() and np.all(a >= 0) for a in (p2, p3)), "invalid UV")
        mass = eps * cosmo.omega_b / cosmo.omega_m * np.where(known, halo, 0.0)
        safe_age = np.where(known, age, 0.0)
        bright = uv_luminosity_to_muv(p2 + eps * p3) <= config["uv_cut"]
        rate = mass * kernel.rate(safe_age)
        add("source_rate", rate.sum(axis=1) * w)
        add("marigo_fate_127_252_source_rate", (mass * marigo_fate.rate(safe_age)).sum(axis=1) * w)
        add("uv_bright_source_rate", (rate * bright).sum(axis=1) * w)
        add("burst_below_260_msun_source_rate", (rate * (mass < 260)).sum(axis=1) * w)
        add(
            "expected_count_below_one_source_rate",
            (rate * (mass * kernel.imf.yield_per_msun() < 1)).sum(axis=1) * w,
        )
        add("hydrogen_only_source_rate", (mass * hydrogen_kernel.rate(safe_age)).sum(axis=1) * w)
        for window in config["windows_myr"]:
            add(
                f"window_{window:g}_myr_source_rate",
                (mass * kernel.rate(safe_age, window)).sum(axis=1) * w,
            )
        mass_hist, age_hist, progenitor_hist = [], [], []
        for m, a, r in zip(mass, safe_age, rate):
            active = r > 0
            mass_hist.append(
                np.histogram(np.log10(m[active]), bins=mass_edges, weights=r[active])[0]
            )
            age_hist.append(np.histogram(a[active], bins=age_edges, weights=r[active])[0])
            progenitor_hist.append(
                np.histogram(
                    np.clip(kernel.inverse(a[active]), 140, 260),
                    bins=progenitor_edges,
                    weights=r[active],
                )[0]
            )
        require(
            np.isclose(np.sum(mass_hist), rate.sum(), rtol=1e-10, atol=0),
            "burst mass histogram truncates rate",
        )
        add("rate_by_log_burst_mass", np.asarray(mass_hist) * w[:, None])
        add("rate_by_age", np.asarray(age_hist) * w[:, None])
        add("rate_by_progenitor_mass", np.asarray(progenitor_hist) * w[:, None])
        runs.append(
            {
                "path": str(path),
                "seed": c["seed"],
                "products": manifest["products"],
                "n_mass": shape[0],
                "n_tracks": shape[1],
                "history_dt_myr": elapsed / (c["n_grid"] - 1),
                "left_censored_minimum_age_myr": elapsed,
            }
        )
        print(f"validated and analyzed {path.name}", flush=True)
    estimates = {}
    for name, rows in statistics.items():
        mean, se = cluster_sum_and_se(rows)
        estimates[name] = {
            "value": np.asarray(mean).tolist(),
            "cluster_mc_se": np.asarray(se).tolist(),
        }
    mean, se = (estimates["source_rate"][k] for k in ("value", "cluster_mc_se"))
    require(mean > 0, "no PISN rate resolved in these samples")
    estimates["observer_rate_per_year_deg2_dz"] = {
        "value": float(observer_rate_per_deg2(mean, z, astro)),
        "cluster_mc_se": float(observer_rate_per_deg2(se, z, astro)),
    }
    return {
        "z": z,
        "runs": runs,
        "estimates": estimates,
        "log_burst_mass_edges": mass_edges.tolist(),
        "progenitor_mass_edges_msun": progenitor_edges.tolist(),
        "age_edges_myr": age_edges.tolist(),
        "uv_bright_rate_fraction": estimates["uv_bright_source_rate"]["value"] / mean,
        "expected_count_below_one_rate_fraction": estimates["expected_count_below_one_source_rate"][
            "value"
        ]
        / mean,
        "burst_below_260_msun_rate_fraction": estimates["burst_below_260_msun_source_rate"]["value"]
        / mean,
    }


def plot_results(summary, kernel, hydrogen_kernel, output):
    plt.style.use("apj")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), layout="constrained")
    mass = np.geomspace(1, 500, 1500)
    phi = kernel.imf.number_density(mass) * mass * np.log(10)
    axes[0].plot(mass, phi)
    axes[0].fill_between(mass, 0, phi, where=(mass >= 140) & (mass <= 260), alpha=0.25)
    axes[0].set(
        xscale="log",
        xlabel=r"Initial stellar mass [$M_\odot$]",
        ylabel=r"Number / initial $M_\odot$ / dex",
        xlim=(1, 500),
        ylim=(0, None),
    )
    age = np.linspace(2.0, 3.2, 3000)
    axes[1].plot(age, kernel.rate(age) * 1e6, label="H + He burning")
    axes[1].plot(age, hydrogen_kernel.rate(age) * 1e6, "--", label="H burning only")
    axes[1].set(
        xlabel="Burst age [Myr]",
        ylabel=r"PISN / initial $M_\odot$ / Myr",
        xlim=(2, 3.2),
        ylim=(0, None),
    )
    axes[1].legend(fontsize=9)
    fig.savefig(output / "imf_delay.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("source_rate", "observer_rate_per_year_deg2_dz"),
        (r"PISN / source yr / cMpc$^3$", r"PISN / observer yr / deg$^2$ / $\Delta z$"),
    ):
        for case in summary["cases"]:
            e = case["estimates"][key]
            ax.errorbar(
                case["z"], e["value"], yerr=e["cluster_mc_se"], fmt="o", color="C0", capsize=4
            )
        ax.set(xlabel="Redshift", ylabel=label, xticks=[12.5, 14.5], xlim=(12, 15), ylim=(0, None))
    fig.savefig(output / "rates.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    for case in summary["cases"]:
        edges = np.asarray(case["log_burst_mass_edges"])
        values = np.asarray(case["estimates"]["rate_by_log_burst_mass"]["value"])
        ax.stairs(values / (values.sum() * np.diff(edges)), edges, label=f"z = {case['z']}")
    ax.set(
        xlabel=r"$\log_{10}(M_{\star,\mathrm{III}}/M_\odot)$",
        ylabel="Fraction of PISN rate / dex",
        xlim=(2, 8.5),
        ylim=(0, None),
    )
    ax.legend()
    fig.savefig(output / "burst_mass.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/pisn_random_q.toml")
    args = parser.parse_args()
    path, config = read_toml(args.config)
    base = path.parent
    lifetime_path = resolve_path(base, config["lifetimes"])
    kernel = load_marigo_kernel(lifetime_path)
    hydrogen_kernel = load_marigo_kernel(lifetime_path, hydrogen_only=True)
    hashes = {
        str(p): digest(p)
        for p in (
            path,
            lifetime_path,
            Path(__file__),
            resolve_path(base, config["executed_uv_ssp"]),
        )
    }
    record_experiment_sources(hashes)
    summary = {
        "config": config,
        "cosmology": asdict(Cosmology()),
        "imf": asdict(kernel.imf),
        "pisn_mass_bounds_msun": [kernel.lower, kernel.upper],
        "yield_per_initial_msun": kernel.imf.yield_per_msun(),
        "delay_bounds_myr": kernel.delay_bounds_myr.tolist(),
        "hydrogen_only_delay_bounds_myr": hydrogen_kernel.delay_bounds_myr.tolist(),
        "source_hashes": hashes,
        "cases": [],
    }
    for case in config["cases"]:
        summary["cases"].append(analyze_case(case, config, base, kernel, hydrogen_kernel, hashes))
    output = resolve_path(base, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    figures = resolve_path(base, config["figures"])
    plot_results(summary, kernel, hydrogen_kernel, figures)
    summary["figure_sha256"] = {str(p): digest(p) for p in sorted(figures.glob("*.pdf"))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output / "summary.json")


if __name__ == "__main__":
    main()
