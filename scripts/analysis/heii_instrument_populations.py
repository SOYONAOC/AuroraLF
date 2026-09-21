"""Export the populations currently shown in the He II comparison, without editing slides."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.heii import evaluate_kernel, load_kernel
from auroralf.mah import Cosmology
from auroralf.uvlf import uv_luminosity_to_muv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_save/heii_instrument_20260918"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    base = ROOT / "external_data/ssp_spectra/schaerer2010_pop3"
    uv = base / "pop3_ge0_logE_500_001_is5.25"
    line = base / "pop3_ge0_logE_500_001_is5.22"
    kernel = load_kernel(uv, line)
    records = []
    for z, center in [
        (6.639, None),
        (8.1623, None),
        (10.6, None),
        (12.342, -20.53),
        (13.86, -19.0),
    ]:
        dirname = "heii_v24_targets_20260914" if center is None else "heii_exact_targets_20260916"
        directory = ROOT / "data_save" / dirname
        path = directory / f"z{z:g}.npz"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert sha(path) == manifest["products"][path.name]
        with np.load(path) as sample:
            status = sample["status"]
            age = sample["age_myr"]
            luv = sample["popii"] + 0.03 * sample["popiii_per_efficiency"]
            muv = uv_luminosity_to_muv(luv)
            if center is None:
                selected = (status == 1) & (age <= 3) & (muv <= -20)
            else:
                selected = np.abs(muv - center) <= 0.25
            weights = np.broadcast_to(sample["weight_per_track"][:, None], status.shape)
            unknown = float(weights[selected & (status == 2)].sum() / weights[selected].sum())
            selected &= status != 2
            mass = sample["burst_halo_mass_msun"][selected] * 0.03 * cosmo.omega_b / cosmo.omega_m
            resolved = status[selected] == 1
            lum = np.zeros(len(mass))
            lum[resolved] = mass[resolved] * evaluate_kernel(age[selected][resolved], kernel)
            conversion = 1 / (4 * np.pi * astro.luminosity_distance(z).to_value("cm") ** 2)
            # Explicit exploratory continuum: flat fnu (UV beta=-2), same spatial profile
            # as He II; all populations unlensed to isolate the instrumental effect.
            fnu_mjy = luv[selected] * (1 + z) * conversion / 1e-26
            group = np.broadcast_to(np.arange(status.shape[0])[:, None], status.shape)[selected]
            np.savez_compressed(
                OUT / f"population_z{z:g}.npz",
                luminosity=lum,
                flux=lum * conversion,
                fnu_mjy=fnu_mjy,
                weight=weights[selected],
                cluster=group,
                muv=muv[selected],
            )
            records.append(
                dict(
                    z=z,
                    input=str(path.relative_to(ROOT)),
                    input_sha256=sha(path),
                    conversion=conversion,
                    unknown_weight_fraction=unknown,
                    n_known=len(lum),
                    q_log10_mean=0.5,
                    epsilon=0.03,
                    selection="young UV bright" if center is None else "UV window +/-0.25 mag",
                )
            )
    (OUT / "populations.json").write_text(
        json.dumps(
            dict(
                records=records,
                magnification=1,
                continuum="flat fnu, beta=-2; model UV amplitude per object; no dust",
                source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (uv, line)},
            ),
            indent=2,
        )
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
