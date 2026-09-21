"""Pandeia MOS response basis for an isolated He II line and flat-fnu continuum.

All instrument settings and reference-data versions are saved. No slide writes.
Noise is propagated from Pandeia's extracted 1D noise, including a fitted local
continuum. Population synthesis and instrumental selection remain separate.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "external_data/instruments/pandeia"
OUT = ROOT / "data_save/heii_instrument_20260918/response"
FREF = 1e-18
CREF = 1e-4  # mJy = 100 nJy


def configure(z, mode, size, width, line, continuum, nexp=1):
    from pandeia.engine.calc_utils import build_default_calc, build_default_source

    calc = build_default_calc("jwst", "nirspec", "mos")
    wave = 0.1640 * (1 + z)
    inst = calc["configuration"]["instrument"]
    if mode == "prism":
        inst.update(disperser="prism", filter="clear")
    else:
        inst.update(
            disperser="g140m" if wave < 1.7 else "g235m",
            filter="f100lp" if wave < 1.7 else "f170lp",
        )
    calc["configuration"]["detector"].update(
        readout_pattern="nrsirs2", ngroup=19, nint=1, nexp=nexp
    )
    src = build_default_source("jwst", "point" if size == 0 else "gaussian2d")
    if size:
        src["shape"].update(
            major=size / np.sqrt(8 * np.log(2)), minor=size / np.sqrt(8 * np.log(2))
        )
    src["spectrum"]["normalization"].update(norm_wave=wave, norm_flux=continuum)
    # Retain even a zero-strength line: it fixes the same internal wavelength
    # sampling for continuum-only and line+continuum calculations.
    src["spectrum"]["lines"] = [
        dict(
            center=wave,
            width=width,
            strength=line,
            profile="gaussian",
            emission_or_absorption="emission",
        )
    ]
    calc["scene"] = [src]
    calc["strategy"]["reference_wavelength"] = wave
    return calc


def calculate(config, path):
    from pandeia.engine.perform_calculation import perform_calculation

    encoded = json.dumps(config, sort_keys=True, indent=2)
    if Path(str(path) + ".npz").exists():
        if Path(str(path) + ".json").read_text() != encoded:
            raise ValueError(f"Changed configuration for existing response: {path}")
        with np.load(Path(str(path) + ".npz")) as saved:
            return {k: saved[k] for k in saved.files}
    Path(str(path) + ".json").write_text(encoded)
    report = perform_calculation(config, webapp=False)
    wave, rate = np.asarray(report["1d"]["extracted_flux"])
    nwave, noise = np.asarray(report["1d"]["extracted_noise"])
    np.testing.assert_array_equal(wave, nwave)
    result = dict(
        wave=wave,
        rate=rate,
        variance=noise**2,
        exposure_seconds=report["scalar"]["total_exposure_time"],
    )
    Path(str(path) + ".report.json").write_text(
        json.dumps(
            dict(
                scalar=report["scalar"],
                warnings=report.get("warnings", {}),
                information=report["information"],
            ),
            indent=2,
            default=lambda x: np.asarray(x).tolist(),
        )
    )
    np.savez_compressed(Path(str(path) + ".npz"), **result)
    return result


def reduce_basis(reports, center):
    zero, line, cont, line2, cont2, combined, check = reports
    wave = zero["wave"]
    for r in reports[1:]:
        np.testing.assert_array_equal(r["wave"], wave)
    lr = line["rate"] - zero["rate"]
    cr = cont["rate"] - zero["rate"]
    good = np.isfinite(zero["variance"]) & (zero["variance"] > 0) & (cr > 0)
    local = good & (np.abs(wave - center) < 0.12 * center)
    if np.count_nonzero(local) < 8:
        raise ValueError("Insufficient good pixels around line")
    positive = np.where(local, np.maximum(lr, 0), 0)
    cdf = np.cumsum(positive) / positive.sum()
    lo, hi = np.searchsorted(cdf, [0.025, 0.975])
    peak = int(np.argmax(positive))
    radius = max(hi - peak, peak - lo, 1)
    ix = np.arange(len(wave))
    core = good & (ix >= lo) & (ix <= hi)
    side = good & (np.abs(ix - peak) >= 3 * radius) & (np.abs(ix - peak) <= 7 * radius)
    if side.sum() < 4 or cr[side].sum() <= 0:
        raise ValueError("Insufficient continuum sidebands")
    # Unbiased local continuum amplitude, with its measurement variance retained.
    weights = core.astype(float) - side * (cr[core].sum() / cr[side].sum())
    active = weights != 0
    w = weights[active]
    signal = float(w @ lr[active])
    terms = [float(w**2 @ r["variance"][active]) for r in reports]
    b = terms[0]
    ff = (terms[3] - 2 * terms[1] + b) / 2
    cc = (terms[4] - 2 * terms[2] + b) / 2
    fc = terms[5] - terms[1] - terms[2] + b
    a, c = terms[1] - b - ff, terms[2] - b - cc
    if min(signal, b, a, c) <= 0:
        raise ValueError("Nonpositive signal/noise basis")
    predicted = b + 0.37 * a + 0.63 * c + 0.37**2 * ff + 0.63**2 * cc + 0.37 * 0.63 * fc
    error = abs(terms[6] / predicted - 1)
    if error > 0.005:
        raise ValueError(f"Pandeia variance basis disagrees by more than 0.5%: {error}")
    np.testing.assert_allclose(
        check["rate"][active] - zero["rate"][active],
        0.37 * lr[active] + 0.63 * cr[active],
        rtol=2e-4,
        atol=1e-12,
    )
    return dict(
        signal_per_fref=signal,
        variance_background=b,
        variance_line_per_fref=a,
        variance_continuum_per_cref=c,
        variance_ff=ff,
        variance_cc=cc,
        variance_fc=fc,
        variance_validation_relative_error=error,
        core_pixels=int(core.sum()),
        sideband_pixels=int(side.sum()),
        extracted_line_rate_per_fref=float(lr[core].sum()),
        core_line_fraction=float(lr[core].sum() / positive.sum()),
        exposure_seconds=float(zero["exposure_seconds"]),
        fref=FREF,
        cref_mjy=CREF,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--z", type=float)
    args = parser.parse_args()
    os.environ["pandeia_refdata"] = str(BASE / "pandeia_data-2026.7-jwst")
    os.environ["PSF_DIR"] = str(BASE / "pandeia_psfs-2026.7rc1-jwst")
    for name, version in [("pandeia_refdata", "VERSION_DATA"), ("PSF_DIR", "VERSION_PSF")]:
        if not (Path(os.environ[name]) / version).is_file():
            raise FileNotFoundError(Path(os.environ[name]) / version)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    zs = [args.z] if args.z is not None else [6.639, 8.1623, 10.6, 12.342, 13.86]
    for z in zs:
        for mode in ["prism", "medium"]:
            for size in [0, 0.1, 0.2]:
                for width in [50, 500]:
                    key = f"z{z:g}_{mode}_size{size:g}_width{width}"
                    reports = []
                    for tag, f, c in [
                        ("zero", 0, 0),
                        ("line", FREF, 0),
                        ("cont", 0, CREF),
                        ("line2", 2 * FREF, 0),
                        ("cont2", 0, 2 * CREF),
                        ("combined", FREF, CREF),
                        ("check", 0.37 * FREF, 0.63 * CREF),
                    ]:
                        config = configure(z, mode, size, width, f, c)
                        reports.append(calculate(config, OUT / f"{key}_{tag}"))
                        print(key, tag, "complete", flush=True)
                    result = dict(
                        z=z,
                        mode=mode,
                        size_fwhm_arcsec=size,
                        line_fwhm_kms=width,
                        **reduce_basis(reports, 0.164 * (1 + z)),
                    )
                    for tag in ["zero", "line", "cont"]:
                        report = json.loads((OUT / f"{key}_{tag}.report.json").read_text())
                        result[f"saturation_bound_{tag}"] = report["scalar"]["fraction_saturation"]
                    (OUT / f"{key}_basis.json").write_text(json.dumps(result, indent=2))
                    records.append(result)
                    print(json.dumps(result), flush=True)
                    if args.smoke:
                        return
    (OUT / ("basis.json" if args.z is None else f"basis_z{args.z:g}.json")).write_text(
        json.dumps(records, indent=2)
    )


if __name__ == "__main__":
    main()
