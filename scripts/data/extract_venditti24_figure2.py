"""Extract published Figure 2 vector coordinates, not new ETC calculations."""

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "slides/popiii_heii_pisn_complete_20260916/assets/venditti24_fig2.pdf"
PAPER = ROOT / "external_data/literature_sources/heii_observability/HEII-OBS-001_Venditti24.pdf"
DEST = PAPER.parent / "venditti24_fig2_vectors.json"
SVG = ROOT / "outputs/heii_v24_overlay_20260917/v24-figure2.svg"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    SVG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(FIGURE), str(SVG)], check=True)
    elements = list(ET.parse(SVG).iter())
    # Original Figure 2 ticks: x=6.5 and 10; log10(L)=41 and 42.
    # Coordinates precede the common (0.71,0,0,-0.71,5.74,182.026) transform.
    y41, y42 = 112.030458, 200.44375
    x_ticks = {"IFU": (61.199164, 334.202025), "MOS": (388.801496, 661.798856)}
    paths = [e.get("d", "") for e in elements]
    for tick in (f"M 61.199164 {y41}", f"M 61.199164 {y42}"):
        if not any(d.startswith(tick) for d in paths):
            raise ValueError("Figure calibration changed")

    def luminosity(y):
        return 10 ** (41 + (y - y41) / (y42 - y41))

    curves, bands = [], []
    for element in elements:
        style, path = element.get("style", ""), element.get("d", "")
        properties = dict(p.strip().split(":", 1) for p in style.split(";") if ":" in p)
        match = re.fullmatch(r"M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)\s*", path)
        if "stroke-dasharray" in properties and match:
            x0, y0, x1, y1 = map(float, match.groups())
            if x1 - x0 < 200 or y0 == y1:
                continue  # labels and legend segments
            mode = "IFU" if x0 < 300 else "MOS"
            xa, xb = x_ticks[mode]
            width = float(properties["stroke-width"])
            resolution = {1.0: 100, 2.0: 1000, 3.0: 2700}[width]
            color = properties["stroke"]
            if color == "rgb(100%,0%,0%)":
                exposure, snr = 50, 5
            elif color.startswith("rgb(43.920898%"):
                exposure, snr = 10, 3
            elif color.startswith("rgb(72.155762%"):
                exposure, snr = 50, 3
            else:
                raise ValueError(f"Unrecognized sensitivity color: {color}")
            dash = float(properties["stroke-dasharray"].split(",")[0]) / width
            if abs(dash - 3.7) < 0.01:
                velocity = 500
            elif abs(dash - 1.0) < 0.01:
                velocity = 50
            else:
                raise ValueError("Unknown line-width style")
            curves.append(
                dict(
                    mode=mode,
                    resolving_power=resolution,
                    exposure_h=exposure,
                    integrated_snr=snr,
                    linewidth_kms=velocity,
                    z=[6.5 + (x - xa) * 3.5 / (xb - xa) for x in (x0, x1)],
                    luminosity_erg_s=[luminosity(y0), luminosity(y1)],
                    svg_path=path,
                )
            )
        if properties.get("fill-opacity") == "0.25" and path.startswith("M "):
            coords = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]
            if len(coords) != 10:
                continue
            xs, ys = coords[::2], coords[1::2]
            if max(xs) - min(xs) < 200:
                continue
            if properties.get("fill") == "rgb(54.508972%,0%,0%)" and min(xs) < 100:
                lo, hi = sorted(luminosity((182.026 - y) / 0.71) for y in (min(ys), max(ys)))
                bands.append(dict(model="No ML", luminosity_range_erg_s=[lo, hi]))
            if properties.get("fill") == "rgb(0%,54.508972%,54.508972%)" and min(xs) < 100:
                bands.append(
                    dict(
                        model="Strong ML",
                        luminosity_range_erg_s=[luminosity(min(ys)), luminosity(max(ys))],
                    )
                )
    if len(curves) != 24 or len(bands) != 2:
        raise ValueError(f"Expected 12 curves per mode and two bands: {len(curves)}, {len(bands)}")
    for band in bands:
        lo, hi = band["luminosity_range_erg_s"]
        if abs(hi / lo / 30 - 1) > 0.001:
            raise ValueError("Band must cover eta=0.01--0.3")
    result = dict(
        source="Venditti+2024, arXiv:2405.10940v2, Figure 2, sections II.2/III.1",
        url="https://arxiv.org/html/2405.10940v2#S3.F2",
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [PAPER, FIGURE]},
        extraction="PDF vector paths via pdftocairo SVG; calibrated against axis ticks. "
        "Values reproduce published artwork, not unrounded author tables or a fresh ETC run.",
        calibration=dict(y41=y41, y42=y42, x_ticks=x_ticks),
        band_display_z=[6.5, 10.7],
        eta_range=[0.01, 0.3],
        bands=bands,
        curves=curves,
        assumptions="JWST ETC v4.0; point source, no continuum, medium background at GN-z11; "
        "IFU aperture0.09arcsec, sky annulus0.3--0.9arcsec; MOS three-shutter slitlet. "
        "Published unlensed luminosity thresholds; not target-specific delensed detection limits.",
        restriction="Keep endpoint-connecting lines only on their published z interval. "
        "Do not extrapolate sensitivity or bands to z12--14. Bands are efficiency/model ranges, "
        "not confidence intervals, and eta_III is not AuroraLF epsilon_b.",
    )
    DEST.write_text(json.dumps(result, indent=2) + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
