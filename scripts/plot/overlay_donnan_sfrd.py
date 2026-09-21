"""Append model markers to the unchanged vector artwork of Donnan24 Fig. 8."""

import hashlib
import json
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "external_data/literature_sources/papers/Donnan2024UVLF/source.tar.gz"
DATA = ROOT / "outputs/sfrd_fraction_audit_20260917/comparison.json"
OUT = ROOT / "outputs/donnan_sfrd_overlay_20260917"
ASSET = ROOT / "slides/atomic_crossing_z6/assets/donnan24_sfrd_overlay.pdf"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def tag(name):
    return f"{{{NS}}}{name}"


def star(parent, x, y, radius=6.0):
    angles = np.arange(10) * np.pi / 5 - np.pi / 2
    radii = np.where(np.arange(10) % 2 == 0, radius, radius * 0.43)
    points = " ".join(
        f"{x + r * np.cos(a):.6f},{y + r * np.sin(a):.6f}"
        for r, a in zip(radii, angles, strict=True)
    )
    ET.SubElement(
        parent,
        tag("polygon"),
        {
            "points": points,
            "fill": "#ffb000",
            "stroke": "#732a00",
            "stroke-width": "0.9",
            "stroke-linejoin": "round",
        },
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        member = archive.getmember("figures/rho_uv.pdf")
        assert member.isfile()
        pdf_bytes = archive.extractfile(member).read()
    source_pdf = OUT / "rho_uv.pdf"
    source_pdf.write_bytes(pdf_bytes)
    source_svg = OUT / "original.svg"
    subprocess.run(["pdftocairo", "-svg", str(source_pdf), str(source_svg)], check=True)
    root = ET.parse(source_svg).getroot()
    original_children = [ET.tostring(e) for e in root]
    original_attributes = dict(root.attrib)
    # Identify the bottom x ticks and left y ticks from their vector strokes.
    segments = []
    for e in root.iter(tag("path")):
        match = re.fullmatch(r"M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)\s*", e.get("d", ""))
        if match and e.get("transform"):
            segments.append((e, np.array([float(x) for x in match.groups()])))
    ticks = [
        (e, v)
        for e, v in segments
        if "stroke-width:1;" in e.get("style", "")
        and abs(v[1] - 38.017851) < 1e-6
        and abs(v[0] - v[2]) < 1e-6
        and 3 < v[3] - v[1] < 4
    ]
    ticks.sort(key=lambda ev: ev[1][0])
    assert len(ticks) == 18
    matrix = ticks[0][0].get("transform")
    assert all(e.get("transform") == matrix for e, _ in ticks)
    a, b, c, d, dx, dy = map(float, re.search(r"matrix\(([^)]+)\)", matrix)[1].split(","))
    assert b == c == 0 and a > 0 and d < 0
    group = ET.SubElement(root, tag("g"), {"id": "auroralf-overlay"})
    rows = json.loads(DATA.read_text())["uv_inferred"]
    positions = []
    for panel, offset in enumerate((0, 9)):
        xp = np.array([v[0] for _, v in ticks[offset : offset + 9]])
        sx = (xp[-1] - xp[0]) / 8
        np.testing.assert_allclose(xp, xp[0] + np.arange(9) * sx, atol=0.005)
        for row in rows:
            rho = row["models"]["eps0.03_dust"]["rho_uv"]
            np.testing.assert_allclose(
                rho * 1.15e-28, row["models"]["eps0.03_dust"]["uv_inferred_sfrd"]
            )
            x = a * (xp[0] + (row["z"] - 8) * sx) + dx
            # The original left-axis endpoints are log10 rho_UV = 22 and 28.
            y_raw = 38.017851 + (np.log10(rho) - 22) / 6 * (304.126272 - 38.017851)
            y = d * y_raw + dy
            star(group, x, y)
            positions.append(
                {
                    "panel": panel,
                    "z": row["z"],
                    "rho_uv": rho,
                    "log10_rho_uv": float(np.log10(rho)),
                    "x": x,
                    "y": y,
                }
            )
        # Validate all seven original left-axis ticks independently.
        left = 78.334242 if panel == 0 else 546.719478
        yt = sorted(
            v[1]
            for e, v in segments
            if abs(v[0] - left) < 1e-6 and 3 < v[2] - v[0] < 4 and abs(v[1] - v[3]) < 1e-6
        )
        assert len(yt) == 7
        np.testing.assert_allclose(yt, np.linspace(38.017851, 304.126272, 7), atol=0.005)
    star(group, 83, 20)
    # Outlined lettering avoids librsvg's PDF kerning bug for some Arial pairs.
    label = "AuroraLF: Pop II + III, with dust; same UV-to-SFR conversion"
    outline = TextPath(
        (0, 0),
        label,
        size=12,
        prop=FontProperties(
            fname="/home/zhuhourui/.local/share/fonts/microsoft-academic/Arial.TTF"
        ),
    )
    codes = {
        MplPath.MOVETO: "M",
        MplPath.LINETO: "L",
        MplPath.CURVE3: "Q",
        MplPath.CURVE4: "C",
        MplPath.CLOSEPOLY: "Z",
    }
    commands = []
    for vertices, code in outline.iter_segments(curves=True, simplify=False):
        coords = "" if code == MplPath.CLOSEPOLY else " ".join(f"{v:.6f}" for v in vertices)
        commands.append(codes[code] + coords)
    path = ET.SubElement(
        group,
        tag("path"),
        {
            "d": " ".join(commands),
            "transform": "translate(97,24) scale(1,-1)",
            "fill": "#732a00",
        },
    )
    ET.SubElement(path, tag("title")).text = label
    assert dict(root.attrib) == original_attributes
    assert [ET.tostring(e) for e in list(root)[:-1]] == original_children
    result = OUT / "overlay.svg"
    ET.ElementTree(root).write(result, encoding="utf-8", xml_declaration=True)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(ASSET), str(result)], check=True)
    # Render source and composite through the same renderer and check unchanged pixels.
    for source, name in [(source_svg, "original-render.png"), (result, "overlay.png")]:
        subprocess.run(
            ["rsvg-convert", "-w", "1960", "-o", str(OUT / name), str(source)], check=True
        )
    before = np.asarray(Image.open(OUT / "original-render.png").convert("RGBA"))
    after = np.asarray(Image.open(OUT / "overlay.png").convert("RGBA"))
    assert before.shape == after.shape
    allowed = np.zeros(before.shape[:2], dtype=bool)
    scale = before.shape[1] / 979.2
    allowed[: int(35 * scale), :] = True  # New legend in the original top margin.
    for point in positions:
        x, y = point["x"] * scale, point["y"] * scale
        r = 8 * scale
        allowed[max(0, int(y - r)) : int(y + r) + 1, max(0, int(x - r)) : int(x + r) + 1] = True
    changed = np.any(before != after, axis=2)
    assert not np.any(changed & ~allowed), "Original artwork changed outside overlay regions"
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "source": "Donnan et al. 2024, arXiv:2403.03171v3, Fig.8",
                "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                "figure_pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "marker_positions": positions,
                "original_svg_children_unchanged": True,
                "pixels_unchanged_outside_added_markers_and_legend": True,
                "x_tick_calibration_max_tolerance_pt": 0.005,
                "y_tick_calibration_max_tolerance_pt": 0.005,
                "interpolation": "None; only the three available matched-estimator predictions",
                "model_error_bars": "Not propagated; markers represent central model values",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
