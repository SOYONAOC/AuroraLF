"""Overlay current SFRD on observational vectors from Venditti Fig. 1.

Preserves marker paths and error bars instead of claiming access to raw survey
catalogues. Source-PDF hash and SVG structure are checked before extraction.
"""

import copy
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data_save/ionizing_sources/sfrd_v1"
OUT = ROOT / "outputs/popiii_sfrd_observations_20260917"
ASSETS = ROOT / "slides/atomic_crossing_z6/assets"
PAPER = (
    ROOT
    / "external_data/literature_sources/popiii_uvlf_library/papers/Venditti2023ANeedleInA/source/figures/SFRD_pop_av+obs+U12.pdf"
)
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
COLORS = {"total": "#244bf0", "popii": "#20832b", "popiii": "#d62728"}
STYLES = {"total": "-", "popii": "--", "popiii": ":"}
FONT_DIR = Path.home() / ".local/share/fonts/microsoft-academic"
FONTS = {"normal": FONT_DIR / "Arial.TTF", "bold": FONT_DIR / "Arialbd.TTF"}
EXTRA_WIDTH = 300.0
WIDTH = 600 + EXTRA_WIDTH
SOURCE_LEFT = 54.0
SOURCE_WIDTH = 381.601562 - SOURCE_LEFT


def wide_x(x):
    return x + (x - SOURCE_LEFT) * EXTRA_WIDTH / SOURCE_WIDTH


def translated(element, dx):
    group = ET.Element(tag("g"), {"transform": f"translate({dx},0)"})
    group.append(copy.deepcopy(element))
    return group


def widen_segments(element):
    """Reproject straight data/axis segments without changing stroke widths."""
    element = copy.deepcopy(element)
    for node in element.iter():
        if node.get("clip-path"):
            node.set("clip-path", "url(#wide-axes)")
        if node.tag != tag("path"):
            continue
        parts = node.get("d").split()
        assert len(parts) % 3 == 0
        for i in range(0, len(parts), 3):
            assert parts[i] in {"M", "L"}
            x = float(parts[i + 1])
            parts[i + 1] = f"{wide_x(x):.9f}"
            np.testing.assert_allclose(
                (float(parts[i + 1]) - SOURCE_LEFT) / (SOURCE_WIDTH + EXTRA_WIDTH),
                (x - SOURCE_LEFT) / SOURCE_WIDTH,
                atol=1e-10,
            )
        node.set("d", " ".join(parts))
    return element


def tag(name):
    return f"{{{NS}}}{name}"


def text(parent, x, y, words, size=11, weight="normal", color="#222222"):
    # Embed vector glyph outlines: librsvg's PDF text kerning is incorrect for
    # some Arial pairs, despite its PNG rendering being correct.
    outline = TextPath((0, 0), words, size=size, prop=FontProperties(fname=FONTS[weight]))
    commands = {
        MplPath.MOVETO: "M",
        MplPath.LINETO: "L",
        MplPath.CURVE3: "Q",
        MplPath.CURVE4: "C",
        MplPath.CLOSEPOLY: "Z",
    }
    segments = []
    for vertices, code in outline.iter_segments(curves=True, simplify=False):
        coordinates = "" if code == MplPath.CLOSEPOLY else " ".join(f"{v:.6f}" for v in vertices)
        segments.append(commands[code] + coordinates)
    el = ET.SubElement(
        parent,
        tag("path"),
        {
            "transform": f"translate({x},{y}) scale(1,-1)",
            "d": " ".join(segments),
            "fill": color,
        },
    )
    ET.SubElement(el, tag("title")).text = words


def main():
    for font in FONTS.values():
        if not font.is_file():
            raise FileNotFoundError(font)
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RUN / "literature_comparison.json").read_text())
    assert hashlib.sha256(PAPER.read_bytes()).hexdigest() == meta["paper_figure_sha256"]
    assert hashlib.sha256((RUN / "sfrd.csv").read_bytes()).hexdigest() == meta["current_csv_sha256"]
    source_svg = OUT / "source.svg"
    subprocess.run(["pdftocairo", "-svg", str(PAPER), str(source_svg)], check=True)
    source = ET.parse(source_svg).getroot()
    defs, body = list(source)
    children = list(body)
    assert body.get("id") == "surface1" and len(children) == 256
    assert children[68].get("clip-path") == "url(#clip3)"
    assert children[154].get("style", "").find("stroke:rgb(100%,0%,100%)") >= 0
    assert children[155].get("d", "").startswith("M 54 39.601562 L 54 352.800781")
    assert children[205].get("d", "").startswith("M 381.601562 320.398438")

    data = np.genfromtxt(RUN / "sfrd.csv", delimiter=",", names=True)
    data = data[data["window_myr"] == 10]
    assert np.all(np.diff(data["z"]) > 0)
    np.testing.assert_allclose(data["total"], data["popii"] + data["popiii"])
    for key in COLORS:
        assert np.all(np.isfinite(data[key]) & (data[key] > 0))

    # Calibration from original axis ticks, in PDF points (upward y).
    sx = (329.183594 - 67.105469) / 10
    sy = (324.328125 - 39.601562) / 5
    left = 67.105469 + (6.5 - 7) * sx
    bottom = 39.601562
    plt.style.use("apj")
    plt.rcParams["text.usetex"] = False
    # Preserve absolute PDF coordinates; the style enables tight cropping.
    plt.rcParams["savefig.bbox"] = None
    fig = plt.figure(figsize=(WIDTH / 72, 360 / 72))
    ax = fig.add_axes(
        [
            wide_x(left) / WIDTH,
            bottom / 360,
            12.5 * sx * (1 + EXTRA_WIDTH / SOURCE_WIDTH) / WIDTH,
            5.5 * sy / 360,
        ]
    )
    ax.set(xlim=(6.5, 19), ylim=(-7, -1.5))
    ax.set_axis_off()
    for key in COLORS:
        (line,) = ax.plot(
            data["z"], np.log10(data[key]), color=COLORS[key], ls=STYLES[key], lw=2.1, clip_on=True
        )
        line.set_gid(f"auroralf-{key}")
    # Independent coordinate test against four original ticks.
    for z, y, xp, yp in [(7, -7, 67.105469, 39.601562), (17, -2, 329.183594, 324.328125)]:
        pix = ax.transData.transform((z, y))
        np.testing.assert_allclose(pix * 72 / fig.dpi, [wide_x(xp), yp], atol=1e-6)
    model_svg = OUT / "model.svg"
    fig.savefig(model_svg, transparent=True)
    plt.close(fig)
    model = ET.parse(model_svg).getroot()
    np.testing.assert_allclose(
        [float(value) for value in model.get("viewBox").split()], [0, 0, WIDTH, 360]
    )

    result = ET.Element(
        tag("svg"),
        {"width": f"{WIDTH}pt", "height": "385pt", "viewBox": f"0 0 {WIDTH} 385", "version": "1.1"},
    )
    result.append(copy.deepcopy(defs))
    wide_clip = ET.SubElement(result[0], tag("clipPath"), {"id": "wide-axes"})
    ET.SubElement(
        wide_clip,
        tag("rect"),
        {
            "x": str(SOURCE_LEFT),
            "y": str(360 - 352.800781),
            "width": str(SOURCE_WIDTH + EXTRA_WIDTH),
            "height": str(352.800781 - bottom),
        },
    )
    for d in model.findall(tag("defs")):
        result.append(copy.deepcopy(d))
    ET.SubElement(result, tag("rect"), {"width": str(WIDTH), "height": "385", "fill": "white"})
    # Retain original axes and tick labels, without theory bands or annotations.
    for i in list(range(4, 68)) + list(range(155, 159)):
        if 4 <= i <= 21:
            if children[i].tag == tag("g"):
                tick_x = float(children[i - 2].get("d").split()[1])
                result.append(translated(children[i], wide_x(tick_x) - tick_x))
            else:
                result.append(widen_segments(children[i]))
        elif i == 22:
            result.append(translated(children[i], EXTRA_WIDTH / 2))
        elif i in range(24, 45, 4):
            result.append(translated(children[i], EXTRA_WIDTH))
        elif i >= 155:
            result.append(widen_segments(children[i]))
        else:
            result.append(copy.deepcopy(children[i]))
    for key in COLORS:
        line = next(e for e in model.iter() if e.get("id") == f"auroralf-{key}")
        result.append(copy.deepcopy(line))
    # Reproject positions/error extents; keep every marker outline undistorted.
    observations = ET.SubElement(result, tag("g"), {"id": "original-observations"})
    marker_count = error_count = 0
    for i in range(71, 155):
        e = children[i]
        path = e if e.tag == tag("path") else list(e)[0]
        matrix = path.get("transform", "")
        if matrix == "matrix(1,0,0,-1,0,360)":
            observations.append(widen_segments(e))
            error_count += 1
        else:
            if not matrix:
                # Four filled pentagons have separate, co-centred stroke paths.
                assert i in {71, 73, 75, 77}
                matrix = list(children[i + 1])[0].get("transform")
            values = [float(v) for v in matrix.removeprefix("matrix(").removesuffix(")").split(",")]
            assert values[:4] == [1, 0, 0, -1]
            center = values[4]
            moved = translated(e, wide_x(center) - center)
            assert ET.tostring(moved[0]) == ET.tostring(e)
            observations.append(moved)
            marker_count += 1
    assert marker_count + error_count == 84

    sidebar = ET.SubElement(result, tag("g"), {"transform": f"translate({EXTRA_WIDTH},0)"})
    text(sidebar, 400, 28, "AuroraLF", 14, "bold")
    text(sidebar, 400, 45, "10 Myr average; burst efficiency 3%", 10)
    for key, name, yy in [
        ("total", "Pop II + Pop III (total)", 67),
        ("popii", "Pop II", 88),
        ("popiii", "Pop III", 109),
    ]:
        attrs = {
            "x1": "401",
            "x2": "428",
            "y1": str(yy),
            "y2": str(yy),
            "stroke": COLORS[key],
            "stroke-width": "2.1",
        }
        if key == "popii":
            attrs["stroke-dasharray"] = "7.77,3.36"
        if key == "popiii":
            attrs["stroke-dasharray"] = "2.1,3.465"
        ET.SubElement(sidebar, tag("line"), attrs)
        text(sidebar, 436, yy + 4, name, 11)
    text(sidebar, 400, 146, "Total-SFRD observations", 12, "bold")
    legend = ET.SubElement(sidebar, tag("g"), {"transform": "translate(340,-58)"})
    for e in children[187:205]:
        legend.append(copy.deepcopy(e))
    text(sidebar, 400, 287, "Points and errors preserved from", 10)
    text(sidebar, 400, 302, "Venditti et al. (2023), Fig. 1.", 10)
    text(sidebar, 400, 328, "Survey selection / IMF / dust", 10)
    text(sidebar, 400, 343, "conventions are not homogenized.", 10)
    text(
        result,
        54,
        377,
        "Observations constrain total SFRD, not Pop III alone.  No fit performed.",
        10,
    )
    svg = OUT / "sfrd_observations.svg"
    ET.ElementTree(result).write(svg, encoding="utf-8", xml_declaration=True)
    pdf = ASSETS / "sfrd_observations.pdf"
    png = OUT / "sfrd_observations.png"
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-w", "1600", "-o", str(png), str(svg)], check=True)
    manifest = dict(
        status="complete",
        source_url="https://arxiv.org/html/2301.10259v2#S3.F1",
        observations="84 source elements reprojected to wider axes; marker outlines unchanged, error endpoints mapped with axes; not raw catalogue data",
        observational_legend="Original publication legend preserved (some labels use preprint years)",
        model="AuroraLF current random-q, epsilon_b=0.03, 10 Myr newly formed mass, Reed07 HMF, no metallicity veto",
        xlim=[6.5, 19],
        ylim_log10_sfrd=[-7, -1.5],
        source_point_transform=f"x_new = 54 + (x_old - 54) * {1 + EXTRA_WIDTH / SOURCE_WIDTH}; y unchanged; marker shapes not scaled",
        layout={"width_pt": WIDTH, "height_pt": 385, "axes_extra_width_pt": EXTRA_WIDTH},
        observation_elements={"marker_elements": marker_count, "error_elements": error_count},
        validation=[
            "exact marker element equality after translation",
            "error endpoints preserve normalized horizontal coordinates",
            "axis calibration matches original ticks to 1e-6 pt",
            "saved model SVG preserves the wide coordinate system",
            "total=PopII+PopIII",
            "input hashes",
        ],
        limitation="Visual comparison only: full-HMF model versus heterogeneous observational selections; no common IMF, UV limit, or dust conversion applied",
        hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [PAPER, RUN / "sfrd.csv", source_svg, svg, pdf, Path(__file__)]
        },
        fonts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in FONTS.values()},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
