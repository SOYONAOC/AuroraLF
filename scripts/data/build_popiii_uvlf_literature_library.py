#!/usr/bin/env python3
"""Build a local Pop III / UVLF paper source library from arXiv and ADS."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
import re
import shutil
import ssl
import tarfile
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


LIBRARY_ROOT = Path("external_data/literature_sources/popiii_uvlf_library")
ARXIV_API = "https://export.arxiv.org/api/query"
ADS_API = "https://api.adsabs.harvard.edu/v1/search/query"
USER_AGENT = "AuroraLF PopIII UVLF literature library builder"
ADS_FIELDS = ",".join(
    [
        "title",
        "author",
        "year",
        "bibcode",
        "citation_count",
        "identifier",
        "abstract",
        "pub",
        "volume",
        "page",
        "doi",
    ]
)


@dataclass(frozen=True)
class PaperSeed:
    arxiv_id: str
    category: str
    priority: int
    reason: str


PAPER_SEEDS = [
    PaperSeed(
        "2501.11678",
        "direct_popiii_uvlf",
        1,
        "Pop III galaxy candidate with direct observational Pop III UVLF constraints at z~6-7.",
    ),
    PaperSeed(
        "2401.07396",
        "direct_popiii_uvlf",
        1,
        "Semi-analytic Pop III star formation model explicitly quantifying UVLF impact at z=9-16.",
    ),
    PaperSeed(
        "2505.20263",
        "direct_popiii_uvlf",
        1,
        "Models bright Pop III systems in the reionization era and their abundance implications.",
    ),
    PaperSeed(
        "2509.22776",
        "direct_popiii_uvlf",
        2,
        "Compact Pop III cluster observability and cosmic-streaming effects relevant to number counts.",
    ),
    PaperSeed(
        "1710.09878",
        "popiii_galaxy_observability",
        1,
        "Search strategy for Pop III-bright galaxies tied to pristine gas evolution.",
    ),
    PaperSeed(
        "2211.02038",
        "popiii_galaxy_observability",
        1,
        "JWST observability and identification criteria for Pop III galaxies.",
    ),
    PaperSeed(
        "2301.10259",
        "popiii_galaxy_observability",
        1,
        "Environmental conditions for catching Pop III star-forming galaxies during reionization.",
    ),
    PaperSeed(
        "2211.12970",
        "popiii_galaxy_observability",
        1,
        "FOREVER22 predictions for bright first galaxies with Pop III stars and JWST comparisons.",
    ),
    PaperSeed(
        "2207.04751",
        "popiii_galaxy_observability",
        1,
        "Review of why Pop III identification in early galaxies is difficult.",
    ),
    PaperSeed(
        "2305.14413",
        "popiii_galaxy_observability",
        1,
        "Extremely metal-poor star complex approaching Pop III conditions with JWST.",
    ),
    PaperSeed(
        "2212.04476",
        "popiii_galaxy_observability",
        1,
        "Strong He II emitter with very blue UV slope used as a Pop III diagnostic case.",
    ),
    PaperSeed(
        "2506.17400",
        "popiii_galaxy_observability",
        1,
        "Metal-polluted Pop III galaxy diagnostics for JWST high-z surveys.",
    ),
    PaperSeed(
        "2505.03873",
        "popiii_galaxy_observability",
        2,
        "Extremely metal-poor JWST candidates useful as near-Pop III observational comparators.",
    ),
    PaperSeed(
        "2603.23209",
        "popiii_galaxy_observability",
        2,
        "Mass scale of Pop III starbursts in first-galaxy simulations under high Lyman-Werner background.",
    ),
    PaperSeed(
        "1801.03584",
        "popiii_galaxy_observability",
        2,
        "Caustic-transit observability of individual Pop III stars and remnants.",
    ),
    PaperSeed(
        "1204.0517",
        "popiii_galaxy_observability",
        2,
        "Detectability of lensed Pop III galaxies with HST and JWST.",
    ),
    PaperSeed(
        "1206.0007",
        "popiii_galaxy_observability",
        2,
        "Detectability limits for isolated Pop III stars with JWST.",
    ),
    PaperSeed(
        "2502.03525",
        "popiii_galaxy_observability",
        2,
        "Recent cross-spectrum constraints and predictions for closing in on Pop III stars.",
    ),
    PaperSeed(
        "1105.0921",
        "spectra_imf_uv_output",
        1,
        "First-galaxy spectral evolution, JWST detection limits, and Pop III color criteria.",
    ),
    PaperSeed(
        "1102.5150",
        "spectra_imf_uv_output",
        1,
        "Rest-frame UV-to-optical spectra for extremely metal-poor and metal-free galaxies.",
    ),
    PaperSeed(
        "1008.2114",
        "spectra_imf_uv_output",
        1,
        "Predicted UV properties of very metal-poor starbursts used for SSP/UV interpretation.",
    ),
    PaperSeed(
        "astro-ph/0206390",
        "spectra_imf_uv_output",
        1,
        "Classic evolving spectra of Pop III stars and reionization consequences.",
    ),
    PaperSeed(
        "2109.10655",
        "spectra_imf_uv_output",
        1,
        "Birth mass function of Pop III stars, directly relevant to top-heavy IMF assumptions.",
    ),
    PaperSeed(
        "2205.15328",
        "spectra_imf_uv_output",
        2,
        "CMB-regulated IMF shifts for metal-poor stars, relevant to high-z IMF evolution.",
    ),
    PaperSeed(
        "2312.12109",
        "spectra_imf_uv_output",
        1,
        "ASTRAEUS evolving stellar IMF model for early galaxies and reionization.",
    ),
    PaperSeed(
        "2411.17007",
        "spectra_imf_uv_output",
        1,
        "Top-heavy IMF and star-formation efficiency effects on high-z UV luminosities.",
    ),
    PaperSeed(
        "2408.03189",
        "spectra_imf_uv_output",
        2,
        "Nebular continuum effects on high-z bright galaxy problem and UV slopes.",
    ),
    PaperSeed(
        "2412.02002",
        "spectra_imf_uv_output",
        2,
        "Rapidly rotating massive-star signatures in first galaxies.",
    ),
    PaperSeed(
        "2505.21463",
        "spectra_imf_uv_output",
        2,
        "Spectral evolution tracks for rotating Pop III stars.",
    ),
    PaperSeed(
        "2506.20767",
        "spectra_imf_uv_output",
        2,
        "Impact of Pop III IMF assumptions on earliest-galaxy characteristics.",
    ),
    PaperSeed(
        "2006.15260",
        "formation_enrichment_context",
        1,
        "Constraints on when Pop III star formation ends, setting transition/gating context.",
    ),
    PaperSeed(
        "1305.1325",
        "formation_enrichment_context",
        1,
        "Pop III stars and remnants in high-redshift galaxies.",
    ),
    PaperSeed(
        "1003.0472",
        "formation_enrichment_context",
        1,
        "First-galaxy chemical enrichment, mixing, and continued star formation.",
    ),
    PaperSeed(
        "0902.3263",
        "formation_enrichment_context",
        1,
        "Initial first-galaxy starburst signatures.",
    ),
    PaperSeed(
        "0711.4622",
        "formation_enrichment_context",
        2,
        "Occurrence rates for metal-free galaxies in the early Universe.",
    ),
    PaperSeed(
        "astro-ph/0607013",
        "formation_enrichment_context",
        2,
        "Pop III star-formation dependence on formation redshift and environment.",
    ),
    PaperSeed(
        "0706.4416",
        "formation_enrichment_context",
        2,
        "Photodissociating-background effects on Pop III star formation.",
    ),
    PaperSeed(
        "2407.14294",
        "formation_enrichment_context",
        2,
        "Analytical Pop III formation model with feedback and fragmentation.",
    ),
    PaperSeed(
        "2203.07733",
        "formation_enrichment_context",
        2,
        "21-cm absorption implications for high-z star formation and JWST surveys.",
    ),
    PaperSeed(
        "2208.01612",
        "highz_uvlf_baseline",
        1,
        "JWST z~9-16 UVLF baseline frequently used when evaluating Pop III/top-heavy explanations.",
    ),
    PaperSeed(
        "2304.06658",
        "highz_uvlf_baseline",
        1,
        "Spectroscopic JWST UVLF constraints at z~8.6-13.2.",
    ),
    PaperSeed(
        "2503.15594",
        "highz_uvlf_baseline",
        1,
        "Very high-redshift UVLF estimates with explicit discussion of young/Pop III-compatible sources.",
    ),
    PaperSeed(
        "2310.03799",
        "highz_uvlf_baseline",
        2,
        "Supersonic streaming impact on the faint end of the JWST UVLF and Pop III timing.",
    ),
    PaperSeed(
        "2307.15305",
        "highz_uvlf_baseline",
        2,
        "Bursty star formation explanation for bright cosmic-dawn galaxies; comparison baseline to Pop III/IMF models.",
    ),
    PaperSeed(
        "2304.04348",
        "highz_uvlf_baseline",
        2,
        "Standard galaxy-formation baseline for ultra-high-z galaxy abundance.",
    ),
    PaperSeed(
        "2307.12487",
        "highz_uvlf_baseline",
        2,
        "General formation scenario for JWST high-z galaxies in Lambda-CDM.",
    ),
    PaperSeed(
        "2407.02674",
        "highz_uvlf_baseline",
        2,
        "Weakly mass-dependent star-formation efficiency model for elevated UV luminosity density.",
    ),
    PaperSeed(
        "2606.02738",
        "highz_uvlf_baseline",
        2,
        "Recent bursty-star-formation UVLF model for JWST high-redshift galaxies.",
    ),
    PaperSeed(
        "2606.07357",
        "highz_uvlf_baseline",
        2,
        "Recent non-standard-spectrum model for monolithic high-z galaxies and UVLF calculation.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=LIBRARY_ROOT,
        help="Output directory for the local paper library.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Delay between arXiv download requests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download PDFs and source packages even if present.",
    )
    return parser.parse_args()


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError(
            "certifi is required in the project .venv so Python can verify HTTPS certificates"
        ) from exc
    return ssl.create_default_context(cafile=certifi.where())


def request_bytes(
    url: str,
    context: ssl.SSLContext,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers, data=data)
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        return response.read()


def query_arxiv_metadata(arxiv_ids: list[str], context: ssl.SSLContext) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for start in range(0, len(arxiv_ids), 25):
        chunk = arxiv_ids[start : start + 25]
        url = ARXIV_API + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": str(len(chunk))}
        )
        root = ET.fromstring(request_bytes(url, context))
        for entry in root.findall("atom:entry", namespace):
            raw_id = entry.findtext("atom:id", default="", namespaces=namespace).rsplit(
                "/abs/", 1
            )[-1]
            base_id = re.sub(r"v\d+$", "", raw_id)
            links = []
            for link in entry.findall("atom:link", namespace):
                links.append(dict(link.attrib))
            entries[base_id] = {
                "arxiv_id": base_id,
                "arxiv_versioned_id": raw_id,
                "title": normalize_space(
                    entry.findtext("atom:title", default="", namespaces=namespace)
                ),
                "summary": normalize_space(
                    entry.findtext("atom:summary", default="", namespaces=namespace)
                ),
                "authors": [
                    normalize_space(author.findtext("atom:name", default="", namespaces=namespace))
                    for author in entry.findall("atom:author", namespace)
                ],
                "published": entry.findtext("atom:published", default="", namespaces=namespace),
                "updated": entry.findtext("atom:updated", default="", namespaces=namespace),
                "primary_category": next(
                    (
                        category.attrib.get("term", "")
                        for category in entry.findall(
                            "{http://arxiv.org/schemas/atom}primary_category"
                        )
                    ),
                    "",
                ),
                "categories": [
                    category.attrib.get("term", "")
                    for category in entry.findall("atom:category", namespace)
                ],
                "doi": entry.findtext(
                    "{http://arxiv.org/schemas/atom}doi", default=""
                ),
                "journal_ref": entry.findtext(
                    "{http://arxiv.org/schemas/atom}journal_ref", default=""
                ),
                "links": links,
            }
        time.sleep(0.5)
    missing = sorted(set(arxiv_ids) - set(entries))
    if missing:
        raise RuntimeError(f"arXiv metadata missing for: {', '.join(missing)}")
    return entries


def query_ads_metadata(arxiv_id: str, context: ssl.SSLContext) -> dict | None:
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        raise RuntimeError("ADS_API_TOKEN is required for ADS metadata")
    query = f'identifier:"arXiv:{arxiv_id}"'
    url = ADS_API + "?" + urllib.parse.urlencode(
        {
            "q": query,
            "fl": ADS_FIELDS,
            "rows": "2",
            "sort": "citation_count desc",
        }
    )
    data = json.loads(
        request_bytes(url, context, headers={"Authorization": f"Bearer {token}"})
    )
    docs = data.get("response", {}).get("docs", [])
    if not docs:
        return None
    return docs[0]


def fetch_ads_bibtex(bibcodes: list[str], context: ssl.SSLContext) -> str:
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        raise RuntimeError("ADS_API_TOKEN is required for ADS BibTeX export")
    if not bibcodes:
        return ""
    payload = json.dumps({"bibcode": bibcodes}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        exported = json.loads(response.read().decode("utf-8"))
    return exported.get("export", "")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def ascii_slug(value: str, max_words: int = 5) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    if not words:
        return "paper"
    return "".join(word[:1].upper() + word[1:] for word in words[:max_words])


def paper_key(arxiv_entry: dict) -> str:
    author = arxiv_entry["authors"][0].split()[-1] if arxiv_entry["authors"] else "Unknown"
    year = arxiv_entry["published"][:4] or "0000"
    return ascii_slug(author, max_words=1) + year + ascii_slug(arxiv_entry["title"], max_words=4)


def download_file(
    url: str,
    destination: Path,
    context: ssl.SSLContext,
    force: bool,
) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    data = request_bytes(url, context)
    temporary.write_bytes(data)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty file from {url}")
    temporary.replace(destination)


def classify_and_extract_source(source_raw: Path, paper_dir: Path, force: bool) -> str:
    data = source_raw.read_bytes()
    source_dir = paper_dir / "source"
    if force and source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(exist_ok=True)

    if data.startswith(b"%PDF"):
        destination = paper_dir / "source_from_arxiv_eprint.pdf"
        source_raw.replace(destination)
        return "pdf_from_eprint"

    if data.startswith(b"\x1f\x8b"):
        source_tar = paper_dir / "source.tar.gz"
        source_raw.replace(source_tar)
        try:
            with tarfile.open(source_tar, "r:gz") as archive:
                safe_extract(archive, source_dir)
            return "tar_gz"
        except tarfile.TarError:
            with gzip.open(source_tar, "rb") as compressed:
                source_bytes = compressed.read()
            if source_bytes.startswith(b"%!PS"):
                single_gz = paper_dir / "source.ps.gz"
                extracted_name = "source.ps"
                source_type = "single_ps_gz"
            else:
                single_gz = paper_dir / "source.tex.gz"
                extracted_name = "source.tex"
                source_type = "single_tex_gz"
            source_tar.replace(single_gz)
            with gzip.open(single_gz, "rb") as compressed:
                tex_bytes = compressed.read()
            (source_dir / extracted_name).write_bytes(tex_bytes)
            return source_type

    if tarfile.is_tarfile(source_raw):
        source_tar = paper_dir / "source.tar"
        source_raw.replace(source_tar)
        with tarfile.open(source_tar, "r:") as archive:
            safe_extract(archive, source_dir)
        return "tar"

    if data.startswith(b"%!PS"):
        destination = paper_dir / "source.ps"
        source_raw.replace(destination)
        return "postscript"

    destination = paper_dir / "source.raw"
    source_raw.replace(destination)
    return "unknown_raw"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if root != member_path and root not in member_path.parents:
            raise RuntimeError(f"Unsafe path in source archive: {member.name}")
    archive.extractall(destination, filter="data")


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def arxiv_eprint_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{arxiv_id}"


def build_record(seed: PaperSeed, arxiv_entry: dict, ads_entry: dict | None) -> dict:
    authors = arxiv_entry["authors"]
    ads_title = normalize_space((ads_entry or {}).get("title", [""])[0])
    ads_abstract = normalize_space((ads_entry or {}).get("abstract", ""))
    return {
        "key": paper_key(arxiv_entry),
        "arxiv_id": seed.arxiv_id,
        "arxiv_versioned_id": arxiv_entry["arxiv_versioned_id"],
        "title": arxiv_entry["title"],
        "authors": authors,
        "year": arxiv_entry["published"][:4],
        "published": arxiv_entry["published"],
        "updated": arxiv_entry["updated"],
        "category": seed.category,
        "priority": seed.priority,
        "selection_reason": seed.reason,
        "primary_category": arxiv_entry["primary_category"],
        "arxiv_categories": arxiv_entry["categories"],
        "arxiv_summary": arxiv_entry["summary"],
        "doi": arxiv_entry["doi"] or first_value((ads_entry or {}).get("doi")),
        "journal_ref": arxiv_entry["journal_ref"],
        "ads_bibcode": (ads_entry or {}).get("bibcode", ""),
        "ads_title": ads_title,
        "ads_pub": (ads_entry or {}).get("pub", ""),
        "ads_year": (ads_entry or {}).get("year", ""),
        "ads_citation_count": (ads_entry or {}).get("citation_count", 0),
        "ads_abstract": ads_abstract,
        "ads_identifiers": (ads_entry or {}).get("identifier", []),
    }


def first_value(values: list | None) -> str:
    if not values:
        return ""
    return values[0]


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "key",
        "arxiv_id",
        "title",
        "year",
        "category",
        "priority",
        "ads_citation_count",
        "ads_bibcode",
        "doi",
        "journal_ref",
        "source_type",
        "pdf_path",
        "source_path",
        "source_extract_path",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_readme(records: list[dict], path: Path) -> None:
    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)

    lines = [
        "# Pop III / UVLF Literature Library",
        "",
        "Local arXiv source/PDF cache for Pop III, metal-free stellar populations,",
        "top-heavy IMF UV output, and high-redshift UV luminosity-function baselines",
        "used by AuroraLF research work.",
        "",
        "Generated by:",
        "",
        "```bash",
        "PYTHONPATH=. .venv/bin/python scripts/data/build_popiii_uvlf_literature_library.py",
        "```",
        "",
        "## Files",
        "",
        "- `manifest.json`: full metadata, ADS provenance, paths, abstracts, and selection notes.",
        "- `papers.csv`: compact table for quick scanning.",
        "- `references.bib`: ADS BibTeX export for records with ADS bibcodes.",
        "- `papers/<key>/paper.pdf`: arXiv PDF.",
        "- `papers/<key>/source.*`: arXiv e-print source package or arXiv e-print PDF when no TeX source is exposed.",
        "- `papers/<key>/source/`: extracted TeX/source tree when the e-print is a source archive.",
        "",
        "## Categories",
        "",
    ]
    for category in sorted(by_category):
        category_records = sorted(
            by_category[category],
            key=lambda item: (item["priority"], -int(item.get("ads_citation_count") or 0), item["year"]),
        )
        lines.append(f"### {category}")
        lines.append("")
        for record in category_records:
            lines.append(
                f"- `{record['key']}` ({record['year']}): {record['title']} "
                f"[arXiv:{record['arxiv_id']}]. {record['selection_reason']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    context = ssl_context()
    root = args.library_root
    papers_root = root / "papers"
    root.mkdir(parents=True, exist_ok=True)
    papers_root.mkdir(exist_ok=True)

    seeds_by_id = {seed.arxiv_id: seed for seed in PAPER_SEEDS}
    arxiv_ids = list(seeds_by_id)
    arxiv_metadata = query_arxiv_metadata(arxiv_ids, context)

    records: list[dict] = []
    for arxiv_id in arxiv_ids:
        seed = seeds_by_id[arxiv_id]
        ads_metadata = query_ads_metadata(arxiv_id, context)
        record = build_record(seed, arxiv_metadata[arxiv_id], ads_metadata)
        paper_dir = papers_root / record["key"]
        paper_dir.mkdir(exist_ok=True)

        pdf_path = paper_dir / "paper.pdf"
        download_file(arxiv_pdf_url(arxiv_id), pdf_path, context, args.force)
        time.sleep(args.sleep_seconds)

        raw_source = paper_dir / "source.eprint"
        if args.force or not any(
            (paper_dir / name).exists()
            for name in [
                "source.tar.gz",
                "source.tar",
                "source.tex.gz",
                "source.ps.gz",
                "source.ps",
                "source.raw",
                "source_from_arxiv_eprint.pdf",
            ]
        ):
            download_file(arxiv_eprint_url(arxiv_id), raw_source, context, force=True)
            source_type = classify_and_extract_source(raw_source, paper_dir, args.force)
            time.sleep(args.sleep_seconds)
        else:
            source_type = existing_source_type(paper_dir)

        record.update(
            {
                "paper_dir": str(paper_dir),
                "pdf_path": str(pdf_path),
                "source_type": source_type,
                "source_path": str(source_artifact_path(paper_dir, source_type)),
                "source_extract_path": str(paper_dir / "source"),
            }
        )
        records.append(record)

    records.sort(key=lambda item: (item["category"], item["priority"], item["year"], item["key"]))
    (root / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(records, root / "papers.csv")
    write_readme(records, root / "README.md")
    bibcodes = [record["ads_bibcode"] for record in records if record.get("ads_bibcode")]
    (root / "references.bib").write_text(fetch_ads_bibtex(bibcodes, context), encoding="utf-8")

    print(f"Wrote {len(records)} papers to {root}")
    by_source_type: dict[str, int] = {}
    for record in records:
        by_source_type[record["source_type"]] = by_source_type.get(record["source_type"], 0) + 1
    for source_type, count in sorted(by_source_type.items()):
        print(f"{source_type}: {count}")


def existing_source_type(paper_dir: Path) -> str:
    for filename, source_type in [
        ("source.tar.gz", "tar_gz"),
        ("source.tar", "tar"),
        ("source.tex.gz", "single_tex_gz"),
        ("source.ps.gz", "single_ps_gz"),
        ("source.ps", "postscript"),
        ("source.raw", "unknown_raw"),
        ("source_from_arxiv_eprint.pdf", "pdf_from_eprint"),
    ]:
        if (paper_dir / filename).exists():
            return source_type
    raise RuntimeError(f"No source artifact found in {paper_dir}")


def source_artifact_path(paper_dir: Path, source_type: str) -> Path:
    filename_by_type = {
        "tar_gz": "source.tar.gz",
        "tar": "source.tar",
        "single_tex_gz": "source.tex.gz",
        "single_ps_gz": "source.ps.gz",
        "postscript": "source.ps",
        "unknown_raw": "source.raw",
        "pdf_from_eprint": "source_from_arxiv_eprint.pdf",
    }
    return paper_dir / filename_by_type[source_type]


if __name__ == "__main__":
    main()
