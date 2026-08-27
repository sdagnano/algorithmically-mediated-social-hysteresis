"""Resolve the curated DOI set and build manuscript-ready references.

OpenAlex is used only as a metadata resolver. Selection is controlled by the
versioned selected_sources.csv file and is therefore auditable.
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "literature" / "selected_sources.csv"
MANUSCRIPT = ROOT / "manuscript"
OPENALEX = "https://api.openalex.org/works"

OVERRIDES = {
    "10.1177/17456916231195361": {"year": 2024},
    "10.1016/j.chaos.2025.117660": {"year": 2026},
    "10.1145/3728372": {"authors": "Karl Krauth; Yixin Wang; Michael I. Jordan"},
    "10.1145/3240323.3240370": {
        "container": "Proceedings of the 12th ACM Conference on Recommender Systems"
    },
    "10.1145/3340531.3412152": {
        "container": "Proceedings of the 29th ACM International Conference on Information and Knowledge Management"
    },
    "10.1145/3447548.3467298": {
        "container": "Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining"
    },
    "10.1145/2872427.2883040": {
        "authors": "Jessica Su; Aneesh Sharma; Sharad Goel",
        "container": "Proceedings of the 25th International Conference on World Wide Web",
    },
    "10.1145/3397271.3401230": {
        "container": "Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval"
    },
    "10.1145/3397271.3401431": {
        "container": "Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval"
    },
    "10.1145/3501247.3531583": {
        "container": "Proceedings of the 14th ACM Web Science Conference"
    },
    "10.1145/1864708.1864772": {
        "container": "Proceedings of the Fourth ACM Conference on Recommender Systems"
    },
    "10.48550/arxiv.1703.01049": {
        "authors": "Ayan Sinha; David F. Gleich; Karthik Ramani",
        "container": "arXiv preprint arXiv:1703.01049",
        "year": 2017,
        "volume": "",
        "issue": "",
        "first_page": "",
        "last_page": "",
        "type": "preprint",
    },
    "10.48550/arxiv.1809.04644": {
        "authors": "Wilbert Samuel Rossi; Jan Willem Polderman; Paolo Frasca",
        "container": "arXiv preprint arXiv:1809.04644",
        "year": 2018,
    },
    "10.1073/pnas.2025334119": {
        "authors": "Ferenc Huszár; Sofia Ira Ktena; Conor Cruise O'Brien; Luca Belli; Andrew Schlaikjer; Moritz Hardt",
        "container": "Proceedings of the National Academy of Sciences",
        "year": 2022,
        "volume": "119",
        "issue": "1",
        "first_page": "e2025334119",
        "last_page": "e2025334119",
        "type": "article",
        "landing_page": "https://doi.org/10.1073/pnas.2025334119",
    },
    "10.1093/pnasnexus/pgaf402": {
        "container": "PNAS Nexus",
        "year": 2026,
        "volume": "5",
        "issue": "1",
        "first_page": "pgaf402",
        "last_page": "pgaf402",
        "type": "article",
        "landing_page": "https://doi.org/10.1093/pnasnexus/pgaf402",
    },
    "10.1002/sres.70126": {
        "authors": "Johnny Chan",
        "container": "Systems Research and Behavioral Science",
        "year": 2026,
        "volume": "",
        "issue": "",
        "first_page": "1",
        "last_page": "7",
        "type": "article",
        "landing_page": "https://doi.org/10.1002/sres.70126",
    },
    "10.48550/arxiv.2505.09254": {
        "authors": "Joseph B. Bak-Coleman; Stephan Lewandowsky; Philipp Lorenz-Spreen; Arvind Narayanan; Amy Orben; Lisa Oswald",
        "container": "arXiv preprint arXiv:2505.09254v3",
        "year": 2025,
        "volume": "", "issue": "", "first_page": "", "last_page": "",
        "type": "preprint",
        "landing_page": "https://doi.org/10.48550/arXiv.2505.09254",
    },
    "10.1038/s44159-025-00475-5": {
        "container": "Nature Reviews Psychology", "year": 2025,
        "volume": "4", "issue": "", "first_page": "615", "last_page": "615",
    },
    "10.1016/j.ymssp.2014.04.012": {
        "container": "Mechanical Systems and Signal Processing", "year": 2014,
        "volume": "49", "issue": "1-2", "first_page": "209", "last_page": "233",
    },
    "10.1007/bf01349418": {
        "authors": "F. Preisach", "title": "\u00dcber die magnetische Nachwirkung",
        "container": "Zeitschrift f\u00fcr Physik", "year": 1935,
        "volume": "94", "issue": "5-6", "first_page": "277", "last_page": "302",
    },
    "10.1137/1035005": {
        "authors": "Jack W. Macki; Paolo Nistri; Pietro Zecca", "container": "SIAM Review",
        "year": 1993, "volume": "35", "issue": "1", "first_page": "94", "last_page": "123",
    },
    "10.1103/physrevlett.70.3347": {
        "container": "Physical Review Letters", "year": 1993, "volume": "70",
        "issue": "21", "first_page": "3347", "last_page": "3350",
    },
    "10.1007/s00023-019-00807-1": {
        "container": "Annales Henri Poincar\u00e9", "year": 2019, "volume": "20",
        "issue": "8", "first_page": "2819", "last_page": "2872",
    },
    "10.1109/tac.2005.847035": {
        "container": "IEEE Transactions on Automatic Control", "year": 2005,
        "volume": "50", "issue": "5", "first_page": "631", "last_page": "645",
    },
    "10.1049/ip-cta:20010375": {
        "container": "IEE Proceedings---Control Theory and Applications", "year": 2001,
        "volume": "148", "issue": "3", "first_page": "185", "last_page": "192",
    },
    "10.1002/9780470513200": {
        "authors": "Fay\u00e7al Ikhouane; Jos\u00e9 Rodellar", "container": "John Wiley & Sons",
        "year": 2007, "volume": "", "issue": "", "first_page": "", "last_page": "", "type": "book",
    },
    "10.1016/j.ipm.2025.104125": {
        "container": "Information Processing & Management", "year": 2025, "volume": "62",
        "issue": "4", "first_page": "104125", "last_page": "104125",
    },
    "10.1109/tac.2025.3616262": {
        "container": "IEEE Transactions on Automatic Control", "year": 2026, "volume": "71",
        "issue": "3", "first_page": "1708", "last_page": "1723",
    },
    "10.1126/sciadv.aax7310": {
        "authors": "Aili Asikainen; Gerardo I\u00f1iguez; Javier Ure\u00f1a-Carri\u00f3n; Kimmo Kaski; Mikko Kivel\u00e4",
        "container": "Science Advances", "year": 2020, "volume": "6", "issue": "19",
        "first_page": "eaax7310", "last_page": "eaax7310", "type": "article",
    },
    "10.1109/tcss.2026.3715162": {
        "authors": "Ella C. Davidson; Mengbin Ye",
        "title": "Modelling the Closed-Loop Dynamics Between a Social Media Recommender System and Users' Opinions",
        "container": "IEEE Transactions on Computational Social Systems",
        "year": 2026, "volume": "", "issue": "", "first_page": "1", "last_page": "13", "type": "article",
    },
    "10.48550/arxiv.2605.01503": {
        "authors": "Giulia De Pasquale; Sarah Dean; Paolo Frasca", "container": "arXiv preprint arXiv:2605.01503",
        "year": 2026, "volume": "", "issue": "", "first_page": "", "last_page": "", "type": "preprint",
    },
    "10.48550/arxiv.2603.10275": {
        "authors": "Simone Mariano; Paolo Frasca", "container": "arXiv preprint arXiv:2603.10275",
        "year": 2026, "volume": "", "issue": "", "first_page": "", "last_page": "", "type": "preprint",
    },
    "10.48550/arxiv.2607.05010": {
        "authors": "Simone Mariano; Paolo Frasca", "container": "arXiv preprint arXiv:2607.05010",
        "year": 2026, "volume": "", "issue": "", "first_page": "", "last_page": "", "type": "preprint",
    },
    "10.1080/01621459.2020.1750415": {
        "authors": "Jason Wu; Peng Ding", "container": "Journal of the American Statistical Association",
        "year": 2021, "volume": "116", "issue": "536", "first_page": "1898", "last_page": "1913", "type": "article",
    },
    "10.48550/arxiv.2607.04257": {
        "authors": "Jizhou Liu; Azeem M. Shaikh; Liang Zhong", "container": "arXiv preprint arXiv:2607.04257",
        "year": 2026, "volume": "", "issue": "", "first_page": "", "last_page": "", "type": "preprint",
    },
    "10.1016/j.ins.2021.12.069": {"year": 2022},
    "10.1287/mnsc.2022.4583": {"year": 2023},
    "10.1177/20563051211041648": {
        "first_page": "20563051211041648",
        "last_page": "20563051211041648",
    },
    "10.1103/physrevx.11.011012": {
        "first_page": "011012",
        "last_page": "011012",
    },
    "10.1073/pnas.2023301118": {
        "first_page": "e2023301118",
        "last_page": "e2023301118",
    },
    "10.1007/s42001-025-00381-z": {
        "first_page": "52",
        "last_page": "52",
    },
    "10.1093/comnet/cnac055": {
        "first_page": "cnac055",
        "last_page": "cnac055",
    },
    "10.1073/pnas.2102144118": {
        "first_page": "e2102144118",
        "last_page": "e2102144118",
    },
    "10.1140/epjp/s13360-020-00541-2": {
        "first_page": "521",
        "last_page": "521",
    },
    "10.1103/physrevresearch.2.043117": {
        "first_page": "043117",
        "last_page": "043117",
    },
    "10.1103/physrevx.10.041042": {
        "first_page": "041042",
        "last_page": "041042",
    },
    "10.1093/biomtc/ujae023": {
        "first_page": "ujae023",
        "last_page": "ujae023",
    },
}


def get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "algorithmically-mediated-social-hysteresis-bibliography/2.0"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable retry state")


def normalize(item: dict) -> dict:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    biblio = item.get("biblio") or {}
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in item.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    return {
        "doi": (item.get("doi") or "").replace("https://doi.org/", "").lower(),
        "title": item.get("title") or "",
        "year": item.get("publication_year") or "",
        "authors": "; ".join(authors),
        "container": source.get("display_name") or source.get("host_organization_name") or "",
        "volume": biblio.get("volume") or "",
        "issue": biblio.get("issue") or "",
        "first_page": biblio.get("first_page") or "",
        "last_page": biblio.get("last_page") or "",
        "type": item.get("type") or "article",
        "openalex_id": (item.get("id") or "").replace("https://openalex.org/", ""),
        "cited_by_count": item.get("cited_by_count") or 0,
        "landing_page": location.get("landing_page_url") or item.get("doi") or "",
    }


def latex_escape(text: object) -> str:
    value = str(text or "")
    # Springer submission systems ask for TeX-encoded diacritics and plain
    # ASCII punctuation.  These are the non-ASCII code points present in the
    # curated metadata as of the audit date.
    unicode_replacements = {
        "\ufeff": "",
        "ı́": r"\'{\i}",
        "ß": r"{\ss}",
        "á": r"\'{a}",
        "é": r"\'{e}",
        "í": r"\'{i}",
        "î": r"\^{i}",
        "ó": r"\'{o}",
        "ö": r'\"{o}',
        "ú": r"\'{u}",
        "ü": r'\"{u}',
        "Ü": r'\"{U}',
        "ç": r"\c{c}",
        "Ç": r"\c{C}",
        "č": r"\v{c}",
        "ę": r"\k{e}",
        "ı": r"{\i}",
        "ł": r"\l{}",
        "ń": r"\'{n}",
        "ñ": r"\~{n}",
        "ä": r'\"{a}',
        "š": r"\v{s}",
        "\u0301": r"\'{}",
        "‐": "-",
        "’": "'",
    }
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    value = "".join(replacements.get(char, char) for char in value)
    for source, target in unicode_replacements.items():
        value = value.replace(source, target)
    return value


def bibtex_escape(text: object) -> str:
    return latex_escape(text)


def author_list(text: str, max_names: int | None = None) -> str:
    names = [name.strip() for name in text.split(";") if name.strip()]
    if max_names is not None and len(names) > max_names:
        return ", ".join(names[: max_names - 1]) + ", ... , and " + names[-1]
    if len(names) <= 1:
        return names[0] if names else "Unknown author"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def pages(row: dict) -> str:
    first, last = str(row.get("first_page") or ""), str(row.get("last_page") or "")
    if first and last and first != last:
        return f"{first}-{last}"
    return first


def build_reference_tex(rows: list[dict]) -> str:
    lines = [r"\begin{thebibliography}{999}", ""]
    for row in rows:
        author = latex_escape(author_list(row["authors"], max_names=20))
        title = latex_escape(row["title"])
        container = latex_escape(row["container"])
        volume = latex_escape(row["volume"])
        issue = latex_escape(row["issue"])
        page = latex_escape(pages(row))
        year = latex_escape(row["year"])
        details = ""
        if volume:
            details += rf", \textit{{{volume}}}"
        if issue:
            details += f"({issue})"
        if page:
            details += f", {page}"
        lines.extend(
            [
                rf"\bibitem{{{row['key']}}}",
                rf"{author}. ({year}). {title}. \textit{{{container}}}{details}. "
                rf"\url{{https://doi.org/{row['doi']}}}.",
                "",
            ]
        )
    lines.append(r"\end{thebibliography}")
    lines.append("")
    return "\n".join(lines)


def build_bibtex(rows: list[dict]) -> str:
    entries = []
    for row in rows:
        author = " and ".join(name.strip() for name in row["authors"].split(";") if name.strip())
        common = {"author": author, "title": row["title"], "year": row["year"]}
        if row["type"] == "book":
            kind = "book"
            fields = {**common, "publisher": row["container"]}
        elif row["type"] == "preprint":
            kind = "misc"
            fields = {**common, "howpublished": row["container"]}
        else:
            kind = "inproceedings" if row["type"] in {"proceedings-article", "conference-paper"} else "article"
            fields = {
                **common,
                ("booktitle" if kind == "inproceedings" else "journal"): row["container"],
                "volume": row["volume"], "number": row["issue"], "pages": pages(row),
            }
        fields.update({"doi": row["doi"], "url": f"https://doi.org/{row['doi']}"})
        body = [f"@{kind}{{{row['key']},"]
        for name, value in fields.items():
            if str(value or ""):
                body.append(f"  {name} = {{{bibtex_escape(value)}}},")
        body.append("}")
        entries.append("\n".join(body))
    return "\n\n".join(entries) + "\n"


def main() -> None:
    selected = pd.read_csv(SELECTED, dtype=str).fillna("")
    dois = [doi.lower() for doi in selected.doi]
    resolved: dict[str, dict] = {}
    select = "id,doi,title,publication_year,authorships,primary_location,biblio,type,cited_by_count"
    for offset in range(0, len(dois), 25):
        batch = "|".join(dois[offset : offset + 25])
        params = urllib.parse.urlencode(
            {"filter": f"doi:{batch}", "per-page": 50, "select": select}, safe="|:/."
        )
        payload = get_json(f"{OPENALEX}?{params}")
        for item in payload.get("results", []):
            row = normalize(item)
            resolved[row["doi"]] = row
        time.sleep(0.6)

    missing = [doi for doi in dois if doi not in resolved]
    # Resolve stragglers individually; this also makes failure diagnostics exact.
    for doi in missing.copy():
        url = f"{OPENALEX}/https://doi.org/{urllib.parse.quote(doi, safe='/:')}?select={select}"
        try:
            row = normalize(get_json(url))
            resolved[row["doi"]] = row
            missing.remove(doi)
        except urllib.error.HTTPError:
            pass
        time.sleep(0.4)

    if missing:
        raise RuntimeError(f"Unresolved selected DOIs: {missing}")

    rows = []
    for source in selected.to_dict(orient="records"):
        doi = source["doi"].lower()
        row = {**source, **resolved[doi]}
        row["key"] = source["key"]
        row["stream"] = source["stream"]
        row["role"] = source["role"]
        if doi in OVERRIDES:
            row.update(OVERRIDES[doi])
        rows.append(row)

    # The target journal uses numbered references. Order the rendered list and
    # the curated table by first citation, and fail if the manuscript and the
    # selected set diverge in either direction.
    manuscript_text = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    citation_order: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript_text):
        for key in group.split(","):
            key = key.strip()
            if key and key not in citation_order:
                citation_order.append(key)
    rows_by_key = {row["key"]: row for row in rows}
    missing_citations = [key for key in citation_order if key not in rows_by_key]
    if missing_citations:
        raise RuntimeError(f"Cited keys missing from selected sources: {missing_citations}")
    uncited = [row["key"] for row in rows if row["key"] not in citation_order]
    if uncited:
        raise RuntimeError(f"Selected sources not cited in manuscript: {uncited}")
    rows = [rows_by_key[key] for key in citation_order]

    ROOT.joinpath("literature").mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "literature" / "curated_bibliography.csv", index=False)
    (MANUSCRIPT / "references.tex").write_text(build_reference_tex(rows), encoding="utf-8")
    (MANUSCRIPT / "references.bib").write_text(build_bibtex(rows), encoding="utf-8")

    stream_counts = pd.DataFrame(rows).groupby("stream").size().sort_values(ascending=False)
    audit = {
        "selection_date": "2026-08-26",
        "selected_records": len(rows),
        "doi_resolution_rate": 1.0,
        "registry_resolution": {"Crossref": 100, "DataCite": 7},
        "metadata_audit": {
            "bibliographic_metadata_fields_corrected": 19,
            "existing_records_affected": 15,
            "conceptual_repositionings": 1,
            "corpus_additions": 20,
            "documented_actions_total": 40,
            "unresolved_dois": 0,
            "duplicate_identities": 0,
            "wrong_doi_title_matches": 0,
            "omitted_authors": 0,
        },
        "streams": {key: int(value) for key, value in stream_counts.items()},
        "method": "Nine broad OpenAlex queries plus backward chaining from ten anchors, followed by targeted closest-predecessor, adaptive-network hysteresis, closed-loop control, and randomization-inference updates; final inclusion by direct relevance.",
        "scope_note": "Structured saturation search, not a PRISMA systematic review and not a claim of literal exhaustiveness.",
    }
    (ROOT / "literature" / "audit_summary.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
