"""
Paper discovery: search free scholarly databases and normalise the results.

Sources (no API key needed):
  * Europe PMC  - biomedical literature (includes PubMed records)
  * OpenAlex    - broad scholarly index, good for biomedical ENGINEERING (sensors, devices, signal processing)

PaperPilot never invents papers. Everything shown comes from these databases, with a link to the source.
If a database cannot be reached, the status message says so and the other source is still used.
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple

import requests

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "PaperPilot/2.0 (open research assistant)"}
TIMEOUT = 25

DOMAIN_PRESETS = {
    "Medical instrumentation": "medical device OR biomedical instrumentation OR biosensor",
    "Biomedical signals (ECG / EEG / EMG)": "ECG OR EEG OR EMG signal acquisition processing",
    "Wearables & remote monitoring": "wearable sensor remote patient monitoring",
    "Medical imaging": "medical imaging ultrasound OR MRI OR CT image analysis",
    "Rehabilitation engineering": "rehabilitation engineering stroke OR Parkinson sensor",
    "Diagnostics & biosensors": "point-of-care diagnostics biosensor",
    "Clinical AI": "machine learning clinical diagnosis medical device",
}


# ------------------------------------------------------------------ helpers
def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:90]


def _clean_doi(doi: str) -> str:
    doi = (doi or "").strip()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).lower()


def make_rid(rec: Dict[str, Any]) -> str:
    key = rec.get("doi") or rec.get("pmid") or _norm_title(rec.get("title", ""))
    return "R" + hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def abstract_from_inverted_index(index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the text."""
    if not index:
        return ""
    slots: Dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


# ------------------------------------------------------------------ query building
def build_europepmc_query(query: str, year_from: Optional[int] = None, oa_only: bool = False,
                          sort: str = "relevance") -> str:
    q = f"({query.strip()})"
    if year_from:
        q += f" AND (PUB_YEAR:[{int(year_from)} TO 3000])"
    if oa_only:
        q += " AND OPEN_ACCESS:y"
    if sort == "citations":
        q += " sort_cited:y"
    elif sort == "recent":
        q += " sort_date:y"
    return q


def build_openalex_params(query: str, year_from: Optional[int], oa_only: bool, sort: str, limit: int) -> Dict[str, str]:
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if oa_only:
        filters.append("open_access.is_oa:true")
    params = {
        "search": query.strip(),
        "per-page": str(limit),
        "select": "id,doi,title,publication_year,cited_by_count,authorships,abstract_inverted_index,"
                  "open_access,primary_location,best_oa_location,type",
    }
    if filters:
        params["filter"] = ",".join(filters)
    params["sort"] = {"citations": "cited_by_count:desc", "recent": "publication_date:desc"}.get(sort, "relevance_score:desc")
    return params


# ------------------------------------------------------------------ parsers (tested with fixtures)
def parse_europepmc(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in (data.get("resultList", {}) or {}).get("result", []) or []:
        title = _strip_tags(r.get("title", "")).rstrip(".")
        if not title:
            continue
        year = r.get("pubYear")
        pdf_url = None
        for u in ((r.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            if str(u.get("documentStyle", "")).lower() == "pdf" and str(u.get("availability", "")).lower() in ("open access", "free"):
                pdf_url = u.get("url")
                break
        doi = _clean_doi(r.get("doi", ""))
        rec = {
            "source": "Europe PMC",
            "title": title,
            "authors": _strip_tags(r.get("authorString", "")).rstrip("."),
            "year": int(year) if str(year).isdigit() else None,
            "journal": r.get("journalTitle") or "",
            "doi": doi,
            "pmid": str(r.get("pmid") or ""),
            "abstract": _strip_tags(re.sub(r"<h4>\s*(.*?)\s*</h4>", r"\1: ", r.get("abstractText", "") or "")),
            "citations": int(r.get("citedByCount") or 0),
            "is_oa": str(r.get("isOpenAccess", "N")).upper() == "Y",
            "pdf_url": pdf_url,
            "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}",
            "pub_type": r.get("pubType") or "",
        }
        rec["rid"] = make_rid(rec)
        out.append(rec)
    return out


def parse_openalex(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in data.get("results", []) or []:
        title = _strip_tags(r.get("title") or "")
        if not title:
            continue
        names = [((a.get("author") or {}).get("display_name") or "") for a in (r.get("authorships") or [])]
        names = [n for n in names if n]
        authors = ", ".join(names[:6]) + (" et al." if len(names) > 6 else "")
        loc = r.get("primary_location") or {}
        best = r.get("best_oa_location") or {}
        oa = r.get("open_access") or {}
        doi = _clean_doi(r.get("doi") or "")
        rec = {
            "source": "OpenAlex",
            "title": title,
            "authors": authors,
            "year": r.get("publication_year"),
            "journal": ((loc.get("source") or {}).get("display_name")) or "",
            "doi": doi,
            "pmid": "",
            "abstract": abstract_from_inverted_index(r.get("abstract_inverted_index")),
            "citations": int(r.get("cited_by_count") or 0),
            "is_oa": bool(oa.get("is_oa")),
            "pdf_url": best.get("pdf_url") or None,
            "url": f"https://doi.org/{doi}" if doi else (r.get("id") or ""),
            "pub_type": r.get("type") or "",
        }
        rec["rid"] = make_rid(rec)
        out.append(rec)
    return out


# ------------------------------------------------------------------ merge
def merge_results(lists: List[List[Dict[str, Any]]], sort: str = "relevance", limit: int = 12) -> List[Dict[str, Any]]:
    """Remove duplicates (same DOI or same title), keep the richest record, then order the list."""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    # round-robin so that no single source dominates the top of the list
    longest = max((len(l) for l in lists), default=0)
    for i in range(longest):
        for l in lists:
            if i >= len(l):
                continue
            rec = l[i]
            key = rec.get("doi") or _norm_title(rec["title"])
            if key in merged:
                old = merged[key]
                for f in ("abstract", "pdf_url", "journal", "authors", "pmid"):
                    if not old.get(f) and rec.get(f):
                        old[f] = rec[f]
                old["citations"] = max(old.get("citations", 0), rec.get("citations", 0))
                old["is_oa"] = old.get("is_oa") or rec.get("is_oa")
                if rec["source"] not in old["source"]:
                    old["source"] += " + " + rec["source"]
            else:
                merged[key] = dict(rec)
                order.append(key)
    items = [merged[k] for k in order]
    if sort == "citations":
        items.sort(key=lambda r: -r.get("citations", 0))
    elif sort == "recent":
        items.sort(key=lambda r: (-(r.get("year") or 0), -r.get("citations", 0)))
    return items[:limit]


# ------------------------------------------------------------------ network
def search_europepmc(query, year_from=None, oa_only=False, sort="relevance", limit=12) -> List[Dict[str, Any]]:
    params = {"query": build_europepmc_query(query, year_from, oa_only, sort), "format": "json",
              "resultType": "core", "pageSize": str(limit)}
    resp = requests.get(EUROPE_PMC_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return parse_europepmc(resp.json())


def search_openalex(query, year_from=None, oa_only=False, sort="relevance", limit=12) -> List[Dict[str, Any]]:
    resp = requests.get(OPENALEX_URL, params=build_openalex_params(query, year_from, oa_only, sort, limit),
                        headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return parse_openalex(resp.json())


def search_papers(query: str, year_from: Optional[int] = None, oa_only: bool = False, sort: str = "relevance",
                  limit: int = 12, sources: Tuple[str, ...] = ("europepmc", "openalex")):
    """Search the chosen sources. Returns (records, status) where status explains each source's result."""
    funcs = {"europepmc": ("Europe PMC", search_europepmc), "openalex": ("OpenAlex", search_openalex)}
    lists, status = [], {}
    for key in sources:
        name, fn = funcs[key]
        try:
            recs = fn(query, year_from, oa_only, sort, limit)
            lists.append(recs)
            status[name] = f"{len(recs)} result(s)"
        except Exception as err:                      # offline, timeout, rate limit ...
            status[name] = f"not reachable ({type(err).__name__})"
    return merge_results(lists, sort, limit), status


def fetch_pdf(url: str, max_mb: int = 25) -> bytes:
    """Download an open-access PDF. Raises ValueError when the link is not a real PDF (many are web pages)."""
    resp = requests.get(url, headers=HEADERS, timeout=45, stream=True, allow_redirects=True)
    resp.raise_for_status()
    data = b""
    for chunk in resp.iter_content(65536):
        data += chunk
        if len(data) > max_mb * 1024 * 1024:
            raise ValueError("The PDF is larger than the size limit.")
    if not data.startswith(b"%PDF"):
        raise ValueError("The link opens a web page, not a PDF file.")
    return data


def paper_from_record(rec: Dict[str, Any], paper_id: str = "", pdf_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Turn a search result into a library paper. With PDF bytes the full text is analysed; otherwise only the
    abstract is available and the paper is marked source_type='abstract'."""
    import os
    import tempfile
    from .pdf_processor import paper_from_text, process_paper

    meta = {k: rec.get(k) for k in ("rid", "doi", "url", "journal", "citations", "is_oa", "pdf_url", "source", "pub_type")}
    paper = None
    if pdf_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(pdf_bytes)
            tmp.close()
            full = process_paper(tmp.name, rec.get("title", "paper") + ".pdf", paper_id)
            if full.get("word_count", 0) >= 200:
                full.update(title=rec.get("title") or full["title"], authors=rec.get("authors") or full["authors"],
                            year=str(rec.get("year") or full["year"]))
                if not full.get("abstract") or full["abstract"].endswith("..."):
                    full["abstract"] = rec.get("abstract") or full["abstract"]
                full["source_type"] = "fulltext"
                paper = full
        except Exception:
            paper = None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    if paper is None:
        paper = paper_from_text(paper_id, rec.get("title", ""), rec.get("authors", ""), str(rec.get("year") or ""),
                                rec.get("abstract", ""))
        paper["source_type"] = "abstract"
        if not rec.get("abstract"):
            paper["warning"] = "The database has no abstract for this paper, so nothing can be analysed. Open the link to read it."
    paper.update({k: v for k, v in meta.items() if v is not None})
    return paper
