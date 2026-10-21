"""
PDF Processor Module for PaperPilot.
Handles extraction of raw text, section segmentation, metadata detection,
and statistical profiling from research paper PDFs using pypdf.
"""

import io
import re
from typing import Dict, List, Any, Optional
import pypdf

from .textutil import looks_title_case


# Heading keywords -> canonical section name. The FIRST matching entry wins.
HEADING_KEYS = [
    ("Abstract", ["abstract"]),
    ("Introduction", ["introduction", "background"]),
    ("Methodology", ["method", "methodology", "experimental setup", "architecture", "evaluation protocol",
                     "materials", "participants", "study design", "implementation"]),
    ("Results", ["result", "finding", "claims", "observations", "experimental evaluation", "experiments"]),
    ("Discussion & Limitations", ["limitation", "gap", "discussion", "unresolved"]),
    ("Conclusion", ["conclusion", "summary", "concluding"]),
    ("References", ["references", "bibliography"]),
]
# unnumbered one-word headings that are accepted as they are
EXACT_HEADINGS = {"abstract", "introduction", "background", "methods", "method", "methodology", "results",
                  "findings", "discussion", "limitations", "conclusion", "conclusions", "summary",
                  "references", "bibliography"}

_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*[\.\)]?|[IVX]{1,4}[\.\)])\s+(.+)$")

STOPWORDS = set("""about above after again against also among because been before being between both cannot could does doing
during each from further have having here into more most other over same should some such than that their them then there
these they this those through under until very were what when where which while with without would your paper study results
using used based show shown across within however thus therefore""".split())


def paper_id_for(filename: str, index: int) -> str:
    """Use the id at the start of the file name when present (P1_xxx.pdf -> P1), else P<index+1>.
    This keeps ids stable, so ground_truth.json and feedback.json keep matching the right paper."""
    m = re.match(r"^(P\d+)[_\-\s]", filename, flags=re.IGNORECASE)
    return m.group(1).upper() if m else f"P{index + 1}"


def classify_heading(line: str):
    """Decide whether a line is a section heading.
    Returns (section_name, remainder_text) or None.
    A heading must be a SHORT line that looks like a heading. A normal sentence that merely
    starts with a word such as 'Results' or 'Methods' is NOT a heading (that was a bug before)."""
    s = line.strip()
    if not s or len(s) > 90:
        return None

    numbered = False
    title = s
    m = _NUMBERED.match(s)
    if m:
        numbered = True
        title = m.group(2).strip()

    remainder = ""
    head = title
    if ":" in title:
        before, after = title.split(":", 1)
        if len(before.split()) <= 4 and after.strip():
            head, remainder = before.strip(), after.strip()
        else:
            head = title.rstrip(":").strip()
    head = head.rstrip(":").strip()

    words = head.split()
    if not (1 <= len(words) <= 8):
        return None
    if head.endswith((".", ",", ";")) and len(words) > 1:
        return None
    if not any(ch.isalpha() for ch in head):
        return None

    if numbered:
        if not (head[0].isupper() and looks_title_case(words)):
            return None
    else:
        ok = head.isupper() or head.lower() in EXACT_HEADINGS or (len(words) <= 4 and looks_title_case(words))
        if not ok:
            return None

    low = head.lower()
    for section, keys in HEADING_KEYS:
        if any(k in low for k in keys):
            return section, remainder
    return None


def extract_text_from_pdf(pdf_source) -> Dict[str, Any]:
    """
    Extracts text and metadata from a PDF file (either filepath, bytes, or file-like object).
    Returns a dictionary with raw text, pages, page_count, and metadata.
    """
    if isinstance(pdf_source, (str, bytes)):
        if isinstance(pdf_source, str):
            reader = pypdf.PdfReader(pdf_source)
        else:
            reader = pypdf.PdfReader(io.BytesIO(pdf_source))
    elif hasattr(pdf_source, "read"):
        # Streamlit UploadedFile or BytesIO
        content = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        reader = pypdf.PdfReader(io.BytesIO(content))
    else:
        raise ValueError("Unsupported PDF source type.")

    pages_text = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages_text.append({"page_number": idx + 1, "text": page_text})

    full_text = "\n\n".join([p["text"] for p in pages_text])
    
    # Try reading PDF metadata
    pdf_meta = {}
    try:
        if reader.metadata:
            for k, v in reader.metadata.items():
                clean_k = str(k).replace("/", "").lower()
                pdf_meta[clean_k] = str(v)
    except Exception:
        pass

    return {
        "full_text": full_text.strip(),
        "pages": pages_text,
        "page_count": len(pages_text),
        "pdf_metadata": pdf_meta
    }


def segment_sections(text: str) -> Dict[str, str]:
    """
    Split the paper text into canonical sections (Abstract, Introduction, Methodology, Results,
    Discussion & Limitations, Conclusion). The reference list is removed.
    If no heading is found at all, the whole text is returned as one 'Full Text' section.
    """
    order: List[str] = []
    buffers: Dict[str, List[str]] = {}
    current = "Front Matter"
    found_heading = False

    def push(sec: str, line: str):
        if sec not in buffers:
            buffers[sec] = []
            order.append(sec)
        buffers[sec].append(line)

    for line in text.split("\n"):
        hit = classify_heading(line)
        if hit:
            found_heading = True
            current = hit[0]
            if current not in buffers:
                buffers[current] = []
                order.append(current)
            if hit[1]:
                push(current, hit[1])
        else:
            push(current, line)

    if not found_heading:
        return {"Full Text": text.strip()} if text.strip() else {}

    sections = {sec: "\n".join(buffers[sec]).strip() for sec in order}
    sections.pop("References", None)
    if "Abstract" in sections:
        sections.pop("Front Matter", None)      # title / authors block
    return {k: v for k, v in sections.items() if v}


def infer_metadata(text: str, filename: str, pdf_meta: Dict[str, str]) -> Dict[str, Any]:
    """
    Infers Title, Authors, Year, and Key Statistics from text and filename.
    """
    title = pdf_meta.get("title", "")
    if not title or len(title.strip()) < 5 or any(term in title.lower() for term in ["untitled", "anonymous", "none"]):
        # Heuristic: First 1-3 non-empty lines before Abstract or Authors
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        candidate_lines = []
        for l in lines[:10]:
            if any(term in l.lower() for term in ["abstract", "authors:", "institution:", "dr.", "university", "department"]):
                break
            if len(l) > 10:
                candidate_lines.append(l)
                if len(" ".join(candidate_lines)) > 40:
                    break
        if candidate_lines:
            title = " ".join(candidate_lines)
        else:
            clean_name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
            title = clean_name

    # Clean title
    title = re.sub(r'^(paper\s*\d*[\s_:\-]*)', '', title, flags=re.IGNORECASE).strip()
    if not title:
        clean_name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
        title = clean_name

    # Infer Authors
    authors = pdf_meta.get("author") or ""
    if not authors:
        author_match = re.search(r'(?i)(?:authors?:?|by:?)\s*([^\n\r]+)', text)
        if author_match:
            authors = author_match.group(1).strip()
        else:
            # Check for Dr. or Et al.
            dr_match = re.search(r'((?:Dr\.|Prof\.|[A-Z][a-z]+ [A-Z][a-z]+)(?:,\s*(?:Dr\.|Prof\.|[A-Z][a-z]+ [A-Z][a-z]+|et al\.|and [A-Z][a-z]+ [A-Z][a-z]+))+)', text)
            if dr_match:
                authors = dr_match.group(1).strip()
            else:
                authors = "Academic Authors"

    # Infer Year
    year_match = re.search(r'\b(20[12]\d)\b', text[:2000])
    year = year_match.group(1) if year_match else "2024"

    # Words and vocabulary count
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))

    # Key terms: the most frequent meaningful words in the paper (works for any domain)
    freq: Dict[str, int] = {}
    for w in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", text):
        lw = w.lower()
        if lw not in STOPWORDS:
            freq[lw] = freq.get(lw, 0) + 1
    found_keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "word_count": word_count,
        "unique_word_count": unique_words,
        "key_terms": found_keywords[:8],
        "reading_time_minutes": round(word_count / 200, 1)
    }


def process_paper(pdf_source, filename: str, paper_id: str) -> Dict[str, Any]:
    """
    Main pipeline entry to process a single PDF document.
    """
    raw_data = extract_text_from_pdf(pdf_source)
    full_text = raw_data["full_text"]
    meta = infer_metadata(full_text, filename, raw_data["pdf_metadata"])
    sections = segment_sections(full_text)

    # Extract clean abstract
    abstract = sections.get("Abstract", "")
    if not abstract:
        # fallback: take first 600 chars of Introduction or full text
        abstract = full_text[:600] + "..."

    warning = ""
    if meta["word_count"] < 50:
        warning = ("Almost no selectable text was found. This may be a scanned PDF. "
                   "OCR is not supported, so no claims can be extracted from this file.")

    return {
        "paper_id": paper_id,
        "filename": filename,
        "warning": warning,
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"],
        "abstract": abstract.strip(),
        "sections": sections,
        "page_count": raw_data["page_count"],
        "word_count": meta["word_count"],
        "reading_time_min": meta["reading_time_minutes"],
        "key_terms": meta["key_terms"],
        "full_text": full_text
    }


def key_terms_of(text: str, n: int = 8) -> List[str]:
    """Most frequent meaningful words of a text (any domain)."""
    freq: Dict[str, int] = {}
    for w in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", text or ""):
        lw = w.lower()
        if lw not in STOPWORDS:
            freq[lw] = freq.get(lw, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def paper_from_text(paper_id: str, title: str, authors: str, year: str, abstract: str,
                    full_text: str = "", sections: Dict[str, str] = None, filename: str = "",
                    page_count: int = 1) -> Dict[str, Any]:
    """Build a paper dict (same fields as process_paper) from text that is already available.
    Used for papers found by search: when only the abstract is known, the paper has one 'Abstract' section
    and is flagged as abstract-only, so every check is clearly limited to the abstract."""
    abstract = (abstract or "").strip()
    text = (full_text or abstract).strip()
    sections = sections or ({"Abstract": abstract} if abstract else {})
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    return {
        "paper_id": paper_id,
        "filename": filename or "",
        "warning": "" if len(words) >= 50 else "Very little text is available for this paper (abstract only or no abstract).",
        "title": title or "Untitled",
        "authors": authors or "Unknown authors",
        "year": str(year or ""),
        "abstract": abstract,
        "sections": sections,
        "page_count": page_count,
        "word_count": len(words),
        "reading_time_min": round(len(words) / 200, 1),
        "key_terms": key_terms_of(text),
        "full_text": text,
    }
