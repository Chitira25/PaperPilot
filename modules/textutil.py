"""
Small text helpers shared by all PaperPilot modules.
The most important one is quote_is_real(): it is the anti-hallucination check.
"""

import re
from typing import List

_SMALL_WORDS = {"and", "of", "the", "for", "in", "on", "to", "a", "an", "vs", "with", "&"}


def clean_text(s: str) -> str:
    """Lower-case, join words broken by a hyphen at the end of a line, unify quotes/dashes,
    and collapse all white space. Used on BOTH sides of a quote comparison."""
    if not s:
        return ""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    s = re.sub(r"-\s*\n\s*", "", s)          # "hand-\nfunction" -> "handfunction"
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def quote_is_real(quote: str, full_text: str) -> bool:
    """True only if the quote appears (after cleaning) inside the paper text."""
    q = clean_text(quote)
    if len(q) < 15:          # too short to be a meaningful quote
        return False
    t = clean_text(full_text)
    if q in t:
        return True
    # tolerate hyphenation differences such as "single-point" vs "single point"
    q2 = re.sub(r"[-\s]+", " ", q)
    t2 = re.sub(r"[-\s]+", " ", t)
    return q2 in t2


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter that does not break on 'et al.', 'e.g.', 'Fig.' and decimals."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    protected = re.sub(r"\b(et al|e\.g|i\.e|Fig|Figs|Dr|Prof|vs|approx|No)\.", lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean_text(text))


def looks_title_case(words: List[str]) -> bool:
    """Every important word starts with a capital letter (typical for headings)."""
    for w in words:
        base = w.strip("&:.,;()")
        if not base or base.lower() in _SMALL_WORDS:
            continue
        if not base[0].isupper() and not base[0].isdigit():
            return False
    return True
