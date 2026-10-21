"""
Rule-based claim extraction (works offline, no API key).

How it works (simple and checkable):
1. Look at the sections where results are usually written (Abstract, Results, Discussion, Conclusion).
2. Split the text into sentences.
3. Keep sentences that contain a result cue ("improved", "did not improve", "significantly" ...).
4. Tag every kept sentence with the controlled vocabulary (intervention, outcome, population, method,
   dataset) and a direction (positive / none / negative).
   If the sentence does not mention a field, the paper's most frequent label for that field is used.
5. The sentence itself is the quote, so every quote is verbatim by construction.
"""

from typing import Dict, List, Any

from . import vocab
from .textutil import split_sentences, clean_text

# sections searched, in priority order
SECTION_PRIORITY = ["Results", "Abstract", "Conclusion", "Discussion & Limitations", "Full Text"]
MAX_CLAIMS_PER_PAPER = 6
MIN_LEN, MAX_LEN = 35, 340
# sentences that describe a limitation are reported as gaps, not as result claims
LIMITATION_SKIP = ["limitation", "critical gap", "remains an open", "remains unresolved", "unresolved", "future work"]


def _paper_defaults(paper: Dict[str, Any]) -> Dict[str, str]:
    """Most frequent label per field in the whole paper (used when a sentence does not name it)."""
    text = paper.get("full_text", "")
    return {f: vocab.best_label(f, text, prefer="most") for f in vocab.FIELDS}


def tag_claim(sentence: str, defaults: Dict[str, str]) -> Dict[str, str]:
    tags = {}
    for f in vocab.FIELDS:
        label = vocab.best_label(f, sentence, prefer="first")
        tags[f] = label if label != vocab.OTHER else defaults.get(f, vocab.OTHER)
    return tags


def _topic_from_tags(tags: Dict[str, str]) -> str:
    if tags["outcome"] != vocab.OTHER:
        return vocab.label_text(tags["outcome"]).capitalize()
    return "General finding"


def extract_claims_rules(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for idx, paper in enumerate(papers):
        paper_id = paper.get("paper_id", f"P{idx + 1}")
        title = paper.get("title", f"Paper {idx + 1}")
        sections = paper.get("sections", {})
        defaults = _paper_defaults(paper)

        candidates = []
        seen = set()
        for rank, sec in enumerate(SECTION_PRIORITY):
            text = sections.get(sec, "")
            if not text:
                continue
            for sent in split_sentences(text):
                if not (MIN_LEN <= len(sent) <= MAX_LEN):
                    continue
                key = clean_text(sent)
                if key in seen:
                    continue
                if any(c in sent.lower() for c in LIMITATION_SKIP) or "gap:" in sent.lower():
                    continue
                direction = vocab.detect_direction(sent)
                if direction is None:
                    continue
                cues = vocab.count_result_cues(sent)
                tags = tag_claim(sent, defaults)
                if tags["intervention"] == vocab.OTHER and tags["outcome"] == vocab.OTHER and cues < 2:
                    continue  # too vague to be a useful claim
                seen.add(key)
                n_known = sum(1 for f in vocab.FIELDS if tags[f] != vocab.OTHER)
                score = cues * 2 + n_known - rank * 0.5
                candidates.append((score, sec, sent, direction, tags, cues, n_known))

        candidates.sort(key=lambda c: -c[0])
        for n, (score, sec, sent, direction, tags, cues, n_known) in enumerate(candidates[:MAX_CLAIMS_PER_PAPER], start=1):
            claims.append({
                "id": f"{paper_id}_C{n}",
                "paper_id": paper_id,
                "paper_title": title,
                "topic": _topic_from_tags(tags),
                "statement": sent,
                "quote": sent,
                "section": sec,
                # rule-based confidence: more result cues and more recognised fields = more confident
                "confidence": round(min(0.95, 0.45 + 0.1 * cues + 0.05 * n_known), 2),
                "methodology": f"Rule-based extraction from {sec}",
                "intervention": tags["intervention"],
                "outcome": tags["outcome"],
                "population": tags["population"],
                "method": tags["method"],
                "dataset": tags["dataset"],
                "direction": direction,
                "extracted_by": "rules",
            })
    return claims


def extract_limitation_sentences(papers: List[Dict[str, Any]], max_per_paper: int = 3) -> List[Dict[str, Any]]:
    """Real sentences where the authors themselves state a limitation or a missing piece."""
    cues = ["limitation", "remains an open", "remains unresolved", "unresolved", "future work", "lack of",
            "not yet", "has not been", "have not been", "was not tested", "were not tested", "only tested",
            "small sample", "single clinic", "single-center", "single center", "cannot generalize", "gap"]
    out = []
    for idx, paper in enumerate(papers):
        pid = paper.get("paper_id", f"P{idx + 1}")
        sec_text = paper.get("sections", {}).get("Discussion & Limitations", "")
        if not sec_text:
            continue
        found = 0
        for sent in split_sentences(sec_text):
            low = sent.lower()
            if MIN_LEN <= len(sent) <= MAX_LEN and any(c in low for c in cues):
                out.append({"paper_id": pid, "paper_title": paper.get("title", pid), "quote": sent})
                found += 1
                if found >= max_per_paper:
                    break
    return out
