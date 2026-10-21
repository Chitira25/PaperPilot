"""
Quick appraisal of a paper: study design, sample size and a reporting checklist, plus a short extractive summary.

IMPORTANT: this is a CHECKLIST of what the text reports, not a quality score and not medical advice.
When only the abstract is available, a missing item may simply not be in the abstract. The result says so.
"""

import re
from typing import Dict, List, Any, Optional

from . import vocab
from .textutil import split_sentences
from .extraction import extract_limitation_sentences

DESIGN_PATTERNS = [
    ("Systematic review / meta-analysis", r"systematic review|meta-analys"),
    ("Randomised controlled trial", r"randomi[sz]ed (controlled |clinical )*trial|\brct\b"),
    ("Randomised study", r"randomi[sz]ed|randomly assigned"),
    ("Cohort study", r"\bcohort\b"),
    ("Case-control study", r"case-control"),
    ("Cross-sectional study", r"cross-sectional"),
    ("Validation / accuracy study", r"validation study|\bvalidat(ed|ion)\b|agreement with|bland-altman|diagnostic accuracy"),
    ("Pilot / feasibility study", r"pilot (study|trial)|feasibility"),
    ("Case report / series", r"case report|case series"),
    ("Device or method development",
     r"we (design|develop|propose|present|built|build|introduce)\w* (a |an |the )?(novel |new |low-cost |wearable |portable |deep |machine )*"
     r"(device|system|sensor|prototype|method|algorithm|framework|platform|model)"),
    ("Review / overview", r"\breview\b|\boverview\b|\bsurvey\b"),
]

CHECKS = [
    ("Sample size reported", None),   # handled separately
    ("Control or comparison group", r"control group|controls\b|comparison group|sham|placebo|compared (with|to|against)"),
    ("Randomisation", r"randomi[sz]|randomly"),
    ("Blinding", r"\bblind|\bmasked\b"),
    ("Ethics approval or consent", r"ethic(s|al) (committee|approval|board)|institutional review board|\birb\b|informed consent|declaration of helsinki"),
    ("Statistics reported", r"\bp\s*[<=>]\s*0?\.\d+|confidence interval|95\s*%\s*ci|standard deviation|\bsd\b|anova|t-test"),
    ("Performance metrics reported", r"sensitivity|specificity|\bauc\b|\brmse\b|mean absolute error|\bmae\b|correlation|bland-altman|accuracy|f1"),
    ("Limitations stated", None),     # handled separately
]


def _first_sentence_with(text: str, pattern: str) -> str:
    rx = re.compile(pattern, re.IGNORECASE)
    for sent in split_sentences(text):
        if rx.search(sent):
            return sent if len(sent) <= 320 else sent[:317] + "..."
    return ""


def detect_sample_size(text: str):
    """Return (n, sentence) or (None, ''). Looks for 'n = 24' and '24 patients / participants ...'."""
    to_int = lambda s: int(s.replace(",", ""))
    patterns = [
        r"\b[nN]\s*=\s*(\d{1,3}(?:,\d{3})+|\d{1,6})\b",
        r"\b(\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?:\w+[\s-]){0,2}(?:patients|participants|subjects|volunteers|adults|children|individuals|cases|infants)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return to_int(m.group(1)), _first_sentence_with(text, re.escape(m.group(0)))
    return None, ""


def top_labels(text: str, field: str, n: int = 3) -> List[str]:
    hits = sorted(vocab.find_labels(field, text), key=lambda h: (-h[1], h[2]))
    return [vocab.label_text(h[0]) for h in hits[:n]]


def appraise_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    text = paper.get("full_text", "") or paper.get("abstract", "")
    abstract_only = paper.get("source_type") == "abstract" or paper.get("word_count", 0) < 400

    design, design_evidence = "Not stated in the text", ""
    for name, pat in DESIGN_PATTERNS:
        sent = _first_sentence_with(text, pat)
        if sent:
            design, design_evidence = name, sent
            break

    n, n_evidence = detect_sample_size(text)
    limitations = [x["quote"] for x in extract_limitation_sentences([paper], max_per_paper=3)]
    if not limitations:   # abstract-only papers have no limitation section: look in the whole text
        limitations = [s for s in split_sentences(text)
                       if re.search(r"limitation|small sample|single[- ]cent(er|re)|future (work|studies)", s, re.IGNORECASE)][:3]

    checks = []
    for label, pat in CHECKS:
        if label == "Sample size reported":
            checks.append({"label": label, "found": n is not None,
                           "evidence": (f"n = {n}. " if n is not None else "") + n_evidence})
        elif label == "Limitations stated":
            checks.append({"label": label, "found": bool(limitations), "evidence": limitations[0] if limitations else ""})
        else:
            sent = _first_sentence_with(text, pat)
            checks.append({"label": label, "found": bool(sent), "evidence": sent})

    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "design": design,
        "design_evidence": design_evidence,
        "sample_size": n,
        "populations": top_labels(text, "population"),
        "methods": top_labels(text, "method"),
        "outcomes": top_labels(text, "outcome"),
        "interventions": top_labels(text, "intervention"),
        "checks": checks,
        "limitations": limitations,
        "abstract_only": abstract_only,
        "scope_note": ("Only the abstract is available, so a missing item may simply not be mentioned in the abstract."
                       if abstract_only else "Checks cover the full text that PaperPilot could read."),
    }


def appraisal_markdown(a: Dict[str, Any]) -> str:
    L = [f"**{a['title']}** (`{a['paper_id']}`)", ""]
    L.append(f"- **Study design (from the text):** {a['design']}")
    L.append(f"- **Sample size:** {a['sample_size'] if a['sample_size'] is not None else 'not found in the text'}")
    if a["populations"]:
        L.append(f"- **Patient groups mentioned:** {', '.join(a['populations'])}")
    if a["methods"]:
        L.append(f"- **Devices / methods mentioned:** {', '.join(a['methods'])}")
    if a["outcomes"]:
        L.append(f"- **Outcomes mentioned:** {', '.join(a['outcomes'])}")
    L += ["", "| Reporting checklist | Found? | Evidence from the text |", "|---|---|---|"]
    for c in a["checks"]:
        ev = (c["evidence"] or "-").replace("|", "/")
        L.append(f"| {c['label']} | {'Yes' if c['found'] else 'No'} | {ev[:200]} |")
    L += ["", f"*{a['scope_note']} This is a checklist of what the text reports, not a quality score and not medical advice.*"]
    return "\n".join(L)


def summarize_paper(paper: Dict[str, Any], claims: List[Dict[str, Any]], max_claims: int = 4) -> str:
    """Extractive summary: opening of the abstract + the strongest result claims (all sentences are real)."""
    abstract_sents = split_sentences(paper.get("abstract", ""))
    L = [f"**{paper.get('title')}** (`{paper.get('paper_id')}`, {paper.get('year') or 'n.d.'})", ""]
    if abstract_sents:
        L.append("**What it is about (from the abstract):** " + " ".join(abstract_sents[:2]))
    mine = sorted([c for c in claims if c.get("paper_id") == paper.get("paper_id")], key=lambda c: -float(c.get("confidence", 0)))
    if mine:
        L += ["", "**Key findings (result sentences found in the text):**"]
        for c in mine[:max_claims]:
            tag = {"positive": "effect reported", "none": "no effect reported", "negative": "decline reported"}.get(c.get("direction"), "")
            L.append(f"- {c['statement']} *({tag})*")
    else:
        L += ["", "No result sentences were recognised in the available text."]
    L += ["", "*Extractive summary: every sentence above is copied from the paper, nothing is paraphrased or added.*"]
    return "\n".join(L)
