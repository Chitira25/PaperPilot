"""Numbers for the Analytics panel. Everything is computed from real data; nothing is a placeholder."""

from collections import Counter
from datetime import date, timedelta, datetime
from typing import Dict, List, Any, Optional

from . import vocab
from . import feedback as fb
from .evaluate import human_agreement


def activity_series(events: List[Dict[str, Any]], days: int = 14, today: Optional[date] = None) -> Dict[str, List]:
    """Events per day for the last `days` days (zeros included)."""
    today = today or date.today()
    start = today - timedelta(days=days - 1)
    counts: Counter = Counter()
    for e in events:
        try:
            d = datetime.fromisoformat(e["time"]).date()
        except (KeyError, ValueError):
            continue
        if start <= d <= today:
            counts[d] += 1
    labels = [(start + timedelta(days=i)) for i in range(days)]
    return {"day": [d.strftime("%d %b") for d in labels], "events": [counts.get(d, 0) for d in labels]}


def topic_counts(papers: List[Dict[str, Any]], top: int = 8) -> List[tuple]:
    """How many papers mention each recognised topic (population, device/method, outcome, intervention)."""
    counter: Counter = Counter()
    for p in papers:
        text = p.get("full_text", "") or p.get("abstract", "")
        seen = set()
        for field in ("population", "method", "outcome", "intervention"):
            for label, _, _ in vocab.find_labels(field, text):
                seen.add(label)
        counter.update(seen)
    return [(vocab.label_text(k), v) for k, v in counter.most_common(top)]


def summary_numbers(papers: List[Dict[str, Any]], results: Optional[Dict[str, Any]],
                    feedback_entries: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = results or {}
    hyps = results.get("hypotheses", [])
    vers = results.get("verifications", {})
    labels = [v.get("verification_label") for v in vers.values() if v.get("verification_label")]
    supported = sum(1 for l in labels if l in ("VERIFIED", "PARTIALLY_VERIFIED"))
    fulltext = sum(1 for p in papers if p.get("source_type") != "abstract" and p.get("word_count", 0) >= 400)
    return {
        "papers": len(papers),
        "fulltext": fulltext,
        "abstract_only": len(papers) - fulltext,
        "claims": len(results.get("claims", [])),
        "contradictions": len(results.get("contradictions", [])),
        "gaps": len(results.get("research_gaps", [])),
        "hypotheses": len(hyps),
        "decisions": len(fb.latest_by_hypothesis(feedback_entries)),
        "searches": sum(1 for e in events if e.get("kind") == "search"),
        "supported_share": round(100 * supported / len(labels)) if labels else None,
        "agreement": (None if human_agreement(feedback_entries) is None else round(100 * human_agreement(feedback_entries))),
    }
