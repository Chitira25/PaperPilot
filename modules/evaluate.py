"""
Evaluation against a hand-written answer key (ground truth).

ground_truth.json format:
{
  "claims":   [{"paper_id": "P1", "population": "stroke", "method": "continuous_sEMG",
                "outcome": "hand_function", "direction": "positive"}],
  "findings": [{"type": "contradiction", "paper_ids": ["P1","P2","P3","P4","P5"]},
               {"type": "gap", "population": "parkinsons", "method": "continuous_sEMG"}]
}
Every number is None when it cannot be computed (for example, no ground truth loaded).
"""

from typing import Dict, List, Any, Optional

from . import feedback as fb


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def claim_coverage(gt_claims: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> Optional[float]:
    key = lambda c: (c.get("paper_id"), c.get("population"), c.get("method"), c.get("outcome"), c.get("direction"))
    found = {key(c) for c in claims}
    return _ratio(sum(1 for g in gt_claims if key(g) in found), len(gt_claims))


def contradiction_precision(gt_findings: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> Optional[float]:
    truth = [frozenset(f["paper_ids"]) for f in gt_findings if f.get("type") == "contradiction" and f.get("paper_ids")]
    if not contradictions:
        return None
    made = [frozenset(c.get("papers_a", []) + c.get("papers_b", [])) for c in contradictions]
    return _ratio(sum(1 for m in made if m in truth), len(made))


def contradiction_recall(gt_findings: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> Optional[float]:
    truth = [frozenset(f["paper_ids"]) for f in gt_findings if f.get("type") == "contradiction" and f.get("paper_ids")]
    made = {frozenset(c.get("papers_a", []) + c.get("papers_b", [])) for c in contradictions}
    return _ratio(sum(1 for t in truth if t in made), len(truth))


def gap_recall(gt_findings: List[Dict[str, Any]], gaps: List[Dict[str, Any]]) -> Optional[float]:
    truth = [(f.get("population"), f.get("method")) for f in gt_findings if f.get("type") == "gap"]
    made = {(g.get("population"), g.get("method")) for g in gaps if g.get("gap_kind") == "untested_combination"}
    return _ratio(sum(1 for t in truth if t in made), len(truth))


def human_agreement(entries: List[Dict[str, Any]]) -> Optional[float]:
    latest = fb.latest_by_hypothesis(entries)
    reviewed = len(latest)
    good = sum(1 for e in latest.values() if e["action"] in ("approve", "edit"))
    return _ratio(good, reviewed)


def evaluate_all(ground_truth: Optional[Dict[str, Any]], results: Dict[str, Any],
                 feedback_entries: List[Dict[str, Any]], papers_loaded: int) -> Dict[str, Any]:
    gt = ground_truth or {}
    return {
        "papers_tested": papers_loaded,
        "claim_coverage": claim_coverage(gt["claims"], results.get("claims", [])) if gt.get("claims") else None,
        "contradiction_precision": contradiction_precision(gt.get("findings", []), results.get("contradictions", []))
        if gt.get("findings") else None,
        "contradiction_recall": contradiction_recall(gt.get("findings", []), results.get("contradictions", []))
        if gt.get("findings") else None,
        "gap_recall": gap_recall(gt.get("findings", []), results.get("research_gaps", []))
        if gt.get("findings") else None,
        "human_agreement": human_agreement(feedback_entries),
        "hypotheses_reviewed": len(fb.latest_by_hypothesis(feedback_entries)),
    }
