"""
Rule-based Generator. It writes one hypothesis per finding using ONLY facts found in the papers
(populations, methods, outcomes and paper ids). No invented numbers and no invented mechanisms.
"""

from collections import Counter
from typing import Dict, List, Any

from . import vocab
from .rules import stable_id  # noqa: F401  (kept for symmetry; ids are derived from finding ids)

_lt = vocab.label_text


def _dir_text(direction: str) -> str:
    return {"positive": "an improvement", "none": "no change", "negative": "a decline",
            "reported": "a comparable level of performance"}.get(direction, "an effect")


def hypothesis_for_gap(gap: Dict[str, Any], claims_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    interv, p, m = gap["intervention"], gap["population"], gap["method"]
    pool = [claims_by_id[i] for i in gap.get("supporting_claim_ids", []) if i in claims_by_id]
    using_m = [c for c in pool if c.get("method") == m and c.get("intervention") == interv]
    outcome = Counter(c["outcome"] for c in using_m if c.get("outcome") != vocab.OTHER).most_common(1)
    outcome = outcome[0][0] if outcome else vocab.OTHER
    direction = Counter(c["direction"] for c in using_m if c.get("outcome") == outcome).most_common(1)
    direction = direction[0][0] if direction else "positive"
    papers = sorted({c["paper_id"] for c in using_m})
    pops = sorted({c["population"] for c in using_m})
    pops_txt = ", ".join(_lt(x) for x in pops)

    formulation = (f"If {_lt(interv)} is evaluated in {_lt(p)} using {_lt(m)}, then {_lt(outcome)} will show "
                   f"{_dir_text(direction)}, the same direction that {', '.join(papers)} reported for {pops_txt}. "
                   f"No loaded paper tests this combination.")
    return {
        "id": f"HYP_{gap['id']}",
        "title": f"Test {_lt(m)} in {_lt(p)}",
        "formulation": formulation,
        "addresses_gap": gap["id"],
        "resolves_contradiction": "N/A",
        "finding_id": gap["id"],
        "finding_type": "gap",
        "rationale": (f"{', '.join(papers)} report {_dir_text(direction)} in {_lt(outcome)} with {_lt(m)} "
                      f"in {pops_txt}. {_lt(p).capitalize()} have only been studied with other methods."),
        "testability_score": "High",
        "proposed_methodology": (f"Run {_lt(interv)} in a {_lt(p)} cohort and measure {_lt(outcome)} with {_lt(m)}. "
                                 f"Compare the change with the results reported in {', '.join(papers)}."),
        "falsification_criteria": (f"Rejected if {_lt(interv)} shows no change, or a change in the opposite "
                                   f"direction, in {_lt(outcome)} for {_lt(p)} measured with {_lt(m)}."),
        "target": {"intervention": interv, "outcome": outcome, "population": p, "method": m,
                   "direction": direction},
        "generated_by": "rules",
        "status": "pending",
    }


def hypothesis_for_contradiction(cont: Dict[str, Any]) -> Dict[str, Any]:
    diffs = cont.get("differences", {})
    attrs = ", ".join(diffs.keys()) if diffs else "study conditions"
    interv, outcome = cont["intervention"], cont["outcome"]
    pa, pb = ", ".join(cont["papers_a"]), ", ".join(cont["papers_b"])
    formulation = (f"If {_lt(interv)} is tested under matched conditions that vary only {attrs}, then the "
                   f"difference in {_lt(outcome)} reported between {pa} and {pb} will disappear when {attrs} "
                   f"are matched, which would show that {attrs} explain the conflict.")
    return {
        "id": f"HYP_{cont['id']}",
        "title": f"Explain the {_lt(outcome)} conflict",
        "formulation": formulation,
        "addresses_gap": "N/A",
        "resolves_contradiction": cont["id"],
        "finding_id": cont["id"],
        "finding_type": "contradiction",
        "rationale": cont["root_cause"],
        "testability_score": "High" if diffs else "Medium",
        "proposed_methodology": cont["resolution_path"],
        "falsification_criteria": (f"Rejected if the difference in {_lt(outcome)} between the groups remains "
                                   f"after {attrs} are matched."),
        "target": {"intervention": interv, "outcome": outcome},
        "generated_by": "rules",
        "status": "pending",
    }


def generate_hypotheses_rules(contradictions: List[Dict[str, Any]], gaps: List[Dict[str, Any]],
                              claims: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    cmap = {c["id"]: c for c in claims}
    out = [hypothesis_for_contradiction(c) for c in contradictions]
    out += [hypothesis_for_gap(g, cmap) for g in gaps if g.get("gap_kind") == "untested_combination"]
    return out[:max_items]
