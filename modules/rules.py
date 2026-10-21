"""
Analysis rules (no AI, fully repeatable): contradictions and research gaps.

Contradiction rule
    Group claims by (intervention, outcome). If, inside one group, some papers report a positive effect
    and other papers report no effect or a negative effect, that is a conflict. Then compare population,
    method and dataset on both sides to explain WHY the results may differ.

Gap rule
    For every intervention build a table: rows = populations, columns = methods. A cell with no claim,
    where the population and the method both appear elsewhere, is an untested combination.
    Gap score = number of claims using that method + number of claims about that population.
"""

import hashlib
from collections import Counter
from typing import Dict, List, Any

from . import vocab

COMPARE_FIELDS = ["population", "method", "dataset"]


def stable_id(prefix: str, key: str) -> str:
    """Same input -> same id, so feedback given earlier still matches after a re-run."""
    return f"{prefix}_{hashlib.md5(key.encode('utf-8')).hexdigest()[:6]}"


def _lt(v: str) -> str:
    return vocab.label_text(v)


def _claim_ref(c: Dict[str, Any]) -> Dict[str, Any]:
    return {"claim_id": c["id"], "paper": c.get("paper_title", c["paper_id"]), "paper_id": c["paper_id"],
            "statement": c["statement"], "quote": c["quote"]}


def find_contradictions(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in claims:
        if c.get("intervention", vocab.OTHER) == vocab.OTHER or c.get("outcome", vocab.OTHER) == vocab.OTHER:
            continue
        groups.setdefault((c["intervention"], c["outcome"]), []).append(c)

    findings = []
    for (interv, outcome), grp in groups.items():
        side_a = [c for c in grp if c.get("direction") == "positive"]
        side_b = [c for c in grp if c.get("direction") in ("none", "negative")]
        if not side_a or not side_b:
            continue
        pairs = [(a, b) for a in side_a for b in side_b if a["paper_id"] != b["paper_id"]]
        if not pairs:
            continue

        differences = {}
        for f in COMPARE_FIELDS:
            va = sorted({c[f] for c in side_a if c.get(f, vocab.OTHER) != vocab.OTHER})
            vb = sorted({c[f] for c in side_b if c.get(f, vocab.OTHER) != vocab.OTHER})
            if va and vb and set(va) != set(vb):
                differences[f] = {"side_a": va, "side_b": vb}

        a0, b0 = pairs[0]
        papers_a = sorted({c["paper_id"] for c in side_a})
        papers_b = sorted({c["paper_id"] for c in side_b})
        shared = [f for f in COMPARE_FIELDS if f not in differences]

        if differences:
            parts = [f"{f} ({', '.join(_lt(x) for x in d['side_a'])} vs {', '.join(_lt(x) for x in d['side_b'])})"
                     for f, d in differences.items()]
            root = ("The papers that report an effect and the papers that report no effect also differ in: "
                    + "; ".join(parts) + ". The conflict may be explained by these differences.")
            resolution = (f"Test {_lt(interv)} under matched conditions: keep "
                          f"{', '.join(shared) if shared else 'the protocol'} the same and vary only "
                          f"{', '.join(differences.keys())}.")
            severity = "Context-Dependent Conflict"
        else:
            root = ("Population, method and dataset look the same on both sides, but the results differ. "
                    "This is a direct conflict. Check sample size, protocol details and measurement.")
            resolution = "Compare the original protocols and data of the conflicting papers."
            severity = "Direct Contradiction"

        findings.append({
            "id": stable_id("CONT", f"{interv}|{outcome}"),
            "type": "contradiction",
            "topic": f"{_lt(outcome).capitalize()} and {_lt(interv)}: effect reported in {', '.join(papers_a)}, "
                     f"no effect or harm in {', '.join(papers_b)}",
            "severity": severity,
            "intervention": interv,
            "outcome": outcome,
            "claim_a": _claim_ref(a0),
            "claim_b": _claim_ref(b0),
            "claim_ids": [c["id"] for c in side_a] + [c["id"] for c in side_b],
            "side_a_claim_ids": [c["id"] for c in side_a],
            "side_b_claim_ids": [c["id"] for c in side_b],
            "papers_a": papers_a,
            "papers_b": papers_b,
            "differences": differences,
            "root_cause": root,
            "resolution_path": resolution,
            "weight": 1.0,
        })
    return findings


def find_gaps(claims: List[Dict[str, Any]], limitations: List[Dict[str, Any]] = None,
              max_limitation_gaps: int = 4) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    by_interv: Dict[str, List[Dict[str, Any]]] = {}
    for c in claims:
        if c.get("intervention", vocab.OTHER) != vocab.OTHER:
            by_interv.setdefault(c["intervention"], []).append(c)

    for interv, grp in by_interv.items():
        usable = [c for c in grp if c.get("population", vocab.OTHER) != vocab.OTHER
                  and c.get("method", vocab.OTHER) != vocab.OTHER]
        pops = sorted({c["population"] for c in usable})
        methods = sorted({c["method"] for c in usable})
        seen = {(c["population"], c["method"]) for c in usable}
        for p in pops:
            for m in methods:
                if (p, m) in seen:
                    continue
                used_m = [c for c in usable if c["method"] == m]
                used_p = [c for c in usable if c["population"] == p]
                score = len(used_m) + len(used_p)
                papers = sorted({c["paper_id"] for c in used_m + used_p})
                titles = []
                for c in used_m + used_p:
                    t = c.get("paper_title", c["paper_id"])
                    if t not in titles:
                        titles.append(t)
                evidence = used_m[0]
                gaps.append({
                    "id": stable_id("GAP", f"{interv}|{p}|{m}"),
                    "type": "gap",
                    "gap_kind": "untested_combination",
                    "title": f"Untested: {_lt(m)} in {_lt(p)}",
                    "category": "Untested Combination",
                    "intervention": interv,
                    "population": p,
                    "method": m,
                    "source_papers": titles,
                    "paper_ids": papers,
                    "description": (f"For {_lt(interv)}, {_lt(m)} is used with "
                                    f"{', '.join(_lt(x) for x in sorted({c['population'] for c in used_m}))}, and "
                                    f"{_lt(p)} are studied only with "
                                    f"{', '.join(_lt(x) for x in sorted({c['method'] for c in used_p})) or 'no other method'}. "
                                    f"No loaded paper combines {_lt(m)} with {_lt(p)}."),
                    "evidence_quote": evidence["quote"],
                    "evidence_claim_id": evidence["id"],
                    "supporting_claim_ids": [c["id"] for c in used_m + used_p],
                    "impact_score": "Critical" if score >= 6 else ("High" if score >= 4 else "Moderate"),
                    "suggested_direction": f"Evaluate {_lt(interv)} in {_lt(p)} using {_lt(m)}.",
                    "score": score,
                    "weight": 1.0,
                })

    # one stated limitation per paper first, so every paper is represented
    chosen, seen_papers = [], set()
    for lim in (limitations or []):
        if lim["paper_id"] not in seen_papers:
            chosen.append(lim)
            seen_papers.add(lim["paper_id"])
    for lim in chosen[:max_limitation_gaps]:
        gaps.append({
            "id": stable_id("GAP", lim["quote"]),
            "type": "gap",
            "gap_kind": "stated_limitation",
            "title": f"Stated limitation in {lim['paper_title'][:45]}",
            "category": "Stated Limitation",
            "source_papers": [lim["paper_title"]],
            "paper_ids": [lim["paper_id"]],
            "description": "The authors themselves state this limitation or missing piece.",
            "evidence_quote": lim["quote"],
            "supporting_claim_ids": [],
            "impact_score": "Moderate",
            "suggested_direction": "Address the limitation in a follow-up study.",
            "score": 1,
            "weight": 1.0,
        })

    gaps.sort(key=lambda g: (-g.get("weight", 1.0), -g.get("score", 0)))
    return gaps
