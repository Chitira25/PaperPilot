"""
Verification: quote check, evidence retrieval, the Critic checks and the final label.

Important design rule: the final label (VERIFIED / PARTIALLY_VERIFIED / UNVERIFIED) is decided by
plain code (final_label), never by an AI model. An AI model may only answer the three true/false
questions (claim, population, method); quotes are always checked by code against the paper text.
"""

import math
from collections import Counter
from typing import Dict, List, Any, Optional

from . import vocab
from .textutil import quote_is_real, split_sentences, tokenize


# ---------------------------------------------------------------- label rule
def final_label(claim_ok: bool, population_ok: bool, method_ok: bool, quotes_ok: bool) -> str:
    if not quotes_ok or not claim_ok:
        return "UNVERIFIED"
    if population_ok and method_ok:
        return "VERIFIED"
    return "PARTIALLY_VERIFIED"


def match_score(claim_ok: bool, population_ok: bool, method_ok: bool, quotes_ok: bool) -> int:
    """0-100: share of the three checks that passed; 0 if a quote is not found in the paper."""
    if not quotes_ok:
        return 0
    return round(100 * (int(claim_ok) + int(population_ok) + int(method_ok)) / 3)


FEASIBILITY_TEXT = {
    "VERIFIED": "Verified",
    "PARTIALLY_VERIFIED": "Partially verified",
    "UNVERIFIED": "Unverified",
}


# ---------------------------------------------------------------- quote check
def papers_by_id(papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {p.get("paper_id", f"P{i + 1}"): p for i, p in enumerate(papers)}


def check_quote(quote: str, paper_id: str, pmap: Dict[str, Dict[str, Any]]) -> bool:
    paper = pmap.get(paper_id)
    return bool(paper) and quote_is_real(quote, paper.get("full_text", ""))


def validate_claim_quotes(claims: List[Dict[str, Any]], papers: List[Dict[str, Any]]):
    """Keep only claims whose quote really exists in their paper. Returns (kept, dropped_count)."""
    pmap = papers_by_id(papers)
    kept, dropped = [], 0
    for c in claims:
        if check_quote(c.get("quote", ""), c.get("paper_id", ""), pmap):
            kept.append(c)
        else:
            dropped += 1
    return kept, dropped


# ---------------------------------------------------------------- retrieval (BM25-lite, no extra packages)
def build_chunks(papers: List[Dict[str, Any]], min_words: int = 50) -> List[Dict[str, Any]]:
    chunks = []
    for i, p in enumerate(papers):
        pid = p.get("paper_id", f"P{i + 1}")
        buf, count = [], 0
        for sent in split_sentences(p.get("full_text", "")):
            buf.append(sent)
            count += len(sent.split())
            if count >= min_words:
                chunks.append({"paper_id": pid, "paper_title": p.get("title", pid), "text": " ".join(buf)})
                buf, count = [], 0
        if buf:
            chunks.append({"paper_id": pid, "paper_title": p.get("title", pid), "text": " ".join(buf)})
    return chunks


def retrieve(query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q = [t for t in tokenize(query) if len(t) > 2]
    k1, b = 1.5, 0.75
    scored = []
    for chunk, d in zip(chunks, docs):
        tf = Counter(d)
        s = 0.0
        for t in set(q):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        if s > 0:
            scored.append((s, chunk))
    scored.sort(key=lambda x: -x[0])
    return [{**c, "score": round(s, 2)} for s, c in scored[:k]]


# ---------------------------------------------------------------- the Critic (rules version)
def best_matching_claim(target: Dict[str, Any], claims: List[Dict[str, Any]]):
    """Find the single claim that matches the hypothesis best.
    Returns (claim or None, claim_ok, population_ok, method_ok).
    The three checks are made on ONE claim so that 'A works in stroke' and 'B was tested in
    Parkinson's' are never mixed into a false 'verified'."""
    best, best_key, best_flags = None, (-1, -1), (False, False, False)
    for c in claims:
        claim_ok = True
        if target.get("intervention") not in (None, vocab.OTHER):
            claim_ok = claim_ok and c.get("intervention") == target["intervention"]
        if target.get("outcome") not in (None, vocab.OTHER):
            claim_ok = claim_ok and c.get("outcome") == target["outcome"]
        if target.get("direction"):
            claim_ok = claim_ok and c.get("direction") == target["direction"]
        pop_ok = target.get("population") in (None, vocab.OTHER) or c.get("population") == target.get("population")
        met_ok = target.get("method") in (None, vocab.OTHER) or c.get("method") == target.get("method")
        key = (int(claim_ok), int(pop_ok) + int(met_ok))
        if key > best_key:
            best, best_key, best_flags = c, key, (claim_ok, pop_ok, met_ok)
    return (best,) + best_flags


def rules_critic(target: Dict[str, Any], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    c, claim_ok, pop_ok, met_ok = best_matching_claim(target, claims)
    notes = []
    if c is None:
        notes.append("No claim in the loaded papers matches this hypothesis.")
    else:
        notes.append(f"Closest source: {c['paper_id']} ({vocab.label_text(c.get('population', 'other'))}, "
                     f"{vocab.label_text(c.get('method', 'other'))}, direction: {c.get('direction')}).")
        if not claim_ok:
            notes.append("The intervention, outcome or direction does not match any source claim.")
        if claim_ok and not pop_ok:
            notes.append("The population in the hypothesis is different from the population in the source.")
        if claim_ok and not met_ok:
            notes.append("The method in the hypothesis is different from the method in the source.")
    return {"claim_match": claim_ok, "population_match": pop_ok, "method_match": met_ok,
            "notes": " ".join(notes), "best_claim_id": c["id"] if c else None}


# ---------------------------------------------------------------- full verification record
def _ev(c: Dict[str, Any], relevance: str) -> Dict[str, Any]:
    return {"paper_title": c.get("paper_title", c["paper_id"]), "paper_id": c["paper_id"],
            "claim_id": c["id"], "quote": c["quote"], "relevance": relevance}


def verify_hypothesis(h: Dict[str, Any], finding: Optional[Dict[str, Any]], claims: List[Dict[str, Any]],
                      papers: List[Dict[str, Any]], chunks: Optional[List[Dict[str, Any]]] = None,
                      critic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the verification record for one hypothesis.
    `critic` may come from an AI model (three booleans + notes). If it is None the rules critic is used."""
    pmap = papers_by_id(papers)
    cmap = {c["id"]: c for c in claims}
    target = h.get("target", {})
    supporting, limiting = [], []

    if h.get("finding_type") == "gap":
        pool = [cmap[i] for i in (finding or {}).get("supporting_claim_ids", []) if i in cmap]
        for c in pool:
            if c.get("method") == target.get("method") and c.get("intervention") == target.get("intervention") \
                    and c.get("direction") == target.get("direction"):
                supporting.append(_ev(c, f"Reports the effect with {vocab.label_text(c['method'])} "
                                         f"in {vocab.label_text(c['population'])}."))
            elif c.get("population") == target.get("population"):
                limiting.append({**_ev(c, ""), "limitation":
                                 f"Same population ({vocab.label_text(c['population'])}) but measured with "
                                 f"{vocab.label_text(c['method'])}; direction: {c.get('direction')}."})
        evidence_claims = pool
    else:
        a = [cmap[i] for i in (finding or {}).get("side_a_claim_ids", []) if i in cmap]
        b = [cmap[i] for i in (finding or {}).get("side_b_claim_ids", []) if i in cmap]
        supporting = [_ev(c, "Reports an effect.") for c in a[:3]]
        limiting = [{**_ev(c, ""), "limitation": "Reports no effect or a negative effect."} for c in b[:3]]
        evidence_claims = a + b

    quotes = [(e["quote"], e["paper_id"]) for e in supporting + limiting]
    quotes_ok = bool(quotes) and all(check_quote(q, pid, pmap) for q, pid in quotes)

    if critic is None:
        if h.get("finding_type") == "gap":
            critic = rules_critic(target, evidence_claims)
        else:
            have_a, have_b = bool(supporting), bool(limiting)
            critic = {"claim_match": have_a and have_b, "population_match": have_a and have_b,
                      "method_match": have_a and have_b,
                      "notes": "The conflict that motivates this hypothesis is found in the sources. "
                               "The explanation itself is still untested."}
    critic = dict(critic)
    critic["quotes_verified"] = quotes_ok

    label = final_label(critic["claim_match"], critic["population_match"], critic["method_match"], quotes_ok)
    score = match_score(critic["claim_match"], critic["population_match"], critic["method_match"], quotes_ok)

    verdicts = {
        "VERIFIED": "All three checks (claim, population, method) match a real source quote.",
        "PARTIALLY_VERIFIED": "The claim is supported by the sources, but the population or the method differs "
                              "from what the hypothesis proposes. That difference is the untested part.",
        "UNVERIFIED": "No real source quote supports this hypothesis.",
    }
    passages = retrieve(f"{h.get('title', '')} {h.get('formulation', '')}", chunks or [], k=3)

    return {
        "hypothesis_id": h["id"],
        "overall_grounding_score": score,
        "feasibility_rating": FEASIBILITY_TEXT[label],
        "verification_label": label,
        "critic": critic,
        "supporting_evidence": supporting,
        "contradicting_or_limiting_evidence": limiting,
        "retrieved_passages": passages,
        "falsification_criteria": h.get("falsification_criteria", "Not specified."),
        "synthesis_verdict": verdicts[label],
    }


def check_custom_hypothesis(target: Dict[str, Any], claims: List[Dict[str, Any]],
                            papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Used by the 'test the Critic' box in the app: check any structured hypothesis against the claims."""
    pmap = papers_by_id(papers)
    critic = rules_critic(target, claims)
    cmap = {c["id"]: c for c in claims}
    best = cmap.get(critic.get("best_claim_id"))
    quotes_ok = bool(best) and check_quote(best["quote"], best["paper_id"], pmap)
    label = final_label(critic["claim_match"], critic["population_match"], critic["method_match"], quotes_ok)
    return {"label": label, "critic": critic, "closest_claim": best, "quotes_verified": quotes_ok,
            "score": match_score(critic["claim_match"], critic["population_match"], critic["method_match"], quotes_ok)}
