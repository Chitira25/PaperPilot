"""
Orchestration for PaperPilot.

Pipeline
    1. Claims         rules engine (offline)  OR  AI extraction (Live LLM mode, quotes checked by code)
    2. Quote check    every claim quote must exist in its paper, otherwise the claim is dropped
    3. Findings       contradictions + gaps found by repeatable rules (rules.py)
    4. Feedback       researcher decisions change finding weights (feedback.py)
    5. Hypotheses     one per finding (rules, or AI Generator in Live LLM mode)
    6. Verification   Critic checks + label decided by code (verify.py)
"""

from typing import Dict, List, Any, Optional

from . import feedback as fb
from . import vocab
from .demo_data import PRECOMPUTED_PAPERS_ANALYSIS
from .extraction import extract_claims_rules, extract_limitation_sentences
from .hypothesis import generate_hypotheses_rules
from .llm_engine import (LLMClient, extract_claims_with_llm, generate_hypothesis_with_llm, critic_with_llm)
from .rules import find_contradictions, find_gaps
from .textutil import quote_is_real
from .verify import (build_chunks, retrieve, verify_hypothesis, validate_claim_quotes)

STATUS_FROM_ACTION = {"approve": "approved", "edit": "edited", "reject": "rejected"}


class PaperPilotAnalyzer:
    def __init__(self, mode: str = "rules", llm_client: Optional[LLMClient] = None,
                 feedback_entries: Optional[List[Dict[str, Any]]] = None):
        mode = mode.lower()
        self.mode = "rules" if mode in ("demo", "rules", "offline") else "llm"
        self.llm_client = llm_client or LLMClient(provider="demo")
        self.feedback_entries = feedback_entries if feedback_entries is not None else fb.load_feedback()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def is_sample_corpus(papers: List[Dict[str, Any]]) -> bool:
        markers = ["Dense Retrieval vs", "Scaling Vector Chunk", "Self-Consistency", "Adaptive Hybrid Retrieval", "Long-Context LLMs"]
        hits = sum(1 for p in papers if any(m.lower() in p.get("title", "").lower() for m in markers))
        return hits >= 2

    def _curated(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = PRECOMPUTED_PAPERS_ANALYSIS
        full = " ".join(p.get("full_text", "") for p in papers)
        ok = sum(1 for c in data["claims"] if quote_is_real(c["quote"], full))
        return {
            "claims": data["claims"], "contradictions": data["contradictions"],
            "research_gaps": data["research_gaps"], "hypotheses": data["hypotheses"],
            "verifications": data["verifications"],
            "engine_used": "Hand-written curated example (NOT computed by the pipeline; scores are illustrative)",
            "curated": True,
            "stats": {"claims_quote_found": ok, "claims_total": len(data["claims"])},
        }

    # ------------------------------------------------------------------ main entry
    def run_full_analysis(self, papers: List[Dict[str, Any]], use_curated: bool = False) -> Dict[str, Any]:
        if not papers:
            return {"claims": [], "contradictions": [], "research_gaps": [], "hypotheses": [],
                    "verifications": {}, "status": "No papers provided."}
        if use_curated and self.is_sample_corpus(papers):
            return self._curated(papers)

        warnings: List[str] = []
        for p in papers:
            if p.get("warning"):
                warnings.append(f"{p.get('paper_id')}: {p['warning']}")

        # 1. claims
        engine = "Offline rules engine (no AI model, repeatable results)"
        claims: List[Dict[str, Any]] = []
        use_llm = self.mode == "llm"
        if use_llm and not self.llm_client.is_configured():
            warnings.append("Live LLM mode needs an API key (or Ollama). The offline rules engine was used instead.")
            use_llm = False
        if use_llm:
            try:
                claims, notes = extract_claims_with_llm(self.llm_client, papers)
                warnings.extend(notes)
                engine = f"Live LLM ({self.llm_client.provider} / {self.llm_client.model or 'default'}) + code checks"
            except Exception as err:      # network, key, rate limit ...
                warnings.append(f"The AI model failed ({str(err)[:120]}). The offline rules engine was used instead.")
                use_llm, claims = False, []
        if not use_llm:
            claims = extract_claims_rules(papers)

        # 2. quote check
        claims, dropped = validate_claim_quotes(claims, papers)
        if dropped:
            warnings.append(f"{dropped} claim(s) were removed because their quote was not found in the paper text.")

        # 3. findings
        limitations = extract_limitation_sentences(papers)
        contradictions = find_contradictions(claims)
        gaps = find_gaps(claims, limitations)

        # 4. feedback weights
        contradictions = fb.apply_feedback(contradictions, self.feedback_entries)
        gaps = fb.apply_feedback(gaps, self.feedback_entries)

        # 5 + 6. hypotheses and verification
        hypotheses = generate_hypotheses_rules(contradictions, gaps, claims)
        findings = {f["id"]: f for f in contradictions + gaps}
        chunks = build_chunks(papers)
        verifications: Dict[str, Any] = {}
        latest = fb.latest_by_hypothesis(self.feedback_entries)

        for i, h in enumerate(hypotheses):
            finding = findings.get(h["finding_id"])
            h["retries"] = 0
            if use_llm and finding is not None:
                try:
                    h, v = self._llm_hypothesis(h, finding, claims, papers, chunks)
                    hypotheses[i] = h
                except Exception as err:
                    warnings.append(f"AI Generator/Critic failed for {h['id']} ({str(err)[:80]}); rules version kept.")
                    v = verify_hypothesis(h, finding, claims, papers, chunks)
            else:
                v = verify_hypothesis(h, finding, claims, papers, chunks)
            h["finding_weight"] = (finding or {}).get("weight", 1.0)
            entry = latest.get(h["id"])
            if entry:
                h["status"] = STATUS_FROM_ACTION[entry["action"]]
                if entry.get("edited_text"):
                    h["edited_text"] = entry["edited_text"]
            verifications[h["id"]] = v

        hypotheses.sort(key=lambda h: -h.get("finding_weight", 1.0))
        result = {
            "claims": claims, "contradictions": contradictions, "research_gaps": gaps,
            "hypotheses": hypotheses, "verifications": verifications,
            "engine_used": engine, "curated": False,
            "stats": {"claims_dropped_unverified_quote": dropped, "papers": len(papers)},
        }
        if warnings:
            result["warning"] = " | ".join(warnings)
        return result

    def _llm_hypothesis(self, h, finding, claims, papers, chunks):
        v0 = verify_hypothesis(h, finding, claims, papers, chunks)           # rules critic + evidence list
        evidence = v0["supporting_evidence"] + v0["contradicting_or_limiting_evidence"]
        passages = v0["retrieved_passages"]
        notes = ""
        for attempt in range(2):                                               # one retry only
            h = generate_hypothesis_with_llm(self.llm_client, h, evidence, passages, notes)
            ai = critic_with_llm(self.llm_client, h, evidence)
            rules = v0["critic"]
            combined = {                                                       # both critics must agree
                "claim_match": ai["claim_match"] and rules["claim_match"],
                "population_match": ai["population_match"] and rules["population_match"],
                "method_match": ai["method_match"] and rules["method_match"],
                "notes": (ai["notes"] + " " + rules["notes"]).strip(),
            }
            v = verify_hypothesis(h, finding, claims, papers, chunks, critic=combined)
            h["retries"] = attempt
            if v["verification_label"] != "UNVERIFIED":
                break
            notes = combined["notes"]
        return h, v

    # ------------------------------------------------------------------ report
    @staticmethod
    def generate_markdown_report(papers: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> str:
        L = ["# PaperPilot: Synthesis Report\n",
             f"**Engine used:** {analysis_results.get('engine_used', 'PaperPilot')}\n",
             f"**Papers analysed:** {len(papers)}\n"]
        L.append("\n## 1. Corpus overview\n")
        for i, p in enumerate(papers):
            L.append(f"### {i + 1}. {p.get('title', 'Paper')}  (`{p.get('paper_id')}`)")
            L.append(f"- **Authors:** {p.get('authors', 'N/A')} ({p.get('year', 'N/A')})")
            L.append(f"- **Words:** {p.get('word_count', 0):,} | **Pages:** {p.get('page_count', 1)}")
            L.append(f"- **Abstract:** {p.get('abstract', '')[:250]}...\n")

        L.append("\n## 2. Extracted claims\n")
        for c in analysis_results.get("claims", []):
            tags = ""
            if c.get("population"):
                tags = (f" `{c.get('intervention')}` / `{c.get('outcome')}` / `{c.get('population')}` / "
                        f"`{c.get('method')}` / direction: `{c.get('direction')}`")
            L.append(f"- **[{c.get('paper_id')}] ({c.get('topic')})**: {c.get('statement')}{tags}")
            L.append(f"  > *\"{c.get('quote')}\"*\n")

        L.append("\n## 3. Contradictions\n")
        for cont in analysis_results.get("contradictions", []):
            L.append(f"### [{cont.get('severity', 'Tension')}] {cont.get('topic')}")
            for tag, key in (("A", "claim_a"), ("B", "claim_b")):
                c = cont.get(key, {})
                L.append(f"- **Claim {tag} ({c.get('paper')}):** {c.get('statement')}")
                L.append(f"  > *\"{c.get('quote')}\"*")
            L.append(f"- **Why they may differ:** {cont.get('root_cause', '')}")
            L.append(f"- **Resolution path:** {cont.get('resolution_path', '')}\n")
        if not analysis_results.get("contradictions"):
            L.append("No contradictions were found by the rules.\n")

        L.append("\n## 4. Research gaps\n")
        for g in analysis_results.get("research_gaps", []):
            L.append(f"### {g.get('title')} `[{g.get('impact_score', '')} impact]`")
            L.append(f"- **Category:** {g.get('category')}")
            L.append(f"- **Description:** {g.get('description')}")
            L.append(f"  > *Evidence: \"{g.get('evidence_quote')}\"*")
            L.append(f"- **Suggested direction:** {g.get('suggested_direction')}\n")
        if not analysis_results.get("research_gaps"):
            L.append("No gaps were found by the rules.\n")

        L.append("\n## 5. Hypotheses and verification\n")
        vers = analysis_results.get("verifications", {})
        for h in analysis_results.get("hypotheses", []):
            v = vers.get(h.get("id"), {})
            L.append(f"### {h.get('id')}: {h.get('title')}")
            L.append(f"**Formulation:** {h.get('edited_text') or h.get('formulation')}\n")
            label = v.get("verification_label", "n/a (curated example)")
            L.append(f"- **Verification:** **{label}** (match score {v.get('overall_grounding_score', 'N/A')}%)")
            if v.get("critic"):
                c = v["critic"]
                L.append(f"- **Critic checks:** claim {c.get('claim_match')}, population {c.get('population_match')}, "
                         f"method {c.get('method_match')}, quotes found in paper {c.get('quotes_verified')}")
            L.append(f"- **Researcher decision:** {h.get('status', 'pending')}")
            L.append(f"- **Rationale:** {h.get('rationale')}")
            L.append(f"- **Proposed experiment:** {h.get('proposed_methodology')}")
            for s in v.get("supporting_evidence", []):
                L.append(f"  - Supporting: *[{s.get('paper_title')}]: \"{s.get('quote')}\"*")
            for ct in v.get("contradicting_or_limiting_evidence", []):
                L.append(f"  - Limiting: *[{ct.get('paper_title')}]: \"{ct.get('quote')}\"* ({ct.get('limitation')})")
            L.append(f"- **Falsification criteria:** {v.get('falsification_criteria', h.get('falsification_criteria', 'N/A'))}")
            L.append(f"- **Verdict:** {v.get('synthesis_verdict', 'N/A')}\n")
        return "\n".join(L)
