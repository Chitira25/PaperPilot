"""
Automated tests for PaperPilot. Run:  python scripts/test_pipeline.py
These tests check CORRECTNESS (not only that outputs exist): quotes must be real, the Critic must
reject a made-up hypothesis, the offline engine must not invent findings, feedback must change weights.
"""

import glob
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.pdf_processor import process_paper, paper_id_for, segment_sections
from modules.analyzer import PaperPilotAnalyzer
from modules.textutil import quote_is_real
from modules.verify import final_label, check_custom_hypothesis
from modules.llm_engine import LLMClient
from modules import feedback as fb
from modules.evaluate import evaluate_all
from modules.rules import find_contradictions, find_gaps

FAILS = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def load(folder):
    files = sorted(glob.glob(os.path.join(ROOT, "data", folder, "*.pdf")))
    return [process_paper(f, os.path.basename(f), paper_id_for(os.path.basename(f), i)) for i, f in enumerate(files)]


def run_rules(papers, entries=None):
    return PaperPilotAnalyzer(mode="rules", feedback_entries=entries or []).run_full_analysis(papers)


def all_quotes_real(res, papers):
    full = {p["paper_id"]: p["full_text"] for p in papers}
    bad = [c["id"] for c in res["claims"] if not quote_is_real(c["quote"], full[c["paper_id"]])]
    for g in res["research_gaps"]:
        if not any(quote_is_real(g["evidence_quote"], t) for t in full.values()):
            bad.append(g["id"])
    for v in res["verifications"].values():
        for e in v["supporting_evidence"] + v["contradicting_or_limiting_evidence"]:
            if not quote_is_real(e["quote"], full[e["paper_id"]]):
                bad.append(v["hypothesis_id"])
    return bad


print("=" * 64 + "\nPaperPilot correctness tests\n" + "=" * 64)

# 1. section splitter bug (body lines that start with a heading word must NOT be treated as headings)
txt = ("Abstract\nWe study grip rehabilitation.\nMethods\nParticipants wore sensors.\n"
       "Results were recorded every week for each patient.\nDiscussion of the sensor placement follows in detail.\n"
       "Results\nGrip improved.\nReferences\n[1] Someone et al.")
sec = segment_sections(txt)
check("Section splitter keeps body lines that start with 'Results' / 'Discussion'",
      "Results were recorded every week" in sec.get("Methodology", "") and "Discussion of the sensor" in sec.get("Methodology", ""))
check("Section splitter removes the reference list", "References" not in sec and "Someone" not in " ".join(sec.values()))

# 2. bundled sample PDFs
rag = load("sample_papers")
check("5 bundled RAG PDFs are parsed with 5+ sections each", len(rag) == 5 and all(len(p["sections"]) >= 5 for p in rag))

# 3. quote check
full = rag[0]["full_text"]
real = full.split(".")[3].strip() + "."
check("quote_is_real accepts a real sentence", quote_is_real(real, full))
check("quote_is_real rejects an invented sentence", not quote_is_real("Graph RAG cured every patient in every trial.", full))
check("quote_is_real tolerates hyphenation differences", quote_is_real("single point sensor was used here", "A single-point sensor was used here."))

# 4. rehab demo: planted contradiction + gap
demo = load("rehab_demo_papers")
res = run_rules(demo)
check("Demo: 5 papers, claims found", len(demo) == 5 and len(res["claims"]) >= 10)
check("Demo: every quote (claims, gaps, evidence) exists in the paper text", all_quotes_real(res, demo) == [], str(all_quotes_real(res, demo)))
check("Demo: a conclusion sentence without a result is not turned into a claim",
      not any("needs a better measurement" in c["quote"] for c in res["claims"]))
conts = res["contradictions"]
check("Demo: planted contradiction found", len(conts) >= 1 and all(c["papers_a"] == ["P1", "P2", "P3"] and c["papers_b"] == ["P4", "P5"] for c in conts))
check("Demo: contradiction explained by population AND method", all({"population", "method"} <= set(c["differences"]) for c in conts))
gap_pairs = {(g.get("population"), g.get("method")) for g in res["research_gaps"] if g["gap_kind"] == "untested_combination"}
check("Demo: planted gap found (parkinsons x continuous_sEMG)", ("parkinsons", "continuous_sEMG") in gap_pairs)
labels = {h["finding_type"]: res["verifications"][h["id"]]["verification_label"] for h in res["hypotheses"]}
check("Demo: gap hypothesis is PARTIALLY_VERIFIED (claim ok, population differs)", labels.get("gap") == "PARTIALLY_VERIFIED", str(labels))
check("Demo: scores are not constant", len({v["overall_grounding_score"] for v in res["verifications"].values()}) > 1)

# 5. the Critic rejects made-up hypotheses and accepts supported ones
good = check_custom_hypothesis({"intervention": "grip_rehab_protocol", "outcome": "hand_function", "population": "stroke",
                                "method": "continuous_sEMG", "direction": "positive"}, res["claims"], demo)
fake = check_custom_hypothesis({"intervention": "grip_rehab_protocol", "outcome": "hand_function", "population": "healthy_adults",
                                "method": "single_point_sensor", "direction": "positive"}, res["claims"], demo)
check("Critic: a supported hypothesis is VERIFIED", good["label"] == "VERIFIED", good["label"])
check("Critic: a made-up hypothesis is NOT verified", fake["label"] != "VERIFIED", fake["label"])
nothing = check_custom_hypothesis({"intervention": "graph_rag", "outcome": "latency", "population": "stroke",
                                   "method": "other", "direction": "positive"}, res["claims"], demo)
check("Critic: a hypothesis with no source at all is UNVERIFIED", nothing["label"] == "UNVERIFIED", nothing["label"])
check("final_label rule table", final_label(True, True, True, True) == "VERIFIED" and final_label(True, False, True, True) == "PARTIALLY_VERIFIED"
      and final_label(False, True, True, True) == "UNVERIFIED" and final_label(True, True, True, False) == "UNVERIFIED")

# 6. honesty: no conflict in the data -> no invented conflict
only_stroke = run_rules(demo[:3])
check("Honest output: 3 agreeing papers give NO contradiction", only_stroke["contradictions"] == [])
check("Honest output: no invented hypotheses when nothing is found", only_stroke["hypotheses"] == [])

# 7. bundled RAG papers through the real pipeline + curated example is labelled
rag_res = run_rules(rag)
check("RAG papers: pipeline runs and every quote is real", all_quotes_real(rag_res, rag) == [])
cur = PaperPilotAnalyzer(mode="rules", feedback_entries=[]).run_full_analysis(rag, use_curated=True)
check("Curated example is flagged as curated, quotes all exist in the PDFs",
      cur.get("curated") is True and cur["stats"]["claims_quote_found"] == cur["stats"]["claims_total"])

# 8. feedback loop
ids = {h["id"]: h["finding_id"] for h in res["hypotheses"]}
hid = next(i for i, f in ids.items() if f.startswith("GAP"))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "feedback.json")
    fb.record_feedback(hid, ids[hid], "reject", path=path)
    entries = fb.load_feedback(path)
    w = fb.compute_weights(entries)[ids[hid]]
    check("Feedback: reject lowers the weight to 0.7", w == 0.7, str(w))
    fb.record_feedback(hid, ids[hid], "approve", path=path)
    w2 = fb.compute_weights(fb.load_feedback(path))[ids[hid]]
    check("Feedback: only the latest decision counts (approve -> 1.2, not 0.9)", w2 == 1.2, str(w2))
    again = run_rules(demo, fb.load_feedback(path))
    check("Feedback: finding ids are stable across runs and weight is applied",
          any(g["id"] == ids[hid] and g["weight"] == 1.2 for g in again["research_gaps"]))
    check("Feedback: hypothesis status is restored after re-run",
          next(h for h in again["hypotheses"] if h["id"] == hid)["status"] == "approved")

# 9. evaluation
gt = json.load(open(os.path.join(ROOT, "data", "rehab_demo_papers", "ground_truth.json")))
m = evaluate_all(gt, res, [{"hypothesis_id": "a", "finding_id": "x", "action": "approve"},
                           {"hypothesis_id": "b", "finding_id": "y", "action": "reject"}], 5)
check("Evaluation: coverage / precision / recall on the demo are 100%",
      m["claim_coverage"] == 1.0 and m["contradiction_precision"] == 1.0 and m["gap_recall"] == 1.0, str(m))
check("Evaluation: human agreement = 50% for 1 approve + 1 reject", m["human_agreement"] == 0.5)
check("Evaluation: no ground truth gives None, not a fake number", evaluate_all(None, res, [], 5)["claim_coverage"] is None)

# 10. Live-LLM path with a stub model (no network): invented quotes are dropped, the label is decided by code
class StubClient(LLMClient):
    def __init__(self):
        super().__init__(provider="openai", api_key="x", use_cache=False)
    def generate_completion(self, system_prompt, user_prompt, temperature=0.0):
        if "extract the claims" in system_prompt:
            pid_text = user_prompt
            real_sentence = next(s for s in pid_text.split("\n") if "improved significantly" in s or "did not improve" in s)
            items = [{"statement": "real", "quote": real_sentence.strip(), "section": "Results",
                      "intervention": "grip_rehab_protocol", "outcome": "hand_function",
                      "population": "stroke" if "improved" in real_sentence else "parkinsons",
                      "method": "continuous_sEMG" if "improved" in real_sentence else "single_point_sensor",
                      "dataset": "other", "direction": "positive" if "improved" in real_sentence else "none"},
                     {"statement": "invented", "quote": "The protocol cured all patients within one day of training.",
                      "section": "Results", "intervention": "grip_rehab_protocol", "outcome": "hand_function",
                      "population": "stroke", "method": "continuous_sEMG", "dataset": "other", "direction": "positive"}]
            return json.dumps(items)
        if "Write ONE testable" in system_prompt:
            return json.dumps({"title": "AI title", "formulation": "AI formulation text.", "rationale": "r",
                               "proposed_methodology": "m", "falsification_criteria": "f"})
        return json.dumps({"claim_match": True, "population_match": True, "method_match": True, "notes": "AI says fine."})

llm_res = PaperPilotAnalyzer(mode="llm", llm_client=StubClient(), feedback_entries=[]).run_full_analysis(demo)
check("LLM path: invented quotes are removed by the code check", all_quotes_real(llm_res, demo) == [] and
      all("cured all patients" not in c["quote"] for c in llm_res["claims"]), "")
check("LLM path: warning reports how many claims were dropped", "removed because their quote was not found" in llm_res.get("warning", ""))
gap_v = next(v for h in llm_res["hypotheses"] if h["finding_type"] == "gap" for v in [llm_res["verifications"][h["id"]]])
check("LLM path: AI critic cannot override the code (gap stays PARTIALLY_VERIFIED, not VERIFIED)",
      gap_v["verification_label"] == "PARTIALLY_VERIFIED", gap_v["verification_label"])
check("LLM path: engine label names the model mode", "Live LLM" in llm_res["engine_used"])
unconfigured = PaperPilotAnalyzer(mode="llm", llm_client=LLMClient(provider="gemini", api_key=""), feedback_entries=[]).run_full_analysis(demo)
check("LLM path without a key falls back to the honest rules engine with a warning",
      "Offline rules engine" in unconfigured["engine_used"] and "API key" in unconfigured.get("warning", ""))

# 11. data contract fields
need_f = {"id", "type", "weight"}
check("Contract: findings have id/type/weight", all(need_f <= set(f) for f in res["contradictions"] + res["research_gaps"]))
need_h = {"id", "finding_id", "formulation", "status", "target"}
check("Contract: hypotheses have required fields", all(need_h <= set(h) for h in res["hypotheses"]))
need_v = {"verification_label", "critic", "supporting_evidence", "overall_grounding_score"}
check("Contract: verifications have required fields", all(need_v <= set(v) for v in res["verifications"].values()))
json.dumps(res)  # must be serialisable for export
check("Results are JSON-serialisable", True)

print("=" * 64)
if FAILS:
    print(f"{len(FAILS)} TEST(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
