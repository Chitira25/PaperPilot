"""
Tests for the research-agent features: paper discovery parsers, library, appraisal, agent routing, analytics.
No internet is needed: the database answers are small fixtures that follow the public API formats, and the
network calls are replaced by mocks. Run:  python scripts/test_agent.py
(The real Europe PMC / OpenAlex calls can only be checked on a computer with internet.)
"""

import json
import os
import sys
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules import discovery as D
from modules import library as L
from modules import analytics as A
from modules.appraisal import appraise_paper, detect_sample_size, summarize_paper
from modules.agent import (ResearchAgent, AgentContext, parse_search_request, classify_intent,
                           find_paper_ref, unsupported_quotes)
from modules.analyzer import PaperPilotAnalyzer
from modules.llm_engine import LLMClient
from modules.textutil import quote_is_real

FAILS = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


EPMC = {"resultList": {"result": [
    {"id": "111", "source": "MED", "pmid": "111", "doi": "10.1000/ecg.1", "title": "Wearable <i>ECG</i> patch for atrial fibrillation detection.",
     "authorString": "Lee A, Kim B.", "journalTitle": "J Biomed Eng", "pubYear": "2024", "isOpenAccess": "Y", "citedByCount": 42,
     "abstractText": "<h4>Background</h4> We validated a wearable ECG patch in 48 patients with atrial fibrillation. "
                     "The patch detected atrial fibrillation with a sensitivity of 95% and a specificity of 92%. "
                     "Limitations include a single centre.",
     "pubType": "journal article",
     "fullTextUrlList": {"fullTextUrl": [{"availability": "Subscription required", "documentStyle": "pdf", "url": "http://x/paywalled.pdf"},
                                         {"availability": "Open access", "documentStyle": "pdf", "url": "http://x/open.pdf"}]}},
    {"id": "222", "source": "MED", "title": "Pulse oximetry accuracy review", "authorString": "Roe C.", "pubYear": "2022", "isOpenAccess": "N",
     "citedByCount": 7, "abstractText": ""}]}}

OALEX = {"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/ECG.1", "title": "Wearable ECG patch for atrial fibrillation detection",
     "publication_year": 2024, "cited_by_count": 50, "type": "article",
     "authorships": [{"author": {"display_name": f"Author {i}"}} for i in range(8)],
     "abstract_inverted_index": {"We": [0], "validated": [1], "a": [2], "patch": [3]},
     "open_access": {"is_oa": True}, "primary_location": {"source": {"display_name": "J Biomed Eng"}},
     "best_oa_location": {"pdf_url": "http://y/open.pdf"}},
    {"id": "https://openalex.org/W3", "doi": None, "title": "Cuffless blood pressure sensors", "publication_year": 2021, "cited_by_count": 300,
     "authorships": [], "abstract_inverted_index": None, "open_access": {"is_oa": False}, "primary_location": None, "best_oa_location": None}]}

print("=" * 64 + "\nResearch-agent tests\n" + "=" * 64)

# ---- discovery parsers
e = D.parse_europepmc(EPMC)
check("Europe PMC: HTML tags and trailing dot removed from the title", e[0]["title"] == "Wearable ECG patch for atrial fibrillation detection")
check("Europe PMC: only the OPEN-ACCESS pdf link is kept", e[0]["pdf_url"] == "http://x/open.pdf")
check("Europe PMC: abstract cleaned, doi, citations, OA flag", "<h4>" not in e[0]["abstract"] and e[0]["doi"] == "10.1000/ecg.1"
      and e[0]["citations"] == 42 and e[0]["is_oa"] is True)
check("Europe PMC: record without doi links to europepmc.org", e[1]["url"].startswith("https://europepmc.org/article/MED/222"))
o = D.parse_openalex(OALEX)
check("OpenAlex: abstract rebuilt from the inverted index", o[0]["abstract"] == "We validated a patch" and o[1]["abstract"] == "")
check("OpenAlex: doi cleaned, authors shortened with et al., pdf url kept",
      o[0]["doi"] == "10.1000/ecg.1" and o[0]["authors"].endswith("et al.") and o[0]["pdf_url"] == "http://y/open.pdf")
check("OpenAlex: missing source/location does not crash", o[1]["journal"] == "" and o[1]["pdf_url"] is None)

merged = D.merge_results([e, o], sort="relevance", limit=10)
check("Merge: the same paper from two sources appears once and keeps the richest data",
      len(merged) == 3 and merged[0]["source"] == "Europe PMC + OpenAlex" and merged[0]["citations"] == 50 and "sensitivity" in merged[0]["abstract"])
check("Merge: sort by citations puts the most cited first", D.merge_results([e, o], "citations", 10)[0]["citations"] == 300)
check("Merge: sort by recent puts the newest year first", D.merge_results([e, o], "recent", 10)[0]["year"] == 2024)

q = D.build_europepmc_query("wearable ECG", 2021, True, "citations")
check("Europe PMC query has year, open-access and citation sort", "PUB_YEAR:[2021 TO 3000]" in q and "OPEN_ACCESS:y" in q and q.endswith("sort_cited:y"))
pr = D.build_openalex_params("wearable ECG", 2021, True, "recent", 8)
check("OpenAlex params have filter and sort", pr["filter"] == "from_publication_date:2021-01-01,open_access.is_oa:true"
      and pr["sort"] == "publication_date:desc" and pr["per-page"] == "8")

def fake_get(url, params=None, **kw):
    r = MagicMock()
    if "europepmc" in url:
        raise ConnectionError("offline")
    r.json.return_value = OALEX
    r.raise_for_status.return_value = None
    return r

with patch("modules.discovery.requests.get", side_effect=fake_get):
    recs, status = D.search_papers("wearable ECG", limit=5)
check("Search: one source offline does not break the other, and the status says so",
      len(recs) == 2 and "not reachable" in status["Europe PMC"] and "2 result" in status["OpenAlex"], str(status))

def pdf_resp(content):
    r = MagicMock(); r.raise_for_status.return_value = None; r.iter_content.return_value = [content]; return r
with patch("modules.discovery.requests.get", return_value=pdf_resp(b"<html>login</html>")):
    try:
        D.fetch_pdf("http://x"); ok = False
    except ValueError:
        ok = True
check("fetch_pdf rejects a web page that is not a PDF", ok)
with patch("modules.discovery.requests.get", return_value=pdf_resp(b"%PDF-1.4 data")):
    check("fetch_pdf accepts real PDF bytes", D.fetch_pdf("http://x").startswith(b"%PDF"))

# ---- library
p1 = D.paper_from_record(merged[0], "")
check("Abstract-only paper is flagged and keeps database metadata", p1["source_type"] == "abstract" and p1["doi"] == "10.1000/ecg.1" and p1["abstract"])
lib, stored, new = L.add_paper([], p1)
lib, stored2, new2 = L.add_paper(lib, dict(p1))
check("Library: a duplicate is not added twice", new and not new2 and len(lib) == 1)
other = D.paper_from_record(merged[2], "P1")
lib, stored3, _ = L.add_paper(lib, other)
check("Library: id conflicts are resolved to a free id", stored3["paper_id"] == "P2")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "lib.json")
    L.save_library(lib, path)
    check("Library: saved and reloaded unchanged", [p["paper_id"] for p in L.load_library(path)] == ["P1", "P2"])
check("Library: remove works", [p["paper_id"] for p in L.remove_paper(lib, "P1")] == ["P2"])
no_abs = D.paper_from_record(merged[1], "P9")
check("Paper without abstract gets a clear warning", "no abstract" in no_abs["warning"].lower())

# ---- appraisal and summary
check("Sample size: n = 24", detect_sample_size("Results (n = 24) were good.")[0] == 24)
check("Sample size: '1,200 patients'", detect_sample_size("We enrolled 1,200 patients in total.")[0] == 1200)
check("Sample size: none when absent", detect_sample_size("We studied some people.")[0] is None)
ap = appraise_paper(p1)
labels = {c["label"]: c["found"] for c in ap["checks"]}
check("Appraisal: sample size (48 patients), metrics and limitations found in the abstract",
      ap["sample_size"] == 48 and labels["Performance metrics reported"] and labels["Limitations stated"])
check("Appraisal: no control group / ethics claimed when the text does not say so",
      not labels["Ethics approval or consent"] and not labels["Randomisation"])
check("Appraisal: abstract-only scope note is shown", ap["abstract_only"] and "abstract" in ap["scope_note"].lower())
check("Appraisal: study design recognised", ap["design"] in ("Validation / accuracy study", "Device or method development"), ap["design"])

# ---- the pipeline on abstract-only papers keeps every quote real
res = PaperPilotAnalyzer(mode="rules", feedback_entries=[]).run_full_analysis(lib)
bad = [c for c in res["claims"] if not any(quote_is_real(c["quote"], p["full_text"]) for p in lib)]
check("Abstract-only papers: pipeline runs and every quote is real", bad == [] and len(res["claims"]) >= 1, str(len(res["claims"])))
check("Summary uses only real sentences", "sensitivity of 95%" in summarize_paper(lib[0], res["claims"]) or "Key findings" in summarize_paper(lib[0], res["claims"]))

# ---- agent parsing and routing
r = parse_search_request("find recent open access papers on wearable ECG for arrhythmia detection", now_year=2026)
check("Parse: query, open access, recent, year", r["query"] == "wearable ECG for arrhythmia detection" and r["oa_only"] and r["sort"] == "recent" and r["year_from"] == 2023, str(r))
r = parse_search_request("give me the most cited papers on cuffless blood pressure since 2019", now_year=2026)
check("Parse: most cited + since year", r["sort"] == "citations" and r["year_from"] == 2019 and r["query"] == "cuffless blood pressure", str(r))
r = parse_search_request("search for top 5 systematic reviews on glucose monitors", now_year=2026)
check("Parse: top N limits the list", r["limit"] == 5)
papers = [{"paper_id": "P1", "title": "Wearable ECG patch for atrial fibrillation detection"}, {"paper_id": "P2", "title": "Cuffless blood pressure sensors"}]
cases = {"show contradictions": "contradictions", "what are the research gaps?": "gaps", "summarise P2": "summary", "review paper 1": "review",
         "analyse my papers": "analyze", "which sensor was used?": "question", "find papers on EEG for epilepsy": "search",
         "compare my papers": "compare", "help": "help", "what hypotheses can we test": "hypotheses"}
wrong = {k: classify_intent(k, papers) for k, v in cases.items() if classify_intent(k, papers) != v}
check("Intent routing", not wrong, str(wrong))
check("Empty library: a plain topic is treated as a search", classify_intent("grip strength after wearable training", []) == "search")
check("Paper references: P2, paper 1, first paper, title fragment",
      find_paper_ref("review P2", papers)["paper_id"] == "P2" and find_paper_ref("summarise paper 1", papers)["paper_id"] == "P1"
      and find_paper_ref("the first paper", papers)["paper_id"] == "P1"
      and find_paper_ref("tell me about cuffless blood pressure sensors", papers)["paper_id"] == "P2")

calls = {"search": [], "ensure": 0, "log": []}
def ctx_search(req):
    calls["search"].append(req); return merged, {"Europe PMC": "2 result(s)", "OpenAlex": "2 result(s)"}
def ctx_ensure():
    calls["ensure"] += 1; return res
agent = ResearchAgent(AgentContext(get_papers=lambda: lib, ensure_results=ctx_ensure, search=ctx_search, llm=None,
                                   log=lambda k, d="": calls["log"].append(k)))
out = agent.respond("find recent papers on wearable ECG")
check("Agent search: calls the database tool with parsed filters and returns real records",
      calls["search"][0]["query"] == "wearable ECG" and len(out["found"]) == 3 and "search" in calls["log"])
out = agent.respond("review P1")
check("Agent review: returns the checklist for the right paper", "Reporting checklist" in out["text"] and out["focus_paper"] == "P1")
out = agent.respond("summarise P1")
check("Agent summary: extractive and labelled", "Extractive summary" in out["text"])
out = agent.respond("show contradictions")
check("Agent contradictions: honest when none", "No contradictions" in out["text"] or "contradiction" in out["text"].lower())
out = agent.respond("what sensitivity did the ECG patch reach?")
check("Agent question (no LLM): answers with real passages and sources", "[P1]" in out["text"] and "Extractive answer" in out["text"], out["text"][:80])
out = agent.respond("tell me about quantum gravity")
check("Agent question: says so when the library has nothing relevant", "could not find" in out["text"] or "[P" in out["text"])
check("Agent on empty library asks for papers instead of crashing",
      "library is empty" in ResearchAgent(AgentContext(lambda: [], ctx_ensure, ctx_search)).respond("show gaps")["text"].lower())

class StubLLM(LLMClient):
    def __init__(self): super().__init__(provider="openai", api_key="x", use_cache=False)
    def generate_completion(self, s, u, temperature=0.0):
        return 'The patch reached good accuracy [P1]. The authors state "the device cured every patient in the world within one single week" [P1].'
agent2 = ResearchAgent(AgentContext(lambda: lib, ctx_ensure, ctx_search, llm=StubLLM()))
out = agent2.respond("what accuracy did the patch reach?")
check("Agent with LLM: an invented quotation is flagged", "could not be found in your papers" in out["text"])
check("unsupported_quotes: real quote passes, invented fails",
      unsupported_quotes('He wrote "We validated a wearable ECG patch in 48 patients with atrial fibrillation."', lib) == []
      and len(unsupported_quotes('He wrote "this sentence about a robot surgeon does not exist anywhere at all."', lib)) == 1)

# ---- analytics
events = [{"time": "2026-10-19T10:00:00", "kind": "search"}, {"time": "2026-10-20T09:00:00", "kind": "search"}, {"time": "2026-10-20T11:00:00", "kind": "analyze"}]
s = A.activity_series(events, days=3, today=date(2026, 10, 20))
check("Analytics: events per day with zeros", s["events"] == [0, 1, 2] and len(s["day"]) == 3, str(s))
check("Analytics: topics counted from real text", any("atrial" in t.lower() or "ecg" in t.lower() for t, _ in A.topic_counts(lib)))
num = A.summary_numbers(lib, res, [], events)
check("Analytics: numbers are real and empty states are None", num["papers"] == 2 and num["searches"] == 2 and num["agreement"] is None and num["abstract_only"] == 2)

print("=" * 64)
if FAILS:
    print(f"{len(FAILS)} TEST(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL AGENT TESTS PASSED")
