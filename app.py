"""
PaperPilot v2: AI chat research assistant for medical instrumentation and health research.

Views:  Assistant (chat) | Discover (search) | Library (paper cards) | Analysis | Paper review | Graph | Quality & export
Right:  Analytics panel (real numbers from your own library and review decisions)

Everything the assistant shows comes from the paper databases or from your own papers.
Quotes are checked by code, labels are decided by code, and the researcher decides what is approved.
"""

import os
import glob
import json
import html as _html
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from modules import vocab
from modules import feedback as fb
from modules import library as lib
from modules import discovery as disc
from modules import analytics as an
from modules.agent import ResearchAgent, AgentContext, HELP_TEXT
from modules.appraisal import appraise_paper, appraisal_markdown, summarize_paper
from modules.pdf_processor import process_paper, paper_id_for
from modules.analyzer import PaperPilotAnalyzer
from modules.llm_engine import LLMClient
from modules.verify import check_custom_hypothesis
from modules.evaluate import evaluate_all
from modules.knowledge_graph import build_knowledge_graph_data, generate_interactive_graph_html

st.set_page_config(page_title="PaperPilot | AI research assistant", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")

VIEWS = ["💬 Assistant", "🔎 Discover", "📚 Library", "🧪 Analysis", "🔬 Paper review", "🕸️ Graph", "📈 Quality & export"]
LABEL_COLORS = {"VERIFIED": "#059669", "PARTIALLY_VERIFIED": "#D97706", "UNVERIFIED": "#DC2626"}
LABEL_TEXT = {"VERIFIED": "VERIFIED", "PARTIALLY_VERIFIED": "PARTIALLY VERIFIED", "UNVERIFIED": "UNVERIFIED"}
EXAMPLE_PROMPTS = [
    "Find recent open-access papers on wearable ECG for arrhythmia detection",
    "Most cited papers on cuffless blood pressure measurement",
    "Find papers on surface EMG biofeedback for stroke rehabilitation",
    "Find systematic reviews on continuous glucose monitoring accuracy",
]

# =============================================================
# Styling (light theme, cards, like the PaperPilot mockup)
# =============================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #E8F1FF 0%, #F6FAFF 38%, #FFFFFF 100%); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3ECF8; }

.pp-hero { text-align: center; margin: 0 0 0.6rem 0; }
.pp-hero h1 { font-size: 2.7rem; font-weight: 800; color: #0B1B3B; margin: 0; letter-spacing: -1px; }
.pp-hero p { color: #3B4A66; font-size: 1.1rem; margin: 0.1rem 0 0 0; }
.pp-logo { display:inline-flex; width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg,#2563EB,#06B6D4);
           color:#fff; font-weight:800; font-size:1.5rem; align-items:center; justify-content:center; vertical-align:middle; margin-right:10px; }

.pp-section { font-weight: 800; letter-spacing: 1.5px; color: #0B1B3B; font-size: 0.95rem; margin: 0.2rem 0 0.5rem 0; }
.pp-title { font-weight: 700; color: #0B1B3B; font-size: 1.02rem; line-height: 1.3; margin: 4px 0; }
.pp-meta { color: #64748B; font-size: 0.8rem; margin: 0 0 6px 0; }
.pp-abs { color: #334155; font-size: 0.84rem; line-height: 1.45; margin: 4px 0 8px 0; }
.pp-badge { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; margin: 0 5px 4px 0; }
.pp-b-blue { background:#DBEAFE; color:#1D4ED8; } .pp-b-green { background:#D1FAE5; color:#047857; }
.pp-b-amber { background:#FEF3C7; color:#B45309; } .pp-b-red { background:#FEE2E2; color:#B91C1C; }
.pp-b-grey { background:#E2E8F0; color:#334155; } .pp-b-purple { background:#EDE9FE; color:#6D28D9; }
.pp-quote { background:#F1F6FF; border-left:4px solid #2563EB; padding:8px 12px; border-radius:6px; color:#1E293B; font-size:0.86rem; margin:6px 0; }
.pp-quote-red { background:#FFF1F2; border-left:4px solid #E11D48; padding:8px 12px; border-radius:6px; color:#1E293B; font-size:0.86rem; margin:6px 0; }
.pp-note { color:#64748B; font-size:0.78rem; }

div[data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border-radius:16px; border:1px solid #E3ECF8 !important;
        box-shadow: 0 4px 18px rgba(30,64,175,0.07); }
div[data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid #E3ECF8; border-radius:16px; box-shadow: 0 2px 10px rgba(30,64,175,0.05); }
div[role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
div[role="radiogroup"] > label { background:#FFFFFF; border:1px solid #D7E3F7; border-radius:999px; padding:4px 14px; }
div[role="radiogroup"] > label:has(input:checked) { background:#2563EB; border-color:#2563EB; }
div[role="radiogroup"] > label:has(input:checked) p { color:#FFFFFF !important; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; border:1px solid #CBD9F2; font-weight:600; width:100%; }
a[data-testid^="stBaseLinkButton"] { border-radius:10px; width:100%; justify-content:center; }
.stButton > button[kind="primary"] { background:#2563EB; border-color:#2563EB; }

.pp-stat { text-align:left; padding: 2px 0; }
.pp-stat-num { font-size: 1.9rem; font-weight: 800; color:#0B1B3B; line-height:1.1; }
.pp-stat-lbl { color:#475569; font-size:0.85rem; }
.pp-donuts { display:flex; gap:18px; justify-content:space-around; }
.pp-donut-wrap { text-align:center; }
.pp-donut { width:84px; height:84px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto; }
.pp-donut-in { width:60px; height:60px; border-radius:50%; background:#fff; display:flex; align-items:center; justify-content:center;
               font-weight:800; color:#0B1B3B; font-size:0.95rem; }
.pp-donut-lbl { color:#475569; font-size:0.76rem; margin-top:4px; max-width:110px; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Helpers
# =============================================================
def esc(x):
    return _html.escape(str(x if x is not None else ""))


def notify(kind, text):
    st.session_state.notice = (kind, text)


def persist():
    lib.save_library(st.session_state.papers)


def log(kind, detail=""):
    try:
        lib.log_event(kind, detail)
    except OSError:
        pass


def make_llm():
    live = st.session_state.get("engine_mode", "").startswith("Live")
    if not live:
        return None
    return LLMClient(provider=st.session_state.get("llm_provider", "gemini"), api_key=st.session_state.get("llm_key", ""),
                     model=st.session_state.get("llm_model", ""), custom_base_url=st.session_state.get("llm_url", ""))


def analysis_signature():
    papers = st.session_state.papers
    return (tuple((p.get("paper_id"), len(p.get("full_text", ""))) for p in papers),
            st.session_state.get("engine_mode"), bool(st.session_state.get("use_curated")))


def run_analysis():
    entries = fb.load_feedback(FEEDBACK_PATH)
    live = st.session_state.get("engine_mode", "").startswith("Live")
    client = make_llm() or LLMClient(provider="demo")
    analyzer = PaperPilotAnalyzer(mode="llm" if live else "rules", llm_client=client, feedback_entries=entries)
    with st.spinner("Analysing papers..."):
        st.session_state.analysis_results = analyzer.run_full_analysis(
            st.session_state.papers, use_curated=bool(st.session_state.get("use_curated")))
    st.session_state.analysis_sig = analysis_signature()
    log("analyze", f"{len(st.session_state.papers)} papers")


def ensure_results():
    """Results for the current library. The offline engine re-runs automatically when the library changed."""
    live = st.session_state.get("engine_mode", "").startswith("Live")
    stale = st.session_state.analysis_results is None or st.session_state.get("analysis_sig") != analysis_signature()
    if st.session_state.papers and (stale or st.session_state.refresh) and (not live or st.session_state.analysis_results is None or st.session_state.refresh):
        run_analysis()
        st.session_state.refresh = False
    return st.session_state.analysis_results or {"claims": [], "contradictions": [], "research_gaps": [], "hypotheses": [], "verifications": {}}


def do_search(req):
    """Search the paper databases and remember the records so that the 'Add' buttons can use them."""
    sources = tuple(st.session_state.get("sources") or ("europepmc", "openalex")) or ("europepmc", "openalex")
    records, status = disc.search_papers(req["query"], req.get("year_from"), req.get("oa_only", False),
                                         req.get("sort", "relevance"), req.get("limit", 10), sources)
    for r in records:
        st.session_state.found[r["rid"]] = r
    return records, status


def in_library(rec):
    return lib.find_existing(st.session_state.papers, rec) is not None


# ---- callbacks (run before the next script run) ----
def cb_add_record(rid, with_pdf):
    rec = st.session_state.found.get(rid)
    if not rec:
        return
    pdf_bytes, extra = None, ""
    if with_pdf and rec.get("pdf_url"):
        try:
            pdf_bytes = disc.fetch_pdf(rec["pdf_url"])
        except Exception as err:
            extra = f" The full text could not be downloaded ({str(err)[:70]}), so only the abstract was added."
    paper = disc.paper_from_record(rec, "", pdf_bytes)
    st.session_state.papers, stored, new = lib.add_paper(st.session_state.papers, paper)
    persist()
    log("add", rec.get("title", "")[:80])
    kind = "abstract only" if stored.get("source_type") == "abstract" else "full text"
    if with_pdf and pdf_bytes and stored.get("source_type") == "abstract" and not extra:
        extra = " The downloaded PDF had too little readable text (it may be a scan), so only the abstract was added."
    notify("success" if new and not extra else "warning",
           (f"Added **{stored['paper_id']}** to your library ({kind})." if new else f"{stored['paper_id']} is already in your library.") + extra)


def cb_remove(pid):
    st.session_state.papers = lib.remove_paper(st.session_state.papers, pid)
    persist()
    notify("info", f"Removed {pid} from the library.")


def cb_open_review(pid):
    st.session_state.selected_paper = pid
    st.session_state.view = "🔬 Paper review"


def cb_chip(text):
    st.session_state.pending_input = text


def cb_feedback(hyp_id, finding_id, action, text_key=None):
    edited = st.session_state.get(text_key) if text_key else None
    fb.record_feedback(hyp_id, finding_id, action, edited_text=edited,
                       reviewer=st.session_state.get("reviewer", "researcher") or "researcher", path=FEEDBACK_PATH)
    log("review", f"{action} {hyp_id}")
    st.session_state.refresh = True


def cb_clear_library():
    st.session_state.papers, st.session_state.analysis_results, st.session_state.ground_truth = [], None, None
    persist()
    notify("info", "The library is empty now.")


def import_pdf_paths(paths):
    added = 0
    for path in paths:
        name = os.path.basename(path)
        try:
            paper = process_paper(path, name, paper_id_for(name, len(st.session_state.papers)))
            paper["source_type"] = "fulltext"
            st.session_state.papers, _, new = lib.add_paper(st.session_state.papers, paper)
            added += int(new)
        except Exception as err:
            st.sidebar.error(f"Could not read {name}: {err}")
    persist()
    return added


# =============================================================
# Session state
# =============================================================
defaults = {"papers": None, "analysis_results": None, "analysis_sig": None, "ground_truth": None, "found": {},
            "chat": [], "discover_ids": [], "discover_status": {}, "selected_paper": None, "notice": None,
            "refresh": False, "view": VIEWS[0], "engine_mode": "Offline rules engine (no API key)"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.papers is None:
    st.session_state.papers = lib.load_library()
if not st.session_state.chat:
    st.session_state.chat = [{"role": "assistant", "found": [], "text": HELP_TEXT}]


# =============================================================
# Sidebar: settings and import
# =============================================================
with st.sidebar:
    st.markdown('<div class="pp-section">SETTINGS</div>', unsafe_allow_html=True)
    st.radio("Engine", ["Offline rules engine (no API key)", "Live LLM (API key)"], key="engine_mode",
             help="Offline: repeatable, never invents text. Live LLM: written answers, AI Generator and Critic; "
                  "quotes and labels are still checked by code.")
    if st.session_state.engine_mode.startswith("Live"):
        st.selectbox("Provider", ["gemini", "openai", "groq", "ollama", "custom"], key="llm_provider")
        st.text_input("API key", type="password", key="llm_key", help="Kept only in this browser session. Not needed for Ollama.")
        st.text_input("Model (optional)", key="llm_model")
        if st.session_state.llm_provider in ("ollama", "custom"):
            st.text_input("Base URL", key="llm_url", placeholder="http://localhost:11434")
    st.multiselect("Search sources", ["europepmc", "openalex"], default=["europepmc", "openalex"], key="sources",
                   format_func=lambda s: {"europepmc": "Europe PMC (biomedical)", "openalex": "OpenAlex (broad, incl. engineering)"}[s])

    st.markdown('<div class="pp-section" style="margin-top:14px;">ADD PAPERS</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("➕ Add uploaded PDFs"):
        paths = []
        for up in uploads:
            tmp = os.path.join(tempfile.gettempdir(), up.name)
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
            paths.append(tmp)
        n = import_pdf_paths(paths)
        notify("success", f"Added {n} PDF(s) to the library.")
    if st.button("📁 Import PDFs from data/papers/"):
        n = import_pdf_paths(sorted(glob.glob(os.path.join(DATA_DIR, "papers", "*.pdf"))))
        st.session_state.ground_truth = None
        notify("success" if n else "warning", f"Added {n} PDF(s)." if n else "No new PDFs found in data/papers/.")
    if st.button("🧪 Load rehab demo (synthetic)"):
        if st.session_state.papers:
            notify("warning", "Clear the library first: the demo answer key matches the ids P1 to P5.")
        else:
            import_pdf_paths(sorted(glob.glob(os.path.join(DATA_DIR, "rehab_demo_papers", "*.pdf"))))
            try:
                with open(os.path.join(DATA_DIR, "rehab_demo_papers", "ground_truth.json"), encoding="utf-8") as fh:
                    st.session_state.ground_truth = json.load(fh)
            except (OSError, ValueError):
                pass
            notify("success", "Added 5 synthetic rehab papers (fictional, with a planted contradiction and gap).")
    if st.button("📚 Add bundled RAG samples"):
        import_pdf_paths(sorted(glob.glob(os.path.join(DATA_DIR, "sample_papers", "*.pdf"))))
        notify("success", "Added the bundled RAG sample papers.")
    st.checkbox("Show hand-written curated example (bundled RAG papers only)", key="use_curated")
    st.text_input("Reviewer name", value="researcher", key="reviewer")
    st.button("🗑️ Clear library", on_click=cb_clear_library)
    st.caption("Tip: name your PDFs P1_..., P2_... to keep paper ids stable.")


# refresh analysis after a review decision
if st.session_state.refresh and st.session_state.papers:
    ensure_results()

papers = st.session_state.papers
live_mode = st.session_state.engine_mode.startswith("Live")


# =============================================================
# Header + navigation + layout
# =============================================================
st.markdown('<div class="pp-hero"><h1><span class="pp-logo">P</span>PaperPilot</h1>'
            '<p>AI chat research assistant for medical instrumentation &amp; health research</p></div>', unsafe_allow_html=True)

main_col, side_col = st.columns([2.4, 1], gap="large")


# =============================================================
# Reusable renderers
# =============================================================
def render_found_cards(rids, prefix):
    for rid in rids:
        rec = st.session_state.found.get(rid)
        if not rec:
            continue
        inlib = in_library(rec)
        badges = (f'<span class="pp-badge pp-b-blue">{esc(rec.get("source"))}</span>'
                  f'<span class="pp-badge pp-b-grey">{rec.get("citations", 0)} citations</span>')
        if rec.get("is_oa"):
            badges += '<span class="pp-badge pp-b-green">Open access</span>'
        if inlib:
            badges += '<span class="pp-badge pp-b-purple">In your library</span>'
        if not rec.get("abstract"):
            badges += '<span class="pp-badge pp-b-amber">No abstract</span>'
        abstract = rec.get("abstract", "")
        with st.container(border=True):
            st.markdown(
                f'<div class="pp-title">{esc(rec.get("title"))}</div>'
                f'<div class="pp-meta">{esc(rec.get("authors"))} &middot; {esc(rec.get("year") or "n.d.")} &middot; {esc(rec.get("journal"))}</div>'
                f'{badges}<div class="pp-abs">{esc(abstract[:300])}{"..." if len(abstract) > 300 else ""}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1.4, 1])
            c1.button("➕ Add", key=f"{prefix}_add_{rid}", on_click=cb_add_record, args=(rid, False), disabled=inlib)
            if rec.get("pdf_url"):
                c2.button("📄 Add + full text", key=f"{prefix}_pdf_{rid}", on_click=cb_add_record, args=(rid, True),
                          disabled=inlib)
            if rec.get("url"):
                c3.link_button("Open ↗", rec["url"])


def render_claims(res, only_paper=None):
    claims = [c for c in res.get("claims", []) if only_paper in (None, c.get("paper_id"))]
    if not claims:
        st.info("No result sentences were recognised in the available text. For abstract-only papers this is common: "
                "use **Add + full text** for open-access papers.")
        return
    st.caption("Every quote was checked by code: it exists in the paper text. Labels come from the shared vocabulary. "
               "Direction: positive = effect reported, none = no effect, negative = decline, reported = a performance value is stated.")
    for c in claims:
        tags = (f'<span class="pp-badge pp-b-grey">{esc(c.get("population"))}</span>'
                f'<span class="pp-badge pp-b-grey">{esc(c.get("method"))}</span>'
                f'<span class="pp-badge pp-b-purple">{esc(c.get("direction"))}</span>') if c.get("population") else ""
        with st.container(border=True):
            st.markdown(f'<span class="pp-badge pp-b-blue">{esc(c.get("id"))}</span><span class="pp-badge pp-b-grey">{esc(c.get("section"))}</span>{tags}'
                        f'<div class="pp-title">{esc(c.get("statement"))}</div>'
                        f'<div class="pp-quote">“{esc(c.get("quote"))}”</div>'
                        f'<div class="pp-note">{esc(c.get("paper_title"))} | {esc(c.get("methodology"))}</div>', unsafe_allow_html=True)


def render_contradictions(res):
    items = res.get("contradictions", [])
    st.caption("Rule: same intervention and outcome, but some papers report an effect and others report no effect or harm. "
               "Highest weight first (your review decisions change the weight).")
    if not items:
        st.info("No contradictions were found. This is a valid result: no intervention/outcome pair is reported with opposite results. "
                "The rules only compare topics found in the vocabulary; extend `data/vocab.json` to teach PaperPilot new terms.")
    for cont in items:
        a, b = cont.get("claim_a", {}), cont.get("claim_b", {})
        diffs = "".join(f'<span class="pp-badge pp-b-amber">{esc(f)}: {esc(", ".join(d["side_a"]))} vs {esc(", ".join(d["side_b"]))}</span>'
                        for f, d in cont.get("differences", {}).items())
        with st.container(border=True):
            st.markdown(
                f'<span class="pp-badge pp-b-red">{esc(cont.get("severity", "Tension"))}</span><span class="pp-badge pp-b-grey">weight {cont.get("weight", 1.0)}</span>{diffs}'
                f'<div class="pp-title">{esc(cont.get("topic"))}</div>'
                f'<div class="pp-abs"><b>Claim A ({esc(a.get("paper"))}):</b> {esc(a.get("statement"))}</div><div class="pp-quote">“{esc(a.get("quote"))}”</div>'
                f'<div class="pp-abs"><b>Claim B ({esc(b.get("paper"))}):</b> {esc(b.get("statement"))}</div><div class="pp-quote-red">“{esc(b.get("quote"))}”</div>'
                f'<div class="pp-abs"><b>Why they may differ:</b> {esc(cont.get("root_cause"))}</div>'
                f'<div class="pp-abs"><b>Resolution path:</b> {esc(cont.get("resolution_path"))}</div>', unsafe_allow_html=True)


def render_gaps(res):
    items = res.get("research_gaps", [])
    st.caption("Untested combinations come from the population x method table. Stated limitations are real sentences where the authors name a limitation.")
    if not items:
        st.info("No research gaps were found by the rules in the loaded papers.")
    for g in items:
        with st.container(border=True):
            st.markdown(
                f'<span class="pp-badge pp-b-amber">{esc(g.get("category"))}</span><span class="pp-badge pp-b-grey">{esc(g.get("impact_score"))} impact</span>'
                f'<span class="pp-badge pp-b-grey">weight {g.get("weight", 1.0)}</span><div class="pp-title">{esc(g.get("title"))}</div>'
                f'<div class="pp-abs">{esc(g.get("description"))}</div><div class="pp-quote">“{esc(g.get("evidence_quote"))}”</div>'
                f'<div class="pp-abs"><b>Suggested direction:</b> {esc(g.get("suggested_direction"))}</div>', unsafe_allow_html=True)


def render_hypotheses(res, claims, curated):
    hyps, vers = res.get("hypotheses", []), res.get("verifications", {})
    st.caption("The label is decided by code, not by an AI. VERIFIED = claim, population and method all match one real source quote. "
               "PARTIALLY VERIFIED = claim supported, but population or method differs (the untested part). UNVERIFIED = no real quote supports it.")
    if not hyps:
        st.info("No hypotheses were generated because no contradiction or untested combination was found.")
    for h in hyps:
        v = vers.get(h["id"], {})
        label = v.get("verification_label")
        badge = (f'<span class="pp-badge" style="background:{LABEL_COLORS[label]};color:#fff;">{LABEL_TEXT[label]} | match {v.get("overall_grounding_score")}%</span>'
                 if label else "")
        text_shown = h.get("edited_text") or h.get("formulation")
        with st.container(border=True):
            st.markdown(
                f'<span class="pp-badge pp-b-purple">{esc(h.get("id"))}</span>{badge}<span class="pp-badge pp-b-grey">decision: {esc(h.get("status", "pending"))}</span>'
                f'<span class="pp-badge pp-b-grey">{esc(h.get("generated_by", "curated"))}</span><div class="pp-title">{esc(h.get("title"))}</div>'
                f'<div class="pp-abs"><b>Hypothesis{" (edited by researcher)" if h.get("edited_text") else ""}:</b> {esc(text_shown)}</div>'
                f'<div class="pp-abs"><b>Rationale:</b> {esc(h.get("rationale"))}</div><div class="pp-abs"><b>Proposed experiment:</b> {esc(h.get("proposed_methodology"))}</div>',
                unsafe_allow_html=True)
            with st.expander("Evidence and Critic checks"):
                crit = v.get("critic")
                if crit:
                    tick = lambda ok: "✅" if ok else "❌"
                    st.write(f"{tick(crit.get('claim_match'))} claim matches a source   {tick(crit.get('population_match'))} population matches   "
                             f"{tick(crit.get('method_match'))} method matches   {tick(crit.get('quotes_verified'))} quotes found in the paper text")
                    st.caption(crit.get("notes", ""))
                for e in v.get("supporting_evidence", []):
                    st.markdown(f'<div class="pp-quote"><b>{esc(e.get("paper_title"))}</b>: “{esc(e.get("quote"))}”<br><span class="pp-note">{esc(e.get("relevance"))}</span></div>', unsafe_allow_html=True)
                for e in v.get("contradicting_or_limiting_evidence", []):
                    st.markdown(f'<div class="pp-quote-red"><b>{esc(e.get("paper_title"))}</b>: “{esc(e.get("quote"))}”<br><span class="pp-note">{esc(e.get("limitation"))}</span></div>', unsafe_allow_html=True)
                for pz in v.get("retrieved_passages", []):
                    st.caption(f"Related passage [{pz['paper_id']}]: {pz['text'][:260]}")
                st.write(f"**Falsification criteria:** {v.get('falsification_criteria', h.get('falsification_criteria', 'N/A'))}")
                st.write(f"**Verdict:** {v.get('synthesis_verdict', 'N/A')}")
            if not curated:
                b1, b2, b3 = st.columns([1, 1, 3])
                b1.button("✅ Approve", key=f"ap_{h['id']}", on_click=cb_feedback, args=(h["id"], h["finding_id"], "approve"))
                b2.button("❌ Reject", key=f"rj_{h['id']}", on_click=cb_feedback, args=(h["id"], h["finding_id"], "reject"))
                with b3.expander("✏️ Edit the hypothesis text"):
                    tkey = f"edit_{h['id']}"
                    st.text_area("Your version", value=text_shown, key=tkey, height=100)
                    st.button("Save edited version", key=f"sv_{h['id']}", on_click=cb_feedback, args=(h["id"], h["finding_id"], "edit", tkey))

    if not curated and claims:
        with st.expander("🧪 Test the Critic with your own hypothesis"):
            st.caption("Pick a combination and see how the Critic judges it against the loaded papers. A combination no paper supports should not come back VERIFIED.")
            t1, t2, t3, t4, t5 = st.columns(5)
            t_int = t1.selectbox("Intervention", vocab.allowed_values("intervention"), key="t_int")
            t_out = t2.selectbox("Outcome", vocab.allowed_values("outcome"), key="t_out")
            t_pop = t3.selectbox("Population", vocab.allowed_values("population"), key="t_pop")
            t_met = t4.selectbox("Method", vocab.allowed_values("method"), key="t_met")
            t_dir = t5.selectbox("Direction", vocab.DIRECTIONS, key="t_dir")
            if st.button("Check this hypothesis"):
                r = check_custom_hypothesis({"intervention": t_int, "outcome": t_out, "population": t_pop, "method": t_met, "direction": t_dir},
                                            claims, st.session_state.papers)
                st.markdown(f'<span class="pp-badge" style="background:{LABEL_COLORS[r["label"]]};color:#fff;">{LABEL_TEXT[r["label"]]}</span>', unsafe_allow_html=True)
                st.write(r["critic"]["notes"])
                if r["closest_claim"]:
                    st.markdown(f'<div class="pp-quote">Closest source ({esc(r["closest_claim"]["paper_id"])}): “{esc(r["closest_claim"]["quote"])}”</div>', unsafe_allow_html=True)


def donut(pct, label, color):
    if pct is None:
        inner, bg = "n/a", "#E5E7EB"
    else:
        inner, bg = f"{pct}%", f"conic-gradient({color} 0 {pct}%, #E5E7EB {pct}% 100%)"
    return (f'<div class="pp-donut-wrap"><div class="pp-donut" style="background:{bg};"><div class="pp-donut-in">{inner}</div></div>'
            f'<div class="pp-donut-lbl">{label}</div></div>')


def stat(num, label, icon):
    return f'<div class="pp-stat"><div class="pp-stat-num">{icon} {num}</div><div class="pp-stat-lbl">{label}</div></div>'


# =============================================================
# Right column: ANALYTICS (real numbers only)
# =============================================================
with side_col:
    results_now = st.session_state.analysis_results
    events = lib.load_activity()
    numbers = an.summary_numbers(papers, results_now, fb.load_feedback(FEEDBACK_PATH), events)
    st.markdown('<div class="pp-section">ANALYTICS</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Research activity** (last 14 days)")
        series = an.activity_series(events, 14)
        st.line_chart(pd.DataFrame({"events": series["events"]}, index=series["day"]), height=130)
    with st.container(border=True):
        st.markdown("**Topics in your library**")
        topics = an.topic_counts(papers)
        if topics:
            st.bar_chart(pd.DataFrame({"papers": [t[1] for t in topics]}, index=[t[0] for t in topics]), height=190)
        else:
            st.caption("Add papers to see which patient groups, devices and outcomes they cover.")
    with st.container(border=True):
        st.markdown("**Insight quality**")
        st.markdown('<div class="pp-donuts">' + donut(numbers["supported_share"], "hypotheses supported by sources", "#7C3AED")
                    + donut(numbers["agreement"], "researcher agreement", "#06B6D4") + '</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.markdown(stat(numbers["papers"], "Papers in library", "📄"), unsafe_allow_html=True)
        c2.markdown(stat(numbers["claims"], "Claims extracted", "💡"), unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.markdown(stat(numbers["decisions"], "Review decisions", "✍️"), unsafe_allow_html=True)
        c4.markdown(stat(numbers["searches"], "Searches made", "🔎"), unsafe_allow_html=True)
        st.caption(f"{numbers['fulltext']} full text, {numbers['abstract_only']} abstract only. All numbers come from your own library and decisions.")


# =============================================================
# Left column: views
# =============================================================
with main_col:
    st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
    view = st.session_state.view

    if st.session_state.notice:
        kind, text = st.session_state.notice
        getattr(st, kind if kind in ("success", "warning", "info", "error") else "info")(text)
        st.session_state.notice = None

    # ------------------------------------------------ ASSISTANT
    if view == VIEWS[0]:
        agent = ResearchAgent(AgentContext(
            get_papers=lambda: st.session_state.papers, ensure_results=ensure_results,
            search=do_search, llm=make_llm(), log=log))
        for i, m in enumerate(st.session_state.chat):
            with st.chat_message(m["role"], avatar="🧭" if m["role"] == "assistant" else "🧑‍🔬"):
                st.markdown(m["text"])
                if m.get("found"):
                    render_found_cards(m["found"], f"c{i}")
        if len(st.session_state.chat) == 1:
            st.markdown('<div class="pp-note">Try one of these:</div>', unsafe_allow_html=True)
            for j, prompt in enumerate(EXAMPLE_PROMPTS):
                st.button(prompt, key=f"chip_{j}", on_click=cb_chip, args=(prompt,))
        typed = st.chat_input("Ask me to find, review or analyse papers (medical devices, signals, imaging, rehabilitation ...)")
        user_text = typed or st.session_state.pop("pending_input", None)
        if user_text:
            st.session_state.chat.append({"role": "user", "text": user_text, "found": []})
            with st.spinner("Working on it..."):
                try:
                    out = agent.respond(user_text)
                except Exception as err:
                    out = {"text": f"Sorry, something went wrong while handling that ({type(err).__name__}: {str(err)[:120]}). "
                                   "Please try rephrasing.", "found": []}
            st.session_state.chat.append({"role": "assistant", "text": out["text"], "found": [r["rid"] for r in out.get("found", [])]})
            st.rerun()

    # ------------------------------------------------ DISCOVER
    elif view == VIEWS[1]:
        st.markdown('<div class="pp-section">DISCOVER PAPERS</div>', unsafe_allow_html=True)
        preset = st.selectbox("Domain", ["Custom"] + list(disc.DOMAIN_PRESETS.keys()), key="d_preset")
        query = st.text_input("Search topic", value=disc.DOMAIN_PRESETS.get(preset, "") if preset != "Custom" else "",
                              key=f"d_query_{preset}", placeholder="e.g. wearable ECG arrhythmia detection")
        f1, f2, f3, f4 = st.columns(4)
        year_from = f1.number_input("From year (0 = any)", min_value=0, max_value=2100, value=0, step=1, key="d_year")
        sort = f2.selectbox("Sort by", ["relevance", "recent", "citations"], key="d_sort")
        limit = f3.slider("How many", 5, 25, 10, key="d_limit")
        oa = f4.checkbox("Open access only", key="d_oa")
        if st.button("🔎 Search", type="primary", disabled=not query.strip()):
            log("search", query)
            with st.spinner("Searching Europe PMC and OpenAlex..."):
                recs, status = do_search({"query": query, "year_from": int(year_from) or None, "oa_only": oa, "sort": sort, "limit": limit})
            st.session_state.discover_ids = [r["rid"] for r in recs]
            st.session_state.discover_status = status
        if st.session_state.discover_status:
            st.caption("Sources: " + " | ".join(f"{k}: {v}" for k, v in st.session_state.discover_status.items()))
        if st.session_state.discover_ids:
            render_found_cards(st.session_state.discover_ids, "d")
        elif st.session_state.discover_status:
            st.info("No papers found. Try fewer words, remove the year filter, or use a more general term.")
        else:
            st.info("Choose a domain or type a topic, then press Search. Results come from real databases, with a link to every source.")

    # ------------------------------------------------ LIBRARY
    elif view == VIEWS[2]:
        st.markdown('<div class="pp-section">RESEARCH PAPERS</div>', unsafe_allow_html=True)
        if not papers:
            st.info("Your library is empty. Ask the assistant to find papers, use **Discover**, or add PDFs in the sidebar.")
        else:
            fcol, scol = st.columns([2, 1])
            needle = fcol.text_input("Search your library", key="lib_filter", placeholder="title, author or keyword").strip().lower()
            order = scol.selectbox("Order", ["Added (oldest first)", "Year (newest first)", "Title (A-Z)"], key="lib_order")
            shown = [p for p in papers if not needle or needle in (p.get("title", "") + " " + str(p.get("authors", "")) + " " + p.get("abstract", "")).lower()]
            if order.startswith("Year"):
                shown.sort(key=lambda p: str(p.get("year", "")), reverse=True)
            elif order.startswith("Title"):
                shown.sort(key=lambda p: p.get("title", "").lower())
            for row in range(0, len(shown), 3):
                cols = st.columns(3)
                for col, p in zip(cols, shown[row:row + 3]):
                    with col, st.container(border=True):
                        kind = ('<span class="pp-badge pp-b-amber">Abstract only</span>' if p.get("source_type") == "abstract"
                                else '<span class="pp-badge pp-b-green">Full text</span>')
                        ab = p.get("abstract", "")
                        st.markdown(f'<span class="pp-badge pp-b-blue">{esc(p.get("paper_id"))}</span>{kind}'
                                    f'<div class="pp-title">{esc(p.get("title"))}</div>'
                                    f'<div class="pp-meta">{esc(p.get("authors"))} &middot; {esc(p.get("year"))}</div>'
                                    f'<div class="pp-abs">{esc(ab[:170])}{"..." if len(ab) > 170 else ""}</div>', unsafe_allow_html=True)
                        b1, b2, b3 = st.columns([1.6, 1, 0.8])
                        b1.button("🔬 Review", key=f"rv_{p['paper_id']}", on_click=cb_open_review, args=(p["paper_id"],))
                        if p.get("url"):
                            b2.link_button("↗", p["url"])
                        b3.button("🗑️", key=f"rm_{p['paper_id']}", on_click=cb_remove, args=(p["paper_id"],))

    # ------------------------------------------------ ANALYSIS
    elif view == VIEWS[3]:
        st.markdown('<div class="pp-section">ANALYSIS</div>', unsafe_allow_html=True)
        if not papers:
            st.info("Add papers first (ask the assistant to find some, or use the sidebar).")
        else:
            if live_mode:
                stale = st.session_state.analysis_results is None or st.session_state.analysis_sig != analysis_signature()
                if st.button("▶️ Run analysis (uses the AI model)", type="primary"):
                    run_analysis()
                elif stale:
                    st.info("Press **Run analysis** to analyse the current library with the AI model.")
            res = ensure_results() if not live_mode else (st.session_state.analysis_results or {})
            if res:
                st.caption(f"Engine: **{res.get('engine_used', '')}**")
                if res.get("warning"):
                    st.warning(res["warning"])
                curated = res.get("curated", False)
                if curated:
                    st.warning("This is a hand-written example for the bundled papers. It was NOT computed by the pipeline; its scores are illustrative.")
                t1, t2, t3, t4 = st.tabs([f"⚡ Contradictions ({len(res.get('contradictions', []))})", f"⚠️ Gaps ({len(res.get('research_gaps', []))})",
                                          f"🔮 Hypotheses ({len(res.get('hypotheses', []))})", f"💡 Claims ({len(res.get('claims', []))})"])
                with t1:
                    render_contradictions(res)
                with t2:
                    render_gaps(res)
                with t3:
                    render_hypotheses(res, res.get("claims", []), curated)
                with t4:
                    render_claims(res)

    # ------------------------------------------------ PAPER REVIEW
    elif view == VIEWS[4]:
        st.markdown('<div class="pp-section">PAPER REVIEW</div>', unsafe_allow_html=True)
        if not papers:
            st.info("Add papers first.")
        else:
            ids = [p["paper_id"] for p in papers]
            if st.session_state.selected_paper not in ids:
                st.session_state.selected_paper = ids[0]
            pid = st.selectbox("Paper", ids, key="selected_paper",
                               format_func=lambda i: f"{i}: {next(p for p in papers if p['paper_id'] == i).get('title', '')[:90]}")
            paper = next(p for p in papers if p["paper_id"] == pid)
            res = ensure_results() if not live_mode else (st.session_state.analysis_results or {"claims": []})
            with st.container(border=True):
                kind = "Abstract only" if paper.get("source_type") == "abstract" else "Full text"
                st.markdown(f'<span class="pp-badge pp-b-blue">{esc(pid)}</span><span class="pp-badge pp-b-grey">{kind}</span>'
                            f'<div class="pp-title">{esc(paper.get("title"))}</div>'
                            f'<div class="pp-meta">{esc(paper.get("authors"))} &middot; {esc(paper.get("year"))} &middot; {esc(paper.get("journal", ""))}</div>',
                            unsafe_allow_html=True)
                if paper.get("warning"):
                    st.warning(paper["warning"])
                if paper.get("url"):
                    st.link_button("Open the paper ↗", paper["url"])
            s1, s2, s3 = st.tabs(["📝 Summary", "✅ Appraisal checklist", "💡 Claims with quotes"])
            with s1:
                st.markdown(summarize_paper(paper, res.get("claims", [])))
            with s2:
                st.markdown(appraisal_markdown(appraise_paper(paper)))
            with s3:
                render_claims(res, only_paper=pid)

    # ------------------------------------------------ GRAPH
    elif view == VIEWS[5]:
        st.markdown('<div class="pp-section">KNOWLEDGE GRAPH</div>', unsafe_allow_html=True)
        if not papers:
            st.info("Add papers first.")
        else:
            res = ensure_results() if not live_mode else (st.session_state.analysis_results or {})
            st.caption("Papers, claims, contradictions, gaps and hypotheses. Drag nodes, hover for details. The first view needs an internet connection to load the drawing library.")
            gd = build_knowledge_graph_data(papers, res.get("claims", []), res.get("contradictions", []), res.get("research_gaps", []), res.get("hypotheses", []))
            components.html(generate_interactive_graph_html(gd, height_px=640), height=680, scrolling=False)

    # ------------------------------------------------ QUALITY & EXPORT
    else:
        st.markdown('<div class="pp-section">QUALITY &amp; EXPORT</div>', unsafe_allow_html=True)
        if not papers:
            st.info("Add papers first.")
        else:
            res = ensure_results() if not live_mode else (st.session_state.analysis_results or {})
            st.caption("Numbers are measured against a hand-written answer key (ground_truth.json). Report the real values, even when they are low.")
            gt_up = st.file_uploader("Upload your ground_truth.json (optional)", type=["json"], key="gt_upload")
            if gt_up is not None:
                try:
                    st.session_state.ground_truth = json.load(gt_up)
                except ValueError:
                    st.error("That file is not valid JSON.")
            entries = fb.load_feedback(FEEDBACK_PATH)
            m = evaluate_all(st.session_state.ground_truth, res, entries, len(papers))
            fmt = lambda x: "n/a" if x is None else f"{x * 100:.0f}%"
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Papers", m["papers_tested"])
            q2.metric("Claim coverage", fmt(m["claim_coverage"]), help="Target: more than 90%")
            q3.metric("Contradiction precision", fmt(m["contradiction_precision"]), help="Target: more than 85%")
            q4.metric("Reviewer agreement", fmt(m["human_agreement"]), help=f"Approved or edited / reviewed ({m['hypotheses_reviewed']} reviewed). Target: more than 80%")
            if st.session_state.ground_truth is None:
                st.info("No ground truth loaded, so coverage and precision show n/a. Copy data/ground_truth_template.json, fill it in by hand and upload it here.")
            st.markdown("---")
            report_md = PaperPilotAnalyzer.generate_markdown_report(papers, res) if res else "No analysis yet."
            e1, e2, e3 = st.columns(3)
            e1.download_button("⬇️ Report (Markdown)", report_md, file_name="PaperPilot_Report.md", mime="text/markdown")
            e2.download_button("⬇️ Results (JSON)", json.dumps(res, indent=2, default=str), file_name="paperpilot_results.json", mime="application/json")
            e3.download_button("⬇️ Library (JSON)", json.dumps(papers, indent=2, default=str), file_name="paperpilot_library.json", mime="application/json")
            with st.expander("Preview the report"):
                st.markdown(report_md)
