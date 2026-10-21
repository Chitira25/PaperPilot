"""
The PaperPilot research agent (chat brain).

It understands a request, picks a tool, runs it and answers in plain language:
    find papers   -> search Europe PMC + OpenAlex (real databases, never invented)
    analyse       -> run the claim / contradiction / gap / hypothesis pipeline on the library
    review        -> study design, sample size and a reporting checklist for one paper
    summarise     -> extractive summary of one paper
    compare       -> compare what the library papers report
    contradictions / gaps / hypotheses -> show the findings, with real quotes
    questions     -> answer from the library (extractive; written answer only when a Live LLM is on)

Routing uses plain rules, so the agent behaves the same every time and works without an API key.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import vocab
from .appraisal import appraise_paper, appraisal_markdown, summarize_paper
from .textutil import quote_is_real
from .verify import build_chunks, retrieve

HELP_TEXT = (
    "I am your research assistant for **medical instrumentation and health research**. I can:\n\n"
    "- **Find papers**: *find recent open-access papers on wearable ECG for arrhythmia detection*\n"
    "- **Analyse your library**: *analyse my papers* (claims, contradictions, gaps, hypotheses)\n"
    "- **Review a paper**: *review P2* (study design, sample size, reporting checklist)\n"
    "- **Summarise a paper**: *summarise paper 3*\n"
    "- **Compare**: *compare my papers*, *show contradictions*, *what are the research gaps?*\n"
    "- **Answer questions** from your papers: *which sensor was used in the stroke studies?*\n\n"
    "Papers come from Europe PMC and OpenAlex. I never make up a paper, a quote or a number."
)


@dataclass
class AgentContext:
    get_papers: Callable[[], List[Dict[str, Any]]]
    ensure_results: Callable[[], Dict[str, Any]]            # runs the analysis if it is missing or out of date
    search: Callable[[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, str]]]
    llm: Any = None                                         # an LLMClient or None
    log: Callable[[str, str], None] = lambda kind, detail="": None


# ------------------------------------------------------------------ parsing
_CMD = (r"(?:please\s+)?(?:can you\s+|could you\s+)?(?:find|search(?: for)?|look(?: for| up)|recommend|suggest|give|get|show|fetch|discover|bring)"
        r"(?:\s+me)?")


def parse_search_request(text: str, now_year: Optional[int] = None) -> Dict[str, Any]:
    """Turn 'find recent open access papers on wearable ECG since 2021' into search settings."""
    year = now_year or datetime.now().year
    t = " " + text.strip() + " "
    low = t.lower()
    oa = bool(re.search(r"open[- ]access|free (full[- ]text|pdf|papers)", low))
    sort = "relevance"
    year_from = None
    limit = 10

    if re.search(r"most[- ]cited|highly[- ]cited|top[- ]cited|influential|landmark|seminal", low):
        sort = "citations"
    elif re.search(r"\b(recent|latest|newest|new)\b", low):
        sort = "recent"
        year_from = year - 3
    m = re.search(r"(?:last|past)\s+(\d{1,2})\s+years?", low)
    if m:
        year_from = year - int(m.group(1))
    m = re.search(r"(?:since|from|after)\s+(20\d\d|19\d\d)", low)
    if m:
        year_from = int(m.group(1))
    m = re.search(r"\btop\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:papers|articles|studies)\b", low)
    if m:
        limit = max(3, min(25, int(m.group(1) or m.group(2))))

    q = t
    q = re.sub(r"(?:last|past)\s+\d{1,2}\s+years?", " ", q, flags=re.I)
    q = re.sub(r"(?:since|from|after)\s+(?:20|19)\d\d", " ", q, flags=re.I)
    q = re.sub(r"\btop\s+\d{1,2}\b|\b\d{1,2}\s+(?:papers|articles|studies)\b", " ", q, flags=re.I)
    q = re.sub(_CMD, " ", q, flags=re.I)
    q = re.sub(r"\b(any|some|the|a|an|me|please|good|best|relevant|research|scientific|published)\b", " ", q, flags=re.I)
    q = re.sub(r"\b(recent|latest|newest|new|most[- ]cited|highly[- ]cited|top[- ]cited|open[- ]access|free)\b", " ", q, flags=re.I)
    q = re.sub(r"\b(papers?|articles?|studies|literature|publications?)\b", " ", q, flags=re.I)
    q = re.sub(r"^\s*(?:on|about|for|regarding|related to|in|of)\b", " ", q.strip(), flags=re.I)
    q = re.sub(r"[?!.]+$", "", re.sub(r"\s+", " ", q).strip())
    if len(q) < 3:
        q = re.sub(r"\s+", " ", text).strip()
    return {"query": q, "year_from": year_from, "oa_only": oa, "sort": sort, "limit": limit}


def find_paper_ref(text: str, papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Which library paper does the message refer to? ('P3', 'paper 2', 'first paper', part of a title)."""
    if not papers:
        return None
    low = text.lower()
    m = re.search(r"\bp(\d+)\b", low)
    if m:
        for p in papers:
            if str(p.get("paper_id", "")).lower() == f"p{m.group(1)}":
                return p
    m = re.search(r"\bpaper\s*#?\s*(\d+)\b", low)
    if m and 1 <= int(m.group(1)) <= len(papers):
        return papers[int(m.group(1)) - 1]
    if re.search(r"\b(first|1st)\s+paper\b", low):
        return papers[0]
    if re.search(r"\b(last|latest added|newest)\s+paper\b", low):
        return papers[-1]
    for p in papers:
        title = (p.get("title") or "").lower()
        if len(title) > 15 and title[:40] in low:
            return p
    return None


def unsupported_quotes(answer: str, papers: List[Dict[str, Any]], min_words: int = 8) -> List[str]:
    """Quoted passages in an AI answer that do NOT appear in any library paper."""
    texts = [p.get("full_text", "") for p in papers]
    bad = []
    for q in re.findall(r"[\"“]([^\"”]{20,})[\"”]", answer):
        if len(q.split()) >= min_words and not any(quote_is_real(q, t) for t in texts):
            bad.append(q)
    return bad


def classify_intent(text: str, papers: List[Dict[str, Any]]) -> str:
    low = text.lower().strip()
    ref = find_paper_ref(text, papers)
    if re.search(r"^(help|hi|hello|hey)\b|what can you do|how do (i|you) use", low):
        return "help"
    if re.search(r"\b(contradict\w*|conflict\w*|disagree\w*|inconsisten\w*)\b", low):
        return "contradictions"
    if re.search(r"\bgaps?\b|untested|future work|missing research|research opportunit", low):
        return "gaps"
    if re.search(r"hypothes|research idea|what (should|could) (we|i) (test|study|try)", low):
        return "hypotheses"
    if re.search(r"\b(summar(y|ise|ize)|tl;?dr|key findings?|main findings?)\b", low) and (ref or not re.search(r"\b(on|about)\b", low)):
        return "summary"
    if re.search(r"\b(review|appraise|appraisal|critique|critically|assess quality|study design|quality of)\b", low) and ref:
        return "review"
    if re.search(r"\bcompar(e|ison)\b|difference between|side by side", low) and papers and not re.search(r"\b(find|search)\b", low):
        return "compare"
    if re.search(r"\b(analy[sz]e|analysis|run the pipeline|process (my|the|all))\b", low) and not re.search(r"\b(find|search|papers? (on|about))\b", low):
        return "analyze"
    if re.search(r"\b(overview|list|what do i have|my library|my papers)\b", low) and papers and not re.search(r"\b(find|search)\b", low):
        return "overview"
    if re.search(rf"{_CMD}\b|papers?\s+(on|about|for|regarding)|literature (on|about)|articles?\s+(on|about)|\b(latest|recent)\b.*\b(papers?|research|studies|articles)\b", low) \
            or re.search(r"\b(find|search|recommend|suggest|discover)\b", low):
        return "search"
    return "question" if papers else "search"


# ------------------------------------------------------------------ the agent
class ResearchAgent:
    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def respond(self, text: str) -> Dict[str, Any]:
        papers = self.ctx.get_papers()
        intent = classify_intent(text, papers)
        handler = getattr(self, f"_do_{intent}")
        out = handler(text, papers)
        out.setdefault("found", [])
        out["intent"] = intent
        return out

    # -- tools -----------------------------------------------------------
    def _do_help(self, text, papers):
        return {"text": HELP_TEXT}

    def _do_search(self, text, papers):
        req = parse_search_request(text)
        self.ctx.log("search", req["query"])
        records, status = self.ctx.search(req)
        notes = " | ".join(f"{k}: {v}" for k, v in status.items())
        filt = []
        if req["year_from"]:
            filt.append(f"since {req['year_from']}")
        if req["oa_only"]:
            filt.append("open access only")
        filt.append({"relevance": "most relevant first", "citations": "most cited first", "recent": "newest first"}[req["sort"]])
        if not records:
            reach = all("not reachable" in v for v in status.values())
            msg = ("I could not reach the paper databases. Check your internet connection and try again." if reach
                   else f"I found no papers for **{req['query']}** with these filters. Try fewer words, remove the year or "
                        "open-access filter, or use a more general term.")
            return {"text": f"{msg}\n\n*Sources: {notes}*", "found": []}
        lines = [f"I found **{len(records)} papers** for **{req['query']}** ({', '.join(filt)}).", "",
                 "Add the ones you want to your library with the buttons below each paper. "
                 "Then ask me to *analyse my papers* or *review P1*.", "", f"*Sources: {notes}*"]
        return {"text": "\n".join(lines), "found": records}

    def _do_analyze(self, text, papers):
        if not papers:
            return {"text": "Your library is empty. Ask me to *find papers on* a topic, add some, and then I can analyse them."}
        res = self.ctx.ensure_results()
        self.ctx.log("analyze", f"{len(papers)} papers")
        abs_only = sum(1 for p in papers if p.get("source_type") == "abstract")
        extra = (f"\n\n*{abs_only} of the {len(papers)} papers are abstract-only, so their claims come from the abstract. "
                 "Use **Add + full text** on open-access papers for deeper analysis.*") if abs_only else ""
        return {"text": (f"I analysed **{len(papers)} papers**: **{len(res.get('claims', []))} claims**, "
                         f"**{len(res.get('contradictions', []))} contradictions**, **{len(res.get('research_gaps', []))} gaps**, "
                         f"**{len(res.get('hypotheses', []))} hypotheses**.\n\nAsk me to *show contradictions*, *show gaps* or "
                         f"*show hypotheses*, or open the **Analysis** view.{extra}")}

    def _need_papers(self, papers):
        if not papers:
            return {"text": "Your library is empty. Ask me to *find papers on* a topic and add some first."}
        return None

    def _do_contradictions(self, text, papers):
        if (r := self._need_papers(papers)):
            return r
        res = self.ctx.ensure_results()
        items = res.get("contradictions", [])
        if not items:
            return {"text": "No contradictions were found. This is a valid result: no intervention/outcome pair is reported with "
                            "opposite results. Note that the rules only compare topics found in the vocabulary "
                            "(you can extend it in `data/vocab.json`)."}
        L = [f"I found **{len(items)} contradiction(s)**:"]
        for c in items[:4]:
            a, b = c.get("claim_a", {}), c.get("claim_b", {})
            L += ["", f"**{c.get('topic')}**", f"- {a.get('paper_id', '')}: “{a.get('quote', '')}”", f"- {b.get('paper_id', '')}: “{b.get('quote', '')}”",
                  f"- *Why they may differ:* {c.get('root_cause', '')}"]
        return {"text": "\n".join(L)}

    def _do_gaps(self, text, papers):
        if (r := self._need_papers(papers)):
            return r
        res = self.ctx.ensure_results()
        items = res.get("research_gaps", [])
        if not items:
            return {"text": "No research gaps were found by the rules in the loaded papers."}
        L = [f"I found **{len(items)} gap(s)** (most important first):"]
        for g in items[:5]:
            L += ["", f"**{g.get('title')}** ({g.get('category')})", f"- {g.get('description')}", f"- Evidence: “{g.get('evidence_quote')}”"]
        return {"text": "\n".join(L)}

    def _do_hypotheses(self, text, papers):
        if (r := self._need_papers(papers)):
            return r
        res = self.ctx.ensure_results()
        hyps, vers = res.get("hypotheses", []), res.get("verifications", {})
        if not hyps:
            return {"text": "No hypotheses were generated because no contradiction or untested combination was found."}
        L = [f"I wrote **{len(hyps)} hypothesis(es)**. Each one is checked against real quotes:"]
        for h in hyps[:5]:
            v = vers.get(h["id"], {})
            L += ["", f"**{h.get('title')}** — {v.get('verification_label', 'n/a').replace('_', ' ')}", f"- {h.get('edited_text') or h.get('formulation')}",
                  f"- *Critic:* {v.get('critic', {}).get('notes', '')}"]
        L += ["", "Open the **Analysis** view to approve, edit or reject them."]
        return {"text": "\n".join(L)}

    def _do_summary(self, text, papers):
        if (r := self._need_papers(papers)):
            return r
        paper = find_paper_ref(text, papers)
        if not paper:
            return {"text": "Which paper? Say for example *summarise P2* or *summarise paper 3*.\n\n" + self._library_table(papers)}
        res = self.ctx.ensure_results()
        return {"text": summarize_paper(paper, res.get("claims", [])), "focus_paper": paper["paper_id"]}

    def _do_review(self, text, papers):
        paper = find_paper_ref(text, papers)
        if not paper:
            return {"text": "Which paper should I review? Say for example *review P2*.\n\n" + self._library_table(papers)}
        self.ctx.log("review", paper["paper_id"])
        return {"text": appraisal_markdown(appraise_paper(paper)), "focus_paper": paper["paper_id"]}

    def _do_overview(self, text, papers):
        return {"text": self._library_table(papers)}

    def _do_compare(self, text, papers):
        res = self.ctx.ensure_results()
        claims = res.get("claims", [])
        rows = ["| Paper | Patient group(s) | Device / method(s) | Reported direction(s) |", "|---|---|---|---|"]
        for p in papers:
            mine = [c for c in claims if c.get("paper_id") == p["paper_id"]]
            pops = sorted({vocab.label_text(c["population"]) for c in mine if c.get("population", "other") != "other"})
            meths = sorted({vocab.label_text(c["method"]) for c in mine if c.get("method", "other") != "other"})
            dirs = sorted({c["direction"] for c in mine if c.get("direction")})
            rows.append(f"| {p['paper_id']} | {', '.join(pops) or '-'} | {', '.join(meths) or '-'} | {', '.join(dirs) or '-'} |")
        cont = res.get("contradictions", [])
        tail = (f"\n\n**{len(cont)} contradiction(s)** were found between these papers. Ask me to *show contradictions*."
                if cont else "\n\nNo contradictions were found between these papers.")
        return {"text": "\n".join(rows) + tail}

    def _do_question(self, text, papers):
        chunks = build_chunks(papers)
        passages = retrieve(text, chunks, k=4)
        if not passages:
            return {"text": "I could not find anything about that in your library. I can **search** for papers on it: "
                            f"say *find papers on {text.strip().rstrip('?')}*."}
        sources = ", ".join(sorted({p["paper_id"] for p in passages}))
        llm = self.ctx.llm
        if llm is not None and getattr(llm, "is_configured", lambda: False)():
            ctx_text = "\n\n".join(f"[{p['paper_id']}] {p['text']}" for p in passages)
            system = ("You answer questions about research papers using ONLY the passages given. Cite the paper id like [P2] after each "
                      "statement. If the passages do not contain the answer, say it is not in the loaded papers. Do not invent facts.")
            try:
                answer = llm.generate_completion(system, f"QUESTION: {text}\n\nPASSAGES:\n{ctx_text}", temperature=0.1)
                warn = unsupported_quotes(answer, papers)
                tail = ("\n\n⚠️ A quotation in this answer could not be found in your papers. Do not trust that quote."
                        if warn else "")
                return {"text": f"{answer}{tail}\n\n*Written by the AI model from these passages. Sources: {sources}.*"}
            except Exception as err:
                note = f"\n\n*The AI model failed ({str(err)[:80]}), so here are the matching passages instead.*"
        else:
            note = "\n\n*Extractive answer: these are the most relevant passages. Turn on Live LLM mode for a written answer.*"
        L = ["Here is what your papers say about this:"]
        for p in passages[:3]:
            L.append(f"\n- **[{p['paper_id']}]** “{p['text'][:320]}{'...' if len(p['text']) > 320 else ''}”")
        return {"text": "\n".join(L) + note}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _library_table(papers: List[Dict[str, Any]]) -> str:
        if not papers:
            return "Your library is empty."
        rows = ["| ID | Year | Title | Text available |", "|---|---|---|---|"]
        for p in papers:
            kind = "abstract only" if p.get("source_type") == "abstract" else "full text"
            rows.append(f"| {p['paper_id']} | {p.get('year') or ''} | {p.get('title', '')[:90]} | {kind} |")
        return "\n".join(rows)
