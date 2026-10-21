# PaperPilot v2 (this version)

| Area | Change |
|---|---|
| Interface | New light theme with cards, pill navigation, a chat assistant, a paper library grid and an analytics panel, following the mockup. Version-safe Streamlit calls (no deprecated arguments) |
| Research agent | Chat that routes requests to tools: search, analyse, review, summarise, compare, contradictions, gaps, hypotheses, question answering. Rule-based routing, works without an API key |
| Finding papers | Europe PMC and OpenAlex search with year, open-access and sort filters, duplicate merging, open-access PDF download. Results are real records with links; nothing is invented |
| Medical domain | Vocabulary extended with devices (ECG, EEG, EMG, PPG, ultrasound, MRI, wearables, glucose and blood-pressure monitors ...), patient groups and outcomes. New claim type "reported" for performance numbers (sensitivity, RMSE ...) which device papers use instead of "improved" |
| Paper review | Study design, sample size, reporting checklist (control group, randomisation, blinding, ethics, statistics, metrics, limitations) with evidence sentences, clearly marked as a checklist and not a quality score |
| Library | Saved between sessions, duplicate detection, abstract-only vs full-text marking |
| Analytics | Activity, topics, insight quality, counters. All computed from real data; empty states show n/a |
| Tests | New `scripts/test_agent.py` (48 checks) |

# v1.1 corrections (previous version)


| # | Problem found in the original | Change |
|---|---|---|
| 1 | Export tab crashed: `json` was used but never imported in `app.py` | Import added; export now offers report, results and feedback downloads |
| 2 | Offline engine invented a contradiction, a fixed gap quote that is not in any paper, a template hypothesis with ">25%", and the same 86% score for everything | Removed. Offline engine now only reports what rules find; empty result is allowed |
| 3 | PDF section splitter treated any line starting with "Results", "Methods" ... as a heading and dropped that line | Headings must be short heading-like lines; body text is kept; reference list removed; scanned-PDF warning added |
| 4 | Claims had no population / method / dataset / direction | Structured claims using a shared vocabulary (`vocab.py`, extendable via `data/vocab.json`) |
| 5 | AI quotes were never checked; the grounding score was whatever the model said | Every quote is checked against the paper text; label decided by code (`verify.final_label`); AI critic can never raise a label |
| 6 | Contradictions were found by hard-coded RAG words | Rule on (intervention, outcome, direction) with population / method / dataset differences as the explanation |
| 7 | No researcher review | Approve / Edit / Reject buttons, `data/feedback.json`, finding weights, stable ids |
| 8 | No evaluation | `evaluate.py`, Quality Report tab, `ground_truth.json` format and template |
| 9 | AI calls: whole-paper JSON cut at 3000 characters, no retry, silent fallback to the fake engine, API key in the URL | One call per paper with sentence-safe text, JSON retry, disk cache, honest fallback with a visible warning, key sent in a header |
| 10 | Graph: unpinned CDN library, invented P1/P2 fallback edges, unescaped text in tooltips, unused `pyvis` | Pinned version with a visible error if it cannot load, no invented edges, HTML escaped, `pyvis` removed |
| 11 | Demo mode returned hard-coded results for the bundled papers | Now optional and labelled "hand-written, not computed"; a synthetic rehab demo runs the real pipeline |
| 12 | Tests only checked that outputs were not empty | 38 correctness tests (real quotes, Critic rejects made-up hypotheses, no invented findings, feedback changes weights, AI path with a stub model) |
