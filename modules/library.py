"""
Paper library (saved between sessions) and the activity log used by the analytics panel.
Files: data/library.json and data/activity.json
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(BASE, "data", "library.json")
ACTIVITY_PATH = os.path.join(BASE, "data", "activity.json")


def _read(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def load_library(path: str = LIB_PATH) -> List[Dict[str, Any]]:
    data = _read(path, [])
    return data if isinstance(data, list) else []


def save_library(papers: List[Dict[str, Any]], path: str = LIB_PATH) -> None:
    _write(path, papers)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:90]


def find_existing(papers: List[Dict[str, Any]], paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for p in papers:
        if paper.get("rid") and p.get("rid") == paper["rid"]:
            return p
        if paper.get("doi") and p.get("doi") == paper["doi"]:
            return p
        if _norm_title(p.get("title")) == _norm_title(paper.get("title")) and _norm_title(paper.get("title")):
            return p
    return None


def next_paper_id(papers: List[Dict[str, Any]]) -> str:
    used = {p.get("paper_id") for p in papers}
    i = 1
    while f"P{i}" in used:
        i += 1
    return f"P{i}"


def add_paper(papers: List[Dict[str, Any]], paper: Dict[str, Any]):
    """Add a paper. Returns (papers, stored_paper, was_new). A duplicate is not added twice."""
    existing = find_existing(papers, paper)
    if existing:
        return papers, existing, False
    if not paper.get("paper_id") or paper["paper_id"] in {p.get("paper_id") for p in papers}:
        paper = dict(paper, paper_id=next_paper_id(papers))
    return papers + [paper], paper, True


def remove_paper(papers: List[Dict[str, Any]], paper_id: str) -> List[Dict[str, Any]]:
    return [p for p in papers if p.get("paper_id") != paper_id]


# ------------------------------------------------------------------ activity log
def log_event(kind: str, detail: str = "", path: str = ACTIVITY_PATH) -> None:
    events = _read(path, [])
    events.append({"time": datetime.now().isoformat(timespec="seconds"), "kind": kind, "detail": detail[:200]})
    _write(path, events[-2000:])


def load_activity(path: str = ACTIVITY_PATH) -> List[Dict[str, Any]]:
    data = _read(path, [])
    return data if isinstance(data, list) else []
