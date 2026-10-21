"""
Researcher feedback loop.

The researcher approves / edits / rejects each hypothesis. The decisions are saved in
data/feedback.json. They change the WEIGHT of the finding (contradiction or gap) behind the hypothesis,
and findings are shown highest weight first. Weights are always recomputed from the full feedback file
(only the latest decision per hypothesis counts), so repeating a run never counts a decision twice.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

DELTA = {"approve": 0.2, "edit": 0.1, "reject": -0.3}
VALID_ACTIONS = tuple(DELTA.keys())

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.json")


def load_feedback(path: str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_feedback(entries: List[Dict[str, Any]], path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)


def record_feedback(hypothesis_id: str, finding_id: str, action: str, edited_text: Optional[str] = None,
                    note: str = "", reviewer: str = "researcher", path: str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}")
    entries = load_feedback(path)
    entries.append({
        "hypothesis_id": hypothesis_id,
        "finding_id": finding_id,
        "action": action,
        "edited_text": edited_text if action == "edit" else None,
        "note": note,
        "reviewer": reviewer,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    save_feedback(entries, path)
    return entries


def latest_by_hypothesis(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for e in entries:                       # later entries overwrite earlier ones
        if e.get("action") in VALID_ACTIONS and e.get("hypothesis_id"):
            latest[e["hypothesis_id"]] = e
    return latest


def compute_weights(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for e in latest_by_hypothesis(entries).values():
        fid = e.get("finding_id")
        if fid:
            weights[fid] = weights.get(fid, 1.0) + DELTA[e["action"]]
    return {k: round(max(0.0, v), 2) for k, v in weights.items()}


def apply_feedback(findings: List[Dict[str, Any]], entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set the weight of every finding from the feedback file and sort (highest weight first)."""
    weights = compute_weights(entries)
    for f in findings:
        f["weight"] = weights.get(f.get("id"), 1.0)
    return sorted(findings, key=lambda f: (-f["weight"], -f.get("score", 0)))
