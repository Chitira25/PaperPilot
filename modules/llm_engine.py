"""
LLM Engine for PaperPilot.

Clean REST client for Gemini, OpenAI, Groq, Ollama and any OpenAI-compatible server, plus the three
AI steps used in Live LLM mode:
    1. extract_claims_with_llm  - reads one paper at a time, returns structured claims
    2. generate_hypothesis_with_llm - the Generator
    3. critic_with_llm - the Critic (answers 3 true/false questions only)
Everything the AI returns is checked by code afterwards (quotes must exist in the paper, labels come
from verify.final_label), so the AI can never "verify itself".
"""

import hashlib
import json
import os
import re
import requests
from typing import Dict, List, Any, Optional

from . import vocab
from .extraction import tag_claim
from .textutil import clean_text

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")


def clean_json_response(raw_text: str) -> Any:
    """Remove markdown fences and parse JSON. Raises ValueError if nothing parses."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse valid JSON from LLM response: {raw_text[:200]}...")


class LLMClient:
    def __init__(self, provider: str = "demo", api_key: str = "", model: str = "",
                 custom_base_url: str = "", use_cache: bool = True):
        self.provider = provider.lower()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.custom_base_url = custom_base_url.strip()
        self.use_cache = use_cache
        self._cache: Optional[Dict[str, str]] = None

    def is_configured(self) -> bool:
        if self.provider == "demo":
            return False            # "demo" means: no AI model is used
        if self.provider == "ollama":
            return True             # Ollama runs locally without an API key
        if self.provider == "custom":
            return bool(self.custom_base_url)
        return bool(self.api_key)

    # ---- tiny disk cache: the same prompt is never paid for twice ----
    def _load_cache(self) -> Dict[str, str]:
        if self._cache is None:
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
        except OSError:
            pass

    def generate_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        key = hashlib.sha256(f"{self.provider}|{self.model}|{temperature}|{system_prompt}|{user_prompt}".encode()).hexdigest()
        if self.use_cache:
            cached = self._load_cache().get(key)
            if cached is not None:
                return cached
        text = self._call_provider(system_prompt, user_prompt, temperature)
        if self.use_cache:
            self._load_cache()[key] = text
            self._save_cache()
        return text

    def _call_provider(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        if self.provider == "gemini":
            model_name = self.model or "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
            }
            # the key goes in a header, not in the URL, so it cannot end up in logs
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            candidates = resp.json().get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
            raise ValueError("Gemini returned no candidates.")

        if self.provider in ("openai", "groq", "ollama", "custom"):
            if self.provider == "groq":
                base_url = "https://api.groq.com/openai/v1/chat/completions"
                model_name = self.model or "llama-3.3-70b-versatile"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            elif self.provider == "ollama":
                base_url = (self.custom_base_url or "http://localhost:11434").rstrip("/") + "/v1/chat/completions"
                model_name = self.model or "llama3"
                headers = {"Content-Type": "application/json"}
            elif self.provider == "custom":
                base_url = self.custom_base_url.rstrip("/") + "/chat/completions"
                model_name = self.model or "gpt-4o"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            else:
                base_url = "https://api.openai.com/v1/chat/completions"
                model_name = self.model or "gpt-4o-mini"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"model": model_name, "temperature": temperature,
                       "messages": [{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}]}
            resp = requests.post(base_url, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        raise ValueError(f"Unsupported provider: {self.provider}")

    def ask_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, retries: int = 1) -> Any:
        """Ask for JSON. If the answer is not valid JSON, ask once more with a stricter reminder."""
        last_err = None
        for attempt in range(retries + 1):
            prompt = user_prompt if attempt == 0 else user_prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No explanation, no markdown."
            raw = self.generate_completion(system_prompt, prompt, temperature)
            try:
                return clean_json_response(raw)
            except ValueError as err:
                last_err = err
                if self.use_cache:     # do not keep a bad answer in the cache
                    self._load_cache().pop(hashlib.sha256(
                        f"{self.provider}|{self.model}|{temperature}|{system_prompt}|{prompt}".encode()).hexdigest(), None)
        raise last_err


# ------------------------------------------------------------------ AI step 1: claim extraction
def _paper_text_for_prompt(paper: Dict[str, Any], limit: int = 7000) -> str:
    """Whole sentences only: each section is cut at a sentence end, never in the middle of a word."""
    parts, used = [], 0
    for name in ("Abstract", "Results", "Discussion & Limitations", "Conclusion", "Full Text"):
        txt = paper.get("sections", {}).get(name, "")
        if not txt:
            continue
        room = max(0, limit - used)
        if room < 200:
            break
        if len(txt) > room:
            cut = txt[:room]
            txt = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
        parts.append(f"[{name}]\n{txt}")
        used += len(txt)
    return "\n\n".join(parts)


def extract_claims_with_llm(client: LLMClient, papers: List[Dict[str, Any]]):
    """Returns (claims, notes). One call per paper. Claims whose quote is not found in the paper are dropped by the caller."""
    allowed = {f: ", ".join(vocab.allowed_values(f)) for f in vocab.FIELDS}
    system_prompt = (
        "You are a careful research assistant. Read the paper text and extract the claims that state a RESULT "
        "(not background and not future work). Return ONLY a JSON list. Each item has the keys: "
        "statement, quote, section, intervention, outcome, population, method, dataset, direction.\n"
        "Rules:\n- quote must be copied EXACTLY from the text, one sentence, no changes.\n"
        "- direction is one of: positive (effect/improvement), none (no effect), negative (harm/decline), reported (a performance value such as sensitivity or RMSE is stated, no comparison).\n"
        f"- intervention is one of: {allowed['intervention']}\n- outcome is one of: {allowed['outcome']}\n"
        f"- population is one of: {allowed['population']}\n- method is one of: {allowed['method']}\n"
        f"- dataset is one of: {allowed['dataset']}\n"
        "- If a field is not stated, use \"other\". Never guess.\n- If there are no result claims, return []."
    )
    claims, notes = [], []
    for idx, p in enumerate(papers):
        pid = p.get("paper_id", f"P{idx + 1}")
        title = p.get("title", pid)
        data = client.ask_json(system_prompt, f"PAPER TEXT:\n{_paper_text_for_prompt(p)}", temperature=0.0)
        if not isinstance(data, list):
            notes.append(f"{pid}: the model did not return a list.")
            continue
        defaults = {f: vocab.best_label(f, p.get("full_text", ""), prefer="most") for f in vocab.FIELDS}
        for n, item in enumerate(data, start=1):
            if not isinstance(item, dict) or not item.get("quote"):
                continue
            quote = str(item["quote"]).strip()
            rule_tags = tag_claim(quote, defaults)
            tags = {}
            for f in vocab.FIELDS:
                v = vocab.normalize(f, item.get(f, ""))
                tags[f] = v if v != vocab.OTHER else rule_tags[f]
            direction = str(item.get("direction", "")).lower()
            if direction not in vocab.DIRECTIONS:
                direction = vocab.detect_direction(quote) or "positive"
            claims.append({
                "id": f"{pid}_C{n}", "paper_id": pid, "paper_title": title,
                "topic": vocab.label_text(tags["outcome"]).capitalize() if tags["outcome"] != vocab.OTHER else "General finding",
                "statement": str(item.get("statement") or quote), "quote": quote,
                "section": str(item.get("section") or "Unknown"), "confidence": 0.8,
                "methodology": "AI extraction (quote checked by code)",
                **tags, "direction": direction, "extracted_by": "llm",
            })
    return claims, notes


# ------------------------------------------------------------------ AI steps 2 and 3
def generate_hypothesis_with_llm(client: LLMClient, base: Dict[str, Any], evidence: List[Dict[str, Any]],
                                 passages: List[Dict[str, Any]], critic_notes: str = "") -> Dict[str, Any]:
    """The Generator. `base` is the rule-based hypothesis for the same finding; the AI rewrites it using only the evidence."""
    system_prompt = (
        "You are a research assistant. Write ONE testable research hypothesis for the FINDING using ONLY the "
        "EVIDENCE quotes. Do not invent facts, numbers, mechanisms or papers. "
        "Return ONLY JSON with keys: title, formulation, rationale, proposed_methodology, falsification_criteria."
    )
    ev = "\n".join(f"- [{e['paper_id']}] {e['quote']}" for e in evidence[:8])
    ps = "\n".join(f"- [{p['paper_id']}] {p['text']}" for p in passages)
    fix = f"\nThe reviewer found these problems, fix them: {critic_notes}" if critic_notes else ""
    user_prompt = (f"FINDING: {base['title']} ({base['finding_type']})\nIDEA TO REFINE: {base['formulation']}\n"
                   f"EVIDENCE:\n{ev}\nRELATED PASSAGES:\n{ps}{fix}")
    data = client.ask_json(system_prompt, user_prompt, temperature=0.2)
    out = dict(base)
    if isinstance(data, dict):
        for k in ("title", "formulation", "rationale", "proposed_methodology", "falsification_criteria"):
            if data.get(k):
                out[k] = str(data[k])
        out["generated_by"] = "llm"
    return out


def critic_with_llm(client: LLMClient, hypothesis: Dict[str, Any], quotes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The Critic: three true/false answers, temperature 0. The label is NOT decided here."""
    system_prompt = (
        "You are a strict reviewer. Check the HYPOTHESIS against the QUOTES only. Answer separately: "
        "claim_match (do the quotes support the main claim?), population_match (same population?), "
        "method_match (same method?). If unsure answer false. "
        "Return ONLY JSON: {\"claim_match\": true/false, \"population_match\": true/false, "
        "\"method_match\": true/false, \"notes\": \"one or two short sentences\"}"
    )
    q = "\n".join(f"- [{x['paper_id']}] {x['quote']}" for x in quotes[:8])
    data = client.ask_json(system_prompt, f"HYPOTHESIS: {hypothesis['formulation']}\nQUOTES:\n{q}", temperature=0.0)
    if not isinstance(data, dict):
        return {"claim_match": False, "population_match": False, "method_match": False, "notes": "Critic gave no valid answer."}
    return {"claim_match": bool(data.get("claim_match")), "population_match": bool(data.get("population_match")),
            "method_match": bool(data.get("method_match")), "notes": str(data.get("notes", ""))}
