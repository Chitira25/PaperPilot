"""
Controlled vocabulary for PaperPilot.

Papers describe the same thing in different words ("stroke survivors", "post-stroke patients").
To compare claims across papers every claim is tagged with ONE fixed label per field.
All modules import this file, so everybody uses the same words.

You can EXTEND the vocabulary without touching code: create data/vocab.json like
    {"population": {"spinal_cord_injury": ["spinal cord injury", "sci patients"]}}
New labels are merged with the defaults below.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional

FIELDS = ["intervention", "outcome", "population", "method", "dataset"]
DIRECTIONS = ["positive", "none", "negative", "reported"]
VERIFICATION_LABELS = ["VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"]
OTHER = "other"

# label -> list of keywords (lower case). Order inside a field does not matter.
DEFAULT_VOCAB: Dict[str, Dict[str, List[str]]] = {
    "intervention": {
        "grip_rehab_protocol": ["grip rehabilitation", "grip rehab", "grip training", "hand rehabilitation protocol", "rehabilitation protocol"],
        "graph_rag": ["graph rag", "graphrag", "knowledge graph", "knowledge-graph-augmented", "graph-augmented"],
        "dense_retrieval": ["dense retrieval", "dense vector", "vector search", "vector retrieval"],
        "hybrid_retrieval": ["hybrid retrieval", "hybrid bm25", "adaptive hybrid"],
        "long_context": ["long-context", "long context"],
        "self_consistency": ["self-consistency", "chain-of-thought"],
        "chunking": ["chunk size", "hierarchical chunking", "chunking"],
    },
    "outcome": {
        "hand_function": ["hand function", "fugl-meyer", "dexterity", "motor function"],
        "grip_strength": ["grip strength"],
        "muscle_activation": ["muscle activation", "emg amplitude", "muscle activity"],
        "hallucination": ["hallucinat", "factuality", "factual accuracy"],
        "latency": ["latency", "inference time", "response time"],
        "multi_hop_reasoning": ["multi-hop", "multi hop"],
        "answer_accuracy": ["answer accuracy", "f1", "accuracy"],
    },
    "population": {
        "stroke": ["post-stroke"],
        "parkinsons": ["parkinson"],
        "healthy_adults": ["healthy adults", "healthy volunteers", "healthy participants", "healthy controls"],
    },
    "method": {
        "continuous_sEMG": ["continuous semg", "continuous surface emg", "continuous emg", "continuous surface electromyography"],
        "single_point_sensor": ["single-point sensor", "single point sensor", "single-point measurement", "single point measurement"],
        "grip_strength_test": ["dynamometer", "grip strength test"],
        "hybrid_bm25_dense": ["bm25-dense", "bm25 dense", "hybrid bm25"],
    },
    "dataset": {
        "dataset_A": ["dataset a"],
        "dataset_B": ["dataset b"],
        "mimic_iv": ["mimic-iv", "mimic iv"],
        "pubmed_qa": ["pubmedqa", "pubmed qa"],
        "bioasq": ["bioasq"],
    },
}


# ---- biomedical instrumentation and health domains (added in v2) ----
_BIO = {
    "population": {
        "stroke": ["re:(?<![a-z0-9])stroke(?![\\s-]*volume)", "post-stroke", "hemiparetic", "hemiparesis"],
        "diabetes": ["diabetes", "diabetic"],
        "cardiovascular": ["heart failure", "cardiac patients", "myocardial infarction", "atrial fibrillation",
                           "arrhythmia", "cardiovascular", "coronary"],
        "respiratory": ["copd", "asthma", "chronic obstructive", "respiratory patients", "sleep apnea", "sleep apnoea"],
        "icu_patients": ["intensive care", "critically ill", "icu patients"],
        "elderly": ["older adults", "elderly", "geriatric"],
        "pediatric": ["children", "pediatric", "paediatric", "neonat", "infants"],
        "cancer": ["cancer", "tumor", "tumour", "oncology"],
    },
    "method": {
        "ecg": ["ecg", "electrocardio"],
        "eeg": ["eeg", "electroencephalo"],
        "emg": ["emg", "electromyograph"],
        "ppg": ["ppg", "photoplethysmograph"],
        "ultrasound": ["ultrasound", "ultrasonograph", "sonograph"],
        "mri": ["mri", "magnetic resonance"],
        "ct_imaging": ["computed tomography", "ct scan"],
        "pulse_oximetry": ["pulse oximet", "spo2"],
        "wearable_sensor": ["wearable"],
        "imu_sensor": ["imu", "inertial measurement", "accelerometer"],
        "biosensor": ["biosensor"],
        "glucose_monitor": ["glucose monitor", "continuous glucose", "cgm"],
        "blood_pressure_monitor": ["blood pressure monitor", "sphygmomanometer", "cuffless"],
        "thermography": ["thermography", "infrared thermometer"],
        "ventilator": ["ventilator", "mechanical ventilation"],
        "infusion_pump": ["infusion pump"],
        "implantable_device": ["pacemaker", "implantable", "neurostimulator"],
        "endoscopy": ["endoscop", "laparoscop"],
        "machine_learning": ["machine learning", "deep learning", "neural network", "random forest"],
    },
    "outcome": {
        "diagnostic_accuracy": ["diagnostic accuracy", "sensitivity", "specificity", "auc", "classification accuracy"],
        "measurement_accuracy": ["measurement accuracy", "measurement error", "rmse", "mean absolute error",
                                 "bland-altman"],
        "signal_quality": ["signal quality", "signal-to-noise", "motion artifact", "motion artefact"],
        "heart_rate": ["heart rate", "heart rate variability", "hrv"],
        "blood_pressure": ["blood pressure"],
        "glycemic_control": ["hba1c", "glycemic", "glycaemic", "glucose level"],
        "mortality": ["mortality", "survival"],
        "pain": ["pain"],
    },
    "intervention": {
        "telemonitoring": ["remote monitoring", "telemonitoring", "telemedicine", "telehealth"],
        "wearable_monitoring": ["wearable monitoring", "wearable device"],
        "robot_assisted_therapy": ["robot-assisted", "exoskeleton"],
        "mirror_therapy": ["mirror therapy", "mirror visual feedback"],
    },
}
for _field, _labels in _BIO.items():
    for _label, _kws in _labels.items():
        if _label in DEFAULT_VOCAB[_field]:
            DEFAULT_VOCAB[_field][_label] = list(dict.fromkeys(DEFAULT_VOCAB[_field][_label] + _kws))
        else:
            DEFAULT_VOCAB[_field][_label] = _kws

# ---- direction cues (checked in this order: none -> negative -> positive) ----
NONE_CUES = [
    "no significant", "not significant", "did not improve", "did not", "no improvement", "no difference",
    "unchanged", "failed to", "no clear", "not improve", "no measurable", "showed no", "does not", "do not",
    "cannot", "no effect", "no change",
]
NEGATIVE_CUES = ["worsen", "worse ", "degrad", "declin", "deteriorat", "impair", "harm"]
POSITIVE_CUES = [
    "improv", "increas", "outperform", "higher", "gain", "enhanc", "better than", "better results",
    "better outcome", "reduc", "lower", "matches or exceeds", "exceed", "achiev", "significantly",
    "significant improvement", "significant increase", "significant gain",
]
# general "this sentence states a result" cues used by the claim extractor
RESULT_CUES = NONE_CUES + NEGATIVE_CUES + POSITIVE_CUES + [
    "we show", "we demonstrate", "we find", "we observe", "results show", "results indicate", "results demonstrate",
    "found that", "reveal",
]

_LABEL_TEXT = {
    "continuous_sEMG": "continuous sEMG",
    "single_point_sensor": "single-point sensor",
    "grip_rehab_protocol": "the grip rehabilitation protocol",
    "parkinsons": "Parkinson's patients",
    "stroke": "stroke patients",
    "healthy_adults": "healthy adults",
}

_compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}
_vocab: Dict[str, Dict[str, List[str]]] = {}


def _kw_regex(kw: str) -> re.Pattern:
    if kw.startswith("re:"):                       # advanced: a raw regular expression
        return re.compile(kw[3:], re.IGNORECASE)
    parts = re.split(r"[-\s]+", kw.strip().lower())
    body = r"[-\s]+".join(re.escape(p) for p in parts)
    # keywords ending in a partial stem (e.g. "hallucinat", "improv") may continue; others need a word end
    return re.compile(r"(?<![a-z0-9])" + body, re.IGNORECASE)


def load_vocab(extra_path: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """(Re)load the default vocabulary and merge data/vocab.json if it exists."""
    global _vocab, _compiled
    vocab = {f: {k: list(v) for k, v in d.items()} for f, d in DEFAULT_VOCAB.items()}
    if extra_path is None:
        extra_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vocab.json")
    if extra_path and os.path.exists(extra_path):
        try:
            with open(extra_path, "r", encoding="utf-8") as fh:
                extra = json.load(fh)
            for field, labels in extra.items():
                if field in vocab and isinstance(labels, dict):
                    for label, kws in labels.items():
                        vocab[field].setdefault(label, [])
                        vocab[field][label].extend(k for k in kws if k not in vocab[field][label])
        except (OSError, ValueError):
            pass
    _vocab = vocab
    _compiled = {f: {lab: [_kw_regex(k) for k in kws] for lab, kws in d.items()} for f, d in vocab.items()}
    return vocab


load_vocab()


def allowed_values(field: str) -> List[str]:
    return list(_vocab.get(field, {}).keys()) + [OTHER]


def find_labels(field: str, text: str) -> List[Tuple[str, int, int]]:
    """Return [(label, hit_count, first_position)] for every label of `field` found in text."""
    hits = []
    for label, patterns in _compiled.get(field, {}).items():
        count, first = 0, None
        for p in patterns:
            for m in p.finditer(text):
                count += 1
                first = m.start() if first is None else min(first, m.start())
        if count:
            hits.append((label, count, first))
    return hits


def best_label(field: str, text: str, prefer: str = "first") -> str:
    """Pick one label for a field. prefer='first' = earliest in text, 'most' = most frequent."""
    hits = find_labels(field, text)
    if not hits:
        return OTHER
    if prefer == "most":
        hits.sort(key=lambda h: (-h[1], h[2]))
    else:
        hits.sort(key=lambda h: (h[2], -h[1]))
    return hits[0][0]


_METRIC_RE = re.compile(
    r"(?:sensitivity|specificity|accuracy|\bauc\b|\brmse\b|\bmae\b|mean absolute error|\bf1\b|precision|recall|kappa|"
    r"correlation|error)[^.\d]{0,45}\d", re.IGNORECASE)


def detect_direction(sentence: str) -> Optional[str]:
    """positive = beneficial/increasing effect reported, none = no effect, negative = harm/decline,
    reported = a performance number is stated (sensitivity, RMSE ...) without a comparison.
    Returns None when the sentence has no direction cue (then it is not treated as a result)."""
    s = " " + sentence.lower().replace("increasingly", "") + " "
    if any(c in s for c in NONE_CUES):
        return "none"
    if any(c in s for c in NEGATIVE_CUES):
        return "negative"
    if any(c in s for c in POSITIVE_CUES):
        return "positive"
    if _METRIC_RE.search(s):
        return "reported"           # a performance value is stated (typical for device papers), no comparison
    return None


def count_result_cues(sentence: str) -> int:
    s = sentence.lower()
    return sum(1 for c in RESULT_CUES if c in s)


def label_text(value: str) -> str:
    """Human-readable form of a vocabulary label."""
    if value in _LABEL_TEXT:
        return _LABEL_TEXT[value]
    return value.replace("_", " ")


def normalize(field: str, value: str) -> str:
    """Map a free-text value (for example from an LLM) to an allowed label, else 'other'."""
    if not value:
        return OTHER
    v = str(value).strip()
    if v in _vocab.get(field, {}):
        return v
    low = v.lower().replace(" ", "_")
    for label in _vocab.get(field, {}):
        if label.lower() == low:
            return label
    found = best_label(field, v)
    return found
