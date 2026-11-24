"""Deterministic prospective scoring; no judge verdict is promoted to ground truth.

MuSiQue: SQuAD answer normalization/F1, max over aliases (reuse evaluate.py).
CUAD: official Jaccard >= .5 and Parties substring matching at a fixed output
point, averaged per question. This adaptation is not official ranked AUPR.
Matching source: The-Atticus-Project/cuad, commit 67faa0e, evaluate.py:53-123.
"""
from __future__ import annotations

import json

from .evaluate import LABELS, f1, normalize
from .k1_data import Reference

CUAD_SCHEMA = {"type": "object", "properties": {"spans": {"type": "array", "items": {"type": "string"}}},
               "required": ["spans"], "additionalProperties": False}
PUBMED_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": list(LABELS)}},
                 "required": ["label"], "additionalProperties": False}
CUAD_IOU_THRESHOLD = .5  # Official matching threshold; fixed benchmark definition.


def _object(content: str):
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def _cuad_match(gold: str, prediction: str, parties: bool) -> bool:
    def words(text):
        for punctuation in (".", ",", ";", ":"):
            text = text.replace(punctuation, "")
        return set(text.lower().replace("/", " ").split(" "))
    g, p = words(gold), words(prediction)
    return len(g & p) / len(g | p) >= CUAD_IOU_THRESHOLD or (parties and gold in prediction)


def score_answer(family: str, qid: str, content: str, reference: Reference, *, status: str = "complete") -> dict:
    if status != "complete":
        return dict(quality=0., valid=False, failure=status)
    if family == "musique":
        if not reference.answers:
            raise ValueError("MuSiQue requires nonempty reference aliases")
        return dict(quality=max(f1(content, answer) for answer in reference.answers), valid=True,
                    exact_match=float(any(normalize(content) == normalize(answer) for answer in reference.answers)))
    payload = _object(content)
    if family == "pubmedqa":
        if len(reference.answers) != 1 or reference.answers[0] not in LABELS:
            raise ValueError("PubMedQA requires one official class label")
        valid = payload is not None and set(payload) == {"label"} and payload["label"] in LABELS
        return dict(quality=float(valid and payload["label"] == reference.answers[0]), valid=valid)
    if family != "cuad":
        raise ValueError(f"unknown benchmark family: {family}")
    valid = (payload is not None and set(payload) == {"spans"} and isinstance(payload["spans"], list)
             and all(isinstance(span, str) and span.strip() for span in payload["spans"]))
    if not valid:
        return dict(quality=0., valid=False, failure="invalid_span_schema")
    predictions = tuple(dict.fromkeys(payload["spans"]))
    parties = "Parties" in qid  # Official task-specific exception, not a semantic heuristic.
    matched = [any(_cuad_match(answer, pred, parties) for pred in predictions) for answer in reference.answers]
    tp, fn = sum(matched), len(matched) - sum(matched)
    fp = sum(not any(_cuad_match(answer, pred, parties) for answer in reference.answers) for pred in predictions)
    denominator = 2 * tp + fp + fn
    quality = 2 * tp / denominator if denominator else 1.
    return dict(quality=quality, valid=True, tp=tp, fp=fp, fn=fn,
                exact_span_set_match=float(set(predictions) == set(reference.answers)))
