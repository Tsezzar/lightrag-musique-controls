"""Answer scoring: deterministic rules first; the LLM judge only for open answers that EM rejects."""
from __future__ import annotations

import collections
import json
import re
import string

from .types import Gold

JUDGE_SCHEMA = {"type": "object", "properties": {"equivalent": {"type": "boolean"}}, "required": ["equivalent"]}
LABELS = ("yes", "no", "maybe")


def normalize(s: str) -> str:
    """SQuAD normalisation, as in run_generation.py:11-13."""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g:
        return float(p == g)
    return 2 * common / (len(p) + len(g)) if common else 0.0


CITATION_ONLY = re.compile(r"^\s*\[\d+\](\s*,?\s*\[\d+\])*\s*\.?\s*$")


def is_unknown(pred: str) -> bool:
    """No answer given: empty, UNKNOWN, or only passage citations like "[2]" (the judge accepted one in the pilot)."""
    return normalize(pred) in ("", "unknown") or bool(CITATION_ONLY.match(pred))


def label_of(pred: str) -> str | None:
    words = normalize(pred).split()
    return words[0] if words and words[0] in LABELS else None


def option_of(pred: str, options: tuple[str, ...]) -> str | None:
    """Map a prediction to an option: by letter, by exact text, else by the unique best token F1."""
    norm = normalize(pred)
    letter = re.fullmatch(r"([a-z])", norm.split(" ")[0]) if norm else None
    if letter and len(norm.split()) == 1 and ord(letter.group(1)) - 97 < len(options):
        return options[ord(letter.group(1)) - 97]
    for o in options:
        if normalize(o) == norm:
            return o
    scores = [f1(pred, o) for o in options]
    best = max(scores, default=0.0)
    if best == 0.0 or scores.count(best) > 1:
        return None
    return options[scores.index(best)]


def rule_score(pred: str, gold: Gold) -> tuple[int, float, bool]:
    """-> (em, f1, needs_judge)."""
    if gold.kind in ("yesno", "binary"):
        em = int(label_of(pred) == gold.answers[0])
        return em, float(em), False
    if gold.kind == "option":
        return int(option_of(pred, gold.options) == gold.answers[0]), f1(pred, gold.answers[0]), False
    em = int(any(normalize(pred) == normalize(a) for a in gold.answers))
    best_f1 = max(f1(pred, a) for a in gold.answers)
    return em, best_f1, (not em and not is_unknown(pred))


def judge_messages(question: str, gold: Gold, pred: str) -> list[dict]:
    answers = " | ".join(gold.answers)
    return [dict(role="user", content=(
        f"Question: {question}\nGold answers (any one is acceptable): {answers}\nPredicted answer: {pred}\n"
        "Does the predicted answer give the same answer as one of the gold answers? Extra correct detail is "
        "fine; a different, partial or contradictory answer is not."))]


def parse_judge(content: str) -> bool | None:
    try:
        value = json.loads(content).get("equivalent")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, bool) else None
