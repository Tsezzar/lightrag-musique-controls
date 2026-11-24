"""Read-only historical exclusions and pre-outcome source eligibility."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .k1_data import DataPool, normalize_text, text_digest


@dataclass(frozen=True)
class HistoricalExclusions:
    parent_titles: frozenset[str]
    text_hashes: frozenset[str]
    question_ids: dict[str, frozenset[str]]


def historical_exclusions(databases: list[Path], corpus_jsonl: list[Path]) -> HistoricalExclusions:
    titles, texts, questions = set(), set(), {}
    for path in databases:
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for title, text in con.execute("SELECT title,text FROM documents"):
                if title:
                    titles.add(normalize_text(title))
                texts.add(text_digest(text))
                # Existing CorpusBuilder prepends the exact title plus '. ' to the body.
                prefix = title + ". " if title else ""
                if prefix and text.startswith(prefix):
                    texts.add(text_digest(text[len(prefix):]))
            for family, qid in con.execute("SELECT dataset,qid FROM queries"):
                questions.setdefault(family, set()).add(str(qid))
        finally:
            con.close()
    for path in corpus_jsonl:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                titles.add(normalize_text(row["title"]))
                texts.add(text_digest(row["context"]))
    return HistoricalExclusions(frozenset(titles), frozenset(texts),
                                 {family: frozenset(ids) for family, ids in questions.items()})


def eligible_pool(pool: DataPool, history: HistoricalExclusions) -> DataPool:
    documents = tuple(d for d in pool.documents if d.text.strip()
                      and normalize_text(d.title) not in history.parent_titles
                      and text_digest(d.text) not in history.text_hashes)
    available = {d.source_id for d in documents}
    questions = tuple(q for q in pool.questions
                      if q.qid not in history.question_ids.get(q.family, ())
                      and pool.references[q.qid].support_ids
                      and set(pool.references[q.qid].support_ids) <= available)
    return DataPool(documents, questions, {q.qid: pool.references[q.qid] for q in questions})
