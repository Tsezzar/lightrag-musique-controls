"""Public benchmark parsers. Source text and questions are separated from references."""
from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .k1_protocol import config_fingerprint


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    family: str
    source_id: str
    parent_id: str
    title: str
    text: str


@dataclass(frozen=True)
class PublicQuestion:
    family: str
    qid: str
    text: str
    kind: str


@dataclass(frozen=True)
class GoldSpan:
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Reference:
    answers: tuple[str, ...]
    support_ids: tuple[str, ...]
    spans: tuple[GoldSpan, ...] = ()


@dataclass(frozen=True)
class DataPool:
    documents: tuple[SourceDocument, ...]
    questions: tuple[PublicQuestion, ...]
    references: dict[str, Reference]


def _add_question(questions: list[PublicQuestion], references: dict[str, Reference],
                  question: PublicQuestion, reference: Reference) -> None:
    if question.qid in references:
        raise ValueError(f"duplicate question identifier: {question.qid}")
    questions.append(question)
    references[question.qid] = reference


def parse_cuad(data: dict) -> DataPool:
    docs, questions, references = [], [], {}
    for contract in data["data"]:
        title = contract["title"]
        source_id = "cuad:" + title
        text = "\n\n".join(p["context"] for p in contract["paragraphs"])
        docs.append(SourceDocument("cuad", source_id, normalize_text(title), title, text))
        offset = 0
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers, spans = [], []
                for answer in qa["answers"]:
                    start, value = answer["answer_start"], answer["text"]
                    if type(start) is not int or start < 0 or context[start:start + len(value)] != value:
                        raise ValueError(f"gold offset mismatch: {qa['id']}")
                    answers.append(value)
                    spans.append(GoldSpan(source_id, offset + start, offset + start + len(value)))
                question = PublicQuestion("cuad", qa["id"], f"For contract {title}: {qa['question']}", "extractive")
                _add_question(questions, references, question,
                              Reference(tuple(dict.fromkeys(answers)), (source_id,), tuple(spans)))
            offset += len(context) + 2
    if len({d.source_id for d in docs}) != len(docs):
        raise ValueError("duplicate CUAD contract identifier")
    return DataPool(tuple(docs), tuple(questions), references)


def parse_musique(rows: Iterable[dict]) -> DataPool:
    docs, questions, references = {}, [], {}
    for row in rows:
        if row["answerable"] is not True:
            continue
        support = []
        for paragraph in row["paragraphs"]:
            title, text = paragraph["title"], paragraph["paragraph_text"]
            # Preserve distinct raw records; normalized fingerprints are for duplicate checks only.
            source_id = "musique:" + config_fingerprint([title, text])
            doc = SourceDocument("musique", source_id, normalize_text(title), title, text)
            if source_id in docs and docs[source_id] != doc:
                raise ValueError("source identity collision")
            docs[source_id] = doc
            if paragraph["is_supporting"]:
                support.append(source_id)
        answers = tuple(dict.fromkeys([row["answer"], *row["answer_aliases"]]))
        _add_question(questions, references, PublicQuestion("musique", row["id"], row["question"], "span"),
                      Reference(answers, tuple(dict.fromkeys(support))))
    return DataPool(tuple(docs.values()), tuple(questions), references)


def parse_pubmedqa(rows: Iterable[dict], *, labeled: bool) -> DataPool:
    docs, questions, references = [], [], {}
    for row in rows:
        pmid = str(row["pubid"])
        source_id = "pubmedqa:" + pmid
        # No long_answer, predictions, reasoning labels or final_decision enter source text.
        text = "\n\n".join(row["context"]["contexts"])
        # The task question is not indexed again as a source heading. These
        # abstract-only records have no separately supplied article title.
        docs.append(SourceDocument("pubmedqa", source_id, pmid, pmid, text))
        if labeled:
            decision = row["final_decision"]
            if decision not in ("yes", "no", "maybe"):
                raise ValueError(f"invalid PubMedQA label: {pmid}")
            _add_question(questions, references, PublicQuestion("pubmedqa", pmid, row["question"], "label"),
                          Reference((decision,), (source_id,)))
    if len({d.source_id for d in docs}) != len(docs):
        raise ValueError("duplicate PubMedQA source identifier")
    return DataPool(tuple(docs), tuple(questions), references)
