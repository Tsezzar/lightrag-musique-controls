"""Prospective corpus curation; references are used here, never exposed to RAG/audit input."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from .k1_blocks import TextBlock, split_document
from .k1_data import DataPool, PublicQuestion, SourceDocument, text_digest
from .k1_protocol import Provenance, assert_independent, config_fingerprint


@dataclass(frozen=True)
class StudyCorpus:
    corpus_id: str
    family: str
    split: str
    documents: tuple[SourceDocument, ...]
    questions: tuple[PublicQuestion, ...]
    blocks: tuple[TextBlock, ...]


def corpus_provenance(corpus: StudyCorpus) -> tuple[Provenance, ...]:
    documents = tuple(Provenance(corpus.family, corpus.corpus_id, corpus.split, doc.source_id,
                                  doc.parent_id, text_digest(doc.text)) for doc in corpus.documents)
    blocks = tuple(Provenance(corpus.family, corpus.corpus_id, corpus.split, block.block_id,
                               block.parent_id, text_digest(block.text)) for block in corpus.blocks if block.text.strip())
    return documents + blocks


def assemble_family(pool: DataPool, distractors: tuple[SourceDocument, ...], tokenizer, *,
                    split_counts: dict[str, int], questions_per_corpus: int, minimum_blocks: int,
                    block_tokens: int, seed: int, answerable_questions: int | None = None,
                    chunk_observer: Callable[[str, float], None] | None = None) -> tuple[StudyCorpus, ...]:
    if (not pool.documents or questions_per_corpus < 1 or minimum_blocks < 1
            or set(split_counts) != {"train", "calibration", "test"}
            or any(type(n) is not int or n < 0 for n in split_counts.values())
            or (answerable_questions is not None and not 0 <= answerable_questions <= questions_per_corpus)):
        raise ValueError("invalid corpus assembly settings")
    family = pool.documents[0].family
    documents = {}
    for doc in (*pool.documents, *distractors):
        if doc.family != family or not doc.text.strip():
            raise ValueError("nonempty documents from one source family required")
        if doc.source_id in documents and documents[doc.source_id] != doc:
            raise ValueError("duplicate source identity with different records")
        documents[doc.source_id] = doc
    for question in pool.questions:
        reference = pool.references[question.qid]
        if question.family != family or not reference.support_ids or not set(reference.support_ids) <= documents.keys():
            raise ValueError("question requires unavailable support")
    derived_seed = int(config_fingerprint([seed, family])[:16], 16)
    rng = random.Random(derived_seed)
    ordered_docs, ordered_questions = list(documents.values()), list(pool.questions)
    rng.shuffle(ordered_docs)
    rng.shuffle(ordered_questions)
    splits = [split for split in ("train", "calibration", "test") for _ in range(split_counts[split])]
    rng.shuffle(splits)
    chunks, identities, occupied, used_questions, corpora = {}, {}, set(), set(), []

    def prepare(doc: SourceDocument):
        if doc.source_id not in chunks:
            start = time.perf_counter()
            chunks[doc.source_id] = split_document(doc, tokenizer, block_tokens)
            identities[doc.source_id] = {("source", family, doc.source_id), ("parent", family, doc.parent_id),
                                         ("text", text_digest(doc.text))}
            identities[doc.source_id].update(("text", text_digest(b.text)) for b in chunks[doc.source_id] if b.text.strip())
            if chunk_observer is not None:
                chunk_observer(doc.source_id, time.perf_counter() - start)
        return chunks[doc.source_id]

    def available(doc: SourceDocument) -> bool:
        prepare(doc)
        return not occupied.intersection(identities[doc.source_id])

    for index, split in enumerate(splits):
        selected: dict[str, SourceDocument] = {}
        questions: list[PublicQuestion] = []
        if answerable_questions is not None:
            # CUAD: collect complete contracts first, so twelve questions do not
            # inadvertently require twelve long contracts in every corpus.
            for doc in ordered_docs:
                if not available(doc):
                    continue
                selected[doc.source_id] = doc
                candidates = [q for q in ordered_questions if q.qid not in used_questions
                              and set(pool.references[q.qid].support_ids) <= selected.keys()]
                positive = [q for q in candidates if pool.references[q.qid].answers]
                negative = [q for q in candidates if not pool.references[q.qid].answers]
                if (len(positive) >= answerable_questions
                        and len(negative) >= questions_per_corpus - answerable_questions
                        and sum(len(chunks[d]) for d in selected) >= minimum_blocks):
                    questions = rng.sample(positive, answerable_questions) + rng.sample(
                        negative, questions_per_corpus - answerable_questions)
                    rng.shuffle(questions)
                    break
        else:
            for question in ordered_questions:
                if question.qid in used_questions:
                    continue
                required = [documents[source_id] for source_id in pool.references[question.qid].support_ids]
                if not all(available(doc) for doc in required):
                    continue
                selected.update((doc.source_id, doc) for doc in required)
                questions.append(question)
                if len(questions) == questions_per_corpus:
                    break
            for doc in ordered_docs:
                if sum(len(chunks[d]) for d in selected) >= minimum_blocks:
                    break
                if doc.source_id not in selected and available(doc):
                    selected[doc.source_id] = doc
        blocks = tuple(block for doc in selected.values() for block in chunks[doc.source_id])
        if len(questions) != questions_per_corpus or len(blocks) < minimum_blocks:
            raise ValueError(f"insufficient independent sources for {family} corpus {index}: "
                             f"{len(questions)} questions, {len(blocks)} blocks")
        corpus = StudyCorpus(f"{family}-{index:03d}", family, split, tuple(selected.values()), tuple(questions), blocks)
        for doc in selected.values():
            occupied.update(identities[doc.source_id])
        used_questions.update(q.qid for q in questions)
        corpora.append(corpus)
    assert_independent(row for corpus in corpora for row in corpus_provenance(corpus))
    return tuple(corpora)
