"""Verified public corpus boundary for retrieval and audit; no gold-file reader."""
from __future__ import annotations

import json
from pathlib import Path

from .k1_assembly import StudyCorpus
from .k1_blocks import TextBlock, token_count
from .k1_data import PublicQuestion, SourceDocument
from .k1_protocol import file_sha256


def load_public_corpus(path: Path, expected_sha256: str, tokenizer, maximum_tokens: int) -> StudyCorpus:
    if file_sha256(path) != expected_sha256:
        raise ValueError("public corpus hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"corpus_id", "family", "split", "documents", "questions", "blocks"}:
        raise ValueError("unexpected public corpus fields")
    blocks = tuple(TextBlock(**row) for row in payload["blocks"])
    if not blocks or len({block.block_id for block in blocks}) != len(blocks):
        raise ValueError("nonempty unique public blocks required")
    documents = []
    for metadata in payload["documents"]:
        if set(metadata) != {"family", "source_id", "parent_id", "title"} or metadata["family"] != payload["family"]:
            raise ValueError("invalid public document metadata")
        selected = [b for b in blocks if b.source_id == metadata["source_id"]]
        if not selected:
            raise ValueError("public document has no retained source blocks")
        end = 0
        for block in selected:
            if (block.start != end or block.end != end + len(block.text) or block.parent_id != metadata["parent_id"]
                    or not 0 < block.tokens <= maximum_tokens or token_count(tokenizer, block.text) != block.tokens):
                raise ValueError("invalid public source offsets or token counts")
            end = block.end
        documents.append(SourceDocument(**metadata, text="".join(b.text for b in selected)))
    sources = {document.source_id for document in documents}
    if len(sources) != len(documents) or any(b.source_id not in sources for b in blocks):
        raise ValueError("ambiguous or missing public source identity")
    questions = tuple(PublicQuestion(**row) for row in payload["questions"])
    if (not questions or len({q.qid for q in questions}) != len(questions)
            or any(q.family != payload["family"] for q in questions)):
        raise ValueError("invalid public questions")
    return StudyCorpus(payload["corpus_id"], payload["family"], payload["split"], tuple(documents), questions, blocks)
