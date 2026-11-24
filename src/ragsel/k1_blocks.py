"""Exact source slices and measured context budgets; no text decoding or gold access."""
from __future__ import annotations

from dataclasses import dataclass

from .k1_data import SourceDocument
from .k1_protocol import config_fingerprint


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    source_id: str
    parent_id: str
    start: int
    end: int
    text: str
    tokens: int


@dataclass(frozen=True)
class DeliveredPiece:
    block_id: str
    source_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class DeliveredContext:
    text: str
    pieces: tuple[DeliveredPiece, ...]
    tokens: int


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def prefix_length(tokenizer, text: str, maximum_tokens: int) -> int:
    if type(maximum_tokens) is not int or maximum_tokens < 0:
        raise ValueError("nonnegative integer token budget required")
    if not text or maximum_tokens == 0:
        return 0
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= maximum_tokens:
        return len(text)
    # Byte-level tokens may share the same Unicode character offset. Slice the
    # original string and re-encode, rather than decoding incomplete UTF-8 bytes.
    end = encoded.offsets[maximum_tokens - 1][1]
    while end > 0 and token_count(tokenizer, text[:end]) > maximum_tokens:
        end -= 1
    return end


def split_document(document: SourceDocument, tokenizer, maximum_tokens: int) -> tuple[TextBlock, ...]:
    if type(maximum_tokens) is not int or maximum_tokens <= 0:
        raise ValueError("positive integer block budget required")
    blocks, start = [], 0
    while start < len(document.text):
        size = prefix_length(tokenizer, document.text[start:], maximum_tokens)
        if size == 0:
            raise ValueError("block budget cannot hold one original character")
        end = start + size
        text = document.text[start:end]
        identity = config_fingerprint([document.source_id, document.text, start, end])
        blocks.append(TextBlock(identity, document.source_id, document.parent_id, start, end,
                                text, token_count(tokenizer, text)))
        start = end
    return tuple(blocks)


def render_context(blocks: tuple[TextBlock, ...], titles: dict[str, str], tokenizer,
                   maximum_tokens: int) -> DeliveredContext:
    parts, positions, offset = [], [], 0
    for block in blocks:
        header = f"[{block.block_id}] {titles[block.source_id]}\n"
        part = header + block.text + "\n\n"
        positions.append((offset + len(header), block))
        parts.append(part)
        offset += len(part)
    text = "".join(parts)
    end = prefix_length(tokenizer, text, maximum_tokens)
    pieces = []
    for position, block in positions:
        length = min(len(block.text), max(0, end - position))
        if length:
            pieces.append(DeliveredPiece(block.block_id, block.source_id, block.start,
                                          block.start + length, block.text[:length]))
    delivered = text[:end]
    return DeliveredContext(delivered, tuple(pieces), token_count(tokenizer, delivered))
