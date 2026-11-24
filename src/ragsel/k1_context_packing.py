"""Pack saved LightRAG evidence and audit original source spans without model calls."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import zip_longest

from .k1_blocks import DeliveredContext, DeliveredPiece, render_context, token_count


def json_sections(text):
    sections = []
    label = None
    rows = None
    for line in text.splitlines(keepends=True):
        if line.strip() == "```json":
            rows = []
        elif rows is not None and line.strip() == "```":
            sections.append((label, rows))
            rows = None
        elif rows is not None:
            if line.strip():
                rows.append(line)
        elif line.strip() and line.strip() != "```":
            label = line.strip()
    return sections


def pack_context(blocks, titles, native, tokenizer, maximum, graph_budget):
    if not 0 < graph_budget < maximum:
        raise ValueError("graph budget must leave room for source text")
    source = render_context(blocks, titles, tokenizer, maximum - graph_budget)
    sections = [(label, rows) for label, rows in json_sections(native)
                if label.startswith("Knowledge Graph Data")]
    kept = [[] for _ in sections]

    def render():
        return "\n".join(f"{label}\n```json\n{''.join(rows)}```\n"
                         for (label, _), rows in zip(sections, kept, strict=True) if rows)

    for row_group in zip_longest(*(rows for _, rows in sections)):
        for i, row in enumerate(row_group):
            if row is None:
                continue
            json.loads(row)
            kept[i].append(row)
            graph = render()
            if token_count(tokenizer, graph) > graph_budget or token_count(tokenizer, source.text + graph) > maximum:
                kept[i].pop()
    text = source.text + render()
    return DeliveredContext(text, source.pieces, token_count(tokenizer, text))


def original_pieces(native, delivered, blocks):
    """Decode only complete characters visible inside the original JSON chunk rows."""
    if not native.startswith(delivered):
        raise ValueError("original delivered context must be a native prefix")
    remaining = iter(blocks)
    pieces = []
    for label, rows in json_sections(native):
        if not label.startswith("Document Chunks ("):
            continue
        offset = native.index(label) + len(label)
        for row in rows:
            start = native.index(row, offset)
            offset = start + len(row)
            data = json.loads(row)
            block = next(remaining, None)
            if block is None or data["content"] != block.text:
                raise ValueError("native context rows do not match the ranked source blocks")
            encoded = json.dumps(block.text, ensure_ascii=False)
            # Locate the value after its key, including the JSON string's opening quote.
            key = '"content": '
            content_start = row.index(key) + len(key)
            if row[content_start:content_start + len(encoded)] != encoded:
                raise ValueError("native source serialization differs")
            visible = min(len(encoded) - 2, len(delivered) - start - content_start - 1)
            if visible <= 0:
                continue
            size = len(block.text)
            if visible < len(encoded) - 2:
                size = 0
                used = 0
                for char in block.text:
                    used += len(json.dumps(char, ensure_ascii=False)) - 2
                    if used > visible:
                        break
                    size += 1
            if size:
                pieces.append(DeliveredPiece(block.block_id, block.source_id, block.start, block.start + size, block.text[:size]))
    if next(remaining, None) is not None:
        raise ValueError("ranked source block absent from the native context")
    return tuple(pieces)


def coverage(family, gold, pieces):
    if family != "cuad":
        support = set(gold["support_ids"])
        return len(support & {p.source_id for p in pieces}), len(support)
    locations = defaultdict(list)
    for piece in pieces:
        locations[piece.source_id].append((piece.start, piece.end))
    covered = 0
    for span in gold["spans"]:
        end = span["start"]
        for start, stop in sorted(locations[span["source_id"]]):
            if start > end:
                break
            end = max(end, stop)
        covered += end >= span["end"]
    return covered, len(gold["spans"])
