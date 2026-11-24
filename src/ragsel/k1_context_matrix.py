"""Six-arm context experiment from one immutable native retrieval pool.

No gold readers, inference, learned settings, or legacy-result modifications.
The original package modules remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .k1_blocks import render_context, token_count
from .k1_context_packing import json_sections
from .k1_protocol import config_fingerprint

ARMS = ("V", "BM", "T100", "T75", "TG75", "G100")
VERSION = "six-arm-context-v1"
QWEN_TEMPLATE_SHA256 = "eb4402837c7829a690fa845de4d7f3fd842c2adee476d5341da8a46ea9255175"


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def graph_rows(native):
    """Keep complete native JSON rows and their exact serialized provenance."""
    sections = [(label, rows) for label, rows in json_sections(native)
                if isinstance(label, str) and label.startswith("Knowledge Graph Data")]
    output = []
    for section_index, (label, rows) in enumerate(sections):
        for rank, row in enumerate(rows):
            parsed = json.loads(row)
            if not isinstance(parsed, dict):
                raise ValueError("native graph row must be a JSON object")
            output.append(dict(section=section_index, label=label, rank=rank,
                               text=row, sha256=text_sha(row)))
    # Match the predeclared alternating entity/relationship priority.
    return sorted(output, key=lambda row: (row["rank"], row["section"]))


def render_graph(rows):
    sections = {}
    for row in rows:
        key = row["section"], row["label"]
        sections.setdefault(key, []).append(row["text"])
    return "\n".join(f"{label}\n```json\n{''.join(values)}```\n"
                     for (_, label), values in sorted(sections.items()))


def pack_graph(rows, tokenizer, total_budget, source_text="", graph_budget=None):
    kept = []
    for row in rows:
        candidate = kept + [row]
        graph = render_graph(candidate)
        if graph_budget is not None and token_count(tokenizer, graph) > graph_budget:
            continue
        if token_count(tokenizer, source_text + graph) > total_budget:
            continue
        kept = candidate
    return render_graph(kept), kept


def validate_ranked(corpus, blocks):
    originals = {block.block_id: block for block in corpus.blocks}
    seen = set()
    for block in blocks:
        if block.block_id in seen or originals.get(block.block_id) != block:
            raise ValueError("ranked blocks are duplicate or differ from the public corpus")
        seen.add(block.block_id)


def build_matrix(corpus, question, ranked, native, tokenizer, budget=4000):
    if type(budget) is not int or budget <= 0:
        raise ValueError("positive integer evidence budget required")
    if question not in corpus.questions:
        raise ValueError("question does not belong to corpus")
    if set(ranked) != {"vector", "bm25", "lightrag"}:
        raise ValueError("exactly three precomputed rankings required")
    for blocks in ranked.values():
        validate_ranked(corpus, blocks)
    titles = {doc.source_id: doc.title for doc in corpus.documents}
    source_budget = budget * 3 // 4
    graph_budget = budget - source_budget
    source100 = render_context(ranked["lightrag"], titles, tokenizer, budget)
    source75 = render_context(ranked["lightrag"], titles, tokenizer, source_budget)
    if not source100.text.startswith(source75.text):
        raise ValueError("tokenizer produced nonnested source prefixes; no silent repacking")
    rows = graph_rows(native)
    mixed_graph, mixed_rows = pack_graph(rows, tokenizer, budget, source75.text, graph_budget)
    only_graph, only_rows = pack_graph(rows, tokenizer, budget)
    sources = {
        "V": render_context(ranked["vector"], titles, tokenizer, budget),
        "BM": render_context(ranked["bm25"], titles, tokenizer, budget),
        "T100": source100, "T75": source75, "TG75": source75, "G100": None,
    }
    variants = []
    for arm in ARMS:
        source = sources[arm]
        source_text = source.text if source else ""
        graph = mixed_graph if arm == "TG75" else only_graph if arm == "G100" else ""
        selected_rows = mixed_rows if arm == "TG75" else only_rows if arm == "G100" else []
        context = source_text + graph
        tokens = token_count(tokenizer, context)
        if tokens > budget:
            raise ValueError("context budget exceeded after serialization")
        pieces = [asdict(piece) for piece in source.pieces] if source else []
        documents = {doc.source_id: doc for doc in corpus.documents}
        for piece in pieces:
            if documents[piece["source_id"]].text[piece["start"]:piece["end"]] != piece["text"]:
                raise ValueError("delivered source interval does not match original document")
        variants.append(dict(arm=arm, text=context, text_sha256=text_sha(context),
            source_text=source_text, source_sha256=text_sha(source_text), graph_text=graph,
            context_tokens=tokens, source_tokens=token_count(tokenizer, source_text),
            graph_tokens=token_count(tokenizer, graph), budget=budget,
            graph_quota=.25 if arm == "TG75" else 1. if arm == "G100" else 0.,
            source_limit=source_budget if arm in ("T75", "TG75") else 0 if arm == "G100" else budget,
            graph_limit=graph_budget if arm == "TG75" else budget if arm == "G100" else 0,
            underfill_tokens=budget - tokens, pieces=pieces, graph_rows=selected_rows,
            text_contrast_informative=source100.text != source75.text))
    if variants[3]["source_text"] != variants[4]["source_text"]:
        raise ValueError("T75 identity invariant violated")
    return variants


def permute_without_reclipping(variant, tokenizer):
    swapped = variant["graph_text"] + variant["source_text"]
    count = token_count(tokenizer, swapped)
    if count > variant["budget"]:
        raise ValueError("reordered identical elements exceed budget; do not clip ablation")
    return dict(text=swapped, tokens=count, source_sha256=variant["source_sha256"],
                graph_rows_sha256=config_fingerprint(variant["graph_rows"]))


def render_qwen_chat(messages):
    """The two-message, no-tools branch of the transmitted Qwen Go template.

    Caller must bind its exact template SHA, explicit system and no history. Runtime
    parity still has to be measured on the actual server during development.
    """
    if [message.role for message in messages] != ["system", "user"]:
        raise ValueError("only the frozen system/user template branch is supported")
    return "".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def condition_key(corpus_id, qid, variant, model, seed, protocol_sha256, pool_sha256):
    return config_fingerprint(dict(version=VERSION, corpus_id=corpus_id, qid=qid,
        arm=variant["arm"], budget=variant["budget"], graph_quota=variant["graph_quota"],
        source_limit=variant["source_limit"], graph_limit=variant["graph_limit"],
        context_sha256=variant["text_sha256"], model=model, seed=seed,
        protocol_sha256=protocol_sha256, pool_sha256=pool_sha256))
