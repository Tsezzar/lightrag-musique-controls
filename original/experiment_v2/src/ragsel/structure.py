"""Topology-only intervention over frozen LightRAG extraction; no labels enter this module.

Entities and relation *records* are nodes. Edges connect relation records to their endpoint entities.
Rewiring changes traversal edges, never record text, embeddings or provenance. Only original chunks
are delivered to the generator. This retriever is deliberately independent of LightRAG internals.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Stable ties: source row order. Small corpora make full sorting inexpensive."""
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def top_distribution(scores: np.ndarray, k: int, temperature: float) -> np.ndarray:
    if temperature <= 0 or k <= 0:
        raise ValueError("temperature and k must be positive")
    scores = np.asarray(scores, float)
    out = np.zeros(scores.shape, dtype=float)
    for col in range(scores.shape[1]):
        ids = top_indices(scores[:, col], k)
        if len(ids):
            weights = np.exp((scores[ids, col] - scores[ids, col].max()) / temperature)
            out[ids, col] = weights / weights.sum()
    return out


def _vectors(rows: list[dict], dim: int) -> np.ndarray:
    vectors = []
    for row in rows:
        raw = zlib.decompress(base64.b64decode(row["vector"]))
        dtype = np.float16 if len(raw) == 2 * dim else np.float32 if len(raw) == 4 * dim else None
        if dtype is None:
            raise ValueError(f"unrecognised stored vector length for {row['__id__']}")
        vectors.append(np.frombuffer(raw, dtype=dtype).astype(np.float32))
    matrix = np.array(vectors, dtype=np.float32).reshape((-1, dim))
    if len(matrix):
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite frozen vectors")
    return matrix


@dataclass
class FrozenExtraction:
    records: list[dict]
    record_vectors: np.ndarray
    chunks: list[dict]
    chunk_vectors: np.ndarray
    provenance: sparse.csr_matrix       # record -> original chunk, normalised per record
    edges: np.ndarray                  # (entity row, relation row), both in global record ids
    n_entities: int
    hashes: dict[str, str]

    @classmethod
    def load(cls, folder: Path) -> "FrozenExtraction":
        names = ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json", "kv_store_text_chunks.json"]
        hashes = {name: sha256(folder / name) for name in names}
        stores = [json.loads((folder / name).read_text(encoding="utf-8")) for name in names[:3]]
        entities, relations, chunks = [store["data"] for store in stores]
        if any(store["embedding_dim"] != 384 for store in stores):
            raise ValueError("this pilot requires the frozen MiniLM 384-dimensional vectors")
        # Copy just the immutable payload needed below, rather than retaining duplicated base64 matrices.
        records = entities + relations
        rv = _vectors(records, 384)
        cv = _vectors(chunks, 384)
        chunk_idx = {row["__id__"]: i for i, row in enumerate(chunks)}
        entity_idx = {row["entity_name"]: i for i, row in enumerate(entities)}
        if len(chunk_idx) != len(chunks) or len(entity_idx) != len(entities):
            raise ValueError("duplicate frozen record ids")
        ri, ci, values, edges = [], [], [], []
        for i, row in enumerate(records):
            sources = sorted({chunk_idx[s] for s in row.get("source_id", "").split("<SEP>") if s in chunk_idx})
            for source in sources:
                ri.append(i)
                ci.append(source)
                values.append(1.0 / len(sources))
        for j, row in enumerate(relations, len(entities)):
            for name in sorted({row["src_id"], row["tgt_id"]}):
                if name not in entity_idx:
                    raise ValueError(f"relation endpoint absent from frozen entity store: {name}")
                edges.append((entity_idx[name], j))
        provenance = sparse.csr_matrix((values, (ri, ci)), shape=(len(records), len(chunks)))
        immutable = [{k: v for k, v in row.items() if k not in ("vector",)} for row in records]
        chunk_payload = [{k: v for k, v in row.items() if k != "vector"} for row in chunks]
        return cls(immutable, rv, chunk_payload, cv, provenance,
                   np.array(edges, dtype=np.int64).reshape((-1, 2)), len(entities), hashes)


def rewire_bipartite(edges: np.ndarray, n_nodes: int, seed: int, swaps_per_edge: int) -> tuple[np.ndarray, dict]:
    """Double-edge swaps retain each node's degree and the entity/relation bipartition exactly."""
    edges = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    out = edges.copy()
    original = set(map(tuple, edges.tolist()))
    current = set(original)
    if len(current) != len(edges):
        raise ValueError("parallel input incidence edges")
    rng = np.random.default_rng(seed)
    target, done, attempts = swaps_per_edge * len(out), 0, 0
    while len(out) >= 2 and done < target and attempts < max(100, target * 30):
        attempts += 1
        i, j = rng.integers(0, len(out), size=2)
        if i == j:
            continue
        a, b = map(int, out[i])
        c, d = map(int, out[j])
        if a == c or b == d or (a, d) in current or (c, b) in current:
            continue
        current.remove((a, b))
        current.remove((c, d))
        current.update(((a, d), (c, b)))
        out[i], out[j] = (a, d), (c, b)
        done += 1
    before = np.bincount(edges.ravel(), minlength=n_nodes)
    after = np.bincount(out.ravel(), minlength=n_nodes)
    if not np.array_equal(before, after) or len(current) != len(edges):
        raise AssertionError("rewiring violated its degree or edge-count invariant")
    return out, dict(seed=seed, edges=len(out), swaps=done, target_swaps=target, attempts=attempts,
                     changed_fraction=1 - len(original & current) / len(original) if original else 0.0,
                     degree_preserved=True)


def pagerank(seed_mass: np.ndarray, edges: np.ndarray, damping: float, max_iter: int,
             tolerance: float) -> tuple[np.ndarray, dict]:
    """Batched PPR. Dangling mass returns to the same query's seed distribution."""
    seed_mass = np.asarray(seed_mass, float)
    n = len(seed_mass)
    if not 0 <= damping < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("invalid PPR parameters")
    if not np.allclose(seed_mass.sum(axis=0), 1):
        raise ValueError("seed distributions must sum to one")
    edges = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if not len(edges):
        return seed_mass.copy(), dict(iterations=0, residual=0.0, converged=True)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = sparse.diags(1 / np.maximum(degree, 1)) @ adjacency
    dangling = degree == 0
    x = seed_mass.copy()
    for iteration in range(1, max_iter + 1):
        new = (1 - damping) * seed_mass + damping * (transition.T @ x + seed_mass * x[dangling].sum(axis=0))
        residual = float(np.abs(new - x).sum(axis=0).max())
        x = new
        if residual < tolerance:
            break
    if not np.allclose(x.sum(axis=0), 1, atol=1e-8) or np.any(x < -1e-12):
        raise AssertionError("PPR lost probability mass")
    return x, dict(iterations=iteration, residual=residual, converged=residual < tolerance)


def chunk_scores(extraction: FrozenExtraction, query_vectors: np.ndarray, cfg: dict) -> tuple[dict, dict]:
    """Return chunk-by-query scores; this function has no access to answers or support."""
    q = np.asarray(query_vectors, np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dense = top_distribution(extraction.chunk_vectors @ q.T, cfg["dense_depth"], cfg["temperature"])
    if not len(extraction.records):
        raise ValueError("empty extraction cannot test a topology effect")
    seed = top_distribution(extraction.record_vectors @ q.T, cfg["seed_records"], cfg["temperature"])
    masses = {"flat_records": seed}
    metadata = {}
    real, info = pagerank(seed, extraction.edges, cfg["damping"], cfg["max_iter"], cfg["tolerance"])
    masses["graph_real"] = real
    metadata["graph_real"] = info
    empty, _ = pagerank(seed, np.empty((0, 2), int), cfg["damping"], cfg["max_iter"], cfg["tolerance"])
    if not np.array_equal(empty, seed):
        raise AssertionError("zero-edge graph is not identical to flat")
    for null_seed in cfg["rewire_seeds"]:
        edges, null_meta = rewire_bipartite(extraction.edges, len(seed), null_seed, cfg["swaps_per_edge"])
        name = f"graph_rewired_{null_seed}"
        masses[name], info = pagerank(seed, edges, cfg["damping"], cfg["max_iter"], cfg["tolerance"])
        metadata[name] = dict(**null_meta, **info)
    weight = cfg["dense_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("dense_weight must be in [0,1]")
    scores = {"dense_chunks": dense}
    for name, mass in masses.items():
        projected = np.asarray(extraction.provenance.T @ mass)
        # Renormalise missing provenance mass equally in every arm; report source integrity separately.
        projected /= np.maximum(projected.sum(axis=0, keepdims=True), 1e-12)
        scores[name] = weight * dense + (1 - weight) * projected
    metadata["frozen"] = dict(entities=extraction.n_entities, relations=len(seed) - extraction.n_entities,
                              chunks=len(extraction.chunks), provenance_missing_records=int(
                                  (np.asarray(extraction.provenance.sum(axis=1)).ravel() == 0).sum()),
                              edges=len(extraction.edges), zero_edge_equals_flat=True)
    return scores, metadata


def pack_context(chunks: list[dict], ranked: list[int], tokenizer, budget: int,
                 dense_ranked: list[int] | None = None, reserve_fraction: float = 0.0) -> dict:
    """Single shared serializer. Optional dense reservation ends at the previous complete chunk."""
    ordered, seen, spent = [], set(), 0
    if dense_ranked is not None and reserve_fraction > 0:
        for i in dense_ranked:
            text = f"[{len(ordered) + 1}] {chunks[i]['content']}\n"
            count = len(tokenizer.encode(text, add_special_tokens=False))
            if spent + count > budget * reserve_fraction:
                break
            ordered.append(i)
            seen.add(i)
            spent += count
    ordered.extend(i for i in ranked if i not in seen)
    context, delivered, fully, token_counts = "", [], [], []
    for i in ordered:
        text = f"[{len(delivered) + 1}] {chunks[i]['content']}\n"
        candidate = context + text
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) <= budget:
            context = candidate
            fully.append(i)
            delivered.append(i)
            token_counts.append(len(ids))
        else:
            if len(tokenizer.encode(context, add_special_tokens=False)) < budget:
                context = tokenizer.decode(ids[:budget], skip_special_tokens=False)
                delivered.append(i)
            break
    tokens = len(tokenizer.encode(context, add_special_tokens=False))
    # Decode/encode at a token boundary can differ by a token; trim to a verified ceiling.
    while tokens > budget:
        context = tokenizer.decode(tokenizer.encode(context, add_special_tokens=False)[:budget - 1],
                                   skip_special_tokens=False)
        tokens = len(tokenizer.encode(context, add_special_tokens=False))
    return dict(context=context, tokens=tokens, delivered_chunks=delivered, full_chunks=fully,
                context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest())
