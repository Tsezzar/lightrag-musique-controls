"""The manuscript's six cheap sample features, with explicit missingness and selected-only access."""
from __future__ import annotations

import itertools
import time
import zipfile
from pathlib import Path, PurePosixPath

import networkx as nx
import numpy as np

from .k1_protocol import AuditSample, file_sha256, freeze_manifest

FEATURE_NAMES = ("lcc", "density", "clustering", "twonn", "hubness", "centroid_distance")


def six_features(embeddings, entity_sets: list[set[str]], knn_k: int = 10) -> dict:
    matrix = np.asarray(embeddings, dtype=np.float64)
    n = len(entity_sets)
    if matrix.ndim != 2 or matrix.shape[0] != n or not n or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("finite aligned nonempty embedding matrix required")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError("nonzero finite embedding vectors required")
    if type(knn_k) is not int or knn_k < 1:
        raise ValueError("positive integer neighbor count required")
    matrix = matrix / norms
    graph = nx.Graph()
    for names in entity_sets:
        graph.add_nodes_from(names)
        graph.add_edges_from(itertools.combinations(sorted(names), 2))
    nodes = graph.number_of_nodes()
    values = dict(lcc=max(map(len, nx.connected_components(graph)), default=0) / nodes if nodes else 0.,
                  density=nx.density(graph), clustering=nx.average_clustering(graph) if nodes else 0.,
                  twonn=None, hubness=None, centroid_distance=None)
    undefined = {}
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        values["centroid_distance"] = float(np.mean(1 - np.clip(matrix @ (centroid / norm), -1, 1)))
    else:
        undefined["centroid_distance"] = "zero centroid"
    similarity = np.clip(matrix @ matrix.T, -1, 1)
    distances = np.sqrt(np.maximum(0, 2 - 2 * similarity))
    np.fill_diagonal(distances, np.inf)
    # Stable order makes exact distance ties reproducible; they are disclosed.
    order = np.argsort(distances, axis=1, kind="stable")
    if n >= 3:
        first, second = (distances[np.arange(n), order[:, i]] for i in (0, 1))
        # Equal rows may round to a tiny positive distance after unit normalization.
        duplicates = len(np.unique(matrix, axis=0)) != n
        if not duplicates and (first > 0).all():
            denominator = float(np.mean(np.log(second / first)))
            if denominator > 0 and np.isfinite(denominator):
                values["twonn"] = 1 / denominator
        if values["twonn"] is None:
            undefined["twonn"] = "duplicate points, zero nearest distance or zero mean log distance ratio"
    else:
        undefined["twonn"] = "fewer than three sampled blocks"
    boundary_ties = None
    if n > knn_k:
        neighbors = order[:, :knn_k]
        occurrences = np.bincount(neighbors.ravel(), minlength=n).astype(float)
        standard_deviation = occurrences.std()
        if standard_deviation > 0:
            values["hubness"] = float(np.mean(((occurrences - occurrences.mean()) / standard_deviation) ** 3))
        else:
            undefined["hubness"] = "constant neighbor occurrence counts"
        if n > knn_k + 1:
            boundary_ties = int(np.sum(distances[np.arange(n), order[:, knn_k - 1]] ==
                                       distances[np.arange(n), order[:, knn_k]]))
    else:
        undefined["hubness"] = "sample does not contain k+1 blocks"
    return dict(values=values, undefined=undefined, diagnostics=dict(sample_blocks=n, nodes=nodes,
                edges=graph.number_of_edges(), knn_k=knn_k, hubness_boundary_ties=boundary_ties))


def sample_fingerprint(sample: AuditSample, nlp, embed, *, batch_size: int = 64, knn_k: int = 10) -> dict:
    start = time.perf_counter()
    texts = tuple(block.text for block in sample.blocks)
    entities = [{" ".join(entity.text.casefold().split()) for entity in doc.ents if entity.text.strip()}
                for doc in nlp.pipe(texts, batch_size=batch_size)]
    result = six_features(embed(texts), entities, knn_k)
    return dict(corpus_id=sample.corpus_id, corpus_blocks=sample.corpus_blocks,
                sample_ids=[block.block_id for block in sample.blocks], **result,
                wall_seconds=time.perf_counter() - start)


def unpack_ner_data(archive: Path, directory: Path, maximum_bytes: int) -> Path:
    """Extract model data only, never package Python; validate all paths before writing."""
    with zipfile.ZipFile(archive) as source:
        configs = [PurePosixPath(name) for name in source.namelist() if name.endswith("/config.cfg")]
        if len(configs) != 1 or len(configs[0].parts) < 3:
            raise ValueError("one uniquely nested NER model configuration required")
        prefix = configs[0].parent
        model = directory / prefix.name
        files, total = [], 0
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if entry.is_dir() or not entry.filename.startswith(str(prefix) + "/"):
                continue
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
                raise ValueError("invalid model archive path")
            relative = path.relative_to(prefix)
            target = model.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(model.resolve()):
                raise ValueError("model archive path escapes its destination")
            if path.suffix.lower() in (".py", ".pyc", ".pyd", ".dll", ".so", ".exe"):
                raise ValueError("executable content is not model data")
            total += entry.file_size
            files.append((entry, target))
        if maximum_bytes <= 0 or total > maximum_bytes:
            raise ValueError("unpacked model exceeds configured size limit")
        for entry, target in files:
            contents = source.read(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != contents:
                    raise ValueError("existing model data changed")
            else:
                with target.open("xb") as stream:
                    stream.write(contents)
    freeze_manifest(directory / "extracted_manifest.json", dict(archive_sha256=file_sha256(archive),
                    files={str(path.relative_to(directory.resolve())): file_sha256(path) for _, path in files}))
    return model
