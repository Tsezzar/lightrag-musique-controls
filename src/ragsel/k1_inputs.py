"""Load the pinned native benchmark formats without Hugging Face cache mutation."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .k1_data import DataPool, parse_cuad, parse_musique, parse_pubmedqa
from .k1_protocol import file_sha256


def verify_inventory(path: Path, root: Path, required_paths=()) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for required in required_paths:
        if str(Path(required).resolve()) not in payload["inputs"]:
            raise ValueError(f"input is not bound to the frozen inventory: {required}")
    for name, expected in payload["inputs"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"frozen input changed: {name}")
    for name, expected in payload["code_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"frozen parser code changed: {name}")
    return payload


def arrow_rows(path: Path):
    with pa.memory_map(str(path), "r") as stream:
        for batch in pa.ipc.open_stream(stream):
            yield from batch.to_pylist()


def parquet_rows(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def load_public_pools(raw: Path, musique_file: Path) -> dict[str, DataPool]:
    with zipfile.ZipFile(raw / "cuad.zip") as archive:
        cuad_data = json.loads(archive.read("CUADv1.json"))
    pubmed_data = json.loads((raw / "pubmedqa_labeled.json").read_text(encoding="utf-8"))
    pubmed_rows = (dict(pubid=pmid, question=row["QUESTION"], context=dict(contexts=row["CONTEXTS"]),
                        final_decision=row["final_decision"]) for pmid, row in pubmed_data.items())
    return {
        "cuad": parse_cuad(cuad_data),
        "musique": parse_musique(arrow_rows(musique_file)),
        "pubmedqa": parse_pubmedqa(pubmed_rows, labeled=True),
        "pubmedqa_unlabeled": parse_pubmedqa(parquet_rows(raw / "pubmedqa_unlabeled.parquet"), labeled=False),
    }
