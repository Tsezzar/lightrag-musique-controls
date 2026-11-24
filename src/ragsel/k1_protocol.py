"""Deterministic boundaries for the prospective K1 study; no LLM or gold access."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Block:
    block_id: str
    parent_id: str
    text: str


@dataclass(frozen=True)
class AuditSample:
    corpus_id: str
    corpus_blocks: int
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Provenance:
    source: str
    corpus_id: str
    split: str
    source_id: str
    parent_id: str
    text_sha256: str


def make_audit_sample(corpus_id: str, blocks: tuple[Block, ...], count: int, seed: int) -> AuditSample:
    if type(count) is not int or not 1 <= count <= len(blocks):
        raise ValueError("audit size must be an integer within the corpus")
    if len({b.block_id for b in blocks}) != len(blocks):
        raise ValueError("block identifiers must be unique within a corpus")
    ids = sorted(random.Random(seed).sample(range(len(blocks)), count))
    return AuditSample(corpus_id, len(blocks), tuple(blocks[i] for i in ids))


def run_sample_audit(sample: AuditSample, extractor: Callable[[tuple[str, ...]], T]) -> T:
    """The callback receives selected text only, never the full corpus or gold."""
    return extractor(tuple(block.text for block in sample.blocks))


def assert_independent(rows: Iterable[Provenance]) -> None:
    owners: dict[tuple[str, ...], tuple[str, str]] = {}
    conflicts: list[str] = []
    for row in rows:
        owner = row.corpus_id, row.split
        keys = (("source", row.source, row.source_id),
                ("parent", row.source, row.parent_id), ("text", row.text_sha256))
        for key in keys:
            if key in owners and owners[key] != owner:
                conflicts.append(f"{key[0]} overlap {key[1:]}: {owners[key]} vs {owner}")
            else:
                owners[key] = owner
    if conflicts:
        raise ValueError("; ".join(conflicts))


def calibration_order(n_calibration: int, coverage: float) -> int | None:
    """None denotes an unavailable finite split-conformal quantile, not a clipped one."""
    if type(n_calibration) is not int or n_calibration < 1 or not 0 < coverage < 1:
        raise ValueError("positive calibration size and coverage in (0,1) required")
    order = math.ceil((n_calibration + 1) * coverage)
    return order if order <= n_calibration else None


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def config_fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(path: Path, payload: object) -> None:
    """Create once; an identical resume is permitted, changed inputs fail closed."""
    path = Path(path)
    encoded = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if canonical_json(json.loads(path.read_text(encoding="utf-8"))) != encoded:
            raise ValueError(f"frozen manifest differs: {path}") from None


def choose_matrix(elapsed_hours: float, forecasts: Mapping[str, float], planned_hours: float,
                  multiplier: float) -> str | None:
    values = (elapsed_hours, planned_hours, multiplier, *forecasts.values())
    if any(not math.isfinite(v) or v < 0 for v in values) or multiplier < 1 or planned_hours <= 0:
        raise ValueError("finite nonnegative times and a safety multiplier >= 1 required")
    return next((name for name, cost in forecasts.items()
                 if elapsed_hours + multiplier * cost <= planned_hours), None)


def validate_quote(contexts: Mapping[str, str], source_id: str, quote: str) -> bool:
    """Checks literal existence only. This function does not establish entailment."""
    return bool(quote.strip()) and source_id in contexts and quote in contexts[source_id]
