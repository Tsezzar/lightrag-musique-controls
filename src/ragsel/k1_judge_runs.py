"""Read-only lineage for interrupted judge runs; no copied inference records or costs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .k1_protocol import file_sha256


def collect_control_rows(con: sqlite3.Connection, directory: Path, seen=None) -> list[dict]:
    seen = set() if seen is None else seen
    identity = str(directory.resolve())
    if identity in seen:
        raise ValueError("cyclic judge run continuation")
    seen.add(identity)
    frozen = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
    rows = []
    if frozen.get("continuation"):
        lineage = frozen["continuation"]
        previous = Path(lineage["directory"])
        for name, expected in lineage["files"].items():
            if file_sha256(previous / name) != expected:
                raise ValueError(f"prior judge artifact changed: {previous / name}")
        old = sqlite3.connect((previous / "calls.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows.extend(collect_control_rows(old, previous, seen))
        finally:
            old.close()
    cur = con.execute("SELECT call_id,role,case_index,repetition,expected,observed FROM judge_controls ORDER BY rowid")
    columns = [col[0] for col in cur.description]
    rows.extend(dict(zip(columns, values)) for values in cur)
    if len({row["call_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate judge verdict identity in continuation")
    return rows
