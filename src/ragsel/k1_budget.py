"""Shared accounting for all pilot inference runs."""
from pathlib import Path
import math
import sqlite3
import json


class IncompleteAccountingError(RuntimeError):
    pass


def pilot_seconds(root: Path) -> float:
    total = 0.0
    for path in sorted(root.rglob("calls.sqlite")):
        annotation = path.parent / "budget_reconciliation.json"
        reconciled = json.loads(annotation.read_text(encoding="utf-8"))["attempts"] if annotation.exists() else []
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for call_id, attempt, status, seconds, identity in con.execute(
                    "SELECT call_id,attempt,status,wall_seconds,request_sha256 FROM k1_calls"):
                if status == "running" or seconds is None or not math.isfinite(seconds) or seconds < 0:
                    matches = [row for row in reconciled if row["call_id"] == call_id and row["attempt"] == attempt
                               and row["request_sha256"] == identity and row["state"] == "interrupted"
                               and row.get("evidence")]
                    if len(matches) != 1:
                        raise IncompleteAccountingError(
                            f"unaccounted inference in {path}: {call_id}, attempt {attempt}; reconcile logs before continuing")
                    seconds = matches[0]["reserved_seconds"]
                    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                        raise IncompleteAccountingError(f"invalid interrupted-call budget reservation in {annotation}")
                total += seconds
        finally:
            con.close()
    return total
