"""Control scheduling and judge diagnostics; repetitions never inflate validation n."""
from __future__ import annotations

import math
import random
from statistics import NormalDist


def validation_schedule(n_controls: int, repeated_controls: int, repetitions: int, seed: int) -> tuple[tuple[int, int], ...]:
    if n_controls < 1 or not 0 <= repeated_controls <= n_controls or repetitions < 1:
        raise ValueError("invalid distinct-control or repetition counts")
    repeated = sorted(random.Random(seed).sample(range(n_controls), repeated_controls))
    return tuple([(i, 0) for i in range(n_controls)] +
                 [(i, rep) for rep in range(1, repetitions) for i in repeated])


def _wilson(errors: int, n: int, alpha: float) -> tuple[float, float] | None:
    if not n:
        return None
    z, p = NormalDist().inv_cdf(1 - alpha / 2), errors / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - width, center + width


def validation_summary(rows: list[dict], n_controls: int, repetitions: int,
                       maximum_error: float, alpha: float = .05) -> dict:
    if not 0 <= maximum_error < 1 or not 0 < alpha < 1:
        raise ValueError("invalid validation error limit or confidence level")
    result = {}
    for role in sorted({row["role"] for row in rows}):
        selected = [row for row in rows if row["role"] == role]
        base = [row for row in selected if row["repetition"] == 0]
        invalid = sum(row["observed"] is None for row in base)
        errors = sum(row["observed"] is not None and row["observed"] != row["expected"] for row in base)
        groups = {i: [r for r in selected if r["case_index"] == i] for i in {r["case_index"] for r in selected}}
        repeated = [group for group in groups.values() if len(group) == repetitions
                    and {r["repetition"] for r in group} == set(range(repetitions))]
        stable = sum(all(r["observed"] is not None for r in group)
                     and len({r["observed"] for r in group}) == 1 for group in repeated)
        rate = (errors + invalid) / len(base) if base else None
        result[role] = dict(distinct_controls_completed=len(base), missing_base_controls=n_controls - len(base),
                            base_errors=errors, base_invalid=invalid, base_error_rate=rate,
                            error_wilson_interval=_wilson(errors + invalid, len(base), alpha),
                            passed_simple_controls=len(base) == n_controls and rate is not None and rate <= maximum_error,
                            complete_repeated_controls=len(repeated), stable_repeated_controls=stable,
                            inference_verdicts=len(selected),
                            interpretation="Synthetic control accuracy and repetition stability, not human agreement or real-domain validity.")
    return result
