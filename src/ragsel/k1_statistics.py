"""Paired corpus bootstrap with fixed source counts and explicit practical margins."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def stratified_comparison(rows: list[tuple[str, str, float]], expected_source_counts: dict[str, int],
                          resamples: int, alpha: float, seed: int) -> dict:
    if (not expected_source_counts or any(type(n) is not int or n < 2 for n in expected_source_counts.values())
            or type(resamples) is not int or resamples < 1 or not 0 < alpha < 1):
        raise ValueError("positive bootstrap settings and at least two corpora per source required")
    if len({cid for cid, _, _ in rows}) != len(rows):
        raise ValueError("duplicate corpus IDs; questions/seeds are not independent observations")
    if dict(Counter(source for _, source, _ in rows)) != expected_source_counts:
        raise ValueError("source counts differ from the frozen comparison population")
    if any(not math.isfinite(value) for _, _, value in rows):
        raise ValueError("finite paired corpus differences required")
    rng = np.random.default_rng(seed)
    bootstrap, means = [], []
    for source in sorted(expected_source_counts):
        values = np.asarray([value for _, family, value in rows if family == source], dtype=float)
        means.append(float(values.mean()))
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        bootstrap.append(values[indices].mean(axis=1))
    distribution = np.mean(bootstrap, axis=0)
    return dict(estimate=float(np.mean(means)),
                interval=[float(np.quantile(distribution, alpha / 2)), float(np.quantile(distribution, 1 - alpha / 2))],
                independent_corpora=len(rows), source_counts=expected_source_counts, source_weights="equal",
                resamples=resamples, alpha=alpha, seed=seed)


def interval_conclusion(interval: list[float], margin: float) -> str:
    if (len(interval) != 2 or not all(math.isfinite(x) for x in interval) or interval[0] > interval[1]
            or not math.isfinite(margin) or margin <= 0):
        raise ValueError("ordered finite interval and positive practical margin required")
    low, high = interval
    if low > margin:
        return "practically_better"
    if high < -margin:
        return "practically_worse"
    if low >= -margin and high <= margin:
        return "within_practical_margin"
    return "inconclusive"
