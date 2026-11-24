"""Corpus-level Ridge rule: train-only transforms, one-sided calibration and public test inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .k1_audit import FEATURE_NAMES
from .k1_protocol import calibration_order


@dataclass(frozen=True)
class FeatureRow:
    corpus_id: str
    family: str
    split: str
    values: dict[str, float | None]


@dataclass(frozen=True)
class FittingRow(FeatureRow):
    gain: float


def _matrix(rows: list[FeatureRow]) -> np.ndarray:
    if not rows or len({row.corpus_id for row in rows}) != len(rows):
        raise ValueError("nonempty rows with no duplicate corpus IDs required")
    for row in rows:
        if set(row.values) != set(FEATURE_NAMES):
            raise ValueError("fixed feature names required")
        if any(value is not None and (not isinstance(value, Real) or isinstance(value, bool)
                                     or not math.isfinite(value)) for value in row.values.values()):
            raise ValueError("features must be finite numeric values or explicit None")
    return np.asarray([[row.values[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


def _with_missing_flags(imputer, matrix):
    # All six flags are retained even when missingness first appears outside
    # train; transforms themselves are still fitted exclusively on train.
    return np.column_stack([imputer.transform(matrix), np.isnan(matrix).astype(float)])


@dataclass
class FittedRidge:
    imputer: SimpleImputer
    scaler: StandardScaler
    regressor: Ridge
    threshold: float | None
    calibration_order: int | None
    coverage: float
    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    train_all_missing: tuple[str, ...]

    def decide(self, rows: list[FeatureRow]) -> list[dict]:
        if any(row.split != "test" for row in rows):
            raise ValueError("test feature rows required for prospective decisions")
        if set(row.corpus_id for row in rows) & set(self.train_ids + self.calibration_ids):
            raise ValueError("test corpus identity overlaps fitting data")
        matrix = _matrix(rows)
        predictions = self.regressor.predict(self.scaler.transform(_with_missing_flags(self.imputer, matrix)))
        if not np.isfinite(predictions).all():
            raise ValueError("nonfinite predicted utility gain")
        result = []
        for row, predicted in zip(rows, predictions):
            lower = None if self.threshold is None else float(predicted - self.threshold)
            result.append(dict(corpus_id=row.corpus_id, family=row.family, predicted_gain=float(predicted),
                               lower_bound=lower, choice="lightrag" if lower is not None and lower > 0 else "vector"))
        return result

    def export(self) -> dict:
        return dict(model="ridge", target="utility_gain_lightrag_minus_vector_seconds_per_query",
                    features=list(FEATURE_NAMES), transformed_features=list(FEATURE_NAMES) + [name + "__missing" for name in FEATURE_NAMES],
                    alpha=float(self.regressor.alpha), imputation_statistics=self.imputer.statistics_.tolist(),
                    scaling_mean=self.scaler.mean_.tolist(), scaling_scale=self.scaler.scale_.tolist(),
                    coefficient=self.regressor.coef_.tolist(), intercept=float(self.regressor.intercept_),
                    calibration_threshold=self.threshold, calibration_order=self.calibration_order, coverage=self.coverage,
                    train_ids=list(self.train_ids), calibration_ids=list(self.calibration_ids),
                    train_all_missing=list(self.train_all_missing),
                    limitation="Nominal calibration under exchangeability; no source-conditional or shifted-domain guarantee.")


def fit_ridge(train: list[FittingRow], calibration: list[FittingRow], alpha: float, coverage: float) -> FittedRidge:
    if not train or any(row.split != "train" for row in train):
        raise ValueError("train labels must come exclusively from train corpora")
    if not calibration or any(row.split != "calibration" for row in calibration):
        raise ValueError("calibration labels must come exclusively from calibration corpora")
    identities = [row.corpus_id for row in train + calibration]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate fitting corpus identities; audit repetitions are not independent rows")
    if not math.isfinite(alpha) or alpha <= 0 or any(not math.isfinite(row.gain) for row in train + calibration):
        raise ValueError("positive regularization and finite measured gains required")
    x_train, x_cal = _matrix(train), _matrix(calibration)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_train)
    transformed = _with_missing_flags(imputer, x_train)
    scaler = StandardScaler().fit(transformed)
    regressor = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(transformed), [row.gain for row in train])
    prediction = regressor.predict(scaler.transform(_with_missing_flags(imputer, x_cal)))
    residuals = prediction - np.asarray([row.gain for row in calibration])
    order = calibration_order(len(calibration), coverage)
    threshold = None if order is None else float(np.sort(residuals)[order - 1])
    return FittedRidge(imputer, scaler, regressor, threshold, order, coverage,
                       tuple(row.corpus_id for row in train), tuple(row.corpus_id for row in calibration),
                       tuple(name for i, name in enumerate(FEATURE_NAMES) if np.isnan(x_train[:, i]).all()))
