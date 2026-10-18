"""Leakage-safe model-specific preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .data import IDENTIFIER, NOMINAL_COLUMNS

# tcprtt equals synack + ackdat (within floating-point precision) in every row.
# The distance pipeline drops the derived total to avoid counting the same timing
# information twice. The tree pipeline retains the total because a threshold on
# the sum supplies a distinct axis-aligned split. The other pairs highlighted by
# the brief remain available because selection-only ablations found no benefit
# from removing sloss, dloss, and ct_ftp_cmd.
REDUNDANT_DROP_COLUMNS = ("tcprtt",)
HIGH_CORRELATION_DROP_COLUMNS = ("sloss", "dloss", "ct_ftp_cmd")


class SkewedLog1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to non-negative columns whose training skew exceeds a threshold."""

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def fit(self, X: np.ndarray | pd.DataFrame, y: object = None) -> SkewedLog1pTransformer:
        values = np.asarray(X, dtype=np.float64)
        frame = pd.DataFrame(values)
        skew = frame.skew(axis=0, skipna=True).abs().to_numpy()
        minima = np.nanmin(values, axis=0)
        self.selected_indices_ = np.flatnonzero((skew >= self.threshold) & (minima >= 0.0))
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32).copy()
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected number of numerical features")
        values[:, self.selected_indices_] = np.log1p(values[:, self.selected_indices_])
        return values


@dataclass(frozen=True)
class PreprocessorSpec:
    name: str
    log_skewed: bool
    scaler: str | None
    drop_redundant: bool = True


PREPROCESSOR_SPECS = {
    "standard": PreprocessorSpec("standard", False, "standard"),
    "standard_all": PreprocessorSpec("standard_all", False, "standard", False),
    "robust": PreprocessorSpec("robust", False, "robust"),
    "robust_all": PreprocessorSpec("robust_all", False, "robust", False),
    "log_standard": PreprocessorSpec("log_standard", True, "standard"),
    "log_standard_all": PreprocessorSpec("log_standard_all", True, "standard", False),
    "tree": PreprocessorSpec("tree", False, None),
    "tree_all": PreprocessorSpec("tree_all", False, None, False),
}


def input_columns(frame: pd.DataFrame, drop_redundant: bool) -> tuple[list[str], list[str]]:
    excluded = {IDENTIFIER}
    if drop_redundant:
        excluded.update(column for column in REDUNDANT_DROP_COLUMNS if column in frame.columns)
    nominal = [column for column in NOMINAL_COLUMNS if column not in excluded]
    numeric = [
        column
        for column in frame.columns
        if column not in excluded and column not in NOMINAL_COLUMNS
    ]
    return numeric, nominal


def build_preprocessor(frame: pd.DataFrame, spec_name: str) -> ColumnTransformer:
    """Create a fresh preprocessing graph for one cross-validation fold."""
    if spec_name not in PREPROCESSOR_SPECS:
        raise ValueError(f"Unknown preprocessor: {spec_name}")
    spec = PREPROCESSOR_SPECS[spec_name]
    numeric_columns, nominal_columns = input_columns(frame, spec.drop_redundant)
    numeric_steps: list[tuple[str, object]] = []
    if spec.log_skewed:
        numeric_steps.append(("log1p_skewed", SkewedLog1pTransformer()))
    if spec.scaler == "standard":
        numeric_steps.append(("scale", StandardScaler()))
    elif spec.scaler == "robust":
        numeric_steps.append(("scale", RobustScaler()))
    numeric_transformer: object = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    nominal_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("nominal", nominal_transformer, nominal_columns),
        ],
        sparse_threshold=0.30,
        verbose_feature_names_out=True,
    )
