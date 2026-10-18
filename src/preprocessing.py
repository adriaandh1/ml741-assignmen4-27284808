"""Label-free distance geometry, fitted exclusively through authorised row indices."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import BINARY, NOMINAL, SOM_COLUMNS, assert_som_features

CONTINUOUS = tuple(c for c in SOM_COLUMNS if c not in (*BINARY, *NOMINAL))


class SOMPreprocessor:
    """Reject labels/extra columns. no target argument exists in the fitting API."""

    def __init__(self, candidate: str):
        if candidate not in {"standard", "log_standard"}:
            raise ValueError("Unknown representation")
        self.candidate = candidate

    def fit(self, features: pd.DataFrame):
        assert_som_features(features)
        if features.empty:
            raise ValueError("Empty fitting data")
        values = features.loc[:, CONTINUOUS].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Numerical inputs must be finite/non-negative")
        skew = pd.DataFrame(values).skew().fillna(0).to_numpy()
        self.log_indices_ = (
            np.flatnonzero(skew >= 2)
            if self.candidate == "log_standard"
            else np.array([], dtype=int)
        )
        values[:, self.log_indices_] = np.log1p(values[:, self.log_indices_])
        self.scaler_ = StandardScaler().fit(values)
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        ).fit(features.loc[:, NOMINAL])
        self.fit_rows_ = features.index.to_numpy(dtype=np.int64).copy()
        self.feature_names_ = (
            [f"numeric__{c}" for c in CONTINUOUS]
            + [f"binary__{c}" for c in BINARY]
            + [f"nominal__{c}" for c in self.encoder_.get_feature_names_out(NOMINAL)]
        )
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        assert_som_features(features)
        if not hasattr(self, "scaler_"):
            raise ValueError("Preprocessor has not been fitted")
        numeric = features.loc[:, CONTINUOUS].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Numerical inputs must be finite/non-negative")
        numeric[:, self.log_indices_] = np.log1p(numeric[:, self.log_indices_])
        binary = features.loc[:, BINARY].to_numpy(dtype=np.float32)
        if not np.isin(binary, [0, 1]).all():
            raise ValueError("Unrepaired binary inputs")
        # Each known categorical mismatch contributes one squared distance unit.
        nominal = self.encoder_.transform(features.loc[:, NOMINAL]) / np.sqrt(2)
        result = np.concatenate((self.scaler_.transform(numeric), binary, nominal), axis=1)
        result = result.astype(np.float32)
        if not np.isfinite(result).all() or result.shape[1] != len(self.feature_names_):
            raise ValueError("Non-finite values/unstable transformed dimension")
        return result

    def metadata(self) -> dict:
        return {
            "candidate": self.candidate,
            "fit_row_count": len(self.fit_rows_),
            "input_columns": list(SOM_COLUMNS),
            "feature_names": self.feature_names_,
            "dimension": len(self.feature_names_),
            "log_columns": [CONTINUOUS[i] for i in self.log_indices_],
            "numeric_columns": list(CONTINUOUS),
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "scaler_variance": self.scaler_.var_.tolist(),
            "categories": {
                c: v.tolist() for c, v in zip(NOMINAL, self.encoder_.categories_, strict=True)
            },
            "unknown_category_rule": "all-zero block, weight 1/sqrt(2)",
            "constant_rule": "scale=1; retained; never select using held-out variance",
        }


def fit_on_training(
    features: pd.DataFrame, rows: np.ndarray, allowed_rows: np.ndarray, candidate: str
) -> SOMPreprocessor:
    if not np.array_equal(np.sort(rows), np.sort(allowed_rows)):
        raise ValueError("Preprocessing fit rows differ from permitted training rows")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Repeated fitting indices")
    if not features.index.is_unique:
        raise ValueError("Feature source indices must be unique")
    return SOMPreprocessor(candidate).fit(features.loc[rows])
