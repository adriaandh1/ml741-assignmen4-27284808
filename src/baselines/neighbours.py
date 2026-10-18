"""Cached exact nearest-neighbour tuning and prediction."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from .config import ExperimentConfig
from .metrics import classification_metrics
from .preprocessing import build_preprocessor


def _as_float32(matrix: object) -> object:
    if sparse.issparse(matrix):
        return matrix.astype(np.float32, copy=False).tocsr()
    return np.asarray(matrix, dtype=np.float32)


def query_neighbours_batched(
    estimator: NearestNeighbors,
    queries: object,
    n_neighbors: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query neighbours in bounded batches to avoid a large distance-result allocation."""
    count = queries.shape[0]
    distances = np.empty((count, n_neighbors), dtype=np.float32)
    indices = np.empty((count, n_neighbors), dtype=np.int32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_distances, batch_indices = estimator.kneighbors(
            queries[start:stop],
            n_neighbors=n_neighbors,
            return_distance=True,
        )
        distances[start:stop] = batch_distances.astype(np.float32, copy=False)
        indices[start:stop] = batch_indices.astype(np.int32, copy=False)
    return distances, indices


def vote_from_neighbours(
    neighbour_labels: np.ndarray,
    distances: np.ndarray,
    k: int,
    weights: str,
    classes: Iterable[int] = range(10),
) -> np.ndarray:
    labels = neighbour_labels[:, :k]
    local_distances = distances[:, :k]
    class_values = np.asarray(tuple(classes), dtype=np.int8)
    if weights == "uniform":
        vote_weights = np.ones_like(local_distances, dtype=np.float32)
    elif weights == "distance":
        zero_mask = local_distances == 0.0
        any_zero = zero_mask.any(axis=1)
        vote_weights = np.divide(
            1.0,
            local_distances,
            out=np.zeros_like(local_distances, dtype=np.float32),
            where=~zero_mask,
        )
        vote_weights[any_zero] = zero_mask[any_zero].astype(np.float32)
    else:
        raise ValueError(f"Unsupported weights: {weights}")
    scores = np.empty((len(labels), len(class_values)), dtype=np.float32)
    for index, class_value in enumerate(class_values):
        scores[:, index] = (vote_weights * (labels == class_value)).sum(axis=1)
    return class_values[np.argmax(scores, axis=1)]


def tune_knn(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    selection_indices: np.ndarray,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    X = features.iloc[selection_indices].reset_index(drop=True)
    y = target.iloc[selection_indices].reset_index(drop=True).to_numpy(dtype=np.int8)
    local_groups = groups.iloc[selection_indices].reset_index(drop=True).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.seed + 10,
    )
    max_k = max(config.knn_values)
    rows: list[dict[str, object]] = []
    for preprocessor_name in config.knn_preprocessors:
        for fold, (train, validation) in enumerate(
            splitter.split(X, y, groups=local_groups), start=1
        ):
            preprocessor = build_preprocessor(X.iloc[train], preprocessor_name)
            transform_start = time.perf_counter()
            X_train = _as_float32(preprocessor.fit_transform(X.iloc[train], y[train]))
            X_validation = _as_float32(preprocessor.transform(X.iloc[validation]))
            transform_seconds = time.perf_counter() - transform_start
            for metric in config.knn_metrics:
                search = NearestNeighbors(
                    n_neighbors=max_k,
                    algorithm="brute",
                    metric=metric,
                    n_jobs=config.n_jobs,
                )
                query_start = time.perf_counter()
                search.fit(X_train)
                distances, neighbour_indices = query_neighbours_batched(
                    search,
                    X_validation,
                    max_k,
                    config.knn_batch_size,
                )
                query_seconds = time.perf_counter() - query_start
                neighbour_labels = y[train][neighbour_indices]
                for k in config.knn_values:
                    for weights in config.knn_weights:
                        predictions = vote_from_neighbours(
                            neighbour_labels,
                            distances,
                            k,
                            weights,
                        )
                        row = {
                            "preprocessor": preprocessor_name,
                            "metric": metric,
                            "k": k,
                            "weights": weights,
                            "fold": fold,
                            "train_count": int(len(train)),
                            "validation_count": int(len(validation)),
                            "transformed_features": int(X_train.shape[1]),
                            "transform_seconds": transform_seconds,
                            "query_seconds": query_seconds,
                        }
                        row.update(classification_metrics(y[validation], predictions))
                        rows.append(row)
    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(destination / "knn_fold_results.csv", index=False)
    group_columns = ["preprocessor", "metric", "k", "weights"]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    summary = (
        fold_results.groupby(group_columns, as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary = summary.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary.to_csv(destination / "knn_candidates.csv", index=False)
    best_row = summary.iloc[0]
    best = {
        "preprocessor": str(best_row["preprocessor"]),
        "metric": str(best_row["metric"]),
        "k": int(best_row["k"]),
        "weights": str(best_row["weights"]),
        "selection_macro_f1_mean": float(best_row["macro_f1_mean"]),
        "selection_macro_f1_std": float(best_row["macro_f1_std"]),
    }
    return summary, best


def fit_predict_knn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    selected: dict[str, object],
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    preprocessor = build_preprocessor(X_train, str(selected["preprocessor"]))
    start = time.perf_counter()
    train_transformed = _as_float32(preprocessor.fit_transform(X_train, y_train))
    test_transformed = _as_float32(preprocessor.transform(X_test))
    transform_seconds = time.perf_counter() - start
    estimator = NearestNeighbors(
        n_neighbors=int(selected["k"]),
        algorithm="brute",
        metric=str(selected["metric"]),
        n_jobs=config.n_jobs,
    )
    query_start = time.perf_counter()
    estimator.fit(train_transformed)
    distances, indices = query_neighbours_batched(
        estimator,
        test_transformed,
        int(selected["k"]),
        config.knn_batch_size,
    )
    query_seconds = time.perf_counter() - query_start
    predictions = vote_from_neighbours(
        y_train[indices],
        distances,
        int(selected["k"]),
        str(selected["weights"]),
    )
    timing = {
        "transform_seconds": transform_seconds,
        "fit_query_seconds": query_seconds,
        "transformed_features": int(train_transformed.shape[1]),
    }
    return predictions, timing
