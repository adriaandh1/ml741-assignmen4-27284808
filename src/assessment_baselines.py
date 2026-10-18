"""Read-only, hashed Assignment 2 algorithm reuse with explicit execution controls."""

import json
from time import perf_counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from config import ROOT


def assignment2_modules():
    from baselines import config, neighbours, preprocessing, tree

    return preprocessing, neighbours, tree, config


def settings():
    selected = json.loads((ROOT / "reference/selected_configurations.json").read_text())
    config = json.loads((ROOT / "reference/baseline_config.json").read_text())
    return selected, config


def baseline_predict(train, training_labels, test, selected, method, fold, seed, batch=1024):
    """No held-out labels argument. original transforms/query/voting are reused."""
    preprocess, neighbours, _, _ = assignment2_modules()
    config = selected[method]
    started = perf_counter()
    transformer = preprocess.build_preprocessor(train, config["preprocessor"])
    x = transformer.fit_transform(train, training_labels)
    held = transformer.transform(test)
    if method == "knn":
        x, held = neighbours._as_float32(x), neighbours._as_float32(held)
    transform_seconds = perf_counter() - started
    started = perf_counter()
    if method == "knn":
        estimator = NearestNeighbors(
            n_neighbors=config["k"], algorithm="brute", metric=config["metric"], n_jobs=4
        ).fit(x)
    elif method == "tree":
        estimator = DecisionTreeClassifier(random_state=seed + fold, **config["params"]).fit(
            x, training_labels
        )
    else:
        raise ValueError("Unknown frozen baseline")
    fit_seconds = perf_counter() - started
    started = perf_counter()
    if method == "knn":
        distances, indices = neighbours.query_neighbours_batched(
            estimator, held, config["k"], batch
        )
        predictions = neighbours.vote_from_neighbours(
            training_labels[indices], distances, config["k"], config["weights"]
        )
    else:
        predictions = estimator.predict(held).astype(np.int8)
    timing = {
        "preprocessing_seconds": transform_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": perf_counter() - started,
        "dimension": x.shape[1],
        "fold": fold,
        "method": method,
        "batch_size": batch if method == "knn" else None,
    }
    return predictions, timing, transformer, estimator
