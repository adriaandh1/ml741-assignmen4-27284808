"""OOF nested aggregation and paired whole-geometry-group bootstrap."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from clustering import METRICS, classification_metrics


def assemble_oof(parts, expected_rows, som_seeds, clustering_seeds):
    """Require exactly one held-out prediction per row and stochastic replicate."""
    lookup = pd.Index(expected_rows)
    predictions = np.full(
        (len(som_seeds), len(clustering_seeds), len(expected_rows)), -1, np.int8
    )
    for part in parts:
        positions = lookup.get_indexer(part["rows"])
        if (positions < 0).any() or len(np.unique(positions)) != len(positions):
            raise ValueError("OOF source identity mismatch")
        s, c = (
            som_seeds.index(part["som_seed"]),
            clustering_seeds.index(part["clustering_seed"]),
        )
        if (predictions[s, c, positions] != -1).any():
            raise ValueError("Duplicate OOF prediction")
        pred = np.asarray(part["predictions"])
        if pred.shape != positions.shape or not np.isin(pred, np.arange(10)).all():
            raise ValueError("Invalid OOF classes/length")
        predictions[s, c, positions] = pred
    if (predictions < 0).any():
        raise ValueError("Incomplete OOF stochastic grid")
    return predictions


def nested_values(values):
    values = np.asarray(values, dtype=float)
    if values.shape[:2] != (5, 5) or not np.isfinite(values).all():
        raise ValueError("Expected complete five-by-five stochastic values")
    som_means = values.mean(axis=1)
    return {
        "mean": som_means.mean(axis=0),
        "som_sd": som_means.std(axis=0, ddof=1),
        "kmeans_sd_mean": values.std(axis=1, ddof=1).mean(axis=0),
        "minimum": values.min(axis=(0, 1)),
        "maximum": values.max(axis=(0, 1)),
    }


def representative(scores, som_seeds, clustering_seeds):
    scores = np.asarray(scores)
    distance = np.abs(scores - np.median(scores))
    candidates = np.argwhere(distance == distance.min())
    return min(
        candidates.tolist(), key=lambda pair: (som_seeds[pair[0]], clustering_seeds[pair[1]])
    )


def macro_from_confusions(confusions):
    counts = np.asarray(confusions, dtype=float).reshape(-1, 10, 10)
    tp = np.diagonal(counts, axis1=1, axis2=2)
    denominator = counts.sum(axis=1) + counts.sum(axis=2)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=1)


def group_confusions(labels, groups, predictions):
    """Sparse sufficient statistics exactly retain each group's full multiplicity."""
    labels, groups, predictions = map(np.asarray, (labels, groups, predictions))
    if (
        predictions.ndim != 2
        or predictions.shape[1] != len(labels)
        or len(groups) != len(labels)
    ):
        raise ValueError("Bootstrap row alignment differs")
    if (
        not np.isin(labels, np.arange(10)).all()
        or not np.isin(predictions, np.arange(10)).all()
    ):
        raise ValueError("Invalid bootstrap classes")
    unique, inverse = np.unique(groups, return_inverse=True)
    row_ids = np.broadcast_to(inverse, predictions.shape).ravel()
    col_ids = (np.arange(len(predictions))[:, None] * 100 + labels * 10 + predictions).ravel()
    stats = csr_matrix(
        (np.ones(len(row_ids), dtype=np.int32), (row_ids, col_ids)),
        shape=(len(unique), len(predictions) * 100),
    )
    stats.sum_duplicates()
    return unique, stats


def group_draw_weights(rng, count):
    return np.bincount(rng.integers(0, count, size=count), minlength=count)


def paired_bootstrap(labels, groups, som_predictions, knn, tree, count=2000, seed=47441):
    som = np.asarray(som_predictions)
    if som.shape != (5, 5, len(labels)):
        raise ValueError("Bootstrap needs all 25 separate OOF SOM vectors")
    predictions = np.concatenate((np.asarray([knn, tree]), som.reshape(25, -1)))
    unique, stats = group_confusions(labels, groups, predictions)
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(count):
        weights = group_draw_weights(rng, len(unique))
        scores = macro_from_confusions(weights @ stats)
        som_score = scores[2:].reshape(5, 5).mean(axis=1).mean()
        records.append(
            {
                "draw": draw + 1,
                "knn": scores[0],
                "tree": scores[1],
                "som": som_score,
                "som_minus_knn": som_score - scores[0],
                "som_minus_tree": som_score - scores[1],
                **{f"som_s{s}_c{c}": scores[2 + s * 5 + c] for s in range(5) for c in range(5)},
            }
        )
        if (draw + 1) % 250 == 0:
            print(f"Group bootstrap {draw + 1}/{count}", flush=True)
    return pd.DataFrame(records)


def metric_records(y, predictions, identity):
    metrics = classification_metrics(y, predictions)
    overall = {**identity, **{name: metrics[name] for name in METRICS}}
    classes = [
        {
            **identity,
            "class": c,
            **{name: metrics[name][c] for name in ("precision", "recall", "f1", "support")},
        }
        for c in range(10)
    ]
    return overall, classes, metrics


def outer_positions(rows, groups, folds, development_rows, fold):
    """Use source identities and reject any development/group/fold leakage."""
    rows, groups, folds = map(np.asarray, (rows, groups, folds))
    if len(rows) != len(groups) or len(rows) != len(folds):
        raise ValueError("Outer identity lengths differ")
    if len(np.unique(rows)) != len(rows) or np.intersect1d(rows, development_rows).size:
        raise ValueError("Duplicate or development rows in assessment")
    if not np.array_equal(np.unique(folds), np.arange(1, 6)) or fold not in range(1, 6):
        raise ValueError("Expected five frozen outer folds")
    check = pd.DataFrame({"group": groups, "fold": folds})
    if (check.groupby("group").fold.nunique() != 1).any():
        raise ValueError("Geometry group crosses outer folds")
    train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if not len(train) or not len(test):
        raise ValueError("Empty outer partition")
    return train, test
