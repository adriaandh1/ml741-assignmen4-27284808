"""Codebook-only KMeans, training-label mapping and nested validation selection."""

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from som import matrix

KMEANS = dict(init="k-means++", n_init=10, max_iter=500, tol=1e-4, algorithm="lloyd")
CONDITION = ["side", "k", "weighting", "mapping"]
METRICS = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]


def integer_vector(values, *, length=None, upper=None):
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError("Expected a one-dimensional integer array")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("Invalid integer array")
    if length is not None and len(values) != length:
        raise ValueError("Integer array length differs")
    if upper is not None and np.any(values >= upper):
        raise ValueError("Integer index out of range")
    return values


def codebook_weights(hits, mode):
    integer_vector(hits)
    if mode == "uniform":
        return np.ones(len(hits), dtype=float)
    if mode == "hit_weighted":
        return hits.astype(float)
    raise ValueError("Unknown clustering weighting")


def cluster_codebook(codebook, hits, k, mode, seed):
    """No labels argument: only prototypes and descriptive training support."""
    w = matrix(codebook)
    integer_vector(hits, length=len(w))
    weights = codebook_weights(hits, mode)
    if not 2 <= k < len(w) or np.count_nonzero(weights) < k:
        raise ValueError("Insufficient positive-weight units for declared K")
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = KMeans(n_clusters=k, random_state=seed, **KMEANS).fit(w, sample_weight=weights)
    runtime = perf_counter() - started
    if caught:
        raise ValueError("KMeans warning: " + "; ".join(str(item.message) for item in caught))
    units = model.predict(w).astype(np.int32)
    if len(np.unique(units)) != k or not np.isfinite(model.cluster_centers_).all():
        raise ValueError("KMeans did not produce exactly K nonempty codebook clusters")
    if not np.isfinite(model.inertia_):
        raise ValueError("Non-finite KMeans inertia")
    unit_counts = np.bincount(units, minlength=k)
    support = np.bincount(units, weights=hits, minlength=k).astype(np.int64)
    return {
        "centres": model.cluster_centers_,
        "unit_clusters": units,
        "unit_counts": unit_counts,
        "cluster_hit_support": support,
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
        "fit_seconds": runtime,
        "silhouette": float(silhouette_score(w, units, metric="euclidean")),
        "training_dead_units": int(np.count_nonzero(hits == 0)),
        "zero_support_clusters": int(np.count_nonzero(support == 0)),
        "warnings": [],
    }


def cluster_mapping(training_labels, train_bmu, unit_clusters, centres, rule):
    y = integer_vector(training_labels, length=len(train_bmu), upper=10)
    integer_vector(train_bmu, upper=len(unit_clusters))
    k = len(centres)
    integer_vector(unit_clusters, upper=k)
    totals = np.bincount(y, minlength=10)
    if not (totals > 0).all():
        raise ValueError("All ten training classes must be available")
    counts = np.zeros((k, 10), dtype=np.int64)
    np.add.at(counts, (unit_clusters[train_bmu], y), 1)
    supported = np.flatnonzero(counts.sum(axis=1) > 0)
    if not len(supported):
        raise ValueError("No supported cluster exists")
    labels = np.full(k, -1, dtype=np.int8)
    for z in supported:
        if rule == "majority":
            labels[z] = np.argmax(counts[z])
        elif rule == "class_balanced":
            # Exact rational comparison: ascending iteration preserves lowest-class ties.
            best = 0
            for c in range(1, 10):
                if int(counts[z, c]) * int(totals[best]) > int(counts[z, best]) * int(
                    totals[c]
                ):
                    best = c
            labels[z] = best
        else:
            raise ValueError("Unknown mapping rule")
    fallback = np.full(k, -1, dtype=np.int32)
    unsupported = np.flatnonzero(counts.sum(axis=1) == 0)
    if len(unsupported):
        distance = cdist(centres[unsupported], centres[supported], metric="euclidean")
        fallback[unsupported] = supported[np.argmin(distance, axis=1)]
        labels[unsupported] = labels[fallback[unsupported]]
    return {
        "counts": counts,
        "class_totals": totals,
        "class_mapping": labels,
        "fallback_to": fallback,
        "unsupported": fallback >= 0,
        "clusters_per_class": np.bincount(labels, minlength=10),
    }


def predict_from_bmu(bmu, unit_clusters, mapping):
    integer_vector(bmu, upper=len(unit_clusters))
    clusters = unit_clusters[bmu]
    return mapping["class_mapping"][clusters], mapping["unsupported"][clusters]


def classification_metrics(y, predicted):
    integer_vector(y, upper=10)
    integer_vector(predicted, length=len(y), upper=10)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    support, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    correct = confusion.diagonal()
    precision = np.divide(correct, predicted_count, out=np.zeros(10), where=predicted_count > 0)
    recall = np.divide(correct, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=precision + recall > 0,
    )
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(correct.sum() / len(y)),
        "weighted_f1": float(f1 @ support / len(y)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion": confusion,
        "predicted_classes": int(np.count_nonzero(predicted_count)),
    }


def structural_metrics(labels, observation_clusters, k):
    counts = np.zeros((k, 10), dtype=np.int64)
    np.add.at(counts, (observation_clusters, labels), 1)
    return {
        "purity": float(counts.max(axis=1).sum() / len(labels)),
        "nmi": float(
            normalized_mutual_info_score(
                labels, observation_clusters, average_method="arithmetic"
            )
        ),
        "ari": float(adjusted_rand_score(labels, observation_clusters)),
    }


def condition_id(side, k, weighting, mapping):
    return f"log_standard-{side}x{side}-k{k}-{weighting}-{mapping}"


def nested_summary(runs):
    rows, som_rows = [], []
    for key, group in runs.groupby(CONDITION, sort=True):
        if group.duplicated(["som_seed", "clustering_seed"]).any():
            raise ValueError("Duplicate stochastic replicate")
        expected = {(s, c) for s in range(47421, 47426) for c in range(47431, 47436)}
        actual = set(zip(group.som_seed, group.clustering_seed, strict=True))
        complete = actual == expected and group.state.eq("complete").all()
        valid = group[group.state == "complete"]
        row = dict(zip(CONDITION, key, strict=True))
        row.update(condition_id=condition_id(*key), eligible=complete, evaluations=len(group))
        for som_seed, subgroup in valid.groupby("som_seed"):
            item = {**dict(zip(CONDITION, key, strict=True)), "som_seed": som_seed}
            for metric in METRICS:
                item[f"{metric}_mean"] = float(subgroup[metric].mean())
                item[f"{metric}_kmeans_sd"] = float(subgroup[metric].std(ddof=1))
            som_rows.append(item)
        for metric in METRICS:
            by_som = valid.groupby("som_seed")[metric]
            means = by_som.mean()
            row[f"{metric}_mean"] = float(means.mean())
            row[f"{metric}_som_sd"] = float(means.std(ddof=1))
            row[f"{metric}_kmeans_sd_mean"] = float(by_som.std(ddof=1).mean())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(som_rows)


def select_condition(summary):
    eligible = summary[summary.eligible].copy()
    if eligible.empty:
        raise ValueError("No complete downstream condition eligible")
    maximum = eligible.macro_f1_mean.max()
    tied = eligible[eligible.macro_f1_mean >= maximum - 1e-12].copy()
    tied["weight_order"] = tied.weighting.map({"uniform": 0, "hit_weighted": 1})
    tied["map_order"] = tied.mapping.map({"majority": 0, "class_balanced": 1})
    return (
        tied.sort_values(
            [
                "balanced_accuracy_mean",
                "macro_f1_som_sd",
                "side",
                "k",
                "weight_order",
                "map_order",
                "condition_id",
            ],
            ascending=[False, True, True, True, True, True, True],
        )
        .iloc[0]
        .to_dict()
    )
