"""Paired resampling of observed geometry groups, with all stochastic pairs retained."""

import numpy as np
import pandas as pd

from assessment_statistics import group_confusions, group_draw_weights, macro_from_confusions
from experiments import fingerprint

PAIRS = [
    ("k150", "k40"),
    ("k400", "k40"),
    ("k400", "k150"),
    ("k150", "knn"),
    ("k150", "tree"),
    ("k400", "knn"),
    ("k400", "tree"),
]
METHOD_ORDER = ["knn", "tree", "k40", "k150", "k400"]
DIFFERENCES = [f"{a}_minus_{b}" for a, b in PAIRS]


def score_pairs(scores):
    """Corresponding SOM/KMeans seed pairs, never randomly matched realisations."""
    scores = np.asarray(scores)
    if scores.shape != (77,):
        raise ValueError("Bootstrap needs two baselines and three complete 25-seed procedures")
    methods = {"knn": np.full((5, 5), scores[0]), "tree": np.full((5, 5), scores[1])}
    for index, name in enumerate(METHOD_ORDER[2:]):
        methods[name] = scores[2 + index * 25 : 2 + (index + 1) * 25].reshape(5, 5)
    return np.array([methods[a] - methods[b] for a, b in PAIRS])


def paired_bootstrap(labels, groups, methods, *, count=2000, seed=47441):
    if set(methods) != set(METHOD_ORDER):
        raise ValueError("Unexpected bootstrap comparison procedures")
    for name in METHOD_ORDER[2:]:
        if methods[name].shape != (5, 5, len(labels)):
            raise ValueError("Incomplete stochastic OOF family in bootstrap")
    if any(methods[name].shape != (len(labels),) for name in METHOD_ORDER[:2]):
        raise ValueError("Incomplete baseline OOF vector")
    predictions = np.concatenate(
        [
            np.array([methods[n] for n in METHOD_ORDER[:2]]),
            *(methods[n].reshape(25, -1) for n in METHOD_ORDER[2:]),
        ]
    )
    unique, sufficient = group_confusions(labels, groups, predictions)
    scores = np.empty((count, 77), float)
    differences = np.empty((count, 7, 5, 5), float)
    fingerprints = []
    rng = np.random.default_rng(seed)
    for draw in range(count):
        weights = group_draw_weights(rng, len(unique))
        scores[draw] = macro_from_confusions(weights @ sufficient)
        differences[draw] = score_pairs(scores[draw])
        fingerprints.append(fingerprint(weights))
        if (draw + 1) % 250 == 0:
            print(f"Post-hoc paired group bootstrap {draw + 1}/{count}", flush=True)
    return {
        "scores": scores,
        "paired_differences": differences,
        "weight_sha256": np.asarray(fingerprints),
        "unique_groups": unique,
        "method_order": np.asarray(METHOD_ORDER),
        "seed": np.array(seed),
    }


def draw_table(result):
    scores, differences = result["scores"], result["paired_differences"]
    count = len(scores)
    rows = {
        "draw": np.arange(1, count + 1),
        "group_weights_sha256": result["weight_sha256"],
        "knn": scores[:, 0],
        "tree": scores[:, 1],
    }
    for index, name in enumerate(METHOD_ORDER[2:]):
        rows[name] = (
            scores[:, 2 + index * 25 : 2 + (index + 1) * 25]
            .reshape(count, 5, 5)
            .mean(axis=2)
            .mean(axis=1)
        )
    mean = differences.mean(axis=3).mean(axis=2)
    for index, name in enumerate(DIFFERENCES):
        rows[name] = mean[:, index]
    return pd.DataFrame(rows)


def verify_original_draws(result, old):
    if len(old) != len(result["scores"]) or not np.array_equal(
        old.draw, np.arange(1, len(old) + 1)
    ):
        raise ValueError("Frozen bootstrap draw identity mismatch")
    scores = result["scores"]
    table = draw_table(result)
    maximum = 0.0
    for name, current in (("knn", scores[:, 0]), ("tree", scores[:, 1]), ("som", table.k40)):
        delta = np.max(np.abs(old[name].to_numpy() - current))
        maximum = max(maximum, float(delta))
        if delta > 1e-15:
            raise ValueError("Original paired-bootstrap scores do not reproduce")
    for s in range(5):
        for c in range(5):
            delta = np.max(np.abs(old[f"som_s{s}_c{c}"].to_numpy() - scores[:, 2 + s * 5 + c]))
            maximum = max(maximum, float(delta))
            if delta > 1e-15:
                raise ValueError("Original per-seed bootstrap score differs")
    return maximum
