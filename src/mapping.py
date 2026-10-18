"""Post-hoc training-only prevalence correction. original endpoint rules unchanged."""

import numpy as np

from clustering import cluster_mapping

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def alpha_key(alpha):
    if alpha not in ALPHAS:
        raise ValueError("Alpha differs from declared grid")
    return f"a{alpha:.2f}"


def prevalence_mapping(training_labels, train_bmu, unit_clusters, centres, alpha):
    """Validation targets are absent. Exact endpoints. float64 intermediate scores."""
    alpha_key(alpha)
    rule = "class_balanced" if alpha == 1 else "majority"
    result = cluster_mapping(training_labels, train_bmu, unit_clusters, centres, rule)
    if alpha not in (0, 1):
        scores = result["counts"].astype(np.float64) / np.power(
            result["class_totals"].astype(np.float64), np.float64(alpha)
        )
        supported = ~result["unsupported"]
        result["class_mapping"][supported] = np.argmax(scores[supported], axis=1)
        missing = result["unsupported"]
        result["class_mapping"][missing] = result["class_mapping"][
            result["fallback_to"][missing]
        ]
        result["clusters_per_class"] = np.bincount(result["class_mapping"], minlength=10)
    return result


def validate_decisions(mapping, alpha, centres):
    """Check cached decisions independently, without fitting/creating a new mapping."""
    from scipy.spatial.distance import cdist

    alpha_key(alpha)
    counts, totals = mapping["counts"], mapping["class_totals"]
    supported = np.flatnonzero(counts.sum(axis=1) > 0)
    unsupported = np.flatnonzero(counts.sum(axis=1) == 0)
    if np.any(totals <= 0) or not len(supported):
        raise ValueError("Invalid cached training support")
    for z in supported:
        if alpha == 1:
            best = 0
            for c in range(1, 10):
                if int(counts[z, c]) * int(totals[best]) > int(counts[z, best]) * int(
                    totals[c]
                ):
                    best = c
        else:
            best = int(
                np.argmax(counts[z].astype(float) / np.power(totals.astype(float), alpha))
            )
        if mapping["class_mapping"][z] != best or mapping["fallback_to"][z] != -1:
            raise ValueError("Cached prevalence decision differs")
    if len(unsupported):
        distances = cdist(centres[unsupported], centres[supported])
        targets = supported[np.argmin(distances, axis=1)]
        if not (
            np.array_equal(mapping["fallback_to"][unsupported], targets)
            and np.array_equal(
                mapping["class_mapping"][unsupported], mapping["class_mapping"][targets]
            )
        ):
            raise ValueError("Cached unsupported-cluster fallback differs")
    if not np.array_equal(mapping["unsupported"], counts.sum(axis=1) == 0):
        raise ValueError("Cached unsupported flags differ")
    if not np.array_equal(
        mapping["clusters_per_class"], np.bincount(mapping["class_mapping"], minlength=10)
    ):
        raise ValueError("Cached mapped-class counts differ")
