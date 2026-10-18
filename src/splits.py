"""Group-disjoint development and locked assessment indices. never evaluate models."""

from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ASSESSMENT_SEED = 47413


def check_partitions(
    parts: dict[str, np.ndarray], target: np.ndarray, groups: np.ndarray
) -> dict:
    for name, rows in parts.items():
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"Invalid row indices: {name}")
        if len(rows) == 0 or len(np.unique(rows)) != len(rows):
            raise ValueError(f"Empty or repeated row indices: {name}")
        if rows.min() < 0 or rows.max() >= len(target):
            raise ValueError(f"Out-of-range row indices: {name}")
        if set(target[rows]) != set(range(10)):
            raise ValueError(f"Missing class/target in partition: {name}")
    for left, right in combinations(parts, 2):
        if np.intersect1d(parts[left], parts[right]).size:
            raise ValueError(f"Row overlap: {left}/{right}")
        if np.intersect1d(groups[parts[left]], groups[parts[right]]).size:
            raise ValueError(f"Duplicate-group overlap: {left}/{right}")
    return {
        name: {
            "rows": len(rows),
            "groups": int(np.unique(groups[rows]).size),
            "class_counts": {str(c): int((target[rows] == c).sum()) for c in range(10)},
        }
        for name, rows in parts.items()
    }


def split_development(rows: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, validation = next(splitter.split(np.zeros(len(rows)), target[rows], groups[rows]))
    return np.sort(rows[train]), np.sort(rows[validation])


def smoke_rows(
    development: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int, per_class: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = set()
    for c in range(10):
        candidates = np.unique(groups[development[target[development] == c]])
        selected.update(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    return development[np.isin(groups[development], list(selected))]
