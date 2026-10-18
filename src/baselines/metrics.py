"""Metric calculation shared by tuning and final assessment."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = np.arange(10, dtype=np.int8)


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }


def per_class_metrics(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> list[dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    return [
        {
            "class": int(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    ]


def confusion_matrices(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    absolute = confusion_matrix(y_true, y_pred, labels=LABELS)
    normalized = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    return absolute, np.nan_to_num(normalized)
