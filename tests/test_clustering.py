import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from clustering import (
    classification_metrics,
    cluster_codebook,
    cluster_mapping,
    codebook_weights,
    condition_id,
    nested_summary,
    predict_from_bmu,
    select_condition,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(47412)
    return rng.normal(size=(64, 4)), np.arange(1, 65, dtype=np.int64)


def test_kmeans_exact_k_determinism_and_seed_distinction(toy):
    w, hits = toy
    with threadpool_limits(limits=1):
        a = cluster_codebook(w, hits, 10, "uniform", 47431)
        b = cluster_codebook(w, hits, 10, "uniform", 47431)
        c = cluster_codebook(w, hits, 10, "uniform", 47432)
    assert len(np.unique(a["unit_clusters"])) == 10
    np.testing.assert_array_equal(a["unit_clusters"], b["unit_clusters"])
    np.testing.assert_array_equal(a["centres"], b["centres"])
    assert not np.array_equal(a["centres"], c["centres"])


def test_weight_modes_exact(toy):
    _, hits = toy
    np.testing.assert_array_equal(codebook_weights(hits, "uniform"), np.ones(64))
    np.testing.assert_array_equal(codebook_weights(hits, "hit_weighted"), hits)


def test_dead_units_still_assigned_and_insufficient_support_rejected(toy):
    w, hits = toy
    hits[:20] = 0
    with threadpool_limits(limits=1):
        fit = cluster_codebook(w, hits, 10, "hit_weighted", 47431)
    assert len(fit["unit_clusters"]) == 64
    assert fit["training_dead_units"] == 20
    assert sum(fit["cluster_hit_support"]) == sum(hits)
    with pytest.raises(ValueError, match="positive-weight"):
        cluster_codebook(w, hits, 50, "hit_weighted", 47431)


def test_labels_cannot_enter_clustering(toy):
    w, hits = toy
    assert "y" not in inspect.signature(cluster_codebook).parameters
    with pytest.raises(TypeError):
        cluster_codebook(w, hits, 10, "uniform", 47431, y=np.arange(64))
    with pytest.raises(TypeError, match="DataFrames"):
        cluster_codebook(pd.DataFrame(w).assign(attack_cat=0), hits, 10, "uniform", 47431)


def mapping_case():
    # Cluster 0 has class0 count 6 vs class1 count 2. class1 is rarer globally.
    y = np.array([0] * 6 + [1] * 2 + [0] * 14 + list(range(2, 10)), dtype=np.int8)
    bmu = np.array([0] * 8 + [1] * 22, dtype=np.int32)
    units = np.array([0, 1], dtype=np.int32)
    centres = np.array([[0.0, 0.0], [2.0, 0.0]])
    return y, bmu, units, centres


def test_majority_balanced_frequencies_and_multiple_clusters():
    y, bmu, units, centres = mapping_case()
    majority = cluster_mapping(y, bmu, units, centres, "majority")
    balanced = cluster_mapping(y, bmu, units, centres, "class_balanced")
    assert majority["class_mapping"].tolist() == [0, 0]
    assert majority["clusters_per_class"][0] == 2
    assert balanced["class_mapping"].tolist() == [1, 2]  # 2..9 tie at full capture
    assert balanced["class_totals"].tolist() == [20, 2] + [1] * 8
    assert balanced["counts"][0, :2].tolist() == [6, 2]


@pytest.mark.parametrize("rule", ["majority", "class_balanced"])
def test_class_ties_lowest_id(rule):
    y = np.arange(10, dtype=np.int8)
    bmu = np.zeros(10, dtype=np.int32)
    mapping = cluster_mapping(y, bmu, np.array([0], dtype=np.int32), np.zeros((1, 2)), rule)
    assert mapping["class_mapping"].tolist() == [0]


def test_fallback_nearest_supported_no_chaining_and_tie():
    y = np.tile(np.arange(10, dtype=np.int8), 2)
    bmu = np.repeat(np.array([0, 1], dtype=np.int32), 10)
    units = np.arange(4, dtype=np.int32)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [3.5, 0.0]])
    mapping = cluster_mapping(y, bmu, units, centres, "majority")
    assert mapping["unsupported"].tolist() == [False, False, True, True]
    assert mapping["fallback_to"].tolist() == [-1, -1, 0, 1]
    assert "validation_labels" not in inspect.signature(cluster_mapping).parameters


def test_prediction_chain():
    units = np.array([2, 0, 1, 2], dtype=np.int32)
    mapping = {
        "class_mapping": np.array([9, 4, 1], dtype=np.int8),
        "unsupported": np.array([False, True, False]),
    }
    prediction, used = predict_from_bmu(np.array([3, 1, 2, 0], dtype=np.int32), units, mapping)
    assert prediction.tolist() == [1, 9, 4, 1]
    assert used.tolist() == [False, False, True, False]


def test_metric_reference_and_no_prediction_classes():
    y = np.repeat(np.arange(10, dtype=np.int8), 2)
    pred = y.copy()
    pred[1::2] = 0
    result = classification_metrics(y, pred)
    assert result["macro_f1"] == pytest.approx(
        f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0)
    )
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    np.testing.assert_array_equal(
        result["confusion"], confusion_matrix(y, pred, labels=list(range(10)))
    )
    zero = classification_metrics(y, np.zeros(len(y), dtype=np.int8))
    assert zero["f1"][1:].tolist() == [0] * 9
    assert zero["macro_f1"] == pytest.approx(1 / 55)


def replicate_grid():
    rows = []
    for s in range(47421, 47426):
        for c in range(47431, 47436):
            value = (s - 47421) * 0.1 + (c - 47431) * 0.01
            rows.append(
                {
                    "side": 8,
                    "k": 10,
                    "weighting": "uniform",
                    "mapping": "majority",
                    "som_seed": s,
                    "clustering_seed": c,
                    "state": "complete",
                    **{
                        metric: value
                        for metric in (
                            "macro_f1",
                            "balanced_accuracy",
                            "accuracy",
                            "weighted_f1",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def test_nested_aggregation_not_flat_sd():
    runs = replicate_grid()
    summary, som = nested_summary(runs)
    assert summary.iloc[0].eligible
    assert summary.iloc[0].macro_f1_mean == pytest.approx(0.22)
    assert summary.iloc[0].macro_f1_som_sd == pytest.approx(
        np.std([0.02, 0.12, 0.22, 0.32, 0.42], ddof=1)
    )
    assert summary.iloc[0].macro_f1_kmeans_sd_mean == pytest.approx(
        np.std([0, 0.01, 0.02, 0.03, 0.04], ddof=1)
    )
    assert len(som) == 5
    assert summary.iloc[0].macro_f1_som_sd != pytest.approx(runs.macro_f1.std())


def test_incomplete_or_failed_replicates_ineligible():
    runs = replicate_grid()
    assert not nested_summary(runs.iloc[:-1])[0].iloc[0].eligible
    runs.loc[0, "state"] = "failed"
    assert not nested_summary(runs)[0].iloc[0].eligible
    with pytest.raises(ValueError, match="No complete"):
        select_condition(nested_summary(runs)[0])
    with pytest.raises(ValueError, match="Duplicate"):
        nested_summary(pd.concat([runs, runs.iloc[:1]]))


def selection_row(**kwargs):
    row = dict(
        side=8,
        k=10,
        weighting="uniform",
        mapping="majority",
        eligible=True,
        macro_f1_mean=0.5,
        balanced_accuracy_mean=0.6,
        macro_f1_som_sd=0.01,
    )
    row.update(kwargs)
    row["condition_id"] = condition_id(row["side"], row["k"], row["weighting"], row["mapping"])
    return row


@pytest.mark.parametrize(
    "key,a,b",
    [
        ("macro_f1_mean", 0.6, 0.5),
        ("balanced_accuracy_mean", 0.7, 0.6),
        ("macro_f1_som_sd", 0.01, 0.02),
        ("side", 8, 16),
        ("k", 10, 15),
        ("weighting", "uniform", "hit_weighted"),
        ("mapping", "majority", "class_balanced"),
    ],
)
def test_each_selection_priority(key, a, b):
    first, second = selection_row(**{key: a}), selection_row(**{key: b})
    chosen = select_condition(pd.DataFrame([second, first]))
    assert chosen[key] == a


def test_primary_tolerance_then_balanced_accuracy_and_lexical():
    first = selection_row(macro_f1_mean=0.5, balanced_accuracy_mean=0.8)
    second = selection_row(macro_f1_mean=0.5 + 5e-13, balanced_accuracy_mean=0.7)
    assert select_condition(pd.DataFrame([second, first]))["balanced_accuracy_mean"] == 0.8
    second["macro_f1_mean"] = 0.5 + 2e-12
    assert select_condition(pd.DataFrame([second, first]))["balanced_accuracy_mean"] == 0.7
    first["condition_id"], second["condition_id"] = "a", "b"
    second.update(macro_f1_mean=0.5, balanced_accuracy_mean=0.8)
    assert select_condition(pd.DataFrame([second, first]))["condition_id"] == "a"
