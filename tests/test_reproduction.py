import ast
import inspect
import io
import tokenize

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

import experiments
from assessment_baselines import baseline_predict, settings
from assessment_models import train_som
from assessment_statistics import (
    assemble_oof,
    group_confusions,
    macro_from_confusions,
    nested_values,
    outer_positions,
)
from clustering import cluster_codebook, cluster_mapping
from config import ROOT
from mapping import prevalence_mapping
from som import SOMSettings


def test_fixed_schedule_and_validation_exclusion():
    rng = np.random.default_rng(47412)
    training = rng.normal(size=(60, 3))
    arguments = {"settings": SOMSettings(2, 40, 0.75, 13)}
    with threadpool_limits(limits=1):
        first = train_som(training, 47421, **arguments)
        second = train_som(training, 47421, validation=np.full((9, 3), 999.0), **arguments)
    np.testing.assert_array_equal(first.codebook, second.codebook)
    assert [r["epoch"] for r in first.history] == list(range(71))
    assert all(r["radius"] == 0.75 for r in first.history[41:])
    assert second.history[-1]["validation_qe"] > first.history[-1]["train_qe"]


@pytest.mark.parametrize("k", [150, 200, 250, 300, 400])
def test_supported_high_k(k):
    rng = np.random.default_rng(47412)
    codebook = rng.normal(size=(1024, 5))
    hits = np.arange(1, 1025)
    with threadpool_limits(limits=1):
        fit = cluster_codebook(codebook, hits, k, "hit_weighted", 47431)
    assert len(np.unique(fit["unit_clusters"])) == k
    assert fit["cluster_hit_support"].sum() == hits.sum()


@pytest.mark.parametrize("alpha,rule", [(0, "majority"), (1, "class_balanced")])
def test_mapping_endpoints(alpha, rule):
    labels = np.concatenate([np.repeat(0, 20), np.arange(1, 10)])
    bmu = np.arange(len(labels)) % 4
    units = np.arange(4)
    centres = np.arange(4).reshape(-1, 1).astype(float)
    expected = cluster_mapping(labels, bmu, units, centres, rule)
    actual = prevalence_mapping(labels, bmu, units, centres, alpha)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_partial_mapping_training_only_and_ties(alpha):
    labels = np.tile(np.arange(10), 3)
    bmu = np.repeat([0, 1, 2], 10)
    units = np.arange(4)
    centres = np.arange(4).reshape(-1, 1).astype(float)
    actual = prevalence_mapping(labels, bmu, units, centres, alpha)
    reference = actual["counts"] / actual["class_totals"] ** alpha
    np.testing.assert_array_equal(actual["class_mapping"][:3], reference[:3].argmax(axis=1))
    np.testing.assert_array_equal(actual["class_mapping"], 0)
    assert actual["fallback_to"][3] == 2
    assert "validation" not in inspect.signature(prevalence_mapping).parameters


def test_group_bootstrap_and_nested_aggregation():
    labels = np.array([0, 1, 0, 1])
    groups = np.array([3, 3, 9, 9])
    predictions = np.array([[0, 1, 1, 1], [1, 1, 0, 0]])
    _, sparse = group_confusions(labels, groups, predictions)
    counts = (np.array([2, 1]) @ sparse).reshape(2, 10, 10)
    expanded = np.array([0, 1, 0, 1, 2, 3])
    for i in range(2):
        brute = np.zeros((10, 10))
        np.add.at(brute, (labels[expanded], predictions[i, expanded]), 1)
        np.testing.assert_array_equal(counts[i], brute)
    assert np.isfinite(macro_from_confusions(counts)).all()
    values = np.arange(25).reshape(5, 5)
    assert nested_values(values)["som_sd"] == values.mean(axis=1).std(ddof=1)


def test_outer_fold_and_oof_guards():
    rows = np.arange(100)
    groups = np.repeat(np.arange(50), 2)
    folds = np.repeat(np.arange(1, 6), 20)
    train, held = outer_positions(rows, groups, folds, np.array([200]), 1)
    assert len(train) == 80 and len(held) == 20
    with pytest.raises(ValueError):
        outer_positions(rows, groups, folds, np.array([0]), 1)
    parts = [
        {
            "rows": rows[:50],
            "predictions": np.zeros(50, dtype=np.int8),
            "som_seed": 47421,
            "clustering_seed": 47431,
        },
        {
            "rows": rows[50:],
            "predictions": np.ones(50, dtype=np.int8),
            "som_seed": 47421,
            "clustering_seed": 47431,
        },
    ]
    result = assemble_oof(parts, rows, [47421], [47431])
    assert result.shape == (1, 1, 100)
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_oof(parts + parts[:1], rows, [47421], [47431])


def test_baselines_are_local_and_training_only(source_frame):
    selected, config = settings()
    assert config["seed"] > 0
    train = source_frame.iloc[:200].drop(columns="attack_cat")
    held = source_frame.iloc[200:].drop(columns="attack_cat")
    labels = source_frame.attack_cat.to_numpy(dtype=np.int8)[:200]
    with threadpool_limits(limits=1):
        for method in ("knn", "tree"):
            result = baseline_predict(train, labels, held, selected, method, 1, config["seed"])
            assert result[0].shape == (40,)
            assert np.isin(result[0], np.arange(10)).all()


def test_cache_rejects_changed_inputs_and_corruption(tmp_path):
    path = tmp_path / "fit"
    path.mkdir()
    (path / "state.txt").write_text("saved")
    expected = {"seed": 47421}
    experiments.finish(path, expected)
    assert experiments.complete(path, expected)
    with pytest.raises(ValueError, match="inputs or settings"):
        experiments.complete(path, {"seed": 47422})
    (path / "state.txt").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        experiments.complete(path, expected)


def test_assessment_orchestration_on_synthetic_data(monkeypatch, tmp_path, source_frame):
    frame = source_frame.iloc[:100].copy()
    frame["attack_cat"] = np.tile(np.arange(10), 10)
    rows = np.arange(100)
    parts = {
        "train": np.array([200]),
        "validation": np.array([201]),
        "assessment": rows,
        "assessment_fold_by_source_row": np.repeat(np.arange(1, 6), 20),
    }
    monkeypatch.setattr(experiments, "OUT", tmp_path / "runs")
    monkeypatch.setattr(experiments, "partitions", lambda: (parts, np.arange(202)))
    monkeypatch.setattr(experiments, "read_source", lambda _: frame)

    class Model:
        codebook = np.arange(1024, dtype=float).reshape(-1, 1)

    monkeypatch.setattr(experiments, "fit_map", lambda *a: Model())
    monkeypatch.setattr(
        experiments,
        "bmus",
        lambda x, *a, **kw: (np.arange(len(x), dtype=np.int32), None, None),
    )
    calls = []

    def fit(directory, model, hits, k, weight, seed, train_bmu, held_bmu, labels):
        calls.append((k, weight, seed, len(labels)))
        return {
            "unit_clusters": np.arange(1024) % k,
            "centres": np.arange(k, dtype=float).reshape(-1, 1),
        }

    monkeypatch.setattr(experiments, "fit_clusters", fit)

    def baseline(train, y, held, *args):
        assert not np.intersect1d(train.index, held.index).size
        return np.zeros(len(held), dtype=np.int8), {}, None, None

    monkeypatch.setattr(experiments, "baseline_predict", baseline)
    result = experiments.assessment()
    assert result["rows"] == 100
    assert len(calls) == 375
    assert {c[0] for c in calls} == {40, 150, 400}
    assert all(c[1] == "hit_weighted" and c[3] == 80 for c in calls)
    with np.load(tmp_path / "runs/assessment/oof.npz", allow_pickle=False) as saved:
        for name in ("k40", "k150", "k400"):
            assert saved[name].shape == (5, 5, 100)
            assert np.isin(saved[name], np.arange(10)).all()
    for fold in range(1, 6):
        metadata = pd.read_json(
            tmp_path / f"runs/assessment/fold_{fold}/preprocessing.json", typ="series"
        )
        assert metadata["fit_row_count"] == 80


def test_comment_style():
    for folder in ("src", "tests", "report/scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text()
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    assert ";" not in token.string and "\u2014" not in token.string
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        assert ";" not in doc and "\u2014" not in doc
