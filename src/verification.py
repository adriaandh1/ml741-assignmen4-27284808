"""Check the submission snapshot and saved prediction metrics."""

import numpy as np
import pandas as pd

from clustering import METRICS, classification_metrics
from config import ROOT
from data import DATA_HASH, METADATA_HASH, verify_file


def verify_local_data():
    for name, digest in [("networkTraffic.csv", DATA_HASH), ("features.csv", METADATA_HASH)]:
        path = ROOT / "data" / name
        if path.exists():
            verify_file(path, digest)


def load_predictions():
    base = ROOT / "results/assessment"
    with np.load(base / "identities.npz", allow_pickle=False) as saved:
        identity = {key: saved[key].copy() for key in saved.files}
    rows, groups, folds = (identity[key] for key in ("rows", "groups", "folds"))
    if len(rows) != 206138 or len(np.unique(rows)) != len(rows):
        raise ValueError("Assessment row identity differs")
    if np.intersect1d(rows, identity["development_rows"]).size:
        raise ValueError("Development rows entered assessment")
    if not np.array_equal(np.unique(folds), np.arange(1, 6)):
        raise ValueError("Assessment folds differ")
    if (pd.DataFrame({"g": groups, "f": folds}).groupby("g").f.nunique() != 1).any():
        raise ValueError("Geometry group crosses folds")
    methods = {}
    for name, path in [
        ("k40", base / "som_oof_predictions.npz"),
        ("k150", ROOT / "results/exploratory/chunk4_6/oof_predictions/k150.npz"),
        ("k400", ROOT / "results/exploratory/chunk4_6/oof_predictions/k400.npz"),
    ]:
        with np.load(path, allow_pickle=False) as saved:
            for key in ("rows", "groups", "folds", "labels"):
                if not np.array_equal(saved[key], identity[key]):
                    raise ValueError("Saved prediction rows or labels differ")
            methods[name] = saved["predictions"].copy()
        if methods[name].shape != (5, 5, len(rows)):
            raise ValueError("Incomplete stochastic predictions")
    for name in ("knn", "tree"):
        prediction = np.full(len(rows), -1, dtype=np.int8)
        for fold in range(1, 6):
            path = base / f"folds/fold_{fold}/{name}/predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                held = np.flatnonzero(folds == fold)
                if not np.array_equal(saved["test_rows"], rows[held]):
                    raise ValueError("Baseline source-row order differs")
                if np.intersect1d(saved["train_rows"], rows[held]).size:
                    raise ValueError("Baseline training rows overlap held-out rows")
                prediction[held] = saved["predictions"]
        methods[name] = prediction
    if any(not np.isin(value, np.arange(10)).all() for value in methods.values()):
        raise ValueError("Invalid or missing class predictions")
    return identity, methods


def verify():
    verify_local_data()
    identity, methods = load_predictions()
    summary = pd.read_csv(
        ROOT / "results/exploratory/chunk4_6/nested_summary.csv",
        float_precision="round_trip",
    )
    for method, predictions in methods.items():
        repetitions = predictions.reshape(-1, len(identity["rows"]))
        scores = [classification_metrics(identity["labels"], p) for p in repetitions]
        for metric in METRICS:
            values = np.array([score[metric] for score in scores])
            if len(values) == 25:
                values = values.reshape(5, 5).mean(axis=1)
            expected = summary.loc[
                (summary.method == method) & (summary.metric == metric), "mean"
            ].item()
            if not np.isclose(values.mean(), expected, rtol=0, atol=2e-15):
                raise ValueError(f"Saved {method} {metric} does not reproduce")
    return {
        "assessment_rows": len(identity["rows"]),
        "stochastic_replicates_per_som_procedure": 25,
        "som_fits": 0,
        "kmeans_fits": 0,
        "baseline_fits": 0,
        "status": "passed",
    }
