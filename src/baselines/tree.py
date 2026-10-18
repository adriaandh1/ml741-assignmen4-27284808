"""Classification-tree tuning and final prediction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig
from .preprocessing import build_preprocessor

TREE_PARAMETER_DISTRIBUTIONS = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "model__max_depth": [None, 8, 12, 16, 24, 32, 48],
    "model__min_samples_split": [2, 5, 10, 25, 50, 100],
    "model__min_samples_leaf": [1, 2, 5, 10, 25, 50],
    "model__max_features": [None, "sqrt", "log2", 0.5, 0.75],
    "model__class_weight": [None, "balanced"],
    "model__ccp_alpha": [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
}


def _parameter_key(params: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """Return a stable key for a scikit-learn parameter dictionary."""
    return tuple((key, repr(value)) for key, value in sorted(params.items()))


def tune_tree(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    selection_indices: np.ndarray,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    X = features.iloc[selection_indices].reset_index(drop=True)
    y = target.iloc[selection_indices].reset_index(drop=True).to_numpy(dtype=np.int8)
    local_groups = groups.iloc[selection_indices].reset_index(drop=True).to_numpy()
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X, "tree_all")),
            ("model", DecisionTreeClassifier(random_state=config.seed)),
        ]
    )
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.seed + 20,
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=TREE_PARAMETER_DISTRIBUTIONS,
        n_iter=config.tree_search_iterations,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "weighted_f1": "f1_weighted",
        },
        refit="macro_f1",
        cv=splitter,
        random_state=config.seed + 21,
        n_jobs=config.n_jobs,
        return_train_score=True,
        verbose=1,
    )
    search.fit(X, y, groups=local_groups)
    randomized_results = pd.DataFrame(search.cv_results_)
    randomized_results["search_stage"] = "randomized"
    randomized_results.to_csv(destination / "tree_candidates_randomized.csv", index=False)

    # A seeded random search explores the broad joint space. A second, selection-only
    # search then varies every control around the leading joint configuration. The local
    # refinement avoids accepting a coarse random value (for example, a leaf size) merely
    # because the stronger neighbouring value was not sampled in the same combination.
    evaluated = {_parameter_key(params) for params in randomized_results["params"]}
    refinement_grids: list[dict[str, list[object]]] = []
    for parameter, values in TREE_PARAMETER_DISTRIBUTIONS.items():
        for value in values:
            candidate = dict(search.best_params_)
            candidate[parameter] = value
            key = _parameter_key(candidate)
            if key in evaluated:
                continue
            evaluated.add(key)
            refinement_grids.append({name: [item] for name, item in candidate.items()})

    result_frames = [randomized_results]
    if refinement_grids:
        refinement = GridSearchCV(
            pipeline,
            param_grid=refinement_grids,
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
                "accuracy": "accuracy",
                "weighted_f1": "f1_weighted",
            },
            refit="macro_f1",
            cv=splitter,
            n_jobs=config.n_jobs,
            return_train_score=True,
            verbose=1,
        )
        refinement.fit(X, y, groups=local_groups)
        refined_results = pd.DataFrame(refinement.cv_results_)
        refined_results["search_stage"] = "local_refinement"
        refined_results.to_csv(destination / "tree_candidates_refined.csv", index=False)
        result_frames.append(refined_results)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    results["rank_test_macro_f1"] = (
        results["mean_test_macro_f1"].rank(method="min", ascending=False).astype(int)
    )
    results = results.sort_values(
        ["rank_test_macro_f1", "mean_test_balanced_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)
    results.to_csv(destination / "tree_candidates.csv", index=False)
    best_row = results.iloc[0]
    best_params = {
        key.removeprefix("model__"): value
        for key, value in dict(best_row["params"]).items()
        if key.startswith("model__")
    }
    for key, value in list(best_params.items()):
        if isinstance(value, np.generic):
            best_params[key] = value.item()
    best = {
        "preprocessor": "tree_all",
        "params": best_params,
        "selection_macro_f1_mean": float(best_row["mean_test_macro_f1"]),
        "selection_macro_f1_std": float(best_row["std_test_macro_f1"]),
        "randomized_candidates": int(len(randomized_results)),
        "refinement_candidates": int(len(results) - len(randomized_results)),
        "total_candidates": int(len(results)),
    }
    return results, best


def fit_predict_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    selected: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int], Pipeline]:
    model = DecisionTreeClassifier(random_state=seed, **dict(selected["params"]))
    preprocessor_name = str(selected.get("preprocessor", "tree_all"))
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(X_train, preprocessor_name)), ("model", model)]
    )
    fit_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = pipeline.predict(X_test).astype(np.int8)
    predict_seconds = time.perf_counter() - predict_start
    fitted_tree = pipeline.named_steps["model"]
    timing = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "tree_depth": int(fitted_tree.get_depth()),
        "tree_leaves": int(fitted_tree.get_n_leaves()),
    }
    return predictions, timing, pipeline
