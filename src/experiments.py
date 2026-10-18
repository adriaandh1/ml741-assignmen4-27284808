"""Reproduce the fixed experiments in a separate output directory."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from artifacts import write_json
from assessment_baselines import baseline_predict, settings
from assessment_models import train_som
from assessment_statistics import nested_values, outer_positions
from clustering import (
    METRICS,
    classification_metrics,
    cluster_codebook,
    nested_summary,
    predict_from_bmu,
    select_condition,
)
from config import ROOT
from data import DATA_HASH, METADATA_HASH, descriptive, sha256, validate_schema, verify_file
from mapping import ALPHAS, prevalence_mapping
from preprocessing import SOMPreprocessor
from som import BatchSOM, SOMSettings, bmus, diagnostics

SOM_SEEDS = tuple(range(47421, 47426))
CLUSTERING_SEEDS = tuple(range(47431, 47436))
SIDES = (8, 16, 24, 32)
REPRESENTATIONS = ("standard", "log_standard")
OUT = ROOT / "runs"


def fingerprint(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def identity(**parameters):
    source = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    return {"data_sha256": DATA_HASH, "implementation": source, **parameters}


def complete(directory, expected):
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record["identity"] != expected:
        raise ValueError(f"Cached inputs or settings differ: {directory}")
    for name, digest in record["files"].items():
        verify_file(directory / name, digest)
    return True


def finish(directory, expected):
    files = {
        p.name: sha256(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "complete.json"
    }
    write_json(directory / "complete.json", {"identity": expected, "files": files})


def partitions():
    with np.load(ROOT / "results/splits/partition_indices.npz", allow_pickle=False) as saved:
        parts = {key: saved[key].copy() for key in saved.files}
    with np.load(ROOT / "results/splits/group_indices.npz", allow_pickle=False) as saved:
        groups = saved["geometry_group"].copy()
    names = ("train", "validation", "assessment")
    if [len(parts[name]) for name in names] != [41229, 10306, 206138]:
        raise ValueError("Frozen partition size differs")
    for i, name in enumerate(names):
        rows = parts[name]
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Repeated source row")
        for other in names[i + 1 :]:
            if np.intersect1d(rows, parts[other]).size:
                raise ValueError("Partition row overlap")
            if np.intersect1d(groups[rows], groups[parts[other]]).size:
                raise ValueError("Partition geometry overlap")
    return parts, groups


def read_source(rows):
    verify_file(ROOT / "data/networkTraffic.csv", DATA_HASH)
    verify_file(ROOT / "data/features.csv", METADATA_HASH)
    retained = set(rows.tolist())
    frame = pd.read_csv(
        ROOT / "data/networkTraffic.csv",
        na_values=["?"],
        keep_default_na=False,
        skiprows=lambda line: line > 0 and line - 1 not in retained,
    )
    frame.index = rows
    validate_schema(frame, full=False)
    if not np.array_equal(frame.id.to_numpy(), rows + 1):
        raise ValueError("Source-row ordering differs")
    return frame


def development():
    parts, _ = partitions()
    rows = np.sort(np.concatenate([parts["train"], parts["validation"]]))
    frame = read_source(rows)
    features = descriptive(frame, reduced=True)
    return parts, frame, features


def prepare():
    parts, _, features = development()
    expected = identity(stage="prepare")
    destination = OUT / "preprocessing"
    if complete(destination, expected):
        return {"stage": "prepare", "cached": True}
    destination.mkdir(parents=True, exist_ok=True)
    for representation in REPRESENTATIONS:
        processor = SOMPreprocessor(representation).fit(features.loc[parts["train"]])
        np.savez_compressed(
            destination / f"{representation}.npz",
            train=processor.transform(features.loc[parts["train"]]),
            validation=processor.transform(features.loc[parts["validation"]]),
        )
        write_json(destination / f"{representation}.json", processor.metadata())
    finish(destination, expected)
    return {"stage": "prepare", "training_rows": 41229, "validation_rows": 10306}


def prepared(representation):
    prepare()
    with np.load(OUT / f"preprocessing/{representation}.npz", allow_pickle=False) as saved:
        return saved["train"].copy(), saved["validation"].copy()


def fit_map(directory, training, validation, seed, side):
    settings_ = SOMSettings(side, 40, 0.75, 2048)
    expected = identity(
        seed=seed,
        settings=asdict(settings_),
        refinement_epochs=30,
        training=fingerprint(training),
        validation=fingerprint(validation),
    )
    if complete(directory, expected):
        return BatchSOM.load(directory / "codebook.npz")
    directory.mkdir(parents=True, exist_ok=True)
    model = train_som(training, seed, settings=settings_, validation=validation)
    model.save(directory / "codebook.npz")
    pd.DataFrame(model.history).to_csv(directory / "history.csv", index=False)
    metrics = {}
    assignments = {}
    for name, values in (("train", training), ("validation", validation)):
        metrics[name], state = diagnostics(
            values, model.codebook, model.coordinates, model.centroid
        )
        assignments[f"{name}_bmu"] = state["first"]
        assignments[f"{name}_hits"] = state["hits"]
    np.savez_compressed(directory / "assignments.npz", **assignments)
    write_json(directory / "metrics.json", metrics)
    finish(directory, expected)
    return model


def som():
    with threadpool_limits(limits=4):
        for representation in REPRESENTATIONS:
            training, validation = prepared(representation)
            for side in SIDES:
                for seed in SOM_SEEDS:
                    name = f"{representation}-{side}x{side}-s{seed}"
                    print(f"SOM {name}", flush=True)
                    fit_map(OUT / "som" / name, training, validation, seed, side)
    return {"stage": "som", "conditions": 8, "maps": 40, "epochs": 70}


def fit_clusters(directory, model, hits, k, weight, seed, train_bmu, validation_bmu, labels):
    expected = identity(
        codebook=fingerprint(model.codebook),
        hits=fingerprint(hits),
        train_bmu=fingerprint(train_bmu),
        validation_bmu=fingerprint(validation_bmu),
        training_labels=fingerprint(labels),
        k=k,
        weighting=weight,
        clustering_seed=seed,
    )
    if complete(directory, expected):
        with np.load(directory / "state.npz", allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}
    fit = cluster_codebook(model.codebook, hits, k, weight, seed)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: fit[name] for name in ("centres", "unit_clusters", "unit_counts")}
    np.savez_compressed(directory / "state.npz", **state)
    write_json(
        directory / "diagnostics.json",
        {key: value for key, value in fit.items() if key not in state},
    )
    finish(directory, expected)
    return state


def write_metrics(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "replicates.csv", index=False)
    summary, by_som = nested_summary(frame)
    summary.to_csv(directory / "summary.csv", index=False)
    by_som.to_csv(directory / "som_means.csv", index=False)
    return summary


def clustering():
    parts, frame, _ = development()
    labels = {
        name: frame.loc[parts[name], "attack_cat"].to_numpy(dtype=np.int8)
        for name in ("train", "validation")
    }
    training, validation = prepared("log_standard")
    studies = [
        ("original", (8, 16, 32), (10, 15, 20, 30, 40), ("uniform", "hit_weighted"), (0, 1)),
        ("sensitivity", (32,), (40, 50, 60, 80, 100, 150), ("hit_weighted",), ALPHAS),
        ("extended", (32,), (200, 250, 300, 400), ("hit_weighted",), ALPHAS),
    ]
    with threadpool_limits(limits=4):
        for study, sides, counts, weights, alphas in studies:
            records = []
            for side in sides:
                for seed in SOM_SEEDS:
                    path = OUT / f"som/log_standard-{side}x{side}-s{seed}/codebook.npz"
                    model = BatchSOM.load(path)
                    train_bmu = bmus(training, model.codebook, second=False)[0]
                    val_bmu = bmus(validation, model.codebook, second=False)[0]
                    hits = np.bincount(train_bmu, minlength=side**2)
                    for k in counts:
                        for weight in weights:
                            for cluster_seed in CLUSTERING_SEEDS:
                                name = f"{side}-s{seed}-k{k}-{weight}-c{cluster_seed}"
                                print(f"Clustering {study} {name}", flush=True)
                                fit = fit_clusters(
                                    OUT / "clustering" / study / name,
                                    model,
                                    hits,
                                    k,
                                    weight,
                                    cluster_seed,
                                    train_bmu,
                                    val_bmu,
                                    labels["train"],
                                )
                                for alpha in alphas:
                                    mapping = prevalence_mapping(
                                        labels["train"],
                                        train_bmu,
                                        fit["unit_clusters"],
                                        fit["centres"],
                                        alpha,
                                    )
                                    prediction, _ = predict_from_bmu(
                                        val_bmu, fit["unit_clusters"], mapping
                                    )
                                    score = classification_metrics(
                                        labels["validation"], prediction
                                    )
                                    records.append(
                                        {
                                            "side": side,
                                            "k": k,
                                            "weighting": weight,
                                            "mapping": f"alpha{alpha}",
                                            "alpha": alpha,
                                            "som_seed": seed,
                                            "clustering_seed": cluster_seed,
                                            "state": "complete",
                                            **{metric: score[metric] for metric in METRICS},
                                        }
                                    )
            summary = write_metrics(OUT / "clustering" / study, records)
            if study == "original":
                summary["mapping"] = summary.mapping.map(
                    {"alpha0": "majority", "alpha1": "class_balanced"}
                )
                from clustering import condition_id

                summary["condition_id"] = [
                    condition_id(r.side, r.k, r.weighting, r.mapping)
                    for r in summary.itertuples()
                ]
                write_json(
                    OUT / "clustering/original/selection.json", select_condition(summary)
                )
    return {"stage": "clustering", "fits": 1000, "validation_evaluations": 2750}


def assessment():
    parts, groups = partitions()
    rows = parts["assessment"]
    folds = parts["assessment_fold_by_source_row"][rows]
    frame = read_source(rows)
    features = descriptive(frame, reduced=True)
    labels = frame.attack_cat.to_numpy(dtype=np.int8)
    raw = frame.drop(columns="attack_cat").copy()
    for column in ("proto", "service", "state"):
        raw[column] = raw[column].astype("string").fillna("<Missing>")
    raw["is_ftp_login"] = raw.is_ftp_login.gt(0).astype("int8")
    procedures = {"k40": (40, 1.0), "k150": (150, 0.5), "k400": (400, 0.5)}
    predictions = {name: np.full((5, 5, len(rows)), -1, dtype=np.int8) for name in procedures}
    predictions.update(
        {name: np.full(len(rows), -1, dtype=np.int8) for name in ("knn", "tree")}
    )
    selected, baseline_config = settings()
    with threadpool_limits(limits=4):
        for fold in range(1, 6):
            train, held = outer_positions(
                rows,
                groups[rows],
                folds,
                np.concatenate([parts["train"], parts["validation"]]),
                fold,
            )
            if np.intersect1d(groups[rows[train]], groups[rows[held]]).size:
                raise ValueError("Geometry group crosses assessment folds")
            processor = SOMPreprocessor("log_standard").fit(features.iloc[train])
            training, validation = (
                processor.transform(features.iloc[position]) for position in (train, held)
            )
            base = OUT / f"assessment/fold_{fold}"
            write_json(base / "preprocessing.json", processor.metadata())
            for s, seed in enumerate(SOM_SEEDS):
                print(f"Assessment fold {fold}, SOM seed {seed}", flush=True)
                model = fit_map(base / f"som_s{seed}", training, validation, seed, 32)
                train_bmu = bmus(training, model.codebook, second=False)[0]
                held_bmu = bmus(validation, model.codebook, second=False)[0]
                hits = np.bincount(train_bmu, minlength=1024)
                for name, (k, alpha) in procedures.items():
                    for c, cluster_seed in enumerate(CLUSTERING_SEEDS):
                        fit = fit_clusters(
                            base / f"{name}_s{seed}_c{cluster_seed}",
                            model,
                            hits,
                            k,
                            "hit_weighted",
                            cluster_seed,
                            train_bmu,
                            held_bmu,
                            labels[train],
                        )
                        mapping = prevalence_mapping(
                            labels[train],
                            train_bmu,
                            fit["unit_clusters"],
                            fit["centres"],
                            alpha,
                        )
                        predictions[name][s, c, held] = predict_from_bmu(
                            held_bmu, fit["unit_clusters"], mapping
                        )[0]
            for method in ("knn", "tree"):
                directory = base / method
                expected = identity(method=method, fold=fold, settings=selected[method])
                if complete(directory, expected):
                    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                        prediction = saved["predictions"]
                else:
                    prediction, timing, _, _ = baseline_predict(
                        raw.iloc[train],
                        labels[train],
                        raw.iloc[held],
                        selected,
                        method,
                        fold,
                        baseline_config["seed"],
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        directory / "predictions.npz",
                        train_rows=rows[train],
                        test_rows=rows[held],
                        predictions=prediction,
                    )
                    write_json(directory / "runtime.json", timing)
                    finish(directory, expected)
                predictions[method][held] = prediction
    if any(not np.isin(value, np.arange(10)).all() for value in predictions.values()):
        raise ValueError("Incomplete assessment predictions")
    destination = OUT / "assessment"
    np.savez_compressed(
        destination / "oof.npz",
        rows=rows,
        labels=labels,
        groups=groups[rows],
        folds=folds,
        **predictions,
    )
    records = []
    for name, values in predictions.items():
        metrics = [
            classification_metrics(labels, prediction)
            for prediction in values.reshape(-1, len(rows))
        ]
        for metric in METRICS:
            scores = np.array([record[metric] for record in metrics])
            stats = (
                nested_values(scores.reshape(5, 5))
                if len(scores) == 25
                else {
                    "mean": scores.mean(),
                    "som_sd": 0.0,
                    "kmeans_sd_mean": 0.0,
                    "minimum": scores.min(),
                    "maximum": scores.max(),
                }
            )
            records.append({"method": name, "metric": metric, **stats})
    pd.DataFrame(records).to_csv(destination / "summary.csv", index=False)
    return {"stage": "assessment", "rows": len(rows), "som_procedures": 3}


def bootstrap():
    from bootstrap import draw_table, paired_bootstrap

    path = OUT / "assessment/oof.npz"
    with np.load(path, allow_pickle=False) as saved:
        labels, groups = saved["labels"], saved["groups"]
        methods = {name: saved[name] for name in ("knn", "tree", "k40", "k150", "k400")}
    result = paired_bootstrap(labels, groups, methods)
    destination = OUT / "bootstrap"
    destination.mkdir(parents=True, exist_ok=True)
    draw_table(result).to_csv(destination / "draws.csv", index=False)
    np.savez_compressed(destination / "draws.npz", **result)
    means = result["paired_differences"].mean(axis=3).mean(axis=2)
    intervals = np.quantile(means, [0.025, 0.975], axis=0)
    from bootstrap import DIFFERENCES

    pd.DataFrame(
        {
            "comparison": DIFFERENCES,
            "lower": intervals[0],
            "upper": intervals[1],
        }
    ).to_csv(destination / "intervals.csv", index=False)
    return {"stage": "bootstrap", "replicates": 2000, "seed": 47441}


def smoke():
    rng = np.random.default_rng(47412)
    values = np.concatenate(
        [
            rng.normal(-2, 0.2, (100, 3)),
            rng.normal(2, 0.2, (100, 3)),
        ]
    )
    with threadpool_limits(limits=1):
        model = train_som(
            values,
            47421,
            settings=SOMSettings(2, 3, 0.75, 32),
            refinement_epochs=2,
            validation=values[::10],
        )
    if not np.isfinite(model.codebook).all():
        raise ValueError("Non-finite synthetic codebook")
    return {"synthetic_rows": len(values), "epochs": 5, "finite": True}
