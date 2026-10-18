from dataclasses import replace

import numpy as np
import pytest

from config import ROOT, load_config
from data import (
    TARGET,
    descriptive,
    group_ids,
    sha256,
    validate_schema,
    verify_file,
)
from preprocessing import SOMPreprocessor, fit_on_training
from splits import check_partitions, smoke_rows, split_development


def test_schema(source_frame):
    validate_schema(source_frame, full=False)
    with pytest.raises(ValueError, match="columns"):
        validate_schema(source_frame.drop(columns=TARGET), full=False)
    with pytest.raises(ValueError, match="row count"):
        validate_schema(source_frame)
    source_frame[TARGET] = source_frame[TARGET].astype(float)
    source_frame.loc[0, TARGET] = 1.5
    with pytest.raises(ValueError, match="Targets"):
        validate_schema(source_frame, full=False)


@pytest.mark.parametrize(
    "column,value", [("dur", np.inf), ("dur", np.nan), ("is_ftp_login", 3), ("id", 2)]
)
def test_schema_violations(source_frame, column, value):
    source_frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_schema(source_frame, full=False)


def test_input_identity(tmp_path):
    path = tmp_path / "source.csv"
    with pytest.raises(FileNotFoundError):
        verify_file(path, "missing")
    path.write_text("immutable")
    expected = sha256(path)
    verify_file(path, expected)
    path.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        verify_file(path, expected)


def test_groups_ignore_labels_preserve_repeats(source_frame):
    original = group_ids(descriptive(source_frame))
    source_frame[TARGET] = 9 - source_frame[TARGET]
    np.testing.assert_array_equal(original, group_ids(descriptive(source_frame)))
    assert original[0] == original[1]
    with pytest.raises(ValueError, match="exclude"):
        group_ids(source_frame)


def test_partition_determinism_and_coverage(source_frame, features):
    target = source_frame[TARGET].to_numpy()
    groups = group_ids(features)
    rows = np.arange(len(features))
    first = split_development(rows, target, groups, 47411)
    second = split_development(rows, target, groups, 47411)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    check_partitions(dict(zip(["train", "validation"], first, strict=True)), target, groups)
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), rows)


def test_overlap_guards(source_frame, features):
    y, g = source_frame[TARGET].to_numpy(), group_ids(features)
    train = np.arange(0, len(features), 2)
    with pytest.raises(ValueError, match="Row overlap"):
        check_partitions({"a": train, "b": train}, y, g)
    with pytest.raises(ValueError, match="Duplicate-group overlap"):
        check_partitions({"a": train, "b": train + 1}, y, g)
    with pytest.raises(ValueError, match="Missing class"):
        check_partitions({"a": np.arange(5)}, y, g)
    with pytest.raises(ValueError, match="Out-of-range"):
        check_partitions({"a": np.array([-1])}, y, g)


def test_smoke_seed(source_frame, features):
    y, g = source_frame[TARGET].to_numpy(), group_ids(features)
    rows = np.arange(len(features))
    a = smoke_rows(rows, y, g, 47412, 5)
    np.testing.assert_array_equal(a, smoke_rows(rows, y, g, 47412, 5))
    assert not np.array_equal(a, smoke_rows(rows, y, g, 47414, 5))
    # Whole feature groups, including conflicting targets, are retained.
    assert not np.intersect1d(g[a], g[np.setdiff1d(rows, a)]).size


@pytest.mark.parametrize("candidate", ["standard", "log_standard"])
def test_label_guard(features, candidate):
    labelled = features.assign(attack_cat=0)
    with pytest.raises(ValueError, match="Target"):
        SOMPreprocessor(candidate).fit(labelled)
    processor = SOMPreprocessor(candidate).fit(features)
    with pytest.raises(ValueError, match="Target"):
        processor.transform(labelled)
    assert not any(TARGET in name for name in processor.feature_names_)
    with pytest.raises(TypeError):
        processor.fit(features, np.zeros(len(features)))


def test_fit_scope_and_unknown_constant(features):
    permitted = np.arange(120)
    with pytest.raises(ValueError, match="permitted"):
        fit_on_training(features, np.arange(121), permitted, "standard")
    processor = fit_on_training(features, permitted, permitted, "standard")
    expected = features.loc[permitted, "dur"].mean()
    assert processor.scaler_.mean_[0] == expected
    np.testing.assert_array_equal(processor.fit_rows_, permitted)
    heldout = features.loc[np.arange(120, 240)].copy()
    heldout["dur"] = 10**9
    heldout["proto"] = "unseen"
    matrix = processor.transform(heldout)
    assert processor.scaler_.mean_[0] == expected
    assert matrix.shape[1] == processor.transform(features.loc[permitted]).shape[1]
    assert np.isfinite(matrix).all()
    assert (processor.scaler_.scale_ > 0).all()
    proto_index = processor.feature_names_.index("nominal__proto_known")
    assert np.all(matrix[:, proto_index] == 0)


def test_log_selection_is_training_only(features):
    train = features.iloc[:120].copy()
    train["sbytes"] = 0
    train.loc[train.index[-1], "sbytes"] = 1000000
    processor = SOMPreprocessor("log_standard").fit(train)
    assert "sbytes" in processor.metadata()["log_columns"]
    heldout = features.iloc[120:].copy()
    heldout["rate"] = 0
    heldout.loc[heldout.index[-1], "rate"] = 1000000
    processor.transform(heldout)
    assert "rate" not in processor.metadata()["log_columns"]


def test_binary_repair_and_nominal_geometry(source_frame, features):
    source_frame.loc[0, "is_ftp_login"] = 4
    repaired = descriptive(source_frame, reduced=True)
    assert repaired.loc[0, "is_ftp_login"] == 1
    features.loc[0, "proto"] = "other"
    features.loc[0, "is_sm_ips_ports"] = 1
    processor = SOMPreprocessor("standard").fit(features)
    values = processor.transform(features)
    nominal = [i for i, name in enumerate(processor.feature_names_) if "nominal__proto" in name]
    assert np.isclose(np.sum((values[0, nominal] - values[1, nominal]) ** 2), 1)
    binary = processor.feature_names_.index("binary__is_sm_ips_ports")
    assert values[0, binary] == 1 and values[1, binary] == 0
    with pytest.raises(ValueError, match="schema"):
        processor.transform(features.assign(id=np.arange(len(features))))


def test_config_strictness(tmp_path):
    a = load_config(ROOT / "configs/full.yaml")
    assert a == load_config(ROOT / "configs/full.yaml")
    assert a.path("data_path").is_absolute()
    for text in (
        "unknown: 1",
        "results_dir: ../assignment-two/results",
        "validation_seed: 42.5",
        "candidates: [tree]",
        "[]",
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)
    assert replace(a, results_dir="results-smoke").path("results_dir").is_relative_to(ROOT)
