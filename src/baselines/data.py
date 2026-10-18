"""Dataset loading, schema validation, checksums, and fixed partitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

EXPECTED_SHA256 = "acde2c646b55c9c6907ade03b1ec985bf58f53926c6045ae221a61ab17535c3b"
EXPECTED_ROWS = 257_673
EXPECTED_COLUMNS = 44
TARGET = "attack_cat"
IDENTIFIER = "id"
NOMINAL_COLUMNS = ("proto", "state", "service")
BINARY_REPAIR_COLUMNS = ("is_ftp_login",)
ATTACK_CATEGORY_NAMES = (
    "Normal",
    "Reconnaissance",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
)


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    numeric_columns: tuple[str, ...]
    nominal_columns: tuple[str, ...] = NOMINAL_COLUMNS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    require_full_shape: bool = True,
) -> DatasetBundle:
    """Load the assignment CSV without altering the source file."""
    data_path = Path(path)
    if verify_checksum:
        observed = sha256_file(data_path)
        if observed != EXPECTED_SHA256:
            raise ValueError(f"Dataset checksum mismatch: {observed}")
    frame = pd.read_csv(data_path, na_values=["?"], keep_default_na=False)
    if require_full_shape and frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)}, found {frame.shape}")
    if len(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_COLUMNS} columns, found {len(frame.columns)}")
    if frame.columns[-1] != TARGET:
        raise ValueError(f"The final column must be {TARGET}")
    if not set(NOMINAL_COLUMNS).issubset(frame.columns):
        raise ValueError("Nominal feature columns are missing")
    if not frame[IDENTIFIER].is_unique:
        raise ValueError("id must contain one unique value per observation")

    target = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if sorted(target.unique().tolist()) != list(range(10)):
        raise ValueError("Target labels must be the integers zero through nine")
    features = frame.drop(columns=[TARGET]).copy()
    numeric_columns = tuple(
        column for column in features.columns if column not in NOMINAL_COLUMNS
    )
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in NOMINAL_COLUMNS:
        features[column] = features[column].astype("string").fillna("<Missing>")
    # Stable group labels are based on the untouched supplied feature values.
    # Model repairs occur afterwards and are separately checked not to merge groups.
    groups = pd.util.hash_pandas_object(
        features.drop(columns=[IDENTIFIER]), index=False
    ).rename("feature_group")
    for column in BINARY_REPAIR_COLUMNS:
        # The supplied schema defines is_ftp_login as binary, but 26 raw values
        # use the positive codes two or four. Preserve the indicator semantics.
        features[column] = features[column].gt(0).astype("int8")
    return DatasetBundle(
        frame=frame,
        features=features,
        target=target,
        groups=groups,
        numeric_columns=numeric_columns,
    )


def stratified_limit(
    indices: np.ndarray,
    target: pd.Series,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    """Deterministically cap a partition while preserving target proportions."""
    if limit is None or len(indices) <= limit:
        return np.asarray(indices, dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    local, _ = next(splitter.split(np.zeros(len(indices)), target.iloc[indices]))
    return np.asarray(indices[local], dtype=np.int64)


def create_partitions(
    target: pd.Series,
    groups: pd.Series,
    selection_fraction: float,
    seed: int,
    selection_limit: int | None = None,
    assessment_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    split_count = round(1.0 / selection_fraction)
    if not np.isclose(selection_fraction, 1.0 / split_count):
        raise ValueError("selection_fraction must be the reciprocal of an integer")
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    assessment, selection = next(splitter.split(np.zeros(len(target)), target, groups=groups))
    selection = stratified_limit(selection, target, selection_limit, seed + 1)
    assessment = stratified_limit(assessment, target, assessment_limit, seed + 2)
    if np.intersect1d(selection, assessment).size:
        raise AssertionError("Selection and assessment partitions overlap")
    if np.intersect1d(groups.iloc[selection], groups.iloc[assessment]).size:
        raise AssertionError("Feature groups overlap between experimental partitions")
    return np.sort(selection), np.sort(assessment)


def save_partitions(
    output_dir: str | Path,
    selection: np.ndarray,
    assessment: np.ndarray,
    groups: pd.Series,
    seed: int,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "partition_indices.npz",
        selection=selection,
        assessment=assessment,
    )
    manifest = {
        "seed": seed,
        "selection_count": int(len(selection)),
        "assessment_count": int(len(assessment)),
        "overlap_count": int(np.intersect1d(selection, assessment).size),
        "selection_group_count": int(groups.iloc[selection].nunique()),
        "assessment_group_count": int(groups.iloc[assessment].nunique()),
        "group_overlap_count": int(
            np.intersect1d(groups.iloc[selection], groups.iloc[assessment]).size
        ),
    }
    (destination / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
