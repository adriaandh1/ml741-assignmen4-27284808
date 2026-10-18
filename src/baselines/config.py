"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """All parameters that can change an experimental result."""

    seed: int = 741
    data_path: str = "networkTraffic.csv"
    results_dir: str = "results"
    selection_fraction: float = 0.20
    selection_limit: int | None = None
    assessment_limit: int | None = None
    inner_folds: int = 3
    outer_folds: int = 5
    n_jobs: int = 12
    knn_batch_size: int = 1024
    knn_values: tuple[int, ...] = (1, 3, 5, 7, 11, 21, 31, 51)
    knn_metrics: tuple[str, ...] = ("euclidean", "manhattan")
    knn_weights: tuple[str, ...] = ("uniform", "distance")
    knn_preprocessors: tuple[str, ...] = ("standard", "robust", "log_standard")
    tree_search_iterations: int = 36
    ablation_repeats: int = 5
    bootstrap_iterations: int = 2000
    run_baselines: bool = True

    def validate(self) -> None:
        if not 0.0 < self.selection_fraction < 1.0:
            raise ValueError("selection_fraction must be between zero and one")
        if self.inner_folds < 2 or self.outer_folds < 2:
            raise ValueError("cross-validation requires at least two folds")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.knn_batch_size < 1:
            raise ValueError("knn_batch_size must be positive")
        if self.ablation_repeats < 1:
            raise ValueError("ablation_repeats must be positive")
        if not self.knn_values or min(self.knn_values) < 1:
            raise ValueError("knn_values must contain positive integers")
        if any(value % 2 == 0 for value in self.knn_values):
            raise ValueError("odd knn_values are required to reduce voting ties")
        if not set(self.knn_weights) <= {"uniform", "distance"}:
            raise ValueError("unsupported k-nearest-neighbour voting rule")
        if not set(self.knn_metrics) <= {"euclidean", "manhattan"}:
            raise ValueError("unsupported nearest-neighbour distance metric")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("knn_values", "knn_metrics", "knn_weights", "knn_preprocessors"):
            data[key] = list(data[key])
        return data


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration with strict field validation."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    tuple_fields = {"knn_values", "knn_metrics", "knn_weights", "knn_preprocessors"}
    for field in tuple_fields & raw.keys():
        raw[field] = tuple(raw[field])
    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = ExperimentConfig(**raw)
    config.validate()
    return config
