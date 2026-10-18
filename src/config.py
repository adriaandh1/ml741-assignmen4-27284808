"""Strict, project-relative deterministic configuration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    data_path: str = "data/networkTraffic.csv"
    metadata_path: str = "data/features.csv"
    results_dir: str = "runs"
    validation_seed: int = 47411
    smoke_seed: int = 47412
    smoke_groups_per_class: int = 30
    candidates: tuple[str, ...] = ("standard", "log_standard")

    def path(self, field: str) -> Path:
        return (ROOT / getattr(self, field)).resolve()

    def snapshot(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    unknown = set(raw) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if "candidates" in raw:
        raw["candidates"] = tuple(raw["candidates"])
    config = Config(**raw)
    for seed in (config.validation_seed, config.smoke_seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be unsigned 32-bit integers")
    if type(config.smoke_groups_per_class) is not int or config.smoke_groups_per_class < 10:
        raise ValueError("Smoke requires at least ten groups per class")
    if not config.candidates or not set(config.candidates) <= {"standard", "log_standard"}:
        raise ValueError("Unsupported representation candidate")
    if len(set(config.candidates)) != len(config.candidates):
        raise ValueError("Duplicate representation candidates")
    for field in ("data_path", "metadata_path", "results_dir"):
        if not isinstance(getattr(config, field), str):
            raise ValueError(f"{field} must be a path string")
    destination = config.path("results_dir")
    if not destination.is_relative_to(ROOT) or destination == ROOT:
        raise ValueError("Results must remain inside Assignment 4")
    return config
