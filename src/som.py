"""Custom rectangular batch SOM, with chunked Euclidean BMUs and no labels API."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def matrix(values: np.ndarray) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        raise TypeError("SOM accepts feature-only NumPy matrices, never labelled DataFrames")
    if not isinstance(values, np.ndarray) or values.ndim != 2 or not values.size:
        raise ValueError("Expected a non-empty two-dimensional feature matrix")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("SOM input must be finite floating-point features")
    return np.ascontiguousarray(values, dtype=np.float64)


def lattice(side: int) -> np.ndarray:
    if type(side) is not int or side < 2:
        raise ValueError("Map side must be an integer >=2")
    return np.indices((side, side)).reshape(2, -1).T


def adjacent(first: np.ndarray, second: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.abs(coordinates[first] - coordinates[second]).sum(axis=1) == 1


def bmus(
    values: np.ndarray, codebook: np.ndarray, chunk_size: int = 2048, *, second: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable BMUs: float64 norm identity, directly refined near the second-place boundary."""
    x, w = matrix(values), matrix(codebook)
    if x.shape[1] != w.shape[1] or (second and len(w) < 2):
        raise ValueError("BMU feature dimension/unit count mismatch")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid BMU chunk size")
    first = np.empty(len(x), dtype=np.int32)
    runners_up = np.full(len(x), -1, dtype=np.int32)
    distances = np.empty(len(x))
    codebook_norm = np.einsum("ij,ij->i", w, w)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        block = x[start:stop]
        input_norm = np.einsum("ij,ij->i", block, block)
        squared = input_norm[:, None] + codebook_norm[None, :] - 2 * (block @ w.T)
        if not np.isfinite(squared).all():
            raise ValueError("Squared-distance arithmetic overflowed")
        np.maximum(squared, 0, out=squared)
        # Conservative forward-error bound for the norm identity. Refine all candidates
        # that might enter the first two, using direct differences in bounded pair batches.
        boundary = np.partition(squared, 1 if len(w) > 1 else 0, axis=1)[
            :, 1 if len(w) > 1 else 0
        ]
        tolerance = (
            16 * np.finfo(float).eps * (x.shape[1] + 2) * (input_norm + codebook_norm.max())
        )
        candidate_rows, candidate_units = np.nonzero(
            squared <= boundary[:, None] + tolerance[:, None]
        )
        if len(candidate_rows) > 8 * len(block):
            # Highly tied blocks: bounded direct distance matrix is cheaper than many gathers.
            squared = cdist(block, w, metric="sqeuclidean")
        else:
            for pair_start in range(0, len(candidate_rows), 4096):
                r = candidate_rows[pair_start : pair_start + 4096]
                u = candidate_units[pair_start : pair_start + 4096]
                differences = block[r] - w[u]
                squared[r, u] = np.einsum("ij,ij->i", differences, differences)
        rows = np.arange(stop - start)
        best = np.argmin(squared, axis=1)  # argmin returns the lowest index on exact ties.
        first[start:stop] = best
        distances[start:stop] = np.sqrt(squared[rows, best])
        if second:
            squared[rows, best] = np.inf
            runners_up[start:stop] = np.argmin(squared, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Euclidean distances overflowed")
    return first, runners_up, distances


def batch_update(
    values: np.ndarray,
    codebook: np.ndarray,
    coordinates: np.ndarray,
    radius: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    """w_j = sum_k h_jk S_k / sum_k h_jk n_k. zero-denominator units stay put."""
    x, w = matrix(values), matrix(codebook)
    if not np.isfinite(radius) or radius <= 0 or len(coordinates) != len(w):
        raise ValueError("Invalid neighbourhood radius/lattice")
    assignments, _, _ = bmus(x, w, chunk_size, second=False)
    counts = np.bincount(assignments, minlength=len(w))
    sums = np.empty_like(w)
    for feature in range(x.shape[1]):
        sums[:, feature] = np.bincount(assignments, weights=x[:, feature], minlength=len(w))
    distances = cdist(coordinates, coordinates, "sqeuclidean")
    influence = np.exp(-distances / (2 * radius**2))
    denominators = influence @ counts
    updated = w.copy()
    positive = denominators > 0
    updated[positive] = (influence @ sums)[positive] / denominators[positive, None]
    if not np.isfinite(updated).all():
        raise ValueError("Non-finite batch SOM update")
    return updated


def diagnostics(
    values: np.ndarray,
    codebook: np.ndarray,
    coordinates: np.ndarray,
    centroid: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[dict, dict]:
    x = matrix(values)
    first, second, distances = bmus(x, codebook, chunk_size)
    hits = np.bincount(first, minlength=len(codebook))
    denominator = float(np.linalg.norm(x - centroid, axis=1).mean())
    qe = float(distances.mean())
    occupied = int(np.count_nonzero(hits))
    metrics = {
        "qe": qe,
        "centroid_qe": denominator,
        "contextual_qe": qe / denominator if denominator > 0 else None,
        "te": float((~adjacent(first, second, coordinates)).mean()),
        "observations": len(x),
        "total_units": len(codebook),
        "occupied_units": occupied,
        "dead_units": len(codebook) - occupied,
        "occupied_fraction": occupied / len(codebook),
        "hit_min": int(hits.min()),
        "hit_max": int(hits.max()),
        "hit_median": float(np.median(hits)),
        "hit_quantiles": np.quantile(hits, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }
    return metrics, {"first": first, "second": second, "distance": distances, "hits": hits}


def neighbour_distance_map(codebook: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Unit-centred mean distance to edge-sharing neighbours (not an expanded U-matrix)."""
    w = matrix(codebook)
    neighbours = cdist(coordinates, coordinates, "cityblock") == 1
    distances = cdist(w, w, "euclidean")
    return np.sum(distances * neighbours, axis=1) / neighbours.sum(axis=1)


@dataclass(frozen=True)
class SOMSettings:
    side: int = 16
    epochs: int = 40
    final_radius: float = 0.75
    chunk_size: int = 2048

    def validate(self) -> None:
        lattice(self.side)
        if type(self.epochs) is not int or self.epochs < 2:
            raise ValueError("At least two fixed-budget epochs required")
        if not np.isfinite(self.final_radius) or self.final_radius <= 0:
            raise ValueError("Invalid final radius")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("Invalid chunk size")

    def radii(self) -> np.ndarray:
        self.validate()
        return np.geomspace(self.side / 2, self.final_radius, self.epochs)


class BatchSOM:
    def __init__(self, settings: SOMSettings, seed: int):
        settings.validate()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an unsigned integer")
        self.settings = settings
        self.seed = seed
        self.coordinates = lattice(settings.side)

    def initialise(self, training: np.ndarray) -> np.ndarray:
        x = matrix(training)
        if len(x) < len(self.coordinates):
            raise ValueError("Observation initialisation requires at least n_units rows")
        self.initial_rows = np.random.default_rng(self.seed).choice(
            len(x), len(self.coordinates), replace=False
        )
        self.codebook = x[self.initial_rows].copy()
        return self.codebook.copy()

    def fit(self, training: np.ndarray, *, validation: np.ndarray | None = None):
        x = matrix(training)
        val = matrix(validation) if validation is not None else None
        if val is not None and val.shape[1] != x.shape[1]:
            raise ValueError("Validation feature dimension mismatch")
        self.initialise(x)
        self.centroid = x.mean(axis=0)
        self.history = []
        started = perf_counter()
        self._checkpoint(x, val, 0, self.settings.side / 2, 0, started)
        for epoch, radius in enumerate(self.settings.radii(), start=1):
            previous = self.codebook
            self.codebook = batch_update(
                x, previous, self.coordinates, float(radius), self.settings.chunk_size
            )
            movement = float(np.linalg.norm(self.codebook - previous, axis=1).mean())
            self._checkpoint(x, val, epoch, float(radius), movement, started)
        self.runtime_seconds = perf_counter() - started
        return self

    def _checkpoint(self, training, validation, epoch, radius, movement, started):
        row = {
            "epoch": epoch,
            "radius": radius,
            "codebook_movement": movement,
            "elapsed_seconds": perf_counter() - started,
        }
        for name, data in (("train", training), ("validation", validation)):
            if data is not None:
                metrics, _ = diagnostics(
                    data,
                    self.codebook,
                    self.coordinates,
                    self.centroid,
                    self.settings.chunk_size,
                )
                row.update(
                    {
                        f"{name}_{k}": metrics[k]
                        for k in ("qe", "contextual_qe", "te", "occupied_fraction")
                    }
                )
        self.history.append(row)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            codebook=self.codebook,
            coordinates=self.coordinates,
            centroid=self.centroid,
            initial_rows=self.initial_rows,
            seed=np.array(self.seed),
            settings=np.array(list(asdict(self.settings).values())),
        )

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as saved:
            side, epochs, radius, chunk = saved["settings"]
            model = cls(
                SOMSettings(int(side), int(epochs), float(radius), int(chunk)),
                int(saved["seed"]),
            )
            model.codebook = matrix(saved["codebook"])
            model.centroid = saved["centroid"].copy()
            model.initial_rows = saved["initial_rows"].copy()
            if not np.array_equal(saved["coordinates"], model.coordinates):
                raise ValueError("Saved lattice differs from declared topology")
            if len(model.codebook) != len(model.coordinates):
                raise ValueError("Saved codebook capacity differs")
        return model
