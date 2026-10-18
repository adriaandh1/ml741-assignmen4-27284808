import numpy as np
import pandas as pd
import pytest

from som import (
    BatchSOM,
    SOMSettings,
    adjacent,
    batch_update,
    bmus,
    diagnostics,
    lattice,
    neighbour_distance_map,
)


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_bmu_reference(chunk):
    rng = np.random.default_rng(117)
    x, w = rng.normal(size=(31, 6)), rng.normal(size=(9, 6))
    brute = np.linalg.norm(x[:, None, :] - w[None, :, :], axis=2)
    order = np.argsort(brute, axis=1, kind="stable")
    first, second, distance = bmus(x, w, chunk)
    np.testing.assert_array_equal(first, order[:, 0])
    np.testing.assert_array_equal(second, order[:, 1])
    np.testing.assert_allclose(distance, brute[np.arange(len(x)), first])


def test_ties_and_cancellation():
    x = np.array([[0.0, 0.0], [1e12, 1e12]])
    w = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1e12 + 1, 1e12], [1e12, 1e12 + 2]])
    first, second, distance = bmus(x, w, 1)
    np.testing.assert_array_equal(first, [0, 3])
    np.testing.assert_array_equal(second, [1, 4])
    np.testing.assert_array_equal(distance, [1, 1])
    duplicate = np.zeros((10, 2))
    a, b, d = bmus(np.zeros((3, 2)), duplicate)
    np.testing.assert_array_equal(a, 0)
    np.testing.assert_array_equal(b, 1)
    np.testing.assert_array_equal(d, 0)


def test_four_neighbour_geometry():
    coordinates = lattice(3)
    for unit, expected in [(0, 2), (1, 3), (4, 4), (8, 2)]:
        neighbours = adjacent(np.full(9, unit), np.arange(9), coordinates)
        assert neighbours.sum() == expected
    assert not adjacent(np.array([0]), np.array([4]), coordinates)[0]
    assert not adjacent(np.array([0]), np.array([2]), coordinates)[0]


def test_batch_update_independent_reference():
    x = np.array([[0.0], [2.0], [4.0]])
    w = np.array([[0.0], [4.0], [10.0], [12.0]])
    coords = lattice(2)
    radius = 0.6
    reference = []
    assigned = [0, 0, 1]  # x=2 tie chooses unit zero.
    for unit in range(4):
        numerator, denominator = 0.0, 0.0
        for row, best in zip(x, assigned, strict=True):
            distance = sum((coords[unit] - coords[best]) ** 2)
            weight = np.exp(-distance / (2 * radius**2))
            numerator += weight * row[0]
            denominator += weight
        reference.append([numerator / denominator])
    np.testing.assert_allclose(batch_update(x, w, coords, radius), reference)


def test_diagnostics_toy():
    coords = lattice(2)
    w = np.array([[0.0], [1.0], [10.0], [11.0]])
    x = np.array([[0.25], [10.25]])
    metrics, assignments = diagnostics(x, w, coords, np.array([5.25]))
    assert metrics["qe"] == 0.25
    assert metrics["contextual_qe"] == 0.05
    assert metrics["te"] == 0
    assert metrics["occupied_units"] == metrics["dead_units"] == 2
    np.testing.assert_array_equal(assignments["hits"], [1, 0, 1, 0])
    w_diagonal = np.array([[0.0], [10.0], [11.0], [1.0]])
    assert diagnostics(x[:1], w_diagonal, coords, np.array([5.0]))[0]["te"] == 1
    assert (
        diagnostics(np.zeros((2, 1)), np.zeros((4, 1)), coords, np.zeros(1))[0]["contextual_qe"]
        is None
    )


def test_neighbour_distance_corner_edge_interior():
    coords = lattice(3)
    w = (np.arange(9) ** 2).astype(float).reshape(-1, 1)
    measured = neighbour_distance_map(w, coords)
    for i in [0, 1, 4, 8]:
        neighbours = [j for j in range(9) if np.abs(coords[i] - coords[j]).sum() == 1]
        expected = np.mean([abs(w[i, 0] - w[j, 0]) for j in neighbours])
        assert measured[i] == expected


def test_seed_training_and_roundtrip(tmp_path):
    rng = np.random.default_rng(111)
    x = np.concatenate(
        [
            rng.normal(-3, 0.5, (50, 2)),
            rng.normal(3, 0.5, (50, 2)),
            rng.normal([3, -3], 0.5, (50, 2)),
        ]
    )
    settings = SOMSettings(2, 12, 0.1, 13)
    a, b = BatchSOM(settings, 47421), BatchSOM(settings, 47421)
    a.fit(x, validation=x[:7])
    b.fit(x)
    np.testing.assert_array_equal(a.codebook, b.codebook)
    assert a.history[-1]["train_qe"] < a.history[0]["train_qe"]
    assert np.isfinite(a.codebook).all()
    other = BatchSOM(settings, 47422)
    other.initialise(x)
    assert not np.array_equal(other.initial_rows, a.initial_rows)
    path = tmp_path / "som.npz"
    a.save(path)
    loaded = BatchSOM.load(path)
    for original, restored in zip(bmus(x, a.codebook), bmus(x, loaded.codebook), strict=True):
        np.testing.assert_array_equal(original, restored)
    assert (
        diagnostics(x, a.codebook, a.coordinates, a.centroid)[0]
        == diagnostics(x, loaded.codebook, loaded.coordinates, loaded.centroid)[0]
    )


def test_label_and_numeric_guards():
    model = BatchSOM(SOMSettings(2, 2), 47421)
    with pytest.raises(TypeError, match="labelled"):
        model.fit(pd.DataFrame({"feature": [0.0, 1.0], "attack_cat": [0, 1]}))
    with pytest.raises(TypeError):
        model.fit(np.ones((4, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.full((4, 2), np.nan))
    with pytest.raises(ValueError, match="rows"):
        model.fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="dimension"):
        model.fit(np.ones((4, 2)), validation=np.ones((4, 3)))
