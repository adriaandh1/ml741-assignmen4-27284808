"""Full V3 schedule from scratch, using the unchanged custom batch SOM kernel."""

from time import perf_counter

import numpy as np

from som import BatchSOM, SOMSettings, batch_update, matrix


def train_som(
    training, seed, *, settings=None, refinement_epochs=30, progress=None, validation=None
):
    settings = settings or SOMSettings(32, 40, 0.75, 2048)
    model = BatchSOM(settings, seed)
    x = matrix(training)
    model.initialise(x)
    model.centroid = x.mean(axis=0)
    model.history = []
    started = perf_counter()
    model._checkpoint(x, validation, 0, settings.side / 2, 0, started)
    radii = np.concatenate(
        (settings.radii(), np.full(refinement_epochs, settings.final_radius))
    )
    for epoch, radius in enumerate(radii, start=1):
        previous = model.codebook
        model.codebook = batch_update(
            x, previous, model.coordinates, float(radius), settings.chunk_size
        )
        movement = float(np.linalg.norm(model.codebook - previous, axis=1).mean())
        model._checkpoint(x, validation, epoch, float(radius), movement, started)
        model.history[-1]["phase"] = "annealing" if epoch <= settings.epochs else "fixed_radius"
        if progress and (epoch % 10 == 0 or epoch == len(radii)):
            progress(epoch, len(radii), perf_counter() - started)
    model.runtime_seconds = perf_counter() - started
    return model
