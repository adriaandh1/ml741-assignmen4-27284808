import numpy as np
import pandas as pd
import pytest

from data import COLUMNS, NOMINAL, descriptive


@pytest.fixture
def source_frame():
    rows = 240
    frame = pd.DataFrame({c: np.zeros(rows) for c in COLUMNS})
    frame["id"] = np.arange(1, rows + 1)
    frame["attack_cat"] = np.repeat(np.arange(10), rows // 10)
    for c in NOMINAL:
        frame[c] = "known"
    frame["dur"] = np.repeat(np.arange(rows // 2), 2).astype(float)
    return frame


@pytest.fixture
def features(source_frame):
    return descriptive(source_frame, reduced=True)
