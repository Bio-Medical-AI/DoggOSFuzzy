import pytest
from functools import partial
import numpy as np

FLOAT_PRECISION = 1e-12

approx = partial(pytest.approx, abs=FLOAT_PRECISION)


def _random_sample(low, high, size):
    return np.random.random_sample(size) * (high - low) + low
