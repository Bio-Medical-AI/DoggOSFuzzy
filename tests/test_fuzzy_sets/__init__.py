import numpy as np


def _random_sample(low, high, size):
    return np.random.random_sample(size) * (high - low) + low
