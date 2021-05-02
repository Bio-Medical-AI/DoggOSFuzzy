import pytest
from functools import partial


FLOAT_PRECISION = 1e-6


approx = partial(pytest.approx, rel=FLOAT_PRECISION, abs=True)
