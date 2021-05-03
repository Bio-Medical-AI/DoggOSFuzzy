import pytest
from functools import partial


FLOAT_PRECISION = 1e-12


approx = partial(pytest.approx, abs=FLOAT_PRECISION)

