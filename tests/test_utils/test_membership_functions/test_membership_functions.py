from doggos.utils.membership_functions import membership_functions
import pytest


def test_gaussian():
    gaussian_set = membership_functions.gaussian(0.4, 0.15, 1)
    assert gaussian_set(0.4) == 1.0
    assert gaussian_set(0.1) == pytest.approx(0.135, 0.01)
    assert gaussian_set(0.9) == pytest.approx(0.00386, 0.01)


def test_sigmoid():
    sigmoid_set = membership_functions.sigmoid(0.5, -15)
    assert sigmoid_set(0.5) == 0.5
    assert sigmoid_set(0.345) == pytest.approx(0.91, 0.01)
    assert sigmoid_set(0.6525) == pytest.approx(0.092, 0.01)


def test_triangular():
    triangular_set = membership_functions.triangular(0.2, 0.3, 0.7)
    assert triangular_set(0.1) == 0
    assert triangular_set(0.2525) == 0.525
    assert triangular_set(0.3) == 1
    assert triangular_set(0.675) == pytest.approx(0.0625, 0.001)


def test_trapezoidal():
    trapezoidal_set = membership_functions.trapezoidal(0.2, 0.3, 0.6, 0.9)
    assert trapezoidal_set(0.15) == 0
    assert trapezoidal_set(0.25) == 0.5
    assert trapezoidal_set(0.4) == 1
    assert trapezoidal_set(0.95) == 0
    assert trapezoidal_set(0.65) == pytest.approx(0.833, 0.01)


def test_linear():
    linear_set = membership_functions.linear(4, -1)
    assert linear_set(0.15) == 0
    assert linear_set(0.5) == 1
    assert linear_set(0.3) == pytest.approx(0.2, 0.01)
    assert linear_set(0.35) == pytest.approx(0.4, 0.01)
