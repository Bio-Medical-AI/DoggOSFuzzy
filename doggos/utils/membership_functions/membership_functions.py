import numpy as np

from typing import Callable


def gaussian(mean: float, sigma: float, max_value: float = 1) -> Callable[[float], float]:
    """Gaussian membership function

    Defines membership function of gaussian distribution shape
    To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      mean:
        Center of gaussian function, expected value.
      sigma:
        Standard deviation of gaussian function
      max_value:
        Maximum value of membership function, height.

    Returns:
      Callable[[float], float]
      Callable which calculates membership values for given input.


    Example usage:
      >>> gaussian_mf = gaussian(0.4, 0.15, 1)
      >>> membership_value = gaussian_mf(0.5)
    """

    def output_mf(value: float) -> float:
        return max_value * np.exp(-(((mean - value) ** 2) / (2 * sigma ** 2)))

    return output_mf


def sigmoid(offset: float, magnitude: float) -> Callable[[float], float]:
    """Sigmoid membership function

    Defines membership function of sigmoid shape
    To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      offset:
        Offset, bias is the center value of the sigmoid, where it has value of 0.5.
        Determines 'lean' of function.
      magnitude:
        Defines width of sigmoidal region around offset. Sign of the value determines which side
        of the function is open.

    Returns:
      Callable[[float], float]
      Callable which calculates membership values for given input.


    Example usage:
      >>> sigmoid_mf = sigmoid(0.5, -15)
      >>> membership_value = sigmoid_mf(0.2)
    """

    def output_mf(value: float) -> float:
        return 1. / (1. + np.exp(- magnitude * (value - offset)))

    return output_mf


def triangular(l_end: float, center: float, r_end: float, max_value: float = 1) -> Callable[[float], float]:
    """Triangular membership function

    Defines membership function of triangular shape
    To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      l_end:
        Left end, vertex of triangle, where value of function is equal to 0.
      center:
        Top vertex of triangle, where value of function is equal to 1.
      r_end:
        Right end, vertex of triangle, where value of function is equal to 0.
      max_value:
        Maximum value of membership function, height.

    Returns:
      Callable[[float], float]
      Callable which calculates membership values for given input.


    Example usage:
      >>> triangle_mf = triangular(0.2, 0.3, 0.7)
      >>> membership_value - triangle_mf(0.6)
    """

    def output_mf(value: float) -> float:
        return np.minimum(1,
                          np.maximum(0, ((max_value * (value - l_end) / (center - l_end)) * (value <= center) +
                                         ((max_value * ((r_end - value) / (r_end - center))) * (
                                                 value > center)))))

    return output_mf


def trapezoidal(l_end: float, l_center: float, r_center: float, r_end: float, max_value: float = 1) -> Callable[[float],
                                                                                                                float]:
    """Triangular membership function

    Defines membership function of trapezoidal shape
    To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      l_end:
        Left end, vertex of trapezoid, where value of function is equal to 0.
      l_center:
        Top left vertex of trapezoid, where value of function is equal to 1.
      r_center:
        Top right vertex of triangle, where value of function is equal to 1.
      r_end:
        Right end, vertex of trapezoid, where value of function is equal to 0.
      max_value:
        Maximum value of membership function, height.

    Returns:
      Callable[[float], float]
      Callable which calculates membership values for given input.


    Example usage:
      >>> trapezoid_mf = trapezoidal(0.2, 0.3, 0.6, 0.7)
      >>> membership_value = trapezoid_mf(0.4)
    """

    def output_mf(value: float) -> float:
        return np.minimum(1, np.maximum(0, (
                (((max_value * ((value - l_end) / (l_center - l_end))) * (value <= l_center)) +
                 ((max_value * ((r_end - value) / (r_end - r_center))) * (value >= r_center))) +
                (max_value * ((value > l_center) * (value < r_center))))))

    return output_mf


def linear(a: float, b: float, max_value: float = 1) -> Callable[[float], float]:
    """Linear membership function

    Defines linear membership function.
    To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      a:
        a factor in: y=ax + b
      b:
        b factor in: y=ax + b
      max_value:
        Maximum value of membership function, height.

    Returns:
      Callable[[float], float]
      Callable which calculates membership values for given input.


    Example usage:
      >>> linear_mf = linear(4, -1)
      >>> membership_value - linear_mf(0.6)
    """

    def output_mf(value: float) -> float:
        return float(np.minimum((value * a) + b, max_value) if ((value * a) + b) > 0 else 0)

    return output_mf
