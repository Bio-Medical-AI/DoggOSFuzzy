import numpy as np


def gaussian(mean, sigma, max_value=1):
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
      >>> gaussian_set = gaussian(0.4, 0.15, 1)
      >>> membership_value = gaussian_set(0.5)
    """
    mf = lambda value: max_value * np.exp(-(((mean - value) ** 2) / (2 * sigma ** 2)))
    return mf


def sigmoid(offset, magnitude):
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
      >>> sigmoid_set = sigmoid(x1, 0.5, -15)
      >>> membership_value = sigmoid_set(0.2)
    """
    mf = lambda value: 1. / (1. + np.exp(- magnitude * (value - offset)))
    return mf


def triangular(l_end, center, r_end, max_value=1):
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
      >>> triangle_set = triangular(x1, 0.2, 0.3, 0.7)
      >>> membership_value - triangle_set(0.6)
    """
    mf = lambda value: np.minimum(1,
                                  np.maximum(0, ((max_value * (value - l_end) / (center - l_end)) * (value <= center) +
                                                 ((max_value * ((r_end - value) / (r_end - center))) * (
                                                             value > center)))))
    return mf


def trapezoidal(l_end, l_center, r_center, r_end, max_value=1):
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
      >>> trapezoid_set = trapezoidal(x1, 0.2, 0.3, 0.6, 0.7)
      >>> membership_value = trapezoid_set(0.4)
    """
    mf = lambda value: np.minimum(1, np.maximum(0, (
                (((max_value * ((value - l_end) / (l_center - l_end))) * (value <= l_center)) +
                 ((max_value * ((r_end - value) / (r_end - r_center))) * (value >= r_center))) +
                (max_value * ((value > l_center) * (value < r_center))))))
    return mf


def linear(a, b, max_value=1):
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
      >>> linear_set = linear(4, -1)
      >>> membership_value - linear_set(0.6)
    """
    mf = lambda value: np.minimum((value * a) + b, max_value) if ((value * a) + b) > 0 else 0
    return mf
